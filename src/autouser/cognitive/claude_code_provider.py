"""Claude Code CLI provider for the cognitive engine.

Shells out to the local `claude` CLI (one process per call) instead of calling
the Anthropic API directly. No API key dependency — `claude` handles its own
keychain/OAuth via interactive `claude` login.

Locked command shape (per an earlier preflight review, 2026-05-29):
    claude -p --model sonnet --output-format json --json-schema <schema> \
           --no-session-persistence --tools "" <prompt>

Guardrails (non-negotiable):
* No `--bare` — that flag skips keychain/OAuth and requires ANTHROPIC_API_KEY,
  which contradicts the "use local claude code cli" direction.
* `--tools ""` — provider asks for a structured action decision; Claude Code
  must not inspect or edit the repo.
* `--no-session-persistence` — one-shot calls; no state bleed across steps.

Boundary shape is intentionally exposed in three pieces so they can be reviewed
and tested independently:
    1. `build_command()`   — pure command builder, no side effects.
    2. `SubprocessRunner`  — injectable runner protocol so tests mock subprocess.
    3. `parse_envelope()`  — CLI envelope parser (separate from the
       schema-constrained payload parse, which lives in the caller).

Typed errors distinguish missing-binary, auth-failure, timeout, and
nonzero-exit so callers (and operators) get an actionable signal.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)


# --- Spawn concurrency cap (PR 2 carry #4) ----------------------------------
#
# `claude` (like similar agent CLIs) forks its own child tree, so a burst of
# concurrent `complete()` calls can drive the per-uid process count toward
# `kern.maxprocperuid` / `RLIMIT_NPROC` and get `fork()` refused with EPERM.
# An upstream agent backend's own concurrency limit bounds *backend
# invocations* but treats one autouser run as a single slot — it cannot see
# the `claude` subprocesses that run spawns *inside itself*. So the cap has to
# live here, at the shared spawn point, mirroring the backend-side fix one
# layer down.
#
# Design constraints (locked across the PR 2 review):
#   * PROCESS-GLOBAL, not per-instance. A per-instance Semaphore bounds nothing
#     if each parity leg constructs its own ClaudeCodeProvider. Hence module
#     scope, shared across all instances.
#   * LOOP-LAZY. A module-level Semaphore built at import time binds to whatever
#     loop exists then (or none); pytest-asyncio spins a fresh loop per test.
#     Bind on first use under the running loop and rebuild if the loop changes.
#   * DEFAULT 1 is a BUDGET decision, not taste: it makes autouser's addend the
#     smallest non-zero contribution to the shared per-uid ceiling, leaving the
#     daemon's term the headroom. Raising it spends budget the daemon also draws
#     on — so the tradeoff is documented here, at the dial, not discovered via
#     EPERM.
_DEFAULT_MAX_SPAWNS = 1
_MAX_SPAWNS_ENV = "AUTOUSER_CLAUDE_CODE_MAX_SPAWNS"

_spawn_semaphore: asyncio.Semaphore | None = None
_spawn_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _read_max_spawns() -> int:
    raw = os.environ.get(_MAX_SPAWNS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_SPAWNS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; falling back to default %d",
            _MAX_SPAWNS_ENV, raw, _DEFAULT_MAX_SPAWNS,
        )
        return _DEFAULT_MAX_SPAWNS
    if value < 1:
        logger.warning(
            "%s=%d is below 1; clamping to 1 (a cap of 0 would deadlock)",
            _MAX_SPAWNS_ENV, value,
        )
        return 1
    return value


def _spawn_gate() -> asyncio.Semaphore:
    """Return the process-global spawn semaphore bound to the running loop.

    Rebuilds if called under a different loop than the one it was bound to, so
    a per-test event loop never blocks on a semaphore owned by a dead loop.
    """
    global _spawn_semaphore, _spawn_semaphore_loop
    loop = asyncio.get_running_loop()
    if _spawn_semaphore is None or _spawn_semaphore_loop is not loop:
        _spawn_semaphore = asyncio.Semaphore(_read_max_spawns())
        _spawn_semaphore_loop = loop
    return _spawn_semaphore


# The fork-refused-at-spawn stderr signature (agent-CLI child startup
# refused near-instantly). Kept narrow on purpose: a bare exit code 1 is NOT
# enough — only this explicit signature reclassifies a nonzero exit as a
# process-limit failure, so ordinary code-1 application errors stay typed as
# ClaudeCodeInvocationError.
#
# BOTH hints require the Rust `std::io::Error` OS-error rendering — the literal
# "(os error N)" suffix the runtime appends when fork()/exec() fails. That
# suffix is the load-bearing discriminator: the bare phrase "resource
# temporarily unavailable" also appears in APPLICATION prose (e.g. an upstream
# "API error: resource temporarily unavailable, retry later"), and matching the
# bare phrase would mis-stamp that ordinary failure as a system process-limit
# error — the exact false-localization carry #5 exists to kill, one notch
# deeper (an earlier review, 2026-05-30). The OS-rendered form
# ("resource temporarily unavailable (os error 11)" for EAGAIN/RLIMIT_NPROC on
# Linux) only originates from the kernel refusing the spawn, never from model
# or API prose. Spawn-time EAGAIN is ALSO caught structurally by the errno gate
# in complete(); this stderr hint only covers the near-instant-nonzero-exit
# path where no OSError reaches us.
_PROCESS_LIMIT_STDERR_HINTS = (
    "operation not permitted (os error 1)",  # EPERM — fork refused
    "resource temporarily unavailable (os error",  # EAGAIN — RLIMIT_NPROC, OS-rendered only
)


def _is_process_limit_signature(stderr: str) -> bool:
    s = stderr.lower()
    return any(hint in s for hint in _PROCESS_LIMIT_STDERR_HINTS)


_PROCESS_LIMIT_MESSAGE = (
    "`claude` could not start — the system hit its per-process limit (a single "
    "budget shared across other agent processes, the daemon, and this run). This run may not be "
    f"the sole cause. Lowering {_MAX_SPAWNS_ENV} helps only if this run is "
    "contributing; restarting the daemon clears an already-grazed ceiling."
)


# --- Errors -----------------------------------------------------------------


class ClaudeCodeError(Exception):
    """Base error for any Claude Code CLI failure."""


class ClaudeCodeNotFoundError(ClaudeCodeError):
    """`claude` binary not on PATH."""


class ClaudeCodeAuthError(ClaudeCodeError):
    """`claude` CLI not authenticated (no keychain / no OAuth login)."""


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Subprocess exceeded the configured wall-clock budget."""


class ClaudeCodeInvocationError(ClaudeCodeError):
    """`claude` exited nonzero for a reason that isn't auth or timeout."""


class ClaudeCodeEnvelopeError(ClaudeCodeError):
    """CLI envelope (stdout) could not be parsed as JSON."""


class ClaudeCodeResponseError(ClaudeCodeError):
    """CLI returned a well-formed envelope with `is_error: true`.

    Distinct from `ClaudeCodeInvocationError` (nonzero process exit) and
    `ClaudeCodeEnvelopeError` (malformed envelope). `is_error` typically
    fires when the CLI exits 0 but the underlying call (schema validation,
    model invocation, internal CLI failure) failed and the CLI reported it
    in-band. Caller treats this as an actionable provider failure rather
    than as a downstream payload-parse failure.
    """


class ClaudeCodeProcessLimitError(ClaudeCodeError):
    """`claude` could not be spawned because the per-uid process ceiling
    (`kern.maxprocperuid` on macOS / `RLIMIT_NPROC` on Linux) was hit and
    `fork()` was refused with EPERM.

    Deliberately distinct from `ClaudeCodeInvocationError`. The discriminator
    is the *spawn-time* signal — a `PermissionError`/`OSError(EPERM)` raised
    before the child runs, OR the "Operation not permitted (os error 1)"
    stderr signature on a near-instant nonzero exit — NOT a bare exit code 1.
    Keying on exit-code-1 alone would mis-stamp this message onto ordinary
    application failures (bad flag, expired auth) that also exit 1, which is
    the false-localization trap this typed error exists to prevent
    (senior-eng / ux, PR 2 carry #5).

    The ceiling is a single budget shared across every process the uid runs
    (the daemon's agent backends, their child trees, this run's `claude`
    spawns, etc.). No single component sees the sum, so this run may not be
    the sole cause — the error copy says so rather than pinning blame on the
    local spawn cap.
    """


# --- Command builder --------------------------------------------------------

_DEFAULT_MODEL = "sonnet"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_TIMEOUT_ENV = "AUTOUSER_CLAUDE_CODE_TIMEOUT"


def _read_default_timeout() -> float:
    """Operator override for the per-call wall-clock budget, in seconds.

    Plan calls on pages with large DOM summaries can exceed the 120s default;
    the env var raises the ceiling without a code change. Explicit constructor
    arg still wins. Invalid or non-positive values fall back to the default.
    """
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; falling back to default %.0fs",
            _TIMEOUT_ENV, raw, _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning(
            "%s=%s is not positive; falling back to default %.0fs",
            _TIMEOUT_ENV, raw, _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    return value


def build_command(
    prompt: str,
    *,
    schema: Mapping[str, object],
    model: str | None = None,
) -> list[str]:
    """Build the argv list for a single Claude Code CLI invocation.

    Pure function: no env reads, no subprocess. Encodes the locked guardrails
    as positional/flag arguments. Unit tests assert presence of each flag.
    """
    schema_json = json.dumps(schema, separators=(",", ":"))
    # The `--` separator before the prompt is defensive: if `--tools` is ever
    # variadic in the CLI (consumes whitespace-separated tokens until the next
    # flag), the empty-string argument could silently absorb the prompt and
    # leave the model with no input. `--` ends option parsing so the prompt
    # is unambiguously positional. Caught at a boundary review (an earlier
    # PR #39 watch-item — verify under the opt-in smoke run too).
    return [
        "claude",
        "-p",
        "--model",
        model or _DEFAULT_MODEL,
        "--output-format",
        "json",
        "--json-schema",
        schema_json,
        "--no-session-persistence",
        "--tools",
        "",
        "--",
        prompt,
    ]


# --- Subprocess runner ------------------------------------------------------


@dataclass(frozen=True)
class CompletedSubprocess:
    """Outcome of a single subprocess call. Decoupled from asyncio internals
    so tests can construct one without touching the event loop."""

    returncode: int
    stdout: str
    stderr: str


SubprocessRunner = Callable[[Sequence[str], float], Awaitable[CompletedSubprocess]]


async def _default_runner(argv: Sequence[str], timeout: float) -> CompletedSubprocess:
    """Default runner: asyncio subprocess with wall-clock timeout.

    Raises FileNotFoundError if the binary is missing (caller translates to
    ClaudeCodeNotFoundError) and asyncio.TimeoutError on timeout (caller
    translates to ClaudeCodeTimeoutError).
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        # Drain pipes via communicate() instead of bare wait() — if the child
        # filled its stdout/stderr PIPE buffer before being killed, wait() can
        # deadlock waiting on the OS to reap the fds. communicate() reads the
        # remaining buffered output, then reaps. Suppress any post-kill error;
        # the timeout itself is the load-bearing signal. (an earlier PR #39
        # boundary review.)
        try:
            await proc.communicate()
        except Exception:
            pass
        raise
    return CompletedSubprocess(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
    )


# --- Envelope parser --------------------------------------------------------


def parse_envelope(raw_stdout: str) -> str:
    """Extract the schema-constrained payload from the CLI's JSON envelope.

    The Claude Code CLI emits a JSON object. Under `--json-schema` the
    schema-conformant payload is in `structured_output` (preferred); `result`
    is the fallback for responses without it. Other envelope fields (session id,
    cost, timing) are intentionally ignored — the caller only consumes the
    payload.

    Failure here is distinct from a payload-parse failure: the CLI returned
    something that isn't a JSON envelope at all, which is an infra failure
    (CLI crashed mid-stream, stderr-to-stdout merging, version skew), not a
    model-output failure. Callers should surface them differently.
    """
    try:
        envelope = json.loads(raw_stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeEnvelopeError(
            f"CLI envelope is not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(envelope, dict):
        raise ClaudeCodeEnvelopeError(
            "CLI envelope is not a JSON object; cannot extract payload"
        )

    # `is_error: true` can fire with returncode 0 (CLI reports schema /
    # invocation failure in-band). Surface it as a typed provider error
    # rather than letting `result` flow downstream as if it were valid
    # payload — caught at an earlier PR #39 boundary review.
    if envelope.get("is_error") is True:
        message = envelope.get("result")
        if not isinstance(message, str):
            message = json.dumps(message) if message is not None else "<no detail>"
        # Lead with the CLI's own message so an operator reading the error
        # sees the actionable failure text first, not internal envelope jargon.
        # (ux PR #39 error-copy polish.)
        raise ClaudeCodeResponseError(
            f"`claude` reported a failure: {message}"
        )

    # Under `--json-schema`, the CLI (>= 2.x) places the schema-conformant
    # object in `structured_output`; `result` then carries only the model's
    # free-text turn, which is frequently empty or prose. Read structured_output
    # FIRST so the schema clamp actually reaches the caller. Verified live
    # against CLI 2.1.158: with --json-schema, result='' / prose while
    # structured_output={"action": ...} holds the payload. This was the s06
    # parity-run defect — `result`-only reads returned '' → json.loads('') →
    # the engine defaulted every claude_code step to `wait`.
    structured = envelope.get("structured_output")
    if structured is not None:
        if isinstance(structured, str):
            return structured
        return json.dumps(structured)

    if "result" in envelope:
        result = envelope["result"]
        if isinstance(result, str):
            return result
        # The CLI sometimes wraps the JSON payload directly as an object;
        # re-serialize so the caller's payload parser gets a string consistently.
        return json.dumps(result)

    raise ClaudeCodeEnvelopeError(
        "CLI envelope missing `result` field; cannot extract payload"
    )


# --- Provider ---------------------------------------------------------------


_AUTH_HINTS = ("auth", "login", "credential", "unauthorized", "not logged in")


class ClaudeCodeProvider:
    """One-shot caller for the local `claude` CLI.

    Each `complete()` call spawns a fresh subprocess. No session is shared
    across calls (enforced via --no-session-persistence). No tools are
    available to the model (enforced via --tools "") so Claude Code cannot
    touch the repo while producing an action decision.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self._model = model or _DEFAULT_MODEL
        # Explicit arg wins; otherwise AUTOUSER_CLAUDE_CODE_TIMEOUT; otherwise 120s.
        self._timeout = timeout if timeout is not None else _read_default_timeout()
        self._runner: SubprocessRunner = runner or _default_runner

    async def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, object],
    ) -> str:
        """Run one CLI invocation and return the schema-constrained payload.

        Payload-to-domain-model parsing (ActionIntent / StepResult) is the
        caller's responsibility — this method only handles the CLI envelope.
        """
        argv = build_command(prompt, schema=schema, model=self._model)
        # carry #4: bound concurrent `claude` spawns process-globally. The gate
        # wraps the call to self._runner (not the runner internals) so the cap
        # is observable through an injected mock runner — that's how the
        # concurrency test asserts it without spawning real subprocesses.
        try:
            async with _spawn_gate():
                completed = await self._runner(argv, self._timeout)
        except FileNotFoundError as exc:
            raise ClaudeCodeNotFoundError(
                "`claude` binary not on PATH. Install Claude Code CLI and "
                "run `claude` once interactively to authenticate."
            ) from exc
        except asyncio.TimeoutError as exc:
            # MUST precede the broad `except OSError` below: in Python 3.11+
            # asyncio.TimeoutError IS the builtin TimeoutError, an OSError
            # subclass. Catch it here or the errno-check clause swallows it
            # and re-raises raw, dropping the typed-timeout translation.
            raise ClaudeCodeTimeoutError(
                f"`claude` did not return within {self._timeout:.0f}s; "
                "retry or raise the provider timeout if this recurs"
            ) from exc
        except OSError as exc:
            # carry #5 (spawn-time discriminator), narrowed per senior-eng's
            # review: gate on errno, NOT on the exception class. `fork()`
            # refused at the per-uid ceiling raises PermissionError(EPERM) or
            # BlockingIOError(EAGAIN) — both OSError subclasses — so this one
            # errno-gated clause covers them. Crucially it does NOT swallow
            # PermissionError(EACCES) (e.g. a present-but-non-executable
            # `claude`): EACCES is a setup error, not the ceiling, and stamping
            # the process-limit message on it would be the exact
            # false-localization this typed error exists to prevent. EACCES /
            # any other errno re-raises unchanged.
            if exc.errno in (errno.EPERM, errno.EAGAIN):
                raise ClaudeCodeProcessLimitError(_PROCESS_LIMIT_MESSAGE) from exc
            raise

        if completed.returncode != 0:
            # carry #5 (exit-signature discriminator): the ceiling can also
            # surface as a near-instant nonzero exit whose stderr carries the
            # "Operation not permitted (os error 1)" signature. Reclassify ONLY
            # on that explicit signature — a bare code-1 stays an invocation
            # error so we never mis-stamp the process-limit message onto an
            # ordinary application failure (the false-localization trap).
            if _is_process_limit_signature(completed.stderr):
                raise ClaudeCodeProcessLimitError(
                    f"{_PROCESS_LIMIT_MESSAGE} stderr={completed.stderr!r}"
                )
            stderr_lc = completed.stderr.lower()
            if any(hint in stderr_lc for hint in _AUTH_HINTS):
                raise ClaudeCodeAuthError(
                    "`claude` CLI is not authenticated. Run `claude` "
                    f"interactively to log in. stderr={completed.stderr!r}"
                )
            raise ClaudeCodeInvocationError(
                f"`claude` exited with code {completed.returncode}. "
                f"stderr={completed.stderr!r}"
            )

        return parse_envelope(completed.stdout)
