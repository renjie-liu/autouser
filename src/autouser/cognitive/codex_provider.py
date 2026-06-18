"""Codex CLI provider for the cognitive engine.

Shells out to the local `codex` CLI (one `codex exec` process per call) instead
of calling an LLM API directly. No API key dependency — `codex` handles its own
auth via `codex login`.

Command shape:
    codex exec [--model M] --skip-git-repo-check \
        --output-schema <schema.json> --output-last-message <message.txt> \
        -c sandbox_mode="read-only" -c approval_policy="never" -- <prompt>

Guardrails (non-negotiable):
* sandbox_mode="read-only" — the provider asks for a structured action decision;
  codex must not write to the repo/filesystem while answering.
* approval_policy="never" — one-shot, non-interactive; never blocks on a prompt.
* --skip-git-repo-check — runs regardless of CWD.

Boundary exposed in pieces for independent review/testing:
    1. build_command()  — pure argv builder, no side effects.
    2. SubprocessRunner — injectable runner protocol so tests mock subprocess.
    3. parse_output()   — final-message reader (the schema-constrained payload).
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)


# --- Errors -----------------------------------------------------------------


class CodexError(Exception):
    """Base error for any codex CLI failure."""


class CodexNotFoundError(CodexError):
    """`codex` binary not on PATH."""


class CodexAuthError(CodexError):
    """`codex` CLI not authenticated (`codex login` not run)."""


class CodexTimeoutError(CodexError):
    """Subprocess exceeded the configured wall-clock budget."""


class CodexInvocationError(CodexError):
    """`codex` exited nonzero for a reason that isn't auth or timeout."""


class CodexOutputError(CodexError):
    """`codex` exited 0 but wrote no usable final message."""


class CodexProcessLimitError(CodexError):
    """`codex` could not be spawned: the per-uid process ceiling
    (`kern.maxprocperuid` / `RLIMIT_NPROC`) was hit and fork() was refused with
    EPERM/EAGAIN. Distinct from CodexInvocationError: the discriminator is the
    spawn-time signal, not a bare nonzero exit."""


# --- Spawn concurrency cap --------------------------------------------------
#
# codex, like other agent CLIs, forks a child tree, so a burst of concurrent
# complete() calls can drive the per-uid process count toward the ceiling and
# get fork() refused with EPERM. The cap is PROCESS-GLOBAL (a per-instance
# semaphore bounds nothing if each leg builds its own provider) and LOOP-LAZY
# (bind on first use under the running loop; rebuild if the loop changes, so a
# per-test event loop never blocks on a semaphore owned by a dead loop).
_DEFAULT_MAX_SPAWNS = 1
_MAX_SPAWNS_ENV = "AUTOUSER_CODEX_MAX_SPAWNS"

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
            "%s=%r is not an integer; using default %d",
            _MAX_SPAWNS_ENV, raw, _DEFAULT_MAX_SPAWNS,
        )
        return _DEFAULT_MAX_SPAWNS
    return value if value >= 1 else 1  # a cap of 0 would deadlock


def _spawn_gate() -> asyncio.Semaphore:
    global _spawn_semaphore, _spawn_semaphore_loop
    loop = asyncio.get_running_loop()
    if _spawn_semaphore is None or _spawn_semaphore_loop is not loop:
        _spawn_semaphore = asyncio.Semaphore(_read_max_spawns())
        _spawn_semaphore_loop = loop
    return _spawn_semaphore


_DEFAULT_TIMEOUT_SECONDS = 120.0
_TIMEOUT_ENV = "AUTOUSER_CODEX_TIMEOUT"


def _read_default_timeout() -> float:
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using default %.0fs",
            _TIMEOUT_ENV, raw, _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_TIMEOUT_SECONDS


# The fork-refused stderr signature. BOTH hints require the OS-error rendering
# ("(os error N)") the runtime appends when fork()/exec() fails — the bare
# phrase "resource temporarily unavailable" also appears in API prose, and
# matching it bare would mis-stamp an ordinary failure as a process-limit error.
_PROCESS_LIMIT_STDERR_HINTS = (
    "operation not permitted (os error 1)",       # EPERM
    "resource temporarily unavailable (os error",  # EAGAIN / RLIMIT_NPROC
)


def _is_process_limit_signature(stderr: str) -> bool:
    s = stderr.lower()
    return any(hint in s for hint in _PROCESS_LIMIT_STDERR_HINTS)


_PROCESS_LIMIT_MESSAGE = (
    "`codex` could not start — the system hit its per-process limit (a single "
    f"budget shared across all agent processes). Lower {_MAX_SPAWNS_ENV} if this "
    "run is contributing, or reduce other concurrent agent processes."
)


# --- Command builder --------------------------------------------------------


def build_command(
    prompt: str,
    *,
    schema_path: str,
    output_path: str,
    model: str | None = None,
) -> list[str]:
    """Build the argv for one `codex exec` invocation. Pure function.

    `model` is omitted when None so codex uses its own ~/.codex/config.toml
    default. The `--` terminator makes the prompt unambiguously positional.
    """
    argv = ["codex", "exec"]
    if model:
        argv += ["--model", model]
    argv += [
        "--skip-git-repo-check",
        "--output-schema", schema_path,
        "--output-last-message", output_path,
        "-c", 'sandbox_mode="read-only"',
        "-c", 'approval_policy="never"',
        "--", prompt,
    ]
    return argv


# --- Subprocess runner ------------------------------------------------------


@dataclass(frozen=True)
class CompletedSubprocess:
    """Outcome of a single subprocess call, decoupled from asyncio internals so
    tests can construct one without an event loop."""

    returncode: int
    stdout: str
    stderr: str


SubprocessRunner = Callable[[Sequence[str], float], Awaitable[CompletedSubprocess]]


async def _default_runner(argv: Sequence[str], timeout: float) -> CompletedSubprocess:
    """Default runner: asyncio subprocess with a wall-clock timeout. Raises
    FileNotFoundError if the binary is missing and asyncio.TimeoutError on
    timeout (the caller translates both to typed errors)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        # Drain via communicate() (not bare wait()) — a child that filled its
        # PIPE buffer before being killed can deadlock wait(). Suppress any
        # post-kill error; the timeout itself is the signal.
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


# --- Output reader ----------------------------------------------------------


def parse_output(text: str) -> str:
    """Return the schema-conformant payload written to --output-last-message.

    Under --output-schema, codex writes the final response (a JSON object as
    text) to the file. The caller parses it into the domain model — this only
    strips and guards emptiness.
    """
    payload = text.strip()
    if not payload:
        raise CodexOutputError("`codex` wrote an empty final message")
    return payload


# --- Provider ---------------------------------------------------------------


_AUTH_HINTS = ("auth", "login", "credential", "unauthorized", "not logged in")


class CodexProvider:
    """One-shot caller for the local `codex` CLI.

    Each complete() call spawns a fresh `codex exec` process with a read-only
    sandbox and no approvals, so codex cannot touch the repo while producing a
    structured action decision.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self._model = model
        self._timeout = timeout if timeout is not None else _read_default_timeout()
        self._runner: SubprocessRunner = runner or _default_runner

    async def complete(self, prompt: str, *, schema: Mapping[str, object]) -> str:
        """Run one `codex exec` and return the schema-constrained payload string.

        Payload-to-domain-model parsing is the caller's responsibility.
        """
        schema_json = json.dumps(schema, separators=(",", ":"))
        with tempfile.TemporaryDirectory(prefix="autouser-codex-") as tmpdir:
            schema_path = os.path.join(tmpdir, "schema.json")
            output_path = os.path.join(tmpdir, "message.txt")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write(schema_json)

            argv = build_command(
                prompt,
                schema_path=schema_path,
                output_path=output_path,
                model=self._model,
            )
            try:
                async with _spawn_gate():
                    completed = await self._runner(argv, self._timeout)
            except FileNotFoundError as exc:
                raise CodexNotFoundError(
                    "`codex` binary not on PATH. Install the Codex CLI and run "
                    "`codex login` once to authenticate."
                ) from exc
            except asyncio.TimeoutError as exc:
                # MUST precede `except OSError`: asyncio.TimeoutError IS the
                # builtin TimeoutError (an OSError subclass) in 3.11+.
                raise CodexTimeoutError(
                    f"`codex` did not return within {self._timeout:.0f}s; retry "
                    "or raise AUTOUSER_CODEX_TIMEOUT if this recurs"
                ) from exc
            except OSError as exc:
                # fork() refused at the per-uid ceiling raises EPERM/EAGAIN
                # (PermissionError/BlockingIOError). Gate on errno, not class,
                # so EACCES (non-executable codex) re-raises unchanged.
                if exc.errno in (errno.EPERM, errno.EAGAIN):
                    raise CodexProcessLimitError(_PROCESS_LIMIT_MESSAGE) from exc
                raise

            if completed.returncode != 0:
                if _is_process_limit_signature(completed.stderr):
                    raise CodexProcessLimitError(
                        f"{_PROCESS_LIMIT_MESSAGE} stderr={completed.stderr!r}"
                    )
                stderr_lc = completed.stderr.lower()
                if any(hint in stderr_lc for hint in _AUTH_HINTS):
                    raise CodexAuthError(
                        "`codex` CLI is not authenticated. Run `codex login`. "
                        f"stderr={completed.stderr!r}"
                    )
                raise CodexInvocationError(
                    f"`codex` exited with code {completed.returncode}. "
                    f"stderr={completed.stderr!r}"
                )

            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    message = f.read()
            except OSError as exc:
                raise CodexOutputError(
                    "`codex` exited 0 but wrote no final-message file"
                ) from exc
            return parse_output(message)
