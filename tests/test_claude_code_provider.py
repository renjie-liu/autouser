"""Mocked-subprocess unit tests for ClaudeCodeProvider.

Subprocess is injected via the `runner` constructor argument; no real
`claude` invocation in this file. The opt-in/skipped-by-default live
smoke test lives in tests/test_claude_code_provider_smoke.py and is
guarded by AUTOUSER_CLAUDE_CODE_SMOKE=1.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import autouser.cognitive.claude_code_provider as ccp
from autouser.cognitive.claude_code_provider import (
    ClaudeCodeAuthError,
    ClaudeCodeEnvelopeError,
    ClaudeCodeInvocationError,
    ClaudeCodeNotFoundError,
    ClaudeCodeProcessLimitError,
    ClaudeCodeProvider,
    ClaudeCodeResponseError,
    ClaudeCodeTimeoutError,
    CompletedSubprocess,
    build_command,
    parse_envelope,
)


_SCHEMA: dict[str, object] = {"type": "object", "properties": {"action": {"type": "string"}}}


# --- build_command ----------------------------------------------------------


def test_build_command_includes_locked_flags() -> None:
    argv = build_command("hello", schema=_SCHEMA)
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert "--no-session-persistence" in argv
    i = argv.index("--tools")
    assert argv[i + 1] == ""
    assert "--model" in argv and argv[argv.index("--model") + 1] == "sonnet"
    assert "--json-schema" in argv


def test_build_command_excludes_bare_flag() -> None:
    """--bare requires ANTHROPIC_API_KEY and bypasses keychain/OAuth — never
    use it in this provider (an earlier preflight ruling)."""
    argv = build_command("hello", schema=_SCHEMA)
    assert "--bare" not in argv


def test_build_command_serializes_schema_compactly() -> None:
    argv = build_command("hello", schema=_SCHEMA)
    schema_arg = argv[argv.index("--json-schema") + 1]
    # Compact separators — no whitespace — so the argv stays cheap to log.
    assert " " not in schema_arg
    assert json.loads(schema_arg) == _SCHEMA


def test_build_command_prompt_is_last() -> None:
    argv = build_command("prompt-text", schema=_SCHEMA)
    assert argv[-1] == "prompt-text"


def test_build_command_separates_prompt_with_double_dash() -> None:
    """Defensive: `--tools ""` could absorb the prompt if --tools is variadic
    in some CLI version. `--` ends option parsing so the prompt is positional.
    (an earlier PR #39 boundary watch-item)."""
    argv = build_command("prompt-text", schema=_SCHEMA)
    dash = argv.index("--")
    assert argv[dash + 1] == "prompt-text"
    # `--tools ""` pair must end before the `--` separator.
    tools_idx = argv.index("--tools")
    assert tools_idx + 1 < dash
    assert argv[tools_idx + 1] == ""


def test_build_command_model_override() -> None:
    argv = build_command("x", schema=_SCHEMA, model="opus")
    assert argv[argv.index("--model") + 1] == "opus"


# --- parse_envelope --------------------------------------------------------


def test_parse_envelope_extracts_string_result() -> None:
    raw = json.dumps({"result": '{"action":"click"}', "cost_usd": 0.001})
    assert parse_envelope(raw) == '{"action":"click"}'


def test_parse_envelope_reserializes_object_result() -> None:
    raw = json.dumps({"result": {"action": "click"}})
    payload = parse_envelope(raw)
    assert json.loads(payload) == {"action": "click"}


def test_parse_envelope_raises_on_invalid_json() -> None:
    with pytest.raises(ClaudeCodeEnvelopeError, match="not valid JSON"):
        parse_envelope("not-json")


def test_parse_envelope_raises_on_missing_result_field() -> None:
    raw = json.dumps({"session_id": "abc"})
    with pytest.raises(ClaudeCodeEnvelopeError, match="missing `result`"):
        parse_envelope(raw)


def test_parse_envelope_prefers_structured_output_over_result() -> None:
    """Under `--json-schema` the CLI puts the schema-conformant object in
    `structured_output` and leaves `result` empty/prose. Reading `result` only
    returned '' → json.loads('') → engine defaulted every step to `wait` (the
    s06 parity-run defect). The schema payload must win. Verified live: CLI
    2.1.158 with --json-schema returns result='' alongside structured_output.
    """
    raw = json.dumps(
        {"result": "", "structured_output": {"action": "click", "target": "#login"}}
    )
    assert json.loads(parse_envelope(raw)) == {"action": "click", "target": "#login"}


def test_parse_envelope_structured_output_string_passthrough() -> None:
    raw = json.dumps({"result": "ignored prose", "structured_output": '{"a":1}'})
    assert parse_envelope(raw) == '{"a":1}'


def test_parse_envelope_falls_back_to_result_when_no_structured_output() -> None:
    """No structured_output (e.g. mocked runners, non-schema responses) → the
    original `result` path still works. Backward-compatible by construction.
    """
    raw = json.dumps({"result": '{"action":"wait"}'})
    assert parse_envelope(raw) == '{"action":"wait"}'


def test_parse_envelope_raises_on_envelope_is_error_true() -> None:
    """Regression: an earlier PR #39 review — `is_error: true` envelopes must surface
    as a typed provider error, not flow through as a valid payload."""
    raw = json.dumps({"is_error": True, "result": "schema validation failed"})
    with pytest.raises(ClaudeCodeResponseError, match="schema validation failed"):
        parse_envelope(raw)


def test_parse_envelope_is_error_without_detail() -> None:
    raw = json.dumps({"is_error": True})
    with pytest.raises(ClaudeCodeResponseError, match="<no detail>"):
        parse_envelope(raw)


def test_parse_envelope_rejects_non_object_top_level() -> None:
    raw = json.dumps(["not", "an", "object"])
    with pytest.raises(ClaudeCodeEnvelopeError, match="not a JSON object"):
        parse_envelope(raw)


def test_parse_envelope_is_error_false_passes_through() -> None:
    """Sanity: explicit `is_error: false` is not the error path."""
    raw = json.dumps({"is_error": False, "result": '{"ok": true}'})
    assert parse_envelope(raw) == '{"ok": true}'


# --- Provider error translation -------------------------------------------


def _runner_returning(returncode: int, stdout: str = "", stderr: str = ""):
    async def runner(argv, timeout):
        return CompletedSubprocess(returncode=returncode, stdout=stdout, stderr=stderr)
    return runner


def _runner_raising(exc: BaseException):
    async def runner(argv, timeout):
        raise exc
    return runner


@pytest.mark.asyncio
async def test_missing_binary_raises_not_found() -> None:
    provider = ClaudeCodeProvider(runner=_runner_raising(FileNotFoundError()))
    with pytest.raises(ClaudeCodeNotFoundError, match="claude.*PATH"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error() -> None:
    provider = ClaudeCodeProvider(
        runner=_runner_raising(asyncio.TimeoutError()), timeout=1.0,
    )
    with pytest.raises(ClaudeCodeTimeoutError, match="1s"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_nonzero_exit_with_auth_stderr_raises_auth_error() -> None:
    provider = ClaudeCodeProvider(
        runner=_runner_returning(1, stderr="Error: not logged in. Run `claude` to authenticate."),
    )
    with pytest.raises(ClaudeCodeAuthError, match="not authenticated"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_nonzero_exit_without_auth_hint_raises_invocation_error() -> None:
    provider = ClaudeCodeProvider(
        runner=_runner_returning(2, stderr="unexpected internal failure"),
    )
    with pytest.raises(ClaudeCodeInvocationError, match="code 2"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_happy_path_returns_payload_string() -> None:
    envelope = json.dumps({"result": '{"action":"click","target":"#go"}'})
    provider = ClaudeCodeProvider(runner=_runner_returning(0, stdout=envelope))
    payload = await provider.complete("hi", schema=_SCHEMA)
    assert json.loads(payload) == {"action": "click", "target": "#go"}


@pytest.mark.asyncio
async def test_default_runner_timeout_does_not_deadlock_on_full_pipes() -> None:
    """Regression: an earlier PR #39 review — when the child fills the stdout PIPE
    buffer before being killed, the old `proc.kill(); await proc.wait()`
    sequence could hang because wait() blocks on the OS reaping fds while
    the unread buffer stays open. Fix uses communicate() to drain after kill.

    The test child prints continuously to stdout; without the fix, this test
    would hang well past the asserted wall-clock bound until the OS killed
    the test runner.
    """
    import sys
    from autouser.cognitive.claude_code_provider import _default_runner

    argv = [
        sys.executable, "-u", "-c",
        "import sys, time\n"
        "while True:\n"
        "    sys.stdout.write('x' * 1024 + '\\n')\n"
        "    sys.stdout.flush()\n",
    ]
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await _default_runner(argv, timeout=0.3)
    elapsed = loop.time() - start
    assert elapsed < 5.0, f"runner deadlocked on full pipe (elapsed={elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_provider_uses_configured_model_in_argv() -> None:
    captured: dict[str, list[str]] = {}

    async def runner(argv, timeout):
        captured["argv"] = list(argv)
        return CompletedSubprocess(
            returncode=0, stdout=json.dumps({"result": "{}"}), stderr=""
        )

    provider = ClaudeCodeProvider(model="opus", runner=runner)
    await provider.complete("hi", schema=_SCHEMA)
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == "opus"


# --- carry #4: process-global spawn cap ------------------------------------


def _reset_spawn_gate() -> None:
    """Force the module-global gate to rebuild on next use, so a test reads the
    current env / loop rather than a semaphore another test bound."""
    ccp._spawn_semaphore = None
    ccp._spawn_semaphore_loop = None


class _ConcurrencyProbe:
    """Records peak concurrency of runner calls that pass the spawn gate."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.release = asyncio.Event()

    def runner(self):
        async def _runner(argv, timeout):
            self.current += 1
            self.peak = max(self.peak, self.current)
            await self.release.wait()  # hold past the gate until released
            self.current -= 1
            return CompletedSubprocess(0, json.dumps({"result": "{}"}), "")
        return _runner


@pytest.mark.asyncio
async def test_spawn_gate_caps_concurrency_at_default_one(monkeypatch) -> None:
    """Default cap is 1: even with 4 concurrent complete() calls, only one
    runner is past the gate at a time. This is the structural bound that
    survives any caller fanning out — not a property of the test running
    sequentially (carry #4)."""
    monkeypatch.delenv("AUTOUSER_CLAUDE_CODE_MAX_SPAWNS", raising=False)
    _reset_spawn_gate()
    probe = _ConcurrencyProbe()
    provider = ClaudeCodeProvider(runner=probe.runner())

    tasks = [
        asyncio.create_task(provider.complete("x", schema=_SCHEMA))
        for _ in range(4)
    ]
    await asyncio.sleep(0.05)  # let all four contend for the gate
    assert probe.current == 1, "more than one runner passed a cap-1 gate"
    assert probe.peak == 1

    probe.release.set()
    await asyncio.gather(*tasks)
    assert probe.peak == 1


@pytest.mark.asyncio
async def test_spawn_gate_respects_env_override(monkeypatch) -> None:
    """AUTOUSER_CLAUDE_CODE_MAX_SPAWNS raises the cap. At 2, peak concurrency
    reaches exactly 2 — confirming the dial is real and process-global."""
    monkeypatch.setenv("AUTOUSER_CLAUDE_CODE_MAX_SPAWNS", "2")
    _reset_spawn_gate()
    probe = _ConcurrencyProbe()
    provider = ClaudeCodeProvider(runner=probe.runner())

    tasks = [
        asyncio.create_task(provider.complete("x", schema=_SCHEMA))
        for _ in range(4)
    ]
    await asyncio.sleep(0.05)
    assert probe.peak == 2, "cap=2 did not allow exactly two concurrent runners"
    assert probe.current == 2

    probe.release.set()
    await asyncio.gather(*tasks)
    _reset_spawn_gate()  # don't leak the raised cap into later tests


@pytest.mark.asyncio
async def test_spawn_gate_is_shared_across_provider_instances(monkeypatch) -> None:
    """The cap must be PROCESS-GLOBAL, not per-instance: two separate
    ClaudeCodeProvider objects still share the single budget. A per-instance
    semaphore would bound nothing when each parity leg builds its own
    provider (senior-eng's load-bearing catch)."""
    monkeypatch.delenv("AUTOUSER_CLAUDE_CODE_MAX_SPAWNS", raising=False)
    _reset_spawn_gate()
    probe = _ConcurrencyProbe()
    provider_a = ClaudeCodeProvider(runner=probe.runner())
    provider_b = ClaudeCodeProvider(runner=probe.runner())

    tasks = [
        asyncio.create_task(provider_a.complete("x", schema=_SCHEMA)),
        asyncio.create_task(provider_b.complete("y", schema=_SCHEMA)),
    ]
    await asyncio.sleep(0.05)
    assert probe.peak == 1, "two providers exceeded the shared cap-1 gate"

    probe.release.set()
    await asyncio.gather(*tasks)


# --- carry #5: typed EPERM / process-limit discrimination ------------------


@pytest.mark.asyncio
async def test_spawn_permission_error_eperm_raises_process_limit(monkeypatch) -> None:
    """fork() refused with PermissionError(EPERM) at spawn → typed
    ClaudeCodeProcessLimitError, NOT NotFound/Invocation."""
    monkeypatch.delenv("AUTOUSER_CLAUDE_CODE_MAX_SPAWNS", raising=False)
    _reset_spawn_gate()
    exc = PermissionError(errno_eperm(), "Operation not permitted")
    provider = ClaudeCodeProvider(runner=_runner_raising(exc))
    with pytest.raises(ClaudeCodeProcessLimitError, match="per-process limit"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_spawn_permission_error_eacces_is_not_process_limit(monkeypatch) -> None:
    """REGRESSION (senior-eng PR 2 review): PermissionError covers BOTH EPERM
    (fork refused — the ceiling) AND EACCES (e.g. a present-but-non-executable
    `claude`). The old broad `except PermissionError` mislabeled EACCES as a
    process-limit failure — the exact false-localization carry #5 exists to
    kill. EACCES must propagate unchanged, NOT become ProcessLimitError. The
    discriminator is errno, never the exception class."""
    monkeypatch.delenv("AUTOUSER_CLAUDE_CODE_MAX_SPAWNS", raising=False)
    _reset_spawn_gate()
    exc = PermissionError(errno_eacces(), "Permission denied")
    provider = ClaudeCodeProvider(runner=_runner_raising(exc))
    with pytest.raises(PermissionError):
        await provider.complete("hi", schema=_SCHEMA)
    # And specifically NOT reclassified as a process-limit error.
    with pytest.raises(PermissionError):
        provider2 = ClaudeCodeProvider(runner=_runner_raising(PermissionError(errno_eacces(), "x")))
        try:
            await provider2.complete("hi", schema=_SCHEMA)
        except ClaudeCodeProcessLimitError as mislabeled:  # pragma: no cover
            raise AssertionError(
                "EACCES was mislabeled as a process-limit error"
            ) from mislabeled


@pytest.mark.asyncio
async def test_spawn_oserror_eagain_raises_process_limit() -> None:
    """RLIMIT_NPROC on Linux surfaces as OSError(EAGAIN) — same ceiling."""
    exc = OSError(errno_eagain(), "Resource temporarily unavailable")
    provider = ClaudeCodeProvider(runner=_runner_raising(exc))
    with pytest.raises(ClaudeCodeProcessLimitError):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_spawn_oserror_other_errno_is_not_reclassified() -> None:
    """A non-EPERM/EAGAIN OSError is NOT a process-limit failure — must
    propagate, not get mis-typed."""
    exc = OSError(errno_eio(), "I/O error")
    provider = ClaudeCodeProvider(runner=_runner_raising(exc))
    with pytest.raises(OSError):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_nonzero_exit_with_eperm_signature_raises_process_limit() -> None:
    """Near-instant nonzero exit carrying the 'Operation not permitted
    (os error 1)' stderr signature → process-limit error."""
    provider = ClaudeCodeProvider(
        runner=_runner_returning(1, stderr="agent: Operation not permitted (os error 1)"),
    )
    with pytest.raises(ClaudeCodeProcessLimitError, match="per-process limit"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_bare_code_one_is_not_mislabeled_as_process_limit() -> None:
    """THE false-localization guard: a plain exit code 1 with no EPERM
    signature (e.g. an application/auth-less failure) must stay a generic
    ClaudeCodeInvocationError, NOT get the process-limit message stamped on
    it. Discriminator is the signature, never the bare exit code."""
    provider = ClaudeCodeProvider(
        runner=_runner_returning(1, stderr="Error: invalid --flag value"),
    )
    with pytest.raises(ClaudeCodeInvocationError, match="code 1"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_nonzero_exit_with_eagain_os_error_signature_raises_process_limit() -> None:
    """RLIMIT_NPROC can surface as a near-instant nonzero exit whose stderr
    carries the OS-rendered EAGAIN form 'resource temporarily unavailable
    (os error 11)'. That OS-error rendering only comes from the kernel
    refusing the spawn → process-limit error."""
    provider = ClaudeCodeProvider(
        runner=_runner_returning(
            1, stderr="claude: Resource temporarily unavailable (os error 11)"
        ),
    )
    with pytest.raises(ClaudeCodeProcessLimitError, match="per-process limit"):
        await provider.complete("hi", schema=_SCHEMA)


@pytest.mark.asyncio
async def test_api_resource_unavailable_prose_is_not_process_limit() -> None:
    """THE carry #5 false-localization guard, one notch deeper (an earlier review):
    an APPLICATION-level message that merely CONTAINS the phrase 'resource
    temporarily unavailable' — without the '(os error N)' OS rendering — must
    stay a generic ClaudeCodeInvocationError. It is an upstream/API failure,
    NOT the kernel refusing fork(). Matching the bare phrase would mis-stamp
    the system process-limit message onto it — the exact misattribution this
    typed error exists to prevent."""
    provider = ClaudeCodeProvider(
        runner=_runner_returning(
            1, stderr="API error: resource temporarily unavailable, retry later"
        ),
    )
    with pytest.raises(ClaudeCodeInvocationError, match="code 1"):
        await provider.complete("hi", schema=_SCHEMA)


def errno_eperm() -> int:
    import errno
    return errno.EPERM


def errno_eacces() -> int:
    import errno
    return errno.EACCES


def errno_eagain() -> int:
    import errno
    return errno.EAGAIN


def errno_eio() -> int:
    import errno
    return errno.EIO
