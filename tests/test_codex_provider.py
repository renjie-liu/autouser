"""Tests for the codex CLI provider (pure; subprocess is mocked)."""

import asyncio
import json

import pytest

from autouser.cognitive.codex_provider import (
    CompletedSubprocess,
    CodexOutputError,
    build_command,
    parse_output,
    _is_process_limit_signature,
    _read_default_timeout,
    _read_max_spawns,
)


def test_build_command_has_guardrails_and_paths():
    argv = build_command("hello", schema_path="/tmp/s.json", output_path="/tmp/o.txt")
    assert argv[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in argv
    assert argv[argv.index("--output-schema") + 1] == "/tmp/s.json"
    assert argv[argv.index("--output-last-message") + 1] == "/tmp/o.txt"
    assert 'sandbox_mode="read-only"' in argv
    assert 'approval_policy="never"' in argv
    # one-shot: no session file persisted to disk (claude_code --no-session-persistence analog)
    assert "--ephemeral" in argv
    # prompt is positional after the `--` option terminator
    assert argv[-2] == "--"
    assert argv[-1] == "hello"
    # no --model unless requested (defer to codex config)
    assert "--model" not in argv


def test_build_command_includes_model_when_given():
    argv = build_command("p", schema_path="/s", output_path="/o", model="o3")
    assert argv[argv.index("--model") + 1] == "o3"
    # guardrails must remain present even when a model is supplied
    assert "--skip-git-repo-check" in argv
    assert "--ephemeral" in argv
    assert 'sandbox_mode="read-only"' in argv
    assert 'approval_policy="never"' in argv


def test_build_command_empty_model_omits_flag():
    argv = build_command("p", schema_path="/s", output_path="/o", model="")
    assert "--model" not in argv


def test_parse_output_returns_stripped_payload():
    assert parse_output('  {"action": "click"}\n') == '{"action": "click"}'


def test_parse_output_empty_raises():
    with pytest.raises(CodexOutputError):
        parse_output("   \n")


def test_read_max_spawns_default_override_and_clamp(monkeypatch):
    monkeypatch.delenv("AUTOUSER_CODEX_MAX_SPAWNS", raising=False)
    assert _read_max_spawns() == 1
    monkeypatch.setenv("AUTOUSER_CODEX_MAX_SPAWNS", "4")
    assert _read_max_spawns() == 4
    monkeypatch.setenv("AUTOUSER_CODEX_MAX_SPAWNS", "0")
    assert _read_max_spawns() == 1  # a cap of 0 would deadlock; clamp to 1
    monkeypatch.setenv("AUTOUSER_CODEX_MAX_SPAWNS", "junk")
    assert _read_max_spawns() == 1


def test_read_default_timeout(monkeypatch):
    monkeypatch.delenv("AUTOUSER_CODEX_TIMEOUT", raising=False)
    assert _read_default_timeout() == 120.0
    monkeypatch.setenv("AUTOUSER_CODEX_TIMEOUT", "30")
    assert _read_default_timeout() == 30.0
    monkeypatch.setenv("AUTOUSER_CODEX_TIMEOUT", "-5")
    assert _read_default_timeout() == 120.0  # non-positive falls back
    monkeypatch.setenv("AUTOUSER_CODEX_TIMEOUT", "junk")
    assert _read_default_timeout() == 120.0  # non-numeric falls back


def test_process_limit_signature():
    assert _is_process_limit_signature("fork: Operation not permitted (os error 1)")
    assert _is_process_limit_signature("resource temporarily unavailable (os error 11)")
    assert not _is_process_limit_signature("API error: resource temporarily unavailable")


def test_completed_subprocess_is_frozen():
    cs = CompletedSubprocess(returncode=0, stdout="x", stderr="")
    assert cs.returncode == 0
    with pytest.raises(Exception):
        cs.returncode = 1  # frozen dataclass


# ---------------------------------------------------------------------------
# Task 2: CodexProvider.complete
# ---------------------------------------------------------------------------

import errno  # noqa: E402

from autouser.cognitive.codex_provider import (  # noqa: E402
    CodexProvider,
    CodexNotFoundError,
    CodexAuthError,
    CodexTimeoutError,
    CodexInvocationError,
    CodexProcessLimitError,
)

_SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}}}


def _runner_writing(payload: str, *, returncode: int = 0, stderr: str = ""):
    """A fake runner that writes `payload` to the --output-last-message path,
    mimicking what real codex does, then returns the given exit/stderr."""

    async def runner(argv, timeout):
        out = argv[argv.index("--output-last-message") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write(payload)
        return CompletedSubprocess(returncode=returncode, stdout="", stderr=stderr)

    return runner


async def test_complete_returns_payload():
    provider = CodexProvider(runner=_runner_writing('{"action": "click"}'))
    out = await provider.complete("decide", schema=_SCHEMA)
    assert json.loads(out) == {"action": "click"}


async def test_complete_passes_schema_and_model():
    seen = {}

    async def runner(argv, timeout):
        seen["argv"] = list(argv)
        schema_path = argv[argv.index("--output-schema") + 1]
        with open(schema_path, encoding="utf-8") as f:
            seen["schema"] = json.loads(f.read())
        out = argv[argv.index("--output-last-message") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write('{"action": "wait"}')
        return CompletedSubprocess(0, "", "")

    provider = CodexProvider(model="o3", runner=runner)
    await provider.complete("decide", schema=_SCHEMA)
    assert seen["schema"] == _SCHEMA
    assert seen["argv"][seen["argv"].index("--model") + 1] == "o3"


async def test_complete_missing_binary_raises_not_found():
    async def runner(argv, timeout):
        raise FileNotFoundError("codex")

    with pytest.raises(CodexNotFoundError):
        await CodexProvider(runner=runner).complete("p", schema=_SCHEMA)


async def test_complete_timeout_raises_typed():
    async def runner(argv, timeout):
        raise asyncio.TimeoutError()

    with pytest.raises(CodexTimeoutError):
        await CodexProvider(runner=runner).complete("p", schema=_SCHEMA)


async def test_complete_auth_failure_from_stderr():
    with pytest.raises(CodexAuthError):
        await CodexProvider(
            runner=_runner_writing("", returncode=1, stderr="Error: not logged in")
        ).complete("p", schema=_SCHEMA)


async def test_complete_nonzero_is_invocation_error():
    with pytest.raises(CodexInvocationError):
        await CodexProvider(
            runner=_runner_writing("", returncode=2, stderr="bad flag")
        ).complete("p", schema=_SCHEMA)


async def test_complete_process_limit_from_stderr_signature():
    with pytest.raises(CodexProcessLimitError):
        await CodexProvider(
            runner=_runner_writing(
                "", returncode=1, stderr="fork: Operation not permitted (os error 1)"
            )
        ).complete("p", schema=_SCHEMA)


async def test_complete_eperm_oserror_is_process_limit():
    async def runner(argv, timeout):
        raise PermissionError(errno.EPERM, "fork refused")

    with pytest.raises(CodexProcessLimitError):
        await CodexProvider(runner=runner).complete("p", schema=_SCHEMA)


async def test_complete_eagain_oserror_is_process_limit():
    async def runner(argv, timeout):
        raise BlockingIOError(errno.EAGAIN, "resource temporarily unavailable")

    with pytest.raises(CodexProcessLimitError):
        await CodexProvider(runner=runner).complete("p", schema=_SCHEMA)


async def test_complete_eacces_oserror_reraises():
    """EACCES (e.g. a present-but-non-executable codex) is NOT the per-uid
    ceiling — it must re-raise unchanged, not be mis-stamped as a process-limit
    error (the false-localization the errno gate exists to prevent)."""
    async def runner(argv, timeout):
        raise PermissionError(errno.EACCES, "permission denied")
    with pytest.raises(PermissionError):
        await CodexProvider(runner=runner).complete("p", schema=_SCHEMA)


async def test_complete_empty_output_raises_output_error():
    with pytest.raises(CodexOutputError):
        await CodexProvider(runner=_runner_writing("   ")).complete("p", schema=_SCHEMA)


async def test_spawn_gate_caps_concurrency(monkeypatch):
    monkeypatch.setenv("AUTOUSER_CODEX_MAX_SPAWNS", "1")
    import autouser.cognitive.codex_provider as cp

    cp._spawn_semaphore = None  # force rebuild under this loop with the new cap
    active = 0
    peak = 0

    async def runner(argv, timeout):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        out = argv[argv.index("--output-last-message") + 1]
        open(out, "w").write('{"action": "x"}')
        return CompletedSubprocess(0, "", "")

    provider = CodexProvider(runner=runner)
    await asyncio.gather(*[provider.complete("p", schema=_SCHEMA) for _ in range(3)])
    assert peak == 1
