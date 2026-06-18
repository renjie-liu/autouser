"""Tests for the codex CLI provider (pure; subprocess is mocked)."""

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
