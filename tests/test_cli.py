"""CLI surface tests: provider resolution, task-spec wiring, persona lookup.

Pure-config tests — no browsers, no subprocesses, no LLM calls. The run-loop
itself is covered by tests/test_scripted_engine.py; these pin the CLI's
translation layer so `autouser run` can't silently drift from the library API.
"""

from __future__ import annotations

import pytest

from autouser.cli import (
    CliConfigError,
    _resolve_personas,
    build_parser,
    build_task_spec,
    resolve_provider,
)
from autouser.persona.registry import ARCHETYPES


# --- resolve_provider --------------------------------------------------------


def test_explicit_provider_wins_over_everything(monkeypatch):
    env = {"AUTOUSER_PROVIDER": "gemini", "ANTHROPIC_API_KEY": "k"}
    assert resolve_provider("claude_code", env=env) == "claude_code"


def test_explicit_unknown_provider_rejected():
    with pytest.raises(CliConfigError, match="unknown provider"):
        resolve_provider("openai", env={})


def test_env_provider_honored():
    assert resolve_provider(None, env={"AUTOUSER_PROVIDER": "gemini"}) == "gemini"


def test_env_provider_invalid_rejected_not_silently_replaced():
    with pytest.raises(CliConfigError, match="AUTOUSER_PROVIDER"):
        resolve_provider(None, env={"AUTOUSER_PROVIDER": "gpt4"})


def test_anthropic_key_beats_gemini_key():
    env = {"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g"}
    assert resolve_provider(None, env=env) == "anthropic"


def test_gemini_key_detected():
    assert resolve_provider(None, env={"GEMINI_API_KEY": "g"}) == "gemini"


def test_claude_binary_fallback(monkeypatch):
    monkeypatch.setattr("autouser.cli.shutil.which", lambda _: "/usr/bin/claude")
    assert resolve_provider(None, env={}) == "claude_code"


def test_no_provider_available_is_config_error(monkeypatch):
    monkeypatch.setattr("autouser.cli.shutil.which", lambda _: None)
    with pytest.raises(CliConfigError, match="no usable LLM provider"):
        resolve_provider(None, env={})


# --- build_task_spec ---------------------------------------------------------


def _parse_run(extra: list[str]):
    return build_parser().parse_args(
        ["run", "--url", "https://x.test/", "--task", "t", "--criteria", "c", *extra]
    )


def test_task_spec_without_predicate():
    spec = build_task_spec(_parse_run([]))
    assert spec.success_predicate is None
    assert spec.max_steps == 15
    assert spec.max_consecutive_failures == 3


def test_task_spec_with_text_predicate():
    spec = build_task_spec(_parse_run(["--success-text", "Order placed."]))
    assert spec.success_predicate is not None
    assert spec.success_predicate.text_contains == "Order placed."
    assert spec.success_predicate.url_pattern is None


def test_task_spec_combines_predicate_fields():
    spec = build_task_spec(
        _parse_run(
            ["--success-text", "Done", "--success-url-pattern", r"/inventory\.html"]
        )
    )
    assert spec.success_predicate.text_contains == "Done"
    assert spec.success_predicate.url_pattern == r"/inventory\.html"


# --- study subcommand ----------------------------------------------------------


def test_study_parser_shares_task_args_and_adds_seeds():
    args = build_parser().parse_args(
        ["study", "--url", "https://x.test/", "--task", "t", "--criteria", "c",
         "--persona", "maria", "--persona", "carlos", "--seeds", "5"]
    )
    assert args.command == "study"
    assert args.seeds == 5
    assert args.persona == ["maria", "carlos"]
    assert args.max_steps == 15  # shared defaults intact


def test_study_seeds_default_three():
    args = build_parser().parse_args(
        ["study", "--url", "https://x.test/", "--task", "t", "--criteria", "c"]
    )
    assert args.seeds == 3


# --- score subcommand ----------------------------------------------------------


def test_score_parser_accepts_thresholds():
    args = build_parser().parse_args(
        ["score", "--benchmark", "b.yaml", "--study", "s.json",
         "--min-precision", "0.7", "--min-recall", "0.5"]
    )
    assert args.command == "score"
    assert args.benchmark == "b.yaml"
    assert args.min_precision == 0.7
    assert args.min_recall == 0.5


def test_audit_flag_defaults_off_on_run():
    args = build_parser().parse_args(
        ["run", "--url", "https://x.test/", "--task", "t", "--criteria", "c"]
    )
    assert args.audit is False


def test_fast_flag_defaults_off_and_parses_on_run_and_study():
    base = ["--url", "https://x.test/", "--task", "t", "--criteria", "c"]
    assert build_parser().parse_args(["run", *base]).fast is False
    assert build_parser().parse_args(["run", *base, "--fast"]).fast is True
    assert build_parser().parse_args(["study", *base, "--fast"]).fast is True


# --- PR #45 review fixes -------------------------------------------------------


def test_perception_accepts_string_values_not_enum_repr():
    """PR#45 finding 3: --perception choices are the value strings, so the
    help/error text is `dom`/`computer_use`, and passing those works."""
    from autouser.runner import PerceptionMode

    base = ["run", "--url", "x", "--task", "t", "--criteria", "c"]
    assert build_parser().parse_args(base).perception == PerceptionMode.DOM
    cu = build_parser().parse_args([*base, "--perception", "computer_use"])
    assert cu.perception == PerceptionMode.COMPUTER_USE
    # The enum-repr string a user might copy from a broken help line is rejected.
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--perception", "PerceptionMode.DOM"])


def test_computer_use_engine_receives_model(monkeypatch):
    """PR#45 finding 2: --model must reach the vision engine, not be dropped
    in favor of ComputerUseEngine's default."""
    import asyncio

    import autouser.cli as cli

    captured = {}

    class _FakeRunner:
        def __init__(self, *, persona, task_spec, screenshot_dir, perception_mode, engine):
            captured["engine"] = engine

        async def run(self):
            from autouser.report.models import FrictionLog, TaskOutcome

            return FrictionLog(
                task="t", success_criteria="c", persona_name="maria",
                persona_summary="s", outcome=TaskOutcome.ABANDONED,
                total_steps=0, steps=[], issues=[], overall_impressions="",
            )

    monkeypatch.setattr(cli, "SimulationRunner", _FakeRunner)

    from autouser.persona.models import Persona
    from autouser.persona.registry import get_archetype
    from autouser.runner import PerceptionMode
    from autouser.session import TaskSpec

    spec = TaskSpec(task="t", success_criteria="c", start_url="https://x.test/")
    asyncio.run(cli._run_one(
        Persona.from_archetype(get_archetype("maria")), spec,
        provider="anthropic", model="claude-opus-4-8",
        perception=PerceptionMode.COMPUTER_USE, out_dir=__import__("pathlib").Path("/tmp/cu_model_test"),
    ))
    from autouser.cognitive.computer_use import ComputerUseEngine

    assert isinstance(captured["engine"], ComputerUseEngine)
    assert captured["engine"].model == "claude-opus-4-8"


# --- personas ----------------------------------------------------------------


def test_resolve_personas_known_names():
    personas = _resolve_personas(["maria", "jake"])
    assert [p.archetype.name for p in personas] == ["maria", "jake"]


def test_resolve_personas_unknown_name_lists_options():
    with pytest.raises(CliConfigError, match="unknown persona"):
        _resolve_personas(["bob"])


def test_parser_default_persona_is_applied_in_run_command():
    # The parser leaves --persona as None; _run_command defaults it to maria.
    args = _parse_run([])
    assert args.persona is None


def test_all_registry_personas_resolve():
    personas = _resolve_personas(sorted(ARCHETYPES))
    assert len(personas) == len(ARCHETYPES)
