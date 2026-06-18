"""Engine-wiring unit tests for the claude_code provider branch.

Verifies the dispatch path through `_call_llm_once` invokes
ClaudeCodeProvider with the right schema for each phase. Subprocess
is mocked end-to-end via the provider's `runner` injection point.
"""

from __future__ import annotations

import json

import pytest

from autouser.cognitive.claude_code_provider import (
    ClaudeCodeProvider,
    CompletedSubprocess,
)
from autouser.cognitive.engine import (
    CognitiveEngine,
    _PLAN_SCHEMA,
    _REFLECT_SCHEMA,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


def _mock_runner_returning(payload: dict) -> tuple[list[list[str]], object]:
    """Returns (captured_argv_list, runner). Runner emits the payload wrapped
    in the CLI envelope. Captured argv exposes the locked-flag check."""
    captured: list[list[str]] = []

    async def runner(argv, timeout):
        captured.append(list(argv))
        envelope = json.dumps({"result": json.dumps(payload)})
        return CompletedSubprocess(returncode=0, stdout=envelope, stderr="")

    return captured, runner


@pytest.fixture
def maria_engine_claude_code() -> CognitiveEngine:
    persona = Persona.from_archetype(get_archetype("maria"))
    return CognitiveEngine(
        persona, "Sign up", "See the dashboard", provider="claude_code",
    )


def test_engine_instantiates_claude_code_client(maria_engine_claude_code) -> None:
    assert isinstance(maria_engine_claude_code._claude_code_client, ClaudeCodeProvider)
    assert maria_engine_claude_code._client is None
    assert maria_engine_claude_code._gemini_client is None


def test_default_model_is_sonnet(maria_engine_claude_code) -> None:
    assert maria_engine_claude_code.model == "sonnet"


@pytest.mark.asyncio
async def test_plan_dispatches_to_claude_code_with_plan_schema(
    maria_engine_claude_code,
) -> None:
    captured, runner = _mock_runner_returning({
        "action": "click",
        "target": "button#go",
        "input_value": None,
        "persona_thought": "I'll click this.",
        "expected_outcome": "The page advances.",
        "confidence": 0.6,
        "recognition": "THINK_SO",
        "prediction": "THINK_SO",
        "progress": "THINK_SO",
    })
    maria_engine_claude_code._claude_code_client = ClaudeCodeProvider(
        model="sonnet", runner=runner,
    )

    from autouser.cognitive.models import UIState
    ui = UIState(
        url="https://example.com/", page_title="Home",
        dom_summary="button#go - 'Go'", visible_text="Go",
    )
    intent = await maria_engine_claude_code.plan(ui)

    assert intent.action.value == "click"
    assert len(captured) == 1
    argv = captured[0]
    # Plan schema went through to the CLI.
    schema_arg = argv[argv.index("--json-schema") + 1]
    assert json.loads(schema_arg) == _PLAN_SCHEMA
    # Locked flags still hold under the engine dispatch path.
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert "--bare" not in argv


@pytest.mark.asyncio
async def test_reflect_dispatches_to_claude_code_with_reflect_schema(
    maria_engine_claude_code,
) -> None:
    captured, runner = _mock_runner_returning({
        "actual_outcome": "Nothing happened.",
        "mismatch": True,
        "reflection": "I'm confused.",
        "emotion": "confused",
        "severity": "medium",
        "category": "feedback",
        "persona_believes_complete": False,
        "observed_success_signal": False,
    })
    maria_engine_claude_code._claude_code_client = ClaudeCodeProvider(
        model="sonnet", runner=runner,
    )

    from autouser.cognitive.models import ActionIntent, ActionType, UIState
    intent = ActionIntent(
        action=ActionType.CLICK, target="button#go",
        persona_thought="...", expected_outcome="...", confidence=0.5,
    )
    ui_before = UIState(url="x", page_title="t", dom_summary="", visible_text="")
    ui_after = UIState(url="x", page_title="t", dom_summary="", visible_text="")
    result = await maria_engine_claude_code.reflect(
        intent, ui_after,
        observation_before=ui_before, observation_after=ui_after,
        step_number=1,
    )

    assert result.mismatch is True
    argv = captured[0]
    schema_arg = argv[argv.index("--json-schema") + 1]
    assert json.loads(schema_arg) == _REFLECT_SCHEMA


def test_detect_provider_honors_explicit_env(monkeypatch) -> None:
    from autouser.cognitive.engine import _detect_provider
    monkeypatch.setenv("AUTOUSER_PROVIDER", "claude_code")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "would-otherwise-win")
    assert _detect_provider() == "claude_code"


def test_detect_provider_ignores_unknown_explicit(monkeypatch) -> None:
    from autouser.cognitive.engine import _detect_provider
    monkeypatch.setenv("AUTOUSER_PROVIDER", "bogus")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _detect_provider() == "anthropic"


@pytest.mark.asyncio
async def test_claude_code_dispatch_with_uninitialized_client_raises_runtime(
    maria_engine_claude_code,
) -> None:
    """F3 regression (distinguished-eng PR #39 static-diff): the runtime
    invariant must hold under `python -O` too. Assertions get stripped;
    an explicit RuntimeError doesn't.
    """
    maria_engine_claude_code._claude_code_client = None
    with pytest.raises(RuntimeError, match="partition violated"):
        await maria_engine_claude_code._call_llm_once(
            "sys", "user", max_tokens=64, schema=_PLAN_SCHEMA,
        )
