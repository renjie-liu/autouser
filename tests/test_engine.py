"""Tests for CognitiveEngine plan() response parsing and integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autouser.cognitive.engine import CognitiveEngine, _SCAFFOLD_FIELDS
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    IssueCategory,
    Severity,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


@pytest.fixture
def maria_engine() -> CognitiveEngine:
    persona = Persona.from_archetype(get_archetype("maria"))
    return CognitiveEngine(persona, "Sign up for an account", "See the dashboard", provider="anthropic")


@pytest.fixture
def jake_engine() -> CognitiveEngine:
    persona = Persona.from_archetype(get_archetype("jake"))
    return CognitiveEngine(persona, "Sign up for an account", "See the dashboard", provider="anthropic")


@pytest.fixture
def sample_ui() -> UIState:
    return UIState(
        url="https://example.com",
        page_title="Example",
        dom_summary='button#submit "Submit"\na.login "Log In"',
        visible_text="Welcome to Example.",
    )


def _mock_llm_response(payload: dict) -> MagicMock:
    """Create a mock Anthropic API response with the given JSON payload."""
    text_block = MagicMock()
    text_block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [text_block]
    return response


# Full LLM response including transient scaffold fields
SAMPLE_LLM_RESPONSE = {
    "recognition": "THINK_SO",
    "prediction": "NOT_SURE",
    "progress": "THINK_SO",
    "action": "click",
    "target": "button#submit",
    "input_value": None,
    "persona_thought": "I see a Submit button so I'll try clicking it.",
    "expected_outcome": "The form submits.",
    "confidence": 0.45,
}


class TestPlanResponseParsing:
    async def test_scaffold_fields_stripped(self, maria_engine, sample_ui):
        """R/P/G fields must not appear in the returned ActionIntent."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            intent = await maria_engine.plan(sample_ui)

        # ActionIntent should not have scaffold fields
        intent_dict = intent.model_dump()
        for field in _SCAFFOLD_FIELDS:
            assert field not in intent_dict

    async def test_action_intent_fields_preserved(self, maria_engine, sample_ui):
        """Core ActionIntent fields must be correctly extracted."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            intent = await maria_engine.plan(sample_ui)

        assert intent.action == ActionType.CLICK
        assert intent.target == "button#submit"
        assert intent.input_value is None
        assert intent.persona_thought == "I see a Submit button so I'll try clicking it."
        assert intent.expected_outcome == "The form submits."
        assert intent.confidence == 0.45

    async def test_markdown_code_fences_handled(self, maria_engine, sample_ui):
        """LLM responses wrapped in ```json fences should still parse."""
        text_block = MagicMock()
        text_block.text = "```json\n" + json.dumps(SAMPLE_LLM_RESPONSE) + "\n```"
        response = MagicMock()
        response.content = [text_block]

        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            intent = await maria_engine.plan(sample_ui)

        assert intent.action == ActionType.CLICK

    async def test_step_count_increments(self, maria_engine, sample_ui):
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            await maria_engine.plan(sample_ui)
            assert maria_engine.step_count == 1
            await maria_engine.plan(sample_ui)
            assert maria_engine.step_count == 2

    async def test_navigate_action_accepted_for_low_tech(self, maria_engine, sample_ui):
        """No schema gating: navigate must be accepted even for low-tech personas."""
        nav_response = {
            **SAMPLE_LLM_RESPONSE,
            "action": "navigate",
            "target": "",
            "input_value": "https://example.com/help",
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(nav_response)
            intent = await maria_engine.plan(sample_ui)

        assert intent.action == ActionType.NAVIGATE
        assert intent.input_value == "https://example.com/help"

    async def test_constraint_violation_logged(self, maria_engine, sample_ui, caplog):
        """Low-tech persona using navigate should be logged as constraint violation."""
        nav_response = {
            **SAMPLE_LLM_RESPONSE,
            "action": "navigate",
            "target": "",
            "input_value": "https://example.com/help",
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(nav_response)
            import logging
            with caplog.at_level(logging.DEBUG):
                await maria_engine.plan(sample_ui)

        assert any("constraint_violation" in r.message for r in caplog.records)

    async def test_scaffold_fields_logged(self, maria_engine, sample_ui, caplog):
        """R/P/G values should be logged at debug level for diagnostics."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            import logging
            with caplog.at_level(logging.DEBUG):
                await maria_engine.plan(sample_ui)

        scaffold_logs = [r for r in caplog.records if "scaffold_" in r.message]
        assert len(scaffold_logs) == 3  # recognition, prediction, progress

    async def test_model_and_temperature_passed(self, sample_ui):
        """Model and temperature from engine init should be used in API call."""
        persona = Persona.from_archetype(get_archetype("jake"))
        engine = CognitiveEngine(
            persona, "Test", "Done", model="claude-haiku-4-5-20251001", temperature=0.3, provider="anthropic"
        )
        with patch.object(
            engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            await engine.plan(sample_ui)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["temperature"] == 0.3

    async def test_system_prompt_stable_across_plans(self, maria_engine, sample_ui):
        """System prompt is built once and reused — byte-stability is what
        makes the prompt-cache breakpoint hit on steps 2+. (Previously it was
        invalidated after every plan to refresh a frustration counter that
        already lives in the user prompt.)"""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            await maria_engine.plan(sample_ui)
            first = maria_engine._system_prompt
            await maria_engine.plan(sample_ui)

        assert first is not None
        assert maria_engine._system_prompt is first


class TestConfidenceBracketValidation:
    async def test_confidence_clamped_when_inconsistent(self, maria_engine, sample_ui):
        """If LLM states NOT_SURE but confidence=0.9, clamp to bracket max (0.54)."""
        response = {
            **SAMPLE_LLM_RESPONSE,
            "recognition": "NOT_SURE",
            "prediction": "NOT_SURE",
            "progress": "THINK_SO",
            "confidence": 0.9,
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(response)
            intent = await maria_engine.plan(sample_ui)

        assert intent.confidence <= 0.54

    async def test_confidence_preserved_when_consistent(self, maria_engine, sample_ui):
        """If confidence is within bracket, it should be preserved as-is."""
        response = {
            **SAMPLE_LLM_RESPONSE,
            "recognition": "THINK_SO",
            "prediction": "THINK_SO",
            "progress": "THINK_SO",
            "confidence": 0.70,
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(response)
            intent = await maria_engine.plan(sample_ui)

        assert intent.confidence == 0.70

    async def test_json_parse_error_returns_wait(self, maria_engine, sample_ui):
        """If LLM returns invalid JSON, default to WAIT action."""
        text_block = MagicMock()
        text_block.text = "I'm not sure what to do here..."
        response = MagicMock()
        response.content = [text_block]

        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            intent = await maria_engine.plan(sample_ui)

        assert intent.action == ActionType.WAIT
        assert intent.confidence == 0.2

    async def test_all_yes_clearly_high_confidence(self, jake_engine, sample_ui):
        """All YES_CLEARLY should allow confidence 0.85-1.0."""
        response = {
            **SAMPLE_LLM_RESPONSE,
            "recognition": "YES_CLEARLY",
            "prediction": "YES_CLEARLY",
            "progress": "YES_CLEARLY",
            "confidence": 0.95,
        }
        with patch.object(
            jake_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(response)
            intent = await jake_engine.plan(sample_ui)

        assert intent.confidence == 0.95

    async def test_any_no_forces_low_confidence(self, maria_engine, sample_ui):
        """Any NO assessment should force confidence to 0.05-0.24 bracket."""
        response = {
            **SAMPLE_LLM_RESPONSE,
            "recognition": "NO",
            "prediction": "THINK_SO",
            "progress": "THINK_SO",
            "confidence": 0.7,
            "action": "wait",
            "target": "",
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(response)
            intent = await maria_engine.plan(sample_ui)

        assert intent.confidence <= 0.24


class TestUserPromptAffectState:
    async def test_affect_state_in_user_prompt(self, maria_engine, sample_ui):
        """Per-step state is the in-character affect line (feeling + time on
        task), not the old numeric frustration counter — and it lives in the
        user prompt so the system prompt stays byte-stable."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_LLM_RESPONSE)
            await maria_engine.plan(sample_ui)

        call_kwargs = mock_create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "How you're feeling:" in user_content
        assert "minute" in user_content


# ============================================================
# Phase 2: reflect() tests
# ============================================================

SAMPLE_REFLECT_RESPONSE = {
    "actual_outcome": "The form submitted and I see a success message.",
    "mismatch": False,
    "reflection": "That worked as I expected. I feel good about this.",
    "emotion": "satisfied",
    "severity": None,
    "category": None,
}

SAMPLE_REFLECT_MISMATCH = {
    "actual_outcome": "Nothing happened when I clicked the button.",
    "mismatch": True,
    "reflection": "I clicked it but nothing changed. That's confusing.",
    "emotion": "confused",
    "severity": "medium",
    "category": "feedback",
}


@pytest.fixture
def sample_intent() -> ActionIntent:
    return ActionIntent(
        action=ActionType.CLICK,
        target="button#submit",
        input_value=None,
        persona_thought="I see a Submit button so I'll try clicking it.",
        expected_outcome="The form submits.",
        confidence=0.45,
    )


@pytest.fixture
def sample_ui_after() -> UIState:
    return UIState(
        url="https://example.com/success",
        page_title="Success",
        dom_summary='div.success "Your account has been created"',
        visible_text="Your account has been created. Welcome!",
    )


class TestReflectResponseParsing:
    async def test_reflect_returns_step_result(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """reflect() should return a fully constructed StepResult."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        assert result.step == 1
        assert result.intent == sample_intent
        assert result.actual_outcome == "The form submitted and I see a success message."
        assert result.mismatch is False
        assert result.emotion == Emotion.SATISFIED
        assert result.severity is None
        assert result.category is None

    async def test_reflect_mismatch_with_severity_and_category(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """When mismatch is true, severity and category should be populated."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_MISMATCH)
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=2,
            )

        assert result.mismatch is True
        assert result.emotion == Emotion.CONFUSED
        assert result.severity == Severity.MEDIUM
        assert result.category == IssueCategory.FEEDBACK

    async def test_reflect_strips_severity_when_no_mismatch(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """severity/category must be None when mismatch is false, even if LLM sets them."""
        bad_response = {
            **SAMPLE_REFLECT_RESPONSE,
            "mismatch": False,
            "severity": "high",
            "category": "navigation",
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(bad_response)
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        assert result.mismatch is False
        assert result.severity is None
        assert result.category is None

    async def test_reflect_give_up_short_circuit(
        self, maria_engine, sample_ui
    ):
        """give_up action should skip the LLM call and return frustrated StepResult."""
        give_up_intent = ActionIntent(
            action=ActionType.GIVE_UP,
            target="",
            input_value=None,
            persona_thought="I can't figure this out.",
            expected_outcome="I stop trying.",
            confidence=0.1,
        )
        # Should NOT call the API
        result = await maria_engine.reflect(
            give_up_intent, sample_ui,
            observation_before=sample_ui,
            observation_after=sample_ui,
            step_number=5,
        )

        assert result.step == 5
        assert result.mismatch is False
        assert result.emotion == Emotion.FRUSTRATED
        assert result.actual_outcome == "Gave up on the task."

    async def test_reflect_json_parse_error_defaults(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """If LLM returns invalid JSON, default to confused mismatch."""
        text_block = MagicMock()
        text_block.text = "I'm not sure what happened..."
        response = MagicMock()
        response.content = [text_block]

        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=3,
            )

        assert result.mismatch is True
        assert result.emotion == Emotion.CONFUSED

    async def test_reflect_markdown_code_fences(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """reflect() should handle markdown code fence wrapping."""
        text_block = MagicMock()
        text_block.text = "```json\n" + json.dumps(SAMPLE_REFLECT_RESPONSE) + "\n```"
        response = MagicMock()
        response.content = [text_block]

        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        assert result.actual_outcome == "The form submitted and I see a success message."

    async def test_reflect_invalid_emotion_defaults_to_uncertain(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """Invalid emotion value should default to uncertain."""
        bad_emotion = {**SAMPLE_REFLECT_RESPONSE, "emotion": "ecstatic"}
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(bad_emotion)
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        assert result.emotion == Emotion.UNCERTAIN

    async def test_reflect_emotion_mismatch_inconsistency_logged(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent, caplog
    ):
        """mismatch=True with confident/satisfied should log warning but not override."""
        inconsistent = {
            **SAMPLE_REFLECT_MISMATCH,
            "emotion": "confident",
        }
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(inconsistent)
            import logging

            with caplog.at_level(logging.WARNING):
                result = await maria_engine.reflect(
                    sample_intent,
                    sample_ui_after,
                    observation_before=sample_ui,
                    observation_after=sample_ui_after,
                    step_number=1,
                )

        # Emotion is preserved (not overridden), but warning is logged
        assert result.emotion == Emotion.CONFIDENT
        assert result.mismatch is True
        assert any("reflect_emotion_mismatch" in r.message for r in caplog.records)

    async def test_reflect_carries_observation_context(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """StepResult must carry both before and after observations."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            result = await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        assert result.observation_before.url == "https://example.com"
        assert result.observation_after.url == "https://example.com/success"

    async def test_reflect_model_and_temperature_passed(
        self, sample_ui, sample_ui_after, sample_intent
    ):
        """Model and temperature from engine init should be used in reflect API call."""
        persona = Persona.from_archetype(get_archetype("jake"))
        engine = CognitiveEngine(
            persona, "Test", "Done", model="claude-haiku-4-5-20251001", temperature=0.3, provider="anthropic"
        )
        with patch.object(
            engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            await engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["temperature"] == 0.3


class TestReflectPromptContent:
    async def test_reflect_prompt_contains_intent_info(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """Reflect user prompt should include the action, target, and expected outcome."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        call_kwargs = mock_create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "click" in user_content
        assert "button#submit" in user_content
        assert "The form submits." in user_content

    async def test_reflect_prompt_contains_before_after_state(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """Reflect user prompt should include both before and after page state."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        call_kwargs = mock_create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "https://example.com" in user_content
        assert "https://example.com/success" in user_content
        assert "Welcome to Example." in user_content
        assert "Your account has been created" in user_content

    async def test_reflect_system_prompt_contains_persona(
        self, maria_engine, sample_ui, sample_ui_after, sample_intent
    ):
        """Reflect system prompt should reference the persona's traits."""
        with patch.object(
            maria_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_llm_response(SAMPLE_REFLECT_RESPONSE)
            await maria_engine.reflect(
                sample_intent, sample_ui_after,
                observation_before=sample_ui,
                observation_after=sample_ui_after,
                step_number=1,
            )

        call_kwargs = mock_create.call_args.kwargs
        system_raw = call_kwargs["system"]
        # system may be a string or a list of content blocks (prompt caching)
        if isinstance(system_raw, list):
            system_content = " ".join(block["text"] for block in system_raw)
        else:
            system_content = system_raw
        assert "Maria" in system_content or "maria" in system_content
        assert "low" in system_content  # tech_literacy


class TestProblemSolvingConstraints:
    """Verify problem-solving constraints appear in system prompts per tech literacy."""

    def test_low_tech_gets_no_workaround_constraint(self, maria_engine):
        prompt = maria_engine._get_system_prompt()
        assert "do NOT look for creative workarounds" in prompt
        assert "retry the SAME thing" in prompt

    def test_high_tech_gets_resourceful_constraint(self, jake_engine):
        prompt = jake_engine._get_system_prompt()
        assert "actively problem-solve" in prompt
        assert "resourceful" in prompt

    def test_low_tech_prohibits_scanning_for_alternatives(self, maria_engine):
        prompt = maria_engine._get_system_prompt()
        assert "do not scan the page for alternative approaches" in prompt

    def test_high_tech_allows_scanning(self, jake_engine):
        prompt = jake_engine._get_system_prompt()
        assert "scan the page for alternative paths" in prompt
