"""Tests for ComputerUseEngine and discriminated target types."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autouser.cognitive.computer_use import ComputerUseEngine, _StepMemory
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CoordinateTarget,
    Emotion,
    SelectorTarget,
    StepResult,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def maria_cu_engine() -> ComputerUseEngine:
    persona = Persona.from_archetype(get_archetype("maria"))
    return ComputerUseEngine(persona, "Sign up for an account", "See the dashboard")


@pytest.fixture
def sample_ui(tmp_path: Path) -> UIState:
    # Create a minimal PNG screenshot (1x1 pixel)
    screenshot = tmp_path / "step_001.png"
    # Minimal valid PNG
    import struct
    import zlib

    def _minimal_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"

        def chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        raw = b"\x00\x00\x00\x00"  # filter byte + 1 RGB pixel
        idat = zlib.compress(raw)
        return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

    screenshot.write_bytes(_minimal_png())

    return UIState(
        url="https://example.com/signup",
        page_title="Sign Up",
        dom_summary='button#submit "Submit"\ninput#email type=email',
        visible_text="Create your account",
        screenshot_path=str(screenshot),
    )


def _mock_anthropic_response(payload: dict) -> MagicMock:
    text_block = MagicMock()
    text_block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.cache_creation_input_tokens = 50
    response.usage.cache_read_input_tokens = 0
    return response


# ---------------------------------------------------------------------------
# Target type tests
# ---------------------------------------------------------------------------


class TestTargetTypes:
    def test_selector_target_str(self):
        t = SelectorTarget(selector="button#submit")
        assert str(t) == "button#submit"
        assert t.kind == "selector"

    def test_coordinate_target_str(self):
        t = CoordinateTarget(x=640, y=360, screenshot_id="step1_abc12345")
        assert "(640, 360)@step1_abc12345" in str(t)
        assert t.kind == "coordinate"

    def test_selector_target_rejects_empty(self):
        with pytest.raises(Exception):
            SelectorTarget(selector="")

    def test_coordinate_target_rejects_negative(self):
        with pytest.raises(Exception):
            CoordinateTarget(x=-1, y=360, screenshot_id="test")

    def test_coordinate_target_requires_screenshot_id(self):
        with pytest.raises(Exception):
            CoordinateTarget(x=100, y=200, screenshot_id="")

    def test_action_intent_with_typed_target(self):
        coord = CoordinateTarget(x=640, y=360, screenshot_id="step1_abc")
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="",
            typed_target=coord,
            persona_thought="I see a button.",
            expected_outcome="It gets clicked.",
            confidence=0.7,
        )
        assert intent.typed_target == coord
        assert intent.resolve_target() == str(coord)

    def test_action_intent_resolve_falls_back_to_str(self):
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="button#submit",
            persona_thought="Clicking submit.",
            expected_outcome="Form submits.",
            confidence=0.8,
        )
        assert intent.resolve_target() == "button#submit"

    def test_action_intent_click_requires_some_target(self):
        with pytest.raises(ValueError, match="requires a non-empty target"):
            ActionIntent(
                action=ActionType.CLICK,
                target="",
                typed_target=None,
                persona_thought="Trying to click.",
                expected_outcome="Something.",
                confidence=0.5,
            )


# ---------------------------------------------------------------------------
# StepMemory tests
# ---------------------------------------------------------------------------


class TestStepMemory:
    def test_empty_history(self):
        mem = _StepMemory("Sign up", "See dashboard")
        assert "No prior actions" in mem.format_history()

    def test_records_and_formats(self):
        mem = _StepMemory("Sign up", "See dashboard")
        mem.record(1, "click", "(640,360)", "Form submits", "Form submitted", 0.8, "confident")
        mem.record(2, "type", "(200,100)", "Email entered", "Email entered", 0.9, "satisfied")
        text = mem.format_history()
        assert "Step 1" in text
        assert "Step 2" in text
        assert "click" in text

    def test_max_steps_truncation(self):
        mem = _StepMemory("Sign up", "See dashboard")
        for i in range(10):
            mem.record(i + 1, "click", f"({i},{i})", "x", "y", 0.5, "uncertain")
        text = mem.format_history(max_steps=3)
        assert "Step 8" in text
        assert "Step 10" in text
        assert "Step 1:" not in text


# ---------------------------------------------------------------------------
# ComputerUseEngine plan() tests
# ---------------------------------------------------------------------------


class TestComputerUsePlan:
    async def test_plan_returns_coordinate_target(self, maria_cu_engine, sample_ui):
        llm_response = {
            "action": "click",
            "x": 640,
            "y": 360,
            "input_value": None,
            "persona_thought": "I see a big Submit button in the middle.",
            "expected_outcome": "The form will submit.",
            "confidence": 0.45,
            "recognition": "THINK_SO",
            "prediction": "NOT_SURE",
            "progress": "THINK_SO",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(llm_response)
            intent = await maria_cu_engine.plan(sample_ui)

        assert intent.action == ActionType.CLICK
        assert intent.typed_target is not None
        assert isinstance(intent.typed_target, CoordinateTarget)
        assert intent.typed_target.x == 640
        assert intent.typed_target.y == 360
        assert intent.typed_target.screenshot_id is not None

    async def test_plan_scaffold_stripped(self, maria_cu_engine, sample_ui):
        llm_response = {
            "action": "scroll",
            "x": 0,
            "y": 0,
            "persona_thought": "I need to scroll to see more.",
            "expected_outcome": "I see more content.",
            "confidence": 0.3,
            "recognition": "NOT_SURE",
            "prediction": "NOT_SURE",
            "progress": "NOT_SURE",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(llm_response)
            intent = await maria_cu_engine.plan(sample_ui)

        assert intent.action == ActionType.SCROLL
        intent_dict = intent.model_dump()
        for field in ("recognition", "prediction", "progress"):
            assert field not in intent_dict

    async def test_plan_zero_coords_forces_wait(self, maria_cu_engine, sample_ui):
        """If the model returns 0,0 for a click, treat as confusion."""
        llm_response = {
            "action": "click",
            "x": 0,
            "y": 0,
            "persona_thought": "Not sure.",
            "expected_outcome": "Something.",
            "confidence": 0.1,
            "recognition": "NO",
            "prediction": "NO",
            "progress": "NO",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(llm_response)
            intent = await maria_cu_engine.plan(sample_ui)

        assert intent.action == ActionType.WAIT

    async def test_plan_no_screenshot_returns_wait(self, maria_cu_engine):
        ui = UIState(
            url="https://example.com",
            page_title="Test",
            dom_summary="test",
            visible_text="test",
            screenshot_path=None,
        )
        intent = await maria_cu_engine.plan(ui)
        assert intent.action == ActionType.WAIT

    async def test_plan_json_parse_failure_returns_wait(self, maria_cu_engine, sample_ui):
        text_block = MagicMock()
        text_block.text = "I don't know what to do"
        response = MagicMock()
        response.content = [text_block]
        response.usage = MagicMock()
        response.usage.input_tokens = 50

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            intent = await maria_cu_engine.plan(sample_ui)

        assert intent.action == ActionType.WAIT

    async def test_plan_confidence_bracket_clamped(self, maria_cu_engine, sample_ui):
        """If stated confidence contradicts R/P/G, it gets clamped."""
        llm_response = {
            "action": "click",
            "x": 400,
            "y": 300,
            "persona_thought": "Clicking something.",
            "expected_outcome": "Something happens.",
            "confidence": 0.95,  # Way too high for NOT_SURE assessments
            "recognition": "NOT_SURE",
            "prediction": "NOT_SURE",
            "progress": "NOT_SURE",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(llm_response)
            intent = await maria_cu_engine.plan(sample_ui)

        assert intent.confidence <= 0.54  # Upper bound for NOT_SURE bracket


# ---------------------------------------------------------------------------
# ComputerUseEngine reflect() tests
# ---------------------------------------------------------------------------


class TestComputerUseReflect:
    async def test_reflect_give_up_short_circuits(self, maria_cu_engine, sample_ui):
        intent = ActionIntent(
            action=ActionType.GIVE_UP,
            target="",
            persona_thought="I can't do this.",
            expected_outcome="Nothing.",
            confidence=0.1,
        )
        result = await maria_cu_engine.reflect(
            intent, sample_ui,
            observation_before=sample_ui,
            observation_after=sample_ui,
            step_number=1,
        )
        assert result.emotion == Emotion.FRUSTRATED
        assert not result.mismatch

    async def test_reflect_records_memory(self, maria_cu_engine, sample_ui):
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="button#x",
            persona_thought="Clicking.",
            expected_outcome="Form submits.",
            confidence=0.7,
        )

        reflect_response = {
            "actual_outcome": "Form submitted successfully.",
            "mismatch": False,
            "reflection": "That worked as expected.",
            "emotion": "satisfied",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(reflect_response)
            result = await maria_cu_engine.reflect(
                intent, sample_ui,
                observation_before=sample_ui,
                observation_after=sample_ui,
                step_number=1,
            )

        assert result.emotion == Emotion.SATISFIED
        assert len(maria_cu_engine._memory.entries) == 1
        assert maria_cu_engine._memory.entries[0]["emotion"] == "satisfied"

    async def test_reflect_preserves_completion_and_memory_fields(
        self, maria_cu_engine, sample_ui
    ):
        """PR#45 finding P1: screenshot-mode reflect must carry
        persona_believes_complete, observed_success_signal, and mental_note —
        the runner's soft-success gate and exploration exit depend on them, and
        without a predicate they are the only way a computer-use run can stop."""
        intent = ActionIntent(
            action=ActionType.CLICK, target="button#x",
            persona_thought="Clicking.", expected_outcome="It completes.",
            confidence=0.8,
        )
        reflect_response = {
            "actual_outcome": "The dashboard appeared.",
            "mismatch": False,
            "reflection": "Looks done.",
            "emotion": "satisfied",
            "persona_believes_complete": True,
            "observed_success_signal": True,
            "mental_note": "Submitting jumps straight to the dashboard.",
        }
        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(reflect_response)
            result = await maria_cu_engine.reflect(
                intent, sample_ui,
                observation_before=sample_ui, observation_after=sample_ui,
                step_number=1,
            )

        assert result.persona_believes_complete is True
        assert result.observed_success_signal is True
        assert result.mental_note == "Submitting jumps straight to the dashboard."

    async def test_reflect_mismatch_with_severity(self, maria_cu_engine, sample_ui):
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="button#x",
            persona_thought="Clicking submit.",
            expected_outcome="Form submits.",
            confidence=0.6,
        )

        reflect_response = {
            "actual_outcome": "Got an error message.",
            "mismatch": True,
            "reflection": "That didn't work. I'm confused.",
            "emotion": "confused",
            "severity": "medium",
            "category": "error_recovery",
        }

        with patch.object(
            maria_cu_engine._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _mock_anthropic_response(reflect_response)
            result = await maria_cu_engine.reflect(
                intent, sample_ui,
                observation_before=sample_ui,
                observation_after=sample_ui,
                step_number=1,
            )

        assert result.mismatch is True
        assert result.severity.value == "medium"
        assert result.category.value == "error_recovery"


# ---------------------------------------------------------------------------
# should_give_up tests
# ---------------------------------------------------------------------------


def _cu_step(n: int, *, action_error: str | None = None, emotion=Emotion.CONFIDENT) -> StepResult:
    before = UIState(url="https://x.test/", page_title="P", dom_summary="d", visible_text="t")
    # After differs from before by default → an ordinary effective step (no
    # "no_effect" frustration); action_error, when set, is the objective signal.
    after = UIState(
        url="https://x.test/", page_title="P", dom_summary="d",
        visible_text=f"changed {n}", action_error=action_error,
    )
    return StepResult(
        step=n,
        intent=ActionIntent(
            action=ActionType.CLICK, target=f"#t{n}",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        ),
        observation_before=before,
        observation_after=after,
        actual_outcome="o", mismatch=False, reflection="r", emotion=emotion,
    )


class TestComputerUseShouldGiveUp:
    def test_not_exhausted_initially(self, maria_cu_engine):
        assert not maria_cu_engine.should_give_up()

    def test_exhausted_on_objective_frustration(self, maria_cu_engine):
        # Maria has patience=6; six failed actions = 6.0 objective frustration.
        # affect.py works in screenshot mode because the raw observations still
        # carry dom_summary/visible_text/focus.
        for i in range(6):
            maria_cu_engine.history.append(_cu_step(i + 1, action_error="click failed"))
        assert maria_cu_engine.should_give_up()

    def test_emotion_labels_alone_do_not_trigger_give_up(self, maria_cu_engine):
        # The circularity fix now applies to screenshot mode too: self-reported
        # frustration with no objective evidence is not termination fuel.
        for i in range(8):
            maria_cu_engine.history.append(_cu_step(i + 1, emotion=Emotion.FRUSTRATED))
        assert not maria_cu_engine.should_give_up()
