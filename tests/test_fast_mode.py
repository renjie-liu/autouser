"""Dual-process fast/slow reflect (cognitive v2, slice 6).

Cruising steps skip the LLM reflect (System 1); surprise, uncertainty, or a
plausible completion escalates to the full LLM reflect (System 2). These tests
pin the gate and the deterministic fast result, and confirm fast mode is OFF
by default (no fidelity change unless asked).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from autouser.cognitive.engine import CognitiveEngine
from autouser.cognitive.models import ActionIntent, ActionType, Emotion, UIState
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


@pytest.fixture
def engine() -> CognitiveEngine:
    persona = Persona.from_archetype(get_archetype("maria"))
    return CognitiveEngine(
        persona, "Add an item to the cart",
        "You reach the checkout page showing your order total",
        provider="anthropic", fast_mode=True,
    )


def _state(url="https://shop.test/", text="Shop home", focused=None, action_error=None) -> UIState:
    return UIState(
        url=url, page_title="Shop", dom_summary="elements",
        visible_text=text, focused_element=focused, action_error=action_error,
    )


def _intent(confidence=0.8, action=ActionType.CLICK, target="#add") -> ActionIntent:
    return ActionIntent(
        action=action, target=target, input_value=None,
        persona_thought="add it", expected_outcome="item added", confidence=confidence,
    )


def _seed_seen(engine: CognitiveEngine, state: UIState) -> None:
    """Put a prior step on this page in history so it is no longer a first
    encounter (the cruise path is only reachable on a familiar page)."""
    from autouser.cognitive.models import StepResult

    engine.history.append(StepResult(
        step=len(engine.history) + 1, intent=_intent(),
        observation_before=state, observation_after=state,
        actual_outcome="o", mismatch=False, reflection="r", emotion=Emotion.CONFIDENT,
    ))


class TestDeliberationGate:
    def test_cruising_step_does_not_deliberate(self, engine):
        before = _state(text="Shop home")
        _seed_seen(engine, before)  # familiar page — past the first-read
        after = _state(url="https://shop.test/cart", text="Item added to cart")
        assert engine._should_deliberate(_intent(), before, after) is False

    def test_action_error_forces_deliberate(self, engine):
        before = _state()
        after = _state(action_error="click failed: timeout")
        assert engine._should_deliberate(_intent(), before, after) is True

    def test_no_effect_forces_deliberate(self, engine):
        same = _state(text="nothing changed")
        assert engine._should_deliberate(_intent(), same, same) is True

    def test_low_confidence_forces_deliberate(self, engine):
        before = _state(text="a")
        after = _state(text="b")
        assert engine._should_deliberate(_intent(confidence=0.4), before, after) is True

    def test_plausible_completion_forces_deliberate(self, engine):
        before = _state(text="Shop home")
        # The post-action page carries the success-criteria words → must reflect
        # to detect success even though nothing looks surprising.
        after = _state(url="https://shop.test/checkout",
                       text="Checkout — review your order total before paying")
        assert engine._should_deliberate(_intent(), before, after) is True

    def test_first_encounter_of_a_page_deliberates(self, engine):
        # No history → this page is being seen for the first time → read it.
        before = _state(url="https://shop.test/new", text="a")
        after = _state(url="https://shop.test/new", text="b")
        assert engine._should_deliberate(_intent(), before, after) is True

    def test_revisited_page_can_cruise(self, engine):
        from autouser.cognitive.models import StepResult
        seen = _state(url="https://shop.test/", text="home")
        engine.history.append(StepResult(
            step=1, intent=_intent(),
            observation_before=seen, observation_after=seen,
            actual_outcome="o", mismatch=False, reflection="r", emotion=Emotion.CONFIDENT,
        ))
        before = _state(url="https://shop.test/", text="home, now familiar")
        after = _state(url="https://shop.test/", text="changed but same page")
        # Same (url, title) as a prior step → not a first encounter → may cruise.
        assert engine._should_deliberate(_intent(), before, after) is False

    def test_exploration_always_deliberates(self):
        persona = Persona.from_archetype(get_archetype("maria"))
        eng = CognitiveEngine(
            persona, "explore", "seen enough",
            provider="anthropic", fast_mode=True, exploration=True,
        )
        before = _state(text="a")
        after = _state(text="b")
        assert eng._should_deliberate(_intent(), before, after) is True


class TestReflectShortCircuit:
    async def test_fast_path_makes_no_llm_call(self, engine):
        before = _state(text="Shop home")
        _seed_seen(engine, before)  # familiar page
        after = _state(url="https://shop.test/cart", text="Item added to cart")
        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            result = await engine.reflect(
                _intent(), after,
                observation_before=before, observation_after=after, step_number=1,
            )
        mock_llm.assert_not_called()
        assert result.mismatch is False
        assert result.mental_note is None
        assert result.observed_success_signal is False
        assert result.persona_believes_complete is False
        assert result.emotion == Emotion.CONFIDENT

    async def test_surprise_step_takes_llm_path(self, engine):
        before = _state(text="Shop home")
        after = _state(action_error="click failed")
        sample = '{"actual_outcome":"nothing","mismatch":true,"reflection":"huh","emotion":"confused","persona_believes_complete":false,"observed_success_signal":false,"mental_note":null,"severity":"medium","category":"feedback"}'
        with patch.object(engine, "_call_llm", new_callable=AsyncMock, return_value=sample) as mock_llm:
            result = await engine.reflect(
                _intent(), after,
                observation_before=before, observation_after=after, step_number=1,
            )
        mock_llm.assert_called_once()
        assert result.mismatch is True

    async def test_fast_mode_off_always_calls_llm(self):
        persona = Persona.from_archetype(get_archetype("maria"))
        eng = CognitiveEngine(
            persona, "t", "c", provider="anthropic", fast_mode=False,
        )
        before = _state(text="a")
        after = _state(text="b")
        sample = '{"actual_outcome":"ok","mismatch":false,"reflection":"fine","emotion":"confident","persona_believes_complete":false,"observed_success_signal":false,"mental_note":null}'
        with patch.object(eng, "_call_llm", new_callable=AsyncMock, return_value=sample) as mock_llm:
            await eng.reflect(
                _intent(), after,
                observation_before=before, observation_after=after, step_number=1,
            )
        mock_llm.assert_called_once()

    async def test_give_up_still_short_circuits_in_fast_mode(self, engine):
        before = _state()
        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            result = await engine.reflect(
                _intent(action=ActionType.GIVE_UP, target=""), before,
                observation_before=before, observation_after=before, step_number=1,
            )
        mock_llm.assert_not_called()
        assert result.emotion == Emotion.FRUSTRATED


class TestSuccessKeywords:
    def test_stopwords_excluded(self, engine):
        # "You reach the checkout page showing your order total"
        kw = engine._success_keywords
        assert "checkout" in kw
        assert "page" not in kw  # stopword
        assert "your" not in kw  # stopword

    def test_empty_criteria_never_plausible(self):
        eng = CognitiveEngine(Persona(), "t", "", provider="anthropic", fast_mode=True)
        assert eng._success_plausible(_state(text="anything at all")) is False
