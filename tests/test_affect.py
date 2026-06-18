"""Affect + simulated time (cognitive v2, slice 3).

The give-up mechanism must run on objective trace evidence, never on
emotion labels — the old mechanism counted self-reported confusion toward
termination, making the measurement its own fuel. These tests pin the
mechanism (which events count, what scales what), not the calibration
constants.
"""

from __future__ import annotations

import pytest

from autouser.affect import (
    READ_FRACTION,
    REVISIT_READ_FRACTION,
    TIME_BUDGET_PER_PATIENCE_SECONDS,
    W_ACTION_FAILED,
    W_BACKTRACK,
    W_EXPECTATION,
    W_REPEATED,
    affect_line,
    extract_events,
    frustration_ratio,
    frustration_score,
    simulated_seconds,
    summarize,
)
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    StepResult,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


@pytest.fixture
def carlos() -> Persona:  # patience 2, reading LOW, motor LOW
    return Persona.from_archetype(get_archetype("carlos"))


@pytest.fixture
def maria() -> Persona:  # patience 6, reading HIGH
    return Persona.from_archetype(get_archetype("maria"))


def _state(url="https://x.test/", text="Welcome", focused=None) -> UIState:
    return UIState(
        url=url, page_title="T", dom_summary="elements",
        visible_text=text, focused_element=focused,
    )


def _step(
    n: int,
    *,
    action: ActionType = ActionType.CLICK,
    target: str = "#a",
    value: str | None = None,
    before: UIState | None = None,
    after: UIState | None = None,
    mismatch: bool = False,
    emotion: Emotion = Emotion.CONFIDENT,
    action_error: str | None = None,
) -> StepResult:
    before = before or _state()
    if after is None:
        # Default: the action visibly changed the page (no no_effect event).
        after = _state(text=f"changed {n}")
    if action_error:
        after = after.model_copy(update={"action_error": action_error})
    return StepResult(
        step=n,
        intent=ActionIntent(
            action=action, target=target, input_value=value,
            persona_thought="t", expected_outcome="e", confidence=0.5,
        ),
        observation_before=before,
        observation_after=after,
        actual_outcome="o",
        mismatch=mismatch,
        reflection="r",
        emotion=emotion,
    )


class TestFrustrationEvents:
    def test_emotion_labels_alone_contribute_nothing(self):
        """THE circularity fix: a step whose only signal is self-reported
        frustration is not termination fuel."""
        history = [
            _step(1, emotion=Emotion.FRUSTRATED),
            _step(2, target="#b", emotion=Emotion.CONFUSED),
        ]
        assert extract_events(history) == []
        assert frustration_score(history) == 0.0

    def test_failed_action_counts(self):
        history = [_step(1, action_error="click failed")]
        events = extract_events(history)
        assert [e.kind for e in events] == ["action_failed"]
        assert frustration_score(history) == W_ACTION_FAILED

    def test_no_effect_click_counts(self):
        same = _state()
        history = [_step(1, before=same, after=same)]
        assert [e.kind for e in extract_events(history)] == ["no_effect"]

    def test_effective_click_is_free(self):
        history = [_step(1)]  # default after differs from before
        assert extract_events(history) == []

    def test_type_never_counts_as_no_effect(self):
        # Input values don't show in innerText — an effective TYPE looks
        # unchanged and must not be charged.
        same = _state()
        history = [_step(1, action=ActionType.TYPE, value="x", before=same, after=same)]
        assert extract_events(history) == []

    def test_press_that_moves_focus_is_free(self):
        before = _state(focused=None)
        after = _state(focused="input#a")
        history = [_step(1, action=ActionType.PRESS, target="", value="Tab",
                         before=before, after=after)]
        assert extract_events(history) == []

    def test_repeated_action_within_lookback(self):
        history = [
            _step(1, target="#retry"),
            _step(2, target="#retry"),
        ]
        kinds = [e.kind for e in extract_events(history)]
        assert kinds == ["repeated_action"]
        assert frustration_score(history) == W_REPEATED

    def test_repeat_outside_lookback_is_free(self):
        history = [
            _step(1, target="#x"),
            _step(2, target="#b"),
            _step(3, target="#c"),
            _step(4, target="#d"),
            _step(5, target="#x"),  # 4 steps later — forgotten
        ]
        assert extract_events(history) == []

    def test_backtrack_counts(self):
        history = [_step(1, action=ActionType.BACK, target="")]
        assert frustration_score(history) == W_BACKTRACK

    def test_mismatch_counts_at_reduced_weight(self):
        history = [_step(1, mismatch=True)]
        assert frustration_score(history) == W_EXPECTATION

    def test_failed_retry_compounds(self):
        history = [
            _step(1, target="#retry", action_error="failed"),
            _step(2, target="#retry", action_error="failed"),
        ]
        # Second step: failure + repeat — more frustrating than either alone.
        assert frustration_score(history) == pytest.approx(
            2 * W_ACTION_FAILED + W_REPEATED
        )


class TestAbandonment:
    def test_low_patience_abandons_on_two_failures(self, carlos):
        history = [
            _step(1, action_error="failed"),
            _step(2, target="#b", action_error="failed"),
        ]
        summary = summarize(history, carlos)
        assert summary.should_abandon
        assert summary.abandon_reason == "frustration"

    def test_high_patience_persists(self, maria):
        history = [
            _step(1, action_error="failed"),
            _step(2, target="#b", action_error="failed"),
        ]
        assert not summarize(history, maria).should_abandon

    def test_time_budget_abandonment(self, carlos):
        # One enormous page read repeatedly: carlos (patience 2) has a
        # 2 × TIME_BUDGET_PER_PATIENCE_SECONDS budget.
        long_page = _state(text="x" * 4000)
        history = [
            _step(n, target=f"#unique{n}", before=long_page,
                  after=_state(url=f"https://x.test/{n}", text="x" * 4000))
            for n in range(1, 30)
        ]
        summary = summarize(history, carlos)
        assert summary.elapsed_seconds >= 2 * TIME_BUDGET_PER_PATIENCE_SECONDS
        assert summary.should_abandon
        assert summary.abandon_reason == "time_budget"

    def test_empty_history_is_calm(self, maria):
        summary = summarize([], maria)
        assert not summary.should_abandon
        assert summary.frustration == 0.0
        assert "calm" in summary.feeling


class TestSimulatedTime:
    def test_thorough_reader_pays_more_than_skimmer(self, maria, carlos):
        page = _state(text="x" * 2000)
        history = [_step(1, before=page)]
        assert simulated_seconds(history, maria) > simulated_seconds(history, carlos)
        # The gap is the READ_FRACTION ratio on the reading component.
        assert READ_FRACTION[maria.reading_comprehension] == 1.0
        assert READ_FRACTION[carlos.reading_comprehension] == 0.2

    def test_revisit_costs_less_than_first_visit(self, maria):
        page = _state(text="x" * 2000)
        first = simulated_seconds([_step(1, before=page)], maria)
        both = simulated_seconds(
            [_step(1, before=page), _step(2, target="#b", before=page)], maria
        )
        second = both - first
        assert second < first * (REVISIT_READ_FRACTION + 0.2)

    def test_new_page_charges_full_read(self, maria):
        page_a = _state(url="https://x.test/a", text="x" * 2000)
        page_b = _state(url="https://x.test/b", text="x" * 2000)
        two_pages = simulated_seconds(
            [_step(1, before=page_a), _step(2, target="#b", before=page_b)], maria
        )
        same_page = simulated_seconds(
            [_step(1, before=page_a), _step(2, target="#b", before=page_a)], maria
        )
        assert two_pages > same_page

    def test_typing_costs_per_character(self, maria):
        short = [_step(1, action=ActionType.TYPE, target="#f", value="ab")]
        long = [_step(1, action=ActionType.TYPE, target="#f", value="ab" * 30)]
        assert simulated_seconds(long, maria) > simulated_seconds(short, maria)


class TestAffectLine:
    def test_line_carries_feeling_and_minutes(self, maria):
        line = affect_line(summarize([_step(1, mismatch=True)], maria))
        assert "How you're feeling:" in line
        assert "minute" in line

    def test_ratio_drives_phrase(self, carlos):
        calm = summarize([], carlos)
        assert "calm" in calm.feeling
        fed_up = summarize(
            [_step(1, action_error="f"), _step(2, target="#b", mismatch=True)], carlos
        )
        assert fed_up.ratio >= 0.75
        assert "quit" in fed_up.feeling or "patience" in fed_up.feeling

    def test_ratio_clamped(self, carlos):
        history = [_step(n, target=f"#t{n}", action_error="f") for n in range(1, 9)]
        assert frustration_ratio(history, carlos) == 1.0
