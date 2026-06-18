"""Journey renderer: the experience trajectory must be fully recorded and readable."""

from __future__ import annotations

import pytest

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    IssueCategory,
    Severity,
    StepResult,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.report.journey_renderer import render_journey
from autouser.report.models import FrictionLog, TaskOutcome


@pytest.fixture
def maria() -> Persona:
    return Persona.from_archetype(get_archetype("maria"))


def _state(title="Home", url="https://x.test/", **kw) -> UIState:
    return UIState(url=url, page_title=title, dom_summary="els", visible_text="txt", **kw)


def _step(n, *, action=ActionType.CLICK, target="#go", note=None, mismatch=False,
          emotion=Emotion.CONFIDENT, after=None, action_error=None) -> StepResult:
    obs_after = after or _state(action_error=action_error)
    return StepResult(
        step=n,
        intent=ActionIntent(
            action=action, target=target, input_value="v" if action == ActionType.TYPE else None,
            persona_thought=f"thought {n}", expected_outcome=f"expect {n}", confidence=0.5,
        ),
        observation_before=_state(),
        observation_after=obs_after,
        actual_outcome=f"outcome {n}",
        mismatch=mismatch,
        severity=Severity.MEDIUM if mismatch else None,
        category=IssueCategory.FEEDBACK if mismatch else None,
        reflection=f"reflect {n}",
        emotion=emotion,
        mental_note=note,
    )


def _log(steps) -> FrictionLog:
    return FrictionLog(
        task="Do the thing.",
        success_criteria="Thing done.",
        persona_name="maria",
        persona_summary="maria summary",
        outcome=TaskOutcome.COMPLETED,
        total_steps=len(steps),
        steps=steps,
        issues=[],
        overall_impressions="It mostly made sense.",
    )


class TestJourneyRenderer:
    def test_full_trajectory_recorded(self, maria):
        steps = [
            _step(1, emotion=Emotion.UNCERTAIN),
            _step(2, mismatch=True, emotion=Emotion.CONFUSED,
                  note="The button only works after naming."),
        ]
        out = render_journey(_log(steps), maria)
        # Every cognitive trace element must appear: thought, expectation,
        # outcome, reflection, emotion, note.
        assert "thought 1" in out and "thought 2" in out
        assert "expect 2" in out
        assert "outcome 2" in out
        assert "reflect 2" in out
        assert "confused" in out
        assert "Note to self: The button only works after naming." in out

    def test_emotional_arc_in_order(self, maria):
        steps = [
            _step(1, emotion=Emotion.CONFIDENT),
            _step(2, emotion=Emotion.CONFUSED),
            _step(3, emotion=Emotion.SATISFIED),
        ]
        out = render_journey(_log(steps), maria)
        arc = out.split("Emotional arc:")[1].splitlines()[0]
        assert arc.index("confident") < arc.index("confused") < arc.index("satisfied")

    def test_mental_model_section_collects_notes_and_places(self, maria):
        steps = [
            _step(1, after=_state(title="Form", url="https://x.test/form"),
                  note="Forms live behind the Start button."),
        ]
        out = render_journey(_log(steps), maria)
        assert "What maria now believes about this site" in out
        assert "Form — https://x.test/form" in out
        assert "1. (step 1) Forms live behind the Start button." in out
        assert "It mostly made sense." in out

    def test_no_notes_renders_honest_absence(self, maria):
        out = render_journey(_log([_step(1)]), maria)
        assert "left without a working model" in out

    def test_failed_action_and_perception_markers(self, maria):
        steps = [
            _step(1, action_error="click failed: timeout"),
            _step(2, after=_state(perceived_via="screen_reader")),
        ]
        out = render_journey(_log(steps), maria)
        assert "did not take effect" in out
        assert "perceived via screen reader" in out

    def test_mismatch_carries_severity_and_category(self, maria):
        out = render_journey(_log([_step(1, mismatch=True, emotion=Emotion.FRUSTRATED)]), maria)
        assert "mismatch (medium/feedback)" in out
