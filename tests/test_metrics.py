"""Tests for M3 measurement harness."""

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    StepResult,
    UIState,
)
from autouser.metrics import extract_metrics, compare_metrics
from autouser.persona import get_archetype
from autouser.persona.models import Persona
from autouser.session import SessionLog, TaskSpec


def _ui(url: str = "https://example.com") -> UIState:
    return UIState(url=url, page_title="Test", dom_summary="<div>test</div>")


def _intent(
    action: ActionType = ActionType.CLICK,
    target: str = "#btn",
    confidence: float = 0.8,
    input_value: str | None = None,
) -> ActionIntent:
    kw: dict = dict(
        action=action,
        target=target,
        persona_thought="thinking",
        expected_outcome="something happens",
        confidence=confidence,
    )
    if input_value is not None:
        kw["input_value"] = input_value
    if action == ActionType.TYPE and "input_value" not in kw:
        kw["input_value"] = "text"
    if action == ActionType.NAVIGATE and "input_value" not in kw:
        kw["input_value"] = "https://example.com"
    return ActionIntent(**kw)


def _step(
    step: int = 1,
    action: ActionType = ActionType.CLICK,
    target: str = "#btn",
    confidence: float = 0.8,
    url_before: str = "https://example.com",
    url_after: str = "https://example.com/next",
) -> StepResult:
    return StepResult(
        step=step,
        intent=_intent(action=action, target=target, confidence=confidence),
        observation_before=_ui(url_before),
        observation_after=_ui(url_after),
        actual_outcome="done",
        mismatch=False,
        reflection="ok",
        emotion=Emotion.CONFIDENT,
    )


def _session(
    persona_name: str = "maria",
    steps: list[StepResult] | None = None,
    terminal_reason: str | None = None,
) -> SessionLog:
    persona = Persona.from_archetype(get_archetype(persona_name))
    spec = TaskSpec(
        task="Test task",
        success_criteria="Done",
        start_url="https://example.com",
    )
    log = SessionLog(task_spec=spec, persona_snapshot=persona)
    for s in steps or []:
        log.append_step(s)
    if terminal_reason:
        log.finalize(terminal_reason)
    return log


class TestEmptySession:
    def test_zero_steps(self):
        mv = extract_metrics(_session(steps=[], terminal_reason="max_steps"))
        assert mv.persona_name == "maria"
        assert mv.confidence.values == ()
        assert mv.confidence.mean == 0.0
        assert mv.unique_urls == 0
        assert mv.non_advancing_rate == 0.0
        assert mv.steps_before_abandonment is None
        assert mv.abandoned is False


class TestConfidence:
    def test_values_extracted(self):
        steps = [
            _step(step=1, confidence=0.3),
            _step(step=2, confidence=0.7),
            _step(step=3, confidence=0.5),
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.confidence.values == (0.3, 0.7, 0.5)
        assert abs(mv.confidence.mean - 0.5) < 0.01
        assert mv.confidence.min == 0.3
        assert mv.confidence.max == 0.7

    def test_single_step_stdev_zero(self):
        mv = extract_metrics(
            _session(steps=[_step(confidence=0.6)], terminal_reason="max_steps")
        )
        assert mv.confidence.stdev == 0.0


class TestExplorationBreadth:
    def test_unique_urls(self):
        steps = [
            _step(step=1, url_before="https://a.com", url_after="https://b.com"),
            _step(step=2, url_before="https://b.com", url_after="https://c.com"),
            _step(step=3, url_before="https://c.com", url_after="https://a.com"),
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.unique_urls == 3

    def test_unique_targets(self):
        steps = [
            _step(step=1, target="#a"),
            _step(step=2, target="#b"),
            _step(step=3, target="#a"),  # duplicate
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.unique_targets == 2


class TestNonAdvancing:
    def test_wait_and_back(self):
        steps = [
            _step(step=1, action=ActionType.CLICK, target="#btn1"),
            _step(step=2, action=ActionType.WAIT, target=""),
            _step(step=3, action=ActionType.BACK, target=""),
            _step(step=4, action=ActionType.CLICK, target="#btn2"),
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.non_advancing_count == 2
        assert mv.non_advancing_rate == 0.5

    def test_repeated_target(self):
        steps = [
            _step(step=1, action=ActionType.CLICK, target="#submit"),
            _step(step=2, action=ActionType.CLICK, target="#submit"),  # repeated
            _step(step=3, action=ActionType.CLICK, target="#next"),
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.non_advancing_count == 1
        assert abs(mv.non_advancing_rate - 0.333) < 0.01

    def test_repeated_target_outside_lookback_window(self):
        """Repeated target beyond 3-step lookback should NOT count."""
        steps = [
            _step(step=1, action=ActionType.CLICK, target="#submit"),
            _step(step=2, action=ActionType.CLICK, target="#a"),
            _step(step=3, action=ActionType.CLICK, target="#b"),
            _step(step=4, action=ActionType.CLICK, target="#c"),
            # Step 5 repeats step 1, but step 1 is outside the 3-step window
            _step(step=5, action=ActionType.CLICK, target="#submit"),
        ]
        mv = extract_metrics(_session(steps=steps, terminal_reason="max_steps"))
        assert mv.non_advancing_count == 0


class TestAbandonment:
    def test_patience_exhausted(self):
        steps = [_step(step=1), _step(step=2)]
        mv = extract_metrics(_session(steps=steps, terminal_reason="patience_exhausted"))
        assert mv.abandoned is True
        assert mv.steps_before_abandonment == 2

    def test_completed(self):
        steps = [_step(step=1), _step(step=2)]
        mv = extract_metrics(_session(steps=steps, terminal_reason="task_completed"))
        assert mv.abandoned is False
        assert mv.steps_before_abandonment is None


class TestPersonaContext:
    def test_persona_params_included(self):
        mv = extract_metrics(
            _session(persona_name="maria", steps=[_step()], terminal_reason="max_steps")
        )
        assert mv.tech_literacy == "low"
        assert mv.patience == 6

    def test_custom_persona(self):
        persona = Persona(patience=3)
        spec = TaskSpec(task="Test", success_criteria="Done", start_url="https://example.com")
        log = SessionLog(task_spec=spec, persona_snapshot=persona)
        log.finalize("max_steps")
        mv = extract_metrics(log)
        assert mv.persona_name == "custom"


class TestCompareMetrics:
    def test_output_contains_persona_names(self):
        mv1 = extract_metrics(
            _session(persona_name="maria", steps=[_step(confidence=0.3)], terminal_reason="patience_exhausted")
        )
        mv2 = extract_metrics(
            _session(persona_name="jake", steps=[_step(confidence=0.9)], terminal_reason="task_completed")
        )
        output = compare_metrics([mv1, mv2])
        assert "maria" in output
        assert "jake" in output
        assert "yes" in output.lower()

    def test_empty_list(self):
        assert "No metrics" in compare_metrics([])
