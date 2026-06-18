"""ScriptedEngine — deterministic engine implementation for CI fixture tests.

Satisfies EngineProtocol. Returns pre-baked ActionIntents from plan() and
constructs StepResults from pre-baked reflection fields in reflect(). No
LLM calls, no network, no nondeterminism. Used by the M5 fixture harness
to exercise SimulationRunner end-to-end without depending on Gemini quotas
or external sites.

Design notes:
- plan() returns intents in order. After the script is exhausted, plan()
  returns a GIVE_UP intent so the runner terminates cleanly rather than
  blowing up with IndexError.
- reflect() pairs with the plan call by step_number (1-indexed in the
  runner loop) — the scripted step at index (step_number - 1) carries the
  reflection fields written into the returned StepResult.
- should_give_up() is driven by an optional fixed step threshold, used by
  tests that want to assert dwell-loop or persona-exhaustion paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    IssueCategory,
    Severity,
    StepResult,
    UIState,
)


@dataclass
class ScriptedStep:
    """One scripted plan+reflect pair. Plan returns `intent`; reflect builds a
    StepResult with the remaining fields, filling observations from runner input."""

    intent: ActionIntent
    actual_outcome: str = "scripted outcome"
    reflection: str = "scripted reflection"
    emotion: Emotion = Emotion.CONFIDENT
    mismatch: bool = False
    severity: Optional[Severity] = None
    category: Optional[IssueCategory] = None
    persona_believes_complete: bool = False
    observed_success_signal: bool = False


def _give_up_intent() -> ActionIntent:
    return ActionIntent(
        action=ActionType.GIVE_UP,
        persona_thought="scripted: end of script",
        expected_outcome="terminate the simulation",
        confidence=1.0,
    )


class ScriptedEngine:
    """EngineProtocol-satisfying engine driven by a pre-baked list of ScriptedSteps."""

    def __init__(
        self,
        steps: list[ScriptedStep],
        give_up_after: Optional[int] = None,
    ) -> None:
        self._steps = steps
        self._plan_idx = 0
        self._give_up_after = give_up_after
        self.history: list[StepResult] = []

    async def plan(self, ui_state: UIState) -> ActionIntent:
        if self._plan_idx >= len(self._steps):
            return _give_up_intent()
        intent = self._steps[self._plan_idx].intent
        self._plan_idx += 1
        return intent

    async def reflect(
        self,
        intent: ActionIntent,
        new_ui_state: UIState,
        *,
        observation_before: UIState,
        observation_after: UIState,
        step_number: int,
    ) -> StepResult:
        idx = step_number - 1
        if 0 <= idx < len(self._steps):
            step = self._steps[idx]
            return StepResult(
                step=step_number,
                intent=intent,
                observation_before=observation_before,
                observation_after=observation_after,
                actual_outcome=step.actual_outcome,
                mismatch=step.mismatch,
                reflection=step.reflection,
                emotion=step.emotion,
                severity=step.severity,
                category=step.category,
                persona_believes_complete=step.persona_believes_complete,
                observed_success_signal=step.observed_success_signal,
            )
        # Past-script reflect (paired with the synthetic GIVE_UP plan): minimal result.
        return StepResult(
            step=step_number,
            intent=intent,
            observation_before=observation_before,
            observation_after=observation_after,
            actual_outcome="scripted: past end of script",
            mismatch=False,
            reflection="scripted",
            emotion=Emotion.CONFIDENT,
        )

    def should_give_up(self) -> bool:
        return self._give_up_after is not None and self._plan_idx >= self._give_up_after
