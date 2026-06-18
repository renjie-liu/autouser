"""Affect state + simulated time: objective give-up mechanics.

docs/cognitive-v2-design-review.md §4. The previous mechanism counted
LLM-labeled emotions ("confused"/"frustrated") toward a step budget — the
measurement (self-reported confusion) was also the mechanism (termination
fuel), circular through one model's self-report, and humans budget minutes,
not steps.

This module derives both signals from evidence the harness can observe:

* **Frustration** accumulates from objective trace events — failed actions,
  clicks that changed nothing, repeated attempts, backtracking — plus
  expectation violations (mismatch) at reduced weight (judged against the
  page, but still a model judgment). Emotion *labels* contribute nothing:
  they remain color in the log, never fuel.
* **Simulated time** charges reading cost proportional to visible text and
  the persona's reading depth (Maria reading everything is genuinely slow;
  Carlos skimming is fast but blind), plus per-action motor costs. Patience
  scales both the frustration threshold and the session time budget.

All functions are pure over (history, persona) — same shape as the memory
composition — so affect is replayable from any recorded session.

Constants are provisional v1 calibration: tests pin the mechanism (which
events count, what scales what), not the literals.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from autouser.cognitive.models import ActionType, StepResult
from autouser.persona.models import Level, Persona

# --- Frustration event weights ------------------------------------------------

W_ACTION_FAILED = 1.0      # action did not execute (selector miss, dead control)
W_NO_EFFECT = 1.0          # click/press executed but nothing observable changed
W_REPEATED = 0.75          # same (action, target, input) as a recent step
W_BACKTRACK = 0.5          # went back — the path was wrong
W_EXPECTATION = 0.5        # mismatch: outcome differed from stated expectation

_REPEAT_LOOKBACK = 3  # mirrors the metrics module's non-advancing window

# --- Simulated time -----------------------------------------------------------

READ_CHARS_PER_SECOND = 25.0  # ~250 wpm adult average
READ_FRACTION = {Level.LOW: 0.2, Level.MEDIUM: 0.5, Level.HIGH: 1.0}
REVISIT_READ_FRACTION = 0.1   # re-scanning a page you've already read
THINK_SECONDS_PER_STEP = 3.0
TYPE_SECONDS_PER_CHAR = 0.25
TYPE_SECONDS_PER_CHAR_LOW_MOTOR = 0.4
TIME_BUDGET_PER_PATIENCE_SECONDS = 90.0

_ACTION_BASE_SECONDS = {
    ActionType.CLICK: 2.0,
    ActionType.TYPE: 1.0,  # plus per-char cost
    ActionType.PRESS: 1.0,
    ActionType.SCROLL: 2.0,
    ActionType.NAVIGATE: 3.0,
    ActionType.BACK: 2.0,
    ActionType.WAIT: 3.0,
    ActionType.GIVE_UP: 0.0,
}


class AffectEvent(BaseModel):
    """One objective frustration-relevant occurrence in the trace."""

    step: int
    kind: str
    weight: float


class AffectSummary(BaseModel):
    """Current affect + time state, derived from the full step history."""

    frustration: float
    threshold: float
    ratio: float  # frustration / threshold, clamped to [0, 1]
    elapsed_seconds: float
    time_budget_seconds: float
    feeling: str
    should_abandon: bool
    abandon_reason: Optional[str] = None  # "frustration" | "time_budget"


def _states_identical(result: StepResult) -> bool:
    before, after = result.observation_before, result.observation_after
    return (
        before.url == after.url
        and before.visible_text == after.visible_text
        and before.dom_summary == after.dom_summary
        and before.focused_element == after.focused_element
    )


def extract_events(history: list[StepResult]) -> list[AffectEvent]:
    """Objective frustration events from the trace.

    Several kinds can fire on one step (a failed retry is both a failure and
    a repeat — and IS more frustrating than either alone). Emotion labels are
    deliberately absent: a step whose only signal is `emotion=frustrated`
    contributes nothing here.
    """
    events: list[AffectEvent] = []
    for i, result in enumerate(history):
        action = result.intent.action

        if result.observation_after.action_error:
            events.append(
                AffectEvent(step=result.step, kind="action_failed", weight=W_ACTION_FAILED)
            )
        elif action in (ActionType.CLICK, ActionType.PRESS) and _states_identical(result):
            # The action ran and visibly did nothing. TYPE is excluded: input
            # values don't appear in innerText, so an effective TYPE looks
            # unchanged to this comparison.
            events.append(
                AffectEvent(step=result.step, kind="no_effect", weight=W_NO_EFFECT)
            )

        signature = (action, result.intent.target, result.intent.input_value)
        for prev in history[max(0, i - _REPEAT_LOOKBACK):i]:
            if (prev.intent.action, prev.intent.target, prev.intent.input_value) == signature:
                events.append(
                    AffectEvent(step=result.step, kind="repeated_action", weight=W_REPEATED)
                )
                break

        if action == ActionType.BACK:
            events.append(
                AffectEvent(step=result.step, kind="backtrack", weight=W_BACKTRACK)
            )

        if result.mismatch:
            events.append(
                AffectEvent(step=result.step, kind="expectation_violation", weight=W_EXPECTATION)
            )
    return events


def frustration_score(history: list[StepResult]) -> float:
    return sum(e.weight for e in extract_events(history))


def frustration_ratio(history: list[StepResult], persona: Persona) -> float:
    threshold = float(max(persona.patience, 1))
    return min(frustration_score(history) / threshold, 1.0)


def simulated_seconds(history: list[StepResult], persona: Persona) -> float:
    """Simulated wall-clock the persona has spent, derived from the trace.

    Reading: full cost on the first encounter of a URL, a small re-scan cost
    afterwards (v1 simplification — same-URL SPA updates are charged as
    revisits). Reading depth scales with reading_comprehension, so thorough
    readers pay in time what skimmers pay in blindness.
    """
    read_fraction = READ_FRACTION[persona.reading_comprehension]
    per_char = (
        TYPE_SECONDS_PER_CHAR_LOW_MOTOR
        if persona.accessibility.motor_precision == Level.LOW
        else TYPE_SECONDS_PER_CHAR
    )

    elapsed = 0.0
    seen_urls: set[str] = set()
    for result in history:
        before = result.observation_before
        if before.url not in seen_urls:
            seen_urls.add(before.url)
            fraction = read_fraction
        else:
            fraction = read_fraction * REVISIT_READ_FRACTION
        elapsed += (len(before.visible_text) * fraction) / READ_CHARS_PER_SECOND

        action = result.intent.action
        elapsed += _ACTION_BASE_SECONDS.get(action, 1.0)
        if action == ActionType.TYPE and result.intent.input_value:
            elapsed += len(result.intent.input_value) * per_char
        elapsed += THINK_SECONDS_PER_STEP
    return elapsed


def feeling_phrase(ratio: float) -> str:
    if ratio < 0.25:
        return "calm — things are going okay so far"
    if ratio < 0.5:
        return "a little annoyed — a few things haven't worked"
    if ratio < 0.75:
        return "getting fed up — too much hasn't worked"
    return "at the end of your patience — about ready to quit"


def summarize(history: list[StepResult], persona: Persona) -> AffectSummary:
    threshold = float(max(persona.patience, 1))
    score = frustration_score(history)
    ratio = min(score / threshold, 1.0)
    elapsed = simulated_seconds(history, persona)
    budget = persona.patience * TIME_BUDGET_PER_PATIENCE_SECONDS

    reason: Optional[str] = None
    if score >= threshold:
        reason = "frustration"
    elif elapsed >= budget:
        reason = "time_budget"

    return AffectSummary(
        frustration=score,
        threshold=threshold,
        ratio=ratio,
        elapsed_seconds=elapsed,
        time_budget_seconds=budget,
        feeling=feeling_phrase(ratio),
        should_abandon=reason is not None,
        abandon_reason=reason,
    )


def affect_line(summary: AffectSummary) -> str:
    """The per-step state line for the plan prompt — in the persona's terms.

    Replaces the old "Patience: X of Y frustration budget used" counter.
    Numbers stay coarse (minutes, not scores) so the persona reasons about
    feeling and time the way a person would, not about harness internals.
    """
    minutes = max(1, round(summary.elapsed_seconds / 60))
    return (
        f"How you're feeling: {summary.feeling}. "
        f"You've been at this about {minutes} minute(s)."
    )
