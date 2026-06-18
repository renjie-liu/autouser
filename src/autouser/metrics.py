"""M3 measurement harness: SessionLog → MetricVector.

Pure function that extracts the 4 M3 gate metrics from a completed session.
No side effects, no contract mutations. Additive module only.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from autouser.cognitive.models import ActionType, StepResult, TerminalReason
from autouser.session import SessionLog


# ---------------------------------------------------------------------------
# Non-advancing action classifier
# ---------------------------------------------------------------------------

NON_ADVANCING_ACTIONS = frozenset({ActionType.WAIT, ActionType.BACK})
_LOOKBACK_WINDOW = 3


def _is_non_advancing(step: StepResult, previous_steps: list[StepResult]) -> bool:
    """Classify a step as non-advancing.

    A step is non-advancing if:
    - Its action is WAIT or BACK, OR
    - Its (action, target) pair matches any of the previous 3 steps
      within the lookback window.
    """
    if step.intent.action in NON_ADVANCING_ACTIONS:
        return True

    if not step.intent.target:
        return False

    window = previous_steps[-_LOOKBACK_WINDOW:]
    return any(
        prev.intent.action == step.intent.action
        and prev.intent.target == step.intent.target
        for prev in window
    )


# ---------------------------------------------------------------------------
# MetricVector — the output of the measurement harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceStats:
    """Descriptive statistics for ActionIntent.confidence across a session."""

    values: tuple[float, ...]
    mean: float
    stdev: float
    min: float
    max: float
    median: float


@dataclass(frozen=True)
class MetricVector:
    """The 4 M3 gate metrics for a single session run, plus persona context.

    Persona parameters are included for human-readable side-by-side
    comparison (PM requirement: the explanation for metric differences
    should be immediately visible).
    """

    # Persona context
    persona_name: str
    tech_literacy: str
    patience: int
    reading_comprehension: str
    domain_familiarity: str
    goal_clarity: str

    # Metric 1: Confidence distribution
    confidence: ConfidenceStats

    # Metric 2: Exploration breadth
    unique_targets: int  # primary: unique elements interacted with
    unique_urls: int  # secondary: unique pages visited

    # Metric 3: Non-advancing action rate
    non_advancing_count: int
    total_steps: int
    non_advancing_rate: float

    # Metric 4: Steps before abandonment
    abandoned: bool
    steps_before_abandonment: Optional[int]
    terminal_reason: Optional[str]


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def extract_metrics(session: SessionLog) -> MetricVector:
    """Extract M3 gate metrics from a completed SessionLog.

    This is a pure function: it reads the session, computes metrics,
    and returns a frozen dataclass. No mutations, no I/O.
    """
    steps = session.steps
    persona = session.persona_snapshot
    persona_name = persona.archetype.name if persona.archetype else "custom"
    total = len(steps)

    # Metric 1: Confidence distribution
    confidences = tuple(s.intent.confidence for s in steps)
    if confidences:
        conf_stats = ConfidenceStats(
            values=confidences,
            mean=statistics.mean(confidences),
            stdev=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            min=min(confidences),
            max=max(confidences),
            median=statistics.median(confidences),
        )
    else:
        conf_stats = ConfidenceStats(
            values=(), mean=0.0, stdev=0.0, min=0.0, max=0.0, median=0.0,
        )

    # Metric 2: Exploration breadth
    # Primary: unique elements interacted with (intent.target)
    targets = {s.intent.target for s in steps if s.intent.target}
    # Secondary: unique URLs visited (before + after observations)
    urls: set[str] = set()
    for s in steps:
        urls.add(s.observation_before.url)
        urls.add(s.observation_after.url)

    # Metric 3: Non-advancing action rate (3-step lookback window)
    non_advancing = 0
    for i, step in enumerate(steps):
        if _is_non_advancing(step, steps[:i]):
            non_advancing += 1
    rate = non_advancing / total if total > 0 else 0.0

    # Metric 4: Steps before abandonment
    # Count both runner-forced abandonment AND LLM-initiated give_up
    # as abandonment. A low-patience persona's prompt constraints may cause
    # the LLM to emit give_up before the runner's threshold kicks in.
    abandoned_reasons = {TerminalReason.ABANDONED, "abandoned", "patience_exhausted"}
    abandoned = session.terminal_reason in abandoned_reasons or any(
        s.intent.action == ActionType.GIVE_UP for s in steps
    )
    steps_before = total if abandoned else None

    return MetricVector(
        persona_name=persona_name,
        tech_literacy=persona.tech_literacy.value,
        patience=persona.patience,
        reading_comprehension=persona.reading_comprehension.value,
        domain_familiarity=persona.domain_familiarity.value,
        goal_clarity=persona.goal_clarity.value,
        confidence=conf_stats,
        unique_targets=len(targets),
        unique_urls=len(urls),
        non_advancing_count=non_advancing,
        total_steps=total,
        non_advancing_rate=round(rate, 3),
        abandoned=abandoned,
        steps_before_abandonment=steps_before,
        terminal_reason=session.terminal_reason,
    )


# ---------------------------------------------------------------------------
# Human-readable comparison output
# ---------------------------------------------------------------------------


def compare_metrics(vectors: list[MetricVector]) -> str:
    """Format multiple MetricVectors as a human-readable comparison table.

    Persona parameters are shown alongside metrics so the explanation
    for differences is immediately visible.
    """
    if not vectors:
        return "No metrics to compare."

    lines: list[str] = []
    sep = "-" * 72

    lines.append(sep)
    lines.append("M3 METRIC COMPARISON")
    lines.append(sep)

    for v in vectors:
        lines.append(f"\n  Persona: {v.persona_name}")
        lines.append(
            f"  tech_literacy={v.tech_literacy}  patience={v.patience}  "
            f"reading_comp={v.reading_comprehension}  "
            f"domain_fam={v.domain_familiarity}  "
            f"goal_clarity={v.goal_clarity}"
        )
        lines.append("")
        lines.append(
            f"  [1] Confidence   mean={v.confidence.mean:.3f}  "
            f"stdev={v.confidence.stdev:.3f}  "
            f"median={v.confidence.median:.3f}  "
            f"range=[{v.confidence.min:.2f}, {v.confidence.max:.2f}]"
        )
        lines.append(
            f"  [2] Exploration   unique_targets={v.unique_targets}  "
            f"unique_urls={v.unique_urls}"
        )
        lines.append(
            f"  [3] Non-advancing {v.non_advancing_count}/{v.total_steps} "
            f"steps ({v.non_advancing_rate:.1%})"
        )
        abandon_str = (
            f"yes (after {v.steps_before_abandonment} steps)"
            if v.abandoned
            else "no"
        )
        lines.append(
            f"  [4] Abandoned     {abandon_str}  "
            f"(terminal_reason={v.terminal_reason})"
        )
        lines.append(sep)

    return "\n".join(lines)
