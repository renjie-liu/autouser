"""Report data models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from autouser.cognitive.models import (
    FeedbackSubtype,
    IssueCategory,
    Severity,
    StepResult,
    TerminalReason,
)


class TaskOutcome(str, Enum):
    """Report-facing outcome. Derived from TerminalReason + friction severity."""
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    ABANDONED = "abandoned"
    TIMED_OUT = "timed_out"
    ERROR = "error"


def outcome_from_terminal_reason(
    reason: TerminalReason,
    has_high_severity_friction: bool = False,
) -> TaskOutcome:
    """Total mapping from TerminalReason to TaskOutcome.

    SUCCESS + high-severity friction promotes to COMPLETED_WITH_ERRORS.
    """
    if reason == TerminalReason.SUCCESS:
        if has_high_severity_friction:
            return TaskOutcome.COMPLETED_WITH_ERRORS
        return TaskOutcome.COMPLETED
    return {
        TerminalReason.ABANDONED: TaskOutcome.ABANDONED,
        TerminalReason.TIMED_OUT: TaskOutcome.TIMED_OUT,
        TerminalReason.ERROR: TaskOutcome.ERROR,
    }[reason]


class Issue(BaseModel):
    """A discrete usability issue found during the simulation."""

    severity: Severity
    category: IssueCategory
    description: str
    step: int
    url: str
    element: Optional[str] = None
    persona_perspective: str = Field(description="Why this is a problem for this persona")
    screenshot_path: Optional[str] = None
    feedback_subtype: Optional[FeedbackSubtype] = Field(
        default=None,
        description=(
            "Per-issue discriminator preserved from FrictionIssue at promotion "
            "time. Lets the renderer distinguish FALSE_SUCCESS / SILENT_SUCCESS "
            "divergence findings from ordinary high-severity issues without "
            "scanning the whole run at render time."
        ),
    )


class Insight(BaseModel):
    """Something the persona figured out about how the site works.

    Distinct from an Issue: insights come from the persona's mental notes
    (reflect), not from mismatches. A workaround the persona had to discover
    ("the field requires E.164 despite the placeholder") is triage-relevant
    even on a run where nothing visibly broke — the discovery cost IS the
    friction.
    """

    step: int
    text: str


class FrictionLog(BaseModel):
    """Complete output of a simulation run."""

    task: str
    success_criteria: str
    persona_name: str
    persona_summary: str
    outcome: TaskOutcome
    total_steps: int
    steps: list[StepResult]
    issues: list[Issue]
    insights: list[Insight] = Field(
        default_factory=list,
        description="The persona's accumulated mental notes — beliefs and "
        "workarounds discovered along the way, in step order.",
    )
    overall_impressions: str = Field(
        description="First impression, confidence trajectory, would this user return?"
    )


class PersonaError(BaseModel):
    """Records a per-persona simulation failure."""

    persona_name: str
    error_type: str
    message: str


class BatchResult(BaseModel):
    """Result of running multiple personas concurrently.

    Partial failures are captured in ``errors`` rather than discarding
    successful runs — if 4 of 5 personas succeed, you get 4 logs + 1 error.
    """

    logs: list[FrictionLog]
    errors: list[PersonaError] = Field(default_factory=list)
