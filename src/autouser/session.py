"""Session-level models: task specification and run logging for replay/calibration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from autouser.cognitive.models import StepResult, SuccessPredicate, TerminalReason
from autouser.persona.models import Persona


class TaskSpec(BaseModel):
    """First-class specification of what the simulated user should attempt."""

    task: str = Field(description="What the user is trying to do, e.g. 'Create an account'")
    success_criteria: str = Field(
        description="How to know the task succeeded, e.g. 'User reaches the dashboard'"
    )
    start_url: str
    max_steps: int = Field(default=20, ge=1, le=100)
    give_up_threshold: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override persona patience. If None, uses persona.patience.",
    )
    success_predicate: Optional[SuccessPredicate] = Field(
        default=None,
        description="Optional deterministic success gate. When set, predicate match is the "
        "authoritative exit condition; LLM soft-gate is disabled.",
    )
    max_consecutive_failures: Optional[int] = Field(
        default=3,
        ge=1,
        description="Consecutive identical-failure steps before the runner triggers ABANDONED. "
        "Compared on (action_type, target, error_text_visible). Set None to disable.",
    )
    exploration: bool = Field(
        default=False,
        description="Open-ended exploration mode: no task to complete. The persona "
        "pokes around an unknown product driven by curiosity; the session ends "
        "SUCCESSfully when they feel they've seen enough (reflect's "
        "persona_believes_complete, reinterpreted), or by frustration/time budget. "
        "Predicate and LLM-soft task gates do not apply.",
    )


class ModelMetadata(BaseModel):
    """Tracks which LLM was used for reproducibility and cost analysis."""

    provider: str = Field(default="anthropic", description="LLM provider")
    model: str = Field(default="", description="Model ID used for cognitive engine calls")
    temperature: float = Field(default=0.7)


class SessionLog(BaseModel):
    """Append-only record of a complete simulation run.

    This is the minimum unit for replay analysis, calibration, and
    false-positive triage. Every field needed to understand what happened
    and why is captured here.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    task_spec: TaskSpec
    persona_snapshot: Persona = Field(
        description="Frozen copy of persona params at run start"
    )
    model_metadata: ModelMetadata = Field(default_factory=ModelMetadata)

    steps: list[StepResult] = Field(default_factory=list)
    terminal_reason: Optional[TerminalReason] = Field(
        default=None,
        description="Why the run ended. Closed enum: success, abandoned, timed_out, error.",
    )

    def append_step(self, step: StepResult) -> None:
        """Append a step result with monotonic index enforcement."""
        expected_index = len(self.steps) + 1
        if step.step != expected_index:
            step.step = expected_index
        self.steps.append(step)

    def finalize(self, reason: TerminalReason) -> None:
        self.terminal_reason = reason
        self.completed_at = datetime.now(timezone.utc)
