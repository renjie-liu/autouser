"""Data models for cognitive model input/output."""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class UIState(BaseModel):
    """Snapshot of the current UI state as perceived by the simulated user."""

    url: str
    page_title: str
    dom_summary: str = Field(description="Simplified/accessible DOM representation")
    screenshot_path: Optional[str] = None
    visible_text: str = Field(default="", description="Extracted visible text content")
    aria_snapshot: Optional[str] = Field(
        default=None,
        description="Accessibility-tree snapshot (role + accessible name, document "
        "order). Raw material for the screen-reader perception filter.",
    )
    perceived_via: Optional[str] = Field(
        default=None,
        description="Set by the perception filter when this state is a persona-"
        "filtered view rather than the raw capture (e.g. 'screen_reader'). "
        "None means unfiltered DOM perception.",
    )
    focused_element: Optional[str] = Field(
        default=None,
        description="Short description of the element holding keyboard focus, "
        "or None when focus is on the document body. Load-bearing for keyboard "
        "navigation: Tab/Enter are blind without knowing where focus sits.",
    )
    action_error: Optional[str] = Field(
        default=None,
        description="Set when the action that produced this state failed to execute "
        "(e.g. selector matched nothing). The simulated user perceives this as "
        "'nothing happened'; this field carries the technical detail for debugging.",
    )

    def has_selector(self, selector: str) -> bool:
        """Check if a CSS selector appears in the DOM summary."""
        return selector in self.dom_summary


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    GIVE_UP = "give_up"
    BACK = "back"
    PRESS = "press"  # keyboard key (Tab/Enter/Space/Escape/arrows) — keyboard navigation


# ---------------------------------------------------------------------------
# Discriminated target types — keeps DOM mode and computer-use mode concerns
# cleanly separated. Each variant is self-validating.
# ---------------------------------------------------------------------------


class SelectorTarget(BaseModel):
    """DOM-mode target: a CSS selector from the page elements list."""

    kind: Literal["selector"] = "selector"
    selector: str = Field(min_length=1, description="CSS selector for the target element")

    def __str__(self) -> str:
        return self.selector


class CoordinateTarget(BaseModel):
    """Computer-use-mode target: pixel coordinates tied to a specific screenshot."""

    kind: Literal["coordinate"] = "coordinate"
    x: int = Field(ge=0, description="X pixel coordinate")
    y: int = Field(ge=0, description="Y pixel coordinate")
    screenshot_id: str = Field(
        min_length=1,
        description="ID of the screenshot these coordinates reference — "
        "executor rejects if stale",
    )

    def __str__(self) -> str:
        return f"({self.x}, {self.y})@{self.screenshot_id}"


Target = Annotated[
    Union[SelectorTarget, CoordinateTarget],
    Field(discriminator="kind"),
]


def _target_from_str(raw: str) -> SelectorTarget:
    """Backward-compat helper: wrap a plain string into a SelectorTarget."""
    return SelectorTarget(selector=raw) if raw else SelectorTarget(selector="__none__")


class ActionIntent(BaseModel):
    """Phase 1 output: what the persona intends to do and why, BEFORE execution."""

    action: ActionType
    target: str = Field(default="", description="CSS selector or description of the target element")
    typed_target: Optional[Union[SelectorTarget, CoordinateTarget]] = Field(
        default=None,
        description="Typed target — preferred over raw 'target' string when set",
    )
    input_value: Optional[str] = Field(default=None, description="Text to type, URL to navigate to")
    persona_thought: str = Field(description="Internal monologue explaining reasoning")
    expected_outcome: str = Field(description="What the persona expects to happen")
    confidence: float = Field(ge=0.0, le=1.0, description="How sure the persona is about this action")

    @model_validator(mode="after")
    def validate_action_fields(self) -> ActionIntent:
        """Reject invalid field combinations for the action type.

        TYPE deliberately does NOT require a target: with an empty target the
        executor types into whatever currently has focus — the keyboard-user
        flow (Tab to a field, then type). Typing with nothing focused is a
        real-user failure mode and surfaces as UIState.action_error, not a
        validation error.
        """
        requires_target = {ActionType.CLICK}
        has_target = bool(self.target) or self.typed_target is not None
        if self.action in requires_target and not has_target:
            raise ValueError(f"Action '{self.action.value}' requires a non-empty target")
        requires_input = {ActionType.TYPE, ActionType.NAVIGATE, ActionType.PRESS}
        if self.action in requires_input and not self.input_value:
            raise ValueError(f"Action '{self.action.value}' requires input_value")
        return self

    def resolve_target(self) -> str:
        """Return the effective target string for execution.

        Prefers typed_target if set, falls back to raw target string.
        """
        if self.typed_target is not None:
            return str(self.typed_target)
        return self.target


class Emotion(str, Enum):
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(str, Enum):
    LABELING = "labeling"
    NAVIGATION = "navigation"
    ACCESSIBILITY = "accessibility"
    FEEDBACK = "feedback"
    ERROR_RECOVERY = "error_recovery"
    LAYOUT = "layout"
    COPY = "copy"


class TerminalReason(str, Enum):
    """Why the simulation loop stopped. Closed 4-value domain."""
    SUCCESS = "success"
    ABANDONED = "abandoned"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class CompletionSource(str, Enum):
    """Which path determined task completion — measurement-quality tier."""
    PREDICATE = "predicate"
    LLM_SOFT = "llm_soft"
    EXPLORATION = "exploration"  # persona felt they'd seen enough (explore mode)
    NONE = "none"


class FeedbackSubtype(str, Enum):
    """Machine-readable discriminator for divergence findings."""
    FALSE_SUCCESS = "false_success"
    SILENT_SUCCESS = "silent_success"


class FrictionIssue(BaseModel):
    """Step-level friction annotation. Used for divergence findings."""
    category: IssueCategory
    severity: Severity
    step: int
    description: str
    feedback_subtype: Optional[FeedbackSubtype] = None


class SuccessPredicate(BaseModel):
    """Optional deterministic success gate on TaskSpec.

    Conjunctive: every configured field must match the post-step state.
    Absent fields are ignored. At least one field must be configured.
    """
    url_pattern: Optional[str] = None
    selector_present: Optional[str] = None
    text_contains: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> SuccessPredicate:
        if all(f is None for f in [self.url_pattern, self.selector_present, self.text_contains]):
            raise ValueError("SuccessPredicate requires at least one configured field")
        return self

    def matches(self, state: UIState) -> bool:
        """All configured fields must match. Returns False if no fields are set."""
        checks: list[bool] = []
        if self.url_pattern is not None:
            checks.append(bool(re.search(self.url_pattern, state.url)))
        if self.selector_present is not None:
            checks.append(state.has_selector(self.selector_present))
        if self.text_contains is not None:
            checks.append(self.text_contains in state.visible_text)
        return bool(checks) and all(checks)


class StepResult(BaseModel):
    """Phase 2 output: post-action reflection including mismatch detection.

    Carries observation context from both before and after the action
    to support mismatch analysis and replay.
    """

    step: int
    intent: ActionIntent
    observation_before: UIState = Field(description="UI state the persona saw before acting")
    observation_after: UIState = Field(description="UI state captured after action executed")
    actual_outcome: str
    mismatch: bool = Field(description="Did the actual outcome differ from expected?")
    reflection: str = Field(description="Post-action persona thought")
    emotion: Emotion
    mental_note: Optional[str] = Field(
        default=None,
        description="What the persona would jot down to remember about how this "
        "site works — one short sentence, or None when the step taught nothing. "
        "Accumulated notes are the persona's evolving mental model and are fed "
        "back into subsequent plan() prompts.",
    )
    severity: Optional[Severity] = Field(
        default=None, description="If mismatch: issue severity"
    )
    category: Optional[IssueCategory] = Field(
        default=None, description="Issue category for filtering and triage"
    )
    screenshot_path: Optional[str] = None
    persona_believes_complete: bool = Field(
        default=False,
        description="Subjective: persona thinks the task is done. From reflect(), never controls exit.",
    )
    observed_success_signal: bool = Field(
        default=False,
        description="LLM's read of whether success criteria appear met in post-action state. "
        "Runner-internal: consumed by the soft gate, never authoritative on its own.",
    )
    success_criteria_met: bool = Field(
        default=False,
        description="Runner-computed. Set by SuccessPredicate match or guarded observed_success_signal. "
        "Never written directly by the LLM.",
    )
    completion_source: CompletionSource = Field(
        default=CompletionSource.NONE,
        description="Which exit path fired: PREDICATE (deterministic), LLM_SOFT (guarded), or NONE.",
    )
    friction_issues: list[FrictionIssue] = Field(default_factory=list)
