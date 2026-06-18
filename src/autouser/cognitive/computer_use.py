"""Computer-use cognitive engine: screenshot-based perception with coordinate actions.

Uses Claude's vision capabilities to "see" the page as a real user would —
via screenshots rather than DOM introspection. The model returns coordinate-
based actions (click at x,y) instead of CSS selectors.

Same two-phase contract as CognitiveEngine (plan/reflect), same persona
constraints, same report output. The difference is perception and targeting.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from pathlib import Path

import anthropic

from autouser import affect
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CoordinateTarget,
    Emotion,
    StepResult,
    UIState,
)
from autouser.cognitive.prompts import (
    TECH_LITERACY_CONSTRAINTS,
    _patience_constraint,
    build_reflect_system_prompt,
    build_reflect_user_prompt,
)
from autouser.persona.models import Persona

logger = logging.getLogger(__name__)

# Bare alias — date-suffixed sonnet-4-6 IDs do not exist and 404 at call time.
_DEFAULT_MODEL = "claude-sonnet-4-6"

# Cap a single mental note (mirrors CognitiveEngine._MENTAL_NOTE_MAX_CHARS).
_MENTAL_NOTE_MAX_CHARS = 240

# ---------------------------------------------------------------------------
# Compact state object carried between steps (replaces full visual context).
# The model loses the screenshot each step to save image tokens, but retains
# a structured text summary of what happened so far.
# ---------------------------------------------------------------------------


class _StepMemory:
    """Lightweight memory carried forward between computer-use steps.

    Keeps task continuity without accumulating screenshots in context.
    """

    def __init__(self, goal: str, success_criteria: str) -> None:
        self.goal = goal
        self.success_criteria = success_criteria
        self.entries: list[dict] = []

    def record(
        self,
        step: int,
        action: str,
        target: str,
        expected: str,
        actual: str,
        confidence: float,
        emotion: str,
    ) -> None:
        self.entries.append({
            "step": step,
            "action": action,
            "target": target,
            "expected": expected,
            "actual": actual,
            "confidence": confidence,
            "emotion": emotion,
        })

    def format_history(self, max_steps: int = 5) -> str:
        if not self.entries:
            return "No prior actions."
        recent = self.entries[-max_steps:]
        lines = []
        for e in recent:
            lines.append(
                f"Step {e['step']}: {e['action']} at {e['target']} "
                f"(conf={e['confidence']:.2f}) → {e['actual']} [{e['emotion']}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt for computer-use plan phase
# ---------------------------------------------------------------------------

_CU_PLAN_SYSTEM = """\
You are {persona_name}, a real person using a website. You are NOT an AI \
assistant — you are a human with specific traits, limitations, and habits. \
Your decisions must reflect who you are, not what is optimal.

## Who You Are

{persona_narrative}

## Your Traits

Tech literacy: {tech_literacy}
{tech_literacy_constraint}

Patience: {patience}/10.
{patience_constraint}

## Your Task

You are trying to: {task_description}
You will know you succeeded when: {success_criteria}

## Instructions

You are looking at a screenshot of a webpage. Decide what to do next.
You can click at specific coordinates, type text, scroll, go back, wait, or give up.

## Response Format

You MUST respond with a single JSON object. No other text before or after.

{{
  "action": "click|type|scroll|navigate|back|wait|give_up",
  "x": 640,
  "y": 360,
  "input_value": "text to type or URL (null if not applicable)",
  "persona_thought": "What you notice and why you're taking this action, 1-2 sentences.",
  "expected_outcome": "What you think will happen, 1 sentence.",
  "confidence": 0.5,
  "recognition": "YES_CLEARLY|THINK_SO|NOT_SURE|NO",
  "prediction": "YES_CLEARLY|THINK_SO|NOT_SURE|NO",
  "progress": "YES_CLEARLY|THINK_SO|NOT_SURE|NO"
}}

IMPORTANT:
- x and y are pixel coordinates on the screenshot where you want to click/type.
- For scroll, back, wait, give_up: set x=0, y=0.
- confidence MUST be consistent with your recognition/prediction/progress assessments.
- persona_thought must be 1-2 sentences maximum."""


def _build_cu_system_prompt(
    persona: Persona,
    task_description: str,
    success_criteria: str,
) -> str:
    """Static for the whole session — per-step affect lives in the user prompt
    (mirrors CognitiveEngine; keeps the system prompt byte-stable)."""
    name = persona.archetype.name if persona.archetype else "a user"
    narrative = persona.archetype.narrative if persona.archetype else "A web user."

    return _CU_PLAN_SYSTEM.format(
        persona_name=name,
        persona_narrative=narrative,
        tech_literacy=persona.tech_literacy.value,
        tech_literacy_constraint=TECH_LITERACY_CONSTRAINTS[persona.tech_literacy],
        patience=persona.patience,
        patience_constraint=_patience_constraint(persona.patience),
        task_description=task_description,
        success_criteria=success_criteria,
    )


# ---------------------------------------------------------------------------
# Scaffold field validation (reused from engine.py logic)
# ---------------------------------------------------------------------------

_SCAFFOLD_FIELDS = {"recognition", "prediction", "progress"}
_VALID_ASSESSMENTS = {"YES_CLEARLY", "THINK_SO", "NOT_SURE", "NO"}


def _confidence_bracket(r: str, p: str, g: str) -> tuple[float, float]:
    assessments = [r, p, g]
    if any(a == "NO" for a in assessments):
        return (0.05, 0.24)
    if any(a == "NOT_SURE" for a in assessments):
        return (0.25, 0.54)
    if all(a == "YES_CLEARLY" for a in assessments):
        return (0.85, 1.0)
    return (0.55, 0.84)


# ---------------------------------------------------------------------------
# ComputerUseEngine
# ---------------------------------------------------------------------------


class ComputerUseEngine:
    """Screenshot-based cognitive engine using Claude's vision capabilities.

    Same two-phase contract as CognitiveEngine:
    - plan(): screenshot in → ActionIntent with CoordinateTarget out
    - reflect(): identical to DOM mode (text-based before/after comparison)

    Key differences from CognitiveEngine:
    - Perception is visual (screenshot) not structural (DOM)
    - Targets are pixel coordinates, not CSS selectors
    - Each screenshot gets a unique ID; coordinates are tied to that ID
    - Compact state memory carried forward (no screenshot accumulation)
    """

    def __init__(
        self,
        persona: Persona,
        task_description: str,
        success_criteria: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        self.persona = persona
        self.task_description = task_description
        self.success_criteria = success_criteria
        self.step_count = 0
        self.history: list[StepResult] = []
        self.model = model or _DEFAULT_MODEL
        self.temperature = temperature

        self._client = anthropic.AsyncAnthropic()
        self._memory = _StepMemory(task_description, success_criteria)

        # Track the most recent screenshot ID for staleness validation
        self._current_screenshot_id: str | None = None

    @property
    def _persona_name(self) -> str:
        return self.persona.archetype.name if self.persona.archetype else "custom"

    async def _call_llm_vision(
        self,
        system_prompt: str,
        user_text: str,
        screenshot_path: str,
        max_tokens: int,
    ) -> str:
        """Send a vision request with screenshot + text to Claude."""
        import asyncio as _asyncio

        # Read and encode screenshot
        img_bytes = Path(screenshot_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Determine media type
        suffix = Path(screenshot_path).suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            },
            {
                "type": "text",
                "text": user_text,
            },
        ]

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_content}],
                )
                usage = response.usage
                if hasattr(usage, "cache_creation_input_tokens"):
                    logger.debug(
                        "cu_cache: persona=%s cache_created=%d cache_read=%d input=%d",
                        self._persona_name,
                        getattr(usage, "cache_creation_input_tokens", 0),
                        getattr(usage, "cache_read_input_tokens", 0),
                        usage.input_tokens,
                    )
                return response.content[0].text.strip()
            except Exception as e:
                retriable = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e)
                if not retriable or attempt == max_retries:
                    raise
                wait = 2 ** attempt * 10
                logger.warning(
                    "cu_retry: attempt=%d/%d error=%s waiting=%ds",
                    attempt + 1, max_retries, type(e).__name__, wait,
                )
                await _asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def _call_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """Text-only LLM call (for reflect phase — no screenshot needed)."""
        import asyncio as _asyncio

        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text.strip()
            except Exception as e:
                retriable = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e)
                if not retriable or attempt == max_retries:
                    raise
                wait = 2 ** attempt * 10
                logger.warning(
                    "cu_retry: attempt=%d/%d error=%s waiting=%ds",
                    attempt + 1, max_retries, type(e).__name__, wait,
                )
                await _asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def plan(self, ui_state: UIState) -> ActionIntent:
        """Phase 1: Look at the screenshot, decide what to do next.

        Returns an ActionIntent with a CoordinateTarget tied to the current
        screenshot ID. The executor validates this ID before acting.
        """
        self.step_count += 1

        # Generate a unique screenshot ID for this step
        screenshot_id = f"step{self.step_count}_{uuid.uuid4().hex[:8]}"
        self._current_screenshot_id = screenshot_id

        system_prompt = _build_cu_system_prompt(
            self.persona,
            self.task_description,
            self.success_criteria,
        )

        # Per-step affect (feeling + time on task) from objective trace
        # evidence, replacing the legacy emotion-count — same as the DOM engine.
        affect_summary = affect.summarize(self.history, self.persona)

        # Build user message with compact history (not accumulated screenshots)
        history_text = self._memory.format_history()
        user_text = (
            f"Step {self.step_count}. {affect.affect_line(affect_summary)}\n\n"
            f"Page URL: {ui_state.url}\n"
            f"Page title: {ui_state.page_title}\n\n"
            f"== Your Recent Actions ==\n{history_text}\n\n"
            f"What do you do next?"
        )

        if not ui_state.screenshot_path:
            logger.warning(
                "cu_no_screenshot: persona=%s step=%d, falling back to wait",
                self._persona_name, self.step_count,
            )
            return ActionIntent(
                action=ActionType.WAIT,
                target="",
                persona_thought="I can't see the page clearly.",
                expected_outcome="Nothing happens while I wait.",
                confidence=0.1,
            )

        raw_text = await self._call_llm_vision(
            system_prompt, user_text, ui_state.screenshot_path, max_tokens=512,
        )

        # Parse JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "cu_parse_error: persona=%s step=%d, defaulting to wait",
                self._persona_name, self.step_count,
            )
            return ActionIntent(
                action=ActionType.WAIT,
                target="",
                persona_thought="I'm confused and not sure what to do.",
                expected_outcome="Nothing happens while I think.",
                confidence=0.2,
            )

        # Extract and validate scaffold fields
        scaffold = {}
        for field in _SCAFFOLD_FIELDS:
            value = parsed.pop(field, None)
            if value:
                scaffold[field] = value
                logger.debug(
                    "cu_scaffold_%s: persona=%s value=%s",
                    field, self._persona_name, value,
                )

        # Validate confidence bracket
        r = scaffold.get("recognition", "")
        p = scaffold.get("prediction", "")
        g = scaffold.get("progress", "")
        if all(a in _VALID_ASSESSMENTS for a in [r, p, g]):
            low, high = _confidence_bracket(r, p, g)
            stated = parsed.get("confidence", 0.5)
            try:
                stated = float(stated)
            except (TypeError, ValueError):
                stated = 0.5
            if not (low <= stated <= high):
                logger.warning(
                    "cu_confidence_bracket_mismatch: persona=%s stated=%.2f "
                    "expected=[%.2f, %.2f]",
                    self._persona_name, stated, low, high,
                )
                parsed["confidence"] = max(low, min(high, stated))

        # Extract coordinates and build typed target
        action_str = parsed.get("action", "wait")
        x = parsed.pop("x", 0)
        y = parsed.pop("y", 0)

        try:
            x = int(x)
            y = int(y)
        except (TypeError, ValueError):
            x, y = 0, 0

        action_type = ActionType(action_str)
        needs_target = action_type in {ActionType.CLICK, ActionType.TYPE}

        typed_target = None
        target_str = ""
        if needs_target and (x > 0 or y > 0):
            typed_target = CoordinateTarget(
                x=x, y=y, screenshot_id=screenshot_id,
            )
            target_str = str(typed_target)
        elif needs_target:
            # Model returned 0,0 for a click/type — treat as confusion
            logger.warning(
                "cu_zero_coordinate: persona=%s action=%s, forcing wait",
                self._persona_name, action_str,
            )
            return ActionIntent(
                action=ActionType.WAIT,
                target="",
                persona_thought="I can see the page but I'm not sure where to click.",
                expected_outcome="Nothing happens while I think.",
                confidence=0.15,
            )

        intent = ActionIntent(
            action=action_type,
            target=target_str,
            typed_target=typed_target,
            input_value=parsed.get("input_value"),
            persona_thought=parsed.get("persona_thought", "Looking at the page."),
            expected_outcome=parsed.get("expected_outcome", "Something should happen."),
            confidence=parsed.get("confidence", 0.5),
        )

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
        """Phase 2: Evaluate what happened after action execution.

        Uses text-based comparison (same as DOM mode) — the reflect phase
        doesn't need vision since it compares structural state changes.
        """
        # Short-circuit for give_up
        if intent.action == ActionType.GIVE_UP:
            result = StepResult(
                step=step_number,
                intent=intent,
                observation_before=observation_before,
                observation_after=observation_after,
                actual_outcome="Gave up on the task.",
                mismatch=False,
                reflection=intent.persona_thought,
                emotion=Emotion.FRUSTRATED,
            )
            self._memory.record(
                step_number, "give_up", "", intent.expected_outcome,
                "Gave up.", intent.confidence, "frustrated",
            )
            return result

        system_prompt = build_reflect_system_prompt(self.persona)
        user_prompt = build_reflect_user_prompt(
            intent, observation_before, observation_after, step_number,
        )

        raw_text = await self._call_llm_text(system_prompt, user_prompt, max_tokens=256)

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "cu_reflect_parse_error: persona=%s step=%d",
                self._persona_name, step_number,
            )
            result = StepResult(
                step=step_number,
                intent=intent,
                observation_before=observation_before,
                observation_after=observation_after,
                actual_outcome="Could not determine what happened.",
                mismatch=True,
                reflection="I'm not sure what just happened.",
                emotion=Emotion.CONFUSED,
            )
            self._memory.record(
                step_number, intent.action.value,
                intent.resolve_target(), intent.expected_outcome,
                "Unknown", intent.confidence, "confused",
            )
            return result

        mismatch = bool(parsed.get("mismatch", False))
        severity = parsed.get("severity") if mismatch else None
        category = parsed.get("category") if mismatch else None

        emotion_str = parsed.get("emotion", "uncertain")
        try:
            emotion = Emotion(emotion_str)
        except ValueError:
            emotion = Emotion.UNCERTAIN

        actual_outcome = parsed.get("actual_outcome", "Unknown outcome.")

        # Completion + memory fields drive the runner's soft-success gate, the
        # exploration "seen enough" exit, and the mental model — they must be
        # carried, not dropped. (Screenshot mode without a predicate can only
        # stop via observed_success_signal.) Mirrors CognitiveEngine.reflect().
        persona_believes_complete = bool(parsed.get("persona_believes_complete", False))
        observed_success_signal = bool(parsed.get("observed_success_signal", False))
        mental_note = parsed.get("mental_note")
        if isinstance(mental_note, str):
            mental_note = mental_note.strip() or None
            if mental_note and len(mental_note) > _MENTAL_NOTE_MAX_CHARS:
                mental_note = mental_note[:_MENTAL_NOTE_MAX_CHARS]
        else:
            mental_note = None

        # Record in compact memory for next plan step
        self._memory.record(
            step_number, intent.action.value,
            intent.resolve_target(), intent.expected_outcome,
            actual_outcome, intent.confidence, emotion.value,
        )

        return StepResult(
            step=step_number,
            intent=intent,
            observation_before=observation_before,
            observation_after=observation_after,
            actual_outcome=actual_outcome,
            mismatch=mismatch,
            reflection=parsed.get("reflection", ""),
            emotion=emotion,
            severity=severity,
            category=category,
            persona_believes_complete=persona_believes_complete,
            observed_success_signal=observed_success_signal,
            mental_note=mental_note,
        )

    def should_give_up(self) -> bool:
        """Abandon on objective trace evidence + simulated time (affect.py),
        not on self-reported emotion labels — same mechanism as the DOM engine
        (slice 3). The raw observations carry dom_summary/visible_text/focus
        even in screenshot mode, so the affect signals apply here too."""
        return affect.summarize(self.history, self.persona).should_abandon

    @property
    def current_screenshot_id(self) -> str | None:
        """The screenshot ID from the most recent plan() call."""
        return self._current_screenshot_id
