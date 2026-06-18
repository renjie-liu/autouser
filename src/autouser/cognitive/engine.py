"""Cognitive engine: LLM-powered decision-making constrained by persona parameters."""

from __future__ import annotations

import json
import logging
import os
import re

import anthropic

from autouser import affect
from autouser.cognitive.claude_code_provider import ClaudeCodeProvider
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    StepResult,
    UIState,
)
from autouser.cognitive.prompts import (
    build_reflect_system_prompt,
    build_reflect_user_prompt,
    build_system_prompt,
    build_user_prompt,
    format_memory,
)
from autouser.persona.models import Persona

logger = logging.getLogger(__name__)

# Default models per provider.
# Anthropic model IDs are bare aliases — date-suffixed variants like
# "claude-sonnet-4-6-20250514" do not exist and 404 at call time.
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.0-flash",
    "claude_code": "sonnet",
}


# JSON schemas constraining ClaudeCodeProvider output. Mirror the inline
# templates in prompts.py — the prompt still describes the format in prose
# so the model sees both signals; `--json-schema` enforces it at the CLI
# envelope level. Keep these in sync with prompts.py if response shape evolves.
_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "click", "type", "scroll", "navigate", "back", "wait",
                "give_up", "press",
            ],
        },
        "target": {"type": "string"},
        "input_value": {"type": ["string", "null"]},
        "persona_thought": {"type": "string"},
        "expected_outcome": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "recognition": {
            "type": "string",
            "enum": ["YES_CLEARLY", "THINK_SO", "NOT_SURE", "NO"],
        },
        "prediction": {
            "type": "string",
            "enum": ["YES_CLEARLY", "THINK_SO", "NOT_SURE", "NO"],
        },
        "progress": {
            "type": "string",
            "enum": ["YES_CLEARLY", "THINK_SO", "NOT_SURE", "NO"],
        },
    },
    "required": [
        "action", "persona_thought", "expected_outcome", "confidence",
        "recognition", "prediction", "progress",
    ],
}

_REFLECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "actual_outcome": {"type": "string"},
        "mismatch": {"type": "boolean"},
        "reflection": {"type": "string"},
        "emotion": {
            "type": "string",
            "enum": ["confident", "uncertain", "confused", "frustrated", "satisfied"],
        },
        "severity": {
            "type": ["string", "null"],
            "enum": ["high", "medium", "low", None],
        },
        "category": {
            "type": ["string", "null"],
            "enum": [
                "labeling", "navigation", "accessibility", "feedback",
                "error_recovery", "layout", "copy", None,
            ],
        },
        "persona_believes_complete": {"type": "boolean"},
        "observed_success_signal": {"type": "boolean"},
        "mental_note": {"type": ["string", "null"]},
    },
    "required": [
        "actual_outcome", "mismatch", "reflection", "emotion",
        "persona_believes_complete", "observed_success_signal", "mental_note",
    ],
}

# Cap a single mental note — it's a jotted memory, not an essay. Longer model
# output is truncated rather than rejected so a verbose note never crashes a step.
_MENTAL_NOTE_MAX_CHARS = 240

# --- Dual-process (fast/slow) reflect ----------------------------------------
# Below this plan confidence the persona was unsure going in, so the step is a
# decision point worth the full LLM reflect even if nothing visibly broke.
_FAST_CONFIDENCE_FLOOR = 0.55

# Stopwords stripped from success criteria before the completion-plausibility
# check, so generic phrasing ("the user sees the page") doesn't force constant
# deliberation. Intentionally small — only the words that appear in almost any
# criterion. Keeping it tight means the gate errs toward deliberating (safe).
_SUCCESS_STOPWORDS = frozenset({
    "after", "with", "when", "that", "this", "your", "page", "user", "sees",
    "should", "have", "been", "into", "onto", "from", "they", "their", "will",
    "able", "then", "once", "while", "where", "which", "there", "here",
})


def _states_unchanged(before: UIState, after: UIState) -> bool:
    """No observable effect between two states (the 'no_effect' signal)."""
    return (
        before.url == after.url
        and before.visible_text == after.visible_text
        and before.dom_summary == after.dom_summary
        and before.focused_element == after.focused_element
    )


def _detect_provider() -> str:
    """Auto-detect provider. Explicit AUTOUSER_PROVIDER env var wins; otherwise
    falls back to API-key presence (Anthropic → Gemini → anthropic default).

    Note: `claude_code` is opt-in via AUTOUSER_PROVIDER=claude_code rather
    than PATH-detected, so adding the binary to PATH doesn't silently switch
    a CI run from API-driven to subprocess-driven.

    F4 (distinguished-eng PR #39 static-diff): AUTOUSER_PROVIDER is honored
    for `anthropic` / `gemini` too, not just `claude_code`. Consequence:
    AUTOUSER_PROVIDER=gemini with no GEMINI_API_KEY will select gemini and
    fail at client init rather than silently falling back to anthropic.
    This is intentional — silent fallback after an explicit opt-in would
    hide operator intent (a CI run that was meant to test the Gemini path
    quietly testing Anthropic instead is the failure mode this prevents).
    """
    explicit = os.environ.get("AUTOUSER_PROVIDER", "").strip()
    if explicit in _DEFAULT_MODELS:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "anthropic"  # fallback — will fail at call time with a clear error

# Transient scaffold fields returned by the LLM but not part of ActionIntent.
_SCAFFOLD_FIELDS = {"recognition", "prediction", "progress"}

# Valid assessment values for bracket validation.
_VALID_ASSESSMENTS = {"YES_CLEARLY", "THINK_SO", "NOT_SURE", "NO"}


def _confidence_bracket(recognition: str, prediction: str, progress: str) -> tuple[float, float]:
    """Return (low, high) confidence bracket for given R/P/G assessments."""
    assessments = [recognition, prediction, progress]
    if any(a == "NO" for a in assessments):
        return (0.05, 0.24)
    if any(a == "NOT_SURE" for a in assessments):
        return (0.25, 0.54)
    if all(a == "YES_CLEARLY" for a in assessments):
        return (0.85, 1.0)
    return (0.55, 0.84)


class CognitiveEngine:
    """Drives the two-phase decision loop for a simulated user.

    Phase 1 (plan): Given persona + UI state + task, produce an ActionIntent.
    Phase 2 (reflect): Given the intent + new UI state, produce a StepResult.

    The engine is responsible for constraining LLM behavior to match persona
    parameters. This is the hardest problem in the system — an unconstrained
    LLM will behave like a power user regardless of persona settings.
    """

    def __init__(
        self,
        persona: Persona,
        task_description: str,
        success_criteria: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        provider: str | None = None,
        exploration: bool = False,
        fast_mode: bool = False,
    ) -> None:
        self.persona = persona
        self.task_description = task_description
        self.success_criteria = success_criteria
        self.exploration = exploration
        self.fast_mode = fast_mode
        self.step_count = 0
        self.history: list[StepResult] = []
        self.provider = provider or _detect_provider()
        self.model = model or _DEFAULT_MODELS.get(self.provider, "claude-sonnet-4-6")
        self.temperature = temperature
        self._system_prompt: str | None = None

        # Initialize provider-specific client
        self._client = None
        self._gemini_client = None
        self._claude_code_client: ClaudeCodeProvider | None = None
        if self.provider == "gemini":
            from google import genai
            self._gemini_client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
            )
        elif self.provider == "claude_code":
            self._claude_code_client = ClaudeCodeProvider(model=self.model)
        else:
            self._client = anthropic.AsyncAnthropic()

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        *,
        schema: dict[str, object] | None = None,
    ) -> str:
        """Provider-agnostic LLM call. Returns raw text response.

        `schema` is required by the claude_code provider (CLI enforces it
        via --json-schema); other providers ignore it (their templates
        describe the format in prose).

        Retries up to 5 times with exponential backoff on transient errors (429, 503).
        """
        import asyncio as _asyncio

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                result = await self._call_llm_once(
                    system_prompt, user_prompt, max_tokens, schema=schema,
                )
                # Rate-limit pacing for free-tier APIs (~2 RPM)
                if self.provider == "gemini":
                    await _asyncio.sleep(15)
                return result
            except Exception as e:
                retriable = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e)
                if not retriable or attempt == max_retries:
                    raise
                wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                logger.warning(
                    "llm_retry: attempt=%d/%d error=%s waiting=%ds",
                    attempt + 1, max_retries, type(e).__name__, wait,
                )
                await _asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def _call_llm_once(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        *,
        schema: dict[str, object] | None = None,
    ) -> str:
        """Single LLM call attempt.

        For Anthropic: uses prompt caching on the system prompt. The persona
        constraints, task description, and response format are ~90% static
        across steps — only frustration count changes. Caching this prefix
        cuts input token costs by ~90% on steps 2+ and reduces latency.
        """
        if self.provider == "gemini":
            from google.genai import types
            response = await self._gemini_client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=self.temperature,
                ),
            )
            return response.text.strip()
        elif self.provider == "claude_code":
            # F3 (distinguished-eng PR #39 static-diff): explicit check over
            # `assert` — assertions get stripped under `python -O` and would
            # turn a real None into an AttributeError further down. The
            # __init__ partition guarantees this is non-None for this branch,
            # but enforce it as a runtime invariant.
            if self._claude_code_client is None:
                raise RuntimeError(
                    "claude_code provider selected but client not initialized; "
                    "engine __init__ partition violated"
                )
            if schema is None:
                raise ValueError(
                    "claude_code provider requires a schema; "
                    "callers must pass schema=_PLAN_SCHEMA or _REFLECT_SCHEMA"
                )
            # F2 (distinguished-eng PR #39 static-diff): max_tokens is
            # deliberately not forwarded to the CLI. `claude -p` has no
            # --max-tokens flag; structured output is bounded by
            # --json-schema at the CLI envelope level (the schema's
            # required fields keep responses small in practice). Caller-side
            # max_tokens is advisory for this provider, authoritative for
            # the others — documented divergence, not a silent drop.
            # CLI takes a single prompt argument; concatenate system + user
            # with an explicit separator so the model sees them as distinct
            # contexts. `claude -p` has no separate system_prompt flag.
            combined = f"{system_prompt}\n\n---\n\n{user_prompt}"
            return await self._claude_code_client.complete(combined, schema=schema)
        else:
            # Structure system prompt for prompt caching.
            # Mark the entire system block as cacheable — Anthropic will
            # cache the prefix and only reprocess the delta on subsequent calls.
            system_blocks = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system_blocks,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Log cache performance for diagnostics
            usage = response.usage
            if hasattr(usage, "cache_creation_input_tokens"):
                logger.debug(
                    "prompt_cache: persona=%s cache_created=%d cache_read=%d input=%d",
                    self._persona_name,
                    getattr(usage, "cache_creation_input_tokens", 0),
                    getattr(usage, "cache_read_input_tokens", 0),
                    usage.input_tokens,
                )
            return response.content[0].text.strip()

    def _get_system_prompt(self) -> str:
        """Build and cache the system prompt — static for the whole session.

        Per-step state (frustration spent) lives in the user prompt only.
        Keeping the system prompt byte-stable is what makes the Anthropic
        prompt-cache breakpoint actually hit on steps 2+; the previous design
        rebuilt it every step to refresh a frustration counter that was
        already present in the user message, paying full input cost per step.
        """
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                self.persona,
                self.task_description,
                self.success_criteria,
                exploration=self.exploration,
            )
        return self._system_prompt

    def _compose_memory(self) -> str:
        """Assemble the persona's mental-model section from step history.

        Places are derived mechanically (the harness never forgets a visited
        page on the persona's behalf); notes are the persona's own jotted
        beliefs from reflect(). Returns "" on step 1 — no memory yet.
        """
        places: list[tuple[int, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for result in self.history:
            for obs in (result.observation_before, result.observation_after):
                key = (obs.page_title, obs.url)
                if key not in seen:
                    seen.add(key)
                    places.append((result.step, obs.page_title, obs.url))
        notes = [
            (result.step, result.mental_note)
            for result in self.history
            if result.mental_note
        ]
        return format_memory(places, notes)

    async def plan(self, ui_state: UIState) -> ActionIntent:
        """Phase 1: Decide what to do next. Returns intent BEFORE action execution.

        Calls the Anthropic API with persona-constrained prompting.
        The prompt includes scaffold fields (recognition/prediction/progress)
        that force grounded confidence scoring. These are logged at debug
        level then stripped — only ActionIntent fields persist.
        """
        self.step_count += 1

        system_prompt = self._get_system_prompt()
        affect_summary = affect.summarize(self.history, self.persona)
        user_prompt = build_user_prompt(
            ui_state, self.step_count, self.history,
            memory_section=self._compose_memory(),
            affect_line=affect.affect_line(affect_summary),
        )

        raw_text = await self._call_llm(
            system_prompt, user_prompt, max_tokens=512, schema=_PLAN_SCHEMA,
        )

        # Parse JSON — handle markdown code fences if the model wraps output
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "plan_parse_error: persona=%s provider=%s, could not parse JSON, "
                "defaulting to wait. raw_text(len=%d)[:400]=%r",
                self._persona_name, self.provider, len(raw_text), raw_text[:400],
            )
            return ActionIntent(
                action=ActionType.WAIT,
                target="",
                input_value=None,
                persona_thought="I'm confused and not sure what to do.",
                expected_outcome="Nothing happens while I think.",
                confidence=0.2,
            )

        # Extract and log scaffold fields for diagnostics, then strip them
        scaffold = {}
        for field in _SCAFFOLD_FIELDS:
            value = parsed.pop(field, None)
            if value:
                scaffold[field] = value
                logger.debug("scaffold_%s: persona=%s value=%s", field, self._persona_name, value)

        # Validate confidence against R/P/G brackets
        r, p, g = scaffold.get("recognition", ""), scaffold.get("prediction", ""), scaffold.get("progress", "")
        if all(a in _VALID_ASSESSMENTS for a in [r, p, g]):
            low, high = _confidence_bracket(r, p, g)
            stated_confidence = parsed.get("confidence", 0.5)
            try:
                stated_confidence = float(stated_confidence)
            except (TypeError, ValueError):
                stated_confidence = 0.5
            if not (low <= stated_confidence <= high):
                logger.warning(
                    "confidence_bracket_mismatch: persona=%s stated=%.2f "
                    "expected=[%.2f, %.2f] R=%s P=%s G=%s",
                    self._persona_name, stated_confidence, low, high, r, p, g,
                )
                # Clamp to bracket — LLM contradicted its own assessment
                parsed["confidence"] = max(low, min(high, stated_confidence))

        # Log constraint violations (e.g., low-literacy persona using navigate)
        action_value = parsed.get("action", "")
        if (
            action_value == "navigate"
            and self.persona.tech_literacy.value == "low"
        ):
            logger.debug(
                "constraint_violation: %s used navigate (tech_literacy=low)",
                self._persona_name,
            )

        # Normalize null input_value
        if parsed.get("input_value") is None:
            parsed["input_value"] = None

        # Defensive: if action is not a valid enum value, infer intent.
        # Gemini sometimes puts the target selector in the action field.
        action_raw = parsed.get("action", "")
        valid_actions = {a.value for a in ActionType}
        if action_raw not in valid_actions:
            logger.warning(
                "action_enum_mismatch: persona=%s got action='%s', "
                "falling back to click with value as target",
                self._persona_name, action_raw,
            )
            if not parsed.get("target"):
                parsed["target"] = action_raw
            parsed["action"] = ActionType.CLICK.value

        return ActionIntent(**parsed)

    @property
    def _persona_name(self) -> str:
        return self.persona.archetype.name if self.persona.archetype else "custom"

    @property
    def _success_keywords(self) -> set[str]:
        """Significant words from the success criteria, cached.

        Used by the dual-process gate to decide whether a step *might* be a
        completion — if so it must go through the full LLM reflect so success
        detection isn't lost on the fast path.
        """
        cached = getattr(self, "_success_kw_cache", None)
        if cached is None:
            words = re.findall(r"[a-z]{4,}", self.success_criteria.lower())
            cached = {w for w in words if w not in _SUCCESS_STOPWORDS}
            self._success_kw_cache = cached
        return cached

    def _success_plausible(self, state: UIState) -> bool:
        kw = self._success_keywords
        if not kw:
            return False
        text = state.visible_text.lower()
        hits = sum(1 for w in kw if w in text)
        return hits / len(kw) >= 0.5

    def _should_deliberate(
        self,
        intent: ActionIntent,
        observation_before: UIState,
        observation_after: UIState,
    ) -> bool:
        """Dual-process gate: True => run the full LLM reflect (System 2).

        Humans deliberate on surprise, uncertainty, and decision points and
        cruise (System 1) otherwise. We deliberate when any of these hold, and
        ONLY then skip the reflect LLM call:
          * exploration mode — the 'seen enough' verdict is subjective, LLM-only;
          * the action failed (action_error) — surprise / friction to capture;
          * the page did not change (no_effect) — the persona expected something;
          * the persona was unsure going in (low confidence);
          * a completion is plausible — success-criteria words appear on the
            page, so reflect must run to detect success (predicate-gated runs
            don't depend on this, but soft-gate ones do).
        """
        if self.exploration:
            return True
        if observation_after.action_error:
            return True
        if _states_unchanged(observation_before, observation_after):
            return True
        if intent.confidence < _FAST_CONFIDENCE_FLOOR:
            return True
        if self._success_plausible(observation_after):
            return True
        if self._first_encounter(observation_before):
            # People read carefully on ARRIVAL at a new page, then cruise.
            # Reflection on a first-seen page is where the persona forms its
            # understanding (and writes the mental note later steps depend on),
            # so a first encounter is a decision point even if nothing broke.
            # Found live: skipping it made careful-reader Priya miss a format
            # hint she'd otherwise have recorded and reused.
            return True
        return False

    def _first_encounter(self, observation_before: UIState) -> bool:
        """True if the persona has not been on this page before this step.

        Page identity is (url, page_title) — enough to distinguish SPA view
        changes that keep the URL constant, without churning on every minor
        in-page content delta.
        """
        sig = (observation_before.url, observation_before.page_title)
        for result in self.history:
            for obs in (result.observation_before, result.observation_after):
                if (obs.url, obs.page_title) == sig:
                    return False
        return True

    def _fast_step_result(
        self,
        intent: ActionIntent,
        observation_before: UIState,
        observation_after: UIState,
        step_number: int,
    ) -> StepResult:
        """Deterministic System-1 reflect: no LLM call.

        A cruising step that went as expected. mismatch=False (so it adds no
        affect frustration), no mental_note (routine steps teach nothing),
        and success flags False — only reachable when `_should_deliberate`
        already ruled out a plausible completion.
        """
        return StepResult(
            step=step_number,
            intent=intent,
            observation_before=observation_before,
            observation_after=observation_after,
            actual_outcome="The page responded the way I expected.",
            mismatch=False,
            reflection="That went how I expected, so I'll keep going.",
            emotion=Emotion.CONFIDENT,
            persona_believes_complete=False,
            observed_success_signal=False,
        )

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

        Compares expected_outcome (from intent) against actual new state.
        Generates mismatch detection, emotional response, and issue categorization.
        Carries both pre- and post-action observations for replay and analysis.

        The caller provides step_number so the returned StepResult is fully
        constructed — no post-hoc mutation needed.
        """
        # Short-circuit for give_up — no LLM call needed
        if intent.action == ActionType.GIVE_UP:
            return StepResult(
                step=step_number,
                intent=intent,
                observation_before=observation_before,
                observation_after=observation_after,
                actual_outcome="Gave up on the task.",
                mismatch=False,
                reflection=intent.persona_thought,
                emotion=Emotion.FRUSTRATED,
            )

        # Dual-process fast path — cruising step, no LLM reflect (System 1).
        if self.fast_mode and not self._should_deliberate(
            intent, observation_before, observation_after
        ):
            logger.debug("fast_reflect: persona=%s step=%d", self._persona_name, step_number)
            return self._fast_step_result(
                intent, observation_before, observation_after, step_number
            )

        system_prompt = build_reflect_system_prompt(
            self.persona, exploration=self.exploration
        )
        user_prompt = build_reflect_user_prompt(
            intent, observation_before, observation_after, step_number,
        )

        raw_text = await self._call_llm(
            system_prompt, user_prompt, max_tokens=256, schema=_REFLECT_SCHEMA,
        )

        # Handle markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "reflect_parse_error: persona=%s provider=%s step=%d, defaulting "
                "to uncertain. raw_text(len=%d)[:400]=%r",
                self._persona_name, self.provider, step_number,
                len(raw_text), raw_text[:400],
            )
            return StepResult(
                step=step_number,
                intent=intent,
                observation_before=observation_before,
                observation_after=observation_after,
                actual_outcome="Could not determine what happened.",
                mismatch=True,
                reflection="I'm not sure what just happened.",
                emotion=Emotion.CONFUSED,
            )

        # Enforce severity/category consistency: only set when mismatch is true
        mismatch = bool(parsed.get("mismatch", False))
        severity = parsed.get("severity") if mismatch else None
        category = parsed.get("category") if mismatch else None

        # Validate emotion is a known value
        emotion_str = parsed.get("emotion", "uncertain")
        try:
            emotion = Emotion(emotion_str)
        except ValueError:
            logger.warning(
                "reflect_invalid_emotion: persona=%s value=%s, defaulting to uncertain",
                self._persona_name, emotion_str,
            )
            emotion = Emotion.UNCERTAIN

        # Log emotion-mismatch inconsistency but don't override — signal, not suppression
        if mismatch and emotion in (Emotion.CONFIDENT, Emotion.SATISFIED):
            logger.warning(
                "reflect_emotion_mismatch: persona=%s step=%d mismatch=True but emotion=%s",
                self._persona_name, step_number, emotion.value,
            )

        persona_believes_complete = bool(parsed.get("persona_believes_complete", False))
        observed_success_signal = bool(parsed.get("observed_success_signal", False))

        # Mental note — the persona's jotted memory of how the site works.
        # Treat empty/whitespace as None; truncate runaway notes.
        mental_note = parsed.get("mental_note")
        if isinstance(mental_note, str):
            mental_note = mental_note.strip() or None
            if mental_note and len(mental_note) > _MENTAL_NOTE_MAX_CHARS:
                mental_note = mental_note[:_MENTAL_NOTE_MAX_CHARS]
        else:
            mental_note = None

        return StepResult(
            step=step_number,
            intent=intent,
            observation_before=observation_before,
            observation_after=observation_after,
            actual_outcome=parsed.get("actual_outcome", "Unknown outcome."),
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
        """Check whether the persona abandons — frustration or time budget.

        Driven by `affect.summarize`: objective trace evidence (failed
        actions, no-effect clicks, repeats, backtracking, expectation
        violations) plus simulated elapsed time, both scaled by patience.
        Emotion labels deliberately contribute nothing — the previous
        mechanism counted self-reported confusion toward termination, making
        the measurement its own fuel (design review §4).
        """
        return affect.summarize(self.history, self.persona).should_abandon
