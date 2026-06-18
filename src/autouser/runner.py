"""Top-level simulation runner: orchestrates the plan→execute→reflect loop."""

from __future__ import annotations

import asyncio
import uuid
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

from autouser import affect
from autouser.cognitive.engine import CognitiveEngine
from autouser.cognitive.models import (
    ActionType,
    CompletionSource,
    FeedbackSubtype,
    FrictionIssue,
    IssueCategory,
    Severity,
    TerminalReason,
)
from autouser.cognitive.protocol import EngineProtocol
from autouser.execution.browser import BrowserExecutor
from autouser.perception import perceive
from autouser.persona.models import Level, Persona
from autouser.report.generator import ReportGenerator
from autouser.report.models import BatchResult, FrictionLog, PersonaError
from autouser.session import SessionLog, TaskSpec


class PerceptionMode(str, Enum):
    """How the cognitive engine perceives the page."""

    DOM = "dom"  # Default: DOM summary + visible text
    COMPUTER_USE = "computer_use"  # Screenshot-based vision


class SimulationRunner:
    """Runs a complete AutoUser simulation.

    Orchestrates the core loop:
    1. Capture UI state
    2. Cognitive model plans next action (Phase 1 — intent before execution)
    3. Execution layer performs the action
    4. Cognitive model reflects on result (Phase 2 — mismatch detection)
    5. Repeat until task complete, patience exhausted, or max steps reached
    6. Generate friction log report
    """

    def __init__(
        self,
        persona: Persona,
        task_spec: TaskSpec,
        screenshot_dir: Optional[Path] = None,
        perception_mode: PerceptionMode | str = PerceptionMode.DOM,
        engine: Optional[EngineProtocol] = None,
        fast_mode: bool = False,
    ) -> None:
        self.persona = persona
        self.task_spec = task_spec
        self.perception_mode = PerceptionMode(perception_mode)
        self._affect_persona = (
            persona.model_copy(update={"patience": task_spec.give_up_threshold})
            if task_spec.give_up_threshold is not None
            else persona
        )

        if engine is not None:
            self.engine = engine
        elif self.perception_mode == PerceptionMode.COMPUTER_USE:
            from autouser.cognitive.computer_use import ComputerUseEngine
            self.engine = ComputerUseEngine(
                persona, task_spec.task, task_spec.success_criteria,
            )
        else:
            self.engine = CognitiveEngine(
                persona, task_spec.task, task_spec.success_criteria,
                exploration=task_spec.exploration,
                fast_mode=fast_mode,
            )

        # Motor fidelity is harness-enforced: low-precision personas get
        # noisy clicks and keystroke typing with typos at the executor level.
        self.executor = BrowserExecutor(
            screenshot_dir,
            imprecise_pointer=(persona.accessibility.motor_precision == Level.LOW),
        )
        self.reporter = ReportGenerator()
        self.session = SessionLog(
            task_spec=task_spec,
            persona_snapshot=persona.model_copy(deep=True),
        )

    def _perceive(self, state):
        """Persona-filtered view for the cognitive engine (DOM mode only).

        The engine reasons over what the persona can perceive; ground-truth
        states stay raw for predicate matching, dwell detection, and replay.
        Perception is affect-modulated: a frustrated persona skims, so the
        engine-facing text shrinks as patience runs down.
        Computer-use mode perceives via screenshots and bypasses this filter.
        """
        if self.perception_mode == PerceptionMode.DOM:
            ratio = affect.frustration_ratio(self.engine.history, self._affect_persona)
            return perceive(state, self.persona, frustration_ratio=ratio)
        return state

    @staticmethod
    def _attach_screenshot(result) -> None:
        """Carry the post-action screenshot onto StepResult for reports/export."""
        if result.screenshot_path is None:
            result.screenshot_path = result.observation_after.screenshot_path

    @staticmethod
    def _has_alert_signal(state) -> bool:
        """Detect structural alert/status regions without matching error copy."""
        aria = (state.aria_snapshot or "").lower()
        dom = (state.dom_summary or "").lower()
        return "alert" in aria or "role=alert" in dom

    @staticmethod
    def _failure_signature(result, after_state):
        """Return an action-failure signature for retry/dwell detection, or None.

        This intentionally avoids English error-word matching. A repeated
        failure is a repeated post-action state on a step the cognitive layer
        classifies as error recovery, where the executor reports an action
        error, or where the page exposes an alert/status region and reflection
        marked the action surprising. Generic no-effect states are handled by
        the affect model rather than forcing an early dwell-loop exit.
        """
        error_recovery = (
            result.category == IssueCategory.ERROR_RECOVERY
            or any(
                issue.category == IssueCategory.ERROR_RECOVERY
                for issue in result.friction_issues
            )
        )
        failed_or_stalled = (
            bool(after_state.action_error)
            or error_recovery
            or (result.mismatch and SimulationRunner._has_alert_signal(after_state))
        )
        if not failed_or_stalled:
            return None
        return (
            after_state.url,
            after_state.visible_text,
            after_state.dom_summary,
            after_state.focused_element,
            after_state.action_error or "",
        )

    def _should_give_up(self) -> bool:
        """Honor custom engine give-up plus TaskSpec patience override."""
        engine_give_up = self.engine.should_give_up()
        if self.task_spec.give_up_threshold is None:
            return engine_give_up
        runner_give_up = affect.summarize(
            self.engine.history,
            self._affect_persona,
        ).should_abandon
        return engine_give_up or runner_give_up

    async def run(self) -> FrictionLog:
        """Execute the full simulation and return a friction log."""
        try:
            ui_state = await self.executor.start(self.task_spec.start_url)

            _dwell_signature: tuple[str, str, str, str | None, str] | None = None
            _dwell_count: int = 0

            for step in range(1, self.task_spec.max_steps + 1):
                # Phase 1: Plan (intent declared BEFORE action). The engine
                # sees the persona-perceived view; `ui_state` stays raw.
                observation_before = self._perceive(ui_state)
                intent = await self.engine.plan(observation_before)

                if intent.action == ActionType.GIVE_UP or self._should_give_up():
                    intent.action = ActionType.GIVE_UP
                    result = await self.engine.reflect(
                        intent, observation_before,
                        observation_before=observation_before,
                        observation_after=observation_before,
                        step_number=step,
                    )
                    self._attach_screenshot(result)
                    self.session.append_step(result)
                    self.engine.history.append(result)
                    self.session.finalize(TerminalReason.ABANDONED)
                    break

                # Execute action in browser — pass screenshot ID for
                # coordinate staleness validation in computer-use mode.
                expected_ss_id = (
                    getattr(self.engine, "current_screenshot_id", None)
                    if self.perception_mode == PerceptionMode.COMPUTER_USE
                    else None
                )
                new_ui_state = await self.executor.execute(
                    intent, step, expected_screenshot_id=expected_ss_id,
                )

                # Phase 2: Reflect (mismatch detection AFTER action) — again
                # on the perceived view; predicate/dwell checks below use the
                # raw `new_ui_state` so objective gates can't be blinded by a
                # persona's perception filter.
                observation_after = self._perceive(new_ui_state)
                result = await self.engine.reflect(
                    intent, observation_after,
                    observation_before=observation_before,
                    observation_after=observation_after,
                    step_number=step,
                )
                self._attach_screenshot(result)

                # --- Termination evaluation (exit-order logic) ---
                persona_name = (
                    self.persona.archetype.name if self.persona.archetype else "persona"
                )

                # 0. Exploration mode — there is no task to complete. The
                #    session ends naturally when the persona feels they've
                #    seen enough (belief reinterpreted by the explore reflect
                #    prompt) on a step that didn't surprise them. Predicate
                #    and LLM-soft task gates don't apply; dwell-loop and
                #    frustration/time exits below still do.
                if self.task_spec.exploration:
                    if result.persona_believes_complete and not result.mismatch:
                        result.success_criteria_met = True
                        result.completion_source = CompletionSource.EXPLORATION
                        self.session.append_step(result)
                        self.engine.history.append(result)
                        self.session.finalize(TerminalReason.SUCCESS)
                        break

                # 1. Predicate path — deterministic, highest priority
                elif self.task_spec.success_predicate is not None:
                    if self.task_spec.success_predicate.matches(new_ui_state):
                        result.success_criteria_met = True
                        result.completion_source = CompletionSource.PREDICATE
                        # Emit silent-success divergence BEFORE finalizing
                        if not result.persona_believes_complete:
                            result.friction_issues.append(FrictionIssue(
                                category=IssueCategory.FEEDBACK,
                                severity=Severity.MEDIUM,
                                step=step,
                                feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
                                description=(
                                    f"{persona_name} objectively completed the task "
                                    f"but did not recognize success."
                                ),
                            ))
                        self.session.append_step(result)
                        self.engine.history.append(result)
                        self.session.finalize(TerminalReason.SUCCESS)
                        break
                    else:
                        # Predicate set but didn't match — log false-success if persona disagrees
                        if result.persona_believes_complete:
                            result.friction_issues.append(FrictionIssue(
                                category=IssueCategory.FEEDBACK,
                                severity=Severity.HIGH,
                                step=step,
                                feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
                                description=(
                                    f"{persona_name} believed the task was complete, "
                                    f"but objective success criteria were not met."
                                ),
                            ))
                        # fall through — persona belief does NOT rescue a failed predicate

                # 2. Soft LLM gate — only when no predicate configured
                #    Uses observed_success_signal (LLM's read of page state vs criteria),
                #    NOT persona_believes_complete (subjective persona feeling).
                elif (
                    result.observed_success_signal
                    and result.intent.confidence >= 0.6
                    and not result.mismatch
                ):
                    result.success_criteria_met = True
                    result.completion_source = CompletionSource.LLM_SOFT
                    self.session.append_step(result)
                    self.engine.history.append(result)
                    self.session.finalize(TerminalReason.SUCCESS)
                    break

                self.session.append_step(result)
                self.engine.history.append(result)

                # 3. Dwell-loop detection — same failed state across steps
                if self.task_spec.max_consecutive_failures is not None:
                    sig = self._failure_signature(result, new_ui_state)
                    if sig and sig == _dwell_signature:
                        _dwell_count += 1
                    elif sig:
                        _dwell_signature = sig
                        _dwell_count = 1
                    else:
                        _dwell_signature = None
                        _dwell_count = 0

                    if _dwell_count >= self.task_spec.max_consecutive_failures:
                        result.friction_issues.append(FrictionIssue(
                            category=IssueCategory.ERROR_RECOVERY,
                            severity=Severity.HIGH,
                            step=step,
                            description=(
                                f"{persona_name} retried the same action "
                                f"{_dwell_count} times without progress and gave up."
                            ),
                        ))
                        self.session.finalize(TerminalReason.ABANDONED)
                        break

                # 4. Frustration — abandoned
                if self._should_give_up():
                    self.session.finalize(TerminalReason.ABANDONED)
                    break

                ui_state = new_ui_state
            else:
                self.session.finalize(TerminalReason.TIMED_OUT)
        except Exception:
            if not self.session.terminal_reason:
                self.session.finalize(TerminalReason.ERROR)
            raise
        finally:
            await self.executor.stop()

        return self.reporter.generate(
            steps=self.engine.history,
            persona=self.persona,
            task=self.task_spec.task,
            success_criteria=self.task_spec.success_criteria,
            terminal_reason=self.session.terminal_reason,
        )


class BatchRunner:
    """Run multiple personas against the same task concurrently.

    Each persona gets its own ``SimulationRunner`` (and therefore its own
    browser context), so N personas execute in parallel and wall-clock time
    is roughly the same as a single run.
    """

    def __init__(
        self,
        personas: Sequence[Persona],
        task_spec: TaskSpec,
        screenshot_dir: Optional[Path] = None,
        perception_mode: PerceptionMode | str = PerceptionMode.DOM,
        *,
        provider: str | None = None,
        model: str | None = None,
        fast_mode: bool = False,
        engine_factory: Callable[[Persona], EngineProtocol] | None = None,
    ) -> None:
        self.personas = list(personas)
        self.task_spec = task_spec
        self.screenshot_dir = screenshot_dir or Path("./screenshots")
        self.perception_mode = PerceptionMode(perception_mode)
        self.provider = provider
        self.model = model
        self.fast_mode = fast_mode
        self.engine_factory = engine_factory

    def _engine_for(self, persona: Persona) -> EngineProtocol | None:
        if self.engine_factory is not None:
            return self.engine_factory(persona)
        if self.provider is None and self.model is None:
            return None
        if self.perception_mode == PerceptionMode.COMPUTER_USE:
            from autouser.cognitive.computer_use import ComputerUseEngine

            return ComputerUseEngine(
                persona,
                self.task_spec.task,
                self.task_spec.success_criteria,
                model=self.model,
            )
        return CognitiveEngine(
            persona,
            self.task_spec.task,
            self.task_spec.success_criteria,
            provider=self.provider,
            model=self.model,
            exploration=self.task_spec.exploration,
            fast_mode=self.fast_mode,
        )

    async def run(self) -> BatchResult:
        """Execute all persona simulations concurrently.

        Returns a ``BatchResult`` containing successful friction logs and any
        per-persona errors.  Partial failures do **not** discard successful
        results — if Carlos crashes but Maria/Jake/Priya/Yuki succeed, you
        get 4 logs + 1 error.
        """
        run_id = uuid.uuid4().hex[:12]

        coros = []
        for persona in self.personas:
            persona_name = persona.archetype.name if persona.archetype else "custom"
            persona_screenshot_dir = self.screenshot_dir / run_id / persona_name
            runner = SimulationRunner(
                persona=persona,
                task_spec=self.task_spec,
                screenshot_dir=persona_screenshot_dir,
                perception_mode=self.perception_mode,
                engine=self._engine_for(persona),
                fast_mode=self.fast_mode,
            )
            coros.append(runner.run())

        results = await asyncio.gather(*coros, return_exceptions=True)

        logs: list[FrictionLog] = []
        errors: list[PersonaError] = []
        for i, result in enumerate(results):
            persona_name = (
                self.personas[i].archetype.name
                if self.personas[i].archetype
                else "custom"
            )
            if isinstance(result, BaseException):
                errors.append(
                    PersonaError(
                        persona_name=persona_name,
                        error_type=type(result).__name__,
                        message=str(result),
                    )
                )
            else:
                logs.append(result)

        return BatchResult(logs=logs, errors=errors)
