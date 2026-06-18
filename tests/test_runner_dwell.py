"""Tests for runner dwell-loop detection (max_consecutive_failures)."""

import pytest
from unittest.mock import AsyncMock, patch

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    IssueCategory,
    Severity,
    StepResult,
    TerminalReason,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.runner import SimulationRunner
from autouser.session import TaskSpec


def _make_ui_state(url: str = "https://example.com/login", error_text: str = "") -> UIState:
    return UIState(
        url=url,
        page_title="Login",
        dom_summary="button#login-btn text='Login'",
        visible_text=f"Username Password {error_text}",
    )


def _make_intent(action: ActionType = ActionType.CLICK, target: str = "button#login-btn") -> ActionIntent:
    return ActionIntent(
        action=action,
        target=target,
        persona_thought="I'll try clicking login",
        expected_outcome="I should log in",
        confidence=0.8,
    )


def _make_step_result(
    step: int,
    intent: ActionIntent,
    ui_before: UIState,
    ui_after: UIState,
) -> StepResult:
    return StepResult(
        step=step,
        intent=intent,
        observation_before=ui_before,
        observation_after=ui_after,
        actual_outcome="Login failed",
        mismatch=True,
        reflection="That didn't work",
        emotion=Emotion.CONFUSED,
        severity=Severity.MEDIUM,
        category=IssueCategory.ERROR_RECOVERY,
    )


@pytest.fixture
def maria_persona() -> Persona:
    return Persona.from_archetype(get_archetype("maria"))


@pytest.fixture
def task_spec_with_retry_budget() -> TaskSpec:
    return TaskSpec(
        task="Log in with username 'locked_out_user' and password 'secret_sauce'.",
        success_criteria="You see the product inventory page.",
        start_url="https://www.saucedemo.com/",
        max_steps=15,
        max_consecutive_failures=3,
    )


@pytest.fixture
def task_spec_no_retry_budget() -> TaskSpec:
    return TaskSpec(
        task="Log in with username 'locked_out_user' and password 'secret_sauce'.",
        success_criteria="You see the product inventory page.",
        start_url="https://www.saucedemo.com/",
        max_steps=15,
        max_consecutive_failures=None,
    )


class TestDwellLoopDetection:
    """Verify the runner terminates after N consecutive identical failures."""

    @pytest.mark.asyncio
    async def test_abandons_after_consecutive_failures(self, maria_persona, task_spec_with_retry_budget):
        error_ui = _make_ui_state(error_text="Sorry, this user has been locked out.")
        intent = _make_intent()

        runner = SimulationRunner(persona=maria_persona, task_spec=task_spec_with_retry_budget)

        with patch.object(runner.executor, "start", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, return_value=intent), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock, side_effect=lambda *a, **kw: _make_step_result(
                 kw.get("step_number", 1), intent, error_ui, error_ui
             )), \
             patch.object(runner.engine, "should_give_up", return_value=False):

            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.ABANDONED
        assert len(runner.session.steps) == 3

    @pytest.mark.asyncio
    async def test_emits_friction_finding_on_dwell_exit(self, maria_persona, task_spec_with_retry_budget):
        error_ui = _make_ui_state(error_text="Sorry, this user has been locked out.")
        intent = _make_intent()

        runner = SimulationRunner(persona=maria_persona, task_spec=task_spec_with_retry_budget)

        with patch.object(runner.executor, "start", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, return_value=intent), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock, side_effect=lambda *a, **kw: _make_step_result(
                 kw.get("step_number", 1), intent, error_ui, error_ui
             )), \
             patch.object(runner.engine, "should_give_up", return_value=False):

            await runner.run()

        last_step = runner.session.steps[-1]
        dwell_findings = [
            fi for fi in last_step.friction_issues
            if fi.category == IssueCategory.ERROR_RECOVERY
            and "retried the same action" in fi.description
        ]
        assert len(dwell_findings) == 1
        assert dwell_findings[0].severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_counter_resets_on_different_error(self, maria_persona, task_spec_with_retry_budget):
        """Counter resets when the error text changes, even with same actions."""
        intent = _make_intent()
        step_counter = 0

        def alternating_execute(*args, **kwargs):
            nonlocal step_counter
            step_counter += 1
            if step_counter % 3 == 0:
                return _make_ui_state(error_text="Invalid password.")
            return _make_ui_state(error_text="Sorry, this user has been locked out.")

        runner = SimulationRunner(persona=maria_persona, task_spec=task_spec_with_retry_budget)

        with patch.object(runner.executor, "start", new_callable=AsyncMock,
                          return_value=_make_ui_state(error_text="Sorry, this user has been locked out.")), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, side_effect=alternating_execute), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, return_value=intent), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock,
                          side_effect=lambda *a, **kw: _make_step_result(
                              kw.get("step_number", 1), intent,
                              _make_ui_state(error_text="Sorry"), _make_ui_state(error_text="Sorry"))), \
             patch.object(runner.engine, "should_give_up", return_value=False):

            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        assert len(runner.session.steps) == 15

    @pytest.mark.asyncio
    async def test_different_actions_same_error_triggers(self, maria_persona, task_spec_with_retry_budget):
        """Multi-step retry cycle with varying actions but same error still triggers."""
        error_ui = _make_ui_state(error_text="Sorry, this user has been locked out.")
        call_count = 0

        targets = ["input#username", "input#password", "button#login-btn"]

        def cycling_plan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            t = targets[call_count % len(targets)]
            action = ActionType.TYPE if "input" in t else ActionType.CLICK
            return ActionIntent(
                action=action, target=t,
                persona_thought="Trying again.", expected_outcome="Login.",
                confidence=0.5, input_value="locked_out_user" if "input" in t else None,
            )

        runner = SimulationRunner(persona=maria_persona, task_spec=task_spec_with_retry_budget)

        with patch.object(runner.executor, "start", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, side_effect=cycling_plan), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock, side_effect=lambda *a, **kw: _make_step_result(
                 kw.get("step_number", 1), _make_intent(), error_ui, error_ui
             )), \
             patch.object(runner.engine, "should_give_up", return_value=False):
            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.ABANDONED
        assert len(runner.session.steps) == 3

    @pytest.mark.asyncio
    async def test_disabled_when_none(self, maria_persona, task_spec_no_retry_budget):
        error_ui = _make_ui_state(error_text="Sorry, this user has been locked out.")
        intent = _make_intent()

        runner = SimulationRunner(persona=maria_persona, task_spec=task_spec_no_retry_budget)

        with patch.object(runner.executor, "start", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, return_value=error_ui), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, return_value=intent), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock, side_effect=lambda *a, **kw: _make_step_result(
                 kw.get("step_number", 1), intent, error_ui, error_ui
             )), \
             patch.object(runner.engine, "should_give_up", return_value=False):

            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        assert len(runner.session.steps) == 15

    @pytest.mark.asyncio
    async def test_give_up_threshold_overrides_persona_patience(self, maria_persona):
        """TaskSpec.give_up_threshold lowers the affect patience budget."""
        initial_ui = _make_ui_state(error_text="")
        unchanged_ui = _make_ui_state(error_text="")
        intent = _make_intent()
        task = TaskSpec(
            task="Click an unresponsive login button.",
            success_criteria="Dashboard appears.",
            start_url="https://example.com/login",
            max_steps=15,
            give_up_threshold=1,
            max_consecutive_failures=None,
        )
        runner = SimulationRunner(persona=maria_persona, task_spec=task)

        with patch.object(runner.executor, "start", new_callable=AsyncMock, return_value=initial_ui), \
             patch.object(runner.executor, "execute", new_callable=AsyncMock, return_value=unchanged_ui), \
             patch.object(runner.executor, "stop", new_callable=AsyncMock), \
             patch.object(runner.engine, "plan", new_callable=AsyncMock, return_value=intent), \
             patch.object(runner.engine, "reflect", new_callable=AsyncMock, side_effect=lambda *a, **kw: _make_step_result(
                 kw.get("step_number", 1), intent, initial_ui, unchanged_ui
             )), \
             patch.object(runner.engine, "should_give_up", return_value=False):

            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.ABANDONED
        assert len(runner.session.steps) == 1
