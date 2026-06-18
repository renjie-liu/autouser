"""Tests for termination detection, success predicates, and divergence findings.

Covers: exit-order logic, predicate evaluation, soft-gate guards,
divergence friction emissions, report outcome derivation, and
completion_source provenance.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CompletionSource,
    Emotion,
    FeedbackSubtype,
    FrictionIssue,
    IssueCategory,
    Severity,
    StepResult,
    SuccessPredicate,
    TerminalReason,
    UIState,
)
from autouser.report.models import TaskOutcome, outcome_from_terminal_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ui_state(
    url: str = "https://example.com",
    dom_summary: str = 'button#submit "Submit"',
    visible_text: str = "Welcome",
) -> UIState:
    return UIState(
        url=url,
        page_title="Test",
        dom_summary=dom_summary,
        visible_text=visible_text,
    )


def _make_step_result(
    step: int = 1,
    persona_believes_complete: bool = False,
    observed_success_signal: bool = False,
    confidence: float = 0.7,
    mismatch: bool = False,
    emotion: Emotion = Emotion.CONFIDENT,
    severity: Severity | None = None,
    category: IssueCategory | None = None,
) -> StepResult:
    # before/after deliberately DIFFER: this helper models ordinary steps
    # whose actions visibly did something. Identical states now feed the
    # affect mechanism as objective "no_effect" frustration (slice 3) —
    # tests that want that signal must construct it explicitly.
    return StepResult(
        step=step,
        intent=ActionIntent(
            action=ActionType.CLICK,
            target="button#submit",
            persona_thought="Clicking submit.",
            expected_outcome="Form submits.",
            confidence=confidence,
        ),
        observation_before=_make_ui_state(),
        observation_after=_make_ui_state(visible_text=f"Result after step {step}"),
        actual_outcome="Form submitted.",
        mismatch=mismatch,
        reflection="Looks good.",
        emotion=emotion,
        severity=severity,
        category=category,
        persona_believes_complete=persona_believes_complete,
        observed_success_signal=observed_success_signal,
    )


# ===========================================================================
# SuccessPredicate tests
# ===========================================================================

class TestSuccessPredicate:
    def test_empty_predicate_raises(self):
        with pytest.raises(ValidationError):
            SuccessPredicate()

    def test_url_pattern_matches(self):
        pred = SuccessPredicate(url_pattern=r"/checkout-complete")
        state = _make_ui_state(url="https://shop.com/checkout-complete")
        assert pred.matches(state)

    def test_url_pattern_no_match(self):
        pred = SuccessPredicate(url_pattern=r"/checkout-complete")
        state = _make_ui_state(url="https://shop.com/cart")
        assert not pred.matches(state)

    def test_selector_present_matches(self):
        pred = SuccessPredicate(selector_present="button#submit")
        state = _make_ui_state(dom_summary='button#submit "Submit"')
        assert pred.matches(state)

    def test_selector_present_no_match(self):
        pred = SuccessPredicate(selector_present="button#checkout")
        state = _make_ui_state(dom_summary='button#submit "Submit"')
        assert not pred.matches(state)

    def test_text_contains_matches(self):
        pred = SuccessPredicate(text_contains="Order confirmed")
        state = _make_ui_state(visible_text="Thank you! Order confirmed.")
        assert pred.matches(state)

    def test_conjunctive_all_must_match(self):
        """When multiple fields set, ALL must match (AND semantics)."""
        pred = SuccessPredicate(
            url_pattern=r"/success",
            text_contains="Order confirmed",
        )
        state_both = _make_ui_state(url="https://shop.com/success", visible_text="Order confirmed")
        state_url_only = _make_ui_state(url="https://shop.com/success", visible_text="Error")
        assert pred.matches(state_both)
        assert not pred.matches(state_url_only)

    def test_none_fields_skipped(self):
        """Unset fields are ignored — only configured fields participate."""
        pred = SuccessPredicate(url_pattern=r"/done", selector_present=None)
        state = _make_ui_state(url="https://example.com/done")
        assert pred.matches(state)


# ===========================================================================
# TerminalReason → TaskOutcome mapping
# ===========================================================================

class TestOutcomeMapping:
    def test_success_no_friction_maps_to_completed(self):
        assert outcome_from_terminal_reason(TerminalReason.SUCCESS) == TaskOutcome.COMPLETED

    def test_success_with_high_friction_promotes(self):
        assert (
            outcome_from_terminal_reason(TerminalReason.SUCCESS, has_high_severity_friction=True)
            == TaskOutcome.COMPLETED_WITH_ERRORS
        )

    def test_abandoned_maps_directly(self):
        assert outcome_from_terminal_reason(TerminalReason.ABANDONED) == TaskOutcome.ABANDONED

    def test_timed_out_maps_directly(self):
        assert outcome_from_terminal_reason(TerminalReason.TIMED_OUT) == TaskOutcome.TIMED_OUT

    def test_error_maps_directly(self):
        assert outcome_from_terminal_reason(TerminalReason.ERROR) == TaskOutcome.ERROR

    def test_mapping_is_total(self):
        """Every TerminalReason variant maps to exactly one TaskOutcome."""
        for reason in TerminalReason:
            result = outcome_from_terminal_reason(reason)
            assert isinstance(result, TaskOutcome)

    def test_abandoned_not_promoted_by_high_friction(self):
        """HIGH severity does NOT promote non-SUCCESS terminal reasons."""
        assert (
            outcome_from_terminal_reason(TerminalReason.ABANDONED, has_high_severity_friction=True)
            == TaskOutcome.ABANDONED
        )

    def test_non_high_severity_does_not_promote(self):
        """MEDIUM/LOW friction on SUCCESS stays COMPLETED."""
        assert (
            outcome_from_terminal_reason(TerminalReason.SUCCESS, has_high_severity_friction=False)
            == TaskOutcome.COMPLETED
        )


# ===========================================================================
# Report outcome promotion rule
# ===========================================================================

class TestReportOutcomePromotion:
    def test_false_success_high_promotes_to_completed_with_errors(self):
        """TerminalReason.SUCCESS + HIGH severity false_success → COMPLETED_WITH_ERRORS."""
        from autouser.report.generator import ReportGenerator
        gen = ReportGenerator()
        step = _make_step_result(step=1)
        step.friction_issues.append(FrictionIssue(
            category=IssueCategory.FEEDBACK,
            severity=Severity.HIGH,
            step=1,
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
            description="test",
        ))
        outcome = gen._determine_outcome([step], TerminalReason.SUCCESS)
        assert outcome == TaskOutcome.COMPLETED_WITH_ERRORS

    def test_silent_success_medium_stays_completed(self):
        """TerminalReason.SUCCESS + MEDIUM severity silent_success → COMPLETED."""
        from autouser.report.generator import ReportGenerator
        gen = ReportGenerator()
        step = _make_step_result(step=1)
        step.friction_issues.append(FrictionIssue(
            category=IssueCategory.FEEDBACK,
            severity=Severity.MEDIUM,
            step=1,
            feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
            description="test",
        ))
        outcome = gen._determine_outcome([step], TerminalReason.SUCCESS)
        assert outcome == TaskOutcome.COMPLETED

    def test_any_high_severity_promotes_not_just_divergence(self):
        """An ordinary labeling issue at HIGH severity also promotes."""
        from autouser.report.generator import ReportGenerator
        gen = ReportGenerator()
        step = _make_step_result(step=1, mismatch=True, severity=Severity.HIGH, category=IssueCategory.LABELING)
        outcome = gen._determine_outcome([step], TerminalReason.SUCCESS)
        assert outcome == TaskOutcome.COMPLETED_WITH_ERRORS


# ===========================================================================
# Finding subtype aggregation (no string matching)
# ===========================================================================

class TestFindingSubtypeAggregation:
    def test_report_counts_by_subtype_not_description(self):
        """ReportGenerator aggregates by feedback_subtype enum, not description text."""
        from autouser.report.generator import ReportGenerator
        gen = ReportGenerator()
        step = _make_step_result(step=1)
        step.friction_issues.extend([
            FrictionIssue(
                category=IssueCategory.FEEDBACK,
                severity=Severity.HIGH,
                step=1,
                feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
                description="Unrelated description that does not mention any enum value.",
            ),
            FrictionIssue(
                category=IssueCategory.FEEDBACK,
                severity=Severity.MEDIUM,
                step=1,
                feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
                description="Another unrelated description.",
            ),
            FrictionIssue(
                category=IssueCategory.FEEDBACK,
                severity=Severity.HIGH,
                step=1,
                feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
                description="Yet another.",
            ),
        ])
        counts = gen._count_findings_by_subtype([step])
        assert counts[FeedbackSubtype.FALSE_SUCCESS] == 2
        assert counts[FeedbackSubtype.SILENT_SUCCESS] == 1


# ===========================================================================
# StepResult field defaults
# ===========================================================================

class TestStepResultDefaults:
    def test_new_fields_have_safe_defaults(self):
        step = _make_step_result()
        assert step.persona_believes_complete is False
        assert step.success_criteria_met is False
        assert step.completion_source == CompletionSource.NONE
        assert step.friction_issues == []

    def test_completion_source_set_on_predicate_exit(self):
        step = _make_step_result()
        step.completion_source = CompletionSource.PREDICATE
        step.success_criteria_met = True
        assert step.completion_source == CompletionSource.PREDICATE
        assert step.success_criteria_met is True

    def test_completion_source_set_on_llm_soft_exit(self):
        step = _make_step_result(persona_believes_complete=True, confidence=0.8)
        step.completion_source = CompletionSource.LLM_SOFT
        step.success_criteria_met = True
        assert step.completion_source == CompletionSource.LLM_SOFT


# ===========================================================================
# Integration: SimulationRunner exit-order logic (mocked)
# ===========================================================================

class TestRunnerTermination:
    """Tests for SimulationRunner.run() exit-order logic.

    These mock the engine and executor to verify the runner's control flow
    without hitting real LLM APIs or browsers.
    """

    @pytest.fixture
    def task_spec_with_predicate(self):
        from autouser.session import TaskSpec
        return TaskSpec(
            task="Complete checkout",
            success_criteria="Reach order confirmation page",
            start_url="https://shop.com",
            max_steps=10,
            success_predicate=SuccessPredicate(url_pattern=r"/order-confirmed"),
        )

    @pytest.fixture
    def task_spec_no_predicate(self):
        from autouser.session import TaskSpec
        return TaskSpec(
            task="Complete checkout",
            success_criteria="Reach order confirmation page",
            start_url="https://shop.com",
            max_steps=10,
        )

    @pytest.fixture
    def persona(self):
        from autouser.persona.models import Persona
        from autouser.persona.registry import get_archetype
        return Persona.from_archetype(get_archetype("maria"))

    @pytest.mark.asyncio
    async def test_predicate_match_stops_at_step_n(self, task_spec_with_predicate, persona):
        """Predicate fires at step 3 → runner stops, completion_source=PREDICATE."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        runner = SimulationRunner(persona, task_spec_with_predicate)

        step_count = 0
        async def mock_start(url):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_execute(intent, step, **kwargs):
            nonlocal step_count
            step_count += 1
            if step_count >= 3:
                return _make_ui_state(url="https://shop.com/order-confirmed")
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#next",
                persona_thought="Moving forward.",
                expected_outcome="Next page.",
                confidence=0.7,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(step=kwargs.get("step_number", 1))

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert report.total_steps == 3
        assert runner.session.terminal_reason == TerminalReason.SUCCESS
        last_step = runner.session.steps[-1]
        assert last_step.completion_source == CompletionSource.PREDICATE
        assert last_step.success_criteria_met is True
        # All prior steps should have NONE
        for s in runner.session.steps[:-1]:
            assert s.completion_source == CompletionSource.NONE

    @pytest.mark.asyncio
    async def test_predicate_never_matches_runs_to_timeout(self, task_spec_with_predicate, persona):
        """Predicate set but never matches → TIMED_OUT at max_steps."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        task_spec_with_predicate.max_steps = 5
        runner = SimulationRunner(persona, task_spec_with_predicate)

        async def mock_start(url):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#next",
                persona_thought="Trying again.",
                expected_outcome="Maybe this time.",
                confidence=0.5,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(step=kwargs.get("step_number", 1))

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert report.total_steps == 5
        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        for s in runner.session.steps:
            assert s.completion_source == CompletionSource.NONE

    @pytest.mark.asyncio
    async def test_llm_soft_exits_when_no_predicate(self, task_spec_no_predicate, persona):
        """No predicate + observed_success_signal=True + confidence≥0.6 + no mismatch → SUCCESS via LLM_SOFT."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        runner = SimulationRunner(persona, task_spec_no_predicate)

        call_count = 0
        async def mock_start(url):
            return _make_ui_state()

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state()

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#submit",
                persona_thought="This looks right.",
                expected_outcome="Done.",
                confidence=0.8,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return _make_step_result(
                    step=kwargs.get("step_number", 1),
                    observed_success_signal=True,
                    confidence=0.8,
                )
            return _make_step_result(step=kwargs.get("step_number", 1))

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        await runner.run()

        assert runner.session.terminal_reason == TerminalReason.SUCCESS
        last_step = runner.session.steps[-1]
        assert last_step.completion_source == CompletionSource.LLM_SOFT
        assert last_step.success_criteria_met is True

    @pytest.mark.asyncio
    async def test_llm_soft_rejects_low_confidence(self, task_spec_no_predicate, persona):
        """No predicate + observed_success_signal=True but confidence=0.4 → loop continues."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        task_spec_no_predicate.max_steps = 3
        runner = SimulationRunner(persona, task_spec_no_predicate)

        async def mock_start(url):
            return _make_ui_state()

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state()

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#submit",
                persona_thought="Not sure.",
                expected_outcome="Maybe.",
                confidence=0.4,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(
                step=kwargs.get("step_number", 1),
                observed_success_signal=True,
                confidence=0.4,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        assert report.total_steps == 3
        for s in runner.session.steps:
            assert s.completion_source == CompletionSource.NONE

    @pytest.mark.asyncio
    async def test_llm_soft_rejects_on_mismatch(self, task_spec_no_predicate, persona):
        """No predicate + observed_success_signal=True + confidence=0.9 but mismatch=True → continues."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        task_spec_no_predicate.max_steps = 3
        runner = SimulationRunner(persona, task_spec_no_predicate)

        async def mock_start(url):
            return _make_ui_state()

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state()

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#submit",
                persona_thought="Hmm.",
                expected_outcome="Should work.",
                confidence=0.9,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(
                step=kwargs.get("step_number", 1),
                observed_success_signal=True,
                confidence=0.9,
                mismatch=True,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        await runner.run()

        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        for s in runner.session.steps:
            assert s.completion_source == CompletionSource.NONE

    @pytest.mark.asyncio
    async def test_false_success_continues_and_logs(self, task_spec_with_predicate, persona):
        """Predicate miss + persona_believes=True → loop continues, FALSE_SUCCESS logged."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        task_spec_with_predicate.max_steps = 3
        runner = SimulationRunner(persona, task_spec_with_predicate)

        async def mock_start(url):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#next",
                persona_thought="I think I'm done.",
                expected_outcome="Checkout complete.",
                confidence=0.8,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(
                step=kwargs.get("step_number", 1),
                persona_believes_complete=True,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT
        assert report.total_steps == 3
        # Every step should have a FALSE_SUCCESS friction issue
        for s in runner.session.steps:
            false_successes = [
                fi for fi in s.friction_issues
                if fi.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS
            ]
            assert len(false_successes) == 1
            assert false_successes[0].severity == Severity.HIGH
            assert false_successes[0].category == IssueCategory.FEEDBACK

    @pytest.mark.asyncio
    async def test_silent_success_exits_and_logs(self, task_spec_with_predicate, persona):
        """Predicate match + persona_believes=False → exit SUCCESS, SILENT_SUCCESS logged."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        runner = SimulationRunner(persona, task_spec_with_predicate)

        async def mock_start(url):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state(url="https://shop.com/order-confirmed")

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#pay",
                persona_thought="Let me try this.",
                expected_outcome="Maybe checkout.",
                confidence=0.5,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(
                step=kwargs.get("step_number", 1),
                persona_believes_complete=False,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert runner.session.terminal_reason == TerminalReason.SUCCESS
        assert report.total_steps == 1
        last_step = runner.session.steps[-1]
        assert last_step.completion_source == CompletionSource.PREDICATE
        assert last_step.success_criteria_met is True
        silent = [
            fi for fi in last_step.friction_issues
            if fi.feedback_subtype == FeedbackSubtype.SILENT_SUCCESS
        ]
        assert len(silent) == 1
        assert silent[0].severity == Severity.MEDIUM
        assert silent[0].category == IssueCategory.FEEDBACK

    @pytest.mark.asyncio
    async def test_abandoned_on_frustration(self, task_spec_no_predicate, persona):
        """Frustration exceeds patience → ABANDONED before max_steps."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        task_spec_no_predicate.max_steps = 20
        runner = SimulationRunner(persona, task_spec_no_predicate)

        async def mock_start(url):
            return _make_ui_state()

        async def mock_execute(intent, step, **kwargs):
            return _make_ui_state()

        step_counter = 0

        async def mock_plan(ui_state):
            nonlocal step_counter
            step_counter += 1
            if step_counter > persona.patience:
                return ActionIntent(
                    action=ActionType.GIVE_UP,
                    target="",
                    persona_thought="I can't do this anymore.",
                    expected_outcome="Give up.",
                    confidence=0.1,
                )
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#try",
                persona_thought="This is confusing.",
                expected_outcome="Something happens.",
                confidence=0.3,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(
                step=kwargs.get("step_number", 1),
                mismatch=True,
                emotion=Emotion.FRUSTRATED,
                severity=Severity.MEDIUM,
                category=IssueCategory.NAVIGATION,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        report = await runner.run()

        assert runner.session.terminal_reason == TerminalReason.ABANDONED
        assert report.total_steps < 20

    @pytest.mark.asyncio
    async def test_executor_exception_yields_error(self, task_spec_no_predicate, persona):
        """Executor crash → ERROR terminal reason."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        runner = SimulationRunner(persona, task_spec_no_predicate)

        async def mock_start(url):
            return _make_ui_state()

        async def mock_execute(intent, step, **kwargs):
            raise RuntimeError("Browser crashed")

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#go",
                persona_thought="Let's go.",
                expected_outcome="Page loads.",
                confidence=0.7,
            )

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan

        with pytest.raises(RuntimeError, match="Browser crashed"):
            await runner.run()

        assert runner.session.terminal_reason == TerminalReason.ERROR

    @pytest.mark.asyncio
    async def test_no_post_success_steps(self, task_spec_with_predicate, persona):
        """Invariant: success_criteria_met=True at step N implies len(steps)==N."""
        from unittest.mock import AsyncMock
        from autouser.runner import SimulationRunner

        runner = SimulationRunner(persona, task_spec_with_predicate)

        call_count = 0
        async def mock_start(url):
            return _make_ui_state(url="https://shop.com/cart")

        async def mock_execute(intent, step, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                return _make_ui_state(url="https://shop.com/order-confirmed")
            return _make_ui_state(url="https://shop.com/step" + str(call_count))

        async def mock_plan(ui_state):
            return ActionIntent(
                action=ActionType.CLICK,
                target="button#next",
                persona_thought="Next step.",
                expected_outcome="Progress.",
                confidence=0.7,
            )

        async def mock_reflect(intent, new_ui_state, **kwargs):
            return _make_step_result(step=kwargs.get("step_number", 1))

        runner.executor.start = mock_start
        runner.executor.execute = mock_execute
        runner.executor.stop = AsyncMock()
        runner.engine.plan = mock_plan
        runner.engine.reflect = mock_reflect

        await runner.run()

        met_step = None
        for s in runner.session.steps:
            if s.success_criteria_met:
                met_step = s.step
                break
        assert met_step is not None
        assert len(runner.session.steps) == met_step
