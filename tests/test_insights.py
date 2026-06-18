"""Insights pipeline: mental notes → FrictionLog.insights → report section.

Motivated by the priya/friction-fixture live run (2026-06-12): 0 issues
extracted (nothing mismatched) yet her note 1 was the planted Pattern-3
finding verbatim. Mismatch-only extraction misses traps the persona AVOIDED;
the discovery cost is friction too.
"""

from __future__ import annotations

import pytest

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    FrictionIssue,
    IssueCategory,
    Severity,
    StepResult,
    TerminalReason,
    UIState,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.report.generator import ReportGenerator
from autouser.report.human_renderer import render_report
from autouser.report.models import FrictionLog, TaskOutcome


@pytest.fixture
def priya() -> Persona:
    return Persona.from_archetype(get_archetype("priya"))


def _step(n: int, note: str | None = None) -> StepResult:
    state = UIState(url="https://x.test/", page_title="T", dom_summary="d", visible_text="v")
    return StepResult(
        step=n,
        intent=ActionIntent(
            action=ActionType.CLICK, target="#a",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        ),
        observation_before=state,
        observation_after=state,
        actual_outcome="ok",
        mismatch=False,
        reflection="fine",
        emotion=Emotion.CONFIDENT,
        mental_note=note,
    )


class TestGeneratorPopulatesInsights:
    def test_notes_become_insights_in_step_order(self, priya):
        steps = [
            _step(1, "Field requires E.164 despite the placeholder."),
            _step(2),
            _step(3, "Validation only fires on submit."),
        ]
        log = ReportGenerator().generate(
            steps=steps, persona=priya, task="t", success_criteria="c",
            terminal_reason=TerminalReason.SUCCESS,
        )
        assert [(i.step, i.text) for i in log.insights] == [
            (1, "Field requires E.164 despite the placeholder."),
            (3, "Validation only fires on submit."),
        ]
        # The motivating case: insights exist even when zero issues extracted.
        assert log.issues == []

    def test_no_notes_means_no_insights(self, priya):
        log = ReportGenerator().generate(
            steps=[_step(1)], persona=priya, task="t", success_criteria="c",
            terminal_reason=TerminalReason.SUCCESS,
        )
        assert log.insights == []

    def test_overall_impressions_are_generated(self, priya):
        log = ReportGenerator().generate(
            steps=[_step(1)], persona=priya, task="t", success_criteria="c",
            terminal_reason=TerminalReason.SUCCESS,
        )
        assert log.overall_impressions
        assert "completed the task" in log.overall_impressions

    def test_issue_screenshot_falls_back_to_observation_after(self, priya):
        step = _step(1)
        step.observation_after.screenshot_path = "/tmp/after.png"
        step.friction_issues.append(
            FrictionIssue(
                category=IssueCategory.FEEDBACK,
                severity=Severity.MEDIUM,
                step=1,
                description="No confirmation appeared.",
            )
        )

        log = ReportGenerator().generate(
            steps=[step], persona=priya, task="t", success_criteria="c",
            terminal_reason=TerminalReason.SUCCESS,
        )

        assert log.issues[0].screenshot_path == "/tmp/after.png"


class TestReportRendersInsights:
    def _log(self, insights_steps) -> FrictionLog:
        steps = [_step(n, text) for n, text in insights_steps]
        return ReportGenerator().generate(
            steps=steps,
            persona=Persona.from_archetype(get_archetype("priya")),
            task="Submit the form",
            success_criteria="Form saved",
            terminal_reason=TerminalReason.SUCCESS,
        )

    def test_insights_section_rendered(self, priya):
        log = self._log([(1, "The placeholder lies about the format.")])
        report = render_report(log, priya)
        assert "What priya had to figure out (1)" in report
        assert "1. (step 1) The placeholder lies about the format." in report

    def test_no_section_without_insights(self, priya):
        log = self._log([(1, None)])
        report = render_report(log, priya)
        assert "had to figure out" not in report
        assert log.outcome == TaskOutcome.COMPLETED

    def test_pipe_and_newline_sanitized(self, priya):
        log = self._log([(1, "weird | note\nwith newline")])
        report = render_report(log, priya)
        assert "weird | note with newline" in report
