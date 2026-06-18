"""Report generator: synthesizes step results into a structured friction log."""

from __future__ import annotations

from autouser.cognitive.models import (
    FeedbackSubtype,
    Severity,
    StepResult,
    TerminalReason,
)
from autouser.persona.models import Persona
from autouser.report.models import (
    FrictionLog,
    Insight,
    Issue,
    TaskOutcome,
    outcome_from_terminal_reason,
)


class ReportGenerator:
    """Transforms raw step results into an actionable friction log.

    Responsibilities:
    - Determine task outcome from TerminalReason enum + friction severity
    - Extract and deduplicate issues from step results
    - Aggregate divergence findings by FeedbackSubtype
    - Generate overall impressions summary
    - Rank issues by severity
    """

    def generate(
        self,
        steps: list[StepResult],
        persona: Persona,
        task: str,
        success_criteria: str,
        terminal_reason: TerminalReason | None = None,
    ) -> FrictionLog:
        """Generate a complete friction log from simulation results."""
        outcome = self._determine_outcome(steps, terminal_reason)
        issues = self._extract_issues(steps)

        impressions = self._overall_impressions(steps, outcome, issues, persona)

        return FrictionLog(
            task=task,
            success_criteria=success_criteria,
            persona_name=persona.archetype.name if persona.archetype else "custom",
            persona_summary=self._summarize_persona(persona),
            outcome=outcome,
            total_steps=len(steps),
            steps=steps,
            issues=sorted(
                issues,
                key=lambda i: {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}[i.severity],
            ),
            # Insights: the persona's mental notes promoted to findings-grade
            # signal. A note can capture a planted issue the persona AVOIDED
            # (read the format hint, sidestepped the trap) — mismatch-only
            # extraction misses those runs entirely (priya/friction-fixture,
            # 2026-06-12: 0 issues, but note 1 was the Pattern-3 finding).
            insights=[
                Insight(step=s.step, text=s.mental_note)
                for s in steps
                if s.mental_note
            ],
            overall_impressions=impressions,
        )

    def _determine_outcome(
        self, steps: list[StepResult], terminal_reason: TerminalReason | None = None,
    ) -> TaskOutcome:
        if not steps:
            return TaskOutcome.ABANDONED
        if terminal_reason is None:
            return TaskOutcome.ABANDONED

        all_friction = [fi for s in steps for fi in s.friction_issues]
        has_high_severity = any(fi.severity == Severity.HIGH for fi in all_friction)
        has_high_mismatch = any(s.mismatch and s.severity == Severity.HIGH for s in steps)

        return outcome_from_terminal_reason(
            terminal_reason,
            has_high_severity_friction=has_high_severity or has_high_mismatch,
        )

    def _count_findings_by_subtype(self, steps: list[StepResult]) -> dict[FeedbackSubtype, int]:
        counts = {subtype: 0 for subtype in FeedbackSubtype}
        for step in steps:
            for issue in step.friction_issues:
                if issue.feedback_subtype is not None:
                    counts[issue.feedback_subtype] += 1
        return counts

    def _extract_issues(self, steps: list[StepResult]) -> list[Issue]:
        # Both extraction paths below can fire on the same step. Per
        # @distinguished-eng's PR #37 architecture observation: this is
        # intentional — agent-reflection findings (Path 1, derived from
        # step.reflection + step.severity) and runner-detection findings
        # (Path 2, structured FrictionIssue emissions from runner.py)
        # are distinct surfaces and both should reach the report. They
        # are NOT deduped on (category, step) because the descriptions
        # come from different sources and a future-reader who collapses
        # them would lose signal. If a true dedup ever becomes desirable,
        # do it at producer (one source of truth per step) not consumer.
        issues = []
        for step in steps:
            screenshot_path = step.screenshot_path or step.observation_after.screenshot_path
            # Step-level mismatch path (reflection-derived findings).
            if step.mismatch and step.severity and step.category:
                issues.append(
                    Issue(
                        severity=step.severity,
                        category=step.category,
                        description=step.reflection,
                        step=step.step,
                        url=step.observation_after.url,
                        persona_perspective=step.intent.persona_thought,
                        screenshot_path=screenshot_path,
                    )
                )
            # Runner-emitted friction findings (dwell-loop, FALSE_SUCCESS,
            # SILENT_SUCCESS). These live on StepResult.friction_issues and
            # carry the divergence the harness is built to surface — must
            # be promoted to FrictionLog.issues or the report's top-friction
            # section renders empty on the runs that matter most. Caught by
            # an earlier draft review of PR #37.
            for fi in step.friction_issues:
                issues.append(
                    Issue(
                        severity=fi.severity,
                        category=fi.category,
                        description=fi.description,
                        step=fi.step,
                        url=step.observation_after.url,
                        persona_perspective=step.intent.persona_thought,
                        screenshot_path=screenshot_path,
                        feedback_subtype=fi.feedback_subtype,
                    )
                )
        return issues

    def _overall_impressions(
        self,
        steps: list[StepResult],
        outcome: TaskOutcome,
        issues: list[Issue],
        persona: Persona,
    ) -> str:
        name = persona.archetype.name if persona.archetype else "The persona"
        if not steps:
            return f"{name} did not get far enough to form a clear impression."

        step_count = len(steps)
        first_emotion = steps[0].emotion.value
        last_emotion = steps[-1].emotion.value
        mismatch_count = sum(1 for step in steps if step.mismatch)
        high_count = sum(1 for issue in issues if issue.severity == Severity.HIGH)
        note_count = sum(1 for step in steps if step.mental_note)

        if outcome == TaskOutcome.COMPLETED:
            opening = f"{name} completed the task in {step_count} step(s)"
        elif outcome == TaskOutcome.COMPLETED_WITH_ERRORS:
            opening = f"{name} completed the task in {step_count} step(s), but with friction"
        elif outcome == TaskOutcome.TIMED_OUT:
            opening = f"{name} ran out of steps before completing the task"
        else:
            opening = f"{name} did not complete the task"

        friction_bits: list[str] = []
        if high_count:
            friction_bits.append(f"{high_count} high-severity issue(s)")
        if mismatch_count:
            friction_bits.append(f"{mismatch_count} expectation mismatch(es)")
        if note_count:
            friction_bits.append(f"{note_count} thing(s) they had to figure out")

        friction = (
            " The main signals were " + ", ".join(friction_bits) + "."
            if friction_bits
            else " The path appeared smooth in the recorded trace."
        )
        emotion = f" Their emotional state moved from {first_emotion} to {last_emotion}."
        return opening + "." + friction + emotion

    def _summarize_persona(self, persona: Persona) -> str:
        parts = [
            f"tech_literacy={persona.tech_literacy.value}",
            f"patience={persona.patience}",
            f"reading={persona.reading_comprehension.value}",
            f"domain={persona.domain_familiarity.value}",
            f"goal_clarity={persona.goal_clarity.value}",
        ]
        return ", ".join(parts)
