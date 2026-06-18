"""False-positive audit (cognitive v2, slice 7).

The audit's prompt builder, response parsing, and aggregation are tested
with an injected judge — no provider needed.
"""

from __future__ import annotations

import json

import pytest

from autouser.audit import AuditReport, audit_log, build_audit_prompt
from autouser.cognitive.models import IssueCategory, Severity
from autouser.report.models import FrictionLog, Issue, TaskOutcome


def _issue(desc, cat=IssueCategory.FEEDBACK, sev=Severity.MEDIUM) -> Issue:
    return Issue(
        severity=sev, category=cat, description=desc, step=1,
        url="https://x.test/", persona_perspective="because reasons",
    )


def _log(issues) -> FrictionLog:
    return FrictionLog(
        task="Buy a widget", success_criteria="c", persona_name="maria",
        persona_summary="s", outcome=TaskOutcome.ABANDONED, total_steps=len(issues),
        steps=[], issues=issues, overall_impressions="",
    )


def _judge_returning(verdicts):
    async def _j(system, user, schema):
        return json.dumps({"verdicts": verdicts})
    return _j


class TestAuditPrompt:
    def test_prompt_enumerates_all_findings(self):
        log = _log([_issue("button is unlabeled"), _issue("error message unclear")])
        prompt = build_audit_prompt(log)
        assert "[0]" in prompt and "[1]" in prompt
        assert "button is unlabeled" in prompt
        assert "Buy a widget" in prompt


class TestAuditAggregation:
    async def test_counts_and_audited_precision(self):
        log = _log([_issue("a"), _issue("b"), _issue("c")])
        judge = _judge_returning([
            {"index": 0, "verdict": "real", "confidence": 0.9, "rationale": "x"},
            {"index": 1, "verdict": "artifact", "confidence": 0.8, "rationale": "y"},
            {"index": 2, "verdict": "real", "confidence": 0.7, "rationale": "z"},
        ])
        report = await audit_log(log, judge=judge)
        assert report.real_count == 2
        assert report.artifact_count == 1
        assert report.uncertain_count == 0
        assert report.audited_precision == pytest.approx(2 / 3)

    async def test_missing_index_becomes_uncertain(self):
        log = _log([_issue("a"), _issue("b")])
        judge = _judge_returning([
            {"index": 0, "verdict": "real", "confidence": 0.9, "rationale": "x"},
        ])
        report = await audit_log(log, judge=judge)
        assert len(report.verdicts) == 2  # every finding accounted for
        assert report.uncertain_count == 1
        assert report.verdicts[1].issue_index == 1

    async def test_hallucinated_index_ignored(self):
        log = _log([_issue("a")])
        judge = _judge_returning([
            {"index": 0, "verdict": "real", "confidence": 0.9, "rationale": "x"},
            {"index": 7, "verdict": "real", "confidence": 0.9, "rationale": "ghost"},
        ])
        report = await audit_log(log, judge=judge)
        assert [v.issue_index for v in report.verdicts] == [0]

    async def test_invalid_verdict_coerced_to_uncertain(self):
        log = _log([_issue("a")])
        judge = _judge_returning([
            {"index": 0, "verdict": "definitely-real", "confidence": 0.9, "rationale": "x"},
        ])
        report = await audit_log(log, judge=judge)
        assert report.verdicts[0].verdict == "uncertain"

    async def test_empty_log_short_circuits(self):
        called = False

        async def judge(system, user, schema):
            nonlocal called
            called = True
            return "{}"

        report = await audit_log(_log([]), judge=judge)
        assert report == AuditReport(verdicts=[], real_count=0, artifact_count=0, uncertain_count=0)
        assert called is False

    async def test_markdown_fenced_json_parsed(self):
        log = _log([_issue("a")])

        async def judge(system, user, schema):
            return '```json\n{"verdicts": [{"index": 0, "verdict": "artifact", "confidence": 0.5, "rationale": "r"}]}\n```'

        report = await audit_log(log, judge=judge)
        assert report.artifact_count == 1
