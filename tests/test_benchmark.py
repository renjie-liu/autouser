"""Planted-issue benchmark scoring (cognitive v2, slice 7).

Pure precision/recall against ground truth — the M5 gate's substrate.
Pins the match relation (category + severity-within-one-tier + keyword/
subtype) and the set-overlap precision/recall semantics.
"""

from __future__ import annotations

from pathlib import Path


from autouser.benchmark import (
    Benchmark,
    GroundTruthIssue,
    score_issues,
    score_logs,
)
from autouser.cognitive.models import FeedbackSubtype, IssueCategory, Severity
from autouser.report.models import FrictionLog, Issue, TaskOutcome

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"


def _issue(cat, sev, desc, *, perspective="", subtype=None) -> Issue:
    return Issue(
        severity=sev, category=cat, description=desc, step=1,
        url="https://x.test/", persona_perspective=perspective,
        feedback_subtype=subtype,
    )


def _log(issues) -> FrictionLog:
    return FrictionLog(
        task="t", success_criteria="c", persona_name="maria",
        persona_summary="s", outcome=TaskOutcome.ABANDONED, total_steps=len(issues),
        steps=[], issues=issues, overall_impressions="",
    )


_PHONE_GT = GroundTruthIssue(
    id="phone", categories=[IssueCategory.ERROR_RECOVERY],
    expected_severity=Severity.HIGH, keywords=["format", "+1", "phone"],
)


class TestMatchRelation:
    def test_category_must_match(self):
        gt = _PHONE_GT
        assert not gt.matches(_issue(IssueCategory.LABELING, Severity.HIGH, "wrong phone format"))
        assert gt.matches(_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone format wrong"))

    def test_keyword_required_when_set(self):
        gt = _PHONE_GT
        assert not gt.matches(_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "some other issue"))

    def test_keyword_matches_in_perspective_too(self):
        gt = _PHONE_GT
        assert gt.matches(
            _issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "an error",
                   perspective="the +1 thing confused me")
        )

    def test_severity_within_one_tier(self):
        gt = _PHONE_GT  # HIGH
        assert gt.matches(_issue(IssueCategory.ERROR_RECOVERY, Severity.MEDIUM, "phone format"))
        assert not gt.matches(_issue(IssueCategory.ERROR_RECOVERY, Severity.LOW, "phone format"))

    def test_feedback_subtype_match_bypasses_keywords(self):
        gt = GroundTruthIssue(
            id="silent", categories=[IssueCategory.FEEDBACK],
            expected_severity=Severity.MEDIUM, keywords=["no confirmation"],
            feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
        )
        # No keyword present, but the subtype matches → still a match.
        assert gt.matches(
            _issue(IssueCategory.FEEDBACK, Severity.MEDIUM, "objectively done",
                   subtype=FeedbackSubtype.SILENT_SUCCESS)
        )


class TestScoring:
    def test_perfect_run(self):
        bench = Benchmark(name="b", issues=[_PHONE_GT])
        score = score_issues([_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone format +1")], bench)
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.false_positive_indices == []

    def test_false_positive_lowers_precision(self):
        bench = Benchmark(name="b", issues=[_PHONE_GT])
        issues = [
            _issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone format"),
            _issue(IssueCategory.COPY, Severity.LOW, "unrelated nitpick"),
        ]
        score = score_issues(issues, bench)
        assert score.precision == 0.5
        assert score.recall == 1.0
        assert score.false_positive_indices == [1]

    def test_miss_lowers_recall(self):
        bench = Benchmark(name="b", issues=[
            _PHONE_GT,
            GroundTruthIssue(id="icon", categories=[IssueCategory.LABELING],
                             expected_severity=Severity.HIGH, keywords=["icon"]),
        ])
        score = score_issues([_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone +1")], bench)
        assert score.recall == 0.5
        assert score.unmatched_ground_truth_ids == ["icon"]

    def test_empty_run_is_zero_precision_zero_recall(self):
        bench = Benchmark(name="b", issues=[_PHONE_GT])
        score = score_issues([], bench)
        assert score.precision == 0.0
        assert score.recall == 0.0

    def test_one_issue_can_match_multiple_truths_and_vice_versa(self):
        # Two personas both complaining about the phone pattern = one truth,
        # two true positives (both trustworthy), recall counts the truth once.
        bench = Benchmark(name="b", issues=[_PHONE_GT])
        issues = [
            _issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone format"),
            _issue(IssueCategory.ERROR_RECOVERY, Severity.MEDIUM, "the +1 format"),
        ]
        score = score_issues(issues, bench)
        assert score.true_positives == 2
        assert score.matched_ground_truth == 1
        assert score.recall == 1.0


class TestUnionScore:
    def test_union_recall_across_runs(self):
        bench = Benchmark(name="b", issues=[
            GroundTruthIssue(id="phone", categories=[IssueCategory.ERROR_RECOVERY],
                             expected_severity=Severity.HIGH, keywords=["phone"]),
            GroundTruthIssue(id="icon", categories=[IssueCategory.LABELING],
                             expected_severity=Severity.HIGH, keywords=["icon"]),
        ])
        # Run A finds phone only; run B finds icon only — union finds both.
        log_a = _log([_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone format")])
        log_b = _log([_issue(IssueCategory.LABELING, Severity.HIGH, "the icon")])
        union = score_logs([log_a, log_b], bench)
        assert union.union_recall == 1.0
        assert union.union_matched_ground_truth == 2
        assert union.mean_precision == 1.0

    def test_mean_precision_skips_empty_runs(self):
        bench = Benchmark(name="b", issues=[
            GroundTruthIssue(id="phone", categories=[IssueCategory.ERROR_RECOVERY],
                             expected_severity=Severity.HIGH, keywords=["phone"]),
        ])
        good = _log([_issue(IssueCategory.ERROR_RECOVERY, Severity.HIGH, "phone")])
        empty = _log([])
        union = score_logs([good, empty], bench)
        # Empty run contributes nothing to precision (no findings to be wrong).
        assert union.mean_precision == 1.0


class TestShippedBenchmark:
    def test_friction_page_benchmark_loads(self):
        bench = Benchmark.from_yaml(BENCHMARKS / "friction_page.yaml")
        assert bench.name == "friction_page"
        assert {g.id for g in bench.issues} == {
            "icon_button", "swapped_buttons", "phone_mismatch",
            "disabled_look_cta", "silent_success",
        }

    def test_silent_success_entry_has_subtype(self):
        bench = Benchmark.from_yaml(BENCHMARKS / "friction_page.yaml")
        silent = next(g for g in bench.issues if g.id == "silent_success")
        assert silent.feedback_subtype == FeedbackSubtype.SILENT_SUCCESS
