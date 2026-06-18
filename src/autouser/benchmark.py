"""Planted-issue benchmarks: deterministic precision/recall (cognitive v2, slice 7).

The trust question — "can we believe what AutoUser reports?" — needs a
ground-truth answer before any human-study calibration. A benchmark declares
the issues a fixture page is KNOWN to contain (planted dark patterns); scoring
a run against it yields precision (of what AutoUser flagged, how much is real)
and recall (of the known issues, how many it found) — the M5 gate's metrics.

Match definition (M5): an AutoUser issue matches a ground-truth issue when it
identifies the same flow AND assigns a severity within one tier. "Same flow"
is approximated here by (category match) AND (a distinctive keyword from the
ground-truth entry appears in the issue's description or the persona's
perspective), or by an explicit feedback_subtype match. Element-level matching
is the eventual M5 definition; the keyword proxy is the honest interim — each
planted pattern has distinctive vocabulary, and a coarse over-match is visible
in the matched-pair listing while an uncountable anecdote is not.

Everything here is pure over (issues, benchmark): a score is recomputable from
a serialized friction_log.json / study.json, same replayability contract as
affect and study.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from autouser.cognitive.models import FeedbackSubtype, IssueCategory, Severity
from autouser.report.models import FrictionLog, Issue

_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


def _within_one_tier(a: Severity, b: Severity) -> bool:
    return abs(_SEVERITY_RANK[a] - _SEVERITY_RANK[b]) <= 1


class GroundTruthIssue(BaseModel):
    """A planted, known-present usability issue on a benchmark page.

    The M5 match definition keys on *flow* + severity, NOT category — and
    category is empirically the fuzziest signal (the LLM calls a misleading-
    placeholder trap `copy` or `error_recovery` interchangeably; see the open
    taxonomy issues). So ``categories`` is an ANY-OF allow-list of the
    categories a real classifier plausibly assigns to this flow, and the
    keywords carry the real flow identification. An empty ``categories`` skips
    the category gate entirely.
    """

    id: str
    categories: list[IssueCategory] = Field(default_factory=list)
    expected_severity: Severity
    keywords: list[str] = Field(
        default_factory=list,
        description="ANY of these (case-insensitive) appearing in an issue's "
        "description or persona_perspective satisfies the 'same flow' half of "
        "the match. Omit only when feedback_subtype alone identifies the flow.",
    )
    feedback_subtype: FeedbackSubtype | None = None
    description: str = ""

    def matches(self, issue: Issue) -> bool:
        if self.categories and issue.category not in self.categories:
            return False
        if not _within_one_tier(issue.severity, self.expected_severity):
            return False
        if self.feedback_subtype is not None and issue.feedback_subtype == self.feedback_subtype:
            return True
        if not self.keywords:
            # No keyword discriminator and subtype didn't match (or none set):
            # category + severity is the whole key.
            return self.feedback_subtype is None
        haystack = f"{issue.description} {issue.persona_perspective}".lower()
        return any(kw.lower() in haystack for kw in self.keywords)


class Benchmark(BaseModel):
    """Ground truth for one fixture page."""

    name: str
    url: str = ""
    issues: list[GroundTruthIssue]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Benchmark:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)


class MatchedPair(BaseModel):
    ground_truth_id: str
    issue_index: int
    issue_description: str


class ScoreResult(BaseModel):
    """Precision/recall of one set of flagged issues against a benchmark."""

    total_flagged: int
    total_ground_truth: int
    true_positives: int  # flagged issues matching >=1 ground truth
    matched_ground_truth: int  # ground truth matched by >=1 flagged issue
    matches: list[MatchedPair]
    unmatched_ground_truth_ids: list[str]
    false_positive_indices: list[int]

    @property
    def precision(self) -> float:
        return self.true_positives / self.total_flagged if self.total_flagged else 0.0

    @property
    def recall(self) -> float:
        return (
            self.matched_ground_truth / self.total_ground_truth
            if self.total_ground_truth
            else 0.0
        )


def score_issues(issues: list[Issue], benchmark: Benchmark) -> ScoreResult:
    """Set-overlap precision/recall: an issue is a true positive if it matches
    any ground-truth entry; a ground-truth is found if any issue matches it.

    Not a strict 1:1 assignment — a single dark pattern legitimately draws
    multiple persona complaints, and one complaint can implicate one pattern.
    """
    matches: list[MatchedPair] = []
    matched_truth_ids: set[str] = set()
    true_positive_indices: set[int] = set()

    for idx, issue in enumerate(issues):
        for gt in benchmark.issues:
            if gt.matches(issue):
                matches.append(
                    MatchedPair(
                        ground_truth_id=gt.id,
                        issue_index=idx,
                        issue_description=issue.description,
                    )
                )
                matched_truth_ids.add(gt.id)
                true_positive_indices.add(idx)

    all_ids = {gt.id for gt in benchmark.issues}
    return ScoreResult(
        total_flagged=len(issues),
        total_ground_truth=len(benchmark.issues),
        true_positives=len(true_positive_indices),
        matched_ground_truth=len(matched_truth_ids),
        matches=matches,
        unmatched_ground_truth_ids=sorted(all_ids - matched_truth_ids),
        false_positive_indices=sorted(set(range(len(issues))) - true_positive_indices),
    )


def score_log(log: FrictionLog, benchmark: Benchmark) -> ScoreResult:
    return score_issues(log.issues, benchmark)


class UnionScore(BaseModel):
    """Cross-run scoring (the M5 gate shape).

    Precision is averaged over runs that flagged anything (each run's
    trustworthiness), while recall is taken over the UNION of all runs —
    "collectively, do the personas find the known issues?" — which is exactly
    how the M5 gate reads recall.
    """

    runs_scored: int
    mean_precision: float
    union_recall: float
    union_matched_ground_truth: int
    total_ground_truth: int
    per_run: list[ScoreResult]


def score_logs(logs: list[FrictionLog], benchmark: Benchmark) -> UnionScore:
    per_run = [score_log(log, benchmark) for log in logs]
    precisions = [r.precision for r in per_run if r.total_flagged]
    union_ids: set[str] = set()
    for r in per_run:
        union_ids.update(m.ground_truth_id for m in r.matches)
    total_gt = len(benchmark.issues)
    return UnionScore(
        runs_scored=len(logs),
        mean_precision=sum(precisions) / len(precisions) if precisions else 0.0,
        union_recall=len(union_ids) / total_gt if total_gt else 0.0,
        union_matched_ground_truth=len(union_ids),
        total_ground_truth=total_gt,
        per_run=per_run,
    )
