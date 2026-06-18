"""Multi-seed studies (cognitive v2, slice 4): variance, aggregation, synthesis.

Pure tests for the sampler and aggregation; one scripted-engine study runs
the full matrix against a local fixture page (real browser, no LLM).
"""

from __future__ import annotations

import importlib.util
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    IssueCategory,
    Severity,
    StepResult,
    SuccessPredicate,
    UIState,
)
from autouser.cognitive.scripted import ScriptedEngine, ScriptedStep
from autouser.persona.models import Level
from autouser.persona.registry import get_archetype
from autouser.persona.variance import sample_variant, variant_summary
from autouser.report.generator import ReportGenerator
from autouser.report.study_renderer import render_study
from autouser.session import TaskSpec
from autouser.study import (
    RunRecord,
    StudyResult,
    cluster_issues,
    compute_persona_stats,
    run_study,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_skip_without_playwright = pytest.mark.skipif(
    importlib.util.find_spec("playwright.async_api") is None,
    reason="Playwright not installed",
)


# --- variance ----------------------------------------------------------------


class TestVarianceSampling:
    def test_seed_zero_is_exact_archetype(self):
        maria = get_archetype("maria")
        variant = sample_variant(maria, 0)
        assert variant.patience == maria.patience
        assert variant.tech_literacy == maria.tech_literacy
        assert variant.reading_comprehension == maria.reading_comprehension

    def test_same_seed_is_deterministic(self):
        maria = get_archetype("maria")
        a = sample_variant(maria, 7)
        b = sample_variant(maria, 7)
        assert a.model_dump(exclude={"archetype"}) == b.model_dump(exclude={"archetype"})

    def test_seeds_produce_variance(self):
        maria = get_archetype("maria")
        variants = [sample_variant(maria, s) for s in range(1, 12)]
        baselines = [
            v for v in variants
            if v.patience == maria.patience
            and v.tech_literacy == maria.tech_literacy
            and v.reading_comprehension == maria.reading_comprehension
            and v.domain_familiarity == maria.domain_familiarity
            and v.goal_clarity == maria.goal_clarity
        ]
        assert len(baselines) < len(variants), "no seed produced any jitter"

    def test_patience_clamped_to_model_range(self):
        carlos = get_archetype("carlos")  # patience 2 — near the floor
        for seed in range(1, 40):
            assert 1 <= sample_variant(carlos, seed).patience <= 10

    def test_accessibility_never_jittered(self):
        priya = get_archetype("priya")
        for seed in range(1, 30):
            variant = sample_variant(priya, seed)
            assert variant.accessibility.screen_reader is True
            assert variant.accessibility.keyboard_only is True
        carlos = get_archetype("carlos")
        for seed in range(1, 30):
            assert sample_variant(carlos, seed).accessibility.motor_precision == Level.LOW

    def test_level_shifts_are_adjacent_only(self):
        maria = get_archetype("maria")  # tech LOW, reading HIGH
        for seed in range(1, 40):
            v = sample_variant(maria, seed)
            # From LOW only LOW/MEDIUM are reachable; from HIGH only HIGH/MEDIUM.
            assert v.tech_literacy in (Level.LOW, Level.MEDIUM)
            assert v.reading_comprehension in (Level.HIGH, Level.MEDIUM)

    def test_variant_summary_matches_friction_log_format(self):
        maria = sample_variant(get_archetype("maria"), 0)
        log = ReportGenerator().generate(
            steps=[], persona=maria, task="t", success_criteria="c",
        )
        assert variant_summary(maria) == log.persona_summary


# --- aggregation ---------------------------------------------------------------


def _ui(url="https://x.test/") -> UIState:
    return UIState(url=url, page_title="T", dom_summary="d", visible_text="v")


def _step_with_issue(n: int, *, url: str, category: IssueCategory,
                     severity: Severity, description: str) -> StepResult:
    return StepResult(
        step=n,
        intent=ActionIntent(
            action=ActionType.CLICK, target="#a",
            persona_thought="t", expected_outcome="e", confidence=0.6,
        ),
        observation_before=_ui(),
        observation_after=_ui(url=url),
        actual_outcome="o",
        mismatch=True,
        severity=severity,
        category=category,
        reflection=description,
        emotion=Emotion.CONFUSED,
    )


def _record(persona: str, seed: int, issues: list[tuple[str, IssueCategory, Severity, str]],
            outcome_steps: int = 3) -> RunRecord:
    steps = [
        _step_with_issue(i + 1, url=url, category=cat, severity=sev, description=desc)
        for i, (url, cat, sev, desc) in enumerate(issues)
    ] or [
        StepResult(
            step=1,
            intent=ActionIntent(action=ActionType.CLICK, target="#a",
                                persona_thought="t", expected_outcome="e", confidence=0.9),
            observation_before=_ui(), observation_after=_ui(url="https://x.test/done"),
            actual_outcome="o", mismatch=False, reflection="fine",
            emotion=Emotion.SATISFIED,
        )
    ]
    persona_obj = sample_variant(get_archetype(persona), seed)
    log = ReportGenerator().generate(
        steps=steps, persona=persona_obj, task="t", success_criteria="c",
    )
    return RunRecord(
        persona_name=persona, seed=seed,
        persona_summary=variant_summary(persona_obj),
        outcome=log.outcome, total_steps=log.total_steps,
        issue_count=len(log.issues), insight_count=0, log=log,
    )


_MODAL = ("https://x.test/modal", IssueCategory.NAVIGATION, Severity.HIGH, "stuck at the modal")
_COPY = ("https://x.test/form", IssueCategory.COPY, Severity.MEDIUM, "placeholder lied")


class TestClustering:
    def test_reproduction_across_runs_ranks_first(self):
        runs = [
            _record("maria", 0, [_MODAL]),
            _record("maria", 1, [_MODAL]),
            _record("carlos", 0, [_MODAL, _COPY]),
            _record("carlos", 1, []),
        ]
        clusters = cluster_issues(runs)
        assert clusters[0].category == IssueCategory.NAVIGATION
        assert clusters[0].run_count == 3
        assert clusters[0].total_runs == 4
        assert clusters[0].personas == {"maria": 2, "carlos": 1}
        assert clusters[1].run_count == 1

    def test_multiple_instances_in_one_run_count_once_for_reproduction(self):
        runs = [_record("maria", 0, [_MODAL, _MODAL])]
        clusters = cluster_issues(runs)
        assert clusters[0].run_count == 1
        assert clusters[0].occurrences == 2

    def test_worst_severity_wins(self):
        low_modal = (_MODAL[0], _MODAL[1], Severity.LOW, "mild")
        runs = [_record("maria", 0, [low_modal]), _record("maria", 1, [_MODAL])]
        assert cluster_issues(runs)[0].worst_severity == Severity.HIGH

    def test_crashed_runs_excluded_from_denominator(self):
        runs = [
            _record("maria", 0, [_MODAL]),
            RunRecord(persona_name="maria", seed=1, persona_summary="s", error="boom"),
        ]
        clusters = cluster_issues(runs)
        assert clusters[0].total_runs == 1


class TestPersonaStats:
    def test_outcome_counts_and_means(self):
        runs = [
            _record("maria", 0, []),          # no issues → completed-ish path
            _record("maria", 1, [_MODAL]),
            RunRecord(persona_name="maria", seed=2, persona_summary="s", error="boom"),
        ]
        stats = compute_persona_stats(runs)
        assert len(stats) == 1
        s = stats[0]
        assert s.runs == 3
        assert s.crashed == 1
        assert s.mean_steps > 0
        assert 0.0 < s.mean_confidence <= 1.0


class TestStudyRenderer:
    def test_synthesis_contains_reproduction_and_stats(self):
        runs = [
            _record("maria", 0, [_MODAL]),
            _record("maria", 1, [_MODAL]),
            _record("carlos", 0, []),
        ]
        result = StudyResult(
            task="Do the thing.", success_criteria="Done.", start_url="https://x.test/",
            seeds_per_persona=2, runs=runs,
            clusters=cluster_issues(runs), persona_stats=compute_persona_stats(runs),
        )
        out = render_study(result)
        assert "2/3" in out                       # reproduction fraction
        assert "stuck at the modal" in out        # example quote
        assert "Behavior by persona" in out
        assert "| maria | 2 |" in out
        assert "Run matrix" in out
        assert "Seed 0 is the unjittered archetype" in out


# --- end-to-end with scripted engines (browser, no LLM) ------------------------


class _QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(directory), **kwargs)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def fixture_server():
    handler = partial(_QuietHandler, directory=FIXTURES_DIR)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@_skip_without_playwright
async def test_study_end_to_end_with_scripted_engines(fixture_server, tmp_path):
    """2 personas × 2 seeds through the real runner + browser, scripted
    cognition: artifacts per cell, aggregate over all four runs."""
    task = TaskSpec(
        task="Press the working save button.",
        success_criteria="Data persisted.",
        start_url=f"{fixture_server}/friction_page.html",
        max_steps=4,
        success_predicate=SuccessPredicate(selector_present="section#save-confirmed"),
    )

    def factory(persona):
        return ScriptedEngine(steps=[
            ScriptedStep(
                intent=ActionIntent(
                    action=ActionType.CLICK, target="#silent-save",
                    persona_thought="press save", expected_outcome="it saves",
                    confidence=0.9,
                ),
                actual_outcome="clicked save",
            ),
        ])

    result = await run_study(
        [get_archetype("maria"), get_archetype("jake")],
        task,
        seeds_per_persona=2,
        out_dir=tmp_path / "study",
        engine_factory=factory,
    )

    assert len(result.runs) == 4
    assert all(r.error is None for r in result.runs)
    assert {(r.persona_name, r.seed) for r in result.runs} == {
        ("maria", 0), ("maria", 1), ("jake", 0), ("jake", 1),
    }
    names = {s.persona_name for s in result.persona_stats}
    assert names == {"maria", "jake"}
    for r in result.runs:
        cell = tmp_path / "study" / f"{r.persona_name}-s{r.seed}"
        assert (cell / "report.md").exists()
        assert (cell / "journey.md").exists()
        assert (cell / "friction_log.json").exists()
        # No audit requested → no audit.json (the default).
        assert not (cell / "audit.json").exists()
    # Study is recomputable from its records (replayability contract).
    assert cluster_issues(result.runs) == result.clusters


@_skip_without_playwright
async def test_study_audit_judge_writes_per_cell_audit(fixture_server, tmp_path):
    """PR#45 finding 1: `study --audit` must actually produce audit.json per
    completed cell — the flag's promise. Uses an injected judge (no provider)."""
    import json

    task = TaskSpec(
        task="Press the working save button.",
        success_criteria="Data persisted.",
        start_url=f"{fixture_server}/friction_page.html",
        max_steps=4,
        success_predicate=SuccessPredicate(selector_present="section#save-confirmed"),
    )

    def factory(persona):
        return ScriptedEngine(steps=[
            ScriptedStep(intent=ActionIntent(
                action=ActionType.CLICK, target="#silent-save",
                persona_thought="press save", expected_outcome="saves", confidence=0.9,
            ), actual_outcome="clicked save"),
        ])

    async def judge(system, user, schema):
        # Verdict per finding index; harmless when a cell has zero findings
        # (audit_log short-circuits before calling the judge).
        return json.dumps({"verdicts": [
            {"index": 0, "verdict": "real", "confidence": 0.9, "rationale": "r"},
        ]})

    result = await run_study(
        [get_archetype("maria")], task,
        seeds_per_persona=1,
        out_dir=tmp_path / "study",
        engine_factory=factory,
        audit_judge=judge,
    )

    assert len(result.runs) == 1
    cell = tmp_path / "study" / "maria-s0"
    assert (cell / "audit.json").exists()
    audit = json.loads((cell / "audit.json").read_text())
    assert {"verdicts", "real_count", "artifact_count", "uncertain_count"} <= audit.keys()
