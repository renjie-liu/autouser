"""Multi-seed studies: from anecdote to evidence (cognitive v2, slice 4).

A study runs M archetypes × N sampled seeds against one task and aggregates:

* **Issue clusters** — findings grouped by (category, url) and ranked by how
  many runs reproduce them. Reproduction frequency is the evidence a single
  friction log can't provide.
* **Persona stats** — per-archetype outcome rates, mean steps, and mean
  confidence across seeds: the behavioral distributions M3's gate needs.

Aggregation functions are pure over run records, so a study is recomputable
from its serialized `study.json` — same replayability contract as `affect`.

Clustering key is deliberately coarse for v1: (category, url). Free-text
descriptions are persona-voiced and can't be keyed; element-level matching
is the M5 match definition and arrives with calibration. A coarse cluster
that over-merges is visible in its example quotes; an uncountable anecdote
is not.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field

from autouser.cognitive.models import IssueCategory, Severity
from autouser.cognitive.protocol import EngineProtocol
from autouser.persona.models import Persona, PersonaArchetype
from autouser.persona.variance import sample_variant, variant_summary
from autouser.report.human_renderer import render_report
from autouser.report.journey_renderer import render_journey
from autouser.report.models import FrictionLog, TaskOutcome
from autouser.runner import PerceptionMode, SimulationRunner
from autouser.session import TaskSpec

logger = logging.getLogger(__name__)

EngineFactory = Callable[[Persona], EngineProtocol]

_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
_MAX_CLUSTER_EXAMPLES = 3


class RunRecord(BaseModel):
    """One (archetype, seed) cell of the study."""

    persona_name: str
    seed: int
    persona_summary: str
    outcome: Optional[TaskOutcome] = None  # None when the run crashed
    total_steps: int = 0
    issue_count: int = 0
    insight_count: int = 0
    error: Optional[str] = None
    log: Optional[FrictionLog] = None


class IssueCluster(BaseModel):
    """Findings that recur across runs, with reproduction evidence."""

    category: IssueCategory
    url: str
    run_count: int = Field(description="Distinct runs in which this cluster appeared")
    total_runs: int
    occurrences: int = Field(description="Total issue instances across all runs")
    personas: dict[str, int] = Field(description="archetype -> affected run count")
    worst_severity: Severity
    feedback_subtypes: dict[str, int] = Field(default_factory=dict)
    examples: list[str] = Field(default_factory=list)

    @property
    def reproduction_rate(self) -> float:
        return self.run_count / self.total_runs if self.total_runs else 0.0


class PersonaStats(BaseModel):
    """Per-archetype behavioral distribution across seeds (M3 substrate)."""

    persona_name: str
    runs: int
    completed: int = 0
    completed_with_errors: int = 0
    abandoned: int = 0
    timed_out: int = 0
    crashed: int = 0
    mean_steps: float = 0.0
    mean_confidence: float = 0.0


class StudyResult(BaseModel):
    task: str
    success_criteria: str
    start_url: str
    seeds_per_persona: int
    runs: list[RunRecord]
    clusters: list[IssueCluster]
    persona_stats: list[PersonaStats]


# --- Aggregation (pure) -------------------------------------------------------


def cluster_issues(runs: list[RunRecord]) -> list[IssueCluster]:
    """Group issues by (category, url); rank by reproduction, then severity."""
    completed_runs = [r for r in runs if r.log is not None]
    total = len(completed_runs)

    buckets: dict[tuple[IssueCategory, str], dict] = {}
    for record in completed_runs:
        seen_in_run: set[tuple[IssueCategory, str]] = set()
        for issue in record.log.issues:
            key = (issue.category, issue.url)
            bucket = buckets.setdefault(
                key,
                {
                    "occurrences": 0,
                    "runs": set(),
                    "personas": defaultdict(set),
                    "worst": issue.severity,
                    "subtypes": defaultdict(int),
                    "examples": [],
                },
            )
            bucket["occurrences"] += 1
            run_id = (record.persona_name, record.seed)
            bucket["runs"].add(run_id)
            bucket["personas"][record.persona_name].add(record.seed)
            if _SEVERITY_RANK[issue.severity] < _SEVERITY_RANK[bucket["worst"]]:
                bucket["worst"] = issue.severity
            if issue.feedback_subtype is not None:
                bucket["subtypes"][issue.feedback_subtype.value] += 1
            # One example per run keeps quotes diverse across seeds.
            if key not in seen_in_run and len(bucket["examples"]) < _MAX_CLUSTER_EXAMPLES:
                bucket["examples"].append(issue.description)
            seen_in_run.add(key)

    clusters = [
        IssueCluster(
            category=key[0],
            url=key[1],
            run_count=len(bucket["runs"]),
            total_runs=total,
            occurrences=bucket["occurrences"],
            personas={name: len(seeds) for name, seeds in bucket["personas"].items()},
            worst_severity=bucket["worst"],
            feedback_subtypes=dict(bucket["subtypes"]),
            examples=bucket["examples"],
        )
        for key, bucket in buckets.items()
    ]
    clusters.sort(
        key=lambda c: (-c.run_count, _SEVERITY_RANK[c.worst_severity], -c.occurrences)
    )
    return clusters


def compute_persona_stats(runs: list[RunRecord]) -> list[PersonaStats]:
    by_name: dict[str, list[RunRecord]] = defaultdict(list)
    for record in runs:
        by_name[record.persona_name].append(record)

    stats: list[PersonaStats] = []
    for name in sorted(by_name):
        records = by_name[name]
        outcomes = defaultdict(int)
        steps: list[int] = []
        confidences: list[float] = []
        for record in records:
            if record.log is None:
                outcomes["crashed"] += 1
                continue
            outcomes[record.log.outcome.value] += 1
            steps.append(record.log.total_steps)
            confidences.extend(s.intent.confidence for s in record.log.steps)
        stats.append(
            PersonaStats(
                persona_name=name,
                runs=len(records),
                completed=outcomes[TaskOutcome.COMPLETED.value],
                completed_with_errors=outcomes[TaskOutcome.COMPLETED_WITH_ERRORS.value],
                abandoned=outcomes[TaskOutcome.ABANDONED.value],
                timed_out=outcomes[TaskOutcome.TIMED_OUT.value],
                crashed=outcomes["crashed"],
                mean_steps=sum(steps) / len(steps) if steps else 0.0,
                mean_confidence=(
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
            )
        )
    return stats


# --- Orchestration -------------------------------------------------------------


async def _run_cell(
    persona: Persona,
    seed: int,
    task_spec: TaskSpec,
    *,
    perception_mode: PerceptionMode,
    engine_factory: Optional[EngineFactory],
    cell_dir: Path,
    audit_judge=None,
) -> FrictionLog:
    engine = engine_factory(persona) if engine_factory else None
    runner = SimulationRunner(
        persona=persona,
        task_spec=task_spec,
        screenshot_dir=cell_dir,
        perception_mode=perception_mode,
        engine=engine,
    )
    log = await runner.run()
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "report.md").write_text(render_report(log, persona), encoding="utf-8")
    (cell_dir / "journey.md").write_text(render_journey(log, persona), encoding="utf-8")
    (cell_dir / "friction_log.json").write_text(
        log.model_dump_json(indent=2), encoding="utf-8"
    )
    if audit_judge is not None:
        from autouser.audit import audit_log

        audit = await audit_log(log, judge=audit_judge)
        (cell_dir / "audit.json").write_text(
            audit.model_dump_json(indent=2), encoding="utf-8"
        )
    return log


async def run_study(
    archetypes: Sequence[PersonaArchetype],
    task_spec: TaskSpec,
    *,
    seeds_per_persona: int = 3,
    out_dir: Path,
    perception_mode: PerceptionMode | str = PerceptionMode.DOM,
    engine_factory: Optional[EngineFactory] = None,
    audit_judge=None,
) -> StudyResult:
    """Run the full study matrix and aggregate.

    Cells run concurrently (`asyncio.gather`); with the claude_code provider
    the process-global spawn gate still serializes LLM calls, so raise
    AUTOUSER_CLAUDE_CODE_MAX_SPAWNS to parallelize a study. A crashed cell
    becomes a RunRecord with `error` set — never a lost study.

    When ``audit_judge`` is provided (the `--audit` flag), each completed cell
    additionally writes an ``audit.json`` — the false-positive audit per run.
    """
    perception = PerceptionMode(perception_mode)
    cells: list[tuple[Persona, int]] = []
    for archetype in archetypes:
        for seed in range(seeds_per_persona):
            cells.append((sample_variant(archetype, seed), seed))

    coros = [
        _run_cell(
            persona, seed, task_spec,
            perception_mode=perception,
            engine_factory=engine_factory,
            cell_dir=out_dir / f"{persona.archetype.name}-s{seed}",
            audit_judge=audit_judge,
        )
        for persona, seed in cells
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    runs: list[RunRecord] = []
    for (persona, seed), result in zip(cells, results):
        name = persona.archetype.name if persona.archetype else "custom"
        if isinstance(result, BaseException):
            logger.warning("study_cell_crashed: %s-s%d: %s", name, seed, result)
            runs.append(
                RunRecord(
                    persona_name=name,
                    seed=seed,
                    persona_summary=variant_summary(persona),
                    error=f"{type(result).__name__}: {result}",
                )
            )
        else:
            runs.append(
                RunRecord(
                    persona_name=name,
                    seed=seed,
                    persona_summary=variant_summary(persona),
                    outcome=result.outcome,
                    total_steps=result.total_steps,
                    issue_count=len(result.issues),
                    insight_count=len(result.insights),
                    log=result,
                )
            )

    return StudyResult(
        task=task_spec.task,
        success_criteria=task_spec.success_criteria,
        start_url=task_spec.start_url,
        seeds_per_persona=seeds_per_persona,
        runs=runs,
        clusters=cluster_issues(runs),
        persona_stats=compute_persona_stats(runs),
    )
