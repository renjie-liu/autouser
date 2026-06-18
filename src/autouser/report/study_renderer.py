"""Study synthesis renderer: reproduction evidence over anecdote.

Renders a StudyResult as the cross-run markdown report: which findings
reproduce, at what rate, across which personas — plus the per-archetype
behavioral distributions (completion rates, steps, confidence) that single
runs can't provide.
"""

from __future__ import annotations

import re

from autouser.study import IssueCluster, StudyResult

_WS_RE = re.compile(r"\s+")


def _cell(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().replace("|", "\\|")


def _cluster_row(rank: int, cluster: IssueCluster) -> str:
    personas = ", ".join(
        f"{name} {count}" for name, count in sorted(cluster.personas.items())
    )
    rate = f"{cluster.run_count}/{cluster.total_runs}"
    subtype = (
        " ".join(f"`{k}`×{v}" for k, v in sorted(cluster.feedback_subtypes.items()))
        or "—"
    )
    return (
        f"| {rank} | {rate} | {cluster.worst_severity.value} | "
        f"{cluster.category.value} | {_cell(cluster.url)} | {personas} | {subtype} |"
    )


def _clusters_section(result: StudyResult) -> str:
    if not result.clusters:
        return "*No issues recorded in any run.*"
    lines = [
        "## Findings ranked by reproduction",
        "",
        "| # | Runs affected | Worst | Category | Where | Personas (runs) | Divergence |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        _cluster_row(i, c) for i, c in enumerate(result.clusters, start=1)
    )
    lines.append("")
    for i, cluster in enumerate(result.clusters, start=1):
        if cluster.examples:
            lines.append(f"**#{i} in the personas' words:**")
            lines.extend(f"> {_cell(e)}" for e in cluster.examples)
            lines.append("")
    return "\n".join(lines).rstrip()


def _stats_section(result: StudyResult) -> str:
    lines = [
        "## Behavior by persona (across seeds)",
        "",
        "| Persona | Runs | Completed | With errors | Abandoned | Timed out | Crashed | Mean steps | Mean confidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in result.persona_stats:
        lines.append(
            f"| {s.persona_name} | {s.runs} | {s.completed} | {s.completed_with_errors} | "
            f"{s.abandoned} | {s.timed_out} | {s.crashed} | "
            f"{s.mean_steps:.1f} | {s.mean_confidence:.2f} |"
        )
    return "\n".join(lines)


def _runs_section(result: StudyResult) -> str:
    lines = [
        "## Run matrix",
        "",
        "| Persona | Seed | Sampled parameters | Outcome | Steps | Issues | Insights |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result.runs:
        outcome = r.outcome.value if r.outcome else f"CRASHED ({_cell(r.error or '')})"
        lines.append(
            f"| {r.persona_name} | {r.seed} | {_cell(r.persona_summary)} | "
            f"{outcome} | {r.total_steps} | {r.issue_count} | {r.insight_count} |"
        )
    return "\n".join(lines)


def render_study(result: StudyResult) -> str:
    n_personas = len({r.persona_name for r in result.runs})
    completed = sum(1 for r in result.runs if r.log is not None)
    header = [
        f"# Study — {result.task.rstrip('.')}.",
        "",
        f"**Success criteria:** {result.success_criteria}",
        f"**Start URL:** {result.start_url}",
        f"**Matrix:** {n_personas} persona(s) × {result.seeds_per_persona} seed(s) "
        f"= {len(result.runs)} runs ({completed} produced logs). "
        f"Seed 0 is the unjittered archetype.",
    ]
    return "\n".join(
        [
            "\n".join(header),
            "",
            _stats_section(result),
            "",
            _clusters_section(result),
            "",
            _runs_section(result),
        ]
    ) + "\n"
