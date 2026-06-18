"""False-positive audit (cognitive v2, slice 7): defend precision.

A persona-constrained model can flag friction that is really an artifact of
the simulation, not the product — "nothing happened" when the harness mis-
targeted, confusion at a state the real UI would have rendered, etc. The
audit runs a SECOND, unconstrained reviewer over the flagged issues and asks
of each: is this plausibly real product friction, or a simulation artifact?

This directly defends the M5 precision gate: issues the reviewer calls
artifacts are the false-positive candidates. It is deliberately independent
of the persona engine — a fresh, capable model with no persona constraints —
so it does not inherit the same blind spots.

The LLM call is behind an injectable ``judge`` seam (same pattern as the
ClaudeCodeProvider's runner), so the prompt builder, response parsing, and
aggregation are all unit-testable without a provider.
"""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable

from pydantic import BaseModel

from autouser.report.models import FrictionLog

logger = logging.getLogger(__name__)

# judge(system_prompt, user_prompt, schema) -> raw JSON string
Judge = Callable[[str, str, dict], Awaitable[str]]

_VERDICTS = ("real", "artifact", "uncertain")

_AUDIT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(_VERDICTS)},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "verdict", "confidence", "rationale"],
            },
        }
    },
    "required": ["verdicts"],
}

_AUDIT_SYSTEM = """\
You are a senior usability reviewer auditing findings produced by a simulated \
user. Your only job is to judge, for each finding, whether it describes \
PLAUSIBLE REAL friction with the product, or an ARTIFACT of the simulation \
(e.g. the simulated user mis-clicked, targeted a wrong element, or reacted to \
a state the real interface would not actually produce).

You are NOT role-playing any persona. Be a skeptical expert. A finding can be \
real even if the simulated user handled it clumsily — judge the PRODUCT issue, \
not the user's competence. Mark 'artifact' only when the finding most likely \
stems from the simulation itself rather than the product. Use 'uncertain' when \
the finding text doesn't give you enough to decide.

Respond with a single JSON object: a "verdicts" array with one entry per \
finding index, each {index, verdict (real|artifact|uncertain), confidence \
0..1, rationale (one sentence)}."""


class AuditVerdict(BaseModel):
    issue_index: int
    verdict: str
    confidence: float
    rationale: str


class AuditReport(BaseModel):
    verdicts: list[AuditVerdict]
    real_count: int
    artifact_count: int
    uncertain_count: int

    @property
    def audited_precision(self) -> float:
        """Reviewer's view of precision: real / judged findings."""
        judged = len(self.verdicts)
        return self.real_count / judged if judged else 0.0


def build_audit_prompt(log: FrictionLog) -> str:
    """One enumerated list of findings for the reviewer to classify."""
    lines = [
        f"Task the simulated user attempted: {log.task}",
        f"Outcome: {log.outcome.value}",
        "",
        "Findings to judge:",
    ]
    for idx, issue in enumerate(log.issues):
        lines.append(
            f"[{idx}] severity={issue.severity.value} category={issue.category.value} "
            f"where={issue.url}"
        )
        lines.append(f"     what: {issue.description}")
        if issue.persona_perspective:
            lines.append(f"     user's view: {issue.persona_perspective}")
    lines.append("")
    lines.append("Return a verdict for every index above.")
    return "\n".join(lines)


def _parse(raw: str, n_issues: int) -> list[AuditVerdict]:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    out: list[AuditVerdict] = []
    seen: set[int] = set()
    for entry in data.get("verdicts", []):
        idx = int(entry["index"])
        if idx < 0 or idx >= n_issues or idx in seen:
            continue  # ignore hallucinated / duplicate indices
        seen.add(idx)
        verdict = entry.get("verdict", "uncertain")
        if verdict not in _VERDICTS:
            verdict = "uncertain"
        out.append(
            AuditVerdict(
                issue_index=idx,
                verdict=verdict,
                confidence=float(entry.get("confidence", 0.5)),
                rationale=str(entry.get("rationale", "")),
            )
        )
    # Any index the reviewer skipped is recorded as uncertain — never silently
    # dropped, so counts always sum to len(issues).
    for idx in range(n_issues):
        if idx not in seen:
            out.append(
                AuditVerdict(
                    issue_index=idx, verdict="uncertain", confidence=0.0,
                    rationale="reviewer did not return a verdict for this finding",
                )
            )
    out.sort(key=lambda v: v.issue_index)
    return out


async def audit_log(log: FrictionLog, *, judge: Judge) -> AuditReport:
    """Run the false-positive audit over a friction log's issues.

    Empty-issue logs short-circuit (nothing to audit). ``judge`` is the
    injectable provider call; ``default_judge`` builds a real one.
    """
    if not log.issues:
        return AuditReport(verdicts=[], real_count=0, artifact_count=0, uncertain_count=0)

    raw = await judge(_AUDIT_SYSTEM, build_audit_prompt(log), _AUDIT_SCHEMA)
    verdicts = _parse(raw, len(log.issues))
    return AuditReport(
        verdicts=verdicts,
        real_count=sum(1 for v in verdicts if v.verdict == "real"),
        artifact_count=sum(1 for v in verdicts if v.verdict == "artifact"),
        uncertain_count=sum(1 for v in verdicts if v.verdict == "uncertain"),
    )


def default_judge(provider: str, model: str | None = None) -> Judge:
    """A judge backed by the configured provider.

    Reuses the engine's per-provider plumbing by borrowing a CognitiveEngine
    purely as a provider client (a neutral persona; the audit prompts are
    passed directly, the persona is never consulted). Keeps audit independent
    of persona construction while not duplicating retry/backoff/auth logic.
    """
    from autouser.cognitive.engine import CognitiveEngine
    from autouser.persona.models import Persona

    engine = CognitiveEngine(
        Persona(), "audit", "audit", provider=provider, model=model, temperature=0.2,
    )

    async def _judge(system_prompt: str, user_prompt: str, schema: dict) -> str:
        return await engine._call_llm(system_prompt, user_prompt, max_tokens=1024, schema=schema)

    return _judge
