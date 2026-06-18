#!/usr/bin/env python3
"""M3 Saucedemo validation matrix.

Runs 4 configs against saucedemo.com with predicate-gated termination,
then prints structured results for M3 scoring.

Matrix:
  A: Maria × standard_user  (baseline — should complete cleanly)
  B: Maria × problem_user   (expect friction from broken images/sort)
  C: Jake  × problem_user   (same site, different persona — differentiation test)
  D: Maria × error_user     (locked out at login — frustration/abandonment path)

Usage:
    python scripts/validate_saucedemo_m3.py

Requires: GEMINI_API_KEY or ANTHROPIC_API_KEY in environment or .env file.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from autouser.cognitive.models import (
    CompletionSource,
    FeedbackSubtype,
    SuccessPredicate,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.report.generator import ReportGenerator
from autouser.runner import SimulationRunner
from autouser.session import TaskSpec

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("saucedemo_m3_debug.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


CHECKOUT_PREDICATE = SuccessPredicate(url_pattern=r"/checkout-complete\.html")

RUNS = [
    {
        "name": "A",
        "persona": "maria",
        "label": "Maria × standard_user",
        "task": (
            "Log in with username 'standard_user' and password 'secret_sauce'. "
            "Add any item to your cart. Go to checkout. Fill in your information "
            "(any name/zip is fine). Complete the purchase."
        ),
        "credentials": "standard_user / secret_sauce",
    },
    {
        "name": "B",
        "persona": "maria",
        "label": "Maria × problem_user",
        "task": (
            "Log in with username 'problem_user' and password 'secret_sauce'. "
            "Add any item to your cart. Go to checkout. Fill in your information "
            "(any name/zip is fine). Complete the purchase."
        ),
        "credentials": "problem_user / secret_sauce",
    },
    {
        "name": "C",
        "persona": "jake",
        "label": "Jake × problem_user",
        "task": (
            "Log in with username 'problem_user' and password 'secret_sauce'. "
            "Add any item to your cart. Go to checkout. Fill in your information "
            "(any name/zip is fine). Complete the purchase."
        ),
        "credentials": "problem_user / secret_sauce",
    },
    {
        "name": "D",
        "persona": "maria",
        "label": "Maria × error_user",
        "task": (
            "Log in with username 'locked_out_user' and password 'secret_sauce'. "
            "Add any item to your cart. Go to checkout. Fill in your information "
            "(any name/zip is fine). Complete the purchase."
        ),
        "credentials": "locked_out_user / secret_sauce",
    },
]

SCREENSHOT_DIR = Path("./screenshots/saucedemo_m3")


async def run_one(run_config: dict) -> dict:
    """Execute a single validation run and return structured results."""
    name = run_config["name"]
    persona_name = run_config["persona"]
    label = run_config["label"]

    task_spec = TaskSpec(
        task=run_config["task"],
        success_criteria="You reach the checkout-complete page confirming your order.",
        start_url="https://www.saucedemo.com/",
        max_steps=20,
        success_predicate=CHECKOUT_PREDICATE,
    )

    archetype = get_archetype(persona_name)
    persona = Persona.from_archetype(archetype)
    persona_dir = SCREENSHOT_DIR / f"run_{name}_{persona_name}"

    runner = SimulationRunner(
        persona=persona,
        task_spec=task_spec,
        screenshot_dir=persona_dir,
    )

    logger.info("=== Starting Run %s: %s ===", name, label)
    await runner.run()
    session = runner.session

    generator = ReportGenerator()
    report = generator.generate(
        steps=session.steps,
        persona=persona,
        task=task_spec.task,
        success_criteria=task_spec.success_criteria,
        terminal_reason=session.terminal_reason,
    )

    all_friction = [fi for s in session.steps for fi in s.friction_issues]
    divergence_findings = [
        fi for fi in all_friction
        if fi.feedback_subtype in (FeedbackSubtype.FALSE_SUCCESS, FeedbackSubtype.SILENT_SUCCESS)
    ]

    completion_sources = [s.completion_source.value for s in session.steps if s.completion_source != CompletionSource.NONE]
    final_source = completion_sources[-1] if completion_sources else "NONE"

    result = {
        "run": name,
        "label": label,
        "persona": persona_name,
        "outcome": report.outcome.value,
        "terminal_reason": session.terminal_reason.value if session.terminal_reason else None,
        "total_steps": report.total_steps,
        "completion_source": final_source,
        "issue_count": len(report.issues),
        "issues": [
            {
                "severity": i.severity.value,
                "category": i.category.value if hasattr(i.category, "value") else str(i.category),
                "description": i.description[:120],
                "step": i.step,
            }
            for i in report.issues
        ],
        "divergence_findings": [
            {
                "subtype": fi.feedback_subtype.value,
                "description": fi.description[:120],
            }
            for fi in divergence_findings
        ],
        "friction_categories": list({
            i.category.value if hasattr(i.category, "value") else str(i.category)
            for i in report.issues
        }),
        "friction_severities": list({i.severity.value for i in report.issues}),
    }

    logger.info(
        "=== Run %s finished: %s, %d steps, %d issues, source=%s ===",
        name, report.outcome.value, report.total_steps, len(report.issues), final_source,
    )
    return result


async def main():
    started = datetime.now(timezone.utc)
    print("\n" + "=" * 70)
    print("SAUCEDEMO M3 VALIDATION MATRIX")
    print(f"Started: {started.isoformat()}")
    print("=" * 70)

    results = []
    for run_config in RUNS:
        result = await run_one(run_config)
        results.append(result)
        print(f"\n  Run {result['run']} ({result['label']}): "
              f"{result['outcome']} in {result['total_steps']} steps, "
              f"{result['issue_count']} issues, source={result['completion_source']}")

    finished = datetime.now(timezone.utc)
    elapsed = (finished - started).total_seconds()

    print("\n" + "=" * 70)
    print("M3 SCORING SUMMARY")
    print("=" * 70)

    # Dimension 1: Persona differentiation (Run B vs Run C)
    run_b = next(r for r in results if r["run"] == "B")
    run_c = next(r for r in results if r["run"] == "C")
    diff_count = abs(run_b["issue_count"] - run_c["issue_count"])
    diff_categories = set(run_b["friction_categories"]) != set(run_c["friction_categories"])
    diff_severities = set(run_b["friction_severities"]) != set(run_c["friction_severities"])
    dim1_pass = diff_count >= 1 or diff_categories or diff_severities

    print("\n  Dimension 1 — Persona Differentiation (Run B vs C):")
    print(f"    Maria issues: {run_b['issue_count']}, categories: {run_b['friction_categories']}")
    print(f"    Jake issues:  {run_c['issue_count']}, categories: {run_c['friction_categories']}")
    print(f"    Delta: {diff_count} issues, category diff={diff_categories}, severity diff={diff_severities}")
    print(f"    VERDICT: {'PASS' if dim1_pass else 'FAIL'}")

    # Dimension 2: Divergence detection (any FALSE_SUCCESS or SILENT_SUCCESS across all runs)
    all_divergences = [d for r in results for d in r["divergence_findings"]]
    dim2_pass = len(all_divergences) > 0

    print("\n  Dimension 2 — Divergence Detection:")
    print(f"    Total divergence findings: {len(all_divergences)}")
    for d in all_divergences:
        print(f"      - {d['subtype']}: {d['description'][:80]}")
    print(f"    VERDICT: {'PASS' if dim2_pass else 'FAIL'}")

    # Dimension 3: Completion accuracy (zero false predicate exits)
    run_a = next(r for r in results if r["run"] == "A")
    predicate_exits = [r for r in results if r["completion_source"] == "predicate"]
    false_exits = [r for r in predicate_exits if r["outcome"] not in ("completed", "completed_with_errors")]
    dim3_pass = len(false_exits) == 0

    print("\n  Dimension 3 — Completion Accuracy:")
    print(f"    Run A (standard_user): {run_a['outcome']}, source={run_a['completion_source']}")
    print(f"    Predicate exits: {len(predicate_exits)}")
    print(f"    False predicate exits: {len(false_exits)}")
    print(f"    VERDICT: {'PASS' if dim3_pass else 'FAIL'}")

    overall = dim1_pass and dim2_pass and dim3_pass
    print(f"\n  {'=' * 50}")
    print(f"  M3 OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("  Provider: Gemini (baseline)")
    print(f"  {'=' * 50}")

    # Dump full results to JSON
    output = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "elapsed_seconds": elapsed,
        "provider": "gemini",
        "runs": results,
        "m3_scoring": {
            "dim1_persona_differentiation": dim1_pass,
            "dim2_divergence_detection": dim2_pass,
            "dim3_completion_accuracy": dim3_pass,
            "overall": overall,
        },
    }
    output_path = SCREENSHOT_DIR / "m3_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Full results: {output_path}")
    print("  Debug log: saucedemo_m3_debug.log")


if __name__ == "__main__":
    _load_env_file()
    asyncio.run(main())
