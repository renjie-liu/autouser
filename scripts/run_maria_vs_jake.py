#!/usr/bin/env python3
"""Maria vs Jake spike experiment.

Runs both personas against the same task on the same site,
then prints raw MetricVector comparison.

Usage:
    python scripts/run_maria_vs_jake.py

Requires: playwright browsers installed (python -m playwright install chromium)
Reads GEMINI_API_KEY or ANTHROPIC_API_KEY from .env or environment.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from autouser.metrics import compare_metrics, extract_metrics
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.runner import SimulationRunner
from autouser.session import SessionLog, TaskSpec


def _load_env_file() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


# Enable debug logging to capture scaffold fields and constraint violations
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("maria_vs_jake_debug.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# --- Experiment config ---
# Using a real e-commerce demo site with forms, navigation, and multi-step flows.
# The Sauce Demo site has login, product listing, cart, and checkout — enough
# interactive complexity to differentiate personas.
TASK_SPEC = TaskSpec(
    task="Log in with the username 'standard_user' and password 'secret_sauce', "
         "then add the cheapest item to your cart and begin checkout.",
    success_criteria="You see the checkout form asking for your first name.",
    start_url="https://www.saucedemo.com/",
    max_steps=15,
    give_up_threshold=10,
)

SCREENSHOT_DIR = Path("./screenshots/maria_vs_jake")


async def run_persona(name: str) -> SessionLog:
    """Run a single persona through the task and return the session log."""
    archetype = get_archetype(name)
    persona = Persona.from_archetype(archetype)
    persona_dir = SCREENSHOT_DIR / name

    runner = SimulationRunner(
        persona=persona,
        task_spec=TASK_SPEC,
        screenshot_dir=persona_dir,
    )

    logger.info("=== Starting %s simulation ===", name.upper())
    friction_log = await runner.run()
    logger.info(
        "=== %s finished: %s in %d steps ===",
        name.upper(),
        friction_log.outcome.value,
        friction_log.total_steps,
    )

    return runner.session


async def main():
    _load_env_file()
    print("\n" + "=" * 60)
    print("MARIA vs JAKE — Spike v1 Experiment")
    print("=" * 60)
    print(f"Task: {TASK_SPEC.task}")
    print(f"Success: {TASK_SPEC.success_criteria}")
    print(f"URL: {TASK_SPEC.start_url}")
    print(f"Max steps: {TASK_SPEC.max_steps}")
    print("=" * 60 + "\n")

    # Run sequentially to avoid port conflicts and keep logs readable
    maria_session = await run_persona("maria")
    jake_session = await run_persona("jake")

    # Extract metrics
    maria_metrics = extract_metrics(maria_session)
    jake_metrics = extract_metrics(jake_session)

    # Print raw comparison
    comparison = compare_metrics([maria_metrics, jake_metrics])

    print("\n" + "=" * 60)
    print("RAW METRIC VECTORS")
    print("=" * 60)
    print(comparison)
    print("=" * 60)

    # Print detailed per-persona breakdown
    for name, metrics in [("MARIA", maria_metrics), ("JAKE", jake_metrics)]:
        print(f"\n--- {name} ---")
        print(f"  Tech literacy: {metrics.tech_literacy}")
        print(f"  Patience: {metrics.patience}")
        print(f"  Goal clarity: {metrics.goal_clarity}")
        print(f"  Confidence: mean={metrics.confidence.mean:.3f} "
              f"stdev={metrics.confidence.stdev:.3f} "
              f"min={metrics.confidence.min:.3f} max={metrics.confidence.max:.3f}")
        print(f"  Exploration: {metrics.unique_targets} unique targets, "
              f"{metrics.unique_urls} unique URLs")
        print(f"  Non-advancing: {metrics.non_advancing_count}/{metrics.total_steps} "
              f"({metrics.non_advancing_rate:.2%})")
        print(f"  Abandoned: {metrics.abandoned} "
              f"(steps: {metrics.steps_before_abandonment}, "
              f"reason: {metrics.terminal_reason})")

    print("\nDebug log written to: maria_vs_jake_debug.log")
    print("Screenshots in: ./screenshots/maria_vs_jake/")


if __name__ == "__main__":
    asyncio.run(main())
