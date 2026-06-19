"""Command-line interface: run simulations without writing Python.

Usage:
    autouser run --url https://example.com/signup \
        --task "Create a new account" \
        --criteria "User reaches the dashboard after signup" \
        --persona maria --persona jake
    autouser personas

Outputs, per persona, under --out/<run_id>/<persona>/:
    report.md          triage-ready friction report (docs/m5-report-spec.md)
    journey.md         experience trajectory: per-step thoughts, expectations,
                       emotions, notes, and the persona's final mental model
    friction_log.json  full machine-readable FrictionLog (the raw trajectory)
    step_NNN.png       per-step screenshots

Exit codes:
    0  every persona simulation produced a friction log (task outcome may
       still be "abandoned" — that is a finding, not a tool failure)
    1  one or more persona simulations crashed
    2  configuration error (bad arguments, no usable LLM provider)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

from autouser.cognitive.engine import CognitiveEngine, _DEFAULT_MODELS
from autouser.cognitive.models import SuccessPredicate
from autouser.persona.models import Persona
from autouser.persona.registry import ARCHETYPES, get_archetype
from autouser.report.human_renderer import render_report
from autouser.report.journey_renderer import render_journey
from autouser.report.models import FrictionLog, PersonaError
from autouser.runner import PerceptionMode, SimulationRunner
from autouser.session import TaskSpec

logger = logging.getLogger(__name__)

_PROVIDERS = sorted(_DEFAULT_MODELS)


class CliConfigError(Exception):
    """Configuration problem the operator must fix (exit code 2)."""


def resolve_provider(explicit: str | None, *, env: dict[str, str] | None = None) -> str:
    """Resolve the LLM provider for a CLI run.

    Precedence: --provider flag > AUTOUSER_PROVIDER env > API-key detection
    (Anthropic, then Gemini) > `claude` binary on PATH (Claude Code CLI,
    no API key needed). Unlike the engine's default, the CLI refuses to
    fall back to a provider that is guaranteed to fail at call time —
    a clear preflight error beats a mid-run stack trace.
    """
    env = os.environ if env is None else env

    if explicit:
        if explicit not in _PROVIDERS:
            raise CliConfigError(
                f"unknown provider {explicit!r}; choose from {', '.join(_PROVIDERS)}"
            )
        return explicit

    env_provider = env.get("AUTOUSER_PROVIDER", "").strip()
    if env_provider:
        if env_provider not in _PROVIDERS:
            raise CliConfigError(
                f"AUTOUSER_PROVIDER={env_provider!r} is not a known provider; "
                f"choose from {', '.join(_PROVIDERS)}"
            )
        return env_provider

    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("GEMINI_API_KEY"):
        return "gemini"
    if shutil.which("claude"):
        logger.info(
            "no API keys found; using the local `claude` CLI (provider=claude_code)"
        )
        return "claude_code"

    raise CliConfigError(
        "no usable LLM provider: set ANTHROPIC_API_KEY or GEMINI_API_KEY, "
        "install + authenticate the `claude` CLI, or pass --provider explicitly"
    )


def build_task_spec(args: argparse.Namespace) -> TaskSpec:
    """Translate CLI arguments into a TaskSpec, including the optional predicate."""
    predicate = None
    if args.success_text or args.success_url_pattern or args.success_selector:
        predicate = SuccessPredicate(
            text_contains=args.success_text,
            url_pattern=args.success_url_pattern,
            selector_present=args.success_selector,
        )
    return TaskSpec(
        task=args.task,
        success_criteria=args.criteria,
        start_url=args.url,
        max_steps=args.max_steps,
        success_predicate=predicate,
        max_consecutive_failures=args.max_consecutive_failures,
    )


def _resolve_personas(names: list[str]) -> list[Persona]:
    personas = []
    for name in names:
        try:
            personas.append(Persona.from_archetype(get_archetype(name)))
        except KeyError:
            raise CliConfigError(
                f"unknown persona {name!r}; choose from {', '.join(sorted(ARCHETYPES))}"
            ) from None
    return personas


async def _run_one(
    persona: Persona,
    task_spec: TaskSpec,
    *,
    provider: str,
    model: str | None,
    perception: PerceptionMode,
    out_dir: Path,
    fast_mode: bool = False,
) -> FrictionLog:
    engine = None
    if perception == PerceptionMode.DOM:
        engine = CognitiveEngine(
            persona, task_spec.task, task_spec.success_criteria,
            provider=provider, model=model,
            exploration=task_spec.exploration,
            fast_mode=fast_mode,
        )
    elif perception == PerceptionMode.COMPUTER_USE:
        # Build the vision engine here too, so --model is honored rather than
        # falling back to ComputerUseEngine's default when SimulationRunner
        # constructs it.
        from autouser.cognitive.computer_use import ComputerUseEngine
        engine = ComputerUseEngine(
            persona, task_spec.task, task_spec.success_criteria, model=model,
        )
    runner = SimulationRunner(
        persona=persona,
        task_spec=task_spec,
        screenshot_dir=out_dir,
        perception_mode=perception,
        engine=engine,
    )
    return await runner.run()


async def _run_command(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    if args.perception == PerceptionMode.COMPUTER_USE and provider != "anthropic":
        raise CliConfigError(
            "--perception computer_use requires the anthropic provider "
            "(vision API); no other provider is supported"
        )
    personas = _resolve_personas(args.persona or ["maria"])
    task_spec = build_task_spec(args)
    return await _execute_runs(args, provider, personas, task_spec)


# Canonical exploration framing — rendered into reports/logs so a reader of
# the artifacts sees what the persona was actually doing.
_EXPLORE_TASK = (
    "Get to know this product: what it is, what you can do here, and whether "
    "it's for you."
)
_EXPLORE_CRITERIA = (
    "You feel you've seen enough to say what this product is and whether "
    "you'd use it."
)


async def _explore_command(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    personas = _resolve_personas(args.persona or ["maria"])
    task_spec = TaskSpec(
        task=_EXPLORE_TASK,
        success_criteria=_EXPLORE_CRITERIA,
        start_url=args.url,
        max_steps=args.max_steps,
        exploration=True,
    )
    args.perception = PerceptionMode.DOM.value  # exploration is DOM-mode only for now
    return await _execute_runs(args, provider, personas, task_spec)


async def _execute_runs(
    args: argparse.Namespace,
    provider: str,
    personas: list[Persona],
    task_spec: TaskSpec,
) -> int:
    run_id = uuid.uuid4().hex[:12]
    run_dir = Path(args.out) / run_id
    print(f"provider={provider} run_id={run_id} out={run_dir}", file=sys.stderr)

    coros = []
    for persona in personas:
        name = persona.archetype.name if persona.archetype else "custom"
        coros.append(
            _run_one(
                persona, task_spec,
                provider=provider, model=args.model,
                perception=args.perception, out_dir=run_dir / name,
                fast_mode=getattr(args, "fast", False),
            )
        )
    results = await asyncio.gather(*coros, return_exceptions=True)

    errors: list[PersonaError] = []
    for persona, result in zip(personas, results):
        name = persona.archetype.name if persona.archetype else "custom"
        persona_dir = run_dir / name
        if isinstance(result, BaseException):
            errors.append(
                PersonaError(
                    persona_name=name,
                    error_type=type(result).__name__,
                    message=str(result),
                )
            )
            print(f"\n=== {name}: ERROR ({type(result).__name__}) ===", file=sys.stderr)
            print(str(result), file=sys.stderr)
            continue

        persona_dir.mkdir(parents=True, exist_ok=True)
        report = render_report(result, persona)
        (persona_dir / "report.md").write_text(report, encoding="utf-8")
        (persona_dir / "journey.md").write_text(
            render_journey(result, persona), encoding="utf-8"
        )
        (persona_dir / "friction_log.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        audit_note = ""
        if getattr(args, "audit", False):
            from autouser.audit import audit_log, default_judge

            audit = await audit_log(result, judge=default_judge(provider, args.model))
            (persona_dir / "audit.json").write_text(
                audit.model_dump_json(indent=2), encoding="utf-8"
            )
            audit_note = (
                f" audit(real={audit.real_count} artifact={audit.artifact_count} "
                f"uncertain={audit.uncertain_count})"
            )
        if not args.quiet:
            print(f"\n{report}\n")
        print(
            f"[{name}] outcome={result.outcome.value} steps={result.total_steps} "
            f"issues={len(result.issues)}{audit_note} -> {persona_dir / 'report.md'}",
            file=sys.stderr,
        )

    if errors:
        summary = json.dumps([e.model_dump() for e in errors], indent=2)
        print(f"\n{len(errors)} persona(s) crashed:\n{summary}", file=sys.stderr)
        return 1
    return 0


async def _study_command(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    if args.perception == PerceptionMode.COMPUTER_USE and provider != "anthropic":
        raise CliConfigError(
            "--perception computer_use requires the anthropic provider "
            "(vision API); no other provider is supported"
        )
    if args.seeds < 1:
        raise CliConfigError("--seeds must be at least 1")

    names = args.persona or ["maria"]
    _resolve_personas(names)  # validate names early; study samples its own variants
    archetypes = [get_archetype(name) for name in names]
    task_spec = build_task_spec(args)

    from autouser.report.study_renderer import render_study
    from autouser.study import run_study

    def engine_factory(persona):
        # Build the right engine per perception mode, honoring --model in both
        # (a None factory would let SimulationRunner construct a default-model
        # ComputerUseEngine, silently ignoring --model).
        if args.perception == PerceptionMode.COMPUTER_USE:
            from autouser.cognitive.computer_use import ComputerUseEngine
            return ComputerUseEngine(
                persona, task_spec.task, task_spec.success_criteria, model=args.model,
            )
        return CognitiveEngine(
            persona, task_spec.task, task_spec.success_criteria,
            provider=provider, model=args.model,
            fast_mode=getattr(args, "fast", False),
        )

    audit_judge = None
    if getattr(args, "audit", False):
        from autouser.audit import default_judge
        audit_judge = default_judge(provider, args.model)

    study_id = uuid.uuid4().hex[:12]
    study_dir = Path(args.out) / study_id
    total = len(archetypes) * args.seeds
    print(
        f"provider={provider} study_id={study_id} matrix={len(archetypes)}x{args.seeds}"
        f"={total} runs out={study_dir}",
        file=sys.stderr,
    )

    result = await run_study(
        archetypes, task_spec,
        seeds_per_persona=args.seeds,
        out_dir=study_dir,
        perception_mode=args.perception,
        engine_factory=engine_factory,
        audit_judge=audit_judge,
    )

    study_dir.mkdir(parents=True, exist_ok=True)
    synthesis = render_study(result)
    (study_dir / "study.md").write_text(synthesis, encoding="utf-8")
    (study_dir / "study.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    if not args.quiet:
        print(f"\n{synthesis}\n")

    crashed = [r for r in result.runs if r.error]
    for record in result.runs:
        status = record.outcome.value if record.outcome else f"ERROR: {record.error}"
        print(
            f"[{record.persona_name}-s{record.seed}] {status} steps={record.total_steps} "
            f"issues={record.issue_count}",
            file=sys.stderr,
        )
    print(f"study report: {study_dir / 'study.md'}", file=sys.stderr)
    return 1 if crashed else 0


def _score_command(args: argparse.Namespace) -> int:
    """Deterministically score a run or study against a planted-issue
    benchmark — no LLM, no browser. Exit 1 only when --min-precision /
    --min-recall thresholds (the M5 gate as a CI check) are set and unmet."""
    from autouser.benchmark import Benchmark, score_logs
    from autouser.report.models import FrictionLog
    from autouser.study import StudyResult

    benchmark = Benchmark.from_yaml(args.benchmark)
    if bool(args.log) == bool(args.study):
        raise CliConfigError("pass exactly one of --log or --study")

    if args.log:
        logs = [FrictionLog.model_validate_json(Path(args.log).read_text())]
    else:
        study = StudyResult.model_validate_json(Path(args.study).read_text())
        logs = [r.log for r in study.runs if r.log is not None]
        if not logs:
            raise CliConfigError("study has no completed runs to score")

    score = score_logs(logs, benchmark)
    print(f"benchmark: {benchmark.name}  ({score.total_ground_truth} planted issues)")
    print(f"runs scored: {score.runs_scored}")
    print(f"mean precision: {score.mean_precision:.0%}")
    print(
        f"union recall:   {score.union_recall:.0%} "
        f"({score.union_matched_ground_truth}/{score.total_ground_truth} planted issues found)"
    )
    found = sorted({m.ground_truth_id for r in score.per_run for m in r.matches})
    missed = sorted(set(g.id for g in benchmark.issues) - set(found))
    if found:
        print(f"found:  {', '.join(found)}")
    if missed:
        print(f"missed: {', '.join(missed)}")

    failed = False
    if args.min_precision is not None and score.mean_precision < args.min_precision:
        print(
            f"FAIL precision {score.mean_precision:.0%} < {args.min_precision:.0%}",
            file=sys.stderr,
        )
        failed = True
    if args.min_recall is not None and score.union_recall < args.min_recall:
        print(
            f"FAIL recall {score.union_recall:.0%} < {args.min_recall:.0%}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


def _personas_command() -> int:
    for name in sorted(ARCHETYPES):
        archetype = ARCHETYPES[name]
        print(f"{name:8s} {archetype.narrative}")
    return 0


def _add_task_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by `run` and `study`."""
    parser.add_argument("--url", required=True, help="start URL (http(s):// or file://)")
    parser.add_argument("--task", required=True, help="what the user is trying to do")
    parser.add_argument(
        "--criteria", required=True,
        help="how the persona knows it succeeded, in user language",
    )
    parser.add_argument(
        "--persona", action="append", default=None, metavar="NAME",
        help=f"persona archetype, repeatable (default: maria; "
        f"available: {', '.join(sorted(ARCHETYPES))})",
    )
    parser.add_argument(
        "--provider", default=None, choices=_PROVIDERS,
        help="LLM provider (default: auto-detect — API keys, then `claude` CLI)",
    )
    parser.add_argument("--model", default=None, help="override the provider's default model")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--success-text", default=None,
        help="objective success: page text that must be visible",
    )
    parser.add_argument(
        "--success-url-pattern", default=None,
        help="objective success: regex the final URL must match",
    )
    parser.add_argument(
        "--success-selector", default=None,
        help="objective success: CSS selector that must be present",
    )
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="identical-failure steps before the run is abandoned (default 3)",
    )
    parser.add_argument(
        "--perception", type=str, default=PerceptionMode.DOM.value,
        choices=[m.value for m in PerceptionMode],
        help="dom (default) or computer_use (anthropic provider only)",
    )
    parser.add_argument(
        "--out", default="./autouser_runs",
        help="output directory for reports, logs, screenshots",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="don't print the rendered report to stdout",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="run the false-positive audit (an unconstrained reviewer judges "
        "each finding real vs simulation-artifact); writes audit.json per persona",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="dual-process mode: skip the deliberate LLM reflect on cruising "
        "steps (no surprise, no plausible completion) — ~halves LLM calls on "
        "smooth runs. Best with an objective success predicate.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autouser",
        description="Simulate persona-driven users against a web app and "
        "produce friction logs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation against a URL")
    _add_task_args(run)

    study = sub.add_parser(
        "study",
        help="run M personas × N sampled seeds and aggregate findings by "
        "reproduction frequency",
    )
    _add_task_args(study)
    study.add_argument(
        "--seeds", type=int, default=3,
        help="seeds per persona (default 3; seed 0 is the unjittered archetype)",
    )

    explore = sub.add_parser(
        "explore",
        help="open-ended first-session exploration: no task — the persona "
        "pokes around and reports what they think the product is",
    )
    explore.add_argument("--url", required=True, help="start URL (http(s):// or file://)")
    explore.add_argument(
        "--persona", action="append", default=None, metavar="NAME",
        help=f"persona archetype, repeatable (default: maria; "
        f"available: {', '.join(sorted(ARCHETYPES))})",
    )
    explore.add_argument(
        "--provider", default=None, choices=_PROVIDERS,
        help="LLM provider (default: auto-detect — API keys, then `claude` CLI)",
    )
    explore.add_argument("--model", default=None, help="override the provider's default model")
    explore.add_argument(
        "--max-steps", type=int, default=25,
        help="hard step ceiling (default 25; the session usually ends earlier "
        "via 'seen enough', frustration, or the patience time budget)",
    )
    explore.add_argument(
        "--out", default="./autouser_runs",
        help="output directory for reports, logs, screenshots",
    )
    explore.add_argument(
        "--quiet", action="store_true",
        help="don't print the rendered report to stdout",
    )

    score = sub.add_parser(
        "score",
        help="score a run/study against a planted-issue benchmark "
        "(precision/recall; deterministic, no LLM)",
    )
    score.add_argument("--benchmark", required=True, help="benchmark YAML path")
    score.add_argument("--log", default=None, help="a friction_log.json to score")
    score.add_argument("--study", default=None, help="a study.json to score (union recall)")
    score.add_argument(
        "--min-precision", type=float, default=None,
        help="exit nonzero if mean precision is below this (0..1) — M5 gate as CI check",
    )
    score.add_argument(
        "--min-recall", type=float, default=None,
        help="exit nonzero if union recall is below this (0..1)",
    )

    sub.add_parser("personas", help="list available persona archetypes")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("AUTOUSER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = build_parser().parse_args(argv)
    try:
        if args.command == "personas":
            return _personas_command()
        if args.command == "score":
            return _score_command(args)
        if args.command == "study":
            return asyncio.run(_study_command(args))
        if args.command == "explore":
            return asyncio.run(_explore_command(args))
        return asyncio.run(_run_command(args))
    except CliConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
