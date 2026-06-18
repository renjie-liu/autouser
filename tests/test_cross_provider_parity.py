"""Cross-provider parity test (PR 2).

Proves a persona behaves identically whether it runs on Gemini or Sonnet
(Claude Code CLI). If the two providers diverge, the simulated user is not a
trustworthy testing instrument — so this test exists to attribute any observed
difference to *provider behavior*, and that attribution is only valid if both
providers have equal expressive capability going in.

This file lands in two layers:

* **Carry #1 — enum-parity precondition (active, pressure-safe).** The
  `assert_schema_covers_domain_enums()` guard runs with zero subprocesses and
  hard-fails *before* any live leg if a schema `enum` has drifted from its
  domain enum. Rationale (ux, first-principles): `--json-schema` is a hard CLI
  clamp, so a schema missing a member silently caps one provider's
  expressiveness ("a persona on Sonnet can never `give_up` because the schema
  dropped that token"). That is a capability gap masquerading as a behavioral
  difference — it corrupts the very comparison this test exists to make. The
  guard uses set **equality** (not subset) so drift in *either* direction
  fails: a schema missing a domain member, or carrying one that parses nowhere.

* **Carry #3 — 6-run acceptance matrix (scaffolded, opt-in/live).** The locked
  matrix is encoded as data and structurally validated here (pressure-safe).
  Its *live* execution spawns real `claude` subprocesses and so is gated behind
  `AUTOUSER_PARITY=1` AND the daemon-restart that clears the per-uid process
  ceiling (an upstream subprocess-pressure class). Sequential-first-run (carry #2) is honored so
  a subprocess-pressure crash isolates to one provider's leg.
"""

from __future__ import annotations

import os

import pytest

from pathlib import Path

from autouser.cognitive.engine import (
    _PLAN_SCHEMA,
    _REFLECT_SCHEMA,
    _VALID_ASSESSMENTS,
    CognitiveEngine,
)
from autouser.cognitive.models import (
    ActionType,
    Emotion,
    FeedbackSubtype,
    IssueCategory,
    Severity,
    SuccessPredicate,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --- Carry #1: enum-parity precondition -------------------------------------


def _schema_enum(schema: dict, prop: str) -> set:
    """The `enum` set for a property, with the `null` sentinel stripped.

    Severity/category schemas carry `None` for the "no mismatch" case while the
    domain enums have no such member, so the null-strip is on the schema side
    only — the right asymmetry (distinguished-eng / ux).
    """
    return set(schema["properties"][prop]["enum"]) - {None}


def assert_schema_covers_domain_enums() -> None:
    """Hard-fail unless every CLI schema `enum` equals its domain enum.

    A precondition, not a deferred check: a parity test whose two providers do
    not have equal expressive capability produces a number that *looks* like
    data but isn't. Refusing to run is strictly better than reporting an
    unverifiable result, so the live matrix below calls this at setup.

    Equality is load-bearing while all personas share one capability surface.
    If per-persona capability scoping is ever added (a deliberate fidelity
    feature), this guard must become per-persona rather than global — logged
    here so a future contributor sees why it is `==` today.
    """
    # action — 7 members incl. `give_up` and `back`
    assert {a.value for a in ActionType} == _schema_enum(_PLAN_SCHEMA, "action")

    # recognition / prediction / progress — the 4-value assessment scaffold.
    # set() wrap is defensive: _VALID_ASSESSMENTS is a set today, but a future
    # change to a tuple/list would make a bare `==` unconditionally False.
    for prop in ("recognition", "prediction", "progress"):
        assert set(_VALID_ASSESSMENTS) == _schema_enum(_PLAN_SCHEMA, prop)

    # emotion — 5 members, lowercase serialization
    assert {e.value for e in Emotion} == _schema_enum(_REFLECT_SCHEMA, "emotion")

    # severity — 3 members, lowercase; schema also carries null
    assert {s.value for s in Severity} == _schema_enum(_REFLECT_SCHEMA, "severity")

    # category — 7 members; schema also carries null
    assert {c.value for c in IssueCategory} == _schema_enum(
        _REFLECT_SCHEMA, "category"
    )


def test_schema_covers_domain_enums() -> None:
    """Carry #1, run on every CI invocation — pressure-safe, no subprocesses."""
    assert_schema_covers_domain_enums()


def test_precondition_uses_equality_not_subset() -> None:
    """A schema *missing* a domain member is the fidelity bug the guard exists
    to catch; a subset check would silently pass it. Pin equality so the guard
    can't be quietly weakened to `>=` later.
    """
    domain = {a.value for a in ActionType}
    schema = _schema_enum(_PLAN_SCHEMA, "action")
    assert domain == schema
    # If the schema dropped `give_up`, a subset check (domain >= schema) would
    # still pass while equality would not — that gap is the whole point.
    capped = schema - {"give_up"}
    assert domain >= capped  # subset would tolerate the cap...
    assert domain != capped  # ...equality refuses it.


def test_precondition_fails_on_dropped_schema_member() -> None:
    """Negative control — the guard must actually BITE when a schema enum drops
    a domain member, else the four positive tests above prove only that today's
    schema happens to match, not that the guard would catch drift.

    Mutate the schema list IN PLACE. Do NOT `monkeypatch.setattr(engine,
    "_PLAN_SCHEMA", ...)`: this test module did `from engine import _PLAN_SCHEMA`,
    so it holds its own binding. Rebinding the *engine* attribute leaves both
    the guard and this module reading the original object — the assertion would
    not raise and the negative control would be a false negative that proves
    nothing (distinguished-eng, s03 review). Mutating the list object in place
    changes the very object every binding points at.
    """
    action_enum = _PLAN_SCHEMA["properties"]["action"]["enum"]
    assert "give_up" in action_enum
    action_enum.remove("give_up")  # in-place: the object the guard reads
    try:
        with pytest.raises(AssertionError):
            assert_schema_covers_domain_enums()
    finally:
        action_enum.append("give_up")  # restore so sibling tests are unaffected
    # confirm restoration actually healed the guard
    assert_schema_covers_domain_enums()


# --- Carry #3: 6-run acceptance matrix ---------------------------------------
#
# Encoded as data so the locked shape is reviewable without a live run. Each
# leg names the persona, scenario, the provider it runs on, and the
# discriminator signal asserted — discriminators are the actual acceptance
# criteria, NOT terminal enums (a review gate: assert the signal, not the
# TaskOutcome/TerminalReason).
PARITY_MATRIX = (
    # (persona, scenario, provider, discriminator)
    ("maria", "locked_out", "gemini", "has_dwell_exit"),
    ("maria", "locked_out", "claude_code", "has_dwell_exit"),
    ("jake", "locked_out", "gemini", "steps_lt_maria_within_provider"),
    ("jake", "locked_out", "claude_code", "steps_lt_maria_within_provider"),
    ("maria", "misleading_checkout", "gemini", "has_false_success"),
    ("maria", "misleading_checkout", "claude_code", "has_false_success"),
)


def test_parity_matrix_is_well_formed() -> None:
    """Carry #3 structural check — pressure-safe. Validates the locked shape:
    3 scenarios x 2 providers = 6 legs, both providers present per scenario.
    """
    assert len(PARITY_MATRIX) == 6
    providers = {leg[2] for leg in PARITY_MATRIX}
    assert providers == {"gemini", "claude_code"}
    # every (persona, scenario) appears once per provider
    by_case: dict[tuple[str, str], set[str]] = {}
    for persona, scenario, provider, _disc in PARITY_MATRIX:
        by_case.setdefault((persona, scenario), set()).add(provider)
    assert len(by_case) == 3
    assert all(p == {"gemini", "claude_code"} for p in by_case.values())


_PARITY_LIVE = os.environ.get("AUTOUSER_PARITY") == "1"

pytestmark = []  # module imports stay pressure-safe; only the live test is gated


# Objective task spec per scenario. Each predicate is a deterministic gate the
# persona's *subjective belief* cannot satisfy on its own — that gap is exactly
# what the discriminators read. Specs are data so the locked shape is reviewable
# without a live run.
from autouser.report.models import TaskOutcome  # noqa: E402
from autouser.runner import SimulationRunner  # noqa: E402
from autouser.persona.models import Persona  # noqa: E402
from autouser.persona.registry import get_archetype  # noqa: E402
from autouser.session import TaskSpec  # noqa: E402

_COMPLETED_OUTCOMES = {TaskOutcome.COMPLETED, TaskOutcome.COMPLETED_WITH_ERRORS}


def _task_spec(scenario: str) -> TaskSpec:
    if scenario == "locked_out":
        # saucedemo's `locked_out_user` authenticates-attempts but is blocked;
        # a low-patience persona retries the same credentials → the dwell-loop
        # exit (runner emits an ERROR_RECOVERY "without progress" issue).
        return TaskSpec(
            task=(
                "Log in with username 'locked_out_user' and password "
                "'secret_sauce', then open the products page."
            ),
            success_criteria="You reach the products / inventory page.",
            start_url="https://www.saucedemo.com/",
            max_steps=8,
            success_predicate=SuccessPredicate(url_pattern=r"/inventory\.html"),
        )
    if scenario == "misleading_checkout":
        # The fixture only ever renders 'Thank you for your order!', never the
        # objective marker 'Order placed.' → predicate stays False while the
        # persona believes it succeeded. That divergence IS false_success.
        return TaskSpec(
            task="Confirm your order to complete the purchase.",
            success_criteria="The page objectively confirms with 'Order placed.'",
            start_url=(_FIXTURES / "misleading_checkout.html").as_uri(),
            max_steps=5,
            success_predicate=SuccessPredicate(text_contains="Order placed."),
        )
    raise ValueError(f"unknown scenario {scenario!r}")


def _discriminator_holds(name, log, *, peers) -> bool:
    """Assert the *signal*, not the terminal enum (a review gate)."""
    if name == "has_dwell_exit":
        # The dwell branch is the only emitter of this issue; the frustration
        # give-up branch finalizes ABANDONED *without* emitting (distinguished-
        # eng's runner.py:217 bug), so checking for the issue — not bare
        # ABANDONED — is what discriminates a genuine dwell exit.
        return any(
            i.category == IssueCategory.ERROR_RECOVERY
            and "without progress" in i.description.lower()
            for i in log.issues
        )
    if name == "has_false_success":
        has_issue = any(
            i.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS for i in log.issues
        )
        final = log.steps[-1] if log.steps else None
        believed = bool(final is not None and final.persona_believes_complete)
        return (has_issue or believed) and log.outcome not in _COMPLETED_OUTCOMES
    if name == "steps_lt_maria_within_provider":
        maria = peers.get("maria")  # same-provider maria locked_out leg
        return maria is not None and log.total_steps < maria.total_steps
    raise ValueError(f"unknown discriminator {name!r}")


@pytest.mark.skipif(
    not _PARITY_LIVE,
    reason=(
        "opt-in live parity run: set AUTOUSER_PARITY=1 (and "
        "AUTOUSER_BROWSER_EXECUTABLE to a launchable chromium). Legs run "
        "sequentially in ONE process/event loop so every ClaudeCodeProvider "
        "shares the module-global spawn gate (concurrent `claude` ≤ 1) — the "
        "single-process invariant the s08 review gates on."
    ),
)
@pytest.mark.asyncio
async def test_cross_provider_parity_live(tmp_path) -> None:
    """Carry #3 live execution — the 6 legs through the real engine run-loop.

    Sequential-first (carry #2): one process, one event loop. Because the
    `claude` spawn gate (`_spawn_gate`, default 1) is module-global, every leg's
    freshly-constructed ClaudeCodeProvider shares it, so autouser's concurrent
    `claude` count never exceeds 1 even across legs. Process-per-leg would each
    re-init the gate to 1 → up to 6 concurrent spawns → re-graze the ceiling;
    that is the trap, and keeping all legs in this single process is the fix.
    """
    # Carry #1 precondition: refuse to run if the providers are not capability-
    # equal — a divergence number is uninterpretable otherwise.
    assert_schema_covers_domain_enums()

    # (persona, scenario) -> FrictionLog, keyed PER provider so jake's
    # steps_lt_maria comparison stays within-provider.
    by_provider: dict[str, dict[str, object]] = {"gemini": {}, "claude_code": {}}
    summary: list[str] = []
    failures: list[str] = []

    for persona_name, scenario, provider, discriminator in PARITY_MATRIX:
        persona = Persona.from_archetype(get_archetype(persona_name))
        spec = _task_spec(scenario)
        engine = CognitiveEngine(
            persona, spec.task, spec.success_criteria, provider=provider
        )
        runner = SimulationRunner(
            persona=persona,
            task_spec=spec,
            screenshot_dir=tmp_path / f"{provider}__{persona_name}__{scenario}",
            engine=engine,
        )
        log = await runner.run()
        by_provider[provider][persona_name] = log

        held = _discriminator_holds(
            discriminator, log, peers=by_provider[provider]
        )
        line = (
            f"[{provider:11s}] {persona_name:5s} / {scenario:19s} "
            f"outcome={log.outcome.value:18s} steps={log.total_steps:2d} "
            f"issues={len(log.issues):2d} {discriminator}={'HOLD' if held else 'MISS'}"
        )
        summary.append(line)
        if not held:
            failures.append(line)

    report = "\n".join(summary)
    print("\n=== cross-provider parity matrix (6 legs) ===\n" + report)

    # Each leg's discriminator is the acceptance criterion. A MISS is real data
    # (characterized divergence for s10), surfaced per-leg with provider/persona/
    # fixture/outcome — never swallowed into an opaque pass.
    assert not failures, (
        "parity discriminator(s) did not hold:\n" + "\n".join(failures)
        + "\n--- full matrix ---\n" + report
    )
