"""ScriptedEngine + SimulationRunner DI seam — slice 2 of the M5 fixture harness.

Verifies:
1. ScriptedEngine satisfies EngineProtocol (structural drift guard).
2. plan() returns intents in order and falls through to GIVE_UP after exhaustion.
3. should_give_up() honors the give_up_after threshold.
4. Happy-path scenario: a ScriptedEngine drives SimulationRunner end-to-end
   against a local fixture page, with a SuccessPredicate that the scripted
   sequence ends up satisfying. Asserts TerminalReason.SUCCESS, no LLM calls,
   exact step count.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    SuccessPredicate,
    UIState,
)
from autouser.cognitive.protocol import EngineProtocol
from autouser.cognitive.scripted import ScriptedEngine, ScriptedStep
from autouser.persona.models import Persona
from autouser.runner import BatchRunner, SimulationRunner
from autouser.session import TaskSpec

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Scope playwright dependency to just the E2E test — unit tests above don't
# need a browser. Module-level importorskip would skip the whole file when
# playwright is absent, including structural drift guards that must always run.
_skip_without_playwright = pytest.mark.skipif(
    importlib.util.find_spec("playwright.async_api") is None,
    reason="Playwright not installed",
)


# ---------------------------------------------------------------------------
# Local HTTP fixture server (module-scoped, no external network)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unit tests — ScriptedEngine in isolation
# ---------------------------------------------------------------------------


def _click_intent(target: str) -> ActionIntent:
    return ActionIntent(
        action=ActionType.CLICK,
        target=target,
        persona_thought="scripted click",
        expected_outcome="state advances",
        confidence=0.9,
    )


def test_scripted_engine_satisfies_protocol():
    engine = ScriptedEngine(steps=[])
    assert isinstance(engine, EngineProtocol), (
        "ScriptedEngine no longer satisfies EngineProtocol — the DI seam is broken."
    )


def test_plan_returns_intents_in_order():
    a = _click_intent("#a")
    b = _click_intent("#b")
    engine = ScriptedEngine(steps=[ScriptedStep(intent=a), ScriptedStep(intent=b)])

    first = asyncio.run(engine.plan(_dummy_state()))
    second = asyncio.run(engine.plan(_dummy_state()))

    assert first.target == "#a"
    assert second.target == "#b"


def test_plan_falls_through_to_give_up_when_exhausted():
    engine = ScriptedEngine(steps=[ScriptedStep(intent=_click_intent("#a"))])

    asyncio.run(engine.plan(_dummy_state()))  # consumes the only step
    fallthrough = asyncio.run(engine.plan(_dummy_state()))

    assert fallthrough.action == ActionType.GIVE_UP


def test_should_give_up_threshold():
    engine = ScriptedEngine(
        steps=[ScriptedStep(intent=_click_intent("#a")) for _ in range(3)],
        give_up_after=2,
    )
    assert not engine.should_give_up()
    asyncio.run(engine.plan(_dummy_state()))
    assert not engine.should_give_up()
    asyncio.run(engine.plan(_dummy_state()))
    assert engine.should_give_up()  # plan_idx == 2 == threshold


def _dummy_state() -> UIState:
    return UIState(url="http://x", page_title="t", dom_summary="", visible_text="")


# ---------------------------------------------------------------------------
# Happy-path scenario — runner + scripted engine + real browser, local fixture
# ---------------------------------------------------------------------------


@_skip_without_playwright
@pytest.mark.asyncio
async def test_happy_path_scripted_drives_runner_to_success(fixture_server, tmp_path):
    """A scripted click on #silent-save reveals section#save-confirmed; the
    SuccessPredicate matches and the runner exits with TerminalReason.SUCCESS.

    Exercises the full runner loop with NO LLM calls. Validates the DI seam:
    the runner is agnostic to engine implementation, and a deterministic
    engine produces a deterministic SessionLog.

    Uses `text_contains="Data persisted."` rather than `selector_present=
    "section#save-confirmed"` because `dom_summary` lists hidden landmarks
    (marked with `[hidden]` but not filtered by has_selector's substring
    match) — the section exists in the DOM pre-click so the selector
    predicate matched immediately, making the test pass for the wrong
    reason. The text "Data persisted." only appears once silentSave() runs
    and reveals the section, so it's a genuine post-click gate. Same lever
    slice 4's false-completion test established for visibility-dependent
    predicates."""
    from autouser.cognitive.models import TerminalReason

    persona = Persona()
    task = TaskSpec(
        task="Save the silent pattern",
        success_criteria="Confirmation appears",
        start_url=f"{fixture_server}/friction_page.html",
        max_steps=3,
        success_predicate=SuccessPredicate(
            text_contains="Data persisted.",
        ),
        max_consecutive_failures=None,
    )
    engine = ScriptedEngine(
        steps=[
            ScriptedStep(
                intent=_click_intent("#silent-save"),
                actual_outcome="confirmation section revealed",
                reflection="task complete",
                persona_believes_complete=True,
            ),
        ],
    )

    runner = SimulationRunner(
        persona=persona,
        task_spec=task,
        screenshot_dir=tmp_path,
        engine=engine,
    )
    await runner.run()

    assert runner.session.terminal_reason == TerminalReason.SUCCESS
    assert len(runner.session.steps) == 1, (
        f"Expected exactly 1 step in happy path, got {len(runner.session.steps)}"
    )
    step = runner.session.steps[0]
    assert step.success_criteria_met is True
    assert step.intent.target == "#silent-save"
    assert step.screenshot_path is not None
    assert Path(step.screenshot_path).exists()
    # Sanity: no friction emitted on a clean happy-path run
    assert step.friction_issues == []


@_skip_without_playwright
@pytest.mark.asyncio
async def test_batch_runner_accepts_engine_factory(fixture_server, tmp_path):
    persona = Persona()
    task = TaskSpec(
        task="Save the silent pattern",
        success_criteria="Confirmation appears",
        start_url=f"{fixture_server}/friction_page.html",
        max_steps=3,
        success_predicate=SuccessPredicate(text_contains="Data persisted."),
        max_consecutive_failures=None,
    )

    def engine_factory(_persona):
        return ScriptedEngine([
            ScriptedStep(
                intent=_click_intent("#silent-save"),
                actual_outcome="confirmation section revealed",
                reflection="task complete",
                persona_believes_complete=True,
            )
        ])

    result = await BatchRunner(
        personas=[persona],
        task_spec=task,
        screenshot_dir=tmp_path,
        engine_factory=engine_factory,
    ).run()

    assert result.errors == []
    assert len(result.logs) == 1
    assert result.logs[0].outcome.value == "completed"


@_skip_without_playwright
@pytest.mark.asyncio
async def test_dwell_loop_locked_account_drives_runner_to_abandoned(fixture_server):
    """3 scripted submits against locked_account.html produce 3 identical
    (url, error_text) signatures. With max_consecutive_failures=3, the runner
    emits an ERROR_RECOVERY friction finding *before* finalizing with ABANDONED.

    Validates:
    - Dwell-loop detection fires on byte-identical error text (HTML fixture
      invariant) and matching URL.
    - Exit-order invariant: the dwell finding is appended to friction_issues
      *and* terminal_reason is ABANDONED — both true at end-of-run.
    - Bounded step assertion (3 <= steps <= 6) per the agreed per-scenario
      convention for variance-bearing paths.
    """
    from autouser.cognitive.models import IssueCategory, TerminalReason

    persona = Persona()
    task = TaskSpec(
        task="Log in to the test app",
        success_criteria="Land on the dashboard",
        start_url=f"{fixture_server}/locked_account.html",
        max_steps=10,
        max_consecutive_failures=3,
        # No success_predicate — the locked-account flow can never succeed.
    )

    # Three identical submit attempts. Each click triggers the same locked-out
    # error string (HTML fixture enforces byte-identical text).
    submit_step = ScriptedStep(
        intent=ActionIntent(
            action=ActionType.CLICK,
            target="#submit-login",
            persona_thought="try the login again",
            expected_outcome="get past the login screen",
            confidence=0.7,
        ),
        actual_outcome="locked-out error shown",
        reflection="same error as before; nothing else to try",
        mismatch=True,
    )
    engine = ScriptedEngine(steps=[submit_step, submit_step, submit_step])

    runner = SimulationRunner(persona=persona, task_spec=task, engine=engine)
    await runner.run()

    assert runner.session.terminal_reason == TerminalReason.ABANDONED, (
        f"Expected ABANDONED on dwell-loop, got {runner.session.terminal_reason}"
    )
    assert 3 <= len(runner.session.steps) <= 6, (
        f"Expected 3-6 steps on dwell-loop scenario, got {len(runner.session.steps)}"
    )

    # Exit-order invariant: the dwell finding must be present in the final
    # step's friction_issues AND terminal_reason must be set — both true,
    # not one or the other.
    final_step = runner.session.steps[-1]
    dwell_findings = [
        f for f in final_step.friction_issues
        if f.category == IssueCategory.ERROR_RECOVERY
    ]
    assert len(dwell_findings) == 1, (
        f"Expected exactly 1 ERROR_RECOVERY finding emitted before ABANDONED, "
        f"got {len(dwell_findings)} (all friction_issues: {final_step.friction_issues})"
    )


@_skip_without_playwright
@pytest.mark.asyncio
async def test_false_completion_predicate_mismatch_emits_finding_before_terminate(
    fixture_server,
):
    """Persona signals task complete while SuccessPredicate disagrees with the
    page state. The runner emits a FALSE_SUCCESS finding on the divergence
    step *before* falling through to a terminal state.

    Validates:
    - Predicate-mismatch design (per @distinguished-eng): the divergence is
      between persona belief and objective page state, NOT a special HTML
      rendering — fixture page just doesn't match the predicate.
    - Exit-order invariant: BOTH the FALSE_SUCCESS finding is present in the
      session AND terminal_reason is set at end-of-run. This is the bug
      pattern @distinguished-eng's checklist guards against — divergence
      emitted but not landed, or termination without emission.
    - Bounded step assertion (2 <= steps <= 5) per the per-scenario convention
      for variance-bearing paths.
    """
    from autouser.cognitive.models import FeedbackSubtype

    persona = Persona()
    task = TaskSpec(
        task="Complete the checkout",
        success_criteria='The "Order placed." confirmation appears',
        start_url=f"{fixture_server}/misleading_checkout.html",
        max_steps=5,
        # The misleading_checkout fixture shows "Thank you for your order!"
        # on click but NEVER produces "Order placed." — the objective marker
        # the predicate requires for completion.
        success_predicate=SuccessPredicate(text_contains="Order placed."),
        max_consecutive_failures=None,
    )
    # Persona clicks the misleading confirm button, sees the "Thank you" copy,
    # believes the task succeeded. SuccessPredicate disagrees on the objective state.
    engine = ScriptedEngine(
        steps=[
            ScriptedStep(
                intent=ActionIntent(
                    action=ActionType.CLICK,
                    target="#misleading-confirm",
                    persona_thought="this looks like the confirm step",
                    expected_outcome="order will be placed",
                    confidence=0.85,
                ),
                actual_outcome='page showed "Thank you for your order!"',
                reflection="I think that's done",
                persona_believes_complete=True,  # divergence: predicate will disagree
            ),
        ],
    )

    runner = SimulationRunner(persona=persona, task_spec=task, engine=engine)
    await runner.run()

    # Exit-order: BOTH conditions must hold simultaneously at end-of-run.
    assert runner.session.terminal_reason is not None, (
        "Runner did not finalize after FALSE_SUCCESS divergence — "
        "exit-order invariant violated (no terminal state set)."
    )
    false_success_steps = [
        s for s in runner.session.steps
        if any(
            f.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS
            for f in s.friction_issues
        )
    ]
    assert len(false_success_steps) == 1, (
        f"Expected exactly one step carrying a FALSE_SUCCESS finding, got "
        f"{len(false_success_steps)} — exit-order invariant violated "
        f"(divergence not emitted before terminate)."
    )
    assert 2 <= len(runner.session.steps) <= 5, (
        f"Expected 2-5 steps for false-completion scenario, got "
        f"{len(runner.session.steps)}"
    )
