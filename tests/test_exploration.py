"""Exploration mode (cognitive v2, slice 5): explore an unknown system.

No task, intrinsic motivation, 'seen enough' as the natural end. The response
schemas are unchanged — only the framing and the runner's exit semantics
differ from goal mode.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from autouser.cli import build_parser
from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CompletionSource,
    Emotion,
    StepResult,
    TerminalReason,
    UIState,
)
from autouser.cognitive.prompts import build_reflect_system_prompt, build_system_prompt
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.runner import SimulationRunner
from autouser.session import TaskSpec


@pytest.fixture
def maria() -> Persona:
    return Persona.from_archetype(get_archetype("maria"))


# --- prompt framing ------------------------------------------------------------


class TestExplorationPrompts:
    def test_exploration_system_prompt_swaps_task_for_curiosity(self, maria):
        prompt = build_system_prompt(maria, "ignored", "ignored", exploration=True)
        assert "Nobody gave you a task" in prompt
        assert "Prefer what you haven't tried yet" in prompt
        assert "seen enough" in prompt
        assert "You are trying to:" not in prompt
        # PROGRESS is reworded for understanding, not task completion.
        assert "what this product is and whether it's for you" in prompt
        assert "move you closer to completing your task" not in prompt

    def test_goal_mode_unchanged(self, maria):
        prompt = build_system_prompt(maria, "Sign up", "See dashboard")
        assert "You are trying to: Sign up" in prompt
        assert "Nobody gave you a task" not in prompt
        assert "move you closer to completing your task" in prompt

    def test_exploration_reflect_reinterprets_belief(self, maria):
        explore = build_reflect_system_prompt(maria, exploration=True)
        goal = build_reflect_system_prompt(maria)
        assert "EXPLORATION MODE" in explore
        assert "tell a friend what this product is" in explore
        assert "EXPLORATION MODE" not in goal

    def test_engine_threads_exploration_flag(self, maria):
        from autouser.cognitive.engine import CognitiveEngine

        engine = CognitiveEngine(
            maria, "t", "c", provider="anthropic", exploration=True
        )
        assert "Nobody gave you a task" in engine._get_system_prompt()


# --- runner exit semantics -------------------------------------------------------


def _ui(text="Welcome") -> UIState:
    return UIState(url="https://x.test/", page_title="T", dom_summary="d", visible_text=text)


def _explore_spec(max_steps: int = 6) -> TaskSpec:
    return TaskSpec(
        task="Get to know this product.",
        success_criteria="Seen enough.",
        start_url="https://x.test/",
        max_steps=max_steps,
        exploration=True,
    )


def _click(target="#a") -> ActionIntent:
    return ActionIntent(
        action=ActionType.CLICK, target=target,
        persona_thought="curious", expected_outcome="something new",
        confidence=0.6,
    )


def _result(step, *, believes=False, mismatch=False) -> StepResult:
    return StepResult(
        step=step,
        intent=_click(f"#t{step}"),
        observation_before=_ui(),
        observation_after=_ui(text=f"changed {step}"),
        actual_outcome="saw something",
        mismatch=mismatch,
        reflection="verdict here" if believes else "still looking",
        emotion=Emotion.CONFIDENT,
        persona_believes_complete=believes,
    )


async def _run_explore(spec: TaskSpec, results_by_step, persona) -> SimulationRunner:
    runner = SimulationRunner(persona, spec)

    async def mock_start(url):
        return _ui()

    async def mock_execute(intent, step, **kwargs):
        return _ui(text=f"changed {step}")

    async def mock_plan(ui_state):
        return _click(f"#t{len(runner.engine.history) + 1}")

    async def mock_reflect(intent, new_ui_state, **kwargs):
        return results_by_step(kwargs.get("step_number", 1))

    runner.executor.start = mock_start
    runner.executor.execute = mock_execute
    runner.executor.stop = AsyncMock()
    runner.engine.plan = mock_plan
    runner.engine.reflect = mock_reflect
    await runner.run()
    return runner


class TestExplorationTermination:
    async def test_seen_enough_finalizes_success(self, maria):
        runner = await _run_explore(
            _explore_spec(),
            lambda n: _result(n, believes=(n == 3)),
            maria,
        )
        assert runner.session.terminal_reason == TerminalReason.SUCCESS
        last = runner.session.steps[-1]
        assert last.step == 3
        assert last.completion_source == CompletionSource.EXPLORATION
        assert last.success_criteria_met is True

    async def test_belief_with_mismatch_does_not_exit(self, maria):
        runner = await _run_explore(
            _explore_spec(max_steps=4),
            lambda n: _result(n, believes=True, mismatch=True),
            maria,
        )
        # Mismatched 'seen enough' never fires the exploration exit; the run
        # ends some other way (frustration from expectation violations, or
        # the step ceiling).
        assert runner.session.terminal_reason in (
            TerminalReason.TIMED_OUT, TerminalReason.ABANDONED,
        )
        assert all(
            s.completion_source != CompletionSource.EXPLORATION
            for s in runner.session.steps
        )

    async def test_no_belief_runs_to_ceiling(self, maria):
        runner = await _run_explore(
            _explore_spec(max_steps=3),
            lambda n: _result(n, believes=False),
            maria,
        )
        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT

    async def test_goal_mode_ignores_exploration_exit(self, maria):
        spec = _explore_spec(max_steps=3).model_copy(update={"exploration": False})
        runner = await _run_explore(
            spec,
            lambda n: _result(n, believes=True),  # belief alone, no success signal
            maria,
        )
        # In goal mode belief without the soft gate's observed_success_signal
        # must not complete the run.
        assert runner.session.terminal_reason == TerminalReason.TIMED_OUT


# --- CLI -------------------------------------------------------------------------


class TestExploreCli:
    def test_explore_parser_minimal_surface(self):
        args = build_parser().parse_args(
            ["explore", "--url", "https://x.test/", "--persona", "yuki"]
        )
        assert args.command == "explore"
        assert args.max_steps == 25
        assert args.persona == ["yuki"]

    def test_explore_has_no_task_args(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["explore", "--url", "https://x.test/", "--task", "t"]
            )
