"""Protocol for the cognitive engine surface consumed by SimulationRunner.

Enables dependency injection of alternative engine implementations (e.g.
ScriptedEngine for deterministic fixture-harness tests) without changing
the runner. Mirrors the live CognitiveEngine surface exactly; if that
surface evolves, mypy / pyright will surface the drift at the seam.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from autouser.cognitive.models import ActionIntent, StepResult, UIState


@runtime_checkable
class EngineProtocol(Protocol):
    """Engine surface required by SimulationRunner.

    Any object satisfying this Protocol can drive a simulation in place of
    the live CognitiveEngine. The Protocol intentionally captures only the
    attributes/methods the runner actually calls — additions to the live
    engine should not expand this surface unless the runner depends on them.
    """

    history: Sequence[StepResult]

    async def plan(self, ui_state: UIState) -> ActionIntent: ...

    async def reflect(
        self,
        intent: ActionIntent,
        new_ui_state: UIState,
        *,
        observation_before: UIState,
        observation_after: UIState,
        step_number: int,
    ) -> StepResult: ...

    def should_give_up(self) -> bool: ...
