"""The live CognitiveEngine must satisfy EngineProtocol — guards the DI seam.

Runtime check (`isinstance(engine, EngineProtocol)` with `@runtime_checkable`)
verifies *attribute presence* only — renames and removals fail here. Signature
drift (e.g., `plan(self, ui_state)` → `plan(self, state, hint='')`) is NOT
caught by this test; that gate is mypy/pyright on the import graph.

So: this test guards "the seam still has the right shape." Static type
checking guards "the seam still has the right contract." Both matter.
"""

from __future__ import annotations

from autouser.cognitive.engine import CognitiveEngine
from autouser.cognitive.protocol import EngineProtocol
from autouser.persona.models import Persona


def _make_engine() -> CognitiveEngine:
    persona = Persona()  # all-defaults persona; the protocol check needs no behavior
    return CognitiveEngine(persona, "test task", "test success criteria")


def test_cognitive_engine_satisfies_protocol():
    engine = _make_engine()
    assert isinstance(engine, EngineProtocol), (
        "CognitiveEngine no longer satisfies EngineProtocol. "
        "The runner DI seam is broken — either restore the surface "
        "(plan, reflect, should_give_up, history) or update EngineProtocol "
        "to reflect the new contract."
    )
