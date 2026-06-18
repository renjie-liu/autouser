"""Mental-model memory + perception filter (cognitive v2, slice 1).

Pure tests for the two harness-level realism mechanisms:
  * memory — places tracked mechanically, beliefs (mental notes) written by
    the persona in reflect() and fed back into plan() prompts;
  * perception — screen-reader personas perceive the accessibility tree,
    not the visual element inventory.
"""

from __future__ import annotations

import pytest

from autouser.cognitive.models import ActionIntent, ActionType, Emotion, StepResult, UIState
from autouser.cognitive.prompts import build_system_prompt, build_user_prompt, format_memory
from autouser.perception import (
    PERCEIVED_SCREEN_READER,
    PERCEIVED_SCREEN_READER_FALLBACK,
    perceive,
)
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype


@pytest.fixture
def maria() -> Persona:
    return Persona.from_archetype(get_archetype("maria"))


@pytest.fixture
def priya() -> Persona:
    return Persona.from_archetype(get_archetype("priya"))


def _state(**overrides) -> UIState:
    defaults = dict(
        url="https://x.test/",
        page_title="Home",
        dom_summary="== Interactive Elements ==\n  button#go | text=\"Go\"",
        visible_text="Welcome",
    )
    defaults.update(overrides)
    return UIState(**defaults)


def _step(step: int, *, note=None, before=None, after=None) -> StepResult:
    return StepResult(
        step=step,
        intent=ActionIntent(
            action=ActionType.CLICK, target="#go",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        ),
        observation_before=before or _state(),
        observation_after=after or _state(),
        actual_outcome="Something happened.",
        mismatch=False,
        reflection="ok",
        emotion=Emotion.CONFIDENT,
        mental_note=note,
    )


# --- format_memory -----------------------------------------------------------


class TestFormatMemory:
    def test_empty_memory_renders_nothing(self):
        assert format_memory([], []) == ""

    def test_single_place_without_notes_renders_nothing(self):
        # One place is just the current page — no recall value yet.
        assert format_memory([(1, "Home", "https://x.test/")], []) == ""

    def test_notes_render_with_step_attribution(self):
        out = format_memory([], [(2, "The Create button needs a name first.")])
        assert "Your Notes About This Site" in out
        assert "(step 2) The Create button needs a name first." in out

    def test_multiple_places_render(self):
        places = [(1, "Home", "https://x.test/"), (3, "Settings", "https://x.test/settings")]
        out = format_memory(places, [])
        assert "Places You've Been" in out
        assert "Settings — https://x.test/settings" in out

    def test_notes_capped_to_most_recent(self):
        notes = [(i, f"note {i}") for i in range(1, 20)]
        out = format_memory([], notes)
        assert "note 19" in out
        assert "note 1)" not in out  # oldest dropped past the cap


class TestPlanPromptCarriesMemory:
    def test_memory_section_lands_in_plan_prompt(self):
        memory = format_memory(
            [(1, "Home", "https://x.test/"), (2, "Form", "https://x.test/form")],
            [(2, "The form hides until you press Start.")],
        )
        prompt = build_user_prompt(
            _state(), 3, [], patience_budget=6, frustration_spent=0,
            memory_section=memory,
        )
        assert "The form hides until you press Start." in prompt
        assert "Places You've Been" in prompt

    def test_no_memory_section_on_first_step(self):
        prompt = build_user_prompt(
            _state(), 1, [], patience_budget=6, frustration_spent=0,
        )
        assert "Your Notes About This Site" not in prompt


class TestEngineComposesMemoryFromHistory:
    def test_compose_memory_collects_places_and_notes(self, maria):
        from autouser.cognitive.engine import CognitiveEngine

        engine = CognitiveEngine.__new__(CognitiveEngine)  # no provider init needed
        engine.persona = maria
        engine.history = [
            _step(1, before=_state(page_title="Home"),
                  after=_state(page_title="Form", url="https://x.test/form")),
            _step(2, note="Submitting needs the phone in a weird format.",
                  before=_state(page_title="Form", url="https://x.test/form"),
                  after=_state(page_title="Form", url="https://x.test/form")),
        ]
        memory = engine._compose_memory()
        assert "Home" in memory
        assert "Form" in memory
        assert "weird format" in memory

    def test_compose_memory_empty_history(self, maria):
        from autouser.cognitive.engine import CognitiveEngine

        engine = CognitiveEngine.__new__(CognitiveEngine)
        engine.persona = maria
        engine.history = []
        assert engine._compose_memory() == ""


# --- perception --------------------------------------------------------------


class TestPerceptionFilter:
    def test_default_personas_perceive_raw_state(self, maria):
        state = _state(aria_snapshot='- button "Go"')
        assert perceive(state, maria) is state

    def test_screen_reader_persona_perceives_accessibility_tree(self, priya):
        state = _state(aria_snapshot='- button "Go"\n- textbox "Email"')
        perceived = perceive(state, priya)
        assert perceived is not state  # raw capture untouched
        assert state.perceived_via is None
        assert perceived.perceived_via == PERCEIVED_SCREEN_READER
        assert "Screen Reader View" in perceived.dom_summary
        assert '- textbox "Email"' in perceived.dom_summary
        # The visual element inventory must NOT leak through.
        assert "button#go" not in perceived.dom_summary

    def test_screen_reader_fallback_when_tree_unavailable(self, priya):
        state = _state(aria_snapshot=None)
        perceived = perceive(state, priya)
        assert perceived.perceived_via == PERCEIVED_SCREEN_READER_FALLBACK
        assert "unavailable" in perceived.dom_summary

    def test_frustrated_persona_skims_long_text(self, maria):
        state = _state(visible_text="x" * 5000)
        perceived = perceive(state, maria, frustration_ratio=0.6)
        assert len(perceived.visible_text) < 5000
        assert "skimming" in perceived.visible_text
        # Raw capture untouched.
        assert len(state.visible_text) == 5000

    def test_calm_persona_reads_everything(self, maria):
        state = _state(visible_text="x" * 5000)
        assert perceive(state, maria, frustration_ratio=0.2) is state

    def test_skim_composes_with_screen_reader(self, priya):
        state = _state(aria_snapshot='- button "Go"', visible_text="x" * 5000)
        perceived = perceive(state, priya, frustration_ratio=0.9)
        assert perceived.perceived_via == PERCEIVED_SCREEN_READER
        assert "skimming" in perceived.visible_text

    def test_screen_reader_system_prompt_teaches_role_targeting(self, priya, maria):
        sr_prompt = build_system_prompt(priya, "Sign up", "See dashboard")
        dom_prompt = build_system_prompt(maria, "Sign up", "See dashboard")
        assert 'role=button[name=' in sr_prompt
        assert "accessible name is unreachable" in sr_prompt
        assert "role=button[name=" not in dom_prompt
        assert "CSS selector" in dom_prompt
