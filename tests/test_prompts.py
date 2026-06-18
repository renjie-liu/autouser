"""Tests for prompt template assembly and constraint correctness."""

from __future__ import annotations

import pytest

from autouser.cognitive.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_history,
    TECH_LITERACY_CONSTRAINTS,
    READING_COMPREHENSION_CONSTRAINTS,
    DOMAIN_FAMILIARITY_CONSTRAINTS,
    GOAL_CLARITY_CONSTRAINTS,
)
from autouser.cognitive.models import ActionIntent, ActionType, StepResult, UIState, Emotion
from autouser.persona.models import Level, Persona
from autouser.persona.registry import get_archetype


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def maria_persona() -> Persona:
    return Persona.from_archetype(get_archetype("maria"))


@pytest.fixture
def jake_persona() -> Persona:
    return Persona.from_archetype(get_archetype("jake"))


@pytest.fixture
def priya_persona() -> Persona:
    return Persona.from_archetype(get_archetype("priya"))


@pytest.fixture
def carlos_persona() -> Persona:
    return Persona.from_archetype(get_archetype("carlos"))


@pytest.fixture
def sample_ui_state() -> UIState:
    return UIState(
        url="https://example.com/dashboard",
        page_title="Dashboard",
        dom_summary='button#submit "Submit Form"\na.nav-link "Home"\ninput#search "Search..."',
        visible_text="Welcome to the dashboard. Click Submit to continue.",
    )


def _make_step_result(step: int, action: str = "click", target: str = "button#submit") -> StepResult:
    return StepResult(
        step=step,
        intent=ActionIntent(
            action=ActionType(action),
            target=target,
            persona_thought="I see a button so I'll click it.",
            expected_outcome="The form submits.",
            confidence=0.7,
        ),
        observation_before=UIState(
            url="https://example.com", page_title="Test", dom_summary="", visible_text=""
        ),
        observation_after=UIState(
            url="https://example.com/result", page_title="Result", dom_summary="", visible_text=""
        ),
        actual_outcome="Form submitted successfully.",
        mismatch=False,
        reflection="That worked as expected.",
        emotion=Emotion.CONFIDENT,
    )


# ============================================================
# System Prompt Assembly
# ============================================================


class TestBuildSystemPrompt:
    def test_contains_persona_name(self, maria_persona: Persona) -> None:
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        assert "maria" in prompt.lower()

    def test_contains_task_description(self, maria_persona: Persona) -> None:
        prompt = build_system_prompt(maria_persona, "Sign up for account", "See dashboard")
        assert "Sign up for account" in prompt

    def test_contains_success_criteria(self, maria_persona: Persona) -> None:
        prompt = build_system_prompt(maria_persona, "Sign up", "See the dashboard page")
        assert "See the dashboard page" in prompt

    def test_system_prompt_carries_no_per_step_state(self, maria_persona: Persona) -> None:
        """The system prompt must be byte-stable for the whole session so the
        Anthropic prompt cache hits on steps 2+. Frustration spend lives in
        the per-step user prompt ("Patience: X of Y...") — re-introducing any
        per-step counter here silently invalidates the cache every step."""
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        assert "frustrated" not in prompt.split("Patience:")[1].split("\n")[0]
        assert "time(s)" not in prompt

    def test_low_tech_uses_soft_navigate_preference(self, maria_persona: Persona) -> None:
        """Engineering constraint: navigate must be soft preference, not prohibition."""
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        # Must NOT contain hard prohibition
        assert "CANNOT use navigate" not in prompt
        assert "cannot use navigate" not in prompt
        assert "do NOT type URLs directly" not in prompt.replace("rarely", "")
        # Must contain soft preference
        assert "rarely type URLs" in prompt or "prefer clicking" in prompt

    def test_all_action_types_listed_for_all_personas(
        self, maria_persona: Persona, jake_persona: Persona
    ) -> None:
        """Engineering constraint: no schema gating. All 8 ActionType values available."""
        for persona in [maria_persona, jake_persona]:
            prompt = build_system_prompt(persona, "Sign up", "See dashboard")
            for action in ["click", "type", "press", "scroll", "navigate", "back", "wait", "give_up"]:
                assert action in prompt, f"Missing action '{action}' for {persona.tech_literacy}"

    def test_confidence_brackets_present(self, maria_persona: Persona) -> None:
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        assert "0.85" in prompt
        assert "YES_CLEARLY" in prompt
        assert "NOT_SURE" in prompt

    def test_persona_thought_no_rpg_tags(self, maria_persona: Persona) -> None:
        """Engineering constraint: persona_thought must be natural language only."""
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        # The persona_thought instruction should NOT require R/P/G tags
        assert "[RECOGNITION:" not in prompt.split('"persona_thought"')[1].split('"expected_outcome"')[0]

    def test_persona_thought_bounded(self, maria_persona: Persona) -> None:
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        assert "1-2 sentences" in prompt

    def test_high_tech_different_from_low_tech(
        self, maria_persona: Persona, jake_persona: Persona
    ) -> None:
        maria_prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        jake_prompt = build_system_prompt(jake_persona, "Sign up", "See dashboard")
        # They should have substantially different constraint text
        assert "hamburger menu" not in maria_prompt or "do not know" in maria_prompt.lower()
        assert "hamburger menus" in jake_prompt

    def test_accessibility_screen_reader_included(self, priya_persona: Persona) -> None:
        prompt = build_system_prompt(priya_persona, "Sign up", "See dashboard")
        assert "screen reader" in prompt.lower()

    def test_accessibility_low_motor_included(self, carlos_persona: Persona) -> None:
        prompt = build_system_prompt(carlos_persona, "Sign up", "See dashboard")
        assert "motor precision" in prompt.lower() or "wrong target" in prompt.lower()

    def test_scaffold_fields_in_response_format(self, maria_persona: Persona) -> None:
        """R/P/G fields should appear in the JSON response format for diagnostic logging."""
        prompt = build_system_prompt(maria_persona, "Sign up", "See dashboard")
        assert '"recognition"' in prompt
        assert '"prediction"' in prompt
        assert '"progress"' in prompt


# ============================================================
# Constraint Block Coverage
# ============================================================


class TestConstraintBlocks:
    def test_all_levels_covered_tech_literacy(self) -> None:
        for level in Level:
            assert level in TECH_LITERACY_CONSTRAINTS

    def test_all_levels_covered_reading(self) -> None:
        for level in Level:
            assert level in READING_COMPREHENSION_CONSTRAINTS

    def test_all_levels_covered_domain(self) -> None:
        for level in Level:
            assert level in DOMAIN_FAMILIARITY_CONSTRAINTS

    def test_all_levels_covered_goal_clarity(self) -> None:
        for level in Level:
            assert level in GOAL_CLARITY_CONSTRAINTS

    def test_low_tech_no_hard_navigate_prohibition(self) -> None:
        constraint = TECH_LITERACY_CONSTRAINTS[Level.LOW]
        assert "CANNOT" not in constraint
        assert "cannot" not in constraint
        assert "do not type URLs directly" not in constraint.lower()
        assert "rarely" in constraint.lower() or "prefer" in constraint.lower()


# ============================================================
# User Prompt Assembly
# ============================================================


class TestBuildUserPrompt:
    def test_contains_page_info(self, sample_ui_state: UIState) -> None:
        prompt = build_user_prompt(sample_ui_state, 1, [], patience_budget=6, frustration_spent=0)
        assert "https://example.com/dashboard" in prompt
        assert "Dashboard" in prompt

    def test_contains_dom_summary(self, sample_ui_state: UIState) -> None:
        prompt = build_user_prompt(sample_ui_state, 1, [], patience_budget=6, frustration_spent=0)
        assert "button#submit" in prompt
        assert "Submit Form" in prompt

    def test_no_history_on_first_step(self, sample_ui_state: UIState) -> None:
        prompt = build_user_prompt(sample_ui_state, 1, [], patience_budget=6, frustration_spent=0)
        assert "Recent Actions" not in prompt

    def test_focus_line_rendered_when_focused(self, sample_ui_state: UIState) -> None:
        focused = sample_ui_state.model_copy(
            update={"focused_element": 'input#search | placeholder="Search..."'}
        )
        prompt = build_user_prompt(focused, 2, [], patience_budget=6, frustration_spent=0)
        assert 'Focused control: input#search | placeholder="Search..."' in prompt

    def test_no_focus_line_when_nothing_focused(self, sample_ui_state: UIState) -> None:
        prompt = build_user_prompt(sample_ui_state, 1, [], patience_budget=6, frustration_spent=0)
        assert "Focused control:" not in prompt

    def test_history_included_after_steps(self, sample_ui_state: UIState) -> None:
        history = [_make_step_result(1), _make_step_result(2)]
        prompt = build_user_prompt(sample_ui_state, 3, history, patience_budget=6, frustration_spent=0)
        assert "Recent Actions" in prompt
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt

    def test_history_capped_at_3_steps(self, sample_ui_state: UIState) -> None:
        history = [_make_step_result(i) for i in range(1, 6)]
        prompt = build_user_prompt(sample_ui_state, 6, history, patience_budget=6, frustration_spent=0)
        # Should only contain last 3 steps
        assert "Step 3:" in prompt
        assert "Step 4:" in prompt
        assert "Step 5:" in prompt
        assert "Step 1:" not in prompt

    def test_visible_text_truncated(self) -> None:
        long_text = "x" * 10000
        ui = UIState(
            url="https://example.com",
            page_title="Test",
            dom_summary="",
            visible_text=long_text,
        )
        prompt = build_user_prompt(ui, 1, [], patience_budget=6, frustration_spent=0)
        # Should be truncated to 4000 chars
        assert len(prompt) < len(long_text)


# ============================================================
# History Formatting
# ============================================================


class TestFormatHistory:
    def test_empty_history(self) -> None:
        assert format_history([]) == ""

    def test_formats_single_step(self) -> None:
        history = [_make_step_result(1)]
        result = format_history(history)
        assert "Step 1:" in result
        assert "click" in result
        assert "button#submit" in result

    def test_max_steps_respected(self) -> None:
        history = [_make_step_result(i) for i in range(1, 10)]
        result = format_history(history, max_steps=3)
        assert "Step 7:" in result
        assert "Step 8:" in result
        assert "Step 9:" in result
        assert "Step 1:" not in result
