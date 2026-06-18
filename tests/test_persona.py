"""Tests for the persona engine."""

import pytest

from autouser.persona import ARCHETYPES, get_archetype
from autouser.persona.models import AccessibilityProfile, Level, Persona
from autouser.cognitive.models import ActionIntent, ActionType
from autouser.session import TaskSpec, SessionLog


def test_all_archetypes_defined():
    assert set(ARCHETYPES.keys()) == {"maria", "jake", "priya", "carlos", "yuki"}


def test_get_archetype():
    maria = get_archetype("maria")
    assert maria.tech_literacy == Level.LOW
    assert maria.reading_comprehension == Level.HIGH
    assert maria.patience == 6


def test_persona_from_archetype():
    archetype = get_archetype("priya")
    persona = Persona.from_archetype(archetype)
    assert persona.accessibility.screen_reader is True
    assert persona.accessibility.keyboard_only is True
    assert persona.tech_literacy == Level.MEDIUM


def test_custom_persona():
    persona = Persona(tech_literacy=Level.HIGH, patience=2, goal_clarity=Level.LOW)
    assert persona.archetype is None
    assert persona.patience == 2


def test_from_config_maps_cognitive_load_to_internal_subdimensions():
    """The external 5-dimension API: cognitive_load_tolerance decomposes into the
    internal reading_comprehension + goal_clarity constraints the engine uses."""
    persona = Persona.from_config(
        tech_literacy=Level.HIGH,
        patience=8,
        domain_familiarity=Level.MEDIUM,
        cognitive_load_tolerance=Level.LOW,
    )
    assert persona.archetype is None
    assert persona.tech_literacy == Level.HIGH
    assert persona.patience == 8
    assert persona.domain_familiarity == Level.MEDIUM
    # cognitive_load_tolerance drives BOTH internal sub-dimensions
    assert persona.reading_comprehension == Level.LOW
    assert persona.goal_clarity == Level.LOW


def test_from_config_defaults_and_accessibility_passthrough():
    profile = AccessibilityProfile(screen_reader=True, keyboard_only=True)
    persona = Persona.from_config(
        cognitive_load_tolerance=Level.HIGH,
        accessibility=profile,
    )
    assert persona.reading_comprehension == Level.HIGH
    assert persona.goal_clarity == Level.HIGH
    assert persona.accessibility.screen_reader is True
    # unspecified dimensions fall back to model defaults
    assert persona.tech_literacy == Level.MEDIUM
    assert persona.patience == 5
    assert persona.domain_familiarity == Level.LOW


def test_carlos_is_impatient():
    carlos = Persona.from_archetype(get_archetype("carlos"))
    assert carlos.patience == 2
    assert carlos.reading_comprehension == Level.LOW


def test_carlos_has_low_motor_precision():
    carlos = Persona.from_archetype(get_archetype("carlos"))
    assert carlos.accessibility.motor_precision == Level.LOW


def test_task_spec_validation():
    spec = TaskSpec(
        task="Sign up for an account",
        success_criteria="User reaches dashboard",
        start_url="https://example.com",
    )
    assert spec.max_steps == 20


def test_session_log_creation():
    spec = TaskSpec(
        task="Sign up",
        success_criteria="Dashboard reached",
        start_url="https://example.com",
    )
    persona = Persona.from_archetype(get_archetype("maria"))
    log = SessionLog(task_spec=spec, persona_snapshot=persona)
    assert log.run_id  # non-empty
    assert len(log.steps) == 0
    assert log.terminal_reason is None


def test_action_intent_requires_input_for_type():
    with pytest.raises(ValueError, match="requires input_value"):
        ActionIntent(
            action=ActionType.TYPE,
            target="#email",
            persona_thought="I need to enter my email",
            expected_outcome="Email field populated",
            confidence=0.8,
        )


def test_action_intent_valid_type():
    intent = ActionIntent(
        action=ActionType.TYPE,
        target="#email",
        input_value="user@example.com",
        persona_thought="I need to enter my email",
        expected_outcome="Email field populated",
        confidence=0.8,
    )
    assert intent.input_value == "user@example.com"


def test_action_intent_click_requires_target():
    with pytest.raises(ValueError, match="requires a non-empty target"):
        ActionIntent(
            action=ActionType.CLICK,
            target="",
            persona_thought="Click something",
            expected_outcome="Something happens",
            confidence=0.5,
        )


def test_action_intent_type_without_target_is_valid():
    """Keyboard-user flow: empty target means 'type into the focused control'.
    Typing with nothing focused is an executor-level action_error, not a
    validation error."""
    intent = ActionIntent(
        action=ActionType.TYPE,
        target="",
        input_value="hello",
        persona_thought="Type into the focused field",
        expected_outcome="Text appears",
        confidence=0.7,
    )
    assert intent.target == ""


def test_action_intent_press_requires_key():
    with pytest.raises(ValueError, match="requires input_value"):
        ActionIntent(
            action=ActionType.PRESS,
            persona_thought="Press something",
            expected_outcome="Focus moves",
            confidence=0.7,
        )


def test_action_intent_press_valid():
    intent = ActionIntent(
        action=ActionType.PRESS,
        input_value="Tab",
        persona_thought="Move to the next control",
        expected_outcome="Focus moves to the next control",
        confidence=0.8,
    )
    assert intent.input_value == "Tab"


def test_archetype_accessibility_is_not_shared():
    """Verify from_archetype deep-copies accessibility to prevent template mutation."""
    archetype = get_archetype("priya")
    p1 = Persona.from_archetype(archetype)
    p2 = Persona.from_archetype(archetype)
    p1.accessibility.zoom_level = 3.0
    assert p2.accessibility.zoom_level == 1.0
    assert archetype.accessibility.zoom_level == 1.0
