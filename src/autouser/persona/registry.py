"""Built-in persona archetypes (v1: Maria, Jake, Priya, Carlos, Yuki)."""

from autouser.persona.models import AccessibilityProfile, Level, PersonaArchetype

ARCHETYPES: dict[str, PersonaArchetype] = {
    "maria": PersonaArchetype(
        name="maria",
        narrative=(
            "Maria, 58, retired teacher, new to web apps. Reads carefully but doesn't know "
            "UI conventions. Catches bad information architecture and unclear labeling."
        ),
        tech_literacy=Level.LOW,
        patience=6,
        reading_comprehension=Level.HIGH,
        domain_familiarity=Level.LOW,
        goal_clarity=Level.MEDIUM,
    ),
    "jake": PersonaArchetype(
        name="jake",
        narrative=(
            "Jake, 26, developer, power user. Impatient with hand-holding. Catches missing "
            "keyboard shortcuts, slow flows, and lack of bulk actions."
        ),
        tech_literacy=Level.HIGH,
        patience=4,
        reading_comprehension=Level.MEDIUM,
        domain_familiarity=Level.HIGH,
        goal_clarity=Level.HIGH,
    ),
    "priya": PersonaArchetype(
        name="priya",
        narrative=(
            "Priya, 34, screen reader user, works in finance. Patient because she's used to "
            "bad UX. Catches accessibility barriers that block task completion."
        ),
        tech_literacy=Level.MEDIUM,
        patience=8,
        reading_comprehension=Level.HIGH,
        domain_familiarity=Level.MEDIUM,
        goal_clarity=Level.HIGH,
        accessibility=AccessibilityProfile(screen_reader=True, keyboard_only=True),
    ),
    "carlos": PersonaArchetype(
        name="carlos",
        narrative=(
            "Carlos, 40, multitasking parent, on mobile. Skims everything, taps wrong targets, "
            "gets interrupted mid-flow. Catches failures under distraction and cognitive load."
        ),
        tech_literacy=Level.MEDIUM,
        patience=2,
        reading_comprehension=Level.LOW,
        domain_familiarity=Level.LOW,
        goal_clarity=Level.LOW,
        accessibility=AccessibilityProfile(motor_precision=Level.LOW),
    ),
    "yuki": PersonaArchetype(
        name="yuki",
        narrative=(
            "Yuki, 22, non-native English speaker, student. Catches jargon, idiom-dependent "
            "labels, and culturally-specific UI metaphors."
        ),
        tech_literacy=Level.MEDIUM,
        patience=5,
        reading_comprehension=Level.LOW,
        domain_familiarity=Level.LOW,
        goal_clarity=Level.MEDIUM,
    ),
}


def get_archetype(name: str) -> PersonaArchetype:
    key = name.lower()
    if key not in ARCHETYPES:
        raise KeyError(f"Unknown archetype '{name}'. Available: {list(ARCHETYPES.keys())}")
    return ARCHETYPES[key]
