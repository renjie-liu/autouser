"""Core persona data models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Level(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AccessibilityProfile(BaseModel):
    """Accessibility-related configuration for a persona."""

    screen_reader: bool = False
    keyboard_only: bool = False
    zoom_level: float = Field(default=1.0, ge=1.0, le=4.0)
    motor_precision: Level = Level.HIGH  # HIGH = precise clicks, LOW = frequent mis-clicks


class PersonaArchetype(BaseModel):
    """A named persona archetype with narrative context and behavioral parameters."""

    name: str
    narrative: str = Field(description="Human-readable backstory, e.g. 'Maria, 58, retired teacher'")
    tech_literacy: Level = Level.MEDIUM
    patience: int = Field(default=5, ge=1, le=10)
    reading_comprehension: Level = Level.MEDIUM
    domain_familiarity: Level = Level.LOW
    goal_clarity: Level = Level.MEDIUM
    accessibility: AccessibilityProfile = Field(default_factory=AccessibilityProfile)


class Persona(BaseModel):
    """A concrete persona instance used during a simulation run.

    Can be created from an archetype or from raw parameters.
    """

    archetype: Optional[PersonaArchetype] = None

    # Behavioral parameters (override archetype if set directly)
    tech_literacy: Level = Level.MEDIUM
    patience: int = Field(default=5, ge=1, le=10)
    reading_comprehension: Level = Level.MEDIUM
    domain_familiarity: Level = Level.LOW
    goal_clarity: Level = Level.MEDIUM
    accessibility: AccessibilityProfile = Field(default_factory=AccessibilityProfile)

    @classmethod
    def from_archetype(cls, archetype: PersonaArchetype) -> Persona:
        return cls(
            archetype=archetype,
            tech_literacy=archetype.tech_literacy,
            patience=archetype.patience,
            reading_comprehension=archetype.reading_comprehension,
            domain_familiarity=archetype.domain_familiarity,
            goal_clarity=archetype.goal_clarity,
            accessibility=archetype.accessibility.model_copy(deep=True),
        )

    @classmethod
    def from_config(
        cls,
        *,
        tech_literacy: Level = Level.MEDIUM,
        patience: int = 5,
        domain_familiarity: Level = Level.LOW,
        cognitive_load_tolerance: Level = Level.MEDIUM,
        accessibility: Optional[AccessibilityProfile] = None,
    ) -> Persona:
        """Build a Persona from the 5 product-level behavioral dimensions.

        These are the dimensions the docs advertise (tech literacy, patience /
        error tolerance, domain familiarity, cognitive load tolerance,
        accessibility). `cognitive_load_tolerance` decomposes into the two
        internal constraints the engine actually uses — `reading_comprehension`
        and `goal_clarity` — which both follow the external level: a user who
        tolerates more cognitive load reads more carefully *and* holds a clearer
        goal. Archetypes may still set the two sub-dimensions independently when
        a persona needs finer nuance (e.g. reads carefully but with a vague
        goal).
        """
        return cls(
            tech_literacy=tech_literacy,
            patience=patience,
            domain_familiarity=domain_familiarity,
            reading_comprehension=cognitive_load_tolerance,
            goal_clarity=cognitive_load_tolerance,
            accessibility=accessibility or AccessibilityProfile(),
        )
