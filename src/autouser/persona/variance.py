"""Within-archetype variance sampling (cognitive v2, slice 4).

One run of one archetype is an anecdote. Real usability studies get variance
across users *of the same type* — so a study samples persona parameters
around the archetype and runs N seeds. Aggregating findings across seeds
converts "maria got confused once" into "7 of 10 low-literacy runs stalled
at the same modal": reproduction frequency becomes severity evidence, and
M3 gets its behavioral distributions from the same data.

Sampling is deterministic per (archetype, seed) so any study is exactly
reproducible. Seed 0 is always the unjittered archetype — every study
includes the canonical baseline.

Accessibility is NEVER jittered: a screen reader, keyboard-only operation,
or low motor precision is identity-defining for the archetype's user
category — a "variant of Priya" without a screen reader isn't a Priya
variant, it's a different study cell.
"""

from __future__ import annotations

import random

from autouser.persona.models import Level, Persona, PersonaArchetype

_LEVEL_ORDER = [Level.LOW, Level.MEDIUM, Level.HIGH]

# Probability that a Level dimension shifts one step (up or down, clamped).
LEVEL_SHIFT_PROBABILITY = 0.25
# Gaussian sigma for patience jitter (clamped to the model's 1..10 range).
PATIENCE_SIGMA = 1.0


def _shift_level(level: Level, rng: random.Random) -> Level:
    if rng.random() >= LEVEL_SHIFT_PROBABILITY:
        return level
    idx = _LEVEL_ORDER.index(level) + rng.choice([-1, 1])
    return _LEVEL_ORDER[min(max(idx, 0), len(_LEVEL_ORDER) - 1)]


def sample_variant(archetype: PersonaArchetype, seed: int) -> Persona:
    """A persona sampled around *archetype*, deterministic per (name, seed).

    Seed 0 returns the exact archetype (the canonical baseline run).
    """
    if seed == 0:
        return Persona.from_archetype(archetype)

    rng = random.Random(f"{archetype.name}:{seed}")
    patience = int(round(archetype.patience + rng.gauss(0, PATIENCE_SIGMA)))
    return Persona(
        archetype=archetype,
        tech_literacy=_shift_level(archetype.tech_literacy, rng),
        patience=min(max(patience, 1), 10),
        reading_comprehension=_shift_level(archetype.reading_comprehension, rng),
        domain_familiarity=_shift_level(archetype.domain_familiarity, rng),
        goal_clarity=_shift_level(archetype.goal_clarity, rng),
        accessibility=archetype.accessibility.model_copy(deep=True),
    )


def variant_summary(persona: Persona) -> str:
    """One-line parameter summary for study tables (matches FrictionLog's)."""
    return (
        f"tech_literacy={persona.tech_literacy.value}, "
        f"patience={persona.patience}, "
        f"reading={persona.reading_comprehension.value}, "
        f"domain={persona.domain_familiarity.value}, "
        f"goal_clarity={persona.goal_clarity.value}"
    )
