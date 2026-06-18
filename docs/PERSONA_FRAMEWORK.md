# Persona Framework

## Overview

AutoUser simulates user behavior by constraining an LLM to act within the behavioral boundaries of a **persona** — a parameterized user profile defined along 5 dimensions. Each dimension controls a different axis of user capability, producing meaningfully different interaction patterns across personas.

Personas are pure data. The simulation runner and cognitive engine operate on persona parameters only; archetype names appear only in reports.

## Dimensions

### Tech Literacy

How familiar the user is with common digital interface patterns.

- **`low`**: Doesn't recognize hamburger menus, doesn't know right-click exists, reads every word before acting. Needs explicit labels and step-by-step guidance.
- **`medium`**: Comfortable with standard web conventions (forms, links, tabs) but unfamiliar with advanced patterns (keyboard shortcuts, drag-and-drop, command palettes).
- **`high`**: Recognizes all standard UI patterns immediately. Expects shortcuts, efficient flows, and power-user features. Gets frustrated by unnecessary hand-holding.

### Patience / Error Tolerance

How many frustrating or confusing steps the persona will tolerate before giving up. Mapped directly to the `patience` parameter (integer, 1-10).

- **Low (1-3)**: Abandons quickly after errors or confusion. Punishes interfaces that don't forgive mistakes.
- **Medium (4-6)**: Will retry a few times but won't persist through repeated friction.
- **High (7-10)**: Keeps trying even when confused. Will read help text, try alternative paths, and work around problems.

In the simulation, patience controls the give-up threshold: the runner tracks steps where mismatches occur or the persona feels confused/frustrated, and ends the simulation when that count reaches the patience value.

### Domain Familiarity

How well the user understands the subject matter of the application (not the UI itself).

- **`low`**: Doesn't know what the domain terms mean. "APR," "webhook," "sprint velocity" are meaningless. Needs contextual definitions and plain-language alternatives.
- **`medium`**: Understands basic domain concepts but may be confused by specialized terminology or non-obvious option consequences.
- **`high`**: Expert in the domain. Knows exactly what they want to do — the only question is whether the UI lets them do it efficiently.

### Cognitive Load Tolerance

How much complexity the user can process at once. This is a composite dimension — internally decomposed into `reading_comprehension` and `goal_clarity` because they produce different behavioral constraints:

- **Reading comprehension** controls how carefully the persona reads text. `low` = skims, misses details, acts on first recognizable option. `high` = reads carefully, notices disclaimers, compares options.
- **Goal clarity** controls how well the persona knows what they're trying to accomplish. `low` = vague goal ("I need to do something with my account"), explores broadly. `high` = specific goal ("I need to change my billing address"), navigates directly.

The external API presents the unified `cognitive_load_tolerance` dimension via `Persona.from_config()`; the internal decomposition into `reading_comprehension` + `goal_clarity` is an implementation detail.

### Accessibility Profile

Physical and assistive technology constraints that affect how the persona interacts with the UI. Defined as a structured `AccessibilityProfile` with:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `screen_reader` | bool | `false` | Persona uses a screen reader; perceives ARIA roles, not visual layout |
| `keyboard_only` | bool | `false` | No mouse; all navigation via Tab, Enter, arrow keys |
| `zoom_level` | float (1.0-4.0) | `1.0` | Browser zoom level; affects visible content and layout |
| `motor_precision` | Level | `high` | `high` = precise clicks, `low` = frequent mis-clicks on small targets |

## Archetypes

Archetypes are named presets that instantiate personas with realistic parameter combinations. They are starting points — teams can modify parameters or create custom personas for their specific needs.

### Maria

**Narrative:** Maria, 58, retired teacher, new to web apps. Reads carefully but doesn't know UI conventions.

| Dimension | Value |
|-----------|-------|
| Tech literacy | `low` |
| Patience | 6 (medium) |
| Domain familiarity | `low` |
| Reading comprehension | `high` |
| Goal clarity | `medium` |
| Accessibility | Default (no assistive tech, full motor precision) |

**Most likely to surface:** `labeling`, `navigation`, `copy` — unclear labels, non-obvious navigation paths, jargon without explanation. She'll try hard but get lost where conventions aren't explicit.

### Jake

**Narrative:** Jake, 26, developer, power user. Impatient with hand-holding. Catches missing keyboard shortcuts, slow flows, and lack of bulk actions.

| Dimension | Value |
|-----------|-------|
| Tech literacy | `high` |
| Patience | 4 (low-medium) |
| Domain familiarity | `high` |
| Reading comprehension | `medium` |
| Goal clarity | `high` |
| Accessibility | Default |

**Most likely to surface:** `layout`, `feedback` — missing keyboard shortcuts, unnecessary confirmation steps, slow multi-step flows that could be streamlined. He'll find the inefficiencies your power users complain about.

### Priya

**Narrative:** Priya, 34, screen reader user, works in finance. Patient because she's used to bad UX. Catches accessibility barriers that block task completion.

| Dimension | Value |
|-----------|-------|
| Tech literacy | `medium` |
| Patience | 8 (high) |
| Domain familiarity | `medium` |
| Reading comprehension | `high` |
| Goal clarity | `high` |
| Accessibility | `screen_reader: true`, `keyboard_only: true` |

**Most likely to surface:** `accessibility`, `layout`, `navigation` — missing ARIA labels, focus traps, non-semantic markup, color-only indicators. She knows what she wants to do — the barrier is whether the UI lets her.

### Carlos

**Narrative:** Carlos, 40, multitasking parent, on mobile. Skims everything, taps wrong targets, gets interrupted mid-flow. Catches failures under distraction and cognitive load.

| Dimension | Value |
|-----------|-------|
| Tech literacy | `medium` |
| Patience | 2 (low) |
| Domain familiarity | `low` |
| Reading comprehension | `low` |
| Goal clarity | `low` |
| Accessibility | `motor_precision: low` |

**Most likely to surface:** `layout`, `error_recovery`, `feedback` — small click targets, hover-dependent UI, forms that clear on error, missing undo. He'll abandon fast and punish interfaces that don't forgive mistakes.

### Yuki

**Narrative:** Yuki, 22, non-native English speaker, student. Catches jargon, idiom-dependent labels, and culturally-specific UI metaphors.

| Dimension | Value |
|-----------|-------|
| Tech literacy | `medium` |
| Patience | 5 (medium) |
| Domain familiarity | `low` |
| Reading comprehension | `low` |
| Goal clarity | `medium` |
| Accessibility | Default |

**Most likely to surface:** `copy`, `labeling`, `navigation` — unexplained domain terminology, unclear option consequences, missing contextual help. She can use any interface — she just doesn't know what the words mean.

> **Note:** The "most likely to surface" mappings are design hypotheses based on the persona dimensions, not validated behavior from a calibrated `reflect()`. Expect the actual distribution of issue categories to shift once the cognitive engine is live and classifying real issues.

## Choosing Your Test Set

Run at minimum 3 dimensionally diverse archetypes per task flow. **Each archetype in your set should be "low" on a different dimension than the others.** Jake (high on everything except patience) is useful as a baseline but doesn't count toward your coverage minimum — he confirms the happy path works, not that the unhappy paths are survivable.

### Recommended Coverage Sets

| Set | Archetypes | Dimensions covered |
|-----|------------|-------------------|
| **General coverage** | Maria + Carlos + Priya | Tech literacy, patience, motor precision, accessibility |
| **Domain-sensitive** | Maria + Yuki + Priya | Tech literacy, domain familiarity, accessibility |
| **Stress test** | Carlos + Yuki + Priya | Patience, domain familiarity, accessibility, motor precision |

### Which set should I pick?

- **"Our users are mostly non-technical"** — General coverage (Maria anchors it)
- **"Our app requires domain knowledge to use"** — Domain-sensitive (Yuki anchors it)
- **"We're worried about users churning from frustration"** — Stress test (Carlos anchors it)
- **"We don't know our users well"** — General coverage (broadest dimensional spread)

### Dimensional diversity principle

Pick archetypes that cover different dimensions, not just different names. Maria + Carlos + Priya covers tech literacy, patience, and accessibility. Maria + Carlos + Yuki covers tech literacy, patience, and domain familiarity — but misses accessibility entirely. Choose based on which gaps matter most for your product.

## Custom Personas

Teams can create custom personas by instantiating `Persona` directly with any parameter combination:

```python
from autouser.persona.models import Persona, AccessibilityProfile, Level

custom_persona = Persona(
    tech_literacy=Level.LOW,
    patience=4,
    domain_familiarity=Level.HIGH,
    reading_comprehension=Level.MEDIUM,
    goal_clarity=Level.LOW,
    accessibility=AccessibilityProfile(
        zoom_level=2.0,
        motor_precision=Level.MEDIUM,
    ),
)
```

Or start from an existing archetype:

```python
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype

maria = Persona.from_archetype(get_archetype("maria"))
maria.patience = 3  # Make Maria less patient for stress testing
```
