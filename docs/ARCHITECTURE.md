# Architecture

## Overview

AutoUser simulates realistic user behavior by running a **plan-execute-reflect** loop inside a real browser. Each iteration captures the user's intent *before* acting and evaluates the outcome *after* acting — producing a behavioral trace that surfaces usability friction from the perspective of a specific persona.

The system has three layers:

```
┌─────────────────────────────────────────────────────┐
│                  SimulationRunner                    │
│         plan → execute → reflect (per step)         │
├──────────────┬─────────────────┬────────────────────┤
│ Persona      │ Cognitive       │ Execution           │
│ Engine       │ Model           │ Layer               │
│              │                 │                     │
│ Archetypes   │ Phase 1: Plan   │ BrowserExecutor     │
│ Parameters   │ Phase 2: Reflect│ Playwright/Chromium │
│ Accessibility│ Emotion model   │ DOM summary + screenshot│
├──────────────┴─────────────────┴────────────────────┤
│                  Report Generator                    │
│       FrictionLog, Issue extraction, Severity ranking│
└─────────────────────────────────────────────────────┘
```

## Data Flow

A single simulation step flows through the system like this:

```
UIState (observation_before)
    │
    ▼
CognitiveEngine.plan(ui_state)
    │  → ActionIntent (action, target, expected_outcome, confidence, persona_thought)
    ▼
BrowserExecutor.execute(intent, step)
    │  → performs action in browser, captures new page state
    ▼
UIState (observation_after)
    │
    ▼
CognitiveEngine.reflect(intent, new_ui_state, observation_before, observation_after, step_number)
    │  → StepResult (actual_outcome, mismatch, emotion, severity, category, reflection)
    ▼
SessionLog.append_step(step_result)
```

The **observation_before/observation_after** separation is load-bearing. It enables:
- Mismatch detection: did what happened match what the persona expected?
- Replay: every step has a complete snapshot pair for debugging
- Calibration: compare simulated friction against real usability study findings

## Layer Details

### Persona Engine

Personas are parameterized along 5 user-facing dimensions:

| Dimension | What it controls |
|-----------|-----------------|
| Tech literacy | Available action vocabulary, prompt complexity |
| Patience / error tolerance | Frustration threshold before give-up (1–10 scale) |
| Domain familiarity | How much context/help text the persona needs |
| Cognitive load tolerance | Reading depth, goal clarity, decision confidence |
| Accessibility profile | Screen reader, keyboard-only, zoom level, motor precision |

**Internal representation:** The code decomposes `cognitive_load_tolerance` into two sub-dimensions — `reading_comprehension` and `goal_clarity` — because they produce different LLM constraints. A user who reads carefully but doesn't know UI conventions (Maria) behaves differently from one who skims everything and has a vague goal (Carlos). A single parameter can't express both axes. The external API presents the unified 5 dimensions via `Persona.from_config()`, which maps `cognitive_load_tolerance` onto both internal sub-dimensions; archetypes set them independently for finer nuance.

**Archetypes** are named presets (Maria, Jake, Priya, Carlos, Yuki) that instantiate personas with realistic parameter combinations. They are pure data — zero runtime branching on archetype names anywhere in the system. The runner and cognitive engine operate on `Persona` parameters only; the archetype name appears only in reports.

`Persona.from_archetype()` deep-copies the accessibility profile to prevent template mutation across runs.

### Cognitive Model

The cognitive engine is the core R&D surface. It runs two phases per step:

**Phase 1 — Plan:** Given the current UI state and persona parameters, declare an `ActionIntent` *before* executing it. This captures what the persona *thinks* will happen (expected outcome), how confident they are, and their internal reasoning (persona_thought).

**Phase 2 — Reflect:** After execution, compare the new UI state against the expected outcome. Produce a `StepResult` with:
- `mismatch`: did the outcome differ from expectation?
- `emotion`: persona's emotional state (CONFIDENT, UNCERTAIN, CONFUSED, FRUSTRATED, SATISFIED)
- `severity` / `category`: if there's an issue, how bad is it and what kind?
- `reflection`: persona's interpretation of what happened

The central challenge is **persona-constrained prompting** — the LLM must degrade its own capabilities to match the persona's tech literacy, reading comprehension, and domain familiarity. A low-tech-literacy persona shouldn't recognize hamburger menus. A screen-reader persona shouldn't reference visual layout.

**Give-up logic:** `should_give_up()` is driven by `affect.summarize()` — pure functions over (history, persona). Frustration accumulates from **objective trace events** (failed actions, clicks that changed nothing, repeated attempts within a 3-step window, backtracking) plus expectation violations (mismatch) at reduced weight; emotion *labels* contribute nothing — they stay color in the log, never termination fuel (the prior emotion-counting mechanism made the measurement its own fuel; see `docs/cognitive-v2-design-review.md` §4). A parallel **simulated-time budget** charges reading cost proportional to visible text × the persona's reading depth (first visit full, revisits cheap) plus per-action motor costs; patience scales both the frustration threshold and the time budget. Current affect also feeds back into behavior: the user prompt carries an in-character feeling/time line, and past half the patience budget the perception filter trims reading depth (frustrated users skim). The **runner** — not the engine — then forces `intent.action = ActionType.GIVE_UP`, executes a final reflect step, and finalizes the session with `terminal_reason="patience_exhausted"`. This boundary is intentional: the cognitive engine is a pure reasoning layer; control flow decisions belong to the orchestration layer. Distinguishing engine-initiated give-up from runner-enforced patience exhaustion in `terminal_reason` is a planned analytics refinement.

### Execution Layer

`BrowserExecutor` wraps Playwright to provide:

- **Action dispatch:** CLICK, TYPE, PRESS, SCROLL, NAVIGATE, BACK, WAIT, GIVE_UP mapped to Playwright calls. PRESS (Tab/Enter/Space/Escape/arrows) enables real keyboard navigation; TYPE with an empty target types into the focused control.
- **State capture:** After each action, captures a `UIState` with:
  - URL and page title
  - Simplified DOM summary (interactive elements + landmarks, capped at 8000 chars)
  - Accessibility-tree snapshot (`aria_snapshot`) — raw material for screen-reader perception
  - The focused element (`focused_element`) — load-bearing for keyboard navigation
  - Visible text (capped at 4000 chars)
  - Screenshot saved to `screenshots/<run_id>/<persona_name>/step_NNN.png`
- **Dialog handling:** Unexpected `alert()` / `confirm()` dialogs are auto-dismissed via async handler
- **Isolation:** Each executor owns its own browser context. `BatchRunner` runs N executors concurrently via `asyncio.gather()` with no shared state.

The DOM summary is a **perception filter** — it extracts only what a real user would notice (interactive elements, landmarks, labels) rather than passing raw HTML to the LLM. Persona-specific filtering lives in `perception.py` and is applied by the runner to engine-facing states only: screen-reader personas perceive the accessibility tree (roles + names, document order) and target via `role=<role>[name="..."]`; raw states stay untouched for predicates, dwell detection, and replay.

**Input fidelity:** personas with low `motor_precision` are harness-enforced — clicks carry gaussian noise around the target center (and can genuinely miss, hitting whatever is adjacent), and typing is real keystrokes with occasional corrected typos so keydown-driven UX reacts. Other personas use `page.fill()` (programmatic, deterministic). See `docs/cognitive-v2-design-review.md` for the constraints-in-the-harness principle.

### Report Generator

`ReportGenerator` consumes the full step history and produces a `FrictionLog`:

- **Outcome classification:** `COMPLETED`, `COMPLETED_WITH_ERRORS`, or `ABANDONED` — determined by terminal reason and mismatch history
- **Issue extraction:** Every step with `mismatch=True` becomes an `Issue` with severity (HIGH/MEDIUM/LOW), category (LABELING, NAVIGATION, ACCESSIBILITY, FEEDBACK, ERROR_RECOVERY, LAYOUT, COPY), the persona's perspective, and the URL/element where it occurred
- **Severity ranking:** Issues sorted HIGH → MEDIUM → LOW for triage priority
- **Persona summary:** Parameters included in the report so readers understand *why* this persona struggled

Severity is **outcome-anchored**: HIGH = blocks task completion, MEDIUM = causes confusion/delay, LOW = minor annoyance. This keeps triage objective.

## Concurrency Model

`BatchRunner` runs multiple personas against the same task concurrently:

```python
results = await asyncio.gather(
    *[run_persona(p) for p in personas],
    return_exceptions=True
)
```

- Each persona gets an isolated `SimulationRunner` with its own `BrowserExecutor`
- Screenshot directories are scoped by `run_id` and persona name — no collisions on reruns or duplicate personas
- Partial failures are captured as `PersonaError` records alongside successful `FrictionLog` results in `BatchResult`
- 4/5 success = 4 logs + 1 error, not a crash

## Key Data Models

| Model | Module | Purpose |
|-------|--------|---------|
| `TaskSpec` | session | What to test (task, success_criteria, start_url, max_steps) |
| `SessionLog` | session | Append-only run record with run_id, persona snapshot, steps, terminal_reason |
| `Persona` | persona.models | Concrete user profile with behavioral parameters |
| `PersonaArchetype` | persona.models | Named preset for common user types |
| `UIState` | cognitive.models | Page snapshot (url, DOM summary, visible text, screenshot) |
| `ActionIntent` | cognitive.models | Phase 1 output: what the persona intends to do and expects |
| `StepResult` | cognitive.models | Phase 2 output: what happened, mismatch detection, emotional response |
| `FrictionLog` | report.models | Complete simulation output with issues and persona perspective |
| `BatchResult` | report.models | Aggregated results from concurrent multi-persona run |

## Design Decisions

**Why phase separation (plan/reflect)?** Without it, the LLM sees the action result simultaneously with choosing the action — it can't express surprise or confusion at an unexpected outcome. Phase separation forces the model to commit to an expectation, then honestly evaluate whether reality matched. This is what produces meaningful mismatch detection.

**Why Pydantic for everything?** Every data structure is a `BaseModel` with field validation. This gives us: type safety at boundaries, JSON serialization for replay/export, validator hooks (e.g., ActionIntent rejects CLICK without a target), and deep-copy semantics for isolation.

**Why DOM summary instead of raw HTML?** Raw HTML is too large and too noisy for LLM context. The DOM summary extracts only perceptible elements (interactive + landmarks) with their labels, roles, and state. This also enables persona-specific perception filtering — different personas "see" different things based on their accessibility profile and tech literacy.

**Why not screenshot-only (vision model)?** Screenshots lose semantic information (ARIA roles, element state, input values). DOM summary + screenshot gives the cognitive engine both the structural and visual signal. The screenshot is primarily for human debugging and report illustration.

## Known Gaps

- **Terminal reason granularity:** `GIVE_UP` vs `patience_exhausted` should be distinct in `terminal_reason` for analytics (planned).
- **Input fidelity (partial):** low-`motor_precision` personas get noisy clicks and keystroke typing with typos; other personas still use `page.fill()`. Hover/tooltips and explicit re-read actions are not modeled yet.
- **Overall impressions:** Report `overall_impressions` is generated deterministically from the trajectory; a richer LLM-written synthesis remains future work.
- **Browser/account scope:** AutoUser drives isolated Playwright browser contexts. It does not yet reuse a user's live Chrome profile, retrieve credentials from macOS Keychain, or handle passkeys/MFA/CAPTCHA handoffs.
- **Calibration:** No validation against real usability study data yet. The contracts support replay and comparison, but the calibration pipeline itself is post-v1.
