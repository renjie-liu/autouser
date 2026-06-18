# Milestones

> **Owner:** PM. Engineering, UX, and stakeholder reviewers.

## Overview

AutoUser v1 is structured as five sequential milestones. Each has a concrete acceptance gate — the milestone is done when the gate passes, not when the code is merged. Later milestones depend on earlier gates passing, so they are sequential by design.

## M1: Scaffold & Contracts

**Goal:** Establish the data model and interface contracts that all subsequent work builds on.

**Deliverables:**
- Pydantic models for `Persona`, `PersonaArchetype`, `TaskSpec`, `SessionLog`, `ActionIntent`, `StepResult`, `UIState`, `FrictionLog`, `Issue`, `BatchResult`
- Enum definitions: `ActionType`, `Emotion`, `Severity`, `IssueCategory`, `TaskOutcome`, `Level`
- `CognitiveEngine` interface with stubbed `plan()`, `reflect()`, `should_give_up()`
- `PersonaArchetype` registry with 5 named presets (Maria, Jake, Priya, Carlos, Yuki)
- Full test coverage of model validation and archetype isolation

**Gate:** All contracts compile, validate, and serialize correctly. Tests pass. No runtime behavior yet — stubs raise `NotImplementedError`.

**Status:** Complete (PR #2 merged).

## M2: Execution Layer

**Goal:** Wire up browser automation so the system can observe and act on real web pages.

**Deliverables:**
- `BrowserExecutor` with Playwright/Chromium integration
- Action dispatch for all `ActionType` values (CLICK, TYPE, SCROLL, NAVIGATE, BACK, WAIT, GIVE_UP)
- `UIState` capture: URL, page title, DOM summary, visible text, screenshot
- `SimulationRunner` orchestrating the plan-execute-reflect loop
- `BatchRunner` for concurrent multi-persona simulation
- Dialog auto-dismiss handler
- Screenshot directory scoping by `run_id` and persona name

**Gate:** `SimulationRunner` completes a multi-step task on a reference site using hardcoded (non-LLM) action sequences. `BatchRunner` runs 3+ personas concurrently with isolated browser contexts and no cross-contamination. Screenshots and session logs are correctly scoped.

**Status:** Complete (PR #4 merged).

## M3: LLM Constraint Spike

**Goal:** Prove that persona-constrained LLM prompting produces measurably different behavior across personas.

**Deliverables:**
- LLM integration for `CognitiveEngine.plan()` and `reflect()`
- Persona-constrained prompting: the LLM degrades its own capabilities to match persona parameters
- `ActionIntent.confidence` as a genuine model self-assessment (not hardcoded)
- Reference task with all 5 personas producing step logs

**Gate:** Simulated personas produce statistically distinguishable distributions on at least 3 of 4 behavioral metrics:
1. **Action confidence distribution** — mean and variance of `ActionIntent.confidence` per persona
2. **Exploration breadth** — unique elements interacted with before reaching target
3. **Non-advancing action rate** — actions that don't move toward task completion
4. **Steps before abandonment** — correlated with patience dimension

**Metric 4 (abandonment) must be among the three.** It is the most directly parameterized behavior — if it doesn't separate, the constraint mechanism is fundamentally broken.

The "3 of 4" threshold gives the spike room to discover that one metric doesn't separate cleanly. M3 asks "do personas behave differently?" not "do they behave realistically?" — calibration is M5.

**Diagnostic plan if confidence doesn't separate:**
1. Check whether persona reasoning text (persona_thought) differentiates even when scores don't
2. If reasoning differentiates but scores don't, restructure the prompt to require explicit uncertainty reasoning before scoring
3. If neither differentiates, the constraint prompts aren't reaching the model's decision process — redesign with few-shot persona exemplars

**Status:** Substrate implemented: LLM providers, scaffolded confidence, metric extraction, study aggregation, and cross-provider parity harness exist. The remaining gate is live statistical validation across the reference matrix.

## M4: Cognitive Engine v1

**Goal:** Full cognitive engine producing actionable friction logs from real web tasks.

**Deliverables:**
- Production-quality `plan()` and `reflect()` implementations
- Issue extraction with severity and category classification
- `overall_impressions` generation
- `ReportGenerator` producing complete `FrictionLog` output
- End-to-end simulation: persona + task → friction log

**Gate:** 3 different personas produce friction logs on 2 different reference tasks (6 total logs). Each log contains:
- At least one correctly categorized issue (category matches human judgment)
- Severity assignments that are outcome-anchored (HIGH = blocks task, MEDIUM = degrades experience, LOW = minor friction)
- `overall_impressions` that accurately reflect the persona's experience trajectory

Human review of all 6 logs confirms they are actionable — a product team could triage and fix the flagged issues from the log alone.

**Status:** Implementation substrate complete: production `plan()`/`reflect()`, issue extraction, deterministic overall impressions, report/journey artifacts, and end-to-end CLI flows exist. The remaining gate is the human review of the six reference logs.

## M5: Calibration & Validation

**Goal:** Establish trust in AutoUser's output by validating against real usability data.

**Deliverables:**
- Reference task suite with known usability issues (ground truth)
- Precision and recall metrics per persona per task:
  - **Precision:** of the issues AutoUser flags, what fraction are real usability issues? (measured against ground truth)
  - **Recall:** of the known usability issues, what fraction does AutoUser find?
- Match definition: an AutoUser issue "matches" a ground truth issue if it identifies the same element/flow AND assigns a severity within one tier
- Cross-persona analysis: do different personas surface different subsets of the ground truth?

**Gate:**
- Precision ≥ 70% across all personas (AutoUser doesn't hallucinate issues)
- Recall ≥ 50% across the union of all personas (collectively, the personas find at least half the known issues)
- At least 3 personas surface at least one unique issue not found by the others (dimensional diversity produces coverage, not redundancy)

**Status:** Partially implemented substrate: planted-issue benchmarks, union precision/recall scoring, and false-positive audit exist. Real usability-study calibration and element-level matching remain.

## Post-v1 candidates

These are tracked but not committed to v1 scope:

- **Cross-persona synthesis report** — automated comparison of friction logs across personas, weighted by user mix. Conditional on M5 calibration data proving which persona signals are most reliable.
- **Confidence-based behavioral annotation** — surfacing hesitation/certainty as qualitative annotations in friction logs ("Maria hesitated here"). Conditional on M3 proving confidence is a trustworthy signal. Annotations should use persona-relative baselines, not absolute thresholds.
- **CRITICAL severity tier** — adding a 4th severity level to distinguish "blocks task completion" from "significantly degrades experience." Revisit after M4 when real severity assignments reveal whether the 3-level scale is sufficient.
- **Input fidelity** — keystroke simulation for typing speed and motor precision errors (replacing `page.fill()`).
- **Category taxonomy review** — evaluate whether the current 7 issue categories (`labeling`, `navigation`, `accessibility`, `feedback`, `error_recovery`, `layout`, `copy`) are the right cuts based on M4 classification data. The UX-designed taxonomy (Nielsen-grounded) may be more intuitive for report consumers.
