# RFC 0001: Targeted Persona Matrix and Evidence Scoring

- **Status:** Draft
- **Date:** 2026-06-13
- **Owner:** TBD
- **Related surfaces:** `autouser study`, `autouser explore`, `StudyResult`,
  `study.md`

## Summary

Add first-class goal and ICP targeting to AutoUser, then use that target to run
a purposeful persona matrix and render an evidence-backed scorecard.

Today AutoUser can run multiple personas and aggregate findings by reproduction
frequency. That is the right substrate, but the operator still has to decide:

- which personas matter for this product,
- what user/business goal the run is measuring,
- which findings are decisive versus incidental,
- and how to compare two products or two versions of the same flow.

This RFC proposes three additions:

1. **TargetSpec**: structured product goal, ICP, journey stage, and success
   criteria carried through reports.
2. **PersonaMatrixSpec**: a goal-aware persona set with explicit slots and
   rationale, instead of an unstructured list of persona names.
3. **EvidenceScorecard**: a scored decision surface backed by concrete run
   evidence, reproduction frequency, objective predicates, screenshots, and
   audit status where available.

The design extends `study`; it does not replace the current single-persona
`run` flow or the existing open-ended `explore` mode.

## Motivation

An early single-persona trial showed that a single persona can be high-signal but incomplete.
Maria found a real public-site clarity issue, but did not exercise signup,
activation, daemon install, or first-value generation. A later trial
showed a different need: evaluating agent products requires following a full
journey from marketing to auth to local runtime connection to real artifact
creation and recovery.

AutoUser should become a repeatable UX lab:

- run the right personas for a stated ICP,
- test against an explicit goal,
- preserve persona-specific qualitative evidence,
- and produce a score that explains itself.

The score is not meant to hide judgment behind a number. It is a triage layer:
the reader should be able to click from any score to the exact persona, step,
URL, issue, quote, and screenshot that caused it.

## Goals

- Let an operator specify the product goal and ICP in CLI/API inputs.
- Generate or select a small persona matrix from that target.
- Explain why each persona is included.
- Aggregate outcomes into a scorecard that is evidence-backed and reproducible.
- Support side-by-side comparisons across products, branches, or versions.
- Preserve current `study` behavior for existing callers.

## Non-goals

- Replace real user research.
- Claim statistical validity from small synthetic studies.
- Generate arbitrary demographic personas as the default behavior.
- Turn AutoUser into a generic product-review LLM with no browser evidence.
- Make score weights opaque or learned in v1.

## Current State

AutoUser already has:

- `autouser run` for one or more personas against one task.
- `autouser explore` for open-ended first-session product understanding.
- `autouser study` for `M personas x N seeds`, with issue clusters ranked by
  reproduction frequency.
- Persona archetypes in `docs/PERSONA_FRAMEWORK.md`.
- `study.md` rendering with findings, persona stats, and run matrix.
- `score` for deterministic benchmark precision/recall against planted issues.

The missing pieces are target intent and decision scoring. Current `study.md`
answers "what happened across these personas?" It does not answer "for this ICP
and this product goal, how well did the journey work?"

## Proposed UX

### CLI: targeted study

```bash
autouser study \
  --url https://example.com/ \
  --task "Connect a local runtime and get one agent to build a small scoped artifact." \
  --goal "A technical founder understands the product, signs up, connects a local runtime, and gets one agent to build a small scoped artifact." \
  --icp "Technical founder or operator evaluating AI-agent team tools for serious work." \
  --journey-stage first_value \
  --criteria "A generated artifact exists locally and the product reports the task complete." \
  --success-selector "[data-testid='task-complete']" \
  --persona-matrix auto \
  --matrix-size 4 \
  --seeds 1 \
  --score
```

### CLI: open-ended targeted exploration

```bash
autouser explore \
  --url https://example.com/ \
  --goal "Understand what this product is and whether it is worth trying." \
  --icp "Nontechnical small-business owner shopping for customer-support automation." \
  --persona-matrix auto \
  --matrix-size 3
```

In v1, `--task` remains the user-facing `TaskSpec.task` for `study`; `--goal`
is target metadata used for matrix selection, score weighting, and reporting.
`explore` can accept `--goal` / `--icp` for targeted matrix selection, but
scorecard output starts with `study` because `explore` lacks an objective task
completion denominator.

### Output files

For a targeted study:

```text
<study_id>/
  study.md
  study.json
  scorecard.md
  scorecard.json
  target.json
  matrix.json
  <persona>-s<seed>/
    report.md
    journey.md
    friction_log.json
    step_000.png
```

`study.md` remains the evidence synthesis. `scorecard.md` is the decision
surface.

## TargetSpec

Add a target object that can be attached to a `TaskSpec` or `StudyResult`.

```python
class JourneyStage(str, Enum):
    UNDERSTAND = "understand"
    SIGNUP = "signup"
    ACTIVATION = "activation"
    FIRST_VALUE = "first_value"
    RECOVERY = "recovery"
    RETENTION = "retention"


class TargetSpec(BaseModel):
    goal: str
    icp: str
    journey_stage: JourneyStage = JourneyStage.UNDERSTAND
    product_context: str = ""
    success_criteria: str
    dealbreakers: list[str] = Field(default_factory=list)
    comparison_label: str | None = None
```

`goal` is what the operator wants the user to accomplish. `icp` is who matters.
`journey_stage` drives default matrix composition and scoring weights.
`dealbreakers` are optional operator-supplied conditions that should force a
low score even if the journey completes, such as "must not expose raw auth
tokens" or "must not require command-line setup for this ICP."

For compatibility, `TargetSpec.success_criteria` can be derived from existing
`TaskSpec.success_criteria` when omitted.

## PersonaMatrixSpec

A matrix is not just a list of names. It should tell the reader what each slot
is testing.

```python
class MatrixSlotKind(str, Enum):
    ICP_CORE = "icp_core"
    ICP_EDGE = "icp_edge"
    LOW_CONTEXT = "low_context"
    LOW_LANGUAGE_CONTEXT = "low_language_context"
    LOW_PATIENCE = "low_patience"
    ACCESSIBILITY = "accessibility"
    POWER_USER = "power_user"
    SKEPTICAL_BUYER = "skeptical_buyer"


class PersonaMatrixSlot(BaseModel):
    kind: MatrixSlotKind
    persona_name: str
    rationale: str
    required: bool = True
    seeds: int = 1


class PersonaMatrixSpec(BaseModel):
    target: TargetSpec
    slots: list[PersonaMatrixSlot]
    selection_mode: Literal["explicit", "auto", "template"]
```

### Default matrix templates

For v1, use existing archetypes rather than generating new personas by LLM.
This keeps behavior deterministic and grounded in the documented persona
framework.

| Stage | Default slots | Default personas |
|---|---|---|
| `understand` | ICP core, low-context, low-language-context, accessibility | Jake, Maria, Yuki, Priya |
| `signup` | ICP core, low-patience, accessibility, low-context | Jake, Carlos, Priya, Maria |
| `activation` | ICP core, low-context, low-patience, accessibility | Jake, Maria, Carlos, Priya |
| `first_value` | ICP core, low-context, low-patience, accessibility | Jake, Maria, Carlos, Priya |
| `recovery` | ICP core, low-patience, low-context, accessibility | Jake, Carlos, Maria, Priya |

The auto-selector maps ICP text to default archetypes conservatively:

- If ICP mentions developer, founder, operator, technical, API, CLI, or agent
  tooling: core = `jake`.
- If ICP mentions nontechnical, consumer, small business, teacher, personal:
  core = `maria` or `yuki`, depending on domain-language complexity.
- Always include at least one stress persona (`carlos`) and one accessibility
  persona (`priya`) for activation/signup/recovery stages unless
  `--matrix-size` is too small.
- V1 templates avoid duplicate archetypes in a single-seed matrix because
  current study artifacts and aggregation are keyed by archetype plus seed.
  If the ICP-core choice would duplicate another slot, the selector picks the
  next compatible archetype for that slot.

If `--persona` is provided explicitly, preserve current behavior and render
those personas as an explicit matrix with default rationales.

## EvidenceScorecard

The scorecard answers: for this target, how well did the product journey work?

```python
class ScoreBand(str, Enum):
    STRONG = "strong"
    ACCEPTABLE = "acceptable"
    WEAK = "weak"
    BLOCKED = "blocked"


class EvidenceRef(BaseModel):
    persona_name: str
    seed: int
    step: int | None = None
    url: str | None = None
    issue_index: int | None = None
    screenshot_path: str | None = None
    quote: str | None = None


class ScoreDimension(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    band: ScoreBand
    rationale: str
    evidence: list[EvidenceRef]


class EvidenceScorecard(BaseModel):
    target: TargetSpec
    matrix: PersonaMatrixSpec
    overall_score: int = Field(ge=0, le=100)
    overall_band: ScoreBand
    confidence: Literal["low", "medium", "high"]
    dimensions: list[ScoreDimension]
    decisive_findings: list[EvidenceRef]
```

Scorecard computation is intentionally not pure over today's `StudyResult`
alone. The scorer needs the study aggregate plus target/task metadata and
optional sidecar facts that are currently written outside `study.json`.

```python
class ScorecardInput(BaseModel):
    study: StudyResult
    target: TargetSpec
    matrix: PersonaMatrixSpec
    success_predicate_present: bool = False
    audit_status_by_cell: dict[str, Literal["absent", "complete"]] = Field(
        default_factory=dict
    )
```

Slice 3 can either add these fields directly to `StudyResult`, or load
`target.json`, `matrix.json`, task metadata, and per-cell `audit.json` sidecars
before constructing `ScorecardInput`. It must not claim that `StudyResult`
alone contains objective predicate presence or audit coverage until the model is
extended to serialize those facts.

### Dimensions

Use five dimensions in v1:

1. **Goal completion**: did target personas complete the objective?
2. **ICP fit**: did ICP-core personas understand value, audience, and next step?
3. **Activation friction**: how much setup/auth/install friction appeared?
4. **Recovery quality**: did follow-up/retry/error paths recover cleanly?
5. **Trust and polish**: did brand, security, UX, or artifact quality undermine
   confidence?

Not every dimension should have equal weight in every stage.

| Stage | Goal | ICP fit | Activation | Recovery | Trust/polish |
|---|---:|---:|---:|---:|---:|
| `understand` | 20 | 40 | 5 | 10 | 25 |
| `signup` | 35 | 15 | 20 | 10 | 20 |
| `activation` | 30 | 10 | 35 | 10 | 15 |
| `first_value` | 35 | 15 | 20 | 15 | 15 |
| `recovery` | 25 | 10 | 10 | 40 | 15 |

### Scoring rules

Scoring should be formula-first and evidence-linked.

For every dimension:

1. Start from the base score.
2. Apply only penalties that have at least one `EvidenceRef`.
3. Apply any per-penalty caps before summing repeated penalties.
4. Clamp the final dimension score with `max(0, min(100, raw_score))`.
5. Compute the weighted overall raw score from clamped dimension scores using
   the stage weights, round half up with `floor(raw_overall + 0.5)`, then clamp
   the rounded integer to `0..100` before constructing `EvidenceScorecard`.

Overall rounding is intentionally not Python's banker rounding. Use half-up so
`74.5` serializes as `75`; tests should cover fractional weighted scores before
Pydantic validates `EvidenceScorecard.overall_score`.

Tests should assert both the raw underflow case and the final Pydantic
validation path: a dimension with enough evidence-backed penalties to subtract
past zero still serializes as `score=0`, never a negative value.

### Band mapping

After clamping, dimensions and overall score use the same deterministic score
thresholds:

| Score | Band |
|---:|---|
| `85..100` | `strong` |
| `70..84` | `acceptable` |
| `40..69` | `weak` |
| `0..39` | `blocked` |

Blocked overrides are deliberately narrow and evidence-backed:

- The **Goal completion** dimension is `blocked` if no target run completed, or
  if any ICP-core run records a false success against an objective predicate.
- Any dimension is `blocked` if an operator-supplied dealbreaker maps to that
  dimension and has at least one `EvidenceRef`.
- The **overall** band is `blocked` if the Goal completion dimension is blocked,
  or if any operator-supplied dealbreaker has evidence.

Otherwise, `band` and `overall_band` are derived only from the clamped numeric
score. Missing audit status or missing success-predicate metadata lowers
confidence; it does not create a blocked band by itself.

#### Goal completion

Base:

```text
100 * completed_target_runs / target_runs
```

`target_runs` is computed from the planned matrix, not from log availability:
count one target run for each required `PersonaMatrixSlot` seed. A crashed cell
that is preserved as `RunRecord.error` with `outcome=None` remains in this
denominator and counts as not completed. Non-required slots are excluded from
the base denominator and can still contribute qualitative evidence or
dimension-specific penalties when cited.

`completed_target_runs` counts runs whose report-facing `TaskOutcome` is
`completed` or `completed_with_errors`. A `completed_with_errors` run reached
the objective for this base formula, but its errors should still feed
evidence-backed penalties in activation friction, recovery quality, or trust and
polish. `abandoned`, `timed_out`, and `error` do not count as completed.

Penalties:

- `-40` for any ICP-core false success.
- `-25` if every completed run required an operator workaround not visible in
  the product.
- `-20` if a required success predicate was absent and completion was persona
  belief only.

#### ICP fit

Start at 100 for ICP-core runs, then subtract:

- `-35` if an ICP-core run abandoned before understanding the product.
- `-20` for each reproduced `copy`, `labeling`, or `navigation` cluster that
  affects ICP-core or low-context slots.
- `-15` if the final mental model contradicts the target ICP or product goal.

#### Activation friction

Start at 100, then subtract:

- `-30` for auth dead-end or external verification block.
- `-25` for install/setup step with unclear recovery path.
- `-20` for raw token/secret exposure in UI or screenshots.
- `-15` for brand mismatch across auth, emails, commands, or daemon logs.
- `-10` for each extra modal/prompt that interrupts the activation path, capped
  at `-30` for this repeated-modal subtype.

#### Recovery quality

Start at 100 for flows with an explicit follow-up or error condition.

- `+0` if no recovery path was tested; mark confidence low.
- `-35` if follow-up is ignored or routes to the wrong agent/persona.
- `-25` if recovery requires manual intervention outside the product.
- `-20` if the product reports success but browser/local QA finds a user-visible
  defect.
- `-10` if progress state is truthful but too opaque for a waiting user.

#### Trust and polish

Start at 100, then subtract:

- `-30` for secret/token exposure.
- `-25` for product self-report contradicting browser evidence.
- `-20` for serious brand mismatch.
- `-15` for visible UX clutter or prompt stacking during the core journey.
- `-10` for harmless but noisy runtime issues such as favicon 404s, capped at
  `-20` for repeated harmless runtime noise.

All penalties must point to at least one `EvidenceRef`. If the renderer cannot
point to evidence, it cannot apply the penalty.

### Confidence

Confidence is about the score, not the product.

| Confidence | Requirement |
|---|---|
| `high` | At least 4 matrix slots, objective success predicate, no crashed cells, evidence refs for every penalty, audit run for high/blocking findings |
| `medium` | At least 3 matrix slots, no crashed ICP-core cell, evidence refs for every major penalty |
| `low` | Fewer than 3 slots, missing objective predicate on completion-sensitive goal, crashed ICP-core cell, or untested recovery dimension |

## Rendered Scorecard Shape

```markdown
# Scorecard — agent-product first-value journey

**Goal:** Technical founder understands the product, connects a local runtime, and gets
one agent to build a scoped artifact.
**ICP:** Technical founder or operator evaluating AI-agent team tools.
**Matrix:** 4 personas x 1 seed. Core ICP: Jake.
**Overall:** 72/100 — Acceptable, confidence medium.

| Dimension | Score | Why |
|---|---:|---|
| Goal completion | 90 | Agent built the artifact and accepted QA follow-up. |
| ICP fit | 70 | Technical path is understandable, but public use-case copy loses low-context users. |
| Activation friction | 55 | Brand mismatch and onboarding modal stacking interrupted setup. |
| Recovery quality | 85 | Review reply reached assigned agent without explicit mention. |
| Trust and polish | 60 | Product naming mismatch and daemon command visibility hurt trust. |

## Decisive Evidence

1. Maria abandoned public exploration after the job-hunting use case stayed in
   agent setup instructions rather than plain-language value explanation.
2. Devon respected the scoped build directory, produced all four files, and
   fixed a favicon issue after QA.
3. Onboarding prompts repeatedly interrupted the actual channel/task path.
```

## Comparison Mode

Once two scorecards exist, comparison is a pure render step:

```bash
autouser compare \
  --left ./runs/baseline/scorecard.json \
  --right ./runs/candidate/scorecard.json
```

The comparison report should show:

- overall score and confidence,
- dimension deltas,
- decisive evidence on each side,
- "winner by goal" rather than generic winner,
- caveats where confidence differs.

This can be M2 for the RFC. The important part is that scorecards are structured
enough to compare without rerunning browser sessions.

## Implementation Plan

### Slice 1: Target metadata and matrix rendering

- Add `TargetSpec`, `PersonaMatrixSpec`, and matrix slot models.
- Add CLI args: `--goal`, `--icp`, `--journey-stage`, `--persona-matrix`,
  `--matrix-size`.
- Keep existing `--task` required for `study`; `--goal` does not replace
  `TaskSpec.task` in v1.
- Preserve current behavior when these args are absent.
- Render target and matrix rationale at the top of `study.md`.
- Write `target.json` and `matrix.json`.

Exit criteria:

- Existing `study` tests pass unchanged.
- A targeted study renders goal, ICP, stage, matrix slots, and rationales.

### Slice 2: Auto matrix selector

- Implement deterministic `select_matrix(target, size)` using existing
  archetypes and stage templates.
- If selector confidence is low, warn and fall back to `maria + carlos + priya`.
- Add unit tests for technical-founder, nontechnical-consumer, and accessibility
  sensitive ICPs.

Exit criteria:

- `--persona-matrix auto --matrix-size 4` picks stable personas with readable
  rationales.
- Explicit `--persona` still wins.

### Slice 3: Evidence scorecard

- Add `EvidenceScorecard` models.
- Add `study --score` as the opt-in CLI flag for writing `scorecard.md` and
  `scorecard.json`; without it, targeted studies still write target/matrix
  metadata only.
- Implement formula-first score computation from `ScorecardInput`, built from
  `StudyResult` plus target/matrix metadata, success-predicate presence, and
  per-cell audit sidecars where available.
- Render `scorecard.md` and `scorecard.json`.
- Every penalty must include an `EvidenceRef`.
- Clamp dimension and overall scores to `0..100` after applying capped,
  evidence-backed penalties and half-up overall rounding before Pydantic
  validation.

Exit criteria:

- Scorecard renders dimensions, score, band, confidence, and decisive evidence.
- Tests prove penalties are not applied without evidence refs.
- Tests prove score underflow clamps to `0` and `ScoreDimension` validation
  still passes.
- Tests prove fractional weighted overall scores use half-up rounding before
  `EvidenceScorecard` validation.
- Tests prove goal completion counts `completed` and `completed_with_errors`,
  and excludes `abandoned`, `timed_out`, and `error`.
- Tests prove goal completion denominator includes crashed required matrix
  cells as non-completions, and excludes non-required slots.
- Tests prove `band` and `overall_band` are derived from the threshold table,
  with blocked overrides applied only when their evidence exists.

### Slice 4: Compare renderer

- Add `autouser compare --left --right`.
- Render dimension deltas and decisive evidence.
- No browser/LLM dependency.

Exit criteria:

- Comparison works from two checked-in fixture scorecards.

## Compatibility

- Existing `run`, `explore`, `study`, and `score` commands remain valid.
- Existing `StudyResult` consumers should tolerate optional fields:
  `target_spec`, `matrix_spec`, `scorecard`.
- Targeted scoring is opt-in behind `--score` for the first implementation.
- `score` against planted benchmarks remains separate from product journey
  scoring. Name collision should be avoided in docs: "benchmark score" vs
  "journey scorecard."

## Risks

### Risk: false precision

A single overall score can look more authoritative than it is.

Mitigation:

- Always render confidence next to score.
- Make evidence refs mandatory for penalties.
- Prefer dimension scores over a single number in prose.
- Render "Not tested" where dimensions lack evidence.

### Risk: auto matrix overfits ICP text

The selector may infer too much from a short ICP description.

Mitigation:

- Use conservative keyword mapping in v1.
- Show matrix rationale prominently.
- Let explicit `--persona` override auto selection.
- Warn when selector confidence is low.

### Risk: scoring duplicates benchmark scoring

AutoUser already has deterministic benchmark precision/recall.

Mitigation:

- Keep benchmark scoring under `autouser score`.
- Treat journey scorecard as product UX decision scoring, not model precision.
- Link docs clearly.

### Risk: target goal leaks into persona behavior too strongly

If personas know the operator's business goal too explicitly, they may act like
evaluators rather than real users.

Mitigation:

- Feed personas only the user-facing task and success criteria.
- Use `TargetSpec` primarily for matrix selection and scoring.
- Do not inject internal scoring weights into cognitive prompts.

## Open Questions

1. Should v1 support custom ICP-derived personas, or only deterministic
   selection among archetypes?
2. Should score weights be configurable by CLI, or only by journey stage?
3. Should targeted `explore` gain scorecards later, and if so what replaces the
   objective completion denominator?
4. Should operator-supplied dealbreakers be hard gates or weighted penalties?
5. Should audit status be required for high-confidence scorecards, or just
   recommended?

## Recommendation

Ship this in three steps:

1. Target metadata and matrix rationale in `study.md`.
2. Deterministic auto matrix selection from goal + ICP + stage.
3. Evidence scorecard with mandatory evidence refs and confidence labels.

That gets AutoUser from "a persona reported friction" to "for this ICP and
goal, here is the evidence-backed state of the journey."
