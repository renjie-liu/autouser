# Cognitive Architecture v2 — Design Review & Direction

> **Status:** Direction accepted 2026-06-12. All seven sequenced slices
> implemented and live-validated (memory + perception + trajectory; action
> fidelity + insights; objective affect + simulated time; multi-seed studies;
> exploration mode; planted-issue benchmarks + false-positive audit;
> dual-process fast/slow loop). The perception-fidelity fixes from the live
> exploration run are folded in. Section markers below record per-slice state.
>
> **Reframed goal (owner, 2026-06-12):** use an agent to simulate human thinking
> and behavior, explore an unknown system, interact with it, and get feedback.
> This is broader than "task-based usability testing" — it adds open-ended
> exploration and makes the *experience trajectory* (thoughts, beliefs, emotions
> over time) a first-class deliverable, not just the extracted issues.

## Diagnosis

The v1 design simulates a **decision policy**, not a **mind**. Each step, a
stateless LLM call receives the current page plus three one-line memories and
picks an action under prose constraints. Missing: an evolving mental model,
accumulating emotional state, perception limits, and time. And most behavioral
constraints live in the wrong place — they are prompt instructions asking the
model to *pretend* to be limited, rather than harness mechanics that *make* it
limited.

### The governing principle: constraints belong in the harness, not the prompt

A model told "you don't notice subtle UI" still receives the full 8KB element
inventory; a model that is only *given* what the persona would perceive doesn't
need to act. Pretend-limitations are exactly where roleplay drifts from human
reality, and M5 calibration will expose that drift. Preference order for
implementing any persona dimension:

1. **Environment-enforced** — the harness makes the limitation real
   (perception filtering, motor noise on clicks, memory caps).
2. **Capability-removed** — the action/percept simply doesn't exist for this
   persona (no TAB action ⇒ add one; SR persona never receives layout).
3. **Prompt-instructed** — last resort, for genuinely cognitive traits
   (interpretation style, emotional disposition).

The live validation run (2026-06-12) demonstrated the asymmetry: no
prompting fixed mis-targeting until the DOM summary itself changed
(`label=` → real attribute names).

## Design changes

### 1. Mental model — the persona remembers and believes ✅ slice 1

Humans explore by building a map. v1 remembered three one-liner actions; it
could not represent *getting lost* — which is a mismatch between mental map and
reality, the most important usability signal there is.

Implemented split, following the harness-first principle:

- **Places (mechanical).** Distinct `(title, url)` pairs visited, tracked by
  the engine from history with zero LLM involvement. Humans don't forget they
  visited a page; the harness shouldn't simulate fake amnesia.
- **Beliefs (interpretive).** `reflect()` now returns an optional
  `mental_note` — one short, in-character sentence about how the *site* works.
  Notes accumulate and are fed back into every `plan()` prompt
  ("Your Notes About This Site"). Beliefs can be wrong; wrong beliefs that
  persist are findings.

Future (slice 2+): structured beliefs with confidence and
held/confirmed/contradicted status; sub-goal stack; **mental-model diff** as a
deliverable — "after ten minutes, users believe your product works like X."

### 2. Perception filter — personas perceive differently ✅ slice 1 (screen reader)

ARCHITECTURE.md named the DOM summary as "the integration point for
persona-specific perception filtering"; v1 never implemented it — every persona
received the same element inventory. Priya was told "you use a screen reader"
while being handed the visual summary: roleplay, not simulation.

Implemented: `perception.perceive(state, persona)` applied by the runner to
every engine-facing state (raw states stay untouched for predicates, dwell
detection, and replay). Screen-reader personas now perceive the **accessibility
tree** (role + accessible name, document order, captured per-state via
`aria_snapshot`) instead of the element list, and target via the
`role=<role>[name="..."]` engine. An element with no accessible name is
genuinely unreachable — if the page's ARIA is broken, the persona's perceived
page is broken, and the struggle IS the finding.

Future: above-the-fold perception gated on actual scrolling (today scroll
exists but full page text is granted regardless); salience-weighted element
lists for skimmers; reading-depth cuts for low reading comprehension.

### 3. Action fidelity — implement the narrated behaviors ✅ slice 2 (keyboard, motor, typing)

- ✅ Keyboard navigation: new `press` action (Tab/Shift+Tab/Enter/Space/Escape/
  arrows) with the focused element captured into every `UIState` and rendered
  as a "Focused control" line — Tab/Enter are blind without it. `type` with an
  empty target types into the focused control; typing with nothing focused is
  an `action_error` (the real failure mode), not a validation error.
- ✅ Motor noise: low `motor_precision` clicks carry gaussian noise around the
  target center at the executor (sigma ∝ target size — small dense targets are
  genuinely harder) and can miss, hitting whatever is adjacent. Carlos's "taps
  wrong targets" is now mechanical truth.
- ✅ Keystroke typing for low-precision personas: real key events with
  occasional adjacent-key typos corrected by Backspace — keydown-driven UX
  observes the fumble. Other personas keep deterministic `page.fill`.
- Remaining: hover (tooltips, menus) and re-read as distinct actions;
  typing-speed time cost (lands with the affect/time slice, §4).

### 4. Affect state + simulated time, replacing the frustration counter ✅ slice 3 (interruptions deferred)

Two defects in the old `should_give_up()` counting LLM-labeled emotions:
(a) the *measurement* (self-reported confusion) was also the *mechanism*
(termination fuel) — circular through one model's self-report; (b) humans
budget minutes, not steps.

Implemented in `affect.py` (pure functions over history + persona, replayable
from any recorded session):

- ✅ Frustration from **objective** trace events: failed actions (1.0),
  no-effect clicks/presses via before/after state comparison (1.0), repeated
  (action, target, input) within a 3-step window (0.75), backtracking (0.5),
  plus expectation violations — mismatch — at reduced weight (0.5; judged
  against the page, but still a model judgment). Emotion labels contribute
  **zero**; a test pins that a frustrated-labeled step with no objective
  signal is not fuel.
- ✅ Simulated clock: reading costs visible-text length × the persona's
  reading depth (Maria full, Carlos 20%) at ~25 chars/s, first visit full
  and revisits cheap, plus per-action motor costs (typing per-char, slower
  for low motor precision). Patience scales the frustration threshold AND
  a time budget (90s per patience point); either exhausting abandons.
- ✅ Behavior modulation: the user prompt carries an in-character affect
  line ("How you're feeling: … about N minute(s)") instead of the numeric
  counter, and past half the patience budget the perception filter trims
  reading depth — frustrated users skim, which makes them miss more.
- Deferred: interruption events (Carlos's mid-flow distraction) — needs a
  distractibility signal; deriving it from `goal_clarity` overloads that
  dimension and adding a persona field belongs with the #5/#7 dimension-
  mapping work. Constants are provisional v1 calibration; tests pin the
  mechanism, not the literals.

### 5. Dual-process loop — reflect on surprise, not on schedule ✅ slice 6

Humans are System 1 most of the time; deliberate reasoning fires on prediction
error. v1 ran full plan + full reflect every step — every persona a careful
deliberator, two LLM round-trips per step.

Implemented as an opt-in `fast_mode` (`--fast`) on the engine: `reflect()`
short-circuits to a deterministic System-1 result (no LLM) on a cruising step,
and escalates to the full LLM reflect on any of — surprise (`action_error`),
no-effect (page unchanged), low plan confidence (< 0.55), a *plausible
completion* (success-criteria keywords present on the page), **first encounter
of a page** ((url, page_title) not seen before — people read on arrival, then
cruise), or exploration mode. The novelty gate was added after live validation:
naive fast mode flipped careful-reader Priya from completed to abandoned on the
phone-format trap because it skipped the first-sight reflection where she'd have
recorded the format insight; deliberating on arrival restored the outcome while
still saving the reflect call on the routine middle step. The completion gate is load-bearing: it's what keeps success detection
from being lost on the fast path (predicate-gated runs detect success on raw
state regardless; soft-gate runs rely on this escalation). The fast result is
mismatch=False with no mental_note, so it adds no affect frustration and
teaches the memory nothing — exactly right for a routine step. Default OFF: no
fidelity change unless asked; recommended for predicate-gated study/benchmark
runs where the cost compounds across the matrix.

### 6. Exploration mode — the unknown-system case ✅ slice 5

Implemented as `autouser explore` / `TaskSpec(exploration=True)`:

- ✅ Intrinsic motivation: the system prompt's task block becomes a
  first-session curiosity brief (no goal; follow your interests; prefer what
  you haven't tried — the §1 memory section makes "haven't tried" concrete);
  PROGRESS is reworded to "is this helping you understand what this product
  is?". Response schemas unchanged (parity guard untouched).
- ✅ Natural ending: reflect's `persona_believes_complete` is reinterpreted
  ("seen enough to tell a friend what this is and whether you'd use it");
  the runner finalizes SUCCESS with `CompletionSource.EXPLORATION` on an
  unmismatched 'seen enough'. Predicate/LLM-soft task gates don't apply;
  dwell-loop and frustration/time-budget exits still do — leaving satisfied
  vs leaving frustrated is the meaningful outcome dichotomy.
- ✅ Output: the journey's mental-model section (places + notes + verdict in
  the final reflection) plus a coverage line (distinct controls tried across
  page states).
- Remaining: discoverability coverage of what was NOT found needs a feature
  inventory/sitemap to compare against (study-level aggregation of
  places/targets is the substrate); exploration is DOM-mode only for now.

### 7. Feedback: from anecdote to evidence

- **Within-archetype variance** ✅ slice 4: `persona/variance.py` samples
  deterministic variants around the archetype (seed 0 = unjittered baseline;
  patience gaussian-jittered, Level dimensions shift one step with p=0.25;
  accessibility never jittered — it's identity-defining). `autouser study`
  runs M personas × N seeds, writes per-cell artifacts, and aggregates:
  issue clusters keyed (category, url) ranked by reproduction rate with
  per-persona counts and example quotes, plus per-archetype outcome/steps/
  confidence distributions (the M3 substrate). Aggregation is pure over run
  records — a study is recomputable from `study.json`. v1 cluster key is
  deliberately coarse; element-level matching is the M5 match definition.
- **Positive signal / insights** ✅ slice 2 (notes as findings): the persona's
  mental notes are promoted to `FrictionLog.insights` and rendered in the
  report ("What the persona had to figure out"). Live evidence for why:
  priya/friction-fixture extracted 0 issues, yet her note 1 stated the planted
  Pattern-3 finding verbatim — she *avoided* the trap by reading carefully,
  and the discovery cost is friction a skimming user will pay in full.
  Confidence/delight moments remain future work.
- **Severity from evidence:** derive (or at least adjust) severity from
  objective trace facts — blocked-for-k-steps, caused-abandonment, time burned
  — rather than trusting the per-step LLM label. The renderer's Blocking tier
  already gestures here; push it into the generator.
- **Experience trajectory as artifact** ✅ slice 1: every step's thought,
  expectation, outcome, reflection, emotion, and mental note is recorded in
  the FrictionLog (and therefore `friction_log.json`) and rendered as
  `journey.md` — the "what was it like to be this user" view alongside the
  triage-ready report.

### 8. Validity: the roleplay problem, managed ✅ slice 7

The model knows what a hamburger menu is and is asked to forget. Harness-level
constraints (§§1–4) shrink the pretending surface; the remainder is now
measurable:

- ✅ **Planted-issue benchmarks** (`benchmark.py`, `autouser score`): a
  benchmark YAML declares the issues a fixture page is KNOWN to contain;
  deterministic scoring yields precision (of what was flagged, how much is
  real) and recall (of the known issues, how many found), with the M5
  union-recall shape across runs/personas. The match relation is flow +
  severity-within-one-tier (M5's definition): category is an any-of allow-list
  (it's the fuzziest signal — the live runs categorize the same phone trap as
  `copy` or `error_recovery`), and distinctive keywords / feedback_subtype
  carry flow identification. Element-level matching remains the eventual
  refinement. `--min-precision`/`--min-recall` turn it into a CI gate. Shipped
  benchmark: `benchmarks/friction_page.yaml` (the 5 planted patterns); a real
  recorded Carlos run scores 100% precision / 20% recall (narrow task → walked
  one pattern), demonstrating recall is task-coverage-bound, which is exactly
  why M5 reads recall over the union.
- ✅ **False-positive audit pass** (`audit.py`, `--audit`): an unconstrained
  reviewer (no persona) judges each flagged finding real / artifact /
  uncertain with a rationale, defending the M5 precision ≥ 70% gate. Behind an
  injectable judge seam; every finding is accounted for (skipped indices →
  uncertain, never dropped).
- M3 statistical separation remains necessary but insufficient (separable ≠
  realistic); M5 against real study data stays the real gate.

## Quick wins (engineering debt noticed during the live validation)

- ✅ **Prompt-cache repair** (slice 1): the system prompt was invalidated every
  step to refresh a frustration counter duplicated in the user prompt — full
  input cost per step on the anthropic provider. System prompt is now
  byte-stable per session; per-step state lives only in the user message.
- ✅ `TaskSpec.give_up_threshold` now feeds the runner's effective patience
  budget while preserving custom engine give-up hooks.
- ✅ Dwell-loop detection no longer greps English error words; it keys repeated
  action-error, alert-region, or error-recovery states from the actual
  post-action trace.
- Reflect receives two full-text excerpts; give it an explicit diff.
- ✅ `BatchRunner` can take provider/model knobs or an engine factory.
- `observed_success_signal` vs `persona_believes_complete` overlap confusingly
  in the reflect schema; consider one "evidence of success: none/weak/clear".

## Strategic note

Browser-driving agents are commoditizing (computer-use APIs, Playwright MCPs).
The durable asset is the **persona-constrained cognition plus calibration data**
proving simulated friction predicts real friction. Every change that moves
realism from prompt to mechanism deepens that moat: prompt-only personas are
trivially copyable; mechanism + validation data are not.

## Sequencing vs. milestones

| Order | Work | Serves |
|-------|------|--------|
| 1 ✅ | Mental model (places + notes), SR perception filter, journey artifact, cache fix | M3 (better separation), exploration groundwork |
| 2 ✅ | Action fidelity: keyboard nav + focus tracking, motor noise, keystroke typing; insights in report | M3/M4 (Priya & Carlos become real) |
| 3 ✅ | Affect state + simulated time (interruptions deferred to the dimension-mapping work) | M3 metric 4 (abandonment) gets a defensible mechanism |
| 4 ✅ | Multi-seed variance + aggregated synthesis report (`autouser study`) | M3 statistics; M4 actionability |
| 5 ✅ | Exploration mode (`autouser explore`: curiosity brief, 'seen enough' exit, coverage) | the reframed goal; new product surface |
| 6 ✅ | Dual-process fast/slow loop (`--fast`: skip reflect on cruising steps) | cost/latency; realism |
| 7 ✅ | Planted-issue benchmarks + false-positive audit (`autouser score`, `--audit`) | M5 precision gate |

Each slice lands independently behind the existing `EngineProtocol` /
runner DI seams; none requires a rewrite.
