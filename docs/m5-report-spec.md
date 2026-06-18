# AutoUser Friction Report — M5 Specification (v2)

**Status:** v2 floor. Satisfies M5 exit criterion #2 as-is. Two ceiling-raising items are open as Saturday-decision-pending follow-ups (§5.1); neither blocks report impl. If Saturday lands cleanly, fold ceiling upgrades as follow-up commits to this PR; otherwise both roll to M6.

**Changelog vs. v1:**
- §3.2 — Persona render rule made explicit (archetype.narrative preferred; structured-field composition fallback)
- §5 — Persona row corrected to actual schema fields (replacing speculative `behavioral_attributes`)
- §5.1 — expanded to two parallel open decisions (`retry_count` + `narrative-source`)
- §7.1 — annotated as captured-(i) ceiling; templated-(iii) floor paragraph added alongside
- §7.2 — rewritten in templated-(iii) form as the primary example; ceiling shown for comparison

---

## 1. Purpose

Defines the human-readable report AutoUser produces from a single simulation run. Turns structured output (`SessionLog`, `Finding[]`) into a document a non-engineer can read and act on.

**M5 exit criterion this spec satisfies:** *a non-engineer can read the output and identify the top friction issue without parsing JSON.*

---

## 2. Audience

Two readers. Same surface, different navigation, different load-bearing sections.

**QA lead — "Did something break, and can I reproduce it?"** Reads Headline → Outcome → Top friction → **Reproduction**. May expand: Technical details. Anchor: Reproduction must be self-sufficient.

**Product manager — "How bad is this, and should we prioritize?"** Reads Headline → Persona → Outcome → **Top friction** → All findings. Skips Technical details. Anchor: Top friction's "Why this matters" paragraph.

**Design principle:** plain language leads, jargon hides. Serves the lower-context reader without losing anything for the higher-context one.

---

## 3. Report structure

Eight elements, in order (plus one conditional element added with cognitive v2):

| # | Element | Purpose |
|---|---|---|
| 1 | **Headline** | One sentence — persona + task + outcome verdict |
| 2 | **Persona** (2 lines) | Behavioral attributes needed to interpret findings. Not demographics |
| 3 | **Task** | What the user was trying to do, in user language |
| 4 | **Outcome** | What happened; subjective/objective divergence explicit when present |
| 5 | **Top friction** | Highest-severity finding, narrative form: where + what happened + why it matters |
| 6 | **All findings** | Severity-ranked table |
| 6b | **What the persona had to figure out** (conditional) | The persona's accumulated mental notes (`FrictionLog.insights`) — workarounds and system quirks discovered along the way. Findings-grade signal the mismatch-driven extraction misses: a trap the persona *avoided* by reading carefully still cost discovery effort. Rendered only when notes exist. Added 2026-06-12 with cognitive v2 slice 2 (see `docs/cognitive-v2-design-review.md` §7). |
| 7 | **Reproduction** | Concrete steps to recreate the run |
| 8 | **Technical details** (collapsed) | Engineering internals; default hidden |

### 3.1 Outcome rendering rule (false-completion is first-class)

Two render modes, selected by comparing the agent's `persona_believes_complete` flag against `SuccessPredicate` evaluation on final state.

- **Convergent:** single sentence.
  > *Outcome:* ❌ Maria did not complete the task — she gave up at login.
  > *Outcome:* ✅ Maria completed the task.
- **Divergent:** two clauses, divergence explicit.
  > *Outcome:* ⚠️ Maria *believed* she completed the task, but the order was not placed.

Divergence is the M5 system's most distinctive finding. Burying it would invert the report's value prop on the case the harness is specifically designed to catch.

### 3.2 Persona section render rule

```
if persona.archetype and persona.archetype.narrative:
    return f"{persona.name} — {persona.archetype.narrative}"
else:
    parts = []
    for field in [tech_literacy, patience, reading_comprehension,
                  domain_familiarity, goal_clarity, accessibility]:
        if field.value != field.default:
            parts.append(f"{field.label}: {field.value}")
    return f"{persona.name} — " + "; ".join(parts)
```

Prefer `archetype.narrative` (human-authored prose) when set; fall back to a composed line from non-default structured fields. Same Persona section across all findings in a report — no per-finding heuristics.

**In:** behavioral patterns under friction, decision style, tolerance thresholds.
**Out:** demographics (age, occupation, location).

The Persona section earns its place by making findings *interpretable*: "Maria retried 3 times" reads as site friction (because Maria is non-experimenting) rather than as a simulation bug.

### 3.3 Severity signaling convention

Severity is conveyed by **both a text label and a visual marker** — never one alone. The visual marker is supplementary to the text; the text is canonical.

- **Marker + text** is the default rendering surface (GFM, Slack, email-with-emoji): `🔴 Blocking`, `🟡 High`, `🟢 Medium`, `⚪ Low`.
- **Text alone** is acceptable on surfaces that cannot render the marker (CLI dumps, plain-text export, screen-reader-only modes). The label by itself still carries full meaning.
- **Marker alone is out of scope.** Color-only signaling fails WCAG and is unreadable to non-sighted users; even in compact tabular formats, the text label must accompany the marker.

The same principle applies to the Outcome line: the emoji (`❌` / `✅` / `⚠️`) accompanies prose, never substitutes for it.

**Why this rule:** the report's §2 audience is mixed in context and assistive-technology use. Redundant signaling means at-a-glance readers benefit from visual hierarchy without making the report unreadable for anyone else. This rule is durable across M6+ surfaces (compact CLI summaries, email digests, machine-readable exports) — if a new surface can't render emoji, drop to text-only; never invert to emoji-only.

Implementation reference: `src/autouser/report/human_renderer.py` — `_TIER_LABELS` returns marker+text together; `_outcome_section` maps `TaskOutcome` → marker; both inherited by all sections that render severity or outcome (currently §5, §6).

---

## 4. Severity ladder

| Severity | Label | Criterion (user-observable) | Examples |
|---|---|---|---|
| **Blocking** | 🔴 Blocking | Persona could not progress | Locked out with no recovery; required information missing; dead-end error |
| **High** | 🟡 High | Progressed only after repeated attempts or backtracking | Retried 3+ times; navigated backward; abandoned and restarted |
| **Medium** | 🟢 Medium | Hesitated/dwelled but recovered on first retry | Single dwell event; one backtrack; brief confusion resolved without help |
| **Low** | ⚪ Low | Friction noted but did not measurably delay or frustrate | Visual-hierarchy hesitation under 5s; minor inconsistencies |

**Ordering:** Blocking > High > Medium > Low. **Ties broken by step count at the friction point** (later in the session = worse).

**Outcome-divergence addendum:** a divergent outcome where objective state *failed* is **Blocking** regardless of in-run retry counts. A user who thinks they finished will not seek help — strictly worse than visible failure.

**Signaling convention applies to the Label column — see §3.3.**

---

## 5. Field mapping

| Report element | Source field(s) |
|---|---|
| Headline | `Persona.name`, `Task.description`, derived outcome verb |
| Persona | `archetype.narrative` if set; else compose from `tech_literacy`, `patience`, `reading_comprehension`, `domain_familiarity`, `goal_clarity`, `accessibility` (non-default values only) — render rule §3.2 |
| Task | `Task.description` |
| Outcome | `Agent.persona_believes_complete` (final step) + `SuccessPredicate.evaluate(final_state)` — pair selects render mode |
| Top friction (narrative) | `Finding[0]` after sort + persona context + `category` + `page_url` — composed per §5.1.2 narrative-source decision |
| All findings (table) | `Finding[]` sorted by severity then step count |
| Reproduction | `Task.target_url`, `SessionLog.steps[].action` |
| Technical details | `SessionLog.terminal_reason`, `Finding[].signature`, `Finding[].category`, run ID, total step count |

### 5.1 Open implementation questions (Saturday-decision-pending)

Both are *ceiling-raising* decisions, not floor-completing. The v2 floor (Option A + path iii) satisfies criterion #2 as specified. If Saturday's pass identifies a clean path to either ceiling without scope expansion beyond ~5 lines at producer sites, fold as a follow-up commit; otherwise both roll to M6.

**Saturday discipline guardrail:** if a ceiling path is selected, scope is exactly *one schema add + one render path change*. Backfilling fixtures, refactoring producers, or "while we're here" enrichment is M6, not M5.

#### 5.1.1 `retry_count` derivation path

The "High" severity criterion needs retry counts at friction points.

- **Option A (floor)** — derive at render time. Report walks `SessionLog.steps` between finding emission and prior page-state match. Zero schema change.
- **Option B (ceiling)** — first-class `Finding.retry_count`. Populated at detection site, ~5 lines at producers.

**Saturday criterion:** does a shared helper `count_retries(finding, session_log)` prevent drift between detection's notion of "retry" and rendering's notion? Yes → A safe. No → B required for *correctness*.

#### 5.1.2 `narrative-source` derivation path

The Top-friction "Why this matters" paragraph is the PM-load-bearing artifact per §2.

- **Path (iii) (floor)** — templated at render time. Composed from `Finding.category`, `Finding.signature`, persona attributes, dwell metrics. Loses UI-specific specifics ("no password reset link", "green check icon").
- **Path (i) (ceiling)** — captured at emission. `Finding.narrative` populated by detection site, ideally from the agent's perception-bearing decision step (`AgentDecision.rationale` or equivalent). Preserves UI specificity.

**Saturday criterion (same shape):** does `compose_narrative(finding, session_log, persona)` produce text satisfying the §2 PM-load-bearing criterion working only from structured state? Yes → (iii) fine. No → (i) required for *correctness*.

**Architectural note:** the agent is the only entity in the pipeline with *perception*. Structured fields don't carry interpretation. (i) is architecturally where the explanation belongs; (iii) is the pragmatic fallback. Saturday's call is about feasibility within M5 scope, not architectural preference.

---

## 6. Out of scope (M6+ candidates)

- Cross-run aggregation and trend dashboards
- Screenshots, video, click heatmaps
- **Persona-tuning prescriptions** ("simplify login copy") — diagnostic-only locked
- **Fix recommendations** — diagnostic-only locked
- Multi-persona comparison reports
- Severity baselines requiring historical data

---

## 7. Worked examples

### 7.1 Convergent outcome — dwell-loop abandonment

#### Templated-(iii) form — v2 floor (what implementation will produce)

> ### Maria gave up trying to log in to saucedemo.
>
> **Persona:** Maria — careful reader; low tech literacy, low patience, high goal clarity, low domain familiarity.
> **Task:** Buy a backpack on saucedemo.com.
>
> *Outcome:* ❌ Maria did not complete the task — she gave up at login.
>
> ### 🔴 Top friction — Blocking
>
> **Login error with no recovery action.** Maria submitted credentials at the login page, received an error, and re-submitted the same credentials three times before abandoning the task. The page presented no recovery affordance the persona acted on.
>
> *Where:* Login page (`/`)
> *Why this matters:* The persona exhibited an abandonment pattern characteristic of cautious, low-patience users encountering an error state they cannot recover from. The dwell signature confirms page state did not change across retries; the persona had no path forward visible from the error alone.
>
> ### All findings (1)
> | # | Severity | Where | What happened |
> |---|---|---|---|
> | 1 | 🔴 Blocking | Login page | Submitted credentials, retried 3×, abandoned. No recovery action taken. |
>
> ### Reproduction
> 1. Open `https://www.saucedemo.com/`
> 2. Enter `locked_out_user` / `secret_sauce`
> 3. Click **Login** — observe error
> 4. (Persona retries identical credentials 2 more times before abandoning)
>
> <details><summary>Technical details</summary>
>
> - `terminal_reason`: `DWELL_LOOP_ABANDONMENT`
> - dwell signature: `("/", "Epic sadface: Sorry, this user has been locked out.")` matched on retries 2 and 3
> - finding category: `recovery_path_missing`
> - retry count at top finding: 3
> - session step count: 4
> - session reference: `saucedemo-maria-2026-05-21-0042`
>
> </details>

#### Captured-(i) form — ceiling (what implementation could produce if §5.1.2 resolves to path i)

> **Locked-account error with no recovery path.** Maria entered her credentials, saw "Epic sadface: Sorry, this user has been locked out.", and had no visible next action — no password reset link, no support contact, no explanation. She retried the same credentials twice more (cautious personas verify, they don't experiment), saw the same error, and abandoned the task.
>
> *Why this matters:* A locked account is recoverable in principle, but the persona had no visible path to recovery. Maria does not have the technical instinct to hunt for password reset — she tried what worked before, hit the same wall, and left.

The two paragraphs differ in *specificity*, not severity logic or report structure. Both identify a Blocking finding at the login page. The ceiling adds UI-specific interpretation the floor cannot produce from structured state alone.

### 7.2 Divergent outcome — false completion

#### Templated-(iii) form — v2 floor

> ### Maria thinks she checked out, but no order was placed.
>
> **Persona:** Maria — careful reader; low tech literacy, low patience, high goal clarity.
> **Task:** Buy a backpack on saucedemo.com.
>
> *Outcome:* ⚠️ Maria **believed** she completed the task, but the order was not placed. She closed the session confident the purchase succeeded.
>
> ### 🔴 Top friction — Blocking
>
> **Misleading completion state at checkout.** Maria signaled that she believed the task was complete while on the checkout-step-two page (`/checkout-step-two.html`). The order-confirmation predicate did not fire — the URL never advanced and no confirmation indicator was detected. From the persona's perspective the task succeeded; from the site's perspective no order was placed.
>
> *Where:* Checkout step 2 (`/checkout-step-two.html`)
> *Why this matters:* This is a divergent-outcome finding. The persona walked away believing success. Users in this state do not escalate — they will not contact support, retry, or revisit the cart. Visible failures are recoverable; this is not.
>
> ### All findings (1)
> | # | Severity | Where | What happened |
> |---|---|---|---|
> | 1 | 🔴 Blocking | Checkout step 2 | Persona believed completion; objective state shows order not placed. No escalation. |
>
> ### Reproduction
> 1. Open `https://www.saucedemo.com/`
> 2. Log in as `standard_user`
> 3. Add backpack to cart, proceed to checkout
> 4. Fill checkout form, click **Finish**
> 5. Observe the page state the persona read as success
> 6. Confirm URL did not advance and no confirmation indicator is present
>
> <details><summary>Technical details</summary>
>
> - `terminal_reason`: `FALSE_COMPLETION`
> - `persona_believes_complete`: `true` at final step
> - `SuccessPredicate.evaluate(final_state)`: `false`
> - finding category: `misleading_confirmation_state`
> - session step count: 8
> - session reference: `saucedemo-maria-2026-05-22-0019` (fixture-derived)
>
> </details>

#### Captured-(i) form — ceiling

> *Why this matters:* This is the worst class of failure — the user does not know to escalate. Maria saw a green check icon and "Thank you" copy and concluded the order was placed. She will not contact support, will not retry, will not return to the cart. From her point of view, the task succeeded.

Same severity logic, same structural placement. The ceiling adds *what specifically Maria interpreted as success* — the design-actionable signal that path (iii) cannot produce from structured state.

---

## 8. Sign-off path

- **Structural sign-off:** ✅ done
- **Floor satisfies M5 exit criterion #2:** ✅ yes (this v2)
- **Saturday ceiling decisions:** ⏳ pending — neither gates report impl pickup
- **M5 close bar:** report impl PR merged on v2 floor (with or without ceiling upgrade)

End of spec.
