# Persona-Fidelity Criteria — Track 2 (Cross-Provider Validation)

**Status:** Acceptance contract for Track 2's parity-test PR. Locked through the static-diff + runtime-workspace-code-grounded + spec-author review triangle; all assertions verified against the actual model surfaces in `src/autouser/cognitive/models.py` and `src/autouser/report/models.py`.

**Purpose:** define what *"persona fidelity holds across providers"* means in observable terms, so Track 2's success criterion is testable rather than subjective. Parallel to the M5 severity ladder: same principle (user-observable rules, not implementation-internal heuristics), same load-bearing role for the milestone.

**Audience:** the parity-test PR asserts against these directly; engineering builds against them; the question being answered is *"is AutoUser's persona simulation an architectural claim or a Gemini artifact?"*

---

## Principle

Persona fidelity is preserved if a report produced from a Sonnet-driven run is **structurally indistinguishable** from one produced from a Gemini-driven run, on the same scenario. **Magnitude variation is acceptable; categorical variation is not.**

Reason: LLMs differ in verbosity, latency, and prose style — that's expected and orthogonal to persona behavior. What must hold is the *shape* of the persona's response to friction: same outcome class, same severity profile, same behavioral pattern, same termination cause.

---

## Criteria (all four must hold; any divergence is a finding)

### 1. Outcome class same

| Convergent-fail (ABANDONED) | Stays ABANDONED on both providers |
| Divergent-fail (FALSE_SUCCESS) | Stays FALSE_SUCCESS on both providers |
| Convergent-success (COMPLETED) | Stays COMPLETED on both providers |

The convergent-vs-divergent classification cannot flip. A persona that gets stuck on Gemini and walks away thinking they're done on Sonnet is not "the same persona under different LLMs" — it's two different personas.

### 2. Top-friction severity tier same

If Gemini's Maria-on-saucedemo produces a **Blocking** finding, Sonnet's must too. Severity tier (Blocking / High / Medium / Low) is the load-bearing assertion via `_severity_label`; specific finding `category` may shift if the persona reasonably interprets the same situation slightly differently.

### 3. Behavioral signature same

Each persona has a characteristic action pattern under friction. The pattern must hold across providers.

- **Maria (cautious-non-experimenting):** same `(page_url, error_text)` state revisited across consecutive observations without state expansion. The dwell-loop signature firing is the load-bearing proxy. If Sonnet-Maria visits a password-reset page, clicks support, or navigates away from the error state when Gemini-Maria did not, that's *state expansion* — a persona-fidelity failure. Action variation within the same state (e.g., Enter vs. button click) is acceptable.

  > **Note:** This criterion verifies the persona reached the same stuck state, not that the path-to-stuck was identical action-by-action. Action-diversity bound is permissive in v1.1; if multi-provider runs reveal action diversity diverging meaningfully, tighten in a follow-up.

- **Jake (impatient):** terminates faster than Maria on the same scenario. Assertion: `jake_run.total_steps < maria_run.total_steps` evaluated *within each provider separately* (so Sonnet/Gemini verbosity differences don't confound the persona contrast). If Jake doesn't beat Maria on time-to-abandon within a given provider, the persona-impatience marker isn't holding.

### 4. Termination cause same

Termination class same, evidenced by per-scenario-class helpers. The bare `TerminalReason.ABANDONED` enum is overloaded across three runner paths (dwell-loop detection, patience exhaustion, script-exhausted/`GIVE_UP`) and cannot stand alone — the discriminator pattern PR #37 established applies here too.

- **Dwell-loop class** (Maria-locked-out): `assert has_dwell_exit(gemini_run) and has_dwell_exit(sonnet_run)`

  ```python
  def has_dwell_exit(s):
      return (
          s.terminal_reason == TerminalReason.ABANDONED
          and any(i.category == IssueCategory.ERROR_RECOVERY for i in s.issues)
      )
  ```

- **False-completion class** (Maria-misleading-checkout): `has_false_success(log)` is the load-bearing check; terminal enum is secondary because the runner may abandon or time out (`TerminalReason.ABANDONED` or `TerminalReason.TIMED_OUT`) after the false-success signal depending on provider behavior.

  ```python
  def has_false_success(s):
      return any(i.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS for i in s.issues)
  ```

The same discriminator-at-construction pattern PR #37 already established (`Issue.feedback_subtype`, `Issue.category` populated at emission) makes both helpers thin — no schema work, just two functions in the parity test file.

---

## What's allowed to vary (magnitude tolerance)

These are *not* fidelity failures:

- Step count within ±2 on the same scenario (LLMs vary in pre-action reasoning verbosity)
- Specific `Issue.description` wording (different prose, same meaning)
- Order of secondary findings when multiple have the same severity
- Latency / wall-clock time (provider-specific, persona-independent)
- Persona's internal `thought` content (different LLMs verbalize differently; what matters is the resulting action, not the soliloquy)

---

## What this means concretely for Track 2's parity test

**Minimum acceptance shape — 6 live runs per opt-in execution (3 per provider × 2 providers):**

1. **Maria-locked-out** (Gemini + Sonnet): assert `has_dwell_exit(log)` on both runs.
2. **Jake-locked-out** (Gemini + Sonnet): assert `jake_run.total_steps < maria_run.total_steps` within each provider.
3. **Maria-misleading-checkout** (Gemini + Sonnet): assert `has_false_success(log)` on both runs.

Jake-misleading-checkout is excluded from acceptance — criterion 3 tests impatience under friction, which the locked-out scenario already covers; running Jake on the false-completion fixture tests "Jake believes success faster than Maria," which is a different persona property, not impatience. Optional symmetric matrix is a small follow-up if anyone wants it later.

If all four criteria hold across all three required scenarios, **persona fidelity is architectural, not provider-specific** — the central Track 2 claim is validated.

If any of the four diverges, the divergence is a *finding* requiring investigation, surfaced via the same M5 report renderer the team shipped. The first question to answer: provider artifact (e.g., Sonnet's CLI envelope parse failed and we fell back to a degraded path) or genuine persona drift (Sonnet actually interpreted "cautious shopper" differently)? Different diagnoses, different fixes.

---

## What's deliberately not in this criteria set

- **Output prose comparison** — too noisy; different LLMs phrase things differently, that's expected.
- **Latency parity** — irrelevant to fidelity; subprocess-spawn is known slower.
- **Token-cost parity** — out of scope.
- **Cross-Opus comparison** — deferred; Sonnet first as the cheap parity validation. Opus comparison is a follow-up if Sonnet shows differential persona-fidelity behavior worth investigating.
- **Tooling parity** (e.g., whether both providers use the same JSON-mode mechanism) — implementation concern, not fidelity concern.

---

## Sign-off path

- **Criteria sign-off:** ✅ closed under static-diff + runtime-workspace + spec-author triangle review.
- **Track 2 acceptance:** all four criteria pass on the three required scenarios across Gemini and Sonnet.
- **Failure mode:** any criterion failing surfaces as an investigation item, not a Track 2 blocker — the goal is to *learn* whether fidelity is architectural, not to force a pass.

End of criteria.
