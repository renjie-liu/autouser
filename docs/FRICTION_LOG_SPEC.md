# Friction Log Spec

## Overview

A **friction log** is the primary output of an AutoUser simulation run. It captures what happened when a specific persona attempted a specific task — including every step taken, every issue encountered, and the persona's overall experience. Friction logs are designed to be actionable for product and QA teams without requiring familiarity with the simulation internals.

## Output Structure

A friction log is a `FrictionLog` object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | What the persona was trying to do |
| `success_criteria` | string | How to determine task completion |
| `persona_name` | string | Archetype name (e.g., "maria") |
| `persona_summary` | string | Human-readable description of the persona's profile |
| `outcome` | TaskOutcome | How the task ended |
| `total_steps` | integer | Number of steps in the simulation |
| `steps` | list[StepResult] | Full step-by-step trace |
| `issues` | list[Issue] | Discrete usability issues extracted from the trace |
| `overall_impressions` | string | First impression, confidence trajectory, would this user return? |

### Task Outcomes

| Value | Meaning |
|-------|---------|
| `completed` | Persona reached the success criteria without significant difficulty |
| `completed_with_errors` | Persona reached the goal but encountered mismatches or confusion along the way |
| `abandoned` | Persona gave up before completing the task |

## Issues

Each issue represents a discrete usability problem encountered during the simulation.

### Issue Fields

| Field | Type | Description |
|-------|------|-------------|
| `severity` | Severity | How much this issue impacted the task |
| `category` | IssueCategory | What kind of usability problem this is |
| `description` | string | Plain-language description of what went wrong |
| `step` | integer | Which step this issue occurred at |
| `url` | string | Page URL where the issue was encountered |
| `element` | string or null | CSS selector of the problematic element, if identifiable |
| `persona_perspective` | string | Why this is a problem *for this persona specifically* |
| `screenshot_path` | string or null | Path to screenshot captured at this step |

### Severity Scale

Severity is **outcome-anchored** — defined by impact on task completion, not subjective difficulty.

| Level | Meaning | Triage guidance |
|-------|---------|-----------------|
| `high` | Blocks task completion or forces significant workarounds. The persona cannot proceed without help, or the task outcome changes because of this issue. | Fix before shipping. This is what users will contact support about — or leave over. |
| `medium` | Causes noticeable confusion or delay but the persona can eventually proceed. The task takes significantly longer or requires retry. | Fix this cycle. Users will complain but won't churn immediately. |
| `low` | Minor friction that the persona notices but navigates without significant delay. Suboptimal but not broken. | Backlog. Address when working in this area. |

### Issue Categories

| Category | What it covers | Typical fix |
|----------|---------------|-------------|
| `labeling` | Unclear or misleading labels, button text, field names, icon meanings. The user can see the element but doesn't understand what it does. | Rename the element, add a tooltip, use a more recognizable icon. |
| `navigation` | Can't find where to go next. Unclear information architecture, missing breadcrumbs, dead-end pages, hidden menu items. | Restructure IA, add wayfinding cues, improve menu visibility. |
| `accessibility` | Issues specific to assistive technology or physical constraints. Missing ARIA labels, focus traps, keyboard-inaccessible controls, color-only indicators, small touch targets for motor-impaired users. | Add ARIA attributes, fix focus order, increase target size, add non-color indicators. |
| `feedback` | UI doesn't communicate what happened after an action. Missing confirmation, silent failures, ambiguous loading states, no progress indication. | Add confirmation messages, loading indicators, error states. |
| `error_recovery` | User made an error and can't get back on track. No undo, destructive actions without confirmation, forms that clear on error, no back button. | Add undo, confirmation dialogs, preserve form state on error. |
| `layout` | UI arrangement makes interaction difficult. Elements too small, poorly positioned, overlapping, or requiring hover to reveal. Crowded interfaces that overwhelm. | Increase target size, improve spacing, show controls persistently. |
| `copy` | Written content is confusing, misleading, or insufficient. Help text that doesn't help, jargon without explanation, ambiguous error messages, missing instructions. | Rewrite in plain language, add contextual help, define domain terms. |

## Step Trace

Each step in the `steps` list is a `StepResult` capturing the full plan-execute-reflect cycle:

| Field | Type | Description |
|-------|------|-------------|
| `step` | integer | Step number (1-indexed) |
| `intent.action` | ActionType | What the persona did (`click`, `type`, `scroll`, `navigate`, `wait`, `back`, `give_up`) |
| `intent.target` | string | CSS selector or description of the target element |
| `intent.input_value` | string or null | Text typed or URL navigated to |
| `intent.persona_thought` | string | Internal monologue explaining why the persona chose this action |
| `intent.expected_outcome` | string | What the persona expected to happen |
| `intent.confidence` | float (0.0-1.0) | How sure the persona was about this action |
| `observation_before` | UIState | Page state before the action |
| `observation_after` | UIState | Page state after the action |
| `actual_outcome` | string | What actually happened |
| `mismatch` | boolean | Did the actual outcome differ from the expected outcome? |
| `reflection` | string | Post-action persona interpretation |
| `emotion` | Emotion | Persona's emotional state: `confident`, `uncertain`, `confused`, `frustrated`, `satisfied` |
| `severity` | Severity or null | If mismatch: issue severity |
| `category` | IssueCategory or null | If mismatch: issue category |

## Example Output

```json
{
  "task": "Purchase a monthly subscription",
  "success_criteria": "User reaches the payment confirmation page",
  "persona_name": "maria",
  "persona_summary": "Maria, 58, retired teacher. Low tech literacy, high patience, uses email and Facebook but rarely new apps.",
  "outcome": "completed_with_errors",
  "total_steps": 12,
  "steps": [
    {
      "step": 1,
      "intent": {
        "action": "click",
        "target": "a.pricing-link",
        "input_value": null,
        "persona_thought": "I need to find the pricing page. I see a link that says 'Pricing' in the top menu, that seems right.",
        "expected_outcome": "A page showing the different subscription options and their prices",
        "confidence": 0.7
      },
      "observation_before": {
        "url": "https://example.com",
        "page_title": "Example App - Home",
        "dom_summary": "nav: [Home] [Features] [Pricing] [Sign In] ...",
        "visible_text": "Welcome to Example App...",
        "screenshot_path": null
      },
      "observation_after": {
        "url": "https://example.com/pricing",
        "page_title": "Example App - Plans",
        "dom_summary": "h1: 'Choose Your Plan' ...",
        "visible_text": "Choose Your Plan. Starter: $9/mo. Pro: $29/mo...",
        "screenshot_path": "screenshots/a1b2c3/maria/step_001.png"
      },
      "actual_outcome": "Navigated to pricing page showing three plan options",
      "mismatch": false,
      "reflection": "Good, I found the pricing page. I can see different plans listed.",
      "emotion": "confident",
      "severity": null,
      "category": null
    },
    {
      "step": 5,
      "intent": {
        "action": "click",
        "target": "button.select-plan",
        "input_value": null,
        "persona_thought": "I want the monthly plan. There are two buttons that say 'Select' but I'm not sure which one is for monthly vs yearly.",
        "expected_outcome": "It selects the monthly plan and takes me to payment",
        "confidence": 0.3
      },
      "observation_before": {
        "url": "https://example.com/pricing",
        "page_title": "Example App - Plans",
        "dom_summary": "button.select-plan: 'Select' (x2), toggle: 'Monthly / Yearly' ...",
        "visible_text": "Monthly / Yearly. Pro: $29/mo...",
        "screenshot_path": null
      },
      "observation_after": {
        "url": "https://example.com/checkout?plan=yearly-pro",
        "page_title": "Checkout - Yearly Pro",
        "dom_summary": "h1: 'Checkout' form: [card number] [expiry] ...",
        "visible_text": "Checkout. Yearly Pro Plan - $290/year...",
        "screenshot_path": "screenshots/a1b2c3/maria/step_005.png"
      },
      "actual_outcome": "Selected the yearly plan instead of monthly. The checkout page shows yearly pricing.",
      "mismatch": true,
      "reflection": "This isn't what I wanted. I wanted monthly, not yearly. The toggle must have been set to yearly but I didn't notice. Now I need to go back and try again.",
      "emotion": "confused",
      "severity": "high",
      "category": "labeling"
    }
  ],
  "issues": [
    {
      "severity": "high",
      "category": "labeling",
      "description": "Monthly/yearly toggle is not clearly associated with the plan selection buttons. User selected the wrong billing cycle.",
      "step": 5,
      "url": "https://example.com/pricing",
      "element": "button.select-plan",
      "persona_perspective": "Maria doesn't recognize toggles as interactive controls. The 'Select' buttons don't indicate which billing cycle they apply to, so she clicked the first one assuming it matched her intent.",
      "screenshot_path": "screenshots/a1b2c3/maria/step_005.png"
    }
  ],
  "overall_impressions": "Maria found the pricing page easily but struggled with the plan selection toggle. She completed the task after going back and re-selecting, but the toggle interaction pattern was unfamiliar and the labeling didn't compensate for that. She would likely call a family member for help in a real scenario."
}
```

## Triage Workflow

### Step 1: Check the outcome

Start with `outcome`. If it's `abandoned`, the task flow has a fundamental problem for this persona type — investigate the last few steps to understand why.

### Step 2: Review issues by severity

Issues are sorted `high` > `medium` > `low`. Start with `high` severity — these are the task blockers.

### Step 3: Read the persona perspective

The `persona_perspective` field on each issue explains *why* this is a problem for this specific persona. A `labeling` issue that confuses Maria (low tech literacy) might not affect Jake (high tech literacy) — but if your user base skews toward Maria's profile, it's your priority.

### Step 4: Check the step trace for context

The full step trace shows what happened before and after each issue. Look at `intent.confidence` to see how uncertain the persona was, and `emotion` to track whether frustration was building across multiple steps. A single `confused` step is different from a pattern of `confused` > `frustrated` > `frustrated` > `give_up`.

## Cross-Persona Analysis

When the same task produces different findings across personas, that's signal — not noise. A confirmation dialog that Carlos flags as `high` severity but Jake breezes through isn't a contradiction; it tells you the dialog's impact depends on user patience and motor precision.

**To interpret conflicting findings: weight by your actual user mix.** If 70% of your users resemble Maria (low tech literacy, careful, patient), then Maria's friction log is your primary signal. Jake's clean run tells you power users are fine — but that's not who you're losing.

For v1, this synthesis is manual. Run 3+ dimensionally diverse archetypes (see [Persona Framework](PERSONA_FRAMEWORK.md) for coverage set recommendations), compare their friction logs side by side, and prioritize issues that appear for the personas closest to your real users.
