# Vision

> **Owner:** PM. Engineering, UX, and stakeholder reviewers.

## What is AutoUser?

AutoUser is an agent that simulates realistic user behavior against web applications. It produces actionable usability feedback — not test pass/fail results, but friction logs that read like usability study notes.

## Problem

Manual usability testing is slow, expensive, and doesn't scale. Automated testing validates *functionality* (does the button work?) but not *usability* (can a real person figure out which button to click?). The gap between "works correctly" and "works for actual humans" is where products lose users.

Teams ship features that pass all tests but frustrate real people — because no test asked "would a non-technical user understand this label?" or "can a screen reader user complete this flow?"

## Solution

AutoUser bridges the gap by simulating diverse user personas — each with different tech literacy, patience, domain familiarity, cognitive capacity, and accessibility needs — and having them attempt real tasks in a real browser. The output is a friction log: a step-by-step record of what the persona tried, what they expected, what actually happened, and where they got stuck.

### Key differentiators

- **Persona-driven, not script-driven.** Tests describe *what* to accomplish ("complete a purchase"), not *how* to do it. The persona figures out the how — and their struggles are the signal.
- **Behavioral realism over coverage.** A low-tech-literacy persona doesn't magically know that a hamburger icon is a menu. A low-patience persona abandons after two confusing screens. These aren't bugs — they're the user experience.
- **Friction logs, not pass/fail.** The output is qualitative: severity-ranked issues with the persona's perspective ("I wasn't sure what 'Submit' would do"). Product teams can read these like usability study findings.
- **Dimensional diversity.** Five behavioral dimensions produce meaningfully different action patterns. Running 3 diverse personas covers more of the user experience space than running one persona 100 times.

## Target users

- **QA teams** who want usability coverage alongside functional testing
- **Product managers** who need usability signal without scheduling user research
- **Accessibility engineers** who want to validate assistive technology flows at scale
- **Design teams** who want to identify friction points before user testing, not instead of it

## What AutoUser is NOT

- Not a replacement for real user research. It surfaces *likely* friction, not *confirmed* friction. Calibration against real studies (M5) determines how much to trust it.
- Not a functional test framework. It doesn't assert "the cart total should be $42.00." It asserts "the persona could find and complete the checkout flow."
- Not an accessibility audit tool. It simulates *users with accessibility needs*, not WCAG compliance checkers. Priya (screen reader user) surfaces what a real screen reader user would hit — but she won't catch every `aria-label` violation.

## Success criteria

AutoUser succeeds when:
1. **Different personas produce different findings** on the same task (M3 gate)
2. **Findings are actionable** — product teams can triage and fix issues from the friction log alone (M4 gate)
3. **Findings correlate with real usability issues** — friction flagged by AutoUser matches friction observed in actual user studies (M5 gate)
