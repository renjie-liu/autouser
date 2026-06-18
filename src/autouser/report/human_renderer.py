"""Human-readable report renderer per docs/m5-report-spec.md (v2 floor).

Renders a FrictionLog + Persona into the eight-section markdown report a
non-engineer can read to identify the top friction issue without parsing
JSON. M5 exit criterion #2.

This is the templated-(iii) floor implementation: narratives are composed
from structured fields (finding category, persona attributes, page URL,
terminal reason). Saturday's `narrative-source` decision (§5.1.2 of the
spec) may upgrade to captured-(i) `Finding.narrative` if compose_narrative
cannot prevent producer/consumer drift; bundled inline if so per the
scope guardrail.

DRAFT — sections being implemented incrementally; see PR description for
the live progress checklist.
"""

from __future__ import annotations

import re

from autouser.cognitive.models import ActionType, FeedbackSubtype, Severity
from autouser.persona.models import Persona
from autouser.report.models import FrictionLog, TaskOutcome


_BEHAVIORAL_FIELDS = [
    "tech_literacy",
    "patience",
    "reading_comprehension",
    "domain_familiarity",
    "goal_clarity",
]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _headline(log: FrictionLog, persona_name: str) -> str:
    """Section 1: one-sentence verdict — persona + task + outcome verb.

    Strips a trailing period from log.task so the headline doesn't end with
    a double period when the caller's task string already ends in `.`.
    """
    verb = _outcome_verb(log)
    task = log.task.rstrip(".")
    return f"# {persona_name} {verb} the task: {task}."


def _format_accessibility(profile) -> str:
    """Render the non-default sub-fields of an AccessibilityProfile.

    Returns "" if every sub-field is at its default. Sub-field defaults
    are read from the AccessibilityProfile model itself (single source
    of truth) so adding a sub-field there auto-extends this surface.
    """
    parts = []
    fields = type(profile).model_fields
    if profile.screen_reader != fields["screen_reader"].default:
        parts.append("screen reader")
    if profile.keyboard_only != fields["keyboard_only"].default:
        parts.append("keyboard only")
    if profile.zoom_level != fields["zoom_level"].default:
        parts.append(f"zoom {profile.zoom_level}x")
    if profile.motor_precision != fields["motor_precision"].default:
        parts.append(f"motor precision: {profile.motor_precision.value}")
    if not parts:
        return ""
    return "accessibility: " + ", ".join(parts)


def _persona_section(persona: Persona, persona_name: str) -> str:
    """Section 2: behavioral 2-line summary per spec §3.2.

    Prefer archetype.narrative when set; fall back to composed non-default
    structured fields. Demographics excluded. Accessibility (per spec §3.2)
    is a nested AccessibilityProfile — rendered via `_format_accessibility`
    when any sub-field is non-default.
    """
    if persona.archetype is not None and persona.archetype.narrative:
        return f"**Persona:** {persona_name} — {persona.archetype.narrative}"

    # Fallback: compose from non-default structured fields.
    parts = []
    for field_name in _BEHAVIORAL_FIELDS:
        field_value = getattr(persona, field_name)
        field_default = type(persona).model_fields[field_name].default
        if field_value != field_default:
            label = field_name.replace("_", " ")
            parts.append(f"{label}: {field_value}")
    accessibility_line = _format_accessibility(persona.accessibility)
    if accessibility_line:
        parts.append(accessibility_line)
    if not parts:
        return f"**Persona:** {persona_name} — default behavioral profile."
    return f"**Persona:** {persona_name} — " + "; ".join(parts)


def _task_section(log: FrictionLog) -> str:
    """Section 3: what the user was trying to do, in user language.
    Strips a trailing period so a caller-supplied task ending in `.`
    doesn't render as `Task: foo..`."""
    return f"**Task:** {log.task.rstrip('.')}."


def _is_blocking(log: FrictionLog, issue) -> bool:
    """Spec §4 severity-tier mapping (render-layer derivation per option (3)).

    Per-issue check, not run-level. Two paths:
    1. The issue itself is a FALSE_SUCCESS divergence — inherently Blocking
       because the persona doesn't know they're stuck.
    2. The issue is HIGH severity in an ABANDONED/TIMED_OUT run — the persona
       could not progress past it.

    An earlier PR #37 follow-on review caught that a run-level "any
    FALSE_SUCCESS in the run" scan could promote an unrelated HIGH issue
    to Blocking. The fix is the discriminator-at-construction pattern:
    Issue.feedback_subtype is populated when the FrictionIssue is promoted,
    so the renderer checks per-issue without scanning siblings.
    """
    if issue.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS:
        return True
    if (
        issue.severity == Severity.HIGH
        and log.outcome in (TaskOutcome.ABANDONED, TaskOutcome.TIMED_OUT)
    ):
        return True
    return False


def _issue_tier(log: FrictionLog, issue) -> int:
    """Derived report tier (0=Blocking, 1=High, 2=Medium, 3=Low) — used for
    ranking so Blocking findings can't be outranked by ordinary HIGH issues
    sharing the same raw Severity enum value."""
    if _is_blocking(log, issue):
        return 0
    return {Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}[issue.severity]


# Tier label includes word + emoji — single source of truth so §5 and §6
# both inherit. Per UX's spec v3 call (normative emoji styling): redundant
# signal (word + emoji always together), a11y-safe for screen readers
# verbalizing "red circle Blocking" / "green circle Medium".
_TIER_LABELS = {
    0: "🔴 Blocking",
    1: "🟡 High",
    2: "🟢 Medium",
    3: "⚪ Low",
}


def _severity_label(log: FrictionLog, issue) -> str:
    return _TIER_LABELS[_issue_tier(log, issue)]


def _rank_issues(log: FrictionLog, issues):
    """Spec ranking: derived report tier ASC (Blocking < High < Medium < Low),
    ties broken by later step (more friction invested in the session) worse.
    Shared by §5 and §6 to prevent drift.

    Ranks on the derived report tier (via `_issue_tier`) rather than raw
    schema severity — otherwise an unrelated HIGH issue at a later step
    could outrank a FALSE_SUCCESS issue at an earlier step within the same
    raw enum value. Returns a NEW sorted list; does not mutate the input.
    """
    return sorted(issues, key=lambda i: (_issue_tier(log, i), -i.step))


def _top_friction_section(log: FrictionLog) -> str:
    """Section 5: highest-severity finding rendered as narrative per spec §3.

    Floor (templated-iii): uses Issue.description for the lead sentence and
    Issue.persona_perspective for the "Why this matters" paragraph. Both are
    populated by ReportGenerator from StepResult fields the agent emits —
    no new data path needed for the floor.

    Ranks via the shared `_rank_issues` helper rather than trusting the
    generator's sort order (caller could pass an unsorted log) — same
    ranking will be used by §6 to prevent drift between sections.
    """
    if not log.issues:
        return "*No friction findings on this run.*"

    top = _rank_issues(log, log.issues)[0]
    label = _severity_label(log, top)
    # Sanitize per @distinguished-eng's PR #37 review Finding B and
    # an earlier final-pass catch: description + perspective are wrapped
    # in / followed by emphasis markers, so they need both whitespace
    # collapse AND `*`/`\` escape (via `_prose_escape`); URL only needs
    # whitespace collapse — never wrapped in emphasis.
    description = _prose_escape(top.description)
    url = _collapse_ws(top.url)
    perspective = _prose_escape(top.persona_perspective)
    return (
        f"### Top friction — {label}\n\n"
        f"**{description}**\n\n"
        f"*Where:* {url}\n"
        f"*Why this matters:* {perspective}"
    )


_COMPLETED_OUTCOMES = {TaskOutcome.COMPLETED, TaskOutcome.COMPLETED_WITH_ERRORS}


_TABLE_CELL_WS_RE = re.compile(r"\s+")


def _collapse_ws(s: str) -> str:
    """Collapse any whitespace run (newlines, tabs, etc.) to a single space.

    Shared by §6 (table cells) and §7 (numbered list items) — any newline
    in an agent/UI-derived string would split a list item or table row.
    """
    if not s:
        return ""
    return _TABLE_CELL_WS_RE.sub(" ", s).strip()


def _table_cell(s: str) -> str:
    """Sanitize free text for markdown table cells: collapse whitespace +
    escape literal `|` so it can't terminate the cell."""
    return _collapse_ws(s).replace("|", "\\|")


_PROSE_ESCAPE_RE = re.compile(r"([\\*])")


def _prose_escape(s: str) -> str:
    """Sanitize free text for §5 prose fields wrapped in emphasis markers.

    The Top friction lead is rendered as `**{description}**` and the
    perspective is rendered inline; either field containing literal `*`
    (or `**`) would split or close the emphasis span prematurely.
    Collapses whitespace via `_collapse_ws` then backslash-escapes `\`
    and `*` per an earlier PR #37 final-pass catch. Same shape as
    `_table_cell` and `_quote_value`: collapse + escape the structural
    characters specific to the wrapping context.
    """
    return _PROSE_ESCAPE_RE.sub(r"\\\1", _collapse_ws(s))


def _all_findings_section(log: FrictionLog) -> str:
    """Section 6: severity-ranked table of all findings per spec §3 element 6.

    Ranks via the shared `_rank_issues` and labels via the shared
    `_severity_label` — same helpers as §5 so the two sections can't
    drift on tier semantics or tiebreak rules. When there are no issues,
    returns empty string (§5 already renders the no-friction line).

    Cell values come from agent/UI-derived free text — `_table_cell()`
    sanitizes whitespace and escapes `|` so a finding description with
    a pipe or newline can't break the markdown table.
    """
    if not log.issues:
        return ""

    ranked = _rank_issues(log, log.issues)
    header = f"### All findings ({len(ranked)})\n\n"
    table = "| # | Severity | Where | What happened |\n|---|---|---|---|\n"
    rows = []
    for idx, issue in enumerate(ranked, start=1):
        label = _severity_label(log, issue)
        rows.append(
            f"| {idx} | {label} | {_table_cell(issue.url)} | {_table_cell(issue.description)} |"
        )
    return header + table + "\n".join(rows)


def _insights_section(log: FrictionLog, persona_name: str) -> str:
    """User-insights section: what the persona had to figure out.

    Insights are the persona's own mental notes — beliefs and workarounds
    discovered along the way. They are findings-grade signal that the
    mismatch-driven issue extraction misses: a trap the persona *avoided*
    by reading carefully still cost discovery effort, and a user who reads
    less carefully will fall in. Rendered after the findings table so triage
    reads issues first, then the user's working theory of the product.
    """
    if not log.insights:
        return ""
    header = f"### What {persona_name} had to figure out ({len(log.insights)})\n\n"
    lines = [
        f"{i}. (step {insight.step}) {_collapse_ws(insight.text)}"
        for i, insight in enumerate(log.insights, start=1)
    ]
    return header + "\n".join(lines)


def _quote_value(s: str) -> str:
    """Wrap an input value for the reproduction list.

    Uses double quotes (not backticks) so an embedded backtick in the
    value can't break a markdown inline-code span. Embedded double quotes
    are escaped. Caught by an earlier PR #37 §7 review — original used
    backticks which broke on `a\\`b`.
    """
    return '"' + s.replace('"', '\\"') + '"'


def _format_action(intent) -> str:
    """Translate an ActionIntent into a user-readable reproduction step.

    Floor-level (iii): uses structured fields (action type, target, input_value)
    and renders in user voice rather than dumping raw ActionIntent JSON. CSS
    selectors and form values are not user-readable labels but they're what
    the floor has; truly label-based reproduction needs the (i) ceiling
    (a captured DOM-text label per element).

    Every ActionType value is handled explicitly — no fall-through to raw
    enum text. Returns "" for GIVE_UP (termination, not reproducible).
    """
    action = intent.action
    target = _collapse_ws(intent.resolve_target() if (intent.target or intent.typed_target) else "")
    value = _collapse_ws(intent.input_value) if intent.input_value else ""

    if action == ActionType.CLICK:
        return f"Click {target}" if target else "Click"
    if action == ActionType.TYPE:
        if target and value:
            return f"Enter {_quote_value(value)} into {target}"
        if value:
            return f"Enter {_quote_value(value)}"
        return "Type"
    if action == ActionType.NAVIGATE:
        return f"Navigate to {value}" if value else "Navigate"
    if action == ActionType.PRESS:
        return f"Press {value}" if value else "Press a key"
    if action == ActionType.SCROLL:
        return "Scroll"
    if action == ActionType.WAIT:
        return "Wait"
    if action == ActionType.BACK:
        return "Go back"
    if action == ActionType.GIVE_UP:
        return ""
    # Defensive: any future ActionType added to the enum will land here
    # until the spec author chooses user-readable wording. Surfacing as
    # the raw enum lets the test suite trip on the gap rather than letting
    # silent leakage ship.
    return _collapse_ws(action.value)


def _reproduction_section(log: FrictionLog) -> str:
    """Section 7: numbered reproduction steps per spec §3 element 7.

    Walks `SessionLog.steps`. Each step's intent is translated into a
    user-readable line via `_format_action`. The first step's
    `observation_before.url` provides the entry-point "Open <url>" line.
    GIVE_UP steps are skipped — they're terminations, not reproducible
    actions.

    All cell text is run through `_collapse_ws` to keep each step on a
    single line; the same shape as §6's cell sanitizer (factored into a
    shared helper so §7 can't drift from §6's whitespace handling).
    """
    if not log.steps:
        return ""

    lines = ["### Reproduction\n"]
    counter = 1

    first = log.steps[0]
    start_url = _collapse_ws(first.observation_before.url)
    if start_url:
        lines.append(f"{counter}. Open {start_url}")
        counter += 1

    for step in log.steps:
        action_line = _format_action(step.intent)
        if action_line:
            lines.append(f"{counter}. {action_line}")
            counter += 1

    if counter == 1:
        return ""  # nothing reproducible (e.g. all GIVE_UP)
    return "\n".join(lines)


def _technical_details_section(log: FrictionLog) -> str:
    """Section 8: technical details for engineers debugging the simulation.

    Floor renders as a GitHub-flavored Markdown `<details>/<summary>` block;
    GitHub collapses it by default. Non-GFM consumers (Slack, plain-text
    email, CLI dumps) may render the tags literally — M6 candidate to add
    a target-aware renderer that emits a plain header on non-GFM surfaces.

    Always renders at minimum the outcome + total step count — there's no
    "empty technical details" floor state. The summary text is the static
    string "Technical details"; never agent-derived, so no sanitization
    risk on the anchor. All category/subtype values come from closed enums
    but pass through `_collapse_ws` defensively — same shape as §6/§7.
    """
    cats = sorted({_collapse_ws(i.category.value) for i in log.issues})
    subtypes_counts: dict[FeedbackSubtype, int] = {}
    for issue in log.issues:
        if issue.feedback_subtype is not None:
            subtypes_counts[issue.feedback_subtype] = (
                subtypes_counts.get(issue.feedback_subtype, 0) + 1
            )

    lines = [
        f"- outcome: `{_collapse_ws(log.outcome.value)}`",
        f"- total steps: {log.total_steps}",
    ]
    if cats:
        lines.append(
            "- finding categories: " + ", ".join(f"`{c}`" for c in cats)
        )
    if subtypes_counts:
        ordered = sorted(subtypes_counts.items(), key=lambda kv: kv[0].value)
        lines.append(
            "- divergence findings: "
            + ", ".join(
                f"{count}× `{_collapse_ws(subtype.value)}`"
                for subtype, count in ordered
            )
        )

    body = "\n".join(lines)
    return (
        "<details><summary>Technical details</summary>\n\n"
        f"{body}\n\n"
        "</details>"
    )


def _outcome_section(log: FrictionLog, persona_name: str) -> str:
    """Section 4: outcome with subjective/objective divergence explicit per §3.1.

    Divergent rendering fires when the run did NOT meet the objective
    success criteria AND there's a per-issue FALSE_SUCCESS finding (or
    the final step itself carries persona_believes_complete=True — both
    are valid signals of the persona believing completion). Discriminator
    is per-issue (`Issue.feedback_subtype`), not a final-step proxy —
    real runs typically have FALSE_SUCCESS emitted on an earlier step
    and a terminal GIVE_UP step where persona_believes_complete=False
    on the final-step proxy alone.

    COMPLETED_WITH_ERRORS is still an objective-completed state (predicate
    matched, friction emerged along the way) — rendering "objective success
    criteria were not met" there would contradict the headline.
    """
    # Outcome-line emoji per UX's spec v3 call (normative): ✅ convergent
    # success, ❌ convergent fail, ⚠️ divergent. Always paired with the
    # text clause so the signal is redundant, not emoji-only.
    if log.outcome == TaskOutcome.COMPLETED:
        return f"*Outcome:* ✅ {persona_name} completed the task."
    if log.outcome == TaskOutcome.COMPLETED_WITH_ERRORS:
        return (
            f"*Outcome:* ✅ {persona_name} completed the task, "
            f"with friction along the way."
        )

    has_false_success_issue = any(
        i.feedback_subtype == FeedbackSubtype.FALSE_SUCCESS for i in log.issues
    )
    final_step = log.steps[-1] if log.steps else None
    final_step_believes_complete = bool(
        final_step is not None and final_step.persona_believes_complete
    )
    is_divergent = (
        log.outcome not in _COMPLETED_OUTCOMES
        and (has_false_success_issue or final_step_believes_complete)
    )
    if is_divergent:
        return (
            f"*Outcome:* ⚠️ {persona_name} **believed** the task was complete, "
            f"but objective success criteria were not met."
        )
    return f"*Outcome:* ❌ {persona_name} did not complete the task."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome_verb(log: FrictionLog) -> str:
    if log.outcome == TaskOutcome.COMPLETED:
        return "completed"
    if log.outcome == TaskOutcome.COMPLETED_WITH_ERRORS:
        return "completed (with friction)"
    return "did not complete"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_report(log: FrictionLog, persona: Persona) -> str:
    """Render a FrictionLog as a markdown human-readable report.

    Spec: docs/m5-report-spec.md (v2 floor). Templated narratives composed
    from structured state.
    """
    persona_name = log.persona_name
    sections = [
        _headline(log, persona_name),
        "",
        _persona_section(persona, persona_name),
        "",
        _task_section(log),
        "",
        _outcome_section(log, persona_name),
        "",
        _top_friction_section(log),
    ]
    all_findings = _all_findings_section(log)
    if all_findings:
        sections.extend(["", all_findings])
    insights = _insights_section(log, persona_name)
    if insights:
        sections.extend(["", insights])
    reproduction = _reproduction_section(log)
    if reproduction:
        sections.extend(["", reproduction])
    sections.extend(["", _technical_details_section(log)])
    return "\n".join(sections)
