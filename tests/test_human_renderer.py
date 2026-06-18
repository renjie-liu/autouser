"""Tests for the human-readable report renderer per docs/m5-report-spec.md.

Sections 1-4 (headline, persona, task, outcome) only — sections 5-8 are
stubs in this initial commit and will be tested as they land.
"""

from __future__ import annotations


from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    Emotion,
    FeedbackSubtype,
    FrictionIssue,
    IssueCategory,
    Severity,
    StepResult,
    UIState,
)
from autouser.persona.models import Persona, PersonaArchetype
from autouser.report.human_renderer import render_report
from autouser.report.models import FrictionLog, Issue, TaskOutcome


def _ui_state(url: str = "http://x") -> UIState:
    return UIState(url=url, page_title="t", dom_summary="", visible_text="")


def _step(*, step: int = 1, persona_believes_complete: bool = False) -> StepResult:
    intent = ActionIntent(
        action=ActionType.CLICK,
        target="#x",
        persona_thought="scripted",
        expected_outcome="state advances",
        confidence=0.8,
    )
    return StepResult(
        step=step,
        intent=intent,
        observation_before=_ui_state(),
        observation_after=_ui_state(),
        actual_outcome="x",
        mismatch=False,
        reflection="y",
        emotion=Emotion.CONFIDENT,
        persona_believes_complete=persona_believes_complete,
    )


def _log(
    *,
    persona_name: str = "Maria",
    task: str = "Buy a backpack",
    outcome: TaskOutcome,
    steps: list[StepResult] | None = None,
) -> FrictionLog:
    return FrictionLog(
        task=task,
        success_criteria="Reach the dashboard",
        persona_name=persona_name,
        persona_summary="careful reader",
        outcome=outcome,
        total_steps=len(steps or []),
        steps=steps or [],
        issues=[],
        overall_impressions="",
    )


# ---------------------------------------------------------------------------
# Section 1: Headline
# ---------------------------------------------------------------------------


def test_headline_completed():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "# Maria completed the task: Buy a backpack." in out


def test_headline_abandoned():
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), Persona())
    assert "# Maria did not complete the task: Buy a backpack." in out


# ---------------------------------------------------------------------------
# Section 2: Persona — archetype.narrative when set, fallback otherwise
# ---------------------------------------------------------------------------


def test_persona_uses_archetype_narrative_when_set():
    arch = PersonaArchetype(
        name="Maria",
        narrative="cautious shopper, low tolerance for unclear errors",
    )
    persona = Persona.from_archetype(arch)
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), persona)
    assert "**Persona:** Maria — cautious shopper, low tolerance for unclear errors" in out


def test_persona_falls_back_to_structured_when_no_archetype():
    # Default-everything persona → fallback path, no behavioral parts
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), Persona())
    assert "**Persona:** Maria — default behavioral profile." in out


# ---------------------------------------------------------------------------
# Section 3: Task
# ---------------------------------------------------------------------------


def test_task_section_rendered():
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), Persona())
    assert "**Task:** Buy a backpack." in out


# ---------------------------------------------------------------------------
# Section 4: Outcome — convergent vs. divergent (false-completion)
# ---------------------------------------------------------------------------


def test_outcome_convergent_completed():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "*Outcome:* ✅ Maria completed the task." in out


def test_outcome_convergent_abandoned():
    out = render_report(
        _log(outcome=TaskOutcome.ABANDONED, steps=[_step()]),
        Persona(),
    )
    assert "*Outcome:* ❌ Maria did not complete the task." in out


def test_outcome_divergent_false_completion():
    """If final step's persona_believes_complete=True but outcome != COMPLETED,
    render the divergent two-clause form per spec §3.1."""
    divergent_step = _step(step=1, persona_believes_complete=True)
    out = render_report(
        _log(outcome=TaskOutcome.ABANDONED, steps=[divergent_step]),
        Persona(),
    )
    assert "**believed** the task was complete" in out
    assert "objective success criteria were not met" in out


# ---------------------------------------------------------------------------
# Section 5: Top friction — narrative form + Blocking derivation per option (3)
# ---------------------------------------------------------------------------


def _issue(
    *,
    severity: Severity = Severity.HIGH,
    description: str = "Maria retried the same login 3 times.",
    url: str = "/login",
    perspective: str = "She had no recovery path visible.",
) -> Issue:
    return Issue(
        severity=severity,
        category=IssueCategory.ERROR_RECOVERY,
        description=description,
        step=1,
        url=url,
        persona_perspective=perspective,
    )


def _step_with_false_success(*, step: int = 1) -> StepResult:
    s = _step(step=step, persona_believes_complete=True)
    s.friction_issues.append(
        FrictionIssue(
            category=IssueCategory.FEEDBACK,
            severity=Severity.HIGH,
            step=step,
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
            description="x",
        )
    )
    return s


def test_top_friction_empty_when_no_issues():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "*No friction findings on this run.*" in out


def test_top_friction_renders_lead_where_and_why():
    """Top friction must include the description, the location, and the
    persona-perspective 'Why this matters' paragraph."""
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[_step()],
    )
    log.issues = [_issue()]
    out = render_report(log, Persona())
    assert "### Top friction — 🔴 Blocking" in out
    assert "**Maria retried the same login 3 times.**" in out
    assert "*Where:* /login" in out
    assert "*Why this matters:* She had no recovery path visible." in out


def test_top_friction_blocking_label_when_high_severity_and_abandoned():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    assert "### Top friction — 🔴 Blocking" in render_report(log, Persona())


def test_top_friction_blocking_label_when_issue_is_false_success_divergence():
    """A FALSE_SUCCESS divergence finding is inherently Blocking — the
    persona doesn't know they're stuck, so they won't escalate. The check
    is per-issue (via Issue.feedback_subtype), not run-level."""
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    log.issues = [
        Issue(
            severity=Severity.HIGH,
            category=IssueCategory.FEEDBACK,
            description="Persona believed task complete; predicate disagreed.",
            step=1,
            url="/",
            persona_perspective="thought it worked",
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
        ),
    ]
    out = render_report(log, Persona())
    assert "### Top friction — 🔴 Blocking" in out


def test_top_friction_high_label_when_high_severity_but_no_blocking_signal():
    """HIGH severity without ABANDONED/TIMED_OUT/FALSE_SUCCESS stays as High."""
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    assert "### Top friction — 🟡 High" in out
    assert "Blocking" not in out


def test_top_friction_medium_label_passthrough():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.MEDIUM)]
    out = render_report(log, Persona())
    assert "### Top friction — 🟢 Medium" in out
    assert "Blocking" not in out


def test_top_friction_step_count_tiebreak_later_step_worse():
    """Per spec §4: ties broken by step count, later step = worse.
    Two HIGH issues at step 3 and step 7 — step 7 must lead the report."""
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    early = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.NAVIGATION,
        description="Early issue at step 3.",
        step=3,
        url="/early",
        persona_perspective="early perspective",
    )
    late = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.ERROR_RECOVERY,
        description="Late issue at step 7.",
        step=7,
        url="/late",
        persona_perspective="late perspective",
    )
    log.issues = [early, late]
    out = render_report(log, Persona())
    # Renderer must rank, not trust list order
    assert "**Late issue at step 7.**" in out
    assert "*Where:* /late" in out
    assert "**Early issue at step 3.**" not in out


# ---------------------------------------------------------------------------
# Integration: real ReportGenerator path — friction_issues must reach the
# top-friction section, not just hand-populated log.issues. Caught by
# an earlier draft review (PR #37): step-level findings from the runner
# (dwell-loop, FALSE_SUCCESS) live on StepResult.friction_issues and were
# not being promoted to FrictionLog.issues.
# ---------------------------------------------------------------------------


def test_friction_issues_from_step_reach_top_friction_render():
    """ReportGenerator must promote StepResult.friction_issues into
    FrictionLog.issues; renderer must then surface them in §5. Without this
    end-to-end test, hand-populated log.issues hides the missing producer
    extraction path."""
    from autouser.cognitive.models import TerminalReason
    from autouser.report.generator import ReportGenerator

    divergent_step = _step(step=1, persona_believes_complete=True)
    divergent_step.friction_issues.append(
        FrictionIssue(
            category=IssueCategory.FEEDBACK,
            severity=Severity.HIGH,
            step=1,
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
            description="Persona believed task complete; predicate disagreed.",
        )
    )
    log = ReportGenerator().generate(
        steps=[divergent_step],
        persona=Persona(),
        task="Complete checkout",
        success_criteria="Order placed.",
        terminal_reason=TerminalReason.ABANDONED,
    )

    # Generator must populate log.issues from the friction finding
    assert len(log.issues) == 1
    assert log.issues[0].category == IssueCategory.FEEDBACK
    assert log.issues[0].severity == Severity.HIGH

    out = render_report(log, Persona())
    # Renderer must surface it, not print the no-friction fallback
    assert "*No friction findings on this run.*" not in out
    assert "### Top friction — 🔴 Blocking" in out
    assert "Persona believed task complete" in out


# ---------------------------------------------------------------------------
# An earlier PR #37 follow-on review: ranking must respect derived tier so an
# unrelated HIGH issue at a later step can't outrank — and falsely get
# labeled Blocking — when a FALSE_SUCCESS exists at an earlier step.
# Plus: COMPLETED_WITH_ERRORS must NOT render the divergent two-clause form.
# ---------------------------------------------------------------------------


def test_false_success_outranks_unrelated_high_at_later_step():
    """A FALSE_SUCCESS finding at an EARLIER step must rank above an
    unrelated HIGH navigation issue at a LATER step. Without per-issue
    feedback_subtype on Issue, the later step's raw HIGH would win the
    tiebreak and then borrow the Blocking label via run-level scan —
    the exact bug an earlier review caught."""
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    false_success = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.FEEDBACK,
        description="Persona believed task complete; predicate disagreed.",
        step=2,
        url="/checkout",
        persona_perspective="thought it worked",
        feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
    )
    unrelated_high = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.NAVIGATION,
        description="Confusing nav structure at later step.",
        step=7,
        url="/nav",
        persona_perspective="got disoriented",
    )
    log.issues = [false_success, unrelated_high]
    out = render_report(log, Persona())
    assert "### Top friction — 🔴 Blocking" in out
    assert "Persona believed task complete" in out
    # Unrelated HIGH at later step must NOT lead and must NOT be labeled Blocking
    assert "**Confusing nav structure at later step.**" not in out


def test_unrelated_high_in_completed_with_errors_is_not_blocking():
    """A plain HIGH issue in a COMPLETED_WITH_ERRORS run (no FALSE_SUCCESS,
    no ABANDONED/TIMED_OUT) must render as High, not Blocking."""
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    assert "### Top friction — 🟡 High" in out
    assert "Blocking" not in out


# ---------------------------------------------------------------------------
# An earlier PR #37 follow-on review: §4 Outcome must NOT render divergent wording
# for COMPLETED_WITH_ERRORS — that's an objective-completed state.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section 6: All findings table — reuses _rank_issues / _severity_label
# ---------------------------------------------------------------------------


def test_all_findings_section_skipped_when_no_issues():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "### All findings" not in out


def test_all_findings_table_rendered_with_header_and_count():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    assert "### All findings (1)" in out
    assert "| # | Severity | Where | What happened |" in out
    assert "|---|---|---|---|" in out


def test_all_findings_table_uses_derived_tier_label():
    """Severity column in §6 must use the same Blocking-derivation as §5
    (via the shared `_severity_label` helper) — not raw enum names."""
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    # HIGH severity in ABANDONED run => Blocking via _is_blocking
    assert "| 1 | 🔴 Blocking | /login | Maria retried the same login 3 times. |" in out


# ---------------------------------------------------------------------------
# Section 7: Reproduction — numbered steps walked from SessionLog.steps
# ---------------------------------------------------------------------------


def _step_with_intent(
    *, step: int, action: ActionType, target: str = "", value: str = "",
    url_before: str = "http://x", url_after: str = "http://x",
) -> StepResult:
    intent = ActionIntent(
        action=action,
        target=target,
        input_value=value if value else None,
        persona_thought="x",
        expected_outcome="y",
        confidence=0.8,
    )
    return StepResult(
        step=step,
        intent=intent,
        observation_before=UIState(url=url_before, page_title="t", dom_summary="", visible_text=""),
        observation_after=UIState(url=url_after, page_title="t", dom_summary="", visible_text=""),
        actual_outcome="x",
        mismatch=False,
        reflection="y",
        emotion=Emotion.CONFIDENT,
    )


def test_reproduction_skipped_when_no_steps():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "### Reproduction" not in out


def test_reproduction_renders_open_url_first_then_steps():
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[
            _step_with_intent(
                step=1, action=ActionType.CLICK, target="#submit-login",
                url_before="http://app.test/login",
            ),
            _step_with_intent(
                step=2, action=ActionType.TYPE, target="#username",
                value="locked_out_user",
            ),
            _step_with_intent(
                step=3, action=ActionType.NAVIGATE, value="http://app.test/dash",
            ),
        ],
    )
    out = render_report(log, Persona())
    assert "### Reproduction" in out
    assert "1. Open http://app.test/login" in out
    assert "2. Click #submit-login" in out
    assert '3. Enter "locked_out_user" into #username' in out
    assert "4. Navigate to http://app.test/dash" in out


def test_reproduction_skips_give_up_steps():
    """GIVE_UP is a termination, not a reproducible action — must not appear
    in the numbered list."""
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[
            _step_with_intent(step=1, action=ActionType.CLICK, target="#x",
                              url_before="http://app/"),
            _step_with_intent(step=2, action=ActionType.GIVE_UP),
        ],
    )
    out = render_report(log, Persona())
    assert "1. Open http://app/" in out
    assert "2. Click #x" in out
    # No 3rd step rendered for GIVE_UP
    assert "3." not in out.split("### Reproduction")[1]


def test_reproduction_sanitizes_newlines_in_targets_and_values():
    """Same shape as the §6 sanitization: agent/UI-derived target or value
    text with embedded newlines must collapse to single-line list items."""
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[
            _step_with_intent(
                step=1, action=ActionType.TYPE,
                target="#username\nbad",
                value="hi\tthere\n",
                url_before="http://app/",
            ),
        ],
    )
    out = render_report(log, Persona())
    repro = out.split("### Reproduction")[1]
    # Numbered list line for the TYPE action stays on a single physical line.
    # Find the "2." line and confirm it has no embedded newline before the
    # next list item or section break.
    assert '2. Enter "hi there" into #username bad' in repro


# ---------------------------------------------------------------------------
# Section 8: Technical details — collapsed <details> block (GFM)
# ---------------------------------------------------------------------------


def test_technical_details_renders_collapsible_block():
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "<details><summary>Technical details</summary>" in out
    assert "</details>" in out


def test_technical_details_includes_outcome_and_step_count_minimum():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step(), _step(step=2)])
    out = render_report(log, Persona())
    assert "- outcome: `abandoned`" in out
    assert "- total steps: 2" in out


def test_technical_details_lists_finding_categories_sorted_and_deduped():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        Issue(severity=Severity.HIGH, category=IssueCategory.NAVIGATION,
              description="d1", step=1, url="/", persona_perspective="p"),
        Issue(severity=Severity.MEDIUM, category=IssueCategory.FEEDBACK,
              description="d2", step=2, url="/", persona_perspective="p"),
        Issue(severity=Severity.LOW, category=IssueCategory.NAVIGATION,
              description="d3", step=3, url="/", persona_perspective="p"),
    ]
    out = render_report(log, Persona())
    # Sorted alphabetically, navigation dedup'd
    assert "- finding categories: `feedback`, `navigation`" in out


def test_technical_details_lists_divergence_subtype_counts():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        Issue(
            severity=Severity.HIGH, category=IssueCategory.FEEDBACK,
            description="d", step=1, url="/", persona_perspective="p",
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
        ),
        Issue(
            severity=Severity.HIGH, category=IssueCategory.FEEDBACK,
            description="d", step=2, url="/", persona_perspective="p",
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
        ),
        Issue(
            severity=Severity.MEDIUM, category=IssueCategory.FEEDBACK,
            description="d", step=3, url="/", persona_perspective="p",
            feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
        ),
    ]
    out = render_report(log, Persona())
    # Sorted by enum value: false_success, silent_success
    assert "- divergence findings: 2× `false_success`, 1× `silent_success`" in out


def test_technical_details_summary_text_is_static():
    """Earlier-review pattern: anchor text must not be agent-derived. The
    <summary> string should be the literal 'Technical details' regardless
    of log content — verified across COMPLETED and ABANDONED outcomes."""
    for outcome in (TaskOutcome.COMPLETED, TaskOutcome.ABANDONED,
                    TaskOutcome.COMPLETED_WITH_ERRORS):
        out = render_report(_log(outcome=outcome), Persona())
        assert "<summary>Technical details</summary>" in out


def test_reproduction_handles_scroll_wait_back_explicitly():
    """An earlier PR #37 §7 review: SCROLL / WAIT / BACK must render as
    user-readable verbs, not "Perform scroll" fall-through."""
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[
            _step_with_intent(step=1, action=ActionType.SCROLL,
                              url_before="http://app/"),
            _step_with_intent(step=2, action=ActionType.WAIT),
            _step_with_intent(step=3, action=ActionType.BACK),
        ],
    )
    out = render_report(log, Persona())
    assert "2. Scroll" in out
    assert "3. Wait" in out
    assert "4. Go back" in out
    # No raw-enum leak
    assert "Perform scroll" not in out
    assert "Perform wait" not in out
    assert "Perform back" not in out


def test_reproduction_escapes_backticks_in_typed_values():
    """An earlier PR #37 §7 review: TYPE values were originally wrapped in
    markdown inline-code backticks. A backtick inside the value (`a\\`b`)
    would close the code span early and break rendering. Switching to
    double-quoted text avoids the inline-code escape problem entirely;
    embedded double quotes are backslash-escaped."""
    log = _log(
        outcome=TaskOutcome.ABANDONED,
        steps=[
            _step_with_intent(
                step=1, action=ActionType.TYPE, target="#password",
                value='a`b"c', url_before="http://app/",
            ),
        ],
    )
    out = render_report(log, Persona())
    # Backtick + double-quote inside the value: backtick sits as plain
    # text inside the double-quoted span (not a markdown delimiter);
    # embedded `"` is backslash-escaped so the quote span stays bounded.
    assert '2. Enter "a`b\\"c" into #password' in out


def test_all_findings_table_sanitizes_pipe_and_newlines_in_cells():
    """An earlier PR #37 §6 review: cell values come from agent/UI free text.
    A pipe or newline in `Issue.description` (or `url`) would break the
    markdown table. _table_cell must collapse whitespace and escape `|`."""
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        Issue(
            severity=Severity.HIGH,
            category=IssueCategory.ERROR_RECOVERY,
            description="Error text: A | B\nthen retry\twith pipes",
            step=1,
            url="/login|nope",
            persona_perspective="x",
        ),
    ]
    out = render_report(log, Persona())
    # Pipes must be escaped; whitespace runs collapsed
    expected_row = (
        "| 1 | 🔴 Blocking | /login\\|nope | "
        "Error text: A \\| B then retry with pipes |"
    )
    assert expected_row in out
    # The table contains exactly one data row (no row-bleeding from newlines)
    assert out.count("\n| 1 |") == 1


def test_all_findings_table_ranks_via_shared_helper():
    """Two issues — FALSE_SUCCESS at earlier step + plain HIGH at later step.
    §6 must rank by derived tier (FALSE_SUCCESS first), same as §5."""
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    false_success = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.FEEDBACK,
        description="False success at step 2.",
        step=2,
        url="/checkout",
        persona_perspective="x",
        feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
    )
    plain_high = Issue(
        severity=Severity.HIGH,
        category=IssueCategory.NAVIGATION,
        description="Nav issue at step 7.",
        step=7,
        url="/nav",
        persona_perspective="x",
    )
    log.issues = [plain_high, false_success]  # deliberately mis-ordered
    out = render_report(log, Persona())
    # FALSE_SUCCESS row must come before nav row, and only it carries Blocking
    fs_idx = out.index("False success at step 2.")
    nav_idx = out.index("Nav issue at step 7.")
    assert fs_idx < nav_idx
    assert "| 1 | 🔴 Blocking | /checkout | False success at step 2. |" in out
    assert "| 2 | 🟡 High | /nav | Nav issue at step 7. |" in out


def test_completed_with_errors_does_not_render_divergent_wording():
    """COMPLETED_WITH_ERRORS means the predicate matched. Even when the
    final step's persona_believes_complete=True (which would normally
    trip the divergent two-clause form for ABANDONED/TIMED_OUT), the
    Outcome line must render the completion-with-friction form, not the
    'objective success criteria were not met' line."""
    step = _step(step=1, persona_believes_complete=True)
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[step])
    out = render_report(log, Persona())
    assert "✅ Maria completed the task, with friction along the way" in out
    assert "**believed** the task was complete" not in out
    assert "objective success criteria were not met" not in out


# ---------------------------------------------------------------------------
# Section 8: Technical details — collapsed <details> block
# ---------------------------------------------------------------------------


def test_technical_details_always_renders_min_outcome_and_step_count():
    """§8 never renders empty — at minimum outcome + step count always
    present, so there's no `<details>` block with nothing inside."""
    out = render_report(_log(outcome=TaskOutcome.COMPLETED), Persona())
    assert "<details><summary>Technical details</summary>" in out
    assert "</details>" in out
    assert "- outcome: `completed`" in out
    assert "- total steps: 0" in out


def test_technical_details_lists_finding_categories_when_issues_present():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        _issue(severity=Severity.HIGH),  # ERROR_RECOVERY per _issue default
        Issue(
            severity=Severity.MEDIUM,
            category=IssueCategory.NAVIGATION,
            description="x",
            step=2,
            url="/y",
            persona_perspective="x",
        ),
    ]
    out = render_report(log, Persona())
    assert "- finding categories: `error_recovery`, `navigation`" in out


def test_technical_details_counts_divergence_subtypes():
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        Issue(
            severity=Severity.HIGH,
            category=IssueCategory.FEEDBACK,
            description="a",
            step=1,
            url="/",
            persona_perspective="x",
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
        ),
        Issue(
            severity=Severity.HIGH,
            category=IssueCategory.FEEDBACK,
            description="b",
            step=2,
            url="/",
            persona_perspective="x",
            feedback_subtype=FeedbackSubtype.FALSE_SUCCESS,
        ),
        Issue(
            severity=Severity.MEDIUM,
            category=IssueCategory.FEEDBACK,
            description="c",
            step=3,
            url="/",
            persona_perspective="x",
            feedback_subtype=FeedbackSubtype.SILENT_SUCCESS,
        ),
    ]
    out = render_report(log, Persona())
    assert "- divergence findings: 2× `false_success`, 1× `silent_success`" in out


# ---------------------------------------------------------------------------
# E2E: real SessionLog from the false-completion harness scenario, piped
# through ReportGenerator.generate() and render_report(). Structural
# assertions per @distinguished-eng's "don't snapshot full markdown" note —
# checks parsed-shape properties, not literal byte positions. This locks
# the producer→render path that an earlier review's catch #1 broke.
# ---------------------------------------------------------------------------


def test_e2e_false_completion_harness_to_human_report():
    """Drive the real SimulationRunner with ScriptedEngine against the
    misleading_checkout fixture (slice 4's scenario), then pipe the
    resulting SessionLog through ReportGenerator + render_report. Asserts
    structural properties of the output, not literal markdown bytes.

    This is the test that would have caught an earlier review's catch #1 (friction
    findings not promoted to FrictionLog.issues) end-to-end."""
    import importlib.util
    import threading
    from functools import partial
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from pathlib import Path

    if importlib.util.find_spec("playwright.async_api") is None:
        import pytest as _pytest
        _pytest.skip("Playwright not installed")

    import asyncio

    from autouser.cognitive.models import (
        ActionIntent as _AI,
        ActionType as _AT,
        SuccessPredicate,
    )
    from autouser.cognitive.scripted import ScriptedEngine, ScriptedStep
    from autouser.report.generator import ReportGenerator
    from autouser.runner import SimulationRunner
    from autouser.session import TaskSpec

    FIXTURES_DIR = Path(__file__).parent / "fixtures"

    class _QH(SimpleHTTPRequestHandler):
        def __init__(self, *a, directory=None, **kw):
            super().__init__(*a, directory=str(directory), **kw)
        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), partial(_QH, directory=FIXTURES_DIR))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{port}"
        task = TaskSpec(
            task="Complete the checkout",
            success_criteria='The "Order placed." confirmation appears',
            start_url=f"{url}/misleading_checkout.html",
            max_steps=5,
            success_predicate=SuccessPredicate(text_contains="Order placed."),
            max_consecutive_failures=None,
        )
        engine = ScriptedEngine(
            steps=[
                ScriptedStep(
                    intent=_AI(
                        action=_AT.CLICK,
                        target="#misleading-confirm",
                        persona_thought="this looks like the confirm step",
                        expected_outcome="order will be placed",
                        confidence=0.85,
                    ),
                    actual_outcome='page showed "Thank you for your order!"',
                    reflection="I think that's done",
                    persona_believes_complete=True,
                ),
            ],
        )
        persona = Persona()
        runner = SimulationRunner(persona=persona, task_spec=task, engine=engine)
        asyncio.run(runner.run())
    finally:
        server.shutdown()

    log = ReportGenerator().generate(
        steps=runner.session.steps,
        persona=persona,
        task=task.task,
        success_criteria=task.success_criteria,
        terminal_reason=runner.session.terminal_reason,
    )
    out = render_report(log, persona)

    # Structural assertions only — no full-markdown snapshot.
    # 1. The producer→render path must surface the FALSE_SUCCESS finding.
    assert len(log.issues) >= 1, (
        "FrictionLog.issues empty — friction_issues not being promoted "
        "(this is the producer-to-consumer bug an earlier review caught)"
    )
    # 2. §4 Outcome renders the divergent two-clause form.
    assert "**believed** the task was complete" in out
    assert "objective success criteria were not met" in out
    # 3. §5 / §6 surface a Blocking finding tied to the false-completion path.
    assert "Top friction — 🔴 Blocking" in out
    # 4. §7 Reproduction includes the open-url line and the misleading-confirm click.
    assert "### Reproduction" in out
    assert "Open http://127.0.0.1:" in out
    assert "Click #misleading-confirm" in out
    # 5. §8 Technical details lists the false_success divergence.
    assert "<details><summary>Technical details</summary>" in out
    assert "false_success" in out
    # 6. Sanity: the run terminated (not None).
    assert log.outcome is not None


# ---------------------------------------------------------------------------
# Pre-merge bundle (PR #37 final review): accessibility field, §5 sanitization,
# headline trailing-period strip, emoji regressions.
# ---------------------------------------------------------------------------


def test_persona_renders_accessibility_when_any_subfield_non_default():
    """Spec §3.2 includes `accessibility` among the composed fields. Non-
    default sub-fields of AccessibilityProfile must surface in the Persona
    section (not silently dropped). Caught by @distinguished-eng's PR #37
    review Finding A."""
    from autouser.persona.models import AccessibilityProfile
    persona = Persona(
        accessibility=AccessibilityProfile(screen_reader=True, zoom_level=2.0),
    )
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), persona)
    assert "accessibility: screen reader, zoom 2.0x" in out


def test_persona_omits_accessibility_when_all_defaults():
    """All-default AccessibilityProfile must not add a noisy
    `accessibility: ...` clause."""
    out = render_report(_log(outcome=TaskOutcome.ABANDONED), Persona())
    assert "accessibility:" not in out


def test_top_friction_sanitizes_newlines_and_asterisks_in_description():
    """Spec §5 prose: newlines split the bold span; literal `*`/`**` in
    description close emphasis prematurely. Per @distinguished-eng's
    Finding B and an earlier final-pass catch — collapse whitespace
    AND backslash-escape `*`/`\\` in description + persona_perspective."""
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [
        Issue(
            severity=Severity.HIGH,
            category=IssueCategory.ERROR_RECOVERY,
            description="Click **Save**\nthen retry",
            step=1,
            url="/login\nbroken",
            persona_perspective="*starred* and\nmulti-line",
        ),
    ]
    out = render_report(log, Persona())
    # Bold span stays bounded — embedded ** is escaped
    assert "**Click \\*\\*Save\\*\\* then retry**" in out
    # URL only collapses whitespace; no emphasis-escape needed there
    assert "*Where:* /login broken" in out
    # Perspective also escapes leading/embedded *
    assert "*Why this matters:* \\*starred\\* and multi-line" in out
    # No raw newline inside the §5 block
    section = out.split("### Top friction")[1].split("### All findings")[0]
    assert "Click **Save**" not in section
    assert "then retry" in section


def test_headline_strips_trailing_period_from_task():
    """Avoid double-period when caller passes task ending in `.`."""
    out = render_report(
        _log(outcome=TaskOutcome.COMPLETED, task="Buy a backpack."),
        Persona(),
    )
    assert "# Maria completed the task: Buy a backpack." in out
    assert "Buy a backpack.." not in out


def test_outcome_divergent_renders_warning_emoji():
    """Divergent outcome line must carry the ⚠️ emoji per the spec v3
    normative styling (paired with text, never emoji-only)."""
    divergent_step = _step(step=1, persona_believes_complete=True)
    out = render_report(
        _log(outcome=TaskOutcome.ABANDONED, steps=[divergent_step]),
        Persona(),
    )
    assert "*Outcome:* ⚠️ Maria **believed**" in out


def test_severity_labels_carry_emojis():
    """All four severity-tier labels must render with their normative
    emoji prefix (single source of truth via _severity_label, so both
    §5 and §6 inherit)."""
    # Blocking
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    assert "🔴 Blocking" in out
    # High (non-Blocking — COMPLETED_WITH_ERRORS doesn't promote)
    log = _log(outcome=TaskOutcome.COMPLETED_WITH_ERRORS, steps=[_step()])
    log.issues = [_issue(severity=Severity.HIGH)]
    out = render_report(log, Persona())
    assert "🟡 High" in out
    # Medium
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.MEDIUM)]
    out = render_report(log, Persona())
    assert "🟢 Medium" in out
    # Low
    log = _log(outcome=TaskOutcome.ABANDONED, steps=[_step()])
    log.issues = [_issue(severity=Severity.LOW)]
    out = render_report(log, Persona())
    assert "⚪ Low" in out
