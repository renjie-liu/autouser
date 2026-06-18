"""Experience-journey renderer: the persona's thought process over time.

The friction report (human_renderer) answers "what should the product team
fix?"; the journey answers "what was it like to be this user?" — the full
trajectory of thoughts, expectations, outcomes, emotions, and the mental
model the persona built along the way. Everything rendered here is already
recorded in the FrictionLog steps (and therefore in friction_log.json);
this is the human-readable view of that trajectory.
"""

from __future__ import annotations

from autouser.cognitive.models import ActionType
from autouser.persona.models import Persona
from autouser.report.models import FrictionLog

_EMOTION_GLYPHS = {
    "confident": "🙂",
    "uncertain": "🤔",
    "confused": "😕",
    "frustrated": "😖",
    "satisfied": "😌",
}


def _emotion_label(emotion) -> str:
    value = emotion.value if hasattr(emotion, "value") else str(emotion)
    glyph = _EMOTION_GLYPHS.get(value, "")
    return f"{glyph} {value}".strip()


def _action_line(intent) -> str:
    action = intent.action.value
    if intent.action == ActionType.GIVE_UP:
        return "give up"
    target = intent.target or ""
    if intent.input_value and intent.action == ActionType.TYPE:
        dest = f"into {target}" if target else "into the focused control"
        return f'{action} "{intent.input_value}" {dest}'
    if intent.input_value and intent.action == ActionType.NAVIGATE:
        return f"{action} to {intent.input_value}"
    if intent.action == ActionType.PRESS:
        return f"{action} {intent.input_value or '(no key)'}"
    return f"{action} {target}".strip()


def _perception_marker(step) -> str:
    via = step.observation_after.perceived_via
    if not via:
        return ""
    return f" _(perceived via {via.replace('_', ' ')})_"


def _step_section(step) -> str:
    lines = [f"## Step {step.step} — {_action_line(step.intent)}"]
    lines.append(f"- Thought: {step.intent.persona_thought}")
    lines.append(f"- Expected: {step.intent.expected_outcome}")
    happened = step.actual_outcome
    if step.observation_after.action_error:
        happened += " (the attempted action did not take effect)"
    lines.append(f"- What happened: {happened}{_perception_marker(step)}")
    feeling = _emotion_label(step.emotion)
    if step.mismatch:
        detail = []
        if step.severity:
            detail.append(step.severity.value)
        if step.category:
            detail.append(step.category.value)
        suffix = f" — mismatch ({'/'.join(detail)})" if detail else " — mismatch"
        feeling += suffix
    lines.append(f"- Felt: {feeling}")
    lines.append(f"- Reflection: {step.reflection}")
    if step.mental_note:
        lines.append(f"- Note to self: {step.mental_note}")
    return "\n".join(lines)


def _emotion_trajectory(log: FrictionLog) -> str:
    if not log.steps:
        return ""
    values = [_emotion_label(s.emotion) for s in log.steps]
    return "**Emotional arc:** " + " → ".join(values)


def _mental_model_section(log: FrictionLog, persona_name: str) -> str:
    """What the persona believes about the site when the run ends.

    Places are mechanical (every distinct page seen); beliefs are the
    persona's own accumulated notes. Reading this against the actual
    product shows where users' mental models diverge from reality.
    """
    lines = [f"## What {persona_name} now believes about this site"]

    places: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for step in log.steps:
        for obs in (step.observation_before, step.observation_after):
            key = (obs.page_title, obs.url)
            if key not in seen:
                seen.add(key)
                places.append(key)
    if places:
        lines.append("\n**Places visited:**")
        lines.extend(f"- {title or '(untitled)'} — {url}" for title, url in places)

    tried = {
        (s.intent.action.value, s.intent.target or s.intent.input_value or "")
        for s in log.steps
        if s.intent.action not in (ActionType.WAIT, ActionType.GIVE_UP)
    }
    if tried:
        lines.append(
            f"\n**Coverage:** tried {len(tried)} distinct control(s)/action(s) "
            f"across {len(places)} page state(s)."
        )

    notes = [(s.step, s.mental_note) for s in log.steps if s.mental_note]
    if notes:
        lines.append("\n**Accumulated understanding (the persona's own notes):**")
        lines.extend(f"{i}. (step {step}) {note}" for i, (step, note) in enumerate(notes, 1))
    else:
        lines.append("\n*No notes accumulated — the persona left without a working model of the site.*")

    if log.overall_impressions:
        lines.append(f"\n**Overall impression:** {log.overall_impressions}")
    return "\n".join(lines)


def render_journey(log: FrictionLog, persona: Persona) -> str:
    """Render the full experience trajectory as markdown."""
    persona_name = log.persona_name
    header = [
        f"# Experience journey — {persona_name}",
        "",
        f"**Task:** {log.task.rstrip('.')}.",
        f"**Outcome:** {log.outcome.value} in {log.total_steps} step(s).",
    ]
    arc = _emotion_trajectory(log)
    if arc:
        header.extend(["", arc])

    sections = ["\n".join(header)]
    sections.extend(_step_section(step) for step in log.steps)
    sections.append(_mental_model_section(log, persona_name))
    return "\n\n".join(sections) + "\n"
