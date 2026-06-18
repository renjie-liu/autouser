"""Persona-specific perception filtering.

ARCHITECTURE.md names the DOM summary as "the integration point for
persona-specific perception filtering" — this module is that integration
point. The principle (docs/cognitive-v2-design-review.md): constraints
belong in the harness, not the prompt. A persona told "you use a screen
reader" but handed the full visual element inventory is roleplaying a
limitation; a persona that is only *given* what assistive tech exposes
doesn't have to pretend.

v2 slice 1 implements the screen-reader filter (Priya). Future filters
(above-the-fold gating, salience weighting, reading-depth cuts) extend
`perceive()` — the runner already routes every engine-facing state
through here.

The filter returns a *new* UIState (model_copy); the raw capture is
never mutated, so screenshots and replay artifacts keep ground truth.
"""

from __future__ import annotations

from autouser.cognitive.models import UIState
from autouser.persona.models import Persona

# Marker values for UIState.perceived_via — trajectory artifacts use these to
# show *how* the persona experienced each state.
PERCEIVED_SCREEN_READER = "screen_reader"
PERCEIVED_SCREEN_READER_FALLBACK = "screen_reader_fallback"

_SCREEN_READER_HEADER = (
    "== Screen Reader View (linear, document order — no visual layout) ==\n"
    "You hear the page as this sequence of roles and names. Anything not "
    "listed here does not exist for you.\n"
)

_SCREEN_READER_FALLBACK_NOTE = (
    "\n[accessibility tree unavailable in this capture; element list shown "
    "instead — treat unnamed elements as unreachable]"
)

# Frustration-driven skim: past this ratio of the patience budget, the persona
# stops reading carefully. Affect modulating perception is the design-review
# §4 behavior loop — frustrated users read less, which makes them miss more.
_SKIM_FRUSTRATION_RATIO = 0.5
_SKIM_VISIBLE_CHARS = 1200
_SKIM_NOTE = "\n[you're frustrated and skimming — you didn't read the rest of the page]"


def perceive(
    state: UIState,
    persona: Persona,
    *,
    frustration_ratio: float = 0.0,
) -> UIState:
    """Return the UI state as *this persona* experiences it right now.

    Identity for personas with default perception in a calm state. The runner
    applies this to every state handed to the cognitive engine (DOM mode);
    raw states remain untouched for replay and debugging.
    ``frustration_ratio`` (0..1, from ``affect.frustration_ratio``) trims
    reading depth once the persona is past half their patience.
    """
    if persona.accessibility.screen_reader:
        state = _screen_reader_view(state)
    if frustration_ratio >= _SKIM_FRUSTRATION_RATIO:
        state = _skim_view(state)
    return state


def _skim_view(state: UIState) -> UIState:
    if len(state.visible_text) <= _SKIM_VISIBLE_CHARS:
        return state
    return state.model_copy(
        update={"visible_text": state.visible_text[:_SKIM_VISIBLE_CHARS] + _SKIM_NOTE}
    )


def _screen_reader_view(state: UIState) -> UIState:
    """Replace the visual element inventory with the accessibility tree.

    A screen-reader user perceives roles + accessible names in document
    order — no layout, no color, no position, and crucially no unnamed
    elements. If the page's ARIA is broken, this view is broken, and the
    persona's resulting struggle IS the finding.

    When the capture has no aria_snapshot (older playwright, capture
    failure), fall back to the element list with an explicit honesty note
    rather than silently granting full vision — and mark the state so the
    trajectory shows the degradation.
    """
    if state.aria_snapshot:
        return state.model_copy(
            update={
                "dom_summary": _SCREEN_READER_HEADER + state.aria_snapshot,
                "perceived_via": PERCEIVED_SCREEN_READER,
            }
        )
    return state.model_copy(
        update={
            "dom_summary": state.dom_summary + _SCREEN_READER_FALLBACK_NOTE,
            "perceived_via": PERCEIVED_SCREEN_READER_FALLBACK,
        }
    )
