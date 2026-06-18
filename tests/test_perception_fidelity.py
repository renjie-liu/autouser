"""Perception/targeting fidelity fixes from the live exploration run.

Maria exploring a live SPA (2026-06-13) looped re-typing into a form she'd
already filled and every Create-Project click timed out. Two harness gaps,
neither a real product issue:

  1. Filled input values were invisible in the perceived state, so a SIGHTED
     persona couldn't tell her typing registered (false 'no confirmation'
     findings + endless re-typing).
  2. The model built `button.foo[text="Create Project"]` from the summary's
     `text="..."` display token — invalid CSS, matches nothing, 5s timeout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autouser.cognitive.models import ActionIntent, ActionType
from autouser.execution.browser import BrowserExecutor, _normalize_text_selector

_FORM_PAGE = "data:text/html," + """
<!DOCTYPE html>
<html lang="en">
<head><title>Form</title></head>
<body>
  <main>
    <input id="name" type="text" placeholder="My Project" />
    <input id="pw" type="password" placeholder="Password" />
    <input id="agree" type="checkbox" />
    <button class="btn px-4" onclick="document.getElementById('out').textContent='SAVED'">Create Project</button>
    <button class="btn px-4">+ Create Project</button>
    <p id="out"></p>
  </main>
</body>
</html>
""".strip().replace("\n", "")


def _intent(action, target="", value=None):
    return ActionIntent(
        action=action, target=target, input_value=value,
        persona_thought="t", expected_outcome="e", confidence=0.7,
    )


@pytest.fixture
async def executor(tmp_path: Path):
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.start(_FORM_PAGE)
    yield ex
    await ex.stop()


class TestInputValuesPerceptible:
    async def test_typed_value_appears_in_dom_summary(self, executor):
        state = await executor.execute(
            _intent(ActionType.TYPE, target="#name", value="My First Project"),
            step=1,
        )
        assert 'contains "My First Project"' in state.dom_summary

    async def test_empty_field_shows_no_contents(self, executor):
        state = await executor.capture_state(step=0)
        assert "contains" not in state.dom_summary

    async def test_password_is_masked(self, executor):
        state = await executor.execute(
            _intent(ActionType.TYPE, target="#pw", value="hunter2"), step=1
        )
        assert "hunter2" not in state.dom_summary
        assert "(filled)" in state.dom_summary

    async def test_checkbox_state_shown(self, executor):
        state = await executor.execute(_intent(ActionType.CLICK, target="#agree"), step=1)
        assert "(checked)" in state.dom_summary


class TestTextAttrSelectorRewrite:
    def test_bare_text_becomes_exact(self):
        assert _normalize_text_selector("text=Create Project") == 'text="Create Project"'

    def test_text_pseudo_attr_becomes_text_is(self):
        assert (
            _normalize_text_selector('button.px-4[text="Create Project"]')
            == 'button.px-4:text-is("Create Project")'
        )

    def test_single_quoted_text_attr(self):
        assert (
            _normalize_text_selector("button[text='Save']")
            == 'button:text-is("Save")'
        )

    def test_plain_selectors_untouched(self):
        assert _normalize_text_selector("button#go") == "button#go"
        assert (
            _normalize_text_selector('input[placeholder="Email"]')
            == 'input[placeholder="Email"]'
        )

    async def test_text_attr_selector_clicks_exact_button(self, executor):
        # The misfire from the live run: a CSS prefix + [text="..."] must now
        # hit the exact-text button, not time out, and not hit "+ Create Project".
        state = await executor.execute(
            _intent(ActionType.CLICK, target='button.px-4[text="Create Project"]'),
            step=1,
        )
        assert state.action_error is None
        assert "SAVED" in state.visible_text
