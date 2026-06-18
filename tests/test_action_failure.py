"""Action-failure resilience: a bad selector is an observation, not a crash.

Found live against a real SPA: the LLM built the selector
input[label="My Project"] from the DOM summary's generic `label=` key, the
locator matched nothing, Page.fill timed out after 60s, and the WHOLE
simulation died with a PersonaError. A real user who clicks something
unresponsive just sees nothing happen — the loop must perceive failure the
same way and let mismatch/patience machinery react in character.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autouser.cognitive.models import ActionIntent, ActionType, UIState
from autouser.cognitive.prompts import build_reflect_user_prompt, build_user_prompt
from autouser.execution.browser import BrowserExecutor

_PAGE = "data:text/html," + """
<!DOCTYPE html>
<html lang="en">
<head><title>Failure Page</title></head>
<body>
  <main>
    <input id="name" type="text" placeholder="Your name" />
    <button id="go" aria-label="Go now">Go</button>
  </main>
</body>
</html>
""".strip().replace("\n", "")


@pytest.fixture
async def executor(tmp_path: Path):
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.start(_PAGE)
    yield ex
    await ex.stop()


class TestFailedActionsDoNotCrash:
    async def test_click_on_missing_selector_returns_action_error(self, executor):
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="button[label=\"Does Not Exist\"]",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        )
        state = await executor.execute(intent, step=1)
        assert isinstance(state, UIState)
        assert state.action_error is not None
        assert "click" in state.action_error
        assert state.page_title == "Failure Page"  # loop continues with live page

    async def test_fill_on_missing_selector_returns_action_error(self, executor):
        intent = ActionIntent(
            action=ActionType.TYPE,
            target="input[label=\"Nope\"]",
            input_value="hello",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        )
        state = await executor.execute(intent, step=1)
        assert state.action_error is not None
        assert "type" in state.action_error

    async def test_successful_action_has_no_action_error(self, executor):
        intent = ActionIntent(
            action=ActionType.TYPE,
            target="#name",
            input_value="maria",
            persona_thought="t", expected_outcome="e", confidence=0.9,
        )
        state = await executor.execute(intent, step=1)
        assert state.action_error is None


_AMBIGUOUS_TEXT_PAGE = "data:text/html," + """
<!DOCTYPE html>
<html lang="en">
<head><title>Ambiguous</title></head>
<body>
  <main>
    <button id="bg" onclick="document.getElementById('out').textContent='WRONG'">+ Create Thing</button>
    <button id="exact" onclick="document.getElementById('out').textContent='RIGHT'">Create Thing</button>
    <p id="out"></p>
  </main>
</body>
</html>
""".strip().replace("\n", "")


class TestTextSelectorExactMatching:
    async def test_bare_text_selector_resolves_to_exact_match(self, tmp_path):
        """`text=Create Thing` must hit the exact-text button, not the
        substring match that appears first in DOM order — the live-SPA
        modal-vs-background bug class."""
        ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
        await ex.start(_AMBIGUOUS_TEXT_PAGE)
        try:
            intent = ActionIntent(
                action=ActionType.CLICK, target="text=Create Thing",
                persona_thought="t", expected_outcome="e", confidence=0.9,
            )
            state = await ex.execute(intent, step=1)
            assert state.action_error is None
            assert "RIGHT" in state.visible_text
            assert "WRONG" not in state.visible_text
        finally:
            await ex.stop()

    def test_normalization_quotes_bare_text(self):
        from autouser.execution.browser import _normalize_text_selector

        assert _normalize_text_selector("text=Create Thing") == 'text="Create Thing"'
        # Already-quoted and non-text selectors pass through untouched.
        assert _normalize_text_selector('text="Create Thing"') == 'text="Create Thing"'
        assert _normalize_text_selector("button#exact") == "button#exact"
        # Embedded quotes are escaped, not broken.
        assert _normalize_text_selector('text=Say "hi"') == 'text="Say \\"hi\\""'


class TestDomSummaryRealAttributeNames:
    async def test_placeholder_and_aria_label_emitted_under_real_names(self, executor):
        state = await executor.capture_state(step=2)
        # The summary must teach attribute selectors that actually match the DOM.
        assert 'placeholder="Your name"' in state.dom_summary
        assert 'aria-label="Go now"' in state.dom_summary
        assert 'label="' not in state.dom_summary.replace(
            'aria-label="', ""
        )  # no bare invented `label=` key remains


class TestPromptsSurfaceActionFailure:
    def _state(self, action_error=None) -> UIState:
        return UIState(
            url="https://x.test/", page_title="T", dom_summary="== Interactive Elements ==",
            visible_text="hello", action_error=action_error,
        )

    def _intent(self) -> ActionIntent:
        return ActionIntent(
            action=ActionType.CLICK, target="#a",
            persona_thought="t", expected_outcome="e", confidence=0.5,
        )

    def test_plan_prompt_includes_note_when_failed(self):
        prompt = build_user_prompt(
            self._state(action_error="click on '#a' failed: timeout"),
            step_number=2, history=[], patience_budget=6, frustration_spent=1,
        )
        assert "Nothing happened when you tried your last action" in prompt
        # Technical selector detail must NOT leak into the persona's perception.
        assert "timeout" not in prompt

    def test_plan_prompt_clean_when_no_failure(self):
        prompt = build_user_prompt(
            self._state(), step_number=2, history=[],
            patience_budget=6, frustration_spent=0,
        )
        assert "Nothing happened when you tried" not in prompt

    def test_reflect_prompt_includes_note_when_failed(self):
        prompt = build_reflect_user_prompt(
            self._intent(),
            self._state(),
            self._state(action_error="click on '#a' failed: timeout"),
            step_number=3,
        )
        assert "did not take effect" in prompt
        assert "timeout" not in prompt

    def test_reflect_prompt_clean_when_no_failure(self):
        prompt = build_reflect_user_prompt(
            self._intent(), self._state(), self._state(), step_number=3,
        )
        assert "did not take effect" not in prompt
