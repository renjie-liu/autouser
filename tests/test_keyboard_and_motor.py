"""Keyboard navigation + motor fidelity (cognitive v2, slice 2).

Harness-enforced behavior dimensions:
  * press — Tab/Enter keyboard navigation with focus tracked in UIState;
  * type with empty target — types into the focused control, and typing with
    nothing focused is an action_error (the real-user failure mode);
  * imprecise pointer — low motor_precision clicks carry gaussian noise and
    can genuinely miss; typing is real keystrokes with corrected typos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import autouser.execution.browser as browser_mod
from autouser.cognitive.models import ActionIntent, ActionType
from autouser.execution.browser import BrowserExecutor

_PAGE = "data:text/html," + """
<!DOCTYPE html>
<html lang="en">
<head><title>Keyboard Page</title></head>
<body>
  <main>
    <input id="first" type="text" placeholder="First field" />
    <input id="second" type="text" placeholder="Second field"
           onkeydown="this.dataset.keydowns = (parseInt(this.dataset.keydowns || 0) + 1)" />
    <button id="go" onclick="document.getElementById('out').textContent = 'ACTIVATED'">Go</button>
    <p id="out"></p>
  </main>
</body>
</html>
""".strip().replace("\n", "")


def _intent(action: ActionType, target: str = "", value: str | None = None) -> ActionIntent:
    return ActionIntent(
        action=action, target=target, input_value=value,
        persona_thought="t", expected_outcome="e", confidence=0.7,
    )


@pytest.fixture
async def executor(tmp_path: Path):
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.start(_PAGE)
    yield ex
    await ex.stop()


class TestPressAndFocus:
    async def test_tab_moves_focus_and_focus_is_captured(self, executor):
        state = await executor.execute(_intent(ActionType.PRESS, value="Tab"), step=1)
        assert state.action_error is None
        assert state.focused_element is not None
        assert "first" in state.focused_element
        assert 'placeholder="First field"' in state.focused_element

        state = await executor.execute(_intent(ActionType.PRESS, value="Tab"), step=2)
        assert "second" in state.focused_element

    async def test_enter_activates_focused_button(self, executor):
        for _ in range(3):  # Tab to first → second → button
            state = await executor.execute(_intent(ActionType.PRESS, value="Tab"), step=1)
        assert "go" in state.focused_element
        state = await executor.execute(_intent(ActionType.PRESS, value="Enter"), step=2)
        assert "ACTIVATED" in state.visible_text

    async def test_invalid_key_degrades_to_action_error(self, executor):
        state = await executor.execute(
            _intent(ActionType.PRESS, value="NotARealKey"), step=1
        )
        assert state.action_error is not None
        assert "press" in state.action_error

    async def test_no_focus_captured_on_body(self, executor):
        state = await executor.capture_state(step=1)
        assert state.focused_element is None


class TestTypeIntoFocused:
    async def test_type_into_focused_field(self, executor):
        await executor.execute(_intent(ActionType.PRESS, value="Tab"), step=1)
        state = await executor.execute(
            _intent(ActionType.TYPE, value="hello"), step=2
        )
        assert state.action_error is None
        value = await executor._page.input_value("#first")
        assert value == "hello"

    async def test_type_with_nothing_focused_is_action_error(self, executor):
        state = await executor.execute(
            _intent(ActionType.TYPE, value="hello"), step=1
        )
        assert state.action_error is not None
        assert "focus" in state.action_error


class TestImprecisePointer:
    @pytest.fixture
    async def imprecise(self, tmp_path: Path):
        ex = BrowserExecutor(screenshot_dir=tmp_path / "shots", imprecise_pointer=True)
        await ex.start(_PAGE)
        yield ex
        await ex.stop()

    async def test_zero_noise_hits_target(self, imprecise, monkeypatch):
        monkeypatch.setattr(browser_mod, "_pointer_noise", lambda w, h: (0.0, 0.0))
        state = await imprecise.execute(_intent(ActionType.CLICK, target="#go"), step=1)
        assert state.action_error is None
        assert "ACTIVATED" in state.visible_text

    async def test_large_noise_misses_target(self, imprecise, monkeypatch):
        # Force the click 300px below the button — it lands on empty page,
        # nothing activates, and the run does NOT crash. The persona will
        # perceive "nothing happened" via the unchanged page.
        monkeypatch.setattr(browser_mod, "_pointer_noise", lambda w, h: (0.0, 300.0))
        state = await imprecise.execute(_intent(ActionType.CLICK, target="#go"), step=1)
        assert state.action_error is None  # the click executed; it just missed
        assert "ACTIVATED" not in state.visible_text

    async def test_typing_is_keystrokes_with_corrected_typo(self, imprecise, monkeypatch):
        # Force a typo before the first character only.
        calls = {"n": 0}

        def fake_typo(char: str):
            calls["n"] += 1
            return "x" if calls["n"] == 1 else None

        monkeypatch.setattr(browser_mod, "_typo_for", fake_typo)
        state = await imprecise.execute(
            _intent(ActionType.TYPE, target="#second", value="ab"), step=1
        )
        assert state.action_error is None
        page = imprecise._page
        # Final value is corrected...
        assert await page.input_value("#second") == "ab"
        # ...but the fumble happened as real key events: typo + backspace +
        # 2 chars ≥ 4 keydowns (fill() would produce none).
        keydowns = int(await page.get_attribute("#second", "data-keydowns") or 0)
        assert keydowns >= 4

    async def test_precise_executor_still_uses_fill(self, executor):
        state = await executor.execute(
            _intent(ActionType.TYPE, target="#second", value="ab"), step=1
        )
        assert state.action_error is None
        page = executor._page
        assert await page.input_value("#second") == "ab"
        # fill() is programmatic — no per-key keydown events.
        keydowns = int(await page.get_attribute("#second", "data-keydowns") or 0)
        assert keydowns == 0
