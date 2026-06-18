"""Tests for the BrowserExecutor (execution layer).

These are integration tests that require Playwright browsers to be installed.
Run: playwright install chromium
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autouser.cognitive.models import ActionIntent, ActionType, UIState
from autouser.execution.browser import BrowserExecutor

# ---------------------------------------------------------------------------
# A tiny HTML page served inline via a data-URI so tests need no network.
# ---------------------------------------------------------------------------
_TEST_PAGE = "data:text/html," + (
    "<html><head><title>Test Page</title></head><body>"
    "<h1>Welcome</h1>"
    '<button id="btn-signup">Sign Up</button>'
    '<input id="email" type="email" placeholder="Enter email" />'
    '<a href="data:text/html,<h1>Page2</h1>" id="link-next">Next</a>'
    "</body></html>"
)


@pytest.fixture
async def executor(tmp_path: Path):
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    yield ex
    await ex.stop()


# ------------------------------------------------------------------
# Lifecycle & state capture
# ------------------------------------------------------------------


async def test_start_returns_ui_state(executor: BrowserExecutor, tmp_path: Path):
    state = await executor.start(_TEST_PAGE)

    assert isinstance(state, UIState)
    assert "Test Page" in state.page_title
    assert "Welcome" in state.visible_text
    assert state.screenshot_path is not None
    assert Path(state.screenshot_path).exists()


async def test_dom_summary_contains_interactive_elements(executor: BrowserExecutor):
    state = await executor.start(_TEST_PAGE)

    assert "btn-signup" in state.dom_summary
    assert "email" in state.dom_summary
    assert "link-next" in state.dom_summary


# ------------------------------------------------------------------
# Action dispatch
# ------------------------------------------------------------------


async def test_click_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.CLICK,
        target="#btn-signup",
        persona_thought="I see a sign-up button, let me click it.",
        expected_outcome="A sign-up form appears",
        confidence=0.7,
    )
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


async def test_type_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.TYPE,
        target="#email",
        input_value="user@example.com",
        persona_thought="I need to enter my email.",
        expected_outcome="Email field is filled in",
        confidence=0.9,
    )
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


async def test_scroll_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.SCROLL,
        target="",
        persona_thought="Let me scroll down to see more.",
        expected_outcome="More content visible",
        confidence=0.5,
    )
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


async def test_navigate_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.NAVIGATE,
        target="",
        input_value="data:text/html,<h1>Other</h1>",
        persona_thought="Navigate somewhere else.",
        expected_outcome="New page loads",
        confidence=0.8,
    )
    state = await executor.execute(intent, step=1)
    assert "Other" in state.visible_text


async def test_back_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    # Navigate away first, then go back.
    nav_intent = ActionIntent(
        action=ActionType.NAVIGATE,
        target="",
        input_value="data:text/html,<h1>Away</h1>",
        persona_thought="Go away.",
        expected_outcome="New page",
        confidence=0.8,
    )
    await executor.execute(nav_intent, step=1)

    back_intent = ActionIntent(
        action=ActionType.BACK,
        target="",
        persona_thought="Go back.",
        expected_outcome="Previous page",
        confidence=0.8,
    )
    state = await executor.execute(back_intent, step=2)
    assert "Welcome" in state.visible_text


async def test_wait_action(executor: BrowserExecutor):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.WAIT,
        target="",
        persona_thought="Let me wait a moment.",
        expected_outcome="Page stays the same",
        confidence=0.5,
    )
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


async def test_give_up_is_noop(executor: BrowserExecutor):
    state_before = await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.GIVE_UP,
        target="",
        persona_thought="I give up.",
        expected_outcome="Nothing",
        confidence=0.0,
    )
    state_after = await executor.execute(intent, step=1)
    assert state_after.url == state_before.url


# ------------------------------------------------------------------
# Screenshot persistence
# ------------------------------------------------------------------


async def test_screenshots_saved_per_step(executor: BrowserExecutor, tmp_path: Path):
    await executor.start(_TEST_PAGE)

    intent = ActionIntent(
        action=ActionType.SCROLL,
        target="",
        persona_thought="scroll",
        expected_outcome="more content",
        confidence=0.5,
    )
    await executor.execute(intent, step=1)
    await executor.execute(intent, step=2)

    shots_dir = tmp_path / "shots"
    pngs = list(shots_dir.glob("*.png"))
    # step 0 (start) + step 1 + step 2 = 3 screenshots
    assert len(pngs) == 3


# ------------------------------------------------------------------
# Dialog dismissal
# ------------------------------------------------------------------

_ALERT_PAGE = "data:text/html," + (
    "<html><head><title>Alert Page</title></head><body>"
    '<button id="trigger" onclick="alert(\'hello\')">Trigger Alert</button>'
    "</body></html>"
)


async def test_dialog_auto_dismissed(executor: BrowserExecutor):
    """Clicking a button that fires alert() should not hang."""
    await executor.start(_ALERT_PAGE)

    intent = ActionIntent(
        action=ActionType.CLICK,
        target="#trigger",
        persona_thought="Click the button.",
        expected_outcome="Something happens",
        confidence=0.5,
    )
    # If dismiss() isn't properly awaited, this will hang or error.
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


# ------------------------------------------------------------------
# Executor cleanup on failure
# ------------------------------------------------------------------


async def test_stop_is_idempotent(tmp_path: Path):
    """Calling stop() on an executor that was never started should not raise."""
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.stop()  # should be a no-op, not an error


async def test_stop_after_start(tmp_path: Path):
    """Resources are released after stop()."""
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.start(_TEST_PAGE)
    await ex.stop()
    assert ex._browser is None or ex._browser.is_connected() is False


# ------------------------------------------------------------------
# ActionIntent model validation
# ------------------------------------------------------------------


def test_click_requires_target():
    with pytest.raises(ValueError, match="requires a non-empty target"):
        ActionIntent(
            action=ActionType.CLICK,
            target="",
            persona_thought="click something",
            expected_outcome="something",
            confidence=0.5,
        )


def test_type_requires_input_value():
    with pytest.raises(ValueError, match="requires input_value"):
        ActionIntent(
            action=ActionType.TYPE,
            target="#field",
            input_value=None,
            persona_thought="type something",
            expected_outcome="something",
            confidence=0.5,
        )


def test_type_without_target_is_valid_keyboard_flow():
    """TYPE with an empty target types into the focused control (keyboard-user
    flow, cognitive v2 slice 2). Typing with nothing focused surfaces as
    UIState.action_error at the executor, not as a validation error here."""
    intent = ActionIntent(
        action=ActionType.TYPE,
        target="",
        input_value="text",
        persona_thought="type into the focused field",
        expected_outcome="something",
        confidence=0.5,
    )
    assert intent.target == ""


def test_navigate_requires_url():
    with pytest.raises(ValueError, match="requires input_value"):
        ActionIntent(
            action=ActionType.NAVIGATE,
            target="",
            input_value=None,
            persona_thought="go somewhere",
            expected_outcome="new page",
            confidence=0.5,
        )


def test_wait_and_give_up_accept_empty_target():
    """WAIT, GIVE_UP, SCROLL, BACK don't require a target."""
    for action in (ActionType.WAIT, ActionType.GIVE_UP, ActionType.SCROLL, ActionType.BACK):
        intent = ActionIntent(
            action=action,
            target="",
            persona_thought="doing something",
            expected_outcome="something",
            confidence=0.5,
        )
        assert intent.action == action


# ------------------------------------------------------------------
# Confirm dialog dismissal
# ------------------------------------------------------------------

_CONFIRM_PAGE = "data:text/html," + (
    "<html><head><title>Confirm Page</title></head><body>"
    '<button id="trigger" onclick="confirm(\'sure?\')">Confirm</button>'
    "</body></html>"
)


async def test_confirm_dialog_dismissed(executor: BrowserExecutor):
    """confirm() should be auto-dismissed without hanging the executor."""
    await executor.start(_CONFIRM_PAGE)
    intent = ActionIntent(
        action=ActionType.CLICK,
        target="#trigger",
        persona_thought="Click confirm button.",
        expected_outcome="Confirm dismissed",
        confidence=0.8,
    )
    state = await executor.execute(intent, step=1)
    assert isinstance(state, UIState)


# ------------------------------------------------------------------
# Double stop safety
# ------------------------------------------------------------------


async def test_double_stop_is_safe(tmp_path: Path):
    """Calling stop() twice must not raise."""
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    await ex.start(_TEST_PAGE)
    await ex.stop()
    await ex.stop()  # second call — should be no-op


# ------------------------------------------------------------------
# Concurrent executor isolation
# ------------------------------------------------------------------


async def test_concurrent_executors_isolated(tmp_path: Path):
    """Multiple executors running concurrently should not interfere."""
    urls = [
        "data:text/html,<h1>Page A</h1>",
        "data:text/html,<h1>Page B</h1>",
        "data:text/html,<h1>Page C</h1>",
    ]

    async def run_one(url: str, label: str) -> UIState:
        ex = BrowserExecutor(screenshot_dir=tmp_path / label)
        try:
            return await ex.start(url)
        finally:
            await ex.stop()

    states = await asyncio.gather(
        run_one(urls[0], "a"),
        run_one(urls[1], "b"),
        run_one(urls[2], "c"),
    )

    assert "Page A" in states[0].visible_text
    assert "Page B" in states[1].visible_text
    assert "Page C" in states[2].visible_text
    assert (tmp_path / "a").exists()
    assert (tmp_path / "b").exists()
    assert (tmp_path / "c").exists()


# ------------------------------------------------------------------
# Configurable Chromium binary (no real browser spawn — pure kwargs wiring)
# ------------------------------------------------------------------


def test_launch_kwargs_default_omits_executable_path(monkeypatch, tmp_path: Path):
    """With no override, launch kwargs are byte-identical to the original
    launch(headless=True) — executable_path must NOT be present, so default
    environments keep resolving Playwright's managed revision unchanged."""
    monkeypatch.delenv("AUTOUSER_BROWSER_EXECUTABLE", raising=False)
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    assert ex._launch_kwargs() == {"headless": True}


def test_launch_kwargs_uses_explicit_executable(monkeypatch, tmp_path: Path):
    """An explicit constructor arg is forwarded as executable_path."""
    monkeypatch.delenv("AUTOUSER_BROWSER_EXECUTABLE", raising=False)
    ex = BrowserExecutor(
        screenshot_dir=tmp_path / "shots", browser_executable="/opt/chrome/chrome"
    )
    assert ex._launch_kwargs() == {
        "headless": True,
        "executable_path": "/opt/chrome/chrome",
    }


def test_launch_kwargs_reads_env_override(monkeypatch, tmp_path: Path):
    """When the arg is unset, AUTOUSER_BROWSER_EXECUTABLE supplies the path."""
    monkeypatch.setenv("AUTOUSER_BROWSER_EXECUTABLE", "/cache/chrome-1208/chrome")
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    assert ex._launch_kwargs()["executable_path"] == "/cache/chrome-1208/chrome"


def test_explicit_executable_beats_env(monkeypatch, tmp_path: Path):
    """Explicit arg wins over the env override."""
    monkeypatch.setenv("AUTOUSER_BROWSER_EXECUTABLE", "/from/env/chrome")
    ex = BrowserExecutor(
        screenshot_dir=tmp_path / "shots", browser_executable="/from/arg/chrome"
    )
    assert ex._launch_kwargs()["executable_path"] == "/from/arg/chrome"


def test_empty_env_collapses_to_default(monkeypatch, tmp_path: Path):
    """An empty-string env value must collapse to the default (no
    executable_path), not pass "" through and break launch."""
    monkeypatch.setenv("AUTOUSER_BROWSER_EXECUTABLE", "")
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    assert ex._launch_kwargs() == {"headless": True}
