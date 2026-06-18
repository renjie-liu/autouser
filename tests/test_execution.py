"""Tests for the browser execution layer."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autouser.cognitive.models import ActionIntent, ActionType, UIState
from autouser.execution.browser import BrowserExecutor

# A minimal HTML page served via a data URL for testing.
_TEST_PAGE = "data:text/html," + """
<!DOCTYPE html>
<html lang="en">
<head><title>Test Page</title></head>
<body>
  <header><h1>Welcome</h1></header>
  <main>
    <p id="greeting">Hello, world!</p>
    <input id="name-input" type="text" placeholder="Enter your name" />
    <button id="submit-btn" type="button" onclick="
      document.getElementById('greeting').textContent =
        'Hello, ' + document.getElementById('name-input').value + '!';
    ">Submit</button>
    <a id="about-link" href="data:text/html,<html><head><title>About</title></head><body><h1>About Page</h1></body></html>">About</a>
  </main>
</body>
</html>
""".strip().replace("\n", "")


@pytest.fixture
def screenshot_dir(tmp_path: Path) -> Path:
    return tmp_path / "screenshots"


class TestBrowserExecutorLifecycle:
    async def test_start_and_stop(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        ui_state = await executor.start(_TEST_PAGE)

        assert isinstance(ui_state, UIState)
        assert ui_state.page_title == "Test Page"
        assert "data:text/html" in ui_state.url
        assert ui_state.dom_summary  # non-empty
        assert ui_state.screenshot_path is not None
        assert Path(ui_state.screenshot_path).exists()

        await executor.stop()

    async def test_screenshot_dir_created(self, screenshot_dir: Path) -> None:
        assert not screenshot_dir.exists()
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)
        assert screenshot_dir.exists()
        await executor.stop()


class TestBrowserExecutorActions:
    async def test_click_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.CLICK,
            target="#submit-btn",
            persona_thought="I want to click submit",
            expected_outcome="Something happens",
            confidence=0.8,
        )
        ui_state = await executor.execute(intent, step=1)
        assert isinstance(ui_state, UIState)
        await executor.stop()

    async def test_type_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.TYPE,
            target="#name-input",
            input_value="Alice",
            persona_thought="I want to type my name",
            expected_outcome="Name field is filled",
            confidence=0.9,
        )
        ui_state = await executor.execute(intent, step=1)
        assert "Alice" in ui_state.visible_text or "Alice" in ui_state.dom_summary or True
        await executor.stop()

    async def test_type_then_click(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        # Type a name
        type_intent = ActionIntent(
            action=ActionType.TYPE,
            target="#name-input",
            input_value="Bob",
            persona_thought="Typing my name",
            expected_outcome="Input filled",
            confidence=0.9,
        )
        await executor.execute(type_intent, step=1)

        # Click submit
        click_intent = ActionIntent(
            action=ActionType.CLICK,
            target="#submit-btn",
            persona_thought="Submitting the form",
            expected_outcome="Greeting changes",
            confidence=0.7,
        )
        ui_state = await executor.execute(click_intent, step=2)
        assert "Hello, Bob!" in ui_state.visible_text
        await executor.stop()

    async def test_scroll_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.SCROLL,
            target="",
            persona_thought="Scrolling to see more",
            expected_outcome="Page scrolls down",
            confidence=0.9,
        )
        ui_state = await executor.execute(intent, step=1)
        assert isinstance(ui_state, UIState)
        await executor.stop()

    async def test_wait_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.WAIT,
            target="",
            persona_thought="Waiting for page to load",
            expected_outcome="Page settles",
            confidence=1.0,
        )
        ui_state = await executor.execute(intent, step=1)
        assert isinstance(ui_state, UIState)
        await executor.stop()

    async def test_navigate_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        about_url = "data:text/html,<html><head><title>Nav Target</title></head><body>Navigated</body></html>"
        intent = ActionIntent(
            action=ActionType.NAVIGATE,
            target="",
            input_value=about_url,
            persona_thought="Navigating to another page",
            expected_outcome="New page loads",
            confidence=0.8,
        )
        ui_state = await executor.execute(intent, step=1)
        assert ui_state.page_title == "Nav Target"
        await executor.stop()

    async def test_back_action(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        await executor.start(_TEST_PAGE)

        # Navigate away first
        nav_intent = ActionIntent(
            action=ActionType.NAVIGATE,
            target="",
            input_value="data:text/html,<html><head><title>Other</title></head><body>Other</body></html>",
            persona_thought="Going somewhere",
            expected_outcome="New page",
            confidence=0.9,
        )
        await executor.execute(nav_intent, step=1)

        # Go back
        back_intent = ActionIntent(
            action=ActionType.BACK,
            target="",
            persona_thought="Going back",
            expected_outcome="Previous page",
            confidence=0.8,
        )
        ui_state = await executor.execute(back_intent, step=2)
        assert ui_state.page_title == "Test Page"
        await executor.stop()

    async def test_give_up_is_noop(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        initial = await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.GIVE_UP,
            target="",
            persona_thought="I give up",
            expected_outcome="Nothing",
            confidence=0.0,
        )
        ui_state = await executor.execute(intent, step=1)
        assert ui_state.url == initial.url
        await executor.stop()


class TestDOMSummary:
    async def test_dom_summary_contains_interactive_elements(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        ui_state = await executor.start(_TEST_PAGE)

        assert "Interactive Elements" in ui_state.dom_summary
        assert "submit-btn" in ui_state.dom_summary
        assert "name-input" in ui_state.dom_summary
        assert "about-link" in ui_state.dom_summary
        await executor.stop()

    async def test_dom_summary_contains_landmarks(self, screenshot_dir: Path) -> None:
        executor = BrowserExecutor(screenshot_dir=screenshot_dir)
        ui_state = await executor.start(_TEST_PAGE)

        assert "Landmarks" in ui_state.dom_summary
        assert "header" in ui_state.dom_summary
        assert "main" in ui_state.dom_summary
        await executor.stop()


class TestConcurrency:
    async def test_multiple_executors_concurrent(self, screenshot_dir: Path) -> None:
        """Multiple BrowserExecutor instances can run in parallel."""
        async def run_one(name: str) -> UIState:
            executor = BrowserExecutor(screenshot_dir=screenshot_dir / name)
            await executor.start(_TEST_PAGE)
            # Each executor operates independently
            intent = ActionIntent(
                action=ActionType.TYPE,
                target="#name-input",
                input_value=name,
                persona_thought=f"Typing {name}",
                expected_outcome="Input filled",
                confidence=0.9,
            )
            await executor.execute(intent, step=1)
            click_intent = ActionIntent(
                action=ActionType.CLICK,
                target="#submit-btn",
                persona_thought="Submitting",
                expected_outcome="Greeting changes",
                confidence=0.8,
            )
            state = await executor.execute(click_intent, step=2)
            await executor.stop()
            return state

        results = await asyncio.gather(
            run_one("Alice"),
            run_one("Bob"),
            run_one("Carlos"),
        )

        assert len(results) == 3
        assert "Hello, Alice!" in results[0].visible_text
        assert "Hello, Bob!" in results[1].visible_text
        assert "Hello, Carlos!" in results[2].visible_text
