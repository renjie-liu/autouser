"""Tests for coordinate-based action execution in BrowserExecutor.

These are integration tests that require Playwright browsers to be installed.
Run: playwright install chromium
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CoordinateTarget,
    UIState,
)
from autouser.execution.browser import BrowserExecutor

_TEST_PAGE = "data:text/html," + (
    "<html><head><title>Coordinate Test</title></head><body>"
    "<h1>Welcome</h1>"
    '<button id="btn" style="position:absolute;left:100px;top:100px;'
    'width:200px;height:50px;">Click Me</button>'
    '<input id="email" type="email" style="position:absolute;left:100px;'
    'top:200px;width:300px;height:40px;" placeholder="Email" />'
    '<div id="result"></div>'
    "<script>"
    'document.getElementById("btn").addEventListener("click", function() {'
    '  document.getElementById("result").textContent = "CLICKED";'
    "});"
    "</script>"
    "</body></html>"
)


@pytest.fixture
async def executor(tmp_path: Path):
    ex = BrowserExecutor(screenshot_dir=tmp_path / "shots")
    yield ex
    await ex.stop()


class TestCoordinateClick:
    async def test_click_at_coordinates(self, executor: BrowserExecutor, tmp_path):
        await executor.start(_TEST_PAGE)

        coord = CoordinateTarget(x=200, y=125, screenshot_id="step1_test")
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="",
            typed_target=coord,
            persona_thought="I see a button.",
            expected_outcome="It gets clicked.",
            confidence=0.8,
        )

        state = await executor.execute(intent, step=1, expected_screenshot_id="step1_test")
        assert "CLICKED" in state.visible_text

    async def test_out_of_bounds_raises(self, executor: BrowserExecutor, tmp_path):
        await executor.start(_TEST_PAGE)

        coord = CoordinateTarget(x=5000, y=5000, screenshot_id="step1_test")
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="",
            typed_target=coord,
            persona_thought="Clicking.",
            expected_outcome="Something.",
            confidence=0.5,
        )

        with pytest.raises(ValueError, match="out of viewport"):
            await executor.execute(intent, step=1, expected_screenshot_id="step1_test")

    async def test_stale_screenshot_raises(self, executor: BrowserExecutor, tmp_path):
        await executor.start(_TEST_PAGE)

        coord = CoordinateTarget(x=200, y=125, screenshot_id="old_screenshot")
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="",
            typed_target=coord,
            persona_thought="Clicking.",
            expected_outcome="Something.",
            confidence=0.5,
        )

        with pytest.raises(ValueError, match="Stale screenshot"):
            await executor.execute(intent, step=1, expected_screenshot_id="current_screenshot")

    async def test_no_expected_id_skips_staleness_check(self, executor: BrowserExecutor, tmp_path):
        """When expected_screenshot_id is None, staleness check is skipped."""
        await executor.start(_TEST_PAGE)

        coord = CoordinateTarget(x=200, y=125, screenshot_id="any_id")
        intent = ActionIntent(
            action=ActionType.CLICK,
            target="",
            typed_target=coord,
            persona_thought="Clicking.",
            expected_outcome="Button clicked.",
            confidence=0.8,
        )

        # Should not raise — no expected ID to check against
        state = await executor.execute(intent, step=1)
        assert "CLICKED" in state.visible_text


class TestCoordinateType:
    async def test_type_at_coordinates(self, executor: BrowserExecutor, tmp_path):
        await executor.start(_TEST_PAGE)

        coord = CoordinateTarget(x=250, y=220, screenshot_id="step1_test")
        intent = ActionIntent(
            action=ActionType.TYPE,
            target="",
            typed_target=coord,
            input_value="test@example.com",
            persona_thought="Typing email.",
            expected_outcome="Email entered.",
            confidence=0.7,
        )

        state = await executor.execute(intent, step=1, expected_screenshot_id="step1_test")
        assert isinstance(state, UIState)


class TestSelectorFallback:
    async def test_selector_still_works(self, executor: BrowserExecutor, tmp_path):
        """CSS selector targeting (DOM mode) continues to work unchanged."""
        await executor.start(_TEST_PAGE)

        intent = ActionIntent(
            action=ActionType.CLICK,
            target="#btn",
            persona_thought="Clicking button.",
            expected_outcome="Button clicked.",
            confidence=0.9,
        )

        state = await executor.execute(intent, step=1)
        assert "CLICKED" in state.visible_text
