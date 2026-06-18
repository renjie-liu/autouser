"""Tests for the friction fixture page — selector smoke tests and predicate validation.

Validates that SuccessPredicate.selector_present strings match the actual
dom_summary output from BrowserExecutor's _DOM_SUMMARY_JS, preventing
false negatives from selector-format mismatches.
"""

from __future__ import annotations

import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial

import pytest
import yaml

from autouser.cognitive.models import (
    FeedbackSubtype,
    SuccessPredicate,
    UIState,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_HTML = FIXTURES_DIR / "friction_page.html"
EXPECTED_FRICTION = FIXTURES_DIR / "expected_friction.yaml"


# ---------------------------------------------------------------------------
# Fixture server — serves the HTML page on a random port
# ---------------------------------------------------------------------------

class _QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(directory), **kwargs)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def fixture_server():
    handler = partial(_QuietHandler, directory=FIXTURES_DIR)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def expected_friction():
    with open(EXPECTED_FRICTION) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Unit tests — SuccessPredicate matching against UIState (no browser needed)
# ---------------------------------------------------------------------------

class TestSelectorSmokeUnit:
    """Validate selector_present strings match dom_summary format conventions."""

    def _make_state(self, dom_summary: str, url: str = "http://localhost") -> UIState:
        return UIState(
            url=url,
            page_title="Test",
            dom_summary=dom_summary,
            visible_text="",
        )

    def test_tag_id_positive_match(self):
        state = self._make_state('section#save-confirmed | role=status | text="Data persisted."')
        assert state.has_selector("section#save-confirmed")

    def test_tag_id_wrong_tag_no_match(self):
        state = self._make_state('section#save-confirmed | role=status | text="Data persisted."')
        assert not state.has_selector("button#save-confirmed")

    def test_class_notation_no_match_when_id_used(self):
        state = self._make_state('section#save-confirmed | role=status | text="Data persisted."')
        assert not state.has_selector(".save-confirmed")

    def test_bare_id_is_substring_match(self):
        """#save-confirmed IS a substring of section#save-confirmed — this is
        expected behavior from the raw substring match, not a bug. Document it."""
        state = self._make_state('section#save-confirmed | role=status | text="Data persisted."')
        assert state.has_selector("#save-confirmed")

    def test_landmark_section_appears_in_summary(self):
        summary = (
            "== Landmarks ==\n"
            "  section#pattern-icon\n"
            "  section#pattern-swap\n"
            "  nav#swap-actions | role=navigation\n"
            "  section#pattern-phone\n"
            "  section#pattern-disabled\n"
            "  section#pattern-silent\n"
            "  section#save-confirmed | role=status\n"
            "\n"
            "== Interactive Elements ==\n"
            '  button#icon-submit | text="✔"\n'
            '  button#swap-cancel | text="Cancel"\n'
            '  button#swap-continue | text="Continue"\n'
        )
        state = self._make_state(summary)
        assert state.has_selector("section#save-confirmed")
        assert state.has_selector("section#pattern-icon")
        assert state.has_selector("button#icon-submit")
        assert not state.has_selector("div#save-confirmed")
        assert not state.has_selector("button#nonexistent")


class TestPredicateYamlConsistency:
    """Validate that YAML predicate definitions produce valid SuccessPredicate objects."""

    def test_all_yaml_predicates_are_constructible(self, expected_friction):
        patterns = expected_friction["patterns"]
        for name, pattern in patterns.items():
            pred_def = pattern["success_predicate"]
            pred = SuccessPredicate(**pred_def)
            assert pred.selector_present or pred.url_pattern or pred.text_contains, (
                f"Pattern '{name}' has an empty predicate"
            )

    def test_silent_success_predicate_uses_landmark_selector(self, expected_friction):
        pred_def = expected_friction["patterns"]["silent_success"]["success_predicate"]
        pred = SuccessPredicate(**pred_def)
        assert pred.selector_present is not None
        assert pred.selector_present.startswith("section#"), (
            f"Silent-success selector should use a landmark tag, got: {pred.selector_present}"
        )

    def test_silent_success_expects_feedback_subtype(self, expected_friction):
        maria = expected_friction["patterns"]["silent_success"]["expected"]["maria"]
        assert maria.get("feedback_subtype") == FeedbackSubtype.SILENT_SUCCESS.value


# ---------------------------------------------------------------------------
# Integration tests — require Playwright (skip if not available)
# ---------------------------------------------------------------------------

playwright = pytest.importorskip("playwright.async_api", reason="Playwright not installed")


@pytest.mark.asyncio
class TestDomSummarySmoke:
    """Validate selector strings against actual BrowserExecutor dom_summary output."""

    async def test_fixture_page_dom_summary_contains_landmarks(self, fixture_server):
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"{fixture_server}/friction_page.html")
            await page.wait_for_load_state("domcontentloaded")

            from autouser.execution.browser import _DOM_SUMMARY_JS
            dom_summary = await page.evaluate(_DOM_SUMMARY_JS)

            assert "section#pattern-icon" in dom_summary
            assert "section#pattern-swap" in dom_summary
            assert "section#pattern-phone" in dom_summary
            assert "section#pattern-disabled" in dom_summary
            assert "section#pattern-silent" in dom_summary

            assert "button#icon-submit" in dom_summary
            assert "button#swap-cancel" in dom_summary
            assert "button#swap-continue" in dom_summary
            assert "button#phone-submit" in dom_summary
            assert "button#disabled-look-btn" in dom_summary
            assert "button#silent-save" in dom_summary

            await browser.close()

    async def test_silent_save_reveals_confirmed_section(self, fixture_server):
        """After clicking Save, section#save-confirmed should appear in dom_summary."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"{fixture_server}/friction_page.html")
            await page.wait_for_load_state("domcontentloaded")

            await page.click("#silent-save")
            await page.wait_for_timeout(200)

            from autouser.execution.browser import _DOM_SUMMARY_JS
            dom_summary = await page.evaluate(_DOM_SUMMARY_JS)

            assert "section#save-confirmed" in dom_summary, (
                f"Expected section#save-confirmed in dom_summary after Save click. "
                f"Got:\n{dom_summary}"
            )

            await browser.close()

    async def test_negative_selectors_do_not_match(self, fixture_server):
        """Wrong-tag and class-notation selectors must NOT match dom_summary."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"{fixture_server}/friction_page.html")
            await page.wait_for_load_state("domcontentloaded")

            from autouser.execution.browser import _DOM_SUMMARY_JS
            dom_summary = await page.evaluate(_DOM_SUMMARY_JS)

            # Wrong tag for existing id
            assert "button#pattern-icon" not in dom_summary
            assert "div#save-confirmed" not in dom_summary
            # Class notation — elements use id, not class-based selectors
            assert ".pattern-icon" not in dom_summary

            await browser.close()

    async def test_predicate_matches_against_live_dom(self, fixture_server):
        """SuccessPredicate.matches() works against real BrowserExecutor output."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"{fixture_server}/friction_page.html")
            await page.wait_for_load_state("domcontentloaded")

            from autouser.execution.browser import _DOM_SUMMARY_JS
            dom_summary = await page.evaluate(_DOM_SUMMARY_JS)
            visible_text = await page.evaluate("() => document.body.innerText")

            state = UIState(
                url=page.url,
                page_title=await page.title(),
                dom_summary=dom_summary,
                visible_text=visible_text,
            )

            icon_pred = SuccessPredicate(
                selector_present="section#pattern-icon",
                text_contains="Pattern 1",
            )
            assert icon_pred.matches(state)

            wrong_pred = SuccessPredicate(
                selector_present="button#nonexistent",
            )
            assert not wrong_pred.matches(state)

            await browser.close()
