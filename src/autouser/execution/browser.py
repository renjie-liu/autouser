"""Playwright-based browser executor."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

from playwright.async_api import Dialog, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from autouser.cognitive.models import (
    ActionIntent,
    ActionType,
    CoordinateTarget,
    UIState,
)

logger = logging.getLogger(__name__)

# Operator override for the Chromium binary Playwright launches. When set,
# it is passed straight through as `executable_path`. This exists because
# Playwright pins an exact managed-browser *revision* to the installed
# playwright package, and on some platforms that revision can't be fetched
# (`playwright install chromium` has no build — e.g. arm64 / newer Ubuntu),
# even though an older cached revision is present and works. Pointing this at
# that cached `chrome` binary lets the prototype run where the managed install
# can't. Unset → Playwright resolves its default managed revision exactly as
# before (no behavior change for environments that have the matching build).
_BROWSER_EXECUTABLE_ENV = "AUTOUSER_BROWSER_EXECUTABLE"

# Max characters for DOM summary to keep LLM context reasonable
_DOM_SUMMARY_MAX_CHARS = 8000

# --- Motor fidelity (low motor_precision personas) ---------------------------
#
# Harness-enforced, not prompt-described: a low-precision persona's clicks land
# with gaussian noise around the target's center, so some genuinely miss and
# hit whatever is adjacent — the executor doesn't pretend, it perturbs. Sigma
# is proportional to target size: big well-separated buttons are easy (the
# noise rarely escapes the box), small dense targets are hard — which is
# exactly the accessibility guidance about touch-target sizing.
_POINTER_NOISE_SIGMA_RATIO = 0.30  # ≈90% per-axis hit rate on the target box

# Per-character probability of a typo (adjacent-key press then Backspace) when
# typing with low motor precision. Keystrokes are real key events, so
# keydown-driven UX (live validation, character counters) sees the fumble.
_TYPO_PROBABILITY = 0.04
_TYPE_DELAY_MS = 30

_ADJACENT_KEYS = {
    "a": "s", "b": "v", "c": "x", "d": "f", "e": "r", "f": "g", "g": "h",
    "h": "j", "i": "o", "j": "k", "k": "l", "l": "k", "m": "n", "n": "m",
    "o": "p", "p": "o", "q": "w", "r": "t", "s": "d", "t": "y", "u": "i",
    "v": "b", "w": "e", "x": "c", "y": "u", "z": "x",
    "1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7", "7": "8",
    "8": "9", "9": "0", "0": "9",
}


def _pointer_noise(width: float, height: float) -> tuple[float, float]:
    """Sample a click offset from the target center for an imprecise pointer.

    Module-level so tests can monkeypatch it deterministic (force a miss or
    a hit) without touching the executor internals.
    """
    return (
        random.gauss(0, width * _POINTER_NOISE_SIGMA_RATIO),
        random.gauss(0, height * _POINTER_NOISE_SIGMA_RATIO),
    )


def _typo_for(char: str) -> Optional[str]:
    """Return the adjacent-key typo to make before *char*, or None.

    Module-level for deterministic monkeypatching in tests.
    """
    neighbor = _ADJACENT_KEYS.get(char.lower())
    if neighbor and random.random() < _TYPO_PROBABILITY:
        return neighbor
    return None


# JS snippet describing the element that currently holds keyboard focus.
# Returns '' when focus sits on <body> (i.e. nothing meaningfully focused).
# Attribute names are REAL (aria-label=/placeholder=/...) for the same reason
# as the DOM summary: anything shown to the LLM may be copied into a selector.
_FOCUSED_ELEMENT_JS = """
() => {
    const el = document.activeElement;
    if (!el || el === document.body || el === document.documentElement) return '';
    const tag = el.tagName.toLowerCase();
    let selector = tag;
    if (el.id) selector += '#' + el.id;
    let label = '';
    let labelAttr = '';
    for (const attr of ['aria-label', 'placeholder', 'alt', 'title']) {
        const v = el.getAttribute(attr);
        if (v) { label = v; labelAttr = attr; break; }
    }
    const text = (el.textContent || '').trim().slice(0, 40);
    const parts = [selector];
    if (label) parts.push(labelAttr + '="' + label + '"');
    else if (text) parts.push('text="' + text + '"');
    return parts.join(' | ');
}
"""


# The DOM summary shows an element's text content as `text="..."` after a
# `|`. The model frequently copies that token into a CSS attribute filter —
# `button.foo[text="Create Project"]` — which is not valid CSS (`text` is no
# attribute), matches nothing, and times out. The summary even invites it
# ("append ONE attribute shown on that line"). Rewrite that pseudo-attribute
# into Playwright's `:text-is()` exact-text pseudo-class, preserving the CSS
# prefix that disambiguates which element. Caught on a live SPA
# exploration run, where every Create-Project click failed this way.
_TEXT_ATTR_RE = re.compile(r"""\[text=(?P<q>["'])(?P<val>.*?)(?P=q)\]""")


def _normalize_text_selector(selector: str) -> str:
    """Make the cognitive layer's text-based targets executable.

    Two rewrites:
    * Bare ``text=foo`` → ``text="foo"`` — Playwright's ``text=foo`` is a
      *substring* match, so with both a modal "Create Project" and a
      background "+ Create Project" present it resolves to the occluded
      background button and the click times out. Exact matching is the
      intended semantics (the model copies exact visible text).
    * ``sel[text="foo"]`` → ``sel:text-is("foo")`` — the invalid CSS
      pseudo-attribute the model builds from the summary's `text="..."`
      display token, converted to an exact-text pseudo-class that keeps the
      CSS prefix.

    Truncated texts (>80 chars in the summary) will miss under exact
    matching — that degrades into an action_error the persona reacts to,
    which beats silently clicking the wrong element.
    """
    if selector.startswith("text="):
        rest = selector[5:]
        if rest and not rest.startswith(('"', "'")):
            escaped = rest.replace('"', '\\"')
            return f'text="{escaped}"'
        return selector

    def _repl(m: re.Match) -> str:
        val = m.group("val").replace('"', '\\"')
        return f':text-is("{val}")'

    return _TEXT_ATTR_RE.sub(_repl, selector)

# JS snippet that produces a simplified, accessible DOM representation.
# Returns text descriptions of interactive and landmark elements — the kind of
# information a real user would perceive — rather than raw HTML.
_DOM_SUMMARY_JS = """
() => {
    const INTERACTIVE = 'a,button,input,select,textarea,[role="button"],[role="link"],[role="tab"],[role="menuitem"]';
    const LANDMARK = 'header,footer,main,nav,aside,section,form,dialog,[role="banner"],[role="navigation"],[role="main"],[role="dialog"]';

    function describe(el) {
        const tag = el.tagName.toLowerCase();
        const role = el.getAttribute('role') || '';
        // Report the label under its REAL attribute name. A generic `label="..."`
        // key teaches the LLM an attribute that doesn't exist in the DOM — it
        // then emits selectors like input[label="Name"] which match nothing.
        // aria-label="..." / placeholder="..." etc. are valid CSS attribute
        // selectors the model can copy verbatim.
        let label = '';
        let labelAttr = '';
        for (const attr of ['aria-label', 'alt', 'placeholder', 'title']) {
            const v = el.getAttribute(attr);
            if (v) { label = v; labelAttr = attr; break; }
        }
        const text = (el.textContent || '').trim().slice(0, 80);
        const type = el.getAttribute('type') || '';
        const href = el.getAttribute('href') || '';
        const disabled = el.disabled ? ' [disabled]' : '';
        const hidden =
            el.offsetParent === null && el.tagName !== 'BODY' ? ' [hidden]' : '';

        // Current field contents — what a SIGHTED user sees after typing.
        // Omitting it made filled inputs look unchanged, so a persona kept
        // re-typing, blind to their own input (false 'no confirmation'
        // findings). Passwords are masked; checkboxes report checked state.
        let contents = '';
        if (tag === 'input' && (type === 'checkbox' || type === 'radio')) {
            contents = el.checked ? ' (checked)' : ' (unchecked)';
        } else if (tag === 'input' || tag === 'textarea') {
            const v = el.value || '';
            if (v) contents = type === 'password'
                ? ' (filled)'
                : ' contains "' + v.slice(0, 60) + '"';
        }

        let selector = tag;
        if (el.id) selector += '#' + el.id;
        else if (el.className && typeof el.className === 'string')
            selector += '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.');

        const parts = [selector];
        if (role) parts.push('role=' + role);
        if (type) parts.push('type=' + type);
        if (label) parts.push(labelAttr + '="' + label + '"');
        if (text && text !== label) parts.push('text="' + text + '"');
        if (href) parts.push('href="' + href.slice(0, 120) + '"');
        parts.push(contents + disabled + hidden);
        return parts.filter(Boolean).join(' | ');
    }

    const lines = [];
    lines.push('== Landmarks ==');
    document.querySelectorAll(LANDMARK).forEach(el => {
        lines.push('  ' + describe(el));
    });
    lines.push('');
    lines.push('== Interactive Elements ==');
    document.querySelectorAll(INTERACTIVE).forEach(el => {
        lines.push('  ' + describe(el));
    });
    return lines.join('\\n');
}
"""


class BrowserExecutor:
    """Executes actions in a real browser and captures UI state.

    Wraps Playwright to provide:
    - Action execution (click, type, scroll, navigate, back)
    - UI state capture (DOM summary, screenshot, visible text)
    - Page state management (wait for navigation, handle dialogs)

    Designed so that the caller can instantiate multiple ``BrowserExecutor``
    instances (one per persona) and run them concurrently — each owns its own
    browser context.
    """

    def __init__(
        self,
        screenshot_dir: Optional[Path] = None,
        *,
        browser_executable: Optional[str] = None,
        imprecise_pointer: bool = False,
    ) -> None:
        self.screenshot_dir = screenshot_dir or Path("./screenshots")
        # Low motor_precision personas: clicks carry gaussian noise (may miss
        # and hit neighbors) and typing is real keystrokes with occasional
        # adjacent-key typos. Set by the runner from the persona profile.
        self.imprecise_pointer = imprecise_pointer
        # Explicit arg wins; otherwise fall back to the env override; otherwise
        # None → Playwright's default managed revision. `or None` collapses an
        # empty-string env value to the default rather than passing "" through.
        self._browser_executable = (
            browser_executable or os.environ.get(_BROWSER_EXECUTABLE_ENV) or None
        )
        self._playwright = None
        self._browser = None
        self._context = None
        self._page: Optional[Page] = None

    def _launch_kwargs(self) -> dict[str, object]:
        """Build the kwargs for `chromium.launch()`.

        Pure (reads only constructed state) so the executable-override wiring
        can be unit-tested without spawning a real browser. `executable_path`
        is included ONLY when an override is set — when absent, the call is
        byte-identical to the original `launch(headless=True)`.
        """
        kwargs: dict[str, object] = {"headless": True}
        if self._browser_executable:
            kwargs["executable_path"] = self._browser_executable
        return kwargs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, url: str) -> UIState:
        """Launch browser, navigate to *url*, and return the initial UI state."""
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            **self._launch_kwargs()
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(60000)
        self._page.set_default_navigation_timeout(60000)

        # Auto-dismiss unexpected dialogs so the loop doesn't hang.
        # dismiss() is async in the Playwright async API — fire-and-forget via task.
        async def _dismiss_dialog(dialog: Dialog) -> None:
            await dialog.dismiss()

        self._page.on("dialog", lambda d: asyncio.ensure_future(_dismiss_dialog(d)))

        await self._page.goto(url, wait_until="domcontentloaded")
        return await self.capture_state(step=0)

    async def stop(self) -> None:
        """Close the browser and release Playwright resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _validate_coordinate_target(
        self,
        coord: CoordinateTarget,
        expected_screenshot_id: str | None = None,
    ) -> None:
        """Validate coordinate target against current viewport bounds.

        Raises ValueError if coordinates are out of bounds or the screenshot
        ID is stale (doesn't match the most recent capture).
        """
        if expected_screenshot_id and coord.screenshot_id != expected_screenshot_id:
            raise ValueError(
                f"Stale screenshot: target references {coord.screenshot_id!r} "
                f"but current is {expected_screenshot_id!r}. "
                f"Page may have changed since coordinates were computed."
            )
        # Validate against viewport bounds (set at context creation: 1280x720)
        viewport = self._page.viewport_size if self._page else None
        if viewport:
            w, h = viewport["width"], viewport["height"]
            if coord.x < 0 or coord.x >= w or coord.y < 0 or coord.y >= h:
                raise ValueError(
                    f"Coordinate ({coord.x}, {coord.y}) out of viewport "
                    f"bounds ({w}x{h})"
                )

    async def execute(
        self,
        intent: ActionIntent,
        step: int,
        *,
        expected_screenshot_id: str | None = None,
    ) -> UIState:
        """Execute *intent* in the browser and return the resulting UI state.

        If the intent has a CoordinateTarget, validates bounds and screenshot
        freshness before executing. Raises ValueError on stale or out-of-bounds
        coordinates — fail closed, don't click at wrong positions.
        """
        assert self._page is not None, "call start() before execute()"
        page = self._page

        action = intent.action
        value = intent.input_value

        # Determine targeting mode
        coord_target = (
            intent.typed_target
            if isinstance(intent.typed_target, CoordinateTarget)
            else None
        )
        selector_target = intent.target if not coord_target else None
        if selector_target:
            selector_target = _normalize_text_selector(selector_target)

        # Validate coordinate targets before execution
        if coord_target:
            self._validate_coordinate_target(coord_target, expected_screenshot_id)

        # A failed action (selector matched nothing, element not interactable)
        # must NOT crash the simulation — a real user who clicks something
        # unresponsive just sees nothing happen and tries again. Capture the
        # failure as UIState.action_error so reflect() can react in character;
        # the loop's mismatch/patience machinery handles the rest.
        action_error: Optional[str] = None
        try:
            if action == ActionType.CLICK:
                if coord_target:
                    logger.debug(
                        "coordinate_click: (%d, %d) screenshot=%s",
                        coord_target.x, coord_target.y, coord_target.screenshot_id,
                    )
                    await page.mouse.click(coord_target.x, coord_target.y)
                elif self.imprecise_pointer:
                    await self._imprecise_click(page, selector_target)
                else:
                    await page.click(selector_target, timeout=5000)

            elif action == ActionType.TYPE:
                if coord_target:
                    # Click to focus the field first, then type
                    await page.mouse.click(coord_target.x, coord_target.y)
                    await page.keyboard.type(value or "")
                elif not selector_target:
                    # Keyboard-user flow: type into whatever holds focus.
                    # Typing with nothing focused does nothing — exactly like
                    # a real user typing into a page with no active field.
                    focused = await page.evaluate(_FOCUSED_ELEMENT_JS)
                    if not focused:
                        raise PlaywrightError(
                            "no element has keyboard focus; typed input went nowhere"
                        )
                    await self._type_text(page, value or "")
                elif self.imprecise_pointer:
                    # Real keystrokes (with possible typos) so keydown-driven
                    # UX reacts; focus via a precise click — motor noise
                    # applies to CLICK actions, not to the focusing step,
                    # otherwise typing runs compound two noise sources.
                    await page.click(selector_target, timeout=5000)
                    await self._type_text(page, value or "")
                else:
                    # fill() sets the value programmatically. Does not simulate
                    # keystrokes, so won't trigger keydown-driven UX. Acceptable for v1.
                    # Same bounded timeout as click — a bad selector should fail in
                    # 5s, not consume the 60s page default per attempt.
                    await page.fill(selector_target, value or "", timeout=5000)

            elif action == ActionType.PRESS:
                # Keyboard navigation: Tab/Shift+Tab to move focus,
                # Enter/Space to activate, Escape, arrows. Invalid key names
                # raise and degrade into action_error like any failed action.
                await page.keyboard.press(value or "")

            elif action == ActionType.SCROLL:
                await page.evaluate("window.scrollBy(0, 500)")

            elif action == ActionType.NAVIGATE:
                if value:
                    await page.goto(value, wait_until="domcontentloaded")

            elif action == ActionType.BACK:
                await page.go_back(wait_until="domcontentloaded")

            elif action == ActionType.WAIT:
                await page.wait_for_timeout(1000)

            elif action == ActionType.GIVE_UP:
                pass  # no-op — runner handles termination
        except PlaywrightError as exc:
            detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            target_desc = (
                f"({coord_target.x}, {coord_target.y})" if coord_target
                else repr(selector_target)
            )
            action_error = f"{action.value} on {target_desc} failed: {detail}"
            logger.warning("action_failed: %s", action_error)

        # Let the page settle after the action (SPA transitions, animations).
        await page.wait_for_load_state("domcontentloaded")

        return await self.capture_state(step, action_error=action_error)

    async def _imprecise_click(self, page: Page, selector: str) -> None:
        """Click with motor noise: gaussian offset around the target center.

        The perturbed point may land outside the target — the click then hits
        whatever occupies that spot (a neighbor, empty space), with real
        consequences the persona must perceive and react to. Falls back to a
        precise click when the element has no bounding box (e.g. covered):
        the actionability error path is more informative than noise there.
        """
        box = await page.locator(selector).first.bounding_box()
        if box is None:
            await page.click(selector, timeout=5000)
            return
        dx, dy = _pointer_noise(box["width"], box["height"])
        x = box["x"] + box["width"] / 2 + dx
        y = box["y"] + box["height"] / 2 + dy
        viewport = page.viewport_size or {"width": 1280, "height": 720}
        x = min(max(x, 0), viewport["width"] - 1)
        y = min(max(y, 0), viewport["height"] - 1)
        missed = not (
            box["x"] <= x <= box["x"] + box["width"]
            and box["y"] <= y <= box["y"] + box["height"]
        )
        if missed:
            logger.debug(
                "imprecise_click_missed: target=%s point=(%.0f, %.0f)", selector, x, y
            )
        await page.mouse.click(x, y)

    async def _type_text(self, page: Page, text: str) -> None:
        """Type *text* as real keystrokes into the focused element.

        With an imprecise pointer, occasionally fat-fingers an adjacent key
        and corrects it with Backspace — keydown-driven UX (live validation,
        counters) observes the fumble, and the final value is still correct.
        """
        for char in text:
            if self.imprecise_pointer:
                typo = _typo_for(char)
                if typo:
                    await page.keyboard.type(typo, delay=_TYPE_DELAY_MS)
                    await page.keyboard.press("Backspace")
            await page.keyboard.type(char, delay=_TYPE_DELAY_MS)

    # ------------------------------------------------------------------
    # State capture
    # ------------------------------------------------------------------

    async def capture_state(
        self, step: int, *, action_error: Optional[str] = None
    ) -> UIState:
        """Capture the current page state: URL, title, DOM summary, screenshot.

        ``action_error`` annotates the snapshot with a preceding action failure
        so the cognitive layer can perceive "nothing happened" honestly instead
        of hallucinating an outcome.
        """
        assert self._page is not None, "call start() before capture_state()"
        page = self._page

        url = page.url
        title = await page.title()

        # DOM summary — a simplified, accessibility-oriented view of the page.
        dom_summary: str = await page.evaluate(_DOM_SUMMARY_JS)
        if len(dom_summary) > _DOM_SUMMARY_MAX_CHARS:
            dom_summary = dom_summary[:_DOM_SUMMARY_MAX_CHARS] + "\n[...truncated]"

        # Visible text — extract the readable body text.
        visible_text: str = await page.evaluate(
            "() => (document.body && document.body.innerText || '').slice(0, 4000)"
        )

        # Accessibility tree (role + accessible name, document order) — raw
        # material for the screen-reader perception filter. Captured for every
        # state so a Priya run perceives what assistive tech would actually
        # expose, not the visual element inventory. Defensive: aria_snapshot()
        # exists from playwright 1.49; degrade to None rather than fail capture.
        aria_snapshot: Optional[str] = None
        try:
            aria_snapshot = await page.locator("body").aria_snapshot()
            if aria_snapshot and len(aria_snapshot) > _DOM_SUMMARY_MAX_CHARS:
                aria_snapshot = aria_snapshot[:_DOM_SUMMARY_MAX_CHARS] + "\n[...truncated]"
        except (PlaywrightError, AttributeError) as exc:
            logger.debug("aria_snapshot unavailable: %s", exc)

        # Keyboard focus — '' (→ None) when focus sits on <body>.
        focused_element: Optional[str] = await page.evaluate(_FOCUSED_ELEMENT_JS) or None

        # Screenshot
        screenshot_path = str(self.screenshot_dir / f"step_{step:03d}.png")
        await page.screenshot(path=screenshot_path, full_page=False)

        return UIState(
            url=url,
            page_title=title,
            dom_summary=dom_summary,
            visible_text=visible_text,
            aria_snapshot=aria_snapshot,
            focused_element=focused_element,
            screenshot_path=screenshot_path,
            action_error=action_error,
        )
