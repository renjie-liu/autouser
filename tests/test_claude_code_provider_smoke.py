"""Opt-in live smoke test for ClaudeCodeProvider.

Skipped by default. Run with:

    AUTOUSER_CLAUDE_CODE_SMOKE=1 pytest tests/test_claude_code_provider_smoke.py

Requires the `claude` CLI on PATH and an authenticated session
(`claude` invoked interactively at least once on this machine). The test
spawns a real subprocess and asserts the provider produces a payload
conforming to the requested schema.

Not part of the default CI suite — exists to validate the provider end-to-end
during local development and before the parity test (PR 2) lands. The parity
test exercises the full engine path; this test isolates the provider boundary.
"""

from __future__ import annotations

import json
import os

import pytest

from autouser.cognitive.claude_code_provider import ClaudeCodeProvider


_SMOKE = os.environ.get("AUTOUSER_CLAUDE_CODE_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not _SMOKE,
    reason="opt-in: set AUTOUSER_CLAUDE_CODE_SMOKE=1 and authenticate `claude` to run",
)


@pytest.mark.asyncio
async def test_live_claude_returns_schema_conforming_payload() -> None:
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
            "answer": {"type": "integer"},
        },
        "required": ["greeting", "answer"],
    }
    provider = ClaudeCodeProvider(timeout=60.0)
    payload_str = await provider.complete(
        "Return greeting='hello' and answer=42 as JSON matching the schema.",
        schema=schema,
    )
    payload = json.loads(payload_str)
    assert isinstance(payload.get("greeting"), str)
    assert isinstance(payload.get("answer"), int)
