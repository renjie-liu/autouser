"""Opt-in live smoke test for the codex provider — spawns the real `codex`.

Skipped unless AUTOUSER_CODEX_SMOKE=1 (and a logged-in `codex` CLI on PATH).
Mirrors tests/test_claude_code_provider_smoke.py.
"""

import json
import os
import shutil

import pytest

from autouser.cognitive.codex_provider import CodexProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOUSER_CODEX_SMOKE") != "1" or shutil.which("codex") is None,
    reason="set AUTOUSER_CODEX_SMOKE=1 with a logged-in `codex` CLI to run",
)

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_codex_returns_schema_conformant_json():
    provider = CodexProvider()
    payload = await provider.complete(
        'Reply with JSON only: {"answer": "pong"}. Do not run any tools.',
        schema=_SCHEMA,
    )
    obj = json.loads(payload)
    assert "answer" in obj
