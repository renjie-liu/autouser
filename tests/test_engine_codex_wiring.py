"""provider='codex' constructs a CodexProvider and routes _call_llm_once to it."""

import pytest

from autouser.cognitive.engine import CognitiveEngine
from autouser.cognitive.codex_provider import CodexProvider
from autouser.persona.models import Persona


def _engine() -> CognitiveEngine:
    return CognitiveEngine(
        Persona(),
        task_description="t",
        success_criteria="c",
        provider="codex",
    )


def test_codex_provider_constructs_codex_client():
    engine = _engine()
    assert engine.provider == "codex"
    assert isinstance(engine._codex_client, CodexProvider)


@pytest.mark.asyncio
async def test_call_llm_once_routes_to_codex(monkeypatch):
    engine = _engine()
    calls = {}

    async def fake_complete(prompt, *, schema):
        calls["prompt"] = prompt
        calls["schema"] = schema
        return '{"ok": true}'

    monkeypatch.setattr(engine._codex_client, "complete", fake_complete)
    out = await engine._call_llm_once(
        "sys", "usr", 256, schema={"type": "object"}
    )
    assert out == '{"ok": true}'
    assert calls["schema"] == {"type": "object"}


def test_codex_defers_model_to_codex_config():
    engine = _engine()  # provider="codex", no model passed
    assert engine.model is None
    assert engine._codex_client._model is None


def test_codex_is_env_selectable(monkeypatch):
    monkeypatch.setenv("AUTOUSER_PROVIDER", "codex")
    from autouser.cognitive.engine import _detect_provider
    assert _detect_provider() == "codex"
