# Codex CLI provider

**Status:** Design (approved 2026-06-18)
**Scope:** Add a `codex` cognitive-engine provider that drives the local OpenAI
`codex` CLI, mirroring the existing `claude_code` provider — no API key, uses the
user's authenticated `codex` CLI.

## Motivation

The cognitive engine supports `anthropic` and `gemini` (SDK + API key) and
`claude_code` (spawns the local `claude` CLI, no key). A symmetric `codex`
provider lets users drive simulations with an existing OpenAI/ChatGPT `codex`
subscription with zero key setup — the same value proposition as `claude_code`.

## The codex interface

`codex exec` runs Codex non-interactively (verified against codex-cli 0.133.0).
Relevant flags:

- `[PROMPT]` positional (or stdin) — the instructions.
- `-m, --model <MODEL>` — model override (optional; otherwise codex's
  `~/.codex/config.toml` default).
- `--output-schema <FILE>` — a JSON Schema file constraining the model's final
  response shape (schema-enforced structured output).
- `-o, --output-last-message <FILE>` — write only the final agent message to a
  file.
- `--skip-git-repo-check` — allow running outside a git repository.
- `-c <key=value>` — config overrides (TOML), e.g. `-c sandbox_mode="read-only"`,
  `-c approval_policy="never"`.

This gives schema-enforced output (like `claude --json-schema`) plus a clean way
to read just the final answer.

## Design

New module `src/autouser/cognitive/codex_provider.py`, mirroring
`claude_code_provider.py`'s three-piece, independently-testable boundary.

### 1. Command builder (pure)

`build_command(prompt, *, schema_path, output_path, model=None) -> list[str]`:

```
codex exec
  [--model <model>]                 # omitted when model is None -> codex config default
  --skip-git-repo-check
  --output-schema <schema_path>
  --output-last-message <output_path>
  -c sandbox_mode="read-only"
  -c approval_policy="never"
  -- <prompt>
```

Guardrails (the analog of `claude_code`'s `--tools ""` + `--no-session-persistence`):

- `sandbox_mode="read-only"` — codex may not write to the repo/filesystem.
- `approval_policy="never"` — codex never blocks on an interactive approval
  prompt (one-shot, non-interactive).
- `--skip-git-repo-check` — runs regardless of CWD.
- `--` terminates option parsing so the prompt is unambiguously positional.

### 2. Subprocess runner (injectable)

Reuse the same `SubprocessRunner` shape
(`Callable[[Sequence[str], float], Awaitable[CompletedSubprocess]]`) and a
`_default_runner` (asyncio subprocess + wall-clock timeout, kill+drain on
timeout). Tests inject a mock runner.

### 3. Output reader

After a zero-exit run, read the `--output-last-message` file; its contents are
the schema-conformant JSON payload string. `parse_output(text)`:

- empty / missing file -> `CodexOutputError`
- otherwise return the text as-is (the caller parses it to the domain model —
  same contract as `claude_code`'s `complete()` returning the payload string).

We use `--output-last-message` rather than parsing the `--json` JSONL event
stream: fewer moving parts and more version-stable.

### Provider class

`CodexProvider.complete(prompt, *, schema) -> str`:

1. Write `schema` to a temp file; allocate a temp output-message file.
2. Spawn under the process-global spawn semaphore (`AUTOUSER_CODEX_MAX_SPAWNS`,
   default 1) with timeout (`AUTOUSER_CODEX_TIMEOUT`, default 120s; explicit
   constructor arg wins).
3. Translate failures to typed errors; on success read + return the payload.
4. Always clean up temp files.

### Typed errors

Mirror `claude_code`'s taxonomy: `CodexNotFoundError` (binary missing),
`CodexAuthError` (not logged in — stderr auth hints), `CodexTimeoutError`,
`CodexInvocationError` (nonzero exit), `CodexOutputError` (missing/empty final
message), and `CodexProcessLimitError` (EPERM/EAGAIN spawn-time ceiling — codex
also forks a child tree, so reuse the same errno + stderr-signature
discriminator).

### Concurrency + timeout

Reuse the loop-lazy, process-global spawn-semaphore pattern with codex-specific
env vars `AUTOUSER_CODEX_MAX_SPAWNS` (default 1) and `AUTOUSER_CODEX_TIMEOUT`
(default 120s). Codex, like claude, forks children; the per-uid process ceiling
is shared, so the cap lives at the spawn point.

## Engine wiring (`engine.py`)

- Client init: `elif self.provider == "codex": self._codex_client = CodexProvider(model=self.model)`.
- `_call_llm_once`: add a `codex` branch that calls
  `self._codex_client.complete(prompt, schema=schema)`, reusing the SAME schema
  already built for the `claude_code` path (`_PLAN_SCHEMA` / `_REFLECT_SCHEMA`).
  Retry/parse handling is shared.
- `_DEFAULT_MODELS`: add a `"codex": None` entry. The `None` value makes the model
  resolution (`model or _DEFAULT_MODELS.get(provider, ...)`) yield `None` for codex —
  so no `--model` is emitted and codex uses its own config default — while the key's
  *presence* lets `_detect_provider` honor `AUTOUSER_PROVIDER=codex` and lets the CLI's
  `_PROVIDERS = sorted(_DEFAULT_MODELS)` offer `--provider codex`. An explicit
  `CognitiveEngine(model=...)` / `--model` still overrides.
- `_detect_provider`: codex stays **opt-in** — selected only via
  `AUTOUSER_PROVIDER=codex` or `CognitiveEngine(provider="codex")`, never
  auto-selected from key presence. Same posture as `claude_code`.

## CLI

`autouser run --provider codex ...` works through the existing `--provider`
plumbing — no new CLI flags. The CLI's no-key auto-fallback remains
`claude_code`; codex is explicit-only.

## Testing

- `tests/test_codex_provider.py` (pure, mocked runner): command shape + each
  guardrail flag; schema/output temp-file handling; `parse_output` happy +
  empty/missing; every typed-error path (not-found, auth, timeout, nonzero exit,
  process-limit signature); the spawn-cap concurrency test.
- `tests/test_engine_codex_wiring.py`: `provider="codex"` constructs a
  `CodexProvider` and `_call_llm_once` routes to it (mocked).
- `tests/test_codex_provider_smoke.py` (opt-in live, `AUTOUSER_CODEX_SMOKE=1`):
  spawns the real `codex` to validate one end-to-end plan call. Skips by default.
  Mirrors `test_claude_code_provider_smoke.py`.

## Docs

- README provider table: add a `codex` row ("authenticated `codex` CLI, no API
  key").
- CLAUDE.md provider notes: codex knobs (`AUTOUSER_CODEX_MAX_SPAWNS`,
  `AUTOUSER_CODEX_TIMEOUT`) and the opt-in posture.
- `.env.example`: note codex needs no key (authenticated CLI).

## Non-goals / risks

- Not auto-selected; explicit opt-in only.
- `codex exec` is an agent, so it is heavier per call than a raw API (sandbox
  spin-up). Acceptable for the opt-in use case; documented like `claude_code`.
- The `read-only` sandbox + `never` approval config is the safety boundary that
  keeps codex from touching the repo while answering. If a future codex version
  renames these config keys, the smoke test catches it.
- Schema-enforced output depends on `--output-schema` / `--output-last-message`
  (present in codex >= 0.133); the smoke test pins the behavior.
