# Contributing to AutoUser

Thanks for your interest in improving AutoUser! This guide covers local setup,
the test/lint workflow, and the conventions the codebase follows.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use uv / your tool of choice
pip install -e ".[dev]"
playwright install chromium
```

If `playwright install chromium` is blocked by network policy, see the
**Setup** section of the [README](README.md) for using a cached Chromium binary
via `AUTOUSER_BROWSER_EXECUTABLE`.

## Tests and linting

```bash
pytest -q                      # full suite (browser tests launch real Chromium)
ruff check src tests scripts   # lint — 100-char lines, target py311
```

The suite runs without any API keys: provider-backed and live tests are
**opt-in** and skip by default. They only run when you set their env flags:

```bash
AUTOUSER_CLAUDE_CODE_SMOKE=1 pytest tests/test_claude_code_provider_smoke.py
AUTOUSER_PARITY=1 pytest tests/test_cross_provider_parity.py   # also needs GEMINI_API_KEY
```

CI (GitHub Actions) runs `ruff` and the full `pytest` suite on Python 3.11–3.13
for every pull request. Please make sure both pass locally first.

## Conventions

These keep the codebase consistent — please follow them in PRs:

- **Pydantic v2 models everywhere**; prefer small, typed modules.
- **Comments explain _why_, not _what_.** Capturing the reasoning behind a
  decision (including review feedback) is part of this repo's style.
- **The runner owns control flow.** Give-up enforcement, predicate gates, and
  dwell-loop detection live in `runner.py`, not in the cognitive engine — the
  engine is a pure reasoning layer. Read the exit-order logic before changing
  termination behavior.
- **Give-up is objective.** It runs on trace events and simulated time
  (`affect.py`), never on emotion labels — emotions are log color only.
- **Failed browser actions must never crash a run.** They become observations
  (`UIState.action_error`) that the persona reacts to in character.
- **Prompt-cache contract:** the plan system prompt must stay byte-stable per
  session; per-step state goes in the user prompt only. `tests/test_prompts.py`
  pins this.
- **Schema parity:** the schema constants in `cognitive/engine.py` must stay
  enum-equal with the domain enums, and `prompts.py` response-format prose must
  match them. `tests/test_cross_provider_parity.py` hard-fails on drift.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline and key
contracts.

## Commit and PR guidelines

- Use [Conventional Commits](https://www.conventionalcommits.org/) for messages,
  matching the existing history — e.g. `feat(runner): ...`, `fix(execution): ...`,
  `docs: ...`.
- Keep PRs focused; one logical change per PR is easier to review.
- Make sure `ruff check src tests scripts` and `pytest -q` pass before opening
  the PR, and add or update tests for behavior changes.
- By contributing, you agree that your contributions are licensed under the
  project's [MIT License](LICENSE).

## Reporting bugs and proposing features

Open a [GitHub issue](https://github.com/renjie-liu/autouser/issues) using the
bug-report or feature-request template. For security issues, **do not** open a
public issue — see [SECURITY.md](SECURITY.md).
