# AutoUser

Simulate realistic user behavior for web application testing and usability evaluation.

AutoUser creates AI-driven simulated users — each with distinct personas, cognitive patterns, and accessibility profiles — that navigate your web application and produce structured friction logs identifying real usability issues.

## Why AutoUser?

Traditional testing checks if features *work*. AutoUser checks if features work *for different kinds of people*. A power user and a first-time visitor experience the same interface differently. AutoUser simulates that difference by constraining an LLM's decision-making to match real behavioral profiles.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  SimulationRunner                    │
│    orchestrates the plan → execute → reflect loop    │
├──────────────┬─────────────────┬────────────────────┤
│  Persona     │  Cognitive      │  Execution         │
│  Engine      │  Model          │  Layer             │
│              │                 │                    │
│  archetypes  │  Phase 1: Plan  │  Playwright        │
│  parameters  │  (intent before │  browser           │
│  profiles    │   action)       │  screenshots       │
│              │  Phase 2: Reflect│  DOM capture       │
│              │  (mismatch      │                    │
│              │   detection)    │                    │
├──────────────┴─────────────────┴────────────────────┤
│                 Report Generator                     │
│    friction log · issues · overall impressions       │
└─────────────────────────────────────────────────────┘
```

The simulation loop runs per step:
1. **Capture** — browser state (DOM summary + screenshot + visible text)
2. **Plan** — cognitive model decides next action based on persona constraints
3. **Execute** — Playwright performs the action in a real browser
4. **Reflect** — cognitive model compares expected vs actual outcome, flags mismatches
5. **Repeat** — until task completes, patience exhausts, or step limit reached
6. **Report** — generate a structured friction log with ranked issues

## Setup

```bash
pip install -e ".[dev]"
playwright install chromium
```

If `playwright install chromium` cannot download (offline / restricted egress) but a
managed Chromium of a *different* revision is already cached, either pin `playwright`
to the version matching the cached revision, or point AutoUser at the cached binary:

```bash
export AUTOUSER_BROWSER_EXECUTABLE=/path/to/cached/chrome
```

## LLM providers

The cognitive engine supports four providers. Auto-detection order (CLI):
explicit `--provider` flag → `AUTOUSER_PROVIDER` env → `ANTHROPIC_API_KEY` →
`GEMINI_API_KEY` → local `claude` CLI on PATH. (`codex` is opt-in only,
never auto-selected, mirroring `claude_code`.)

| Provider | Needs | Notes |
|----------|-------|-------|
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-sonnet-4-6`; prompt caching enabled |
| `gemini` | `GEMINI_API_KEY` | Free-tier rate-limit pacing built in |
| `claude_code` | `claude` CLI installed + authenticated | **No API key needed** — uses your Claude subscription via the local CLI. One subprocess per plan/reflect call; concurrent spawns capped by `AUTOUSER_CLAUDE_CODE_MAX_SPAWNS` (default 1), per-call timeout by `AUTOUSER_CLAUDE_CODE_TIMEOUT` (default 120s) |
| `codex` | `codex` CLI installed + authenticated | **No API key needed** — uses your OpenAI/Codex subscription via the local `codex` CLI (`codex exec`, read-only sandbox). Opt-in via `--provider codex`; model defers to your codex config. Concurrency capped by `AUTOUSER_CODEX_MAX_SPAWNS` (default 1), per-call timeout `AUTOUSER_CODEX_TIMEOUT` (default 120s) |

When using the library directly (not the CLI), `claude_code` is opt-in via
`AUTOUSER_PROVIDER=claude_code` or `CognitiveEngine(..., provider="claude_code")` —
PATH presence alone never silently switches an API-driven run to subprocess-driven.

## Quick start

### CLI

```bash
# List persona archetypes
autouser personas

# Run one persona against a task; per persona this writes:
#   report.md          triage-ready friction report
#   journey.md         experience trajectory — per-step thoughts, expectations,
#                      emotions, notes, and the persona's final mental model
#   friction_log.json  full machine-readable trajectory
#   step_NNN.png       screenshots
autouser run \
  --url https://example.com/signup \
  --task "Create a new account" \
  --criteria "You reach the dashboard after signup" \
  --persona maria --persona jake \
  --success-url-pattern "/dashboard" \
  --max-steps 15 \
  --out ./autouser_runs
```

`--success-text` / `--success-url-pattern` / `--success-selector` configure an
objective success predicate. When set, the persona's *belief* it finished cannot
mark the run successful — divergence between belief and the predicate is itself
reported as a `false_success` finding. Exit code 0 means every persona produced a
friction log (an abandoned task is a finding, not a tool failure); 1 means a
simulation crashed; 2 means a configuration error.

Add `--fast` (dual-process mode) to skip the deliberate LLM reflect on cruising
steps — those with no surprise and no plausible completion — roughly halving LLM
calls on smooth runs. It escalates to full reflection on any mismatch, dead
action, low-confidence decision, or possible completion, so findings and success
detection are preserved. Best paired with a success predicate.

### Exploration: first-session impressions of an unknown product

```bash
# No task — the persona pokes around and reports what they think this is
autouser explore --url https://example.com/ --persona maria --persona jake
```

`explore` gives the persona no goal: they follow their own curiosity (steered
toward what they haven't tried yet by their accumulated notes and places), and
the session ends when they feel they've *seen enough* to say what the product
is and whether they'd use it — or when frustration/time runs out. The
`journey.md` ends with their mental model of your product and a coverage line.
Leaving satisfied vs leaving frustrated is the outcome that matters.

### Trust: benchmarks and the false-positive audit

Can you believe what AutoUser reports? Two answers:

```bash
# Deterministic precision/recall against a planted-issue benchmark (no LLM)
autouser score --benchmark benchmarks/friction_page.yaml --study <study.json>
autouser score --benchmark benchmarks/friction_page.yaml --log <friction_log.json> \
  --min-precision 0.7 --min-recall 0.5      # exit nonzero if unmet (CI gate)

# Have an unconstrained reviewer judge each finding real vs simulation-artifact
autouser run --url ... --task ... --criteria ... --audit
```

A benchmark YAML declares the issues a fixture page is *known* to contain;
scoring reports precision (of what was flagged, how much matches ground truth)
and recall (of the known issues, how many were found — measured across the
*union* of runs, the M5 gate's shape). The `--audit` pass runs a second,
persona-free reviewer over every finding and classifies it real / artifact /
uncertain, writing `audit.json` — a direct defense of precision. See
`benchmarks/friction_page.yaml` for the format.

### Studies: evidence over anecdote

```bash
# M personas × N sampled seeds; findings aggregated by reproduction frequency
autouser study \
  --url https://example.com/signup \
  --task "Create a new account" \
  --criteria "You reach the dashboard after signup" \
  --persona maria --persona carlos --seeds 5 \
  --success-url-pattern "/dashboard"
```

`study` samples persona parameters around each archetype (seed 0 is always the
unjittered baseline; accessibility traits are never jittered) and writes, on top
of the per-run artifacts, a `study.md`/`study.json` synthesis: issues clustered
by (category, URL) and ranked by how many runs reproduce them, with per-persona
counts and example quotes, plus per-archetype outcome/steps/confidence
distributions. With the `claude_code` provider, raise
`AUTOUSER_CLAUDE_CODE_MAX_SPAWNS` to parallelize study cells.

### Single persona (Python API)

```python
import asyncio
from autouser.persona.models import Persona
from autouser.persona.registry import get_archetype
from autouser.session import TaskSpec
from autouser.runner import SimulationRunner

# Pick a persona archetype
maria = Persona.from_archetype(get_archetype("maria"))

# Define what to test
task = TaskSpec(
    task="Create a new account",
    success_criteria="User reaches the dashboard after signup",
    start_url="https://example.com/signup",
    max_steps=20,
)

# Run the simulation
runner = SimulationRunner(persona=maria, task_spec=task)
friction_log = asyncio.run(runner.run())

# Inspect results
print(f"Outcome: {friction_log.outcome.value}")
for issue in friction_log.issues:
    print(f"  [{issue.severity.value}] {issue.category.value}: {issue.description}")
```

### Batch run (multiple personas)

```python
from autouser.runner import BatchRunner
from autouser.persona.registry import ARCHETYPES

# Run all 5 archetypes concurrently
personas = [Persona.from_archetype(a) for a in ARCHETYPES.values()]
batch = BatchRunner(personas=personas, task_spec=task)
result = asyncio.run(batch.run())

# Partial failures don't discard successful runs
print(f"{len(result.logs)} succeeded, {len(result.errors)} failed")
```

## V1 Persona Archetypes

| Name | Profile | Tests for |
|------|---------|-----------|
| **Maria** | 58, retired teacher, low tech literacy | Bad labeling, unclear information architecture |
| **Jake** | 26, developer, power user | Missing keyboard shortcuts, slow flows |
| **Priya** | 34, screen reader user | Accessibility barriers blocking task completion |
| **Carlos** | 40, multitasking parent, impatient | Cognitive load failures, distraction-prone errors |
| **Yuki** | 22, non-native English speaker | Jargon, idiom-dependent labels, cultural assumptions |

Each persona is defined by 5 behavioral dimensions: **tech literacy**, **patience/error tolerance**, **domain familiarity**, **cognitive load tolerance**, and **accessibility profile**. See [Persona Framework](docs/PERSONA_FRAMEWORK.md) for detailed definitions.

To build a custom persona directly from these five dimensions (instead of an archetype), use `Persona.from_config()`:

```python
from autouser.persona.models import Persona, Level

persona = Persona.from_config(
    tech_literacy=Level.LOW,
    patience=3,
    domain_familiarity=Level.LOW,
    cognitive_load_tolerance=Level.LOW,
)
```

**Recommendation:** Run at minimum 3 dimensionally diverse archetypes per task flow. Maria + Carlos + Priya covers tech literacy, patience, and accessibility. See the [coverage matrix](docs/PERSONA_FRAMEWORK.md) for guidance on which personas surface which issue categories.

## Output: Friction Logs

Each simulation produces a `FrictionLog` containing:
- **Task outcome** — `completed`, `completed_with_errors`, or `abandoned`
- **Step-by-step trace** — every action, observation, and reflection
- **Ranked issues** — severity (HIGH/MEDIUM/LOW), category (navigation, labeling, accessibility, etc.), and the persona's perspective on why it's a problem
- **Screenshots** — per-step captures stored in `screenshots/<run_id>/<persona>/`

See [Friction Log Spec](docs/FRICTION_LOG_SPEC.md) for field definitions and triage guidance.

## Run tests

```bash
pytest
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — three-layer pipeline, data flow, key contracts
- [Persona Framework](docs/PERSONA_FRAMEWORK.md) — dimension definitions, archetype coverage matrix
- [Friction Log Spec](docs/FRICTION_LOG_SPEC.md) — output format, severity scale, triage workflow
- [Vision](docs/VISION.md) — product vision, target users, differentiation
- [Milestones](docs/MILESTONES.md) — v1 scope, success criteria, phased roadmap

## License

MIT
