# Security Policy

## Supported versions

AutoUser is pre-1.0 and under active development. Security fixes are applied to
the `main` branch and ship in the next tagged release. Please reproduce against
the latest `main` before reporting.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report it privately through GitHub's built-in flow — **Security → Report a
vulnerability** on this repository
([advisories/new](https://github.com/renjie-liu/autouser/security/advisories/new)).
This keeps the report confidential until a fix is available.

We aim to acknowledge reports within a few days and will coordinate a fix and a
disclosure timeline with you.

## Scope and safe use

AutoUser drives a **real browser** (Playwright/Chromium) and, unless you use the
local `claude_code` provider, sends page content, DOM/accessibility snapshots,
and your task description to third-party LLM providers (Anthropic, Google). Keep
this in mind when running it:

- **Do not** put real credentials, secrets, or personal data in task/criteria
  text, and avoid pointing AutoUser at pages behind your own authenticated
  session — that content leaves your machine for the configured LLM provider.
- Run against test/staging environments or sites you are authorized to test.
- Treat generated friction logs and screenshots as potentially sensitive: they
  capture whatever the browser rendered.
- API keys are read from environment variables / a gitignored `.env`. Never
  commit real keys — use [`.env.example`](.env.example) as the template.
