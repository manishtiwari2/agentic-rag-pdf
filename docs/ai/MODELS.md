# Which AI built what

This file tracks the AI coding assistants that wrote code. The models the app
runs (bge-m3, the reranker, Qwen3-4B) are in `docs/MODEL_SELECTION.md`, not here.

Add a row per merged PR. If no AI was used, write "none". A human
(Manish or Anuradha) reviews every change.

| Date | PR / task | Subsystem | AI tool and model | Human reviewer |
|---|---|---|---|---|
| (next) | AI workflow docs | `CLAUDE.md`, `docs/ai/` | Claude Code, Opus 5.5 | Manish |
| 2026-11 | #52 | Docs fixes | Claude Code, Opus 5.5 | Manish |

Work before November 2026 was not tracked per PR. To find it, see the PR
history on GitHub.

## Model routing (from the global rules)
- Default: Sonnet.
- Opus: architecture, complex planning, or bugs Sonnet failed to fix.
- Haiku: wide read-only searches (Explore agents).
