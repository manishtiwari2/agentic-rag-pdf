# Constraints: never touch without asking

If a task needs to break one of these, stop and ask Manish first. Say which rule,
why, and what the safer option is.

## Logic that must stay
| Rule | Where | Why |
|---|---|---|
| The LLM never sees page numbers. Pages come from chunk metadata. | `src/generation/prompts.py:3-7`, `src/generation/citations.py` | 0 invented pages in 228 answers. This is the project's core trust claim. |
| Out-of-range `[Cn]` markers are dropped, never guessed. | `src/generation/citations.py:130-133` | A wrong page is worse than no page. |
| No evidence means no LLM call, just the fixed refusal. | `src/generation/generator.py` ("no_evidence") | Stops answers made from nothing. |
| `DOCUMENT_TEXT_IS_DATA` in every prompt with document text. | `src/generation/prompts.py:37` | Prompt-injection guard. |
| Conflict policy: prefer newer; unknown date means refuse. | `src/agents/conflicts.py`, `src/versioning.py` | Never serve an outdated rule as current. |
| RRF sums use exact `Fraction`s. | `src/retrieval/fusion.py` | Same ranking on every run. |
| Licence guard and VRAM budget in config validation. | `src/config.py:600-688` (DD-024) | Keeps AGPL code out of runtime and fits a 16 GB T4. |
| Temperature 0. | `src/config.py` | Reproducible runs. |

## Files not to edit
- `benchmark/` question and gold files. They are the test set. Editing them
  invalidates every result. Only `validate-benchmark` may read them for checks.
- `results/` run outputs. They are records of past runs. Add new runs; never rewrite old ones.
- `notebooks/runs/*.ipynb`. They are executed records (the T4 smoke run).
- `LICENSE`, `NOTICE`.
- Old `DD-0xx` entries in `docs/DESIGN_DECISIONS.md`. Add a new entry instead of rewriting history.

## Dependencies
- No AGPL package in runtime code. PyMuPDF is test-only (`requirements-dev.txt`).
- No package outside `docs/ai/DEPENDENCIES.md` without asking.
- No `trust_remote_code=True` when loading models.

## Evaluation rules
- Tune only on the dev split. Never look at eval results while tuning.
- Write the decision rule before a comparison run. Report the CI (paired bootstrap).
- Never present stand-in (CPU) model numbers as real-model numbers.

## Git
- Never commit to `main` directly. Never force-push. Never `git add .` or `-A`.
- No `Co-Authored-By` or "Generated with" lines.
- No secrets, tokens or `.env` files in commits.
