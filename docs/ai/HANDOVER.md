# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p5-agent-quality`, based on PR #61 merge `f1d9716`.
- **P5.1:** DD-080 set the offline reranker floor to 0.01. The cross-encoder floor is opt-in.
- **P5.2:** DD-081 adds a hint for explicit numbers and likely names missing from the selected prompt evidence. Use `--no-missing-term-hint` for the reference run.
- **Dev:** One extra refusal. The scorer counted an excerpt from other rows as correct for question q020 (bug #13). DD-081 reports the paired intervals.
- **Key files:** `src/agents/text.py`, `src/generation/generator.py`, `src/generation/prompts.py`, `src/config.py`.
- **Targeted:** `python -m pytest tests/test_generation.py tests/test_agentic.py tests/test_architecture.py -k "not TestFullAgenticSmokeRun" -o addopts="" -q -rs --tb=short` returned 276 passed. A 20-test subset passed on Python 3.10.
- **Full:** After `docker info`, set the Postgres and Redis test URLs and run `python -m pytest -o addopts="" -q -x -rs --tb=short`. 820 passed and one OCR test skipped because Tesseract was absent.
- **Other checks:** `python -m src.cli validate-benchmark` passed. The offline CLI cited pages 2–3 for FinFET and refused the question with a missing year.
- **Next:** P5.3 adds an adaptive strategy that answers and verifies once before escalating. Keep P5 work on this branch until it is ready for review.
