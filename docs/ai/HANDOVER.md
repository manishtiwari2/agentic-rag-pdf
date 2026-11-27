# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p5-agent-quality`, based on PR #61 merge `f1d9716`.
- **Local commits:** `e4fc5c6` adds the reranker floor (DD-080). `c966c44` adds missing-term hints (DD-081).
- **P5.3:** DD-082 adds opt-in adaptive retrieval. It verifies a first-pass answer and falls back to the agentic loop when needed.
- **Key files:** `src/pipeline.py`, `src/config.py`, `src/api/settings.py`, `src/evaluation/benchmark.py`.
- **Targeted:** `python -m pytest tests/test_agentic.py::TestAdaptive tests/test_conflicts.py::TestPolicy::test_adaptive_preserves_the_conflict_refusal -o addopts="" -q` returned 8 passed on Python 3.10 and 3.13. The CLI exercised both paths.
- **Full:** `python -m pytest -o addopts="" -q -x -rs --tb=short` returned 815 passed and 3 skipped. Docker Desktop was down for the API and Postgres tests. Tesseract was absent for the OCR test.
- **Database checks:** Run `docker info`, set the two test service URLs, then run `python -m pytest tests/test_api.py tests/test_postgres_library.py -o addopts="" -q`.
- **Next:** P5.4 adds arithmetic grounded in cited table cells. Keep P5 tasks in separate commits on this branch.
