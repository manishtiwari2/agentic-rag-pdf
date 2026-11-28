# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-28
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p6-performance`, based on PR #62 merge `98ff9e9`.
- **P6.2:** DD-086 covers cached verified answers, invalidation with migration `0004_library_revision`, SSE text after verification, evidence limits, and `/metrics` latency and first-token times.
- **Key files:** `src/api/app.py`, `src/api/answer_cache.py`, `src/api/llm.py`, `src/api/metrics.py`, `src/storage/library.py`.
- **Targeted:** After `docker info`, set the PostgreSQL and Redis test URLs and run `python -m pytest tests/test_api.py tests/test_postgres_library.py tests/test_generation.py -o addopts="" -q -x -rs --tb=short`. 78 passed on Python 3.10 and 3.13.
- **Full:** With the same services, `python -m pytest -o addopts="" -q -x -rs --tb=short` on Python 3.13 returned 856 passed and 1 skipped (Tesseract absent).
- **Other checks:** `python -m src.cli validate-benchmark` returned `Benchmark OK.`. Migration preserved an existing account. The chat script passed a simulated SSE rendering check.
- **Next:** Complete P6 review, then start P7.1 for the multi-stage Dockerfile.
