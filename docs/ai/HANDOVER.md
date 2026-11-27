# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p6-performance`, based on PR #62 merge `98ff9e9`.
- **P6.1:** DD-085 covers a reusable isolated child, per-job deadlines and memory limits, and configurable embedding batches written in one transaction.
- **Key files:** `src/api/worker.py`, `src/api/ingest_job.py`, `src/storage/library.py`.
- **Targeted:** After `docker info`, set the PostgreSQL and Redis test URLs and run `python -m pytest tests/test_api.py tests/test_postgres_library.py -o addopts="" -q -x -rs --tb=short`. 14 passed on Python 3.10 and 3.13.
- **Full:** With the same services, `python -m pytest -o addopts="" -q -x -rs --tb=short` on Python 3.13 completed with 853 passed and 1 skipped (Tesseract absent).
- **Other checks:** `python -m src.cli validate-benchmark` returned `Benchmark OK.`. Integration used the offline model stack.
- **Next:** P6.2 adds tenant-scoped answer caching with invalidation, streaming and latency metrics.
