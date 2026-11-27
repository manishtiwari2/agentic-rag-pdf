# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p3-api-auth`, based on P2 merge `6d0af06`.
- **Work:** FastAPI, cookie sessions and CSRF, document upload jobs, SSE answers and chat history, limits, health probes, and shared inference client.
- **Decision:** DD-078 covers sessions, durable jobs, and verified SSE responses.
- **Key files:** `src/api/`, `src/storage/library.py`, `migrations/versions/0002_ingestion_uploads.py`, `tests/test_api.py`.
- **Checks:** run `docker info`, set `RAG_TEST_DATABASE_URL` and `RAG_TEST_REDIS_URL`, then run `python -m pytest tests/test_api.py tests/test_postgres_library.py -o addopts="" -q --tb=short`.
- **Results:** 11 targeted passed. Full suite 827 passed, 1 skipped (Tesseract absent), 3 notebook warnings. Benchmark valid. Linux memory limit verified.
- **Next:** review the P3 PR. After merge, start P4 against the documented API in DD-078.
