# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p2-persistent-index`, based on P1 merge `fe36bc5`.
- **Work:** Postgres schema and migration, pgvector adapter, owner-scoped version storage, persistent indexing and duplicate upload protection.
- **Decision:** DD-077 in `docs/DESIGN_DECISIONS.md` covers the cache, isolation and retrieval.
- **Key files:** `src/storage/`, `migrations/`, `src/pipeline.py`, `tests/test_postgres_library.py`.
- **Checks:** run `docker info`, set `RAG_TEST_DATABASE_URL` to a clean Postgres+pgvector database, then run `python -m pytest tests/test_postgres_library.py -o addopts="" -q`.
- **Results:** 4 Postgres tests passed. The full suite had 820 passed, 1 skipped (Tesseract absent), and 3 notebook warnings. Benchmark validation passed.
- **Next:** review P2 tests and diff before committing. Get separate approval for each GitHub write.
