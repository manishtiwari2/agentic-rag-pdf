# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-28
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p7-deployment`, based on PR #63 merge `80a6cae`.
- DD-089 covers the locked Docker build, pinned model revisions, shared model cache, `.env.example`, and PostgreSQL backup and restore.
- `uv lock --check` and `docker build --check .` passed. `docker build -t agentic-pdf-rag:local .` installed CPU PyTorch from the lockfile.
- `python -m pytest tests/test_model_revisions.py tests/test_compose_config.py tests/test_postgres_backup.py -o addopts="" -q` returned 7 passed on Python 3.10 and 3.13.
- Rehearsed `python scripts/postgres_backup.py backup rag.dump` and `python scripts/postgres_backup.py restore rag.dump --database rag_restored` in a disposable CPU Compose project. Switching the stack preserved PDF bytes, vectors, tenant isolation, and citations.
- Full tests use disposable PostgreSQL and Redis after `docker info`. With `RAG_TEST_DOCKER_IMAGE` set to the built tag, `python -m pytest -o addopts="" -q -x -rs --tb=short` returned 864 passed and 1 skipped (host Tesseract absent).
- `python -m src.cli validate-benchmark` returned `Benchmark OK.` GPU configuration and model pin wiring were checked. Runtime smoke used offline models.
- Finish P7 review. P8.1 adds CI quality checks, a Docker build, and a benchmark smoke test.
