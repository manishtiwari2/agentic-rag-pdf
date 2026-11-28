# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-28
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p7-deployment`, based on PR #63 merge `80a6cae`.
- DD-088 covers the P7.2 Compose deployment with default Ollama, a vLLM profile, and an offline CPU override.
- Set the required variables from `ENV.template.md` and run `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build --wait`. Upload, citation, tenant isolation, and volume persistence across restart passed.
- `python -m pytest tests/test_compose_config.py -o addopts="" -q` returned 4 passed on Python 3.10 and 3.13. Ollama and vLLM configs passed. Docker on this host has no NVIDIA adapter.
- After `docker info`, use disposable PostgreSQL and Redis separate from the Compose app database. With `RAG_TEST_DOCKER_IMAGE` set to the built tag, `python -m pytest -o addopts="" -q -x -rs --tb=short` returned 861 passed and 1 skipped (host Tesseract absent).
- `python -m src.cli validate-benchmark` returned `Benchmark OK.`
- P7.3 pins Hugging Face revisions and dependencies and adds `.env.example` and PostgreSQL backup and restore.
