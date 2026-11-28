# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-28
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p7-deployment`, based on PR #63 merge `80a6cae`.
- **P7.1:** DD-087 covers a digest-pinned Python base, multi-stage build, a non-root runtime with English OCR, and a Docker `/health` liveness probe.
- **Build:** `docker build --check .` and `docker build -t agentic-pdf-rag:p7 .`.
- **Targeted:** Set `RAG_TEST_DOCKER_IMAGE` to the built tag and run `python -m pytest tests/test_docker_image.py -o addopts="" -q`. 1 passed.
- **End to end:** `alembic upgrade head` ran from the image. The containerized API and worker indexed a PDF and returned a page citation with offline models.
- **Full:** After `docker info`, point the test URLs at disposable PostgreSQL and Redis. Use a fresh database separate from the app smoke database. `python -m pytest -o addopts="" -q -x -rs --tb=short` returned 857 passed and 1 skipped (host Tesseract absent).
- **Other checks:** `python -m src.cli validate-benchmark` returned `Benchmark OK.`. The image shut down gracefully after serving a cited answer.
- **Next:** P7.2 adds Compose for the API, worker, PostgreSQL, Redis, Ollama and the optional vLLM profile and CPU override.
