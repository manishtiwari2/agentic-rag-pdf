# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p4-web-ui`, based on P3 merge `005b7a8`.
- **Work:** FastAPI pages, upload progress and status, SSE chat, safe message rendering, and a PDF viewer with owner-checked highlights. Gradio was removed.
- **Decision:** DD-079 covers server-rendered pages, safe streaming, and citation matching.
- **Key files:** `src/web/`, `src/ingestion/highlight.py`, `src/api/app.py`, `migrations/versions/0003_job_filename.py`.
- **Targeted:** after `docker info`, set the Postgres and Redis test URLs and run `python -m pytest tests/test_highlight.py tests/test_api.py tests/test_postgres_library.py -o addopts="" -q -rs --tb=short`. 13 passed on Python 3.10 and 3.13.
- **Full:** `python -m pytest -o addopts="" -q -rs --tb=short` on Python 3.13 gave 813 passed and one OCR skip (Tesseract absent).
- **Other checks:** `python -m src.cli validate-benchmark`, `node --check src/web/static/app.js`, the clean wheel, and the offline notebook against `v0.7-gpu-run` passed.
- **Browser:** Manish verified registration, PDF upload, chat and a highlighted citation in Chrome.
- **Next:** after approval, commit P4, run `git log origin/main..HEAD` and `git diff origin/main --stat`, then `git push -u origin feat/p4-web-ui` and open the PR. Start P5 after merge.
