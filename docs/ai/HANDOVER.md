# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p5-agent-quality`, based on PR #61 merge `f1d9716`.
- **P5.1:** Gate generation on raw reranker scores and keep index fingerprints stable when the floor changes.
- **Decision:** DD-080 chose 0.01 on the offline dev split. Refusals and accuracy were unchanged on those questions. The cross-encoder floor is opt-in.
- **Key files:** `src/pipeline.py`, `src/config.py`, `src/api/settings.py`, `tests/test_agentic.py`.
- **Targeted:** after `docker info`, set the Postgres and Redis test URLs. Run `python -m pytest tests/test_agentic.py tests/test_api.py tests/test_postgres_library.py -k "reranker_floor or floor_changes or reuse or versions_change or regenerations_stop_at_the_cap" -o addopts="" -q -rs --tb=short`. Nine passed on Python 3.10 and 3.13.
- **Full:** `python -m pytest -o addopts="" -q -x -rs --tb=short` with Postgres and Redis returned 817 passed and one OCR skip (Tesseract absent).
- **Other checks:** `python -m src.cli validate-benchmark` passed. The offline CLI answered with real pages and refused a question with zero overlap.
- **Next:** P5.2 adds a hint when question terms are missing from the evidence. Keep P5 work on this branch until it is ready for review.
