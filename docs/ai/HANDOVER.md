# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-26
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `fix/p1-rag-safety`, based on `origin/main` at `3616956`.
- **Work:** verifier policy, seven guarded system prompts, model and question deadlines, PDF and OCR limits, request-scoped JSON logs, corrected config test reference.
- **Decision:** DD-076 in `docs/DESIGN_DECISIONS.md` records the defaults and native-call timeout behavior.
- **Key files:** `src/pipeline.py`, `src/config.py`, `src/deadlines.py`, `src/request_log.py`, `src/ingestion/`, `tests/test_safety.py`.
- **Checks:** `python -m pytest -o addopts="" -q -rs` and `python -m src.cli validate-benchmark`.
- **Results:** 816 passed, one real OCR test skipped (Tesseract absent), three notebook warnings. Benchmark valid. Offline FinFET answer cited pages 2–3 of `doc5.pdf`.
- **Next:** review the local P1 diff and approve the commit. Get separate approval for each GitHub write.
