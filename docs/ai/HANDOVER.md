# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p5-agent-quality`, based on PR #61 merge `f1d9716`.
- **P5 complete:** Reranker floor, missing-term hints, opt-in adaptive retrieval and cited table arithmetic. DD-080 through DD-084 record the decisions.
- **Benchmark:** Rule committed as `dc628a3` before the 33-question offline dev runs. Adaptive missed the safety and call-count gates. Agentic stays the default. DD-084 has the paired intervals and commands.
- **Known issue:** Adaptive's first-pass verifier accepted a current-employer excerpt for a previous-employer question (bug #14).
- **Targeted:** `python -m pytest tests/test_statistics.py tests/test_agentic.py::TestAdaptive tests/test_table_calculator.py -o addopts="" -q -rs --tb=short` returned 41 passed.
- **Full:** At implementation commit `4b46118`, `python -m pytest -o addopts="" -q -x -rs --tb=short` with PostgreSQL and Redis returned 852 passed and one OCR skip because Tesseract was absent. P5.5 changes documentation.
- **Next:** After approval, run `git log origin/main..HEAD` and `git diff origin/main --stat`, push `feat/p5-agent-quality`, and open one P5 PR. P6 follows P5.
