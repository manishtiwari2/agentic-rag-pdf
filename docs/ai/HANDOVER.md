# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-27
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Branch:** `feat/p5-agent-quality`, based on PR #61 merge `f1d9716`.
- **Local commits:** `e4fc5c6` adds the reranker floor, `c966c44` adds missing-term hints and `1f1eeab` adds opt-in adaptive retrieval.
- **P5.4:** DD-083 adds decimal arithmetic over validated table-cell coordinates in agentic and adaptive. Model-backed selection asks for coordinates only.
- **Key files:** `src/generation/table_calculator.py`, `src/generation/generator.py`, `src/pipeline.py`, `tests/test_table_calculator.py`.
- **Targeted:** `python -m pytest tests/test_table_calculator.py -o addopts="" -q -rs --tb=short` returned 21 passed on Python 3.10 and 3.13. The CLI refused a missing row.
- **Full:** After `docker info`, set the two test service URLs and run `python -m pytest -o addopts="" -q -x -rs --tb=short`. The run with PostgreSQL and Redis returned 852 passed and one OCR skip because Tesseract was absent.
- **Other checks:** `python -m src.cli validate-benchmark` returned `Benchmark OK.`. Adaptive calculated 1.54 via the authenticated API and cited page 2 using an existing agentic index.
- **Next:** P5.5 writes the decision rule before comparing adaptive and agentic with paired bootstrap on the dev split.
