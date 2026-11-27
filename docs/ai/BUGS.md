# Bugs, AI mistakes and regressions

Add new rows at the top. Status: open, fixed (with commit or PR), or won't fix.
Type: bug, AI mistake (hallucinated or wrong AI output), or regression.

| # | Date | Type | What happened | Where | Status |
|---|---|---|---|---|---|
| 13 | 2026-11-27 | bug | The dev scorer credited a table excerpt about S1 through S7 as the answer about S17. The excerpt omitted S17 and the correct six-year value. | DD-081, dev question q020 | Open. Check row grounding when scoring table answers. |
| 12 | 2026-11-26 | AI mistake | An AI review claimed a "shared reranker sigmoid bug" in all three repos. No sigmoid exists in our `src/reranking`. | `source/three-repo-comparison.md` | Closed: claim was false |
| 11 | 2026-11-26 | AI mistake | A commit and PR body got `Co-Authored-By` and "Generated with Claude Code" lines despite the global rule. | PR #52 | Rule now in `CLAUDE.md` |
| 10 | 2026-11-26 | bug | `src/config.py:4` names `tests/test_config.py`, which does not exist. | `src/config.py:4` | open |
| 9 | 2026-11-26 | bug | Refusal accuracy 0.000 in the hybrid stand-in run. | `src/config.py` | Open. DD-080 adds a floor for weak scores. Dev refusal accuracy was unchanged. |
| 8 | 2026-11-26 | regression | The always-on agent loop lowered faithfulness by -0.079 (CI -0.145 to -0.026). | `src/pipeline.py` agentic | open (ROADMAP P5) |
| 7 | 2026-11-26 | bug | Models reload on every Index click; the index is memory-only. | `src/chat/ui.py:98-99` | open (P2) |
| 6 | 2026-11-26 | bug | `DOCUMENT_TEXT_IS_DATA` is missing from the conflict and chat-rewrite prompts. | `src/agents/conflicts.py:308`, `src/chat/session.py:101` | open (P1) |
| 5 | 2026-11-26 | bug | No upload size, page or magic-byte check; OCR renders every image page. | `src/chat/ui.py:268`, `src/ingestion/parser_base.py:111-159` | open (P1) |
| 4 | 2026-11-26 | bug | No timeout on model calls and no wall-clock budget (max latency 140.8 s in the smoke run). | `src/generation/backends.py:276`, `src/pipeline.py:662` | open (P1) |
| 3 | 2026-11-26 | bug | Silent `except` returns empty text or tables. | `src/ingestion/pdfplumber_parser.py:142,189,197` | open (P1) |
| 2 | 2026-11-26 | bug | No logging anywhere in `src/`. | `src/` | open (P1) |
| 1 | 2026-11-26 | bug | Verifier fails open: an exception becomes "unavailable" and the answer is still served. | `src/pipeline.py:810-811, 942-946` | open (P1) |

P1 branch update (`fix/p1-rag-safety`, pending review): issues 1–6 and 10 have
local fixes. DD-076 records the defaults. Close the rows with the merge hash
after review.

Known flake: issue #46 (a test fails only on Python 3.10).
