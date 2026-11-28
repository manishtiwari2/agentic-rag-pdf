# Approved changes log

One row per merged PR, newest at the top. Fill it in after the merge. Get the
hash with `git log -n 1 --oneline main`.

| Date | PR | Merge hash | What changed | Approved by |
|---|---|---|---|---|
| 2026-11-28 | #63 | 80a6cae | Reused ingestion process, batched embeddings, cached and streamed verified answers, and latency metrics | Manish |
| 2026-11-27 | #62 | 98ff9e9 | Evidence checks, opt-in adaptive retrieval, and cited table arithmetic | Manish |
| 2026-11-27 | #61 | f1d9716 | Authenticated web workspace and highlighted PDF citations | Manish |
| 2026-11-27 | #60 | 005b7a8 | Authenticated FastAPI, background ingestion, and SSE questions | Manish |
| 2026-11-27 | #59 | 6d0af06 | Postgres storage, pgvector search, and persistent PDF indexing | Manish |
| (next) | | | AI workflow docs (`CLAUDE.md`, `docs/ai/`) | Manish |
| before 2026-11-26 | #56 | c5b40aa | Basic prompt-injection guard | Manish |
| before 2026-11-26 | #55 | 83e7c1a | Contributor guide for issues #53, #54, #46 | Manish |
| before 2026-11-26 | #52 | 3444f20 | Fixed stale claims in README, TRADEOFFS and STATUS | Manish |

Earlier history: `git log --merges -n 20 --oneline`.
