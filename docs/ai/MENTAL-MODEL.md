# Mental model: the whole system in plain English

## What it is
You give it PDFs and ask a question. It finds the right passages, writes a
short answer, and shows the page each fact came from. If it cannot find
support, it says so instead of guessing.

## The five ideas that matter
1. **Search twice, then sort.** It searches by meaning (bge-m3 embeddings) and
   by exact words (BM25). It merges the two lists with RRF: each list gives
   `1/(60 + rank)` points, and the points are added. Then a reranker reads each
   (question, passage) pair and sorts them properly.
2. **The model never sees page numbers.** Passages are labelled `[C1]`, `[C2]`...
   The model cites the labels. Code turns each label into the real page. So the
   model cannot invent a page.
3. **Refuse when unsure.** No evidence means no model call and a fixed refusal
   sentence. A failed verification also refuses.
4. **Newer wins, unknown refuses.** If two versions of a document disagree, it
   uses the newer one. If it cannot tell which is newer, it refuses.
5. **Measure honestly.** Tune on dev, test on eval, write the rule before the
   run, and report a confidence interval (paired bootstrap).

## The three strategies
- **dense:** embeddings only. Fast, the default.
- **hybrid:** embeddings + BM25 + reranker.
- **agentic:** hybrid plus a planner, an evidence check, query refinement, and
  a verifier. On the stand-in models it lowered faithfulness (-0.079). That is
  why an "adaptive" mode is on the roadmap: run the loop only when the first
  answer fails.

## Where things live
| You want to change | Look in |
|---|---|
| A setting | `src/config.py` |
| Prompts | `src/generation/prompts.py` |
| Citations | `src/generation/citations.py` |
| Search | `src/retrieval/` |
| The agent steps | `src/agents/` and `src/pipeline.py` |
| PDF reading and OCR | `src/ingestion/` |
| The chat UI | `src/chat/` |
| Benchmarks | `src/evaluation/`, `src/benchmark/`, `benchmark/` |

## Known weak spots (details in BUGS.md)
The verifier lets answers through when it crashes. There are no logs, no
timeouts, no upload limits, no login, and no Docker. Models reload per session.
