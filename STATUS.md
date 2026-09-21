# Project Status

Last updated: 2026-09-21

A factual record of what exists, what it is verified to do, and what blocks the
next step. `EXPERIMENT_PLAN.md` says what order to build in; this says how far
along that order the project actually is.

---

## 1. At a glance

| | |
| --- | --- |
| Phases complete | 1 (ingestion + indexing), 2 (dense baseline) — code |
| Blocking phase | **0 (benchmark construction)** — not started |
| Tests | 292 passing |
| Source | 4,324 lines in `src/`, 2,059 lines in `tests/` |
| Licence | Apache-2.0 (DD-035) |
| Merged | PR #1 and PR #2, both in `main` |
| **Benchmark numbers** | **none exist, and none are claimed** |

---

## 2. Phase status

Against the exit criteria in `EXPERIMENT_PLAN.md` section 2.

| Phase | Exit criterion | State |
| --- | --- | --- |
| **0 — Benchmark** | 5–10 licensed PDFs, ~75 questions, every evidence page verified against the rendered PDF, dev/eval split assigned | **Not started.** `benchmark/documents/` does not exist; `questions.json` is a 10-question template with 19 `REPLACE_WITH` placeholders |
| **1 — Ingestion + indexing** | A PDF parses to page-attributed chunks; page numbers checked against the source; a scanned PDF errors clearly | **Met**, verified mechanically |
| **2 — Dense baseline** | Baseline A answers the benchmark end to end, produces `results/baseline/` | **Code complete, exit criterion blocked on Phase 0.** Answers an arbitrary PDF end to end; `results/baseline/` is empty because there is nothing to run against |
| **3 — Metrics harness** | Retrieval, answer, citation and abstention metrics; error taxonomy; per-question records | Not started, blocked on Phase 0 |
| **4 — Hybrid + reranking** | Baseline B vs A with a paired test (DD-028) | Not started |
| **5 — Agentic** | Planner, evidence controller, refinement, verifier; bounded loops; full trace | Not started |
| **6 — Ablations + write-up** | Seven ablations, comparison table, error analysis | Not started |

`EXPERIMENT_PLAN.md` section 1 argued the benchmark, not the retrieval code,
was the critical path. That is now literally true: the baseline runs and cites
pages correctly, and there is nothing to measure it on.

---

## 3. What runs today

```bash
pip install -r requirements.txt
python -m src.cli ask --pdf paper.pdf --question "What was measured?" --offline
```

```text
3 Hardware

Peak memory during generation reached 9.4 gigabytes with four-bit
weights, leaving sufficient headroom for the key-value cache. [C1]

Evidence:
  [C1] page 3 - 3 Hardware
```

* **Ingestion** — 1-based page numbers; ligature, soft-hyphen, line-break-hyphen
  and running-header normalization; distinct errors for scanned, encrypted and
  blank PDFs, each carrying a remedy.
* **Chunking** — two strategies behind one interface: structure-aware (DD-009)
  and a fixed-size baseline. Tables get their own chunk. Overlap never crosses a
  heading.
* **Retrieval** — dense, over FAISS with an exact numpy fallback.
* **Generation** — grounded answering with deterministic citation resolution,
  and abstention with a fixed canonical refusal sentence.
* **Fallbacks** — a hashing embedder and a scripted extractive backend, both
  real if weak, so the whole pipeline runs and is tested on CPU with no weights.

Module layout is in `ARCHITECTURE.md` section 3. The one deviation:
`src/ingestion/parser_base.py` holds the backend-independent parsing machinery,
added when DD-033 made the PDF library swappable.

---

## 4. Deliberately absent

Not missing — excluded, because DD-013 requires a baseline that lacks them
before their value can be measured:

lexical retrieval · rank fusion · reranking · query planner · evidence
controller · retrieval refinement · verifier · metrics harness

`notebooks/agentic_pdf_rag.ipynb` is still empty. DD-034 settles how it will
obtain `src/` (clone at a pinned ref), which was the blocking decision.

---

## 5. Invariants enforced by tests

Rules that exist only in prose drift. These are executable:

| Invariant | Where |
| --- | --- |
| Page numbers are 1-based | `Page.__post_init__` raises below 1 |
| Every chunk carries page attribution | `Chunk.__post_init__`; `page_number` is derived from `pages`, not supplied |
| Retrieval and generation never import a PDF library | `tests/test_architecture.py` |
| Ingestion never imports downstream layers or `transformers` | `tests/test_architecture.py` |
| No model id appears outside `config.py` (DD-017) | `tests/test_architecture.py` |
| A research-licensed model cannot be the default generator (DD-024) | `RAGConfig.validate()` raises |
| The model stack fits the T4 budget (MODEL_SELECTION 9.1) | `RAGConfig.validate()` raises |
| The prompt contains no page metadata (DD-031) | `tests/test_generation.py` |
| Out-of-range citation markers are dropped, never guessed | `tests/test_generation.py` |
| Both PDF backends produce identical output | ingestion tests parametrized over both |

---

## 6. Verification evidence

* **292 tests pass**, in about 10 seconds. No network, no GPU, no model weights.
* **Page attribution verified mechanically**: across both PDF backends and both
  chunking strategies, zero chunks contain text absent from the pages they
  claim.
* **Both backends agree**: identical block structure, font sizes, table
  detection and chunk counts on the demo corpus.
* **Scanned PDF**: errors with the page counts it saw and an `ocrmypdf` remedy,
  rather than indexing an empty document.

---

## 7. Decisions recorded this phase

| | |
| --- | --- |
| DD-030 | Chunk sizes in characters, not tokens — a token budget binds the chunker to one tokenizer |
| DD-031 | The generator is never shown a page number; code maps markers to pages |
| DD-032 | The fixed-size baseline records no section, so the DD-009 comparison stays honest |
| DD-033 | Default PDF backend is pdfplumber (MIT), not PyMuPDF (AGPL-3.0) |
| DD-034 | The notebook obtains `src/` by cloning at a pinned ref |
| DD-035 | The project is licensed Apache-2.0 |

All five open questions in `EXPERIMENT_PLAN.md` section 5 are now either
decided (4, 5) or correctly deferred pending a dev set (1, 2, 3).

---

## 8. Known limitations

* **The dependency-free fallbacks are weak, and it shows.** On the demo report,
  *"How long did indexing take?"* retrieves the Hardware section instead of
  Latency and answers from it confidently. That is the hashing embedder plus
  lexical extraction doing their best; the citation still correctly points at
  the page the text came from. Exactly the failure Phase 3 exists to count.
* **PyMuPDF is a test-only dependency** (AGPL-3.0). It writes the fixture PDFs,
  which pdfplumber cannot do, and serves as the second parser backend. It is in
  the `dev` extra, never in the runtime dependencies.
* **The copyright line reads `The agentic-pdf-rag authors`** in `LICENSE` and
  `NOTICE` — valid, and avoids asserting a legal name that has not been stated.
* **Chunk size, overlap, fusion weights and `MAX_RETRIEVAL_ITERATIONS` are
  unvalidated.** `EXPERIMENT_PLAN.md` section 5 records them as open pending a
  dev set. The retrieval score threshold is disabled by default for the same
  reason: an untuned threshold is worse than none.

---

## 9. Next step

**Phase 0.** Its hard part is manual and cannot be delegated: choosing
redistributable PDFs, writing ~75 questions, and opening each PDF to confirm
every evidence page by eye. `BENCHMARK_SPEC.md` section 4.1 lists the
categories that are safe to redistribute, and warns that "arXiv" is not a
licence.

The part that can be built first is the tooling that makes the manual work
checkable: a dataset validator enforcing every rule in `BENCHMARK_SPEC.md`
section 5.1, a document manifest with SHA-256 checksums for PDFs that cannot be
committed, and a helper that renders an evidence page so a human can verify it
without leaving the terminal.

A malformed benchmark produces plausible-looking numbers, which is the worst
failure mode an evaluation harness has. The validator should refuse to let a
run start.
