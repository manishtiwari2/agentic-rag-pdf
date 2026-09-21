# Project Status

Last updated: 2026-09-21 (Phase 0 dataset human-verified)

A factual record of what exists, what it is verified to do, and what blocks the
next step. `EXPERIMENT_PLAN.md` says what order to build in; this says how far
along that order the project actually is.

---

## 1. At a glance

| | |
| --- | --- |
| Phases complete | 1 (ingestion + indexing), 2 (dense baseline) — code |
| Blocking phase | **0 (benchmark construction)** — tooling built, 76-question dataset human-verified against the rendered PDFs |
| Tests | 332 passing |
| Source | 5,296 lines in `src/`, 2,544 lines in `tests/` |
| Licence | Apache-2.0 (DD-035) |
| Merged | PR #1 and PR #2, both in `main` |
| **Benchmark numbers** | **none exist, and none are claimed** |

---

## 2. Phase status

Against the exit criteria in `EXPERIMENT_PLAN.md` section 2.

| Phase | Exit criterion | State |
| --- | --- | --- |
| **0 — Benchmark** | 5–10 licensed PDFs, ~75 questions, every evidence page verified against the rendered PDF, dev/eval split assigned | **Met.** `benchmark/documents/` holds 5 PDFs (56 pages total), recorded in `benchmark/documents_metadata.json` with real page counts and SHA-256 checksums; only `doc5.pdf` has a confirmed open licence (CC-BY-4.0, stated in its own text), the other four are `license: "unknown"` per the project owner's explicit sign-off. `questions.json` holds 76 questions (all 10 taxonomy categories, 33 dev / 43 eval); `validate-benchmark` passes cleanly. **Every question and evidence page has now been checked by a human against the rendered PDF** (`inspect-page`, page by page): three evidence-page errors found in the original AI-authored draft (q023, q041, q058 — each citing a page adjacent to, but not containing, part of the supporting text) were corrected; no answer text or unanswerable-question determination needed correction |
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
* **Benchmark tooling** (`src/benchmark/`) — `validate-benchmark` enforces every
  `BENCHMARK_SPEC.md` section 5.1 rule (unique ids, no placeholders, evidence
  pages checked against a real page count read through `PdfParser`, document
  files exist, `question_type`/`split` in the allowed sets) and reports every
  failure in one pass, not just the first; `inspect-page` renders a page to PNG
  and prints its extracted text so a reviewer can verify an evidence page
  without leaving the terminal; `manifest.py` checksum-verifies documents that
  cannot be committed to the repo.

```bash
python -m src.cli validate-benchmark   # exits non-zero on any dataset problem
python -m src.cli inspect-page --pdf benchmark/documents/document_01.pdf --page 4
```

Module layout is in `ARCHITECTURE.md` section 3. The one deviation:
`src/ingestion/parser_base.py` holds the backend-independent parsing machinery,
added when DD-033 made the PDF library swappable. `src/ingestion/render.py`
(page-to-PNG rendering for `inspect-page`) is a second, smaller deviation for
the same reason: PDF-library use stays confined to `src/ingestion/`.

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

## 9. Benchmark documents in place

| File | Pages | Category | Licence |
| --- | --- | --- | --- |
| `doc1.pdf` | 2 | exam_paper (GATE 2027, CS/IT) | unknown |
| `doc2.pdf` | 11 | research_paper (mobile devices in EFL learning) | unknown |
| `doc3.pdf` | 42 | instructional_manual (Git/review/AI-assisted coding) | unknown |
| `doc4.pdf` | 1 | personal_document (resume) | owned by author |
| `doc5.pdf` | 9 | research_paper (semiconductor evolution) | CC-BY-4.0, confirmed in text |

Recorded in `benchmark/documents_metadata.json` with a real page count and
SHA-256 read through `PdfParser` for each file (`Manifest.build_entry`), not a
hand-typed guess. Only `doc5.pdf`'s licence is independently confirmed from
its own text; the rest are marked `"unknown"` and were added on the project
owner's explicit authorization rather than a verified open licence per file.
That is a recorded trade-off, not an oversight — `NOTICE` should be updated
with the same information once real `source_url`/`license` values are filled
in.

## 10. Next step

**Phase 0 is done.** `questions.json` holds 76 questions across all 10
taxonomy categories against the 5 documents in section 9, passes
`validate-benchmark` mechanically, and every question's `answer` and
`evidence_pages` have been checked by a human against the rendered PDF with
`python -m src.cli inspect-page --pdf benchmark/documents/docN.pdf --page M`,
per `EXPERIMENT_PLAN.md` section 2. Three evidence-page errors were found and
fixed in the process (see section 2); no other corrections were needed.

The tooling that made that pass fast stays in place: `src/benchmark/validate.py`
enforces every rule in `BENCHMARK_SPEC.md` section 5.1 and refuses a run on
any failure (`require_valid_benchmark`), `src/benchmark/manifest.py`
checksum-verifies documents that cannot be committed, and
`src/benchmark/inspect.py` renders an evidence page to PNG with its extracted
text so a human can verify it without leaving the terminal.

**Next up per `EXPERIMENT_PLAN.md` section 1:** run Baseline A (Phase 2)
against this now-trusted dataset to populate `results/baseline/`, then build
the metrics harness (Phase 3) to score it.
