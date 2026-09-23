# Project Status

Last updated: 2026-09-21 (Phase 3 metrics harness; first real benchmark numbers)

A factual record of what exists, what it is verified to do, and what blocks the
next step. `EXPERIMENT_PLAN.md` says what order to build in; this says how far
along that order the project actually is.

---

## 1. At a glance

| | |
| --- | --- |
| Phases complete | 0 (benchmark), 1 (ingestion + indexing), 2 (dense baseline), 3 (metrics harness) |
| Next phase | **4 (hybrid retrieval + reranking)** — Baseline A measured, so there is something to measure Baseline B against |
| Tests | 404 passing |
| Source | 7,455 lines in `src/`, 3,643 lines in `tests/` |
| Licence | Apache-2.0 (DD-035) |
| Merged | PR #1–#5, all in `main` |
| **Benchmark numbers** | **They now exist, in `results/baseline/`, from a real 76-question run.** Recall@5 0.886, accuracy (all) 0.171, abstention accuracy 0.167, false-answer rate 0.833. Produced by the **dependency-free fallback stack** (hashing embedder + scripted extractive backend) on a machine with no GPU, not by the model stack in `MODEL_SELECTION.md`. They are real measurements of a real, weak system — not a stub, and not a claim about the intended configuration |

---

## 2. Phase status

Against the exit criteria in `EXPERIMENT_PLAN.md` section 2.

| Phase | Exit criterion | State |
| --- | --- | --- |
| **0 — Benchmark** | 5–10 licensed PDFs, ~75 questions, every evidence page verified against the rendered PDF, dev/eval split assigned | **Met.** `benchmark/documents/` holds 5 PDFs (56 pages total), recorded in `benchmark/documents_metadata.json` with real page counts and SHA-256 checksums; only `doc5.pdf` has a confirmed open licence (CC-BY-4.0, stated in its own text), the other four are `license: "unknown"` per the project owner's explicit sign-off. `questions.json` holds 76 questions (all 10 taxonomy categories, 33 dev / 43 eval); `validate-benchmark` passes cleanly. **Every question and evidence page has now been checked by a human against the rendered PDF** (`inspect-page`, page by page): three evidence-page errors found in the original AI-authored draft (q023, q041, q058 — each citing a page adjacent to, but not containing, part of the supporting text) were corrected; no answer text or unanswerable-question determination needed correction |
| **1 — Ingestion + indexing** | A PDF parses to page-attributed chunks; page numbers checked against the source; a scanned PDF errors clearly | **Met**, verified mechanically |
| **2 — Dense baseline** | Baseline A answers the benchmark end to end, produces `results/baseline/` | **Met.** All 76 questions answered end to end against the 5 benchmark PDFs; `results/baseline/` holds `config.json`, `results.json`, `per_question.json` and `summary.csv`. The exit criterion says "Numbers may be poor. They must exist." They are poor and they exist |
| **3 — Metrics harness** | Retrieval, answer, citation and abstention metrics; error taxonomy; per-question records | **Met.** `src/evaluation/metrics.py` computes Recall@1/3/5/10, Full-Recall@5/10, MRR@10 and nDCG@10 at page level (DD-025), with `unanswerable` questions excluded rather than scored 0.0; deterministic correctness, faithfulness, citation precision/completeness; the four abstention rates over their three distinct denominators. The nine-category error taxonomy (DD-038) assigns exactly one category to every failing record, 65 of 76 here. `src/evaluation/benchmark.py` drives any system through its public `ask`, checkpoints per question, and writes the four files `EVALUATION_PROTOCOL.md` section 27 names. The LLM judge is built, optional and off by default (DD-041) |
| **4 — Hybrid + reranking** | Baseline B vs A with a paired test (DD-028) | Not started. Unblocked: the harness is system-agnostic and the baseline it must beat is measured |
| **5 — Agentic** | Planner, evidence controller, refinement, verifier; bounded loops; full trace | Not started |
| **6 — Ablations + write-up** | Seven ablations, comparison table, error analysis | Not started |

`EXPERIMENT_PLAN.md` section 1 argued the benchmark, not the retrieval code,
was the critical path. It was: the dataset took the longest and everything
downstream was unverifiable until it existed. Both branches below it — the
baseline and the metrics harness — are now closed, and the first measured
result is in section 9.1.

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

* **Evaluation** (`src/evaluation/`) — `metrics.py` scores retrieval at page
  level, answers deterministically, citations, and abstention on all four cells
  of the outcome matrix, then assigns exactly one of nine error categories to
  every failing record; `benchmark.py` runs any system satisfying
  `QueryableSystem` over the dataset, checkpoints after each question and writes
  the four files `EVALUATION_PROTOCOL.md` section 27 names; `judge.py` is the
  optional LLM judge, off by default and unable to block a run (DD-041).

```bash
python -m src.cli validate-benchmark   # exits non-zero on any dataset problem
python -m src.cli inspect-page --pdf benchmark/documents/doc1.pdf --page 4
python -m src.cli run-benchmark --pdf-dir benchmark/documents --out results/baseline --offline
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
controller · retrieval refinement · verifier

The metrics harness has left this list: it is Phase 3's deliverable and is
built. A test in `tests/test_architecture.py` asserts that `src/reranking/` and
`src/agents/` still do not exist, so "deliberately absent" is now enforced
rather than asserted. The taxonomy reserves a `reranking` and a `verification`
category for them, each firing only for a system that declares the stage
(DD-038), so Phases 4 and 5 reuse the harness without it having guessed at
their shape.

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
| Unanswerable questions are excluded from retrieval metrics, not scored 0.0 | `tests/test_evaluation.py`, asserted on the record |
| `over_abstention_rate`'s denominator is the answerable set | `tests/test_evaluation.py` builds a refuse-everything system and requires it to score badly |
| Every failing record gets exactly one error category | `tests/test_evaluation.py`, one test per category plus a population test |
| Benchmark instrumentation does not change answers (ARCHITECTURE 26) | `tests/test_evaluation.py` compares a harness-run `RAGResult` with a bare `ask` |
| Nothing below `src/evaluation/` imports it | `tests/test_architecture.py` |
| `src/reranking/` and `src/agents/` do not exist (DD-013) | `tests/test_architecture.py` |

---

## 6. Verification evidence

* **404 tests pass**, in about 60 seconds. No network, no GPU, no model weights.
  The 76-question benchmark run is one of them, so the harness is exercised
  against the real dataset on every commit rather than only by hand.
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
| DD-036 | Deterministic correctness is token overlap (>= 0.6) plus exact numeric agreement |
| DD-037 | Faithfulness is two numbers; the numeric one is null when the answer has no numbers |
| DD-038 | The error taxonomy is an ordered, versioned, first-match rule set with two reserved categories |
| DD-039 | Retrieval is scored at depth 10 by a read-only probe; answers still come from top_k |
| DD-040 | nDCG is reported under binary relevance and labelled as such |
| DD-041 | The judge is optional, off by default, and a skip is never a zero |
| DD-042 | Citation completeness is sentence-level marker coverage |
| DD-043 | Abstention is scored from the answer text, with the self-report recorded beside it |

DD-036 to DD-043 are Phase 3's. All five open questions in
`EXPERIMENT_PLAN.md` section 5 are either decided (4, 5) or correctly deferred
(1, 2, 3) — and 1, 2 and 3 are now answerable for the first time, because the
dev set has a harness to be swept with.

---

## 8. Known limitations

* **The dependency-free fallbacks are weak, and it shows.** On the demo report,
  *"How long did indexing take?"* retrieves the Hardware section instead of
  Latency and answers from it confidently. That is the hashing embedder plus
  lexical extraction doing their best; the citation still correctly points at
  the page the text came from. Exactly the failure Phase 3 exists to count --
  and now counts: section 9.1 files 46 of 65 failures as `generation`, against a
  Recall@5 that clears its stopping rule.
* **PyMuPDF is a test-only dependency** (AGPL-3.0). It writes the fixture PDFs,
  which pdfplumber cannot do, and serves as the second parser backend. It is in
  the `dev` extra, never in the runtime dependencies.
* **The copyright line reads `The agentic-pdf-rag authors`** in `LICENSE` and
  `NOTICE` — valid, and avoids asserting a legal name that has not been stated.
* **The benchmark numbers are from the fallback stack, not the model stack.**
  Section 9.1 says so in its first line and section 10 says how to replace them.
  No figure there is a claim about `Qwen3-4B-Instruct-2507` plus `bge-m3`.
* **The LLM judge has never been run.** It is built, tested and off by default
  (DD-041); every test pins its behaviour without loading a model, which means
  its behaviour *with* one is untested. DD-027's self-preference disclosure
  applies the moment it is first used.
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

## 9.1 First measured result

`results/baseline/`, 76 questions, Baseline A, **the dependency-free fallback
stack** (hashing embedder + scripted extractive backend). There is no GPU on the
machine that produced it, so this is not the `MODEL_SELECTION.md` stack and no
figure here should be read as one. It is a real end-to-end measurement of a real
system, which is what the Phase 2 exit criterion asked for: *"Numbers may be
poor. They must exist."*

| | Baseline A (fallback stack) | n | Stopping rule (`EXPERIMENT_PLAN.md` 3) |
| --- | ---: | ---: | --- |
| Recall@1 / @3 / @5 / @10 | 0.629 / 0.814 / 0.886 / 0.957 | 70 | Recall@5 >= 0.80 — **met** |
| Full-Recall@5 / @10 | 0.771 / 0.900 | 70 | — |
| Full-Recall@5 (multi_hop + comparison) | 0.733 | 15 | — |
| MRR@10 | 0.736 | 70 | — |
| nDCG@10 (binary, DD-040) | 0.776 | 70 | — |
| accuracy (all) | 0.171 | 76 | — |
| accuracy (answerable) | 0.171 | 70 | — |
| abstention accuracy | 0.167 | 6 | >= 0.70 — **missed** |
| false-answer rate | 0.833 | 6 | <= 0.30 — **missed** |
| over-abstention rate | 0.057 | 70 | — |
| unsupported-answer rate | 0.000 | 76 | — |
| faithfulness (token) | 0.934 | 76 | — |
| citation any-correct | 0.643 | 70 | >= 0.75 — **missed** |
| citation precision | 0.641 | 70 | — |
| median / P95 latency | 0.004 s / 0.010 s | 76 | <= 30 s median — met, on a stack that loads no weights |

Read the three denominators separately, as `EVALUATION_PROTOCOL.md` section 21.1
requires: six unanswerable questions is a small set, and one more false answer
moves abstention accuracy by 17 points. The figure is not a precise value.

### What the error taxonomy says

65 of 76 questions failed, each with exactly one category:

```text
generation          46
abstention           9
retrieval            8
context-selection    1
citation             1
reranking            0   (reserved: DD-013, no reranker in this baseline)
verification         0   (reserved: DD-013, no verifier in this baseline)
hallucination        0
system-runtime       0
```

The diagnosis this is for: **retrieval is not the bottleneck.** Recall@5 clears
its stopping rule, so the evidence usually reaches the generator and the
generator usually fails to use it — which is exactly what a scripted extractive
backend that returns whole sentences does against short reference answers. The
`abstention` count is the second finding and the more serious one: 5 of 6
unanswerable questions were answered rather than refused, because the scripted
backend's match threshold is tuned to find *something*. `unsupported-answer
rate` is 0.000 because that backend copies its numbers verbatim out of the
evidence it cites; it cannot hallucinate a figure, only misread which one
matters.

None of this is a finding about the intended model stack. It is a finding about
the fallback, and it is the first time the project can say something measured
about any of its parts.

---

## 10. Next step

**Phases 0 to 3 are done.** The dataset is verified, the baseline runs against
it end to end, and `src/evaluation/` scores it: page-level retrieval metrics
with unanswerable questions excluded, deterministic answer and citation metrics,
the four abstention outcomes over their three denominators, a nine-category
error taxonomy that assigns exactly one category to every failing record, and
per-question records in the four files `EVALUATION_PROTOCOL.md` section 27
names.

Two things are worth doing before Phase 4, in this order:

1. **Re-run against the model stack.** Every number in section 9.1 comes from
   the fallback components. The same command produces the real figures on a
   machine with a GPU:

   ```bash
   python -m src.cli run-benchmark --out results/baseline
   ```

   Nothing in the harness changes; `--offline` is the only difference. Until
   that run happens, section 9.1 is a measurement of the fallback, and the
   README and write-up must keep saying so.

2. **Phase 4: hybrid retrieval and reranking.** Unblocked, and now meaningful:
   DD-013 wanted a baseline that lacks those components before their value could
   be measured, and there is one, measured. The harness is system-agnostic —
   `QueryableSystem` needs only `index` and `ask` — so Baseline B is scored by
   the same code, and DD-038 already reserves the `reranking` category for the
   component Phase 4 adds.

`EXPERIMENT_PLAN.md` section 5's open questions 1, 2 and 3 (chunk size and
overlap, RRF vs. weighted fusion, `MAX_RETRIEVAL_ITERATIONS`) are answerable for
the first time: they were deferred pending a dev set, and the dev split now has
a harness to be swept with. Sweep on `--split dev` only; the eval split exists
to not be tuned against.
