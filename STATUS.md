# Project Status

Last updated: 2026-09-23 (Phase 5 agentic pipeline; measured against both baselines)

A factual record of what exists, what it is verified to do, and what blocks the
next step. `EXPERIMENT_PLAN.md` says what order to build in; this says how far
along that order the project actually is.

---

## 1. At a glance

| | |
| --- | --- |
| Phases complete | 0 (benchmark), 1 (ingestion + indexing), 2 (dense baseline), 3 (metrics harness), 4 (hybrid + reranking), 5 (agentic) |
| Next phase | **6 (ablations + write-up)** |
| **Phase 5 result** | **The agentic pipeline is not kept: its added complexity did not pay for itself.** No keep-or-drop metric improves over Baseline B. Faithfulness −0.132 [−0.211, −0.053] and citation any-correct −0.143 [−0.229, −0.071] *regress*, because the rule-based evidence controller refused 11 answerable questions. RQ3 not shown, and RQ4 unanswerable offline (floor), as DD-056 declared in advance. Rule-based stand-ins for every agent decision (section 9.3) |
| Tests | 508 passing |
| Source | 9,010 lines in `src/`, 4,652 lines in `tests/` |
| Licence | Apache-2.0 (DD-035) |
| Merged | PR #1–#7 in `main`; Phase 4 is on `feat/phase-4-hybrid-reranking` |
| **Phase 4 result** | **Baseline B vs A, paired bootstrap over 76 questions: no significant difference on either pre-declared primary metric.** Recall@5 +0.057, 95% CI [−0.014, +0.143]; accuracy (all) +0.026, CI [−0.039, +0.092]. Significant on ranking-quality secondaries: MRR@10 +0.091 [+0.019, +0.167], Full-Recall@5 +0.100 [+0.029, +0.186]. Fallback stack on both sides, including a term-overlap stand-in for the reranker (section 9.2) |
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
| **4 — Hybrid + reranking** | Baseline B vs A with a paired test (DD-028) | **Met.** `HybridRAGPipeline` (BM25 + dense, RRF, reranker) shares every other layer with Baseline A and is chosen by `retrieval.strategy`. `results/hybrid/` holds the four section 27 files from a real 76-question run, plus `comparison.json`: the paired bootstrap against `results/baseline/` (2,000 resamples, 19 metrics). The "Reranker OFF" arm is in `results/ablations/reranker_off/`, so the hybrid and reranker effects are measured separately. Findings are in section 9.2; neither primary metric differs significantly |
| **5 — Agentic** | Planner, evidence controller, refinement, verifier; bounded loops; full trace | **Met.** `AgenticRAGPipeline` (`--system agentic`) drives Baseline B's retrieval through a bounded loop of the four components in `src/agents/`, each switchable by one config field. Every per-question record carries the full `agent_trace` with its caps. `results/agentic/` has the four section 27 files plus paired comparisons against A (`comparison.json`) and B (`comparison_vs_hybrid.json`). Findings are in section 9.3: not kept |
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
* **Retrieval** — dense, over FAISS with an exact numpy fallback; BM25 lexical
  retrieval (numbers kept whole, DD-049); Reciprocal Rank Fusion over ranks
  only, with exact arithmetic and a fixed tie-break (DD-045).
* **Reranking** — `bge-reranker-v2-m3` cross-encoder, optional by
  configuration (`reranking.enabled`), with a dependency-free term-overlap
  fallback for `--offline` (DD-048).
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
python -m src.cli run-benchmark --system hybrid --offline                    # results/hybrid
python -m src.cli run-benchmark --system hybrid --no-rerank --offline --out results/ablations/reranker_off
python -m src.cli compare-runs --baseline results/baseline --system results/hybrid
python -m src.cli run-benchmark --system agentic --offline                   # results/agentic
python -m src.cli compare-runs --baseline results/hybrid --system results/agentic --out results/agentic/comparison_vs_hybrid.json
```

`src/evaluation/statistics.py` is the DD-028 machinery. It computes a percentile
bootstrap 95% CI for every headline figure, written into each new
`results.json` as `confidence_intervals`, and runs the paired bootstrap that
`compare-runs` writes to `comparison.json`. Both read the stored
`per_question.json` records, so a published run is compared as published.

Module layout is in `ARCHITECTURE.md` section 3. The one deviation:
`src/ingestion/parser_base.py` holds the backend-independent parsing machinery,
added when DD-033 made the PDF library swappable. `src/ingestion/render.py`
(page-to-PNG rendering for `inspect-page`) is a second, smaller deviation for
the same reason: PDF-library use stays confined to `src/ingestion/`.

---

## 4. Deliberately absent

Not missing: excluded **from both baselines**, because DD-013 requires systems
that lack them before their value can be measured:

query planner · evidence controller · retrieval refinement · verifier

Phase 5 built them, in `src/agents/`, used only by `AgenticRAGPipeline`.
`tests/test_architecture.py` no longer asserts `src/agents/` is absent. It
asserts that Baselines A and B have none of the four components, do not
override the shared `ask`, and declare no agentic stage.

Lexical retrieval, rank fusion and reranking have left this list: they are
Phase 4's deliverable and are built, in Baseline B only. Two tests in
`tests/test_architecture.py` enforce the rest. One asserts `src/agents/` does
not exist. The other asserts Baseline A still builds a dense retriever and no
reranker, since Phase 4 is measured *against* it. The taxonomy's `verification`
category stays reserved for Phase 5. `reranking` now fires for real (DD-044).

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
| `src/agents/` does not exist, and Baseline A has no lexical retriever or reranker (DD-013) | `tests/test_architecture.py` |
| Lexical retrieval finds an exact identifier the offline dense retriever ranks fifth (DD-007) | `tests/test_hybrid.py` |
| RRF output depends on the rankings alone, not on score magnitudes (DD-045) | `tests/test_hybrid.py`, five score transforms including garbage |
| Reranker OFF returns the fused pool field for field and never calls a reranker | `tests/test_hybrid.py`, with a reranker that raises if called |
| A gold page the reranker pushed out is `reranking`, not `retrieval`; one it was never handed is `retrieval` (DD-044) | `tests/test_hybrid.py`, unit and end to end through `HybridRAGPipeline` |
| Baseline A reproduces `results/baseline/per_question.json` after the Phase 4 refactor | `tests/test_hybrid.py`, a fresh 76-question run compared record by record |
| Harness instrumentation does not change Baseline B's answers | `tests/test_hybrid.py`, reranker on and off |
| Paired bootstrap: zero differences give CI [0, 0] and p = 1.0; a constant difference collapses to itself; Baseline A vs itself is "no significant difference" on every metric | `tests/test_statistics.py` |
| A comparison that varies a held-constant field (EVALUATION_PROTOCOL.md 10) is refused | `tests/test_statistics.py` |
| The reranker counts toward the VRAM budget only when used; bf16 4B + both retrieval models is rejected | `tests/test_architecture.py` |

---

## 6. Verification evidence

* **508 tests pass**, in about 165 seconds. No network, no GPU, no model weights.
  Full 76-question runs of both baselines are among them, one of them a
  record-by-record check against `results/baseline/`, so the harness is
  exercised against the real dataset on every commit rather than only by hand.
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

| DD-044 | A gold page is lost "by the reranker" only if Reranker-OFF would have kept it; fixes DD-038 rule 2, which made `reranking` unreachable |
| DD-045 | RRF with k = 60, exact `Fraction` arithmetic, and a fixed tie-break over ranks only |
| DD-046 | Each retriever ranks to 20 and the reranker sees the fused top 20, the same pool with it on or off |
| DD-047 | The paired test's p-value is the one its percentile interval implies; 2,000 resamples, unchanged |
| DD-048 | The offline reranker is term coverage; primary metrics per RQ named before the run |
| DD-049 | BM25 with whole numbers, no stemming, and no zero-score hits |

DD-036 to DD-043 are Phase 3's; DD-044 to DD-049 are Phase 4's. All five open questions in
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

## 9.2 Baseline B vs Baseline A (Phase 4)

`results/hybrid/` (Baseline B: dense + BM25, RRF k = 60, reranker over the fused
top 20), `results/ablations/reranker_off/` (the same system with
`reranking.enabled = False`), both run over the same 76 questions as
`results/baseline/`. **The embedder and generator are the same fallback
components on all three sides**, so the comparisons are fair in the section 10
sense; `compare-runs` checks this and would refuse otherwise. **The reranker
here is the term-overlap stand-in (DD-048), not `bge-reranker-v2-m3`.** There is
still no GPU on this machine.

Paired bootstrap over questions, 2,000 resamples, seed 42 (DD-028, DD-047).
Difference = system − baseline. "n.s." = the 95% CI includes zero, which is
reported as **no significant difference** whatever the sign of the point
estimate.

| Metric (n) | A dense | B hybrid + rerank | B − A, 95% CI | p | Verdict |
| --- | ---: | ---: | --- | ---: | --- |
| **Recall@5** (70) — RQ1 primary | 0.886 | 0.943 | +0.057 [−0.014, +0.143] | 0.190 | n.s. |
| **accuracy (all)** (76) — RQ2 primary | 0.171 | 0.197 | +0.026 [−0.039, +0.092] | 0.570 | n.s. |
| **faithfulness** (76) — RQ2 primary | 0.934 | 0.934 | +0.000 [0, 0] | 1.000 | n.s. (identical) |
| MRR@10 (70) | 0.736 | 0.827 | +0.091 [+0.019, +0.167] | 0.016 | **significant improvement** |
| Full-Recall@5 (70) | 0.771 | 0.871 | +0.100 [+0.029, +0.186] | 0.019 | **significant improvement** |
| Recall@3 (70) | 0.814 | 0.914 | +0.100 [+0.029, +0.186] | 0.021 | **significant improvement** |
| Full-Recall@5, multi-hop + comparison (15) | 0.733 | 0.867 | +0.133 [+0.000, +0.333] | 0.231 | n.s. (CI touches zero) |
| citation any-correct (70) | 0.643 | 0.671 | +0.029 [−0.029, +0.100] | 0.535 | n.s. |
| abstention accuracy (6) | 0.167 | 0.167 | +0.000 [0, 0] | 1.000 | n.s. (identical) |
| false-answer rate (6) | 0.833 | 0.833 | +0.000 [0, 0] | 1.000 | n.s. (identical) |

Baseline B's own 95% CIs (percentile bootstrap, in `results/hybrid/results.json`):

| Metric | Value | 95% CI |
| --- | ---: | --- |
| Recall@5 | 0.943 | [0.886, 0.986] |
| MRR@10 | 0.827 | [0.756, 0.896] |
| accuracy (all) | 0.197 | [0.105, 0.289] |
| citation any-correct | 0.671 | [0.557, 0.771] |
| faithfulness | 0.934 | [0.868, 0.987] |
| abstention accuracy | 0.167 | [0.000, 0.500] |
| false-answer rate | 0.833 | [0.500, 1.000] |

The file has 19 contrasts; 4 are significant. Three are retrieval-ranking
metrics (MRR@10, Full-Recall@5, Recall@3) and the fourth is latency, discussed
below. Section 26.2 asks for the count and for weight on metrics a component
*should* move. Ranking metrics are the ones hybrid retrieval should move, so
they are read as evidence, not as the one in twenty expected by chance.

**Stopping rules.** B reaches citation any-correct 0.671, still below 0.75.
Abstention accuracy (0.167) and false-answer rate (0.833) are unchanged and
still miss. Recall@5 was already met.

### RQ1 — does hybrid retrieval improve evidence retrieval over dense alone?

Read from `results/ablations/reranker_off/comparison.json` (hybrid, reranker
off, vs A), which isolates hybrid retrieval:

* **Primary (Recall@5): no significant difference.** +0.043, CI [−0.029, +0.114],
  n = 70. Dense Recall@5 was already 0.886, so the headroom was 8 questions.
* **Secondaries: significant.** MRR@10 +0.074 [+0.025, +0.129], p = 0.003.
  Full-Recall@5 +0.086 [+0.014, +0.157]. The multi-hop/comparison subset (n = 15)
  is +0.133 [+0.000, +0.333]: n.s.
* Also significant, and not pre-declared: Recall@1 +0.100 and nDCG@10 +0.045.

**Answer:** on this benchmark, with this fallback embedder, hybrid retrieval
puts gold pages *higher*. It is not shown to put them *in the top 5 more
often*. Fused ranking is measurably better and first-order recall is not
distinguishable.

### RQ2 — does reranking improve answer quality enough to justify its cost?

Read from `results/hybrid/comparison_vs_reranker_off.json` (reranker on vs off,
same hybrid pool):

* **Primary: no significant difference.** Accuracy (all) +0.013 [−0.026, +0.053],
  3 questions changed. Faithfulness is identical.
* **Cost: a significant latency regression.** +0.002 s per question
  [+0.001, +0.003]. It is real, and trivial on this stand-in.
* **Secondaries: n.s.** Recall@5 +0.014, MRR@10 +0.017. One undeclared metric
  is significant: Recall@3 +0.071 [+0.014, +0.143]. Section 26.2 says to treat
  that as a hypothesis.
* **Taxonomy:** zero questions are filed as `reranking`. The stand-in pushed no
  gold page off the answer path. It reordered the top 5 on 58 of 76 questions,
  and `retrieval` failures fell from 5 (off) to 4 (on).

**Answer: not shown.** With the term-overlap stand-in, reranking does not
measurably improve answers. **This says nothing about `bge-reranker-v2-m3`.**
RQ2 stays open until the model-stack run (section 10).

### Taken together

**Baseline B does not significantly beat Baseline A on either primary metric.**
That is a legitimate result (EVALUATION_PROTOCOL.md 26.3) and it matches the
expectation `EXPERIMENT_PLAN.md` recorded before the run: section 9.1 had already
shown retrieval was not the bottleneck (46 of 65 failures were `generation`).
Better ranking changed 6 of 76 answers. `generation` is still the largest
category, 47 of 63 failures.

**Latency caveat.** The B − A latency contrast ("significant improvement", −0.001 s)
is not a finding. Baseline A's latencies were measured in a different session,
two days earlier, and both figures are milliseconds on a stack that loads no
weights. Only the on/off contrast was measured in adjacent runs.

**Taxonomy change.** DD-044 changed the error taxonomy version. Baseline A's
stored categories are still valid: a fresh run under the new rules reproduces
them record for record.

---

## 9.3 Agentic system vs both baselines (Phase 5)

`results/agentic/` holds the agentic system. It uses Baseline B's retrieval, fusion and reranking, with the rule-based planner, evidence controller, refinement and verifier (DD-050 to DD-055). It was run over the same 76 questions. **Fallback stack throughout**:
* hashing embedder, scripted generator and term-overlap reranker, as in 9.2;
* rule-based stand-ins for every agent decision. No LLM planned, assessed or verified anything, so every finding below is about the stand-ins.

The metrics were declared before the run, in DD-056, committed ahead of the results. Paired bootstrap, 2,000 resamples, seed 42. `comparison.json` is against A and `comparison_vs_hybrid.json` against B. Each file has **20 comparisons**, 40 in total. A has 6 significant, B has 5.

| Metric (n) | A | B | Agentic | Agentic − B, 95% CI | Verdict vs B |
| --- | ---: | ---: | ---: | --- | --- |
| **accuracy (all)** (76) — keep/drop | 0.171 | 0.197 | 0.184 | −0.013 [−0.066, +0.026] | n.s. |
| **faithfulness** (76) — keep/drop | 0.934 | 0.934 | 0.803 | −0.132 [−0.211, −0.053] | **significant regression** |
| **citation any-correct** (70) — keep/drop | 0.643 | 0.671 | 0.529 | −0.143 [−0.229, −0.071] | **significant regression** |
| **abstention accuracy** (6) — keep/drop | 0.167 | 0.167 | 0.333 | +0.167 [0.000, +0.500] | n.s. (CI touches zero) |
| **Full-Recall@5, multi-hop + comparison** (15) — RQ3 primary | 0.733 | 0.867 | 0.800 | −0.067 [−0.200, 0.000] | n.s. |
| **accuracy, multi-hop + comparison** (15) — RQ3 primary | 0.333 | 0.267 | 0.267 | 0.000 [0, 0] | n.s. (identical) |
| **unsupported-answer rate** (76) — RQ4 primary | 0.000 | 0.000 | 0.000 | 0.000 [0, 0] | n.s. (floor) |
| false-answer rate (6) | 0.833 | 0.833 | 0.667 | −0.167 [−0.500, 0.000] | n.s. |
| over-abstention rate (70) | 0.057 | 0.057 | 0.186 | +0.129 [+0.057, +0.214] | **significant regression** |
| Recall@5 (70) | 0.886 | 0.943 | 0.929 | −0.014 [−0.043, 0.000] | n.s. |
| MRR@10 (70) | 0.736 | 0.827 | 0.814 | −0.013 [−0.040, +0.001] | n.s. |
| latency (76) | 0.005 s | 0.004 s | 0.013 s | +0.009 s | significant regression (cost) |

Against A, the agentic system inherits B's ranking gains: MRR@10 +0.078 and Full-Recall@5 +0.086, both significant. It shows the same faithfulness and citation regressions as against B.

EVALUATION_PROTOCOL.md section 28 costs:
* 1.05 retrieval iterations per question on average, 2 at most;
* 0.87 model calls per question on average, 2 at most. That is below 1, because 13 questions were refused without calling the generator;
* LLM parse-failure rate: undefined, 0 LLM decisions offline.

### Keep-or-drop (EXPERIMENT_PLAN.md section 3)

**The agentic pipeline is not kept. Its added complexity did not pay for itself.** None of the four keep-or-drop metrics improves over Baseline B with a CI excluding zero. Two of them, faithfulness and citation any-correct, *significantly regress*. That is a legitimate result (EVALUATION_PROTOCOL.md 26.3), and it is a finding about the rule-based stand-ins.

The regressions have one cause: **the evidence controller abstains too often.**
* It refused 13 questions, 11 of them answerable. That is the over-abstention regression: 4 answerable questions refused by B, 13 by the agentic system.
* A refusal scores 0 on citation any-correct, and low on token faithfulness, because the refusal sentence's words are not in the evidence.
* Only 3 answers changed correctness against B:
  * q057, unanswerable, is now correctly refused, by the controller;
  * q069 was lost to a controller refusal;
  * q065 was lost to a verifier refusal.

The untuned 0.5 coverage threshold (DD-052) is the obvious suspect. Tuning it belongs on `--split dev` in Phase 6, not here.

### RQ3 — does agentic retrieval improve hard and multi-hop questions?

**Not shown.** Both primary metrics are not significantly different from B on the 15 multi-hop and comparison questions. Accuracy is identical, and Full-Recall@5 is −0.067, with a CI touching zero.
* The rule planner decomposed 10 of 76 questions.
* Refinement ran on 4 questions.
* The scripted generator answers from the single best-matching block, so it cannot combine two hops even when both are retrieved. DD-056 recorded this floor in advance.

**Open question 3** (does iteration 2 ever change an answer?): **once in 4.** Of the 4 refined questions, only q069's draft differed between the first and final round's context. `MAX_RETRIEVAL_ITERATIONS = 2` is almost never exercised by the rule controller: 10 of the 13 insufficient verdicts stopped with `no_new_query`, because the refined query would have repeated one already issued.

### RQ4 — does verification reduce unsupported answers?

**Unanswerable offline, as declared in DD-056.** The unsupported-answer rate is 0.000 for A, B and the agentic system. It cannot fall.
* The rule verifier passed 61 answers and failed 2.
* One of the two failures (q065) rejected a *correct* draft, and is the first record ever filed as `verification`. DD-054's counterfactual rule is what let the taxonomy say so, rather than filing it as `abstention`.
* The abstention gain on q057 came from the evidence controller, not the verifier, and can only be attributed to the controller by Phase 6's controller-OFF arm.

**Taxonomy (vs B):**
* `abstention` rose from 9 to 16;
* `generation` fell from 47 to 39;
* `verification` fires once, and `reranking` once.

---

## 10. Next step

**Phases 0 to 4 are done.** Both baselines run against the verified dataset
end to end. The harness scores both through the same code, and
`src/evaluation/statistics.py` puts a CI on every figure and a paired test on
every comparison.

In this order:

1. **Re-run all three arms against the model stack, on a GPU.** Every number in
   sections 9.1 and 9.2 comes from the fallback components. RQ2 in particular
   is unanswered until `bge-reranker-v2-m3` is the reranker being measured:

   ```bash
   python -m src.cli run-benchmark --out results/baseline
   python -m src.cli run-benchmark --system hybrid
   python -m src.cli run-benchmark --system hybrid --no-rerank --out results/ablations/reranker_off
   python -m src.cli compare-runs --baseline results/baseline --system results/hybrid
   ```

   All three must be re-run together. `compare-runs` refuses to compare a
   model-stack Baseline B with the fallback Baseline A, because the embedder and
   generator would differ (EVALUATION_PROTOCOL.md section 10).

2. **Phase 6: ablations and write-up.** Phase 5 is done (section 9.3). The
   ablation arms are configuration flags: `--no-planner`,
   `--no-evidence-controller`, `--no-refinement`, `--no-verification` and
   `--max-iterations`. The first question is whether the controller alone
   explains the regression. Sweep `agents.sufficiency_threshold` on
   `--split dev` only.

3. *(Done.)* **Phase 5: the agentic pipeline.** Phase 4's result sharpens what it has to
   do. Retrieval is not the bottleneck on this stack. Better ranking moved 6 of
   76 answers, and `generation` and `abstention` are 56 of the 63 failures. The
   planner and refinement loop improve retrieval; the verifier and evidence
   controller target answer grounding and abstention. The latter are the ones
   aimed at the measured failure. The taxonomy's `verification` category stays
   reserved for them.

`EXPERIMENT_PLAN.md` section 5's open questions 1, 2 and 3 (chunk size and
overlap, RRF vs. weighted fusion, `MAX_RETRIEVAL_ITERATIONS`) are answerable for
the first time: they were deferred pending a dev set, and the dev split now has
a harness to be swept with. Sweep on `--split dev` only; the eval split exists
to not be tuned against.
