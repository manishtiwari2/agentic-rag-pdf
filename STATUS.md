# Project Status

Last updated: 2026-09-23 (Phase 6 ablations, final table, error analysis and verdicts)

A factual record of what exists, what it is verified to do, and what blocks the
next step. `EXPERIMENT_PLAN.md` says what order to build in; this says how far
along that order the project actually is.

---

## 1. At a glance

| | |
| --- | --- |
| Phases complete | 0 (benchmark), 1 (ingestion + indexing), 2 (dense baseline), 3 (metrics harness), 4 (hybrid + reranking), 5 (agentic), 6 (ablations + write-up, offline stack) |
| Next step | **GPU model-stack re-run** (section 9.5) |
| **Phase 6 result** | **On the offline stand-in stack, no component earned its cost** (DD-058 rule, DD-060). The evidence controller alone caused Phase 5's regression: removing it recovers faithfulness +0.105 [+0.039, +0.171] and citation +0.086 [+0.029, +0.157]. Threshold tuning on dev (0.5 → 0.3) changed no eval answer. Error analysis found a chunking defect (heading text moved into `chunk.section`, which no component reads) and a citation-parsing defect behind q065 (section 9.4) |
| **Phase 5 result** | **The agentic pipeline is not kept: its added complexity did not pay for itself.** No keep-or-drop metric improves over Baseline B. Faithfulness −0.132 [−0.211, −0.053] and citation any-correct −0.143 [−0.229, −0.071] *regress*, because the rule-based evidence controller refused 11 answerable questions. RQ3 not shown, and RQ4 unanswerable offline (floor), as DD-056 declared in advance. Rule-based stand-ins for every agent decision (section 9.3) |
| Tests | 582 collected: 580 passing, 2 failing. Both failures are in `tests/test_agentic.py` and were already on `main` (missing-number refinement; regeneration cap 2); see section 9.5 |
| Source | 11,933 lines in `src/`, 5,565 lines in `tests/` |
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
| **6 — Ablations + write-up** | Seven ablations, comparison table, error analysis, a verdict per component | **Met, on the offline stand-in stack.** The seven arms (DD-057) are in `results/ablations/`, and the metrics and rules were pre-declared in DD-058, committed before any run. The dev threshold and iteration sweeps plus one eval run are in `results/experiments/` (DD-059). The section 28 table, ablation verdicts and error analysis are generated into `results/final/` by `final-table`. Findings and verdicts are in section 9.4 and DD-060 |

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
| DD-050–DD-056 | Phase 5: the agentic loop, its four rule-based components, and the pre-declared RQ3/RQ4/keep-or-drop metrics |
| DD-057 | The seven ablations are section 24's six components on the agentic system, plus the existing reranker-OFF arm on Baseline B |
| DD-058 | Phase 6's metrics, floors, threshold rule and "earned its cost" rule, declared before any run |
| DD-059 | Sufficiency threshold 0.3, chosen on dev; the iteration cap stays 2 |
| DD-060 | Phase 6 verdicts: no component earned its cost offline; the controller's refusals are two rules, one fed by a chunking defect |

DD-036 to DD-043 are Phase 3's; DD-044 to DD-049 are Phase 4's; DD-050 to DD-056 Phase 5's; DD-057 to DD-060 Phase 6's. Of the five open questions in
`EXPERIMENT_PLAN.md` section 5, 4 and 5 are decided, and 3 is closed for the
offline stack (DD-059). 1 (chunk size) and 2 (fusion) remain open: no Phase 6
arm varies them, and no unplanned sweep was added.

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

## 9.4 Ablations, final comparison table and verdicts (Phase 6)

> **Offline stand-in stack, as in 9.1-9.3.** Hashing embedder, scripted
> extractive generator, term-overlap reranker, and rule-based planner, evidence
> controller and verifier. No figure in this section is about Qwen3-4B, bge-m3
> or bge-reranker-v2-m3.

Everything below was pre-declared before any Phase 6 run:
* DD-057 reconciles "seven" with section 24's six components;
* DD-058 fixes each arm's metrics, the floors, the threshold rule and the
  "earned its cost" rule.

The tables are generated from stored results by `python -m src.cli
final-table`, which writes `results/final/` (`REPORT.md` plus three JSON files).
`tests/test_ablations.py` fails if the committed `results/final/` differs from a
fresh regeneration.

### Final comparison table (EVALUATION_PROTOCOL.md 28), eval split, 43 questions

> Offline stand-in stack. Values with 95% bootstrap CIs. The stored all-question
> runs are restricted to eval by filtering their records, not by re-running them
> (DD-058). The tuned arm was run once, on eval only.

| System | Recall@5 | MRR@10 | Accuracy | Faithfulness | Citation any-correct | Abstention acc. (n=3) | P95 latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense RAG | 0.850 [0.725, 0.950] | 0.733 [0.611, 0.848] | 0.116 [0.023, 0.209] | 0.930 [0.837, 1.000] | 0.600 [0.450, 0.750] | 0.000 [0, 0] | 0.009 [0.007, 0.011] |
| Hybrid RAG | 0.950 [0.875, 1.000] | 0.851 [0.759, 0.933] | 0.186 [0.070, 0.302] | 0.930 [0.837, 1.000] | 0.650 [0.500, 0.775] | 0.000 [0, 0] | 0.011 [0.009, 0.011] |
| Agentic RAG (0.5) | 0.950 [0.875, 1.000] | 0.849 [0.755, 0.933] | 0.140 [0.047, 0.256] | 0.814 [0.698, 0.930] | 0.525 [0.375, 0.675] | 0.000 [0, 0] | 0.026 [0.024, 0.027] |
| Agentic RAG (0.3, tuned on dev) | 0.950 [0.875, 1.000] | 0.849 [0.755, 0.933] | 0.140 [0.047, 0.256] | 0.814 [0.698, 0.930] | 0.525 [0.375, 0.675] | 0.000 [0, 0] | 0.105 [0.026, 0.155] |

| System | Avg retrieval iterations | Avg model calls | Peak VRAM |
| --- | ---: | ---: | --- |
| Dense / Hybrid | null (not reported, DD-055) | null | null: no GPU, not measured |
| Agentic (0.5) | 1.070 [1.000, 1.163] | 0.907 [0.767, 1.023] | null: no GPU, not measured |
| Agentic (0.3) | 1.070 [1.000, 1.163] | 0.907 [0.767, 1.023] | null: no GPU, not measured |

How to read it:
* **None of the three eval unanswerable questions is refused by any system.**
  The agentic system's only abstention gain on all 76 questions, q057, is a dev
  question.
* **Latencies are milliseconds, from different sessions.** The tuned arm's
  higher P95 is a later session on a loaded machine, not a cost of the
  threshold: its answers are identical to the reference's.
* **All-76-question table.** `results/final/REPORT.md` has it, labelled as not a
  headline. Its figures match sections 9.1-9.3.

### Ablations (arm − reference, all 76 questions)

> Offline stand-in stack. The reference is `results/agentic/`, except arm 7,
> whose reference is `results/hybrid/`. **Bold** = 95% CI excludes zero.

| Arm | Declared metrics | Faithfulness / citation (attribution) | Model calls added by component | Answers changed | Verdict (DD-058) |
| --- | --- | --- | ---: | ---: | --- |
| 1 planner OFF | FR@5 multi +0.067 [0, +0.200]; acc. multi 0.000 | 0.000; +0.029 [0, +0.071] | +0.013 | 4 | did not earn its cost |
| 2 hybrid OFF | Recall@5 0.000 [−0.043, +0.043]; MRR@10 +0.003 [−0.010, +0.016] | −0.013; −0.014 | +0.013 | 7 | did not earn its cost |
| 3 reranker OFF (agentic) | MRR@10 −0.011 [−0.067, +0.039]; acc. +0.013 [0, +0.039] | +0.015; −0.014 | 0.000 | 31 | did not earn its cost |
| 4 refinement OFF | over-abst. +0.014 [0, +0.043]; FR@5 0.000 | −0.013; 0.000 | +0.026 | 1 | did not earn its cost |
| 5 evidence controller OFF | abst. acc. −0.167 [−0.500, 0]; **over-abst. −0.100 [−0.172, −0.029]** | **+0.105 [+0.039, +0.171]; +0.086 [+0.029, +0.157]** | **−0.171** | 8 | did not earn its cost: removing it *improves* faithfulness and citation |
| 6 verification OFF | unsupported 0.000 (floor); faithfulness +0.026 [0, +0.066] | +0.026; +0.029 | +0.039 | 3 | did not earn its cost |
| 7 reranker OFF (Baseline B, Phase 4) | acc. −0.013 [−0.053, +0.026]; faithfulness 0.000 | 0.000; −0.043 [−0.100, 0] | n/a | 31 | corroborates arm 3 |

Notes on the table:
* **Latency added by any component** is at most +0.002 s, and none is
  significant.
* **Model calls.** The evidence controller *saves* 0.171 generator calls per
  question, because each refusal skips generation.
* **Iteration cap.** `--max-iterations 1` is behaviourally identical to arm 4
  (DD-057). It is part of the dev sweep, not an arm.

### Verdicts, one per component

On the offline stack, **no component earned its cost.** Every null below is
bounded by a floor DD-058 stated before the run.

* **Hybrid retrieval: did not earn its cost inside the agentic system.**
  * Recall@5 and MRR@10 are unchanged with it off.
  * At the baseline level (Phase 4, RQ1) it *did* significantly improve ranking
    over dense (MRR@10 +0.074), though not its primary metric, Recall@5.
  * Inside the agentic loop that ranking gain disappears. The planner's RRF
    merge over sub-queries does comparable re-ranking.
* **Reranker (term-overlap stand-in): did not earn its cost.** There is no
  significant ranking or answer effect either inside the agentic system
  (arm 3) or on Baseline B (arm 7).
* **Planner: did not earn its cost.**
  * Multi-hop metrics: 0 of 15 answers changed, and Full-Recall@5 is 0.067
    *higher* without it (n.s.).
  * The scripted generator answers from one block, so decomposition cannot
    combine hops (DD-056 floor).
  * One undeclared contrast is significant: nDCG@10 +0.010 without the
    planner. It is isolated, so it is treated as a hypothesis.
* **Refinement: did not earn its cost.** Removing it changed 1 answer in 76, a
  wrong answer becoming a refusal. That is the DD-058 floor: refinement ran on
  4 questions.
* **Evidence controller: did not earn its cost, and is the cause of the Phase 5
  regression.**
  * The pre-declared attribution holds: only this arm recovers faithfulness
    and citation.
  * Its intended benefit, refusing unanswerable questions, is one question
    (q057), and the CI includes zero.
  * It refuses 11 answerable questions.
* **Verifier: did not earn its cost.**
  * Its primary metric sits at the 0.000 floor.
  * Its only action that changed correctness was rejecting a correct answer
    (q065), caused by a parsing defect (below).

**Multiple comparisons.**
* Phase 6 made 168 paired contrasts, 8 files of 21. The 6 arms account for 126
  contrasts, 5 of them significant. The 2 eval files account for 42, also 5
  significant.
* Of the ablation contrasts, 4 significant ones are in the controller arm, on
  metrics the controller plausibly moves. The fifth is the planner's isolated
  nDCG result.
* With Phases 4-5, the project has made 265 contrasts, and this report counts
  them all.

### Evidence-controller threshold and iteration cap (DD-059)

Dev sweep (33 questions, dev-only numbers):

| threshold | 0.2 | **0.3** | 0.4 | 0.5 | 0.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy (all) | 0.212 | **0.242** | 0.242 | 0.242 | 0.242 |
| citation any-correct | 0.567 | **0.533** | 0.533 | 0.533 | 0.467 |
| over-abstention | 0.100 | **0.133** | 0.167 | 0.167 | 0.233 |

* **Choice.** The pre-declared rule chose 0.3: accuracy tied, citation broke
  the tie with 0.6, and over-abstention broke the tie among the rest. The dev
  results and the choice were committed before eval was touched.
* **Eval, one run.** Threshold 0.3 gives answers identical to 0.5 on all 43
  eval questions. Against Baseline B on eval, it still significantly regresses
  faithfulness (−0.116 [−0.209, −0.023]), citation any-correct
  (−0.125 [−0.225, −0.025]) and over-abstention. **Not kept**, under the same
  rule DD-056 applied.
* **Why tuning could not help.** Of the reference's 13 controller refusals, 5
  come from the second rule, "every question number must appear in the
  context". No threshold touches that rule, and 3 of the 6 eval refusals at 0.3
  are number-only.
* **Iteration cap.** A cap of 3 is identical to 2 on dev. A cap of 2 differs
  from 1 on one question, a refusal becoming a wrong answer. Open question 3 is
  closed for the offline stack.
* **Disclosure (section 5.2).** Phase 5's diagnosis, which prompted this tuning,
  cited eval failures (q065, q069). The *value* was chosen on dev alone.

### Error analysis (sections 23, 30)

> Offline stand-in stack. All categories for all 11 runs are in
> `results/final/error_analysis.json`.

| Category | Dense | Hybrid | Agentic | Controller OFF |
| --- | ---: | ---: | ---: | ---: |
| retrieval | 8 | 4 | 4 | 4 |
| reranking | 0 | 0 | 1 | 0 |
| context-selection | 1 | 2 | 2 | 2 |
| abstention | 9 | 9 | 16 | 10 |
| citation | 1 | 1 | 1 | 1 |
| verification | 0 | 0 | 1 | 1 |
| generation | 46 | 47 | 39 | 46 |
| **failures / 76** | 65 | 63 | 64 | 64 |

**Question types that stay hard** (accuracy, best of dense / hybrid / agentic):

| Type (n) | Best accuracy |
| --- | ---: |
| ambiguous (5) | 0.000 |
| summary (3) | 0.000 |
| definition (8) | 0.125 |
| explanation (8) | 0.125 |
| multi_hop (7) | 0.143 |
| table (7) | 0.143 |
| numerical (11) | 0.273 |
| factual (13) | 0.308 |
| unanswerable (6) | 0.333 |
| comparison (8) | 0.500 |

**58 of 76 questions are answered correctly by none of the three systems.**

**Manual inspection.** The sample is the first failures per category in the
agentic reference, plus q065 and q069. What actually went wrong:

* **generation** (q003, q004, q006): the gold page is retrieved, and the
  scripted backend copies the best-matching table row or sentence. Examples:
  * q003: the Digital Logic line with Boolean algebra, not the circuit-design
    clause the question asks about;
  * q004: the Databases heading, not the B/B+ tree example.

  Whole-block extraction against short reference answers fails the 0.6 token
  overlap. This is the fallback generator's ceiling, not a retrieval problem.
* **retrieval** (q020, q023, q028): the gold page is never in the top 5.
  * q020 and q028 retrieve a table-of-contents or header block that shares the
    question's words ("Table 1", `git pull` in a contents list). Hashing and
    BM25 both reward term overlap with navigation text.
  * q023 is a multi-hop question whose two gold pages (6, 8) both lose to
    pages repeating "mobile devices".
* **context-selection** (q009, q022): the gold page is in the depth-10 probe
  but was dropped from the evidence the generator saw.
* **abstention** (q001, q002, q005): answerable questions refused by the
  controller. **The root cause is a chunking defect.**
  * `doc1.pdf` is two pages, and both start with "GATE 2027 IIT Madras |
    Organizing Institute". The structure chunker takes that line as a heading
    and stores it in `chunk.section`, not `chunk.text`. Only citation labels
    read `section`; retrieval, generation and every agent read `text`.
  * The context therefore never contains "2027", and the number rule refuses
    every GATE question that mentions the year.
  * q001's answer, "IIT Madras", is in the deleted line, so no system can
    answer q001.
* **verification** (q065): the verifier rejected a **correct** draft. The draft
  is a verbatim passage from block C1. That passage carries the paper's own
  reference number, "shrinking[3]". `src/generation/citations.py` accepts a bare
  `[3]` as a marker, so the reference became `[C3]`. The verifier then checked
  the preceding claims against C3, a licence notice, and failed them. The
  verifier thresholds are not the cause.
* **reranking** (q073): the pre-rerank context held gold page 5. After the
  planner's sub-queries and the reranker, the final context is pages 1 and 9.
  The answer is copied from a reference-list entry matching "FinFET". The
  DD-044 rule files this as `reranking`.
* **q069** (controller refusal, eval): the question asks about the "iPhone 6".
  The context never contains the token "6" on its own, so the number rule
  refuses.

---

## 9.5 What remains

* **The GPU model-stack run.** Every number in 9.1-9.4 is from the offline
  stand-ins. Re-run baselines, agentic, the seven arms and the eval-split tuned
  arm with Qwen3-4B + bge-m3 + bge-reranker-v2-m3 and LLM agents, then
  `final-table`. RQ2, RQ3 and RQ4 stay open until then. That run also fills in
  peak VRAM, the model-call cost of LLM agents and the LLM parse-failure rate.
* **The notebook** (DD-002, DD-034), which should call `final-table` rather than
  rebuild the table.
* **Follow-ups found by the error analysis.** None was fixed here, because each
  changes the system every stored result measured:
  1. the structure chunker moves heading text into `chunk.section`, where no
     retriever, generator or agent can see it;
  2. bare `[n]` bibliography references parsed as evidence markers;
  3. the controller's number rule refuses on a missing year or model number.
* **Two pre-existing test failures on `main`**, both in `test_agentic.py`:
  * missing-number refinement returns no query;
  * a regeneration cap of 2 verifies one time too few.

  They are not caused by Phase 6. Separately, the `offline_config` fixture
  bug, which broke several other agentic tests, is fixed here.
* **The LLM judge** has still never been run (out of scope).

## 10. Next step

**Phases 0 to 6 are done on the offline stand-in stack** (section 9.4). The
experiment is fully specified, and every table regenerates from stored results
with `python -m src.cli final-table`. What remains is section 9.5: the GPU
model-stack run, the notebook, and three follow-ups found by the error analysis.

In this order:

1. **Re-run everything against the model stack, on a GPU.** Every figure in
   sections 9.1-9.4 comes from the stand-ins. All runs must be re-run together,
   because `compare-runs` refuses to mix stacks (EVALUATION_PROTOCOL.md 10):

   ```bash
   python -m src.cli run-benchmark --system dense --out results/baseline
   python -m src.cli run-benchmark --system hybrid
   python -m src.cli run-benchmark --system hybrid --no-rerank --out results/ablations/reranker_off
   python -m src.cli run-benchmark --system agentic
   python -m src.cli run-benchmark --system agentic --no-planner --out results/ablations/planner_off
   # ... likewise --no-hybrid, --no-rerank, --no-refinement,
   # --no-evidence-controller and --no-verification (DD-057)
   python -m src.cli final-table
   ```

   There is no threshold to re-sweep: `agents.sufficiency_threshold` is read
   only by the rule-based controller (`src/agents/evidence.py`). The LLM
   controller decides sufficiency itself. The iteration-cap sweep *is* worth
   redoing on `--split dev`.

   Step-by-step instructions for all three remaining steps are in
   `NEXT_STEPS.md`.

2. **Decide the three ingestion, citation and controller follow-ups**
   (section 9.5) before the model-stack run. Each changes what every system
   sees, so a fix belongs before the numbers that are meant to last, not after.

3. **The notebook** (DD-002), calling `final-table` rather than rebuilding the
   table by hand.
