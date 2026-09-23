# Experiment Plan

This file was empty and referenced by nothing. It now holds the sequencing,
risk and stopping-rule decisions that the other specifications assume exist
but never state.

`PROJECT_SPEC.md` says *what* to build, `ARCHITECTURE.md` says *how*,
`BENCHMARK_SPEC.md` and `EVALUATION_PROTOCOL.md` say how to measure it. This
file says **in what order, and when to stop**.

---

## 1. Critical path

The blocking task is not the retrieval code. It is the benchmark.

```text
benchmark documents chosen + licensed
        ↓
questions written + ground truth verified
        ↓
  ┌─────┴─────┐
  ↓           ↓
baseline    metrics harness
  └─────┬─────┘
        ↓
hybrid → agentic → ablations → write-up
```

Everything downstream of the dataset is unverifiable without it. Building the
agentic loop first is the tempting order and the wrong one: it produces code
that cannot be shown to work.

---

## 2. Phases and exit criteria

A phase is done when its exit criterion is met, not when the code exists.

### Phase 0 — Benchmark construction

**Exit:** 5-10 licensed PDFs in `benchmark/documents/`, ~75 questions in
`questions.json` passing every check in `BENCHMARK_SPEC.md` 5.1, all evidence
pages verified against the rendered PDFs, dev/eval split assigned.

This is the largest single block of manual work in the project and the easiest
to underestimate. Verifying an evidence page means opening the PDF and looking,
for every question.

### Phase 1 — Ingestion and indexing

**Exit:** a PDF parses to page-attributed chunks; chunk page numbers spot-checked
against the source; a scanned PDF produces a clear error rather than garbage.

**Status: met, 2026-09-21.** Two chunking strategies behind one interface, two
PDF backends behind one protocol, page attribution verified mechanically across
both (zero mis-attributed chunks), scanned/encrypted/blank PDFs each raising a
distinct error with a remedy.

### Phase 2 — Dense baseline

**Exit:** Baseline A answers the benchmark end to end and produces
`results/baseline/`. Numbers may be poor. They must exist.

**Status: met, 2026-09-21.** All 76 questions answered end to end against the
five benchmark PDFs; `results/baseline/` holds the four files
`EVALUATION_PROTOCOL.md` section 27 names. The numbers are poor — accuracy 0.171
over all questions — and they were produced by the dependency-free fallback
stack on a machine with no GPU, which `STATUS.md` section 9.1 states beside
every figure. The criterion asked for numbers that exist, not good ones.

### Phase 3 — Metrics harness

**Exit:** retrieval, answer, citation and abstention metrics computed; error
taxonomy assigns a category to every failure; per-question records saved.

Deliberately before the interesting systems. Without it there is no way to
tell whether the next change helped.

**Status: met, 2026-09-21.** `src/evaluation/metrics.py` and
`src/evaluation/benchmark.py`, driven by `python -m src.cli run-benchmark`.
Retrieval is scored at page level with `unanswerable` questions excluded rather
than scored 0.0 (DD-025); the nine-category taxonomy (DD-038) assigned exactly
one category to each of the 65 failing records in the first run, and the two
categories whose components DD-013 leaves out stay reserved rather than
invented. The harness is system-agnostic, so Phases 4 and 5 reuse it.

### Phase 4 — Hybrid retrieval and reranking

**Exit:** Baseline B measured against Baseline A with a paired test (DD-028).
Answers RQ1 and RQ2.

**Unblocked.** The first thing the baseline measured is that retrieval is not
its bottleneck: Recall@5 is 0.886, clearing the stopping rule below, while 46 of
65 failures are filed as `generation`. Phase 4 improves retrieval, so the honest
expectation is a small gain — which is worth recording now, before the result.

**Status: met, 2026-09-23.** `results/hybrid/comparison.json` is the paired
bootstrap of Baseline B against `results/baseline/`: 76 questions, 2,000
resamples. The expectation held. Neither pre-declared primary metric differs
significantly: Recall@5 +0.057, CI [−0.014, +0.143]; accuracy +0.026, CI
[−0.039, +0.092]. Ranking quality does improve significantly: MRR@10 +0.091,
CI [+0.019, +0.167]. A reranker-off arm (`results/ablations/reranker_off/`)
separates the two questions:

* RQ1: hybrid retrieval ranks gold pages higher, but does not significantly
  raise Recall@5.
* RQ2: with the offline term-overlap stand-in, reranking shows no significant
  answer gain. That finding does not transfer to `bge-reranker-v2-m3`.

Fallback stack throughout; `STATUS.md` section 9.2 has the full table.

### Phase 5 — Agentic pipeline

**Exit:** planner, evidence controller, refinement and verifier working, all
loops bounded, full trace per query. Measured against both baselines.
Answers RQ3 and RQ4.

### Phase 6 — Ablations and write-up

**Exit:** the seven ablations in `EVALUATION_PROTOCOL.md` section 24, the
comparison table, the error analysis, and a statement for each component on
whether it earned its cost.

---

## 3. Stopping rules

The specifications define success qualitatively. These are the quantitative
lines, set **now**, before any results exist — a threshold chosen after seeing
the numbers is not a threshold.

| Criterion | Target | Rationale |
| --- | --- | --- |
| Recall@5 (answerable) | >= 0.80 | below this, generation quality is not the bottleneck and tuning it is wasted effort |
| Abstention accuracy | >= 0.70 | the primary safety property |
| False-answer rate | <= 0.30 | a confident wrong answer is the worst failure mode |
| Citation any-correct | >= 0.75 | citations are a headline feature; below this they mislead |
| Median query latency | <= 30 s on a T4 | beyond this the notebook is unusable interactively |
| Peak VRAM | <= 10 GB | must leave headroom on a 16 GB T4 |

**Minimum publishable result:** Phase 2, 3 and 4 complete with honest numbers.
A measured dense-vs-hybrid comparison on a real benchmark is a genuine result
even if the agentic phase is never reached.

**The agentic pipeline is kept only if** it improves at least one of accuracy,
faithfulness, citation correctness or abstention by a margin whose 95% CI
excludes zero (DD-028). Otherwise the write-up reports that the added
complexity did not pay for itself. That is a legitimate finding and the
project should be willing to publish it.

---

## 4. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Benchmark construction takes far longer than planned | high | high | start Phase 0 first; accept 50 questions over a delayed 75 |
| Ground-truth page numbers are wrong | medium | high | verify every page against the rendered PDF; a silent off-by-one looks like a retrieval bug |
| Small model cannot produce reliable structured output | high | medium | every agent decision needs a deterministic fallback; measure the parse failure rate |
| Agentic pipeline shows no measurable gain | medium | medium | it is a valid result; the plan already accounts for reporting it |
| Colab session expires mid-run | high | medium | checkpoint per-question results as they are produced, not at the end |
| Peak VRAM exceeds the T4 | medium | high | 4-bit quantisation, or sequential model loading |
| Judge self-preference inflates results | high | medium | DD-027: deterministic cross-check plus manual inspection of disagreements |
| Tuning against the evaluation set | medium | high | fixed `split` field; dev set only during development |

---

## 5. Open questions

Decisions deferred until there is evidence, recorded so they are not forgotten:

1. **Chunk size and overlap.** No principled starting value. Sweep on the dev
   set in Phase 1; do not tune further afterwards.
2. **RRF vs. weighted fusion**, and the dense/lexical weights. RRF is the
   default (DD-008) because it needs no calibration, not because it is known
   to be better here. Still open after Phase 4: RRF is built with k = 60
   untuned (DD-045), and `retrieval.fusion` is a field so a weighted method can
   be swept on `--split dev` without code changes elsewhere.
3. **`MAX_RETRIEVAL_ITERATIONS = 2`.** Chosen for latency, not measured. Phase
   5 should check whether iteration 2 ever changes an answer.
4. ~~**How the notebook obtains `src/`.**~~ **Decided 2026-09-21 (DD-034):**
   the setup cell clones the repository at a pinned tag and adds it to
   `sys.path`. The pinned ref is recorded with every run. The notebook is not
   self-contained, which is accepted because it already needs network access
   for the model weights.
5. ~~**Project license.**~~ **Decided 2026-09-21 (DD-035): Apache-2.0.**
   Unblocked by DD-033, which moved the default PDF backend from PyMuPDF
   (AGPL-3.0) to pdfplumber (MIT). `LICENSE` and `NOTICE` are in place and
   `README.md` section 20 is updated. The only licence work left is Phase 0's:
   recording the source and licence of every benchmark PDF.
