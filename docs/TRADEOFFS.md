# Tradeoffs

One section per hard part: the decision, the alternatives, why, the measured
evidence, what I would do at scale, and the design decision (DD) behind it.

> **Every measured figure in this document comes from the offline stand-in
> stack:** hashing embedder, scripted extractive generator, term-overlap
> reranker, rule-based agents. None is a Qwen3-4B / bge-m3 /
> bge-reranker-v2-m3 result. The GPU model-stack run has not been done.

Sections for chunking (DD-071), hybrid search (DD-072), the agent loop,
citations, abstention, OCR and stale/conflicting sources are added as each
work item lands (issue #36).

---

## Re-ranking

**Decision.** A cross-encoder reranker (`bge-reranker-v2-m3` on the model
stack, the term-overlap stand-in offline, DD-048) reorders the fused top 20
(DD-046). It is on by default and can be switched off with one field
(`reranking.enabled`, `--no-rerank`).

**Alternatives considered.**
* No reranker: return the fused RRF order as it is.
* A larger candidate pool (more pairs, more latency).
* An LLM reranker, which would cost one generation call per candidate set on a
  GPU that already holds the generator.

**Why.**
* A bi-encoder embeds the query and each passage separately. A cross-encoder
  reads them together, which is what separates passages that share terms with
  the question from passages that answer it.
* Its cost is bounded and predictable: one forward pass per candidate pair,
  batched, with no generation.
* It is optional by configuration, so its effect is always measured as an
  ablation rather than assumed (DD-044 attributes lost gold pages to it only
  when the reranker-OFF run kept them).

**Measured evidence (offline stand-in stack, all 76 questions, reranker ON −
OFF, paired bootstrap 95% CI).** The full generated table, per question type
and for both systems, is
[`results/experiments/reranker/TABLE.md`](../results/experiments/reranker/TABLE.md)
(`python -m src.cli reranker-table`; no new run, the stored ablations only).
* **Baseline B:**
  * Recall@5 +0.014 [+0.000, +0.043];
  * MRR@10 +0.032 [−0.020, +0.089], 18 questions reordered;
  * accuracy +0.026 [−0.026, +0.079].
  None is significant. The gains sit in explanation and numerical questions;
  comparison and definition questions are ranked slightly worse.
* **Agentic:** the same picture, plus faithfulness −0.013 [−0.039, +0.000],
  which is one numerical question.
* **Cost:** latency is the only significant contrast. Offline it is about 1 ms
  per question (agentic about 5 ms), which says nothing about a cross-encoder
  on a T4.
* **The cost that carries over** is the count of candidate pairs scored:
  * 17.2 per question on Baseline B;
  * 29.7 on the agentic system, which reranks once per retrieval query (1.30
    invocations per question).
  Each pair is one cross-encoder forward pass on the model stack. The agentic
  records store no per-question `rerank_s`, so its offline rerank seconds are
  reported as n/a, not 0.

**Verdict.** No decision changes. On this stack the reranker earns nothing
measurable: the term-overlap stand-in mostly re-sorts what fusion already
ranked well (DD-060 reached the same verdict for the agentic arm). It stays on
by default because RQ2 is a model-stack question: whether a real cross-encoder
pays for about 17-30 passes per question can only be measured with one.

**What I would do at scale.**
* Measure the real cross-encoder's p95 latency per pair batch on the target
  GPU.
* Cap the candidate pool by that budget (the `reranking.candidates` sweep),
  rather than fixing it at 20.
* Skip reranking when fusion's top result is a clear winner, which would save
  most of the passes on easy lexical questions.
* Cache cross-encoder scores per (query, chunk) for repeated queries in chat.

**DDs.** DD-044 (lost-gold attribution), DD-046 (reranker sees the fused top
20), DD-048 (offline reranker, primary metrics named first), DD-060 (Phase 6
verdicts).
