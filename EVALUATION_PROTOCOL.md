# Evaluation Protocol

## 1. Purpose

This document defines the procedure used to evaluate and compare the PDF RAG systems.

The objective is to make benchmark results:

* reproducible
* comparable
* statistically meaningful where practical
* independent of accidental configuration changes

The protocol applies to:

```text id="6t1vlz"
Dense RAG
Hybrid/Reranked RAG
Agentic RAG
Ablation experiments
Model comparisons
```

---

# 2. Evaluation Environment

Every final benchmark run must record:

```text id="6evq7c"
Python version
PyTorch version
Transformers version
embedding library/version
reranker library/version
GPU name
GPU memory
system RAM
CUDA version
```

The final notebook should display this information automatically.

---

# 3. Configuration Snapshot

Every benchmark result must store the complete experiment configuration.

Minimum configuration:

```yaml id="y07l7s"
generation_model:
generation_model_revision:
quantization:

embedding_model:
embedding_model_revision:

reranker_model:
reranker_enabled:

chunking_strategy:
chunk_size:
chunk_overlap:

retrieval_strategy:
top_k:
rerank_k:

max_retrieval_iterations:

temperature:
max_new_tokens:

random_seed:
```

No benchmark result should exist without a corresponding configuration.

---

# 4. Dataset Version

The benchmark dataset must have an explicit version.

Example:

```text id="r2yn0n"
benchmark_version: 1.0
```

If questions, answers, evidence pages, or documents change, increment the benchmark version.

Do not compare results from different benchmark versions without explicitly noting the difference.

---

# 5. Dataset Split

Where practical, maintain:

```text id="2j9vqs"
development set
evaluation set
```

The development set may be used during implementation.

The evaluation set should remain unchanged during final tuning.

This reduces the risk of optimizing directly against the final benchmark.

---

# 6. Evaluation Modes

The system must support two modes.

## Interactive Mode

Used for:

* demonstrations
* debugging
* manual inspection

## Benchmark Mode

Used for:

* automated evaluation
* latency measurement
* metric calculation
* result storage

Benchmark mode must avoid manual intervention.

---

# 7. Baseline Configuration

The first baseline is:

```text id="8q2qj7"
PDF
 ↓
chunking
 ↓
dense retrieval
 ↓
top-k context
 ↓
generation
```

No query planner or agentic retrieval loop should be used.

This establishes the simplest useful reference point.

---

# 8. Hybrid Baseline

The second configuration is:

```text id="z6j5qv"
PDF
 ↓
chunking
 ↓
dense retrieval + lexical retrieval
 ↓
fusion
 ↓
reranking
 ↓
generation
```

This measures the value of retrieval engineering before introducing agents.

---

# 9. Agentic Configuration

The proposed system is:

```text id="4jv7jh"
query
 ↓
query planner
 ↓
retrieval
 ↓
fusion
 ↓
reranking
 ↓
evidence assessment
 ↓
optional refinement
 ↓
context selection
 ↓
generation
 ↓
verification
 ↓
final answer
```

All other benchmark conditions should remain equivalent.

---

# 10. Fair Comparison

When comparing systems, keep constant wherever possible:

```text id="9k3tuj"
documents
questions
ground truth
chunking
generation model
generation parameters
evaluation criteria
```

When testing a specific component, change only that component.

Example:

```text id="ebquzt"
Test reranker
→ keep embedding model and generator unchanged.
```

---

# 11. Randomness

Set deterministic seeds where supported.

At minimum:

```python id="j8jv1h"
SEED = 42
```

Record the seed in every experiment.

Some GPU/model operations may remain nondeterministic.

The notebook should document this rather than claiming perfect determinism.

---

# 12. Generation Parameters

The benchmark should use fixed generation parameters.

Initial defaults:

```text id="16w3at"
temperature = 0.0
```

Use deterministic or near-deterministic generation for benchmark evaluation where supported.

Set a fixed maximum output length.

The exact values must be recorded in the configuration.

---

# 13. Retrieval Parameters

For each benchmark run record:

```text id="svmqgf"
retrieval method
initial top-k
reranking candidate count
final context count
maximum retrieval iterations
```

Do not silently change these values between experiments.

---

# 14. Warm-Up Runs

Before measuring latency:

1. Load all required models.
2. Run one or more warm-up queries.
3. Clear stale timing measurements.
4. Begin benchmark timing.

Model download and model loading time should not be included in per-query inference latency.

Report model loading time separately.

---

# 15. Latency Measurement

Measure at least:

```text id="ezk5a3"
retrieval latency
reranking latency
generation latency
verification latency
total query latency
```

Use a monotonic timer.

For GPU operations, synchronize the GPU before measuring boundaries where necessary.

Report:

```text id="zq8o0x"
mean
median
P95
```

Avoid relying only on average latency.

---

# 16. Resource Measurement

Record:

```text id="8u8hlp"
peak GPU memory
peak system RAM
```

Measure during:

* indexing
* retrieval
* generation
* full query execution

Where supported, reset GPU memory statistics before each measured experiment.

---

# 17. Retrieval Evaluation

For every answerable question, compare retrieved chunks with ground-truth evidence.

Calculate:

```text id="20drk9"
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
```

Where graded relevance is available:

```text id="3k6pby"
nDCG@10
```

Ground-truth evidence may be defined by page or chunk.

The benchmark implementation must clearly document which representation is used.

---

# 18. Answer Evaluation

For every question, evaluate:

```text id="zqj1p2"
correctness
faithfulness
citation correctness
citation completeness
```

Use a combination of:

* deterministic checks where possible
* reference-based comparison
* model-based evaluation
* manual inspection of a sample

Do not rely entirely on an LLM judge.

---

# 19. LLM Judge Protocol

If an LLM judge is used, the judge receives only:

```text id="7fq2cd"
question
reference answer
generated answer
retrieved evidence
generated citations
```

The judge must not use web search or external retrieval.

The judging prompt must be fixed for the experiment.

The judge model and version must be recorded.

---

# 20. Judge Output

Prefer structured output:

```json id="v0n4u1"
{
  "correctness": 0,
  "faithfulness": 0,
  "citation_correctness": 0,
  "reason": "..."
}
```

Use a clearly defined scale.

Example:

```text id="k8m5yn"
0 = incorrect
1 = partially correct
2 = correct
```

The exact scale must remain consistent across experiments.

---

# 21. Abstention Evaluation

For unanswerable questions:

```text id="y0j3h7"
Correct abstention → success
Unsupported answer → failure
Incorrect refusal when evidence exists → failure
```

Calculate:

```text id="mmfl6m"
abstention accuracy
false-answer rate
```

A confident unsupported answer should be treated as a significant failure.

---

# 22. Citation Evaluation

For each generated citation determine:

```text id="1phw9h"
Does the cited page/chunk exist?
Does it support the associated claim?
Is an important claim missing a citation?
```

Report:

```text id="qmxzgd"
citation precision
citation completeness
```

Where exact claim-level evaluation is impractical, perform citation evaluation on a manually reviewed sample.

---

# 23. Error Analysis

Every benchmark run should preserve per-question results.

Failures should be categorized as:

```text id="t8b1tw"
retrieval failure
reranking failure
context selection failure
generation error
hallucination
citation error
abstention error
verification error
system/runtime failure
```

The purpose is diagnosis, not only producing a single score.

---

# 24. Ablation Protocol

Each major component should be removed independently.

Required ablations:

```text id="e0c5wv"
Agent planner OFF
Hybrid retrieval OFF
Reranker OFF
Retrieval refinement OFF
Evidence controller OFF
Verification OFF
```

Use the same evaluation dataset and configuration wherever possible.

---

# 25. Repeated Runs

For deterministic configurations, one complete run may be sufficient for initial development.

For final results, repeat experiments where practical.

Recommended:

```text id="5jvlv3"
3 runs
```

for configurations where model/runtime nondeterminism is significant.

Report mean and standard deviation when repeated runs are performed.

---

# 26. Statistical Reporting

Do not overstate small differences.

If two systems differ only slightly, report the actual values and discuss the uncertainty.

Example:

```text id="q6x6xj"
System A: 78.4%
System B: 79.1%
```

Do not describe this as a major improvement without supporting evidence.

---

# 27. Result Storage

Store experiment outputs under:

```text id="p3j19h"
results/
├── baseline/
├── hybrid/
├── agentic/
├── ablations/
└── final/
```

Each experiment should have:

```text id="1v4y4b"
config.json
results.json
per_question.json
summary.csv
```

Exact formats may be simplified if the implementation has a better consistent structure.

---

# 28. Final Comparison Table

The final notebook should produce a table similar to:

| System      | Recall@5 | MRR | Accuracy | Faithfulness | Citation | Abstention | P95 Latency |
| ----------- | -------: | --: | -------: | -----------: | -------: | ---------: | ----------: |
| Dense RAG   |          |     |          |              |          |            |             |
| Hybrid RAG  |          |     |          |              |          |            |             |
| Agentic RAG |          |     |          |              |          |            |             |

Also report:

```text
Peak GPU memory
Average retrieval iterations
Average model calls
```

for the agentic system.

---

# 29. Final Evaluation Checklist

Before submission verify:

* [ ] Same benchmark documents used.
* [ ] Same benchmark questions used.
* [ ] Ground truth manually reviewed.
* [ ] Configuration saved.
* [ ] Model versions recorded.
* [ ] Retrieval metrics calculated.
* [ ] Answer metrics calculated.
* [ ] Citation metrics calculated.
* [ ] Abstention tested.
* [ ] Latency measured.
* [ ] GPU memory measured.
* [ ] Ablations completed.
* [ ] Per-question failures inspected.
* [ ] Results reproducible from a fresh Colab runtime.

---

# 30. Interpretation Rule

Benchmark results must be reported descriptively.

The project should explain:

* which components improved which metrics,
* where performance degraded,
* what computational cost was introduced,
* which question types remain difficult,
* and what limitations remain.

Do not hide failures behind a single aggregate score.
