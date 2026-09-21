# Design Decisions

## Purpose

This document records important architectural and engineering decisions for the Local Agentic RAG project.

Each decision should capture:

* what was chosen
* why it was chosen
* alternatives considered
* trade-offs
* how the decision will be validated

Decisions should be updated when benchmark evidence justifies a change.

---

# DD-001 — Modular Architecture

**Status:** Accepted

### Decision

Separate the system into independent modules:

```text
ingestion
chunking
embeddings
retrieval
reranking
agents
generation
evaluation
```

### Reason

The project needs to compare components independently and perform ablation experiments.

### Alternative

One monolithic notebook implementation.

### Trade-off

More files and interfaces, but significantly easier testing and experimentation.

---

# DD-002 — Notebook as Entry Point

**Status:** Accepted

### Decision

The Google Colab notebook is the primary user-facing entry point.

Core implementation should live under `src/`.

### Reason

The deliverable must be easy to run in Colab while remaining maintainable.

### Trade-off

Requires some duplication between notebook demonstration and library code.

---

# DD-003 — Local-Only Inference

**Status:** Accepted

### Decision

All generation, embedding, and reranking inference must run locally.

### Reason

The project explicitly targets free Colab and prohibits paid/cloud LLM inference.

### Alternative

Cloud APIs.

### Rejected because

They violate the project constraints.

---

# DD-004 — Small Local Generation Model

**Status:** Accepted

### Decision

Prefer a small instruction-tuned model capable of practical local inference.

Initial candidates:

```text
Qwen3-4B-Instruct-2507
Qwen2.5-3B-Instruct
Qwen2.5-1.5B-Instruct
```

### Reason

Free-Colab compatibility is a hard constraint.

### Validation

Compare answer quality, faithfulness, latency and peak VRAM.

---

# DD-005 — Embedding Model

**Status:** Experimental

### Initial candidate

```text
BAAI/bge-m3
```

### Reason

Designed for retrieval and supports broad text retrieval use cases.

### Validation

Compare retrieval recall and latency against any smaller candidate introduced later.

The final embedding model must be selected using benchmark evidence.

---

# DD-006 — Reranking

**Status:** Experimental

### Initial candidate

```text
BAAI/bge-reranker-v2-m3
```

### Reason

Reranking may improve retrieval precision after initial candidate generation.

### Validation

Compare:

```text
retrieval without reranking
vs.
retrieval with reranking
```

Measure both quality and latency.

---

# DD-007 — Dense + Lexical Retrieval

**Status:** Accepted for experimentation

### Decision

Test both semantic and lexical retrieval.

### Reason

Dense retrieval is strong for semantic similarity, while lexical retrieval can help with:

* exact terms
* names
* identifiers
* numbers
* technical terminology

### Validation

Compare dense-only, lexical-only where practical, and hybrid retrieval.

---

# DD-008 — Reciprocal Rank Fusion

**Status:** Initial candidate

### Decision

Use Reciprocal Rank Fusion as the initial deterministic method for combining retrieval results.

### Reason

It is simple, model-independent and easy to benchmark.

### Alternative

Learned score fusion.

### Trade-off

RRF may not be optimal but provides a strong, interpretable baseline.

---

# DD-009 — Structure-Aware Chunking

**Status:** Accepted

### Decision

Prefer chunks based on document structure and semantic boundaries instead of blindly splitting every N characters.

### Reason

PDFs contain:

* headings
* paragraphs
* sections
* lists
* tables

Preserving these boundaries can improve retrieval context.

### Trade-off

Parsing is more complicated than fixed-size chunking.

### Validation

Compare retrieval performance against a simple fixed-size baseline.

---

# DD-010 — Page Metadata Preservation

**Status:** Accepted

### Decision

Every chunk must retain its source page number.

### Reason

The final answer must provide useful citations.

### Required metadata

```text
document_id
page_number
chunk_id
section
```

This metadata must survive the complete retrieval pipeline.

---

# DD-011 — Explicit Agent State

**Status:** Accepted

### Decision

Represent agent execution using explicit state rather than hidden conversational state.

### Reason

Explicit state improves:

* debugging
* reproducibility
* observability
* testing

Example:

```text
query
query_type
retrieval_strategy
retrieved_chunks
evidence_status
iteration
draft_answer
verification_status
```

---

# DD-012 — Bounded Agent Loop

**Status:** Accepted

### Decision

Agentic retrieval must have a hard maximum number of iterations.

Initial value:

```text
MAX_RETRIEVAL_ITERATIONS = 2
```

### Reason

Prevents:

* infinite loops
* unpredictable latency
* excessive model calls

The value may be changed based on benchmark results.

---

# DD-013 — Baseline Before Agentic RAG

**Status:** Accepted

### Decision

Implement a simple dense RAG baseline before implementing the agentic system.

### Reason

Without a baseline, the value of agentic complexity cannot be measured.

---

# DD-014 — Evidence-First Generation

**Status:** Accepted

### Decision

The generator receives retrieved evidence explicitly and is instructed to answer from that evidence.

### Reason

The primary goal is grounded document QA rather than general chatbot behaviour.

### Failure behaviour

If evidence is insufficient, the system should abstain.

---

# DD-015 — Verification After Generation

**Status:** Experimental

### Decision

Test a verification stage that checks whether generated claims are supported by retrieved evidence.

### Reason

Generation quality alone does not guarantee grounding.

### Validation

Compare:

```text
generation
vs.
generation + verification
```

Measure:

* faithfulness
* hallucination rate
* citation correctness
* latency

---

# DD-016 — No External Knowledge During QA

**Status:** Accepted

### Decision

The answer-generation pipeline must not use web search or external knowledge retrieval.

### Reason

The benchmark evaluates understanding of the uploaded document.

### Exception

External resources may be used during development to obtain libraries, model documentation or benchmark metadata.

They must not supply answer content during evaluation.

---

# DD-017 — Configuration-Driven System

**Status:** Accepted

### Decision

Model names and important parameters must be configurable.

Configuration includes:

```text
models
chunk size
chunk overlap
top-k
rerank-k
retrieval strategy
temperature
max tokens
agent iteration limit
```

### Reason

Enables controlled experiments without modifying implementation code.

---

# DD-018 — Deterministic Components Where Possible

**Status:** Accepted

### Decision

Use normal deterministic code for deterministic operations.

Examples:

```text
ranking
metadata management
score fusion
citation formatting
configuration
result storage
```

Use LLMs only for tasks requiring semantic reasoning.

### Reason

Improves reliability, speed and debuggability.

---

# DD-019 — Evaluation Before Optimization

**Status:** Accepted

### Decision

Do not optimize components based only on intuition.

Changes should be evaluated against the benchmark.

### Reason

RAG systems often have non-obvious interactions between:

```text
chunking
retrieval
reranking
context size
generation
```

---

# DD-020 — Resource-Constrained Design

**Status:** Accepted

### Decision

Treat GPU memory and latency as first-class metrics.

### Reason

A theoretically stronger system that cannot reliably run on free Colab does not satisfy the project objective.

Every final experiment should report:

```text
peak VRAM
latency
model configuration
```

---

# DD-021 — Error Analysis

**Status:** Accepted

### Decision

Benchmarking must preserve per-question results.

### Reason

Aggregate scores cannot identify whether a failure came from:

```text
retrieval
reranking
context selection
generation
citation
verification
```

Error categories will guide future improvements.

---

# DD-022 — No Unnecessary Frameworks

**Status:** Accepted

### Decision

Use frameworks only when they provide clear value.

Prefer lightweight Python components where possible.

### Reason

The project should remain:

* understandable
* reproducible
* easy to run in Colab
* easy to debug

The implementation must not depend on a framework merely to call an LLM or perform simple retrieval.

---

# DD-023 — Final Decisions Must Be Evidence-Based

**Status:** Accepted

Initial choices are hypotheses.

After experiments, update this document with:

```text
Decision
Benchmark evidence
Observed trade-off
Final configuration
```

No component should remain in the final architecture solely because it was part of the original design.

---

# DD-024 — Qwen2.5-3B Cannot Be the Default Generator

**Status:** Accepted

### Decision

`Qwen2.5-3B-Instruct` may be benchmarked but must not be the default model.
The default generator must be Apache-2.0: `Qwen3-4B-Instruct-2507`, or
`Qwen2.5-1.5B-Instruct` in the low-memory configuration.

### Reason

License verification (`MODEL_SELECTION.md` 11.1, checked 2026-09-20) found
that `Qwen2.5-3B-Instruct` is released under `qwen-research`, which is
research/non-commercial only. Every other candidate in the pool is Apache-2.0
or MIT.

The project presents itself as open source. A default configuration that
downloads a research-only model hands every downstream user a licensing
problem they did not choose and probably will not notice.

### Trade-off

If the 3B model turns out to score best, the default is deliberately not the
highest-scoring option. The notebook must state that trade-off rather than
quietly selecting on score alone.

### Note

The restriction is per-checkpoint, not per-vendor. Other Qwen2.5 sizes are
Apache-2.0. Licences must be checked per model, not inferred from the family.

---

# DD-025 — Retrieval Metrics Are Scored at Page Level

**Status:** Accepted

### Decision

Ground-truth evidence is annotated and scored by **page**, not chunk ID.
`evidence_chunk_ids` remains an optional secondary annotation.

### Reason

Chunk IDs are a function of the chunking configuration. Scoring against them
would change the answer key whenever the chunker changes, making the
structure-aware vs. fixed-size comparison in DD-009 meaningless.

Page numbers are stable across configurations and are the unit the system
cites, so retrieval metrics and citation metrics share a scale.

### Consequence

A chunk spanning a page break counts as a hit for every page it covers.

---

# DD-026 — Both "Any" and "All" Recall Are Reported

**Status:** Accepted

### Decision

Report `Recall@K` (any gold page retrieved) and `Full-Recall@K` (all gold
pages retrieved). `Full-Recall@5` is the primary metric for `multi_hop` and
`comparison` questions.

### Reason

They are identical for single-evidence questions and diverge sharply on
multi-hop ones. A system that reliably finds one hop and never the second
scores 1.0 on the first metric and 0.0 on the second. Reporting only the
former would overstate exactly the capability the agentic loop is supposed to
provide.

---

# DD-027 — The Judge Is the Generator, and This Is Disclosed

**Status:** Accepted, with known limitation

### Decision

The LLM judge defaults to the generation model, because a second independent
judge does not fit in the Colab memory budget. Every judged metric is reported
beside a deterministic one, judge/deterministic disagreement is reported as a
number, and the disagreements are manually inspected.

### Reason

Self-grading biases judged scores upward and unevenly. The bias cannot be
removed under the local-only constraint, so it is measured and disclosed
instead of ignored.

### Alternative

A separate judge model, where hardware allows. The change in conclusions
between the two is the measured size of the effect.

---

# DD-028 — Confidence Intervals and Paired Tests Are Mandatory

**Status:** Accepted

### Decision

Every headline figure carries a bootstrap 95% CI. Every system-vs-baseline
claim uses a paired bootstrap over per-question differences. If the CI of the
difference includes zero, the result is reported as "no significant
difference", whatever the point estimate.

### Reason

EVALUATION_PROTOCOL section 26 already forbade overstating small differences
but named no method, which made the rule unenforceable. With 75 questions the
sampling error is roughly ±5 points — larger than most differences the
ablation study will produce.

---

# DD-029 — Unanswerable Questions Count Toward Headline Accuracy

**Status:** Accepted

### Decision

Unanswerable questions score 1 for abstaining and 0 for answering, and are
included in the headline accuracy figure. Accuracy is always reported with its
breakdown (all / answerable / abstention) and with the denominator for each.

### Reason

Excluding them would let a system with a serious hallucination problem post a
good headline number, since the questions that expose the problem would not be
counted.

### Trade-off

The headline figure mixes two scoring rules, which is why the breakdown is
mandatory rather than optional.
