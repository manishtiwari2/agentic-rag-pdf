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
