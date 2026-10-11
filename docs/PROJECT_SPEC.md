# Local Agentic RAG for PDF Question Answering

## 1. Project Overview

Build an open-source, production-oriented **Agentic Retrieval-Augmented Generation (RAG) system** that allows a user to upload a PDF and ask questions about its contents.

The system must:

* Run entirely with locally hosted/open-weight models.
* Be executable on the free Google Colab GPU environment.
* Answer questions using evidence retrieved from the uploaded PDF.
* Provide page-level or chunk-level citations.
* Abstain when the document does not contain enough evidence.
* Include an agentic retrieval/control loop rather than being only a fixed retrieve-then-generate pipeline.
* Provide reproducible benchmark metrics.
* Be packaged as a clean, understandable Google Colab notebook.

The project is intended as an open-source engineering/ML project and should prioritize **correctness, reproducibility, modularity, observability, and measurable performance**.

---

## 2. Problem Statement

Given an arbitrary user-uploaded PDF:

```text
PDF + User Question
        ↓
Document Understanding
        ↓
Retrieval
        ↓
Evidence Selection
        ↓
Answer Generation
        ↓
Grounding Verification
        ↓
Answer + Citations
```

The system should answer only from information supported by the uploaded document.

If sufficient evidence cannot be found, the system should explicitly indicate that the answer cannot be established from the document instead of hallucinating.

---

## 3. Hard Constraints

### Compute

* Target environment: Google Colab free GPU.
* No assumption of an A100/H100 or large-memory GPU.
* Models must be small enough to load and run within practical free-Colab limits.
* GPU memory usage must be measured and documented.

### Model

All inference must be local.

Do not use:

* OpenAI API
* Anthropic API
* Gemini API
* OpenAI-hosted embeddings
* Cloud-hosted inference APIs
* Paid model APIs

Open-weight models downloaded into the Colab runtime are allowed.

### Document Source

The system's knowledge source is the uploaded PDF.

External web search must not be required for answering questions.

### Reproducibility

The notebook must specify:

* model names and versions
* quantization configuration
* embedding model
* chunking configuration
* retrieval configuration
* reranker configuration
* generation parameters
* benchmark dataset
* evaluation methodology

---

## 4. Functional Requirements

### PDF ingestion

The system must:

1. Accept a PDF uploaded by the user.
2. Extract text while preserving page information.
3. Handle multi-page documents.
4. Preserve useful metadata such as:

   * document ID
   * page number
   * section/heading when available
   * chunk ID

The design should allow future support for more sophisticated PDF parsing without rewriting the retrieval system.

### Chunking

Chunks should preserve semantic context where possible.

The implementation should avoid blindly splitting documents at arbitrary character boundaries.

Chunk metadata must remain attached to the source text.

### Retrieval

The system should support at least:

* dense semantic retrieval
* lexical/keyword retrieval or another complementary retrieval mechanism
* candidate fusion
* optional reranking

The exact implementation must be benchmarked rather than assumed to be optimal.

### Question answering

The answer generator must:

* receive retrieved evidence explicitly
* avoid relying on undocumented model knowledge
* produce concise answers
* cite supporting pages/chunks
* distinguish evidence from inference
* abstain when evidence is insufficient

### Agentic behaviour

The system must contain an explicit decision/control mechanism.

At minimum, the system should be capable of deciding:

1. how to retrieve evidence for a query,
2. whether retrieved evidence is sufficient,
3. whether another retrieval attempt is necessary,
4. whether the available evidence supports an answer.

Agentic behaviour must have measurable value.

A simple fixed:

```text
retrieve → generate
```

pipeline must therefore be implemented as a baseline for comparison.

---

## 5. Proposed High-Level Architecture

```text
                    ┌─────────────────┐
                    │   PDF Upload    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ PDF Extraction  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Semantic        │
                    │ Chunking        │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Embeddings      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Retrieval Index │
                    └────────┬────────┘
                             │
                        User Query
                             ↓
                    ┌─────────────────┐
                    │ Query Planner   │
                    └────────┬────────┘
                             ↓
               ┌─────────────┴─────────────┐
               ↓                           ↓
        Dense Retrieval             Lexical Retrieval
               │                           │
               └─────────────┬─────────────┘
                             ↓
                    ┌─────────────────┐
                    │ Candidate Fusion│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Local Reranker  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Evidence        │
                    │ Controller      │
                    └───────┬─────────┘
                            │
                    sufficient evidence?
                       /           \
                     no             yes
                     ↓               ↓
              Retrieve again      Generate
                     │               │
                     └───────┐       ↓
                             │  ┌──────────────┐
                             └→ │ Verification │
                                └──────┬───────┘
                                       ↓
                                Answer + Citations
```

This architecture is a starting hypothesis. Individual components must be validated experimentally.

---

## 6. Baseline and Proposed Systems

The project must contain multiple configurations.

### Baseline A — Dense RAG

```text
PDF
 ↓
Chunk
 ↓
Embedding
 ↓
Vector Search
 ↓
LLM
```

### Baseline B — Retrieval-enhanced RAG

```text
PDF
 ↓
Chunk
 ↓
Hybrid Retrieval
 ↓
Reranker
 ↓
LLM
```

### Proposed — Agentic RAG

```text
PDF
 ↓
Query Planning
 ↓
Adaptive Retrieval
 ↓
Candidate Fusion
 ↓
Reranking
 ↓
Evidence Assessment
 ↓
Optional Retrieval Refinement
 ↓
Answer Generation
 ↓
Grounding Verification
 ↓
Answer + Citations
```

The benchmark must compare these configurations.

---

## 7. Model Requirements

Candidate generation models should be small enough for free Colab GPU execution.

Initial candidates may include:

* Qwen3-4B-Instruct-2507
* Qwen2.5-3B-Instruct
* Qwen2.5-1.5B-Instruct

Candidate embedding models may include:

* BAAI/bge-m3
* other open retrieval models that satisfy the compute and licensing requirements

Candidate rerankers may include:

* BAAI/bge-reranker-v2-m3
* other sufficiently small open rerankers

These are candidates, not predetermined final choices.

The final model configuration must be selected using:

* answer quality
* retrieval quality
* memory usage
* latency
* reliability
* licensing
* Colab compatibility

---

## 8. Quality Requirements

The system should prioritize:

1. Grounded answers
2. Correct retrieval
3. Correct citations
4. Reliable abstention
5. Reasonable latency
6. Low memory usage
7. Reproducibility

The project must not optimize solely for answer fluency.

A fluent hallucination is considered a failure.

---

## 9. Observability

The pipeline should record enough information to diagnose failures.

For each query, where practical, record:

* query
* query classification/plan
* retrieval strategy
* retrieved chunk IDs
* retrieval scores
* reranker scores
* selected evidence
* number of retrieval iterations
* generation latency
* verification result
* final answer
* citations

Debug logging should be separable from normal user-facing output.

---

## 10. Failure Handling

The system must handle:

* empty PDF
* text extraction failure
* very short PDF
* very long PDF
* poor retrieval
* insufficient evidence
* contradictory evidence
* malformed model output
* model loading failure
* GPU memory errors

Failures should produce understandable messages rather than silent errors.

---

## 11. Scope Boundaries

The initial project does not require:

* web search
* multi-user authentication
* cloud deployment
* model fine-tuning
* distributed inference
* persistent production databases
* multimodal image reasoning

These may be future extensions but should not complicate the core Colab implementation.

---

## 12. Deliverable

The primary deliverable is:

```text
notebooks/agentic_pdf_rag.ipynb
```

The notebook must contain:

1. Environment setup
2. Dependency installation
3. Model loading
4. PDF upload
5. PDF parsing
6. Chunking
7. Index construction
8. Baseline RAG
9. Agentic RAG
10. Example queries
11. Benchmark execution
12. Evaluation metrics
13. Ablation results
14. Latency/memory measurements
15. Final demonstration

Supporting implementation should be modularized under:

```text
src/
```

The notebook should act as the reproducible entry point rather than containing the entire implementation as one monolithic code block.

---

## 13. Engineering Principles

The implementation should follow these principles:

### Simple where possible

Do not introduce an agent, framework, or model where deterministic code is sufficient.

### Modular

Each major subsystem should have a clear interface.

### Testable

Important components should be independently testable.

### Measurable

Major architectural decisions should be supported by benchmark results.

### Reproducible

A fresh Colab runtime should be able to reproduce the documented experiment.

### Evidence-first

Retrieved document evidence takes precedence over model prior knowledge.

### Fail safely

When evidence is insufficient, abstain rather than fabricate.

---

## 14. Success Criteria

The project is considered successful when:

* it runs within the intended free-Colab environment,
* a user can upload an unseen PDF,
* the system can answer diverse questions about that PDF,
* answers include useful citations,
* unsupported questions are handled appropriately,
* the agentic pipeline can be compared against simpler baselines,
* benchmark results are reproducible,
* latency and resource usage are reported,
* the architecture and design decisions are understandable from the repository,
* the final notebook can be executed by another user without requiring proprietary APIs.

The goal is not to maximize the number of components.

The goal is to demonstrate that a carefully designed local agentic RAG system provides **measurable improvements in document question answering while remaining practical on constrained hardware**.
