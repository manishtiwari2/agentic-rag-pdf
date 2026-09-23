# System Architecture

## 1. Purpose

This document defines the technical architecture of the local Agentic RAG system described in `PROJECT_SPEC.md`.

The architecture is designed around four requirements:

1. Local inference on free Google Colab GPU.
2. Reliable question answering over arbitrary PDFs.
3. Measurable retrieval and generation quality.
4. A genuine but controlled agentic decision loop.

The architecture should remain modular so that individual components can be replaced and benchmarked independently.

---

# 2. High-Level System

```text
                        ┌──────────────────┐
                        │    PDF Upload    │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Document Parser  │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Document Model   │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Semantic Chunker │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Embedding Model  │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Retrieval Index  │
                        └────────┬─────────┘
                                 │
                           User Question
                                 ↓
                        ┌──────────────────┐
                        │  Query Planner   │
                        └────────┬─────────┘
                                 ↓
                    ┌────────────┴────────────┐
                    ↓                         ↓
             Dense Retrieval          Lexical Retrieval
                    │                         │
                    └────────────┬────────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Candidate Fusion │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │    Reranker      │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ Evidence Agent   │
                        └────────┬─────────┘
                                 ↓
                         Evidence sufficient?
                            /           \
                          NO             YES
                          ↓               ↓
                  Refine / Retrieve    Generator
                          │               │
                          └───────┐       ↓
                                  │  ┌─────────────┐
                                  └→ │ Verifier    │
                                     └──────┬──────┘
                                            ↓
                                      Final Answer
                                      + Citations
```

---

# 3. Core Components

The system consists of the following modules:

```text
src/
├── ingestion/
│   ├── parser.py
│   └── document.py
│
├── chunking/
│   └── chunker.py
│
├── retrieval/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── lexical.py
│   └── retriever.py
│
├── reranking/
│   └── reranker.py
│
├── agents/
│   ├── planner.py
│   ├── evidence.py
│   └── verifier.py
│
├── generation/
│   └── generator.py
│
├── evaluation/
│   ├── metrics.py
│   └── benchmark.py
│
└── pipeline.py
```

The exact file structure can change during implementation if a simpler organization is justified.

As built, `src/agents/` also holds `refinement.py` (query construction for a
further retrieval round), `state.py` (the explicit `AgentState`, section 17)
and `text.py` (the shared tokenizer and strict JSON reader). The agent loop is
`AgenticRAGPipeline` in `src/pipeline.py` (DD-050).

---

# 4. Document Representation

The parser should convert the PDF into a normalized internal representation.

Conceptually:

```python
Document
    document_id
    source_name
    pages: list[Page]

Page
    page_number
    text
    metadata

Chunk
    chunk_id
    document_id
    page_number
    section
    text
    metadata
```

The retrieval layer must not depend directly on the PDF parser.

This separation allows the parser to be replaced without changing retrieval or generation.

---

# 5. PDF Ingestion

Responsibilities:

* load PDF
* extract text
* preserve page numbers
* detect empty/problematic pages
* normalize obvious extraction artifacts
* produce the internal document representation

The ingestion layer should not:

* generate answers
* perform retrieval
* call the LLM
* make agent decisions

---

# 6. Chunking

The chunker converts pages into retrieval units.

Preferred strategy:

```text
document
    ↓
pages
    ↓
sections / paragraphs
    ↓
semantic chunks
```

The chunker should preserve:

* page number
* section title when available
* chunk ID
* document ID

Chunk size and overlap must be configurable.

The implementation should support experimentation with different chunking strategies.

---

# 7. Embedding Layer

The embedding layer converts chunks and queries into vectors.

Interface:

```python
embed_documents(chunks) -> vectors

embed_query(query) -> vector
```

The embedding implementation should hide the underlying model.

Changing the embedding model should not require modifications to retrieval logic.

---

# 8. Retrieval Layer

Retrieval should support multiple strategies.

Initial candidates:

```text
Dense semantic retrieval
+
Lexical retrieval
```

Dense retrieval captures semantic similarity.

Lexical retrieval provides robustness for:

* exact terminology
* names
* identifiers
* numbers
* technical phrases

The retrieval layer returns candidates with:

```text
chunk_id
text
metadata
retrieval_score
retrieval_source
```

---

# 9. Candidate Fusion

Results from different retrievers must be combined using deterministic logic.

Possible approaches include:

* Reciprocal Rank Fusion
* normalized score fusion
* weighted ranking

The initial implementation should prefer a simple deterministic method.

The fusion strategy must be benchmarkable.

---

# 10. Reranking

The reranker receives the user query and candidate chunks.

Input:

```text
query
candidate_chunks
```

Output:

```text
ranked_chunks
```

The reranker should improve precision before context is sent to the generation model.

The reranker must be optional so that its contribution can be measured through ablation experiments.

---

# 11. Query Planner Agent

The query planner is the first agentic component.

Its responsibility is to determine how the question should be handled.

Possible decisions:

```text
QUESTION_TYPE:
    factual
    definition
    comparison
    numerical
    multi-hop
    summary
    unanswerable/uncertain

RETRIEVAL_STRATEGY:
    dense
    lexical
    hybrid

QUERY_REWRITES:
    optional list

REQUIRES_ITERATION:
    yes/no
```

The planner should produce structured output.

It must not directly generate the final answer.

---

# 12. Evidence Controller

The evidence controller determines whether retrieved evidence is sufficient.

Inputs:

```text
original_query
retrieved_chunks
retrieval_scores
```

Output:

```text
sufficient: boolean
confidence: float
reason: string
next_action: answer | retrieve_again | abstain
```

If evidence appears insufficient, the controller may trigger another retrieval iteration.

A hard upper limit must exist on the number of retrieval iterations.

This prevents infinite agent loops.

---

# 13. Retrieval Refinement

If the first retrieval attempt is insufficient, the system may:

1. rewrite the query,
2. change retrieval strategy,
3. broaden or narrow retrieval,
4. retrieve additional candidates.

Example:

```text
Original query
      ↓
Initial retrieval
      ↓
Insufficient evidence
      ↓
Query refinement
      ↓
Second retrieval
      ↓
Evidence assessment
```

The system should not perform unlimited retries.

Default maximum:

```text
MAX_RETRIEVAL_ITERATIONS = 2
```

This value must remain configurable and benchmarkable.

---

# 14. Context Selection

The generator should not automatically receive every retrieved chunk.

A context selection step should:

* remove redundant chunks
* preserve relevant evidence
* respect model context limits
* maintain source metadata
* preserve page information

The final context should contain only the evidence necessary for answering the query.

---

# 15. Answer Generation

The generator receives:

```text
original question
selected evidence
source metadata
```

The prompt must explicitly require:

* answer only from supplied evidence
* no unsupported factual claims
* citations for claims
* explicit uncertainty when evidence is insufficient

The generator must not have access to external search.

---

# 16. Verification

The verification component checks whether the generated answer is supported by retrieved evidence.

Conceptually:

```text
answer
+
evidence
      ↓
Verifier
      ↓
SUPPORTED / UNSUPPORTED / UNCERTAIN
```

The verifier should identify unsupported claims where practical.

If verification fails, the system may:

* regenerate using stronger evidence,
* perform one additional retrieval attempt,
* abstain.

The maximum retry count must be bounded.

---

# 17. Agent State

The agent loop should maintain explicit state.

Conceptual state:

```python
AgentState:
    query
    query_type
    retrieval_strategy
    query_variants
    iteration
    retrieved_chunks
    reranked_chunks
    selected_context
    evidence_status
    draft_answer
    verification_status
    citations
```

The state should be inspectable for debugging and evaluation.

Avoid hidden state inside prompts or global variables.

---

# 18. Pipeline Orchestration

The top-level pipeline should expose a simple interface.

Example:

```python
pipeline = PDFRAGPipeline(config)

pipeline.index(pdf)

result = pipeline.ask(
    "What is the main contribution of the paper?"
)
```

The returned result should contain:

```python
{
    "answer": ...,
    "citations": ...,
    "evidence": ...,
    "metadata": ...
}
```

Metadata may include:

```text
latency
retrieval_iterations
retrieval_strategy
retrieved_chunk_ids
verification_status
```

---

# 19. Baseline Architecture

The baseline must bypass agentic decision making.

```text
query
 ↓
dense retrieval
 ↓
top-k chunks
 ↓
generator
 ↓
answer
```

This provides a reference point for measuring whether the additional agentic complexity is useful.

---

# 20. Agentic Architecture

The proposed system:

```text
query
 ↓
planner
 ↓
retrieval
 ↓
fusion
 ↓
reranking
 ↓
evidence controller
 ↓
 ├── sufficient → context → generation → verification
 │
 └── insufficient → refinement → retrieval
```

The agentic architecture must be compared against the baseline under the same benchmark.

---

# 21. Resource Management

Because the target is free Colab GPU:

* load only required models
* avoid loading multiple large generation models simultaneously
* use quantization where appropriate
* release unused GPU memory
* measure peak GPU memory
* keep retrieval models independent from generation models where possible
* avoid unnecessary model copies

Model configuration must be centralized.

---

# 22. Configuration

Configuration should be separated from implementation.

Example:

```python
Config(
    generation_model=...,
    embedding_model=...,
    reranker_model=...,
    chunk_size=...,
    chunk_overlap=...,
    top_k=...,
    rerank_k=...,
    max_iterations=...,
    temperature=...,
)
```

No hard-coded model names should be scattered throughout the codebase.

---

# 23. Observability

Each query should optionally produce a trace:

```text
Query
 ↓
Planner decision
 ↓
Retrieval strategy
 ↓
Candidates
 ↓
Reranking
 ↓
Evidence decision
 ↓
Generation
 ↓
Verification
 ↓
Final response
```

The trace is primarily for debugging and benchmark analysis.

The normal user interface should remain simple.

---

# 24. Failure Boundaries

Each component should fail explicitly.

Examples:

```text
PDF parsing failure
→ clear ingestion error

No extractable text
→ explain that the PDF may be scanned/image-only

Retrieval failure
→ return controlled failure

Insufficient evidence
→ abstain

Verifier failure
→ mark verification as unavailable rather than claiming success

GPU memory failure
→ provide actionable configuration guidance
```

---

# 25. Design Principle

The system should follow this rule:

> Use deterministic software for deterministic operations and local LLM agents only for decisions that require semantic reasoning.

Examples:

```text
Vector similarity        → deterministic retrieval code
Rank fusion              → deterministic code
Page tracking            → deterministic code
Query interpretation     → LLM agent
Evidence sufficiency     → LLM/model-based reasoning
Answer generation        → LLM
Grounding verification   → model-based verification
```

This keeps the system faster, easier to debug, and easier to benchmark.

---

# 26. Architecture Validation

Before implementation is considered complete, verify:

* every component has a clear responsibility,
* components can be independently tested,
* model choices can be changed through configuration,
* baseline and agentic pipelines share reusable infrastructure,
* agent loops are bounded,
* evidence and citations survive every pipeline stage,
* resource usage can be measured,
* benchmark instrumentation does not change answer behaviour.

Architecture changes should be documented in `DESIGN_DECISIONS.md`.
