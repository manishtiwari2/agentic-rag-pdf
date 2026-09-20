# Reference Notebook Audit

## 1. Reference

The initial reference for this project is:

`atmabodha/OpenNLP/AI for PDFs/Local_Agentic_RAG_System_PDF.ipynb`

This document records useful ideas from the reference implementation and identifies what should be retained, redesigned, or rejected for this project.

The reference is treated as **inspiration and prior art**, not as the implementation specification.

---

# 2. Useful Ideas to Retain

The reference demonstrates the general idea of combining:

```text id="4r5my6"
PDF processing
↓
chunking
↓
embeddings
↓
retrieval
↓
LLM reasoning
```

These are the core building blocks required by this project.

Useful concepts should be extracted into independent modules rather than copied into one large notebook.

---

# 3. Local-First Requirement

The final project has a stricter constraint than many general RAG examples:

```text
All inference must be local.
```

Therefore, any reference implementation using:

* OpenAI API
* Anthropic API
* Gemini API
* hosted embeddings
* hosted rerankers

must not become part of the final inference pipeline.

Local/open-weight replacements must be used.

---

# 4. Notebook vs Production Structure

The reference is useful as a notebook-oriented demonstration.

For this project, the notebook should remain the **entry point**, but core logic should be modular.

Target structure:

```text id="p7w8aw"
src/
├── ingestion/
├── chunking/
├── retrieval/
├── reranking/
├── agents/
├── generation/
└── evaluation/
```

The final notebook should orchestrate these components rather than contain hundreds of duplicated implementation cells.

---

# 5. Document Processing

The PDF processing layer should preserve page-level metadata.

Required information:

```text id="jyv8do"
document_id
page_number
text
section where available
```

This is more important for this project because the final system must provide citations.

If the reference parser loses page information during conversion, that approach should not be copied unchanged.

---

# 6. Chunking

The reference approach demonstrates the importance of chunking before retrieval.

For this project, test at least:

```text id="4w9t4r"
fixed-size baseline
structure-aware chunking
```

Do not assume one chunking strategy is universally optimal.

Chunking must be evaluated through retrieval metrics.

---

# 7. Parent/Child Retrieval

Hierarchical retrieval is a potentially useful idea.

Conceptually:

```text id="7u3c3m"
small retrieval chunk
        ↓
find relevant location
        ↓
larger context chunk
        ↓
generation
```

Potential advantage:

* precise retrieval
* richer generation context

Potential disadvantages:

* additional implementation complexity
* additional storage
* more complicated evaluation

This should be considered as an experiment rather than automatically included.

---

# 8. Hybrid Retrieval

Hybrid retrieval is worth retaining as a major architectural candidate.

Conceptually:

```text id="8qk3r9"
semantic retrieval
+
lexical retrieval
        ↓
result fusion
```

Reason:

Dense retrieval may miss exact identifiers, numbers, names, and terminology.

Lexical retrieval can compensate for these cases.

The final fusion method should be deterministic and benchmarked.

---

# 9. Reranking

A reranking stage is useful after initial candidate retrieval.

Concept:

```text id="0p3j2y"
query
+
top-N candidates
        ↓
reranker
        ↓
top-K evidence
```

The reranker should be evaluated separately.

Do not include it merely because it is common in modern RAG systems.

---

# 10. Agentic Retrieval

The reference direction supports the idea of allowing the system to reassess retrieval.

This is important for the proposed architecture.

The final system should be able to perform:

```text id="h8x1lz"
initial retrieval
      ↓
evidence assessment
      ↓
insufficient?
      ↓
query refinement
      ↓
second retrieval
```

This is more meaningful than simply wrapping a normal RAG chain inside an agent framework.

---

# 11. Query Rewriting

Query rewriting can be useful when:

* the question is ambiguous
* terminology differs from the document
* the first retrieval fails
* the question requires decomposition

However, rewriting should not happen automatically for every query.

The planner should determine whether rewriting is useful.

This reduces unnecessary LLM calls.

---

# 12. Self-Correction

A useful pattern is:

```text id="8f9o2k"
retrieve
↓
evaluate evidence
↓
retry if necessary
```

This should become part of the agentic experiment.

Important constraint:

```text
MAX_RETRIEVAL_ITERATIONS = 2
```

or another benchmark-validated small limit.

Unlimited self-correction is unacceptable for the Colab target.

---

# 13. Agent Framework

The reference ecosystem uses agent orchestration frameworks for graph-based workflows.

A framework may be used if it provides meaningful value.

However, the final implementation should first determine whether a small explicit state machine is sufficient.

Preferred starting point:

```text id="u8s8ms"
Python state
+
explicit routing functions
```

Introduce a framework only if it improves:

* readability
* reliability
* observability
* workflow management

without adding unnecessary dependency complexity.

---

# 14. Retrieval Tool Interface

The agent should interact with retrieval through a narrow interface.

Conceptually:

```python id="n1r2x0"
retrieve(
    query,
    strategy,
    top_k
) -> list[Chunk]
```

This allows the agent to change retrieval behaviour without knowing how the vector index or lexical search is implemented.

---

# 15. Structured Agent Outputs

Agent decisions should use structured schemas.

Example:

```text id="z6grh8"
{
    "query_type": "...",
    "strategy": "...",
    "rewrite": "...",
    "needs_more_retrieval": true
}
```

Avoid parsing arbitrary natural-language agent responses wherever possible.

---

# 16. Evidence Evaluation

The reference direction motivates evaluating retrieved evidence before final generation.

This should become an explicit component:

```text id="3zup6a"
retrieved evidence
        ↓
evidence controller
        ↓
sufficient / insufficient
```

This is a key difference between a basic RAG chain and the proposed agentic pipeline.

---

# 17. Final Answer Grounding

The answer generation stage should receive:

```text id="u8q8ce"
question
+
selected evidence
+
source metadata
```

The model should not be encouraged to fill missing information using general knowledge.

If the evidence is insufficient, the system should abstain.

---

# 18. Citation Design

Citations should be based on preserved document metadata.

Preferred citation information:

```text id="dbn9x2"
document
page
chunk
```

Example:

```text
[Source: document_01.pdf, p. 12]
```

The exact presentation can be improved later.

Citation generation should be deterministic where possible rather than asking the model to invent source identifiers.

---

# 19. UI

A simple Gradio interface may be appropriate for the final Colab demonstration.

The UI should provide:

```text id="3j7yib"
PDF upload
question input
answer
citations
optional evidence display
```

Do not allow UI complexity to affect the core RAG architecture.

---

# 20. Storage

The reference ecosystem demonstrates local vector storage.

For the constrained project, begin with a lightweight local vector index such as FAISS.

Requirements:

* local
* reproducible
* easy to install in Colab
* fast enough for benchmark documents

A heavier vector database should not be introduced unless it provides a measurable benefit.

---

# 21. Evaluation Gap to Address

Tutorial-style RAG implementations often demonstrate successful queries without rigorous comparison.

This project must go further.

Required:

```text id="5w9pzz"
baseline
vs.
hybrid retrieval
vs.
agentic pipeline
```

with:

* retrieval metrics
* answer metrics
* citation metrics
* abstention metrics
* latency
* memory

---

# 22. Benchmark Gap to Address

A few manually selected demonstration questions are insufficient.

The final project needs a fixed benchmark containing:

```text id="f8c4wx"
factual
definition
explanation
comparison
numerical
table
multi-hop
unanswerable
ambiguous
```

This allows the system to be evaluated systematically.

---

# 23. Colab Constraint

The reference architecture must be filtered through the free-Colab constraint.

Before adopting a component ask:

```text id="4u5rza"
Does it fit in GPU memory?
Does it add significant latency?
Does it require an external service?
Does it complicate installation?
Does it improve benchmark results?
```

If the answer to the final question is no, remove it.

---

# 24. What NOT to Copy

Do not blindly copy:

* provider-specific cloud code
* unnecessary framework abstractions
* large monolithic notebook cells
* hard-coded model names
* hard-coded API keys
* unbounded agent loops
* unmeasured retrieval parameters
* UI-specific logic inside the RAG core

---

# 25. What Should Be Improved

The final implementation should improve upon the reference direction by providing:

```text id="frv8qt"
1. Explicit modular architecture
2. Local-only model stack
3. Configurable components
4. Bounded agent loops
5. Structured agent state
6. Page-level citations
7. Baseline comparisons
8. Ablation experiments
9. Quantitative benchmark
10. Resource measurements
11. Error analysis
12. Reproducible Colab execution
```

---

# 26. Reference-Informed Architecture

The useful concepts can therefore be summarized as:

```text id="4w70e6"
PDF
 ↓
Structure-aware parsing
 ↓
Chunking
 ↓
Dense + lexical retrieval
 ↓
Fusion
 ↓
Reranking
 ↓
Evidence assessment
 ↓
Query refinement when necessary
 ↓
Context selection
 ↓
Local LLM
 ↓
Verification
 ↓
Citations
```

This becomes the working architecture, subject to benchmark validation.

---

# 27. Final Principle

The reference implementation answers:

> “How can an Agentic RAG system for PDFs be constructed?”

This project must answer a stronger question:

> “Which architecture provides the best measurable document-QA performance under strict local-compute constraints, and which components actually contribute to that performance?”

The benchmark—not the reference implementation—should determine the final architecture.
