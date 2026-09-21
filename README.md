# Local Agentic RAG for PDF Question Answering

A local, open-source Agentic RAG system for asking questions about uploaded PDFs, designed to run on the free Google Colab GPU environment.

> **Status: Research / Active Development**
>
> This repository is currently being developed. Benchmark numbers and final model selections will be added only after the corresponding experiments have been run.

---

## 1. Project Goal

The goal is to build a practical PDF question-answering system that can:

* accept an uploaded PDF,
* understand its structure,
* retrieve relevant evidence,
* answer questions using only document evidence,
* provide page-level citations,
* recognize when evidence is insufficient,
* perform additional retrieval when necessary,
* and run entirely with local models.

The main research question is:

> **Can a carefully designed Agentic RAG pipeline provide better grounded PDF question answering than simpler RAG pipelines while remaining practical on free Google Colab GPU resources?**

The project therefore focuses on both **quality and efficiency**.

---

## 2. Why Agentic RAG?

A conventional RAG pipeline usually follows:

```text
Question
   ↓
Retrieve
   ↓
Generate
```

This is simple and useful, but retrieval is fixed.

The proposed system introduces controlled decision-making:

```text
Question
   ↓
Query Planning
   ↓
Retrieval
   ↓
Reranking
   ↓
Evidence Assessment
   ↓
Enough Evidence?
   ├── Yes → Generate → Verify → Answer
   │
   └── No  → Refine Query → Retrieve Again
```

The agentic behaviour is intentionally bounded.

The project does not assume that adding agents automatically improves RAG. The benchmark must demonstrate whether the additional complexity provides measurable value.

---

## 3. Key Constraints

This project is designed around strict resource constraints.

### Compute

Target environment:

```text
Google Colab Free GPU
```

The implementation should not assume:

* A100/H100 GPUs
* large dedicated servers
* distributed inference
* permanent GPU availability

### Inference

All model inference must be local.

The final system must not require:

* OpenAI API
* Anthropic API
* Gemini API
* paid inference APIs
* hosted embedding APIs
* hosted reranking APIs

Open-weight models downloaded into the Colab runtime are allowed, subject to their licenses.

---

## 4. Planned Architecture

```text
                         PDF
                          │
                          ▼
                ┌──────────────────┐
                │ PDF Extraction   │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Semantic Chunking│
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Local Embeddings │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Local Vector     │
                │ Index             │
                └────────┬─────────┘
                         │
                      Question
                         ▼
                ┌──────────────────┐
                │ Query Planner    │
                └────────┬─────────┘
                         ▼
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Dense Retrieval       Lexical Retrieval
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌──────────────────┐
                │ Result Fusion    │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Local Reranker   │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Evidence Agent   │
                └────────┬─────────┘
                         ▼
                  Evidence enough?
                    /          \
                  No            Yes
                  │              │
                  ▼              ▼
             Refine Query    Generate Answer
                  │              │
                  └──────┐       ▼
                         │   Verify
                         │       │
                         └───────┤
                                 ▼
                         Answer + Citations
```

This is the planned architecture. Individual components may change based on benchmark evidence.

---

## 5. Planned Components

### PDF Processing

The ingestion layer will extract text while preserving page information.

Each retrieval chunk should retain metadata such as:

```text
document_id
page_number
chunk_id
section
```

This allows retrieved evidence to be traced back to the original PDF.

### Chunking

The project will investigate structure-aware chunking rather than relying exclusively on arbitrary fixed-size text windows.

A simple fixed-size strategy will remain available as a baseline.

### Retrieval

The initial retrieval design will investigate:

```text
Dense semantic retrieval
+
Lexical retrieval
```

The results can then be fused before reranking.

### Reranking

A local reranker may be used to improve the ordering of retrieved candidates.

Its contribution will be measured separately.

### Agentic Control

The agentic layer will handle decisions such as:

* query interpretation,
* retrieval strategy,
* evidence sufficiency,
* retrieval refinement,
* final verification.

The loop will have a hard iteration limit to control latency and model usage.

### Generation

The answer generator will receive the question and selected document evidence.

It should not depend on external web knowledge.

### Verification

The system will attempt to identify unsupported claims before returning the final response.

If evidence remains insufficient, the system should abstain rather than fabricate an answer.

---

## 6. Initial Model Candidates

The final models have **not yet been selected**.

Initial candidates include:

### Generation

```text
Qwen3-4B-Instruct-2507
Qwen2.5-3B-Instruct
Qwen2.5-1.5B-Instruct
```

### Embeddings

```text
BAAI/bge-m3
```

### Reranking

```text
BAAI/bge-reranker-v2-m3
```

These are experimental candidates, not claims that they are the final or best models for this project.

Selection will consider:

* answer quality,
* retrieval quality,
* grounding,
* citation quality,
* latency,
* GPU memory,
* reliability,
* licensing,
* Colab compatibility.

See [`MODEL_SELECTION.md`](MODEL_SELECTION.md).

---

## 7. Baselines

The project will compare progressively stronger systems.

### Baseline 1 — Dense RAG

```text
PDF
 ↓
Chunk
 ↓
Dense Retrieval
 ↓
LLM
```

### Baseline 2 — Hybrid + Reranking

```text
PDF
 ↓
Chunk
 ↓
Dense + Lexical Retrieval
 ↓
Fusion
 ↓
Reranking
 ↓
LLM
```

### Proposed Agentic RAG

```text
Query Planning
 ↓
Adaptive Retrieval
 ↓
Fusion
 ↓
Reranking
 ↓
Evidence Assessment
 ↓
Optional Retrieval Refinement
 ↓
Generation
 ↓
Verification
```

The purpose of these baselines is to measure whether each additional component actually contributes value.

---

## 8. Benchmark

The benchmark will contain multiple PDFs and questions covering different question types.

Planned categories include:

* factual
* definition
* explanation
* comparison
* numerical
* table-based
* multi-hop
* ambiguous
* unanswerable
* summary

Initial target:

```text
5–10 PDFs
50–100 questions
```

The benchmark will include ground-truth answers and supporting evidence locations.

Benchmark details are documented in:

```text
BENCHMARK_SPEC.md
```

---

## 9. Evaluation

The project will evaluate both retrieval and end-to-end answer quality.

### Retrieval

Planned metrics:

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
nDCG@10 where applicable
```

### Answer Quality

Planned metrics:

```text
Answer correctness
Faithfulness
Citation correctness
Citation completeness
Abstention accuracy
```

### Performance

The system will also measure:

```text
PDF ingestion time
Indexing time
Retrieval latency
Reranking latency
Generation latency
Total query latency
Peak GPU memory
Peak system RAM
```

No benchmark result is claimed until it has actually been measured.

---

## 10. Ablation Studies

The project will investigate the contribution of major components.

Planned ablations include:

```text
Agent planner OFF
Hybrid retrieval OFF
Reranker OFF
Retrieval refinement OFF
Evidence controller OFF
Verification OFF
```

The goal is to determine which components provide measurable improvements and which add complexity without sufficient benefit.

---

## 11. Repository Structure

```text
agentic-pdf-rag/
│
├── README.md
├── PROJECT_SPEC.md
├── ARCHITECTURE.md
├── MODEL_SELECTION.md
├── BENCHMARK_SPEC.md
├── EVALUATION_PROTOCOL.md
├── DESIGN_DECISIONS.md
│
├── references/
│   ├── reference_notebook.ipynb
│   └── reference_notes.md
│
├── benchmark/
│   ├── README.md
│   ├── documents/
│   ├── questions.json
│   └── ground_truth.json
│
├── notebooks/
│   └── agentic_pdf_rag.ipynb
│
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── retrieval/
│   ├── reranking/
│   ├── agents/
│   ├── generation/
│   └── evaluation/
│
├── tests/
│
└── results/
    ├── baseline/
    ├── hybrid/
    ├── agentic/
    ├── ablations/
    └── final/
```

The exact structure may evolve during implementation.

---

## 12. Colab Notebook

The final notebook will be the primary entry point for users who want to run the project.

Planned notebook flow:

```text
1. Environment setup
2. Install dependencies
3. Load local models
4. Upload PDF
5. Parse PDF
6. Build chunks
7. Build retrieval index
8. Run baseline RAG
9. Run Agentic RAG
10. Ask questions
11. Display citations
12. Run benchmark
13. Show metrics
14. Show ablation results
15. Show resource usage
```

The notebook should work from a fresh Colab runtime as far as the available free hardware permits.

---

## 13. Quick Start

The final quick-start experience is intended to be:

```text
Open the Colab notebook
        ↓
Run setup cells
        ↓
Upload a PDF
        ↓
Wait for indexing
        ↓
Ask a question
        ↓
Receive answer + citations
```

The exact installation and execution commands will be documented after implementation stabilizes.

At the current stage, the repository is **not yet claiming that the complete pipeline is runnable**.

---

## 14. Example Expected Interaction

For an uploaded document:

```text
Question:
What was the main improvement introduced by the proposed method?
```

The system should return something conceptually similar to:

```text
The proposed method improves X by introducing Y.

Evidence:
- Page 4
- Page 9

The answer is based on the retrieved sections describing the
methodology and experimental results.
```

The exact answer depends on the uploaded document.

For unsupported questions, the system should respond along the lines of:

```text
I could not find sufficient evidence in the uploaded document
to answer this question.
```

The wording may change during implementation.

---

## 15. Design Principles

### Evidence over fluency

A fluent unsupported answer is a failure.

### Local first

The system should remain useful without proprietary inference APIs.

### Simple where possible

Deterministic operations should remain deterministic.

### Agents only where useful

Agentic behaviour should solve a real decision problem.

### Benchmark before claiming improvement

Architectural complexity must be justified experimentally.

### Reproducibility

Model versions, configuration, benchmark data and evaluation procedures should be recorded.

### Explicit limitations

The repository should clearly document known failure modes instead of presenting the system as universally reliable.

---

## 16. Current Limitations

At the current development stage:

* final model selection is incomplete,
* benchmark data is still being constructed,
* quantitative results are not yet available,
* the final agent workflow is not yet validated,
* PDF extraction quality across arbitrary PDFs has not yet been fully evaluated,
* scanned/image-only PDFs may require additional OCR/multimodal processing,
* free-Colab GPU availability and memory can vary between sessions.

These limitations will be updated as development progresses.

---

## 17. Roadmap

### Phase 1 — Specification

* [x] Define project requirements
* [x] Define architecture
* [x] Define model-selection process
* [x] Define benchmark
* [x] Define evaluation protocol
* [x] Audit reference implementation

### Phase 2 — Baseline

* [ ] PDF ingestion
* [ ] Chunking
* [ ] Embeddings
* [ ] Vector retrieval
* [ ] Local generation
* [ ] Baseline benchmark

### Phase 3 — Retrieval Improvements

* [ ] Lexical retrieval
* [ ] Hybrid fusion
* [ ] Reranking
* [ ] Retrieval evaluation

### Phase 4 — Agentic RAG

* [ ] Query planner
* [ ] Evidence controller
* [ ] Retrieval refinement
* [ ] Context selection
* [ ] Verification

### Phase 5 — Evaluation

* [ ] Full benchmark
* [ ] Ablation experiments
* [ ] Error analysis
* [ ] Latency analysis
* [ ] GPU/RAM analysis

### Phase 6 — Finalization

* [ ] Colab notebook
* [ ] Documentation
* [ ] Reproducibility testing
* [ ] Final benchmark
* [ ] Final release

---

## 18. Research Questions

The project will primarily investigate:

### RQ1

Does hybrid retrieval improve document evidence retrieval compared with dense retrieval alone?

### RQ2

Does reranking improve final answer quality enough to justify its computational cost?

### RQ3

Does adaptive/agentic retrieval improve difficult and multi-hop questions?

### RQ4

Does evidence verification reduce unsupported answers?

### RQ5

Can these improvements be achieved within practical free-Colab resource limits?

These questions should be answered using measured experiments rather than assumptions.

---

## 19. Project Status

**Current status: Architecture and evaluation design**

The implementation is intentionally not considered complete until:

* the benchmark exists,
* the baseline works,
* the agentic pipeline works,
* comparative experiments are complete,
* resource usage is measured,
* and the final Colab notebook is reproducible.

Benchmark numbers will be added only after running the experiments.

---

## 20. License

The project code will be released under an open-source license.

The final repository must separately document the licenses of:

* source PDFs,
* model weights,
* datasets,
* third-party libraries.

Model and document licenses must not be assumed from the project code license.

---

## 21. Acknowledgements

This project was initially inspired by existing local/agentic PDF RAG implementations and is intended to build upon those ideas through a more explicit focus on:

* local inference,
* constrained hardware,
* reproducible benchmarking,
* modular engineering,
* citation grounding,
* and ablation-based evaluation.

Reference implementations should be credited appropriately in the final repository.
