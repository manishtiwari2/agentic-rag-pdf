# Local Agentic RAG for PDF Question Answering

A local, open-source Agentic RAG system for asking questions about uploaded PDFs, designed to run on the free Google Colab GPU environment.

> **Status: Research release, offline numbers only**
>
> The full pipeline, a chat interface and the Colab notebook run today
> ([Quick Start](#13-quick-start)). Every benchmark number in this repository comes from
> the **offline stand-in stack** (hashing embedder, scripted generator, rule-based agents)
> on a machine with no GPU. The model-stack run on a Colab T4 is the next step
> ([NEXT_STEPS.md](NEXT_STEPS.md)).

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

[`notebooks/agentic_pdf_rag.ipynb`](notebooks/agentic_pdf_rag.ipynb) is the primary entry
point (DD-002). It follows PROJECT_SPEC section 12, with an explanation before every code
cell:

```text
1. Environment setup (clone at a pinned tag, DD-034)   9. Agent trace
2. Dependencies (incl. Tesseract for OCR)             10. Chat interface (Gradio)
3. Mode switch and model loading                      11. Benchmark (optional, --limit)
4. PDF upload (or a bundled benchmark PDF)            12. Evaluation metrics (from REPORT.md)
5. Parsing, chunking, indexing + a scanned-PDF demo   13. Ablation results (from REPORT.md)
6. Baseline RAG                                       14. Latency and peak VRAM
7. Agentic RAG                                        15. Final demo: a follow-up conversation
8. Dense vs agentic, side by side
```

With no GPU it announces **OFFLINE MODE** and runs the stand-in stack top to bottom on a
CPU; a test executes it that way on every run of the suite. No result number is typed
into it: tables are rendered from the reports `final-table` generates.
[`notebooks/gpu_benchmark_run.ipynb`](notebooks/gpu_benchmark_run.ipynb) runs the full
model-stack benchmark on a Colab T4, resumably, with results on Drive. Both are generated by
`notebooks/build_notebooks.py` and never edited by hand (DD-068).

---

## 13. Quick Start

**In Colab** (recommended):

1. Open [`notebooks/agentic_pdf_rag.ipynb`](notebooks/agentic_pdf_rag.ipynb) in Colab
   (*File → Open notebook → GitHub*).
2. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
3. Set `UPLOAD = True` in section 4 to use your own PDF, text or scanned; section 10 opens a
   chat interface with a public link.

The notebook clones the pinned tag `v0.7-gpu-run` (DD-034); it runs once that tag is pushed
(NEXT_STEPS.md step 1).

**Locally**, on a CPU, with the offline stand-in stack:

```bash
pip install -r requirements.txt
python -m src.cli ask --pdf paper.pdf --question "What was measured?" --offline
```

See section 19.1 for more commands, and `requirements-colab.txt` for the chat interface and
OCR.

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

* **Offline numbers only.** Every benchmark figure so far comes from the offline stand-in
  stack: a hashing embedder, a scripted extractive generator, a term-overlap reranker and
  rule-based agents, on a machine with no GPU. None is a claim about Qwen3-4B, bge-m3 or
  bge-reranker-v2-m3. On that stack no agentic component earned its cost (STATUS.md 9.4).
  The model-stack run is prepared but not yet run (NEXT_STEPS.md).
* **Peak VRAM, the model-call cost of LLM agents and the LLM parse-failure rate are
  unmeasured**, for the same reason; they are reported as null, never 0.
* **The benchmark is small**: 76 questions over 5 PDFs, 6 of them unanswerable, so
  abstention figures move 17 points per question.
* **OCR reads pages with no text layer only**, and its accuracy on real scans has not been
  benchmarked; no benchmark PDF is scanned. Tesseract's per-page time has not been measured
  on the development machine (DD-067).
* **Follow-up rewriting is untested against a benchmark**: no benchmark question is
  multi-turn, and the offline rewriter is a crude rule (DD-065).
* **The LLM judge has never been run.**
* Free-Colab GPU availability and memory vary between sessions; the GPU runner is resumable
  for that reason (DD-064).

---

## 17. Roadmap

### Phase 1 — Specification

* [x] Define project requirements
* [x] Define architecture
* [x] Define model-selection process
* [x] Define benchmark
* [x] Define evaluation protocol
* [x] Audit reference implementation

This checklist predates the project's phase numbering; section 19 maps the two.

### Phase 2 — Baseline

* [x] PDF ingestion, with OCR for pages that have no text layer (DD-067)
* [x] Chunking
* [x] Embeddings
* [x] Vector retrieval
* [x] Local generation
* [x] Baseline benchmark (offline stack)

### Phase 3 — Retrieval Improvements

* [x] Lexical retrieval
* [x] Hybrid fusion
* [x] Reranking
* [x] Retrieval evaluation (offline stack)

### Phase 4 — Agentic RAG

* [x] Query planner
* [x] Evidence controller
* [x] Retrieval refinement
* [x] Context selection
* [x] Verification

### Phase 5 — Evaluation

* [x] Full benchmark (offline stack)
* [x] Ablation experiments (offline stack)
* [x] Error analysis (offline stack)
* [x] Latency analysis (offline stack; milliseconds, no model loaded)
* [ ] GPU/RAM analysis: needs the model-stack run

### Phase 6 — Finalization

* [x] Colab notebook, chat interface and follow-up questions
* [x] Documentation
* [ ] Reproducibility testing: *Run all* in a fresh Colab runtime
* [ ] Final benchmark: the model-stack run (NEXT_STEPS.md)
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

**Current status: Phases 0-6 done on the offline stand-in stack; the Colab chatbot release is
built. The GPU model-stack run is next.**

A fuller record — verification evidence, enforced invariants, known limitations
and the next step — is kept in [STATUS.md](STATUS.md).

| Phase | State |
| --- | --- |
| 0 — Benchmark construction | **Done.** 5 PDFs, 76 human-verified questions, 33 dev / 43 eval |
| 1 — Ingestion and indexing | **Done.** Page-attributed chunks, two chunking strategies, two PDF backends, OCR for scanned pages, explicit failure modes |
| 2 — Dense baseline | **Done.** `results/baseline/` |
| 3 — Metrics harness | **Done.** Retrieval, answer, citation and abstention metrics; nine-category error taxonomy |
| 4 — Hybrid retrieval and reranking | **Done.** `results/hybrid/`; no significant gain on either primary metric |
| 5 — Agentic pipeline | **Done.** `results/agentic/`; not kept |
| 6 — Ablations and write-up | **Done on the offline stand-in stack.** Seven arms in `results/ablations/`; `results/final/` generated by `python -m src.cli final-table`; no component earned its cost offline |
| Release — Colab chatbot | **Built** (STATUS.md 9.5): the deliverable notebook, a Gradio chat interface, follow-up questions, OCR on by default, resumable runs, fixes for three Phase 6 defects, and every offline result regenerated under them (DD-061 to DD-069) |
| Next — model-stack run | **Prepared, not run**: `notebooks/gpu_benchmark_run.ipynb` on a Colab T4 ([NEXT_STEPS.md](NEXT_STEPS.md)) |

**Every result so far is from the offline stand-in stack**, on a machine with no GPU.
`STATUS.md` is the authoritative record.

What runs today:

* **PDF ingestion** with 1-based page numbers, ligature/hyphen/header normalization, **OCR
  on by default** for pages with no text layer (Tesseract, DD-067), and distinct errors, each
  with a remedy, for encrypted, blank and unreadable files and for a missing OCR engine.
* **Two chunkers** behind one interface — structure-aware (DD-009) and a fixed-size
  baseline — preserving `document_id`, `pages`, `page_span`, `section` and `chunk_id`.
  Every chunk carries its section heading, and oversized tables are split by rows with their
  header, so no page line is hidden from retrieval (DD-061).
* **Retrieval**: dense (FAISS, with an exact numpy fallback), BM25, Reciprocal Rank Fusion,
  and a cross-encoder reranker.
* **The agentic system**: planner, evidence controller, refinement and verifier in a bounded
  loop, each switchable, with a full per-question trace.
* **Grounded generation** with deterministic citation resolution: the model emits `[C1]`
  markers against a numbered evidence list and never sees a page number; code maps markers to
  pages (DD-031). A bibliography `[3]` copied from the source is not mistaken for a citation
  (DD-062).
* **Chat**: follow-up questions rewritten into standalone ones (DD-065), and a Gradio app over
  one or more PDFs with citations, refusals and an agent trace (DD-066).
* **Benchmark harness**: `run-benchmark` checkpoints every question and resumes with
  `--resume` (DD-064); `compare-runs` runs the paired bootstrap; `final-table` writes the
  report.
* **Dependency-free fallbacks** — a hashing embedder, a scripted extractive generator, a
  term-overlap reranker and rule-based agents — so everything, notebook included, runs and is
  tested on a CPU with no model weights.

748 tests: 747 pass and 1 is skipped (the real-Tesseract test, where the binary is absent),
in about 4 to 7 minutes on a laptop CPU, with no network, GPU or model weights. The suite
includes full 76-question runs checked record by record against `results/`, and an
execution of the deliverable notebook.

---

## 19.1 Quick start

```bash
pip install -r requirements.txt          # pdfplumber, numpy, faiss-cpu
python -m src.cli ask --pdf paper.pdf --question "What was measured?" --offline
```

`--offline` uses the hashing embedder and the scripted backend: no downloads, no
GPU, weak answers. Drop it to use the real models, which requires
`pip install -r requirements-models.txt` and a GPU.

```text
3 Hardware

Peak memory during generation reached 9.4 gigabytes with four-bit
weights, leaving sufficient headroom for the key-value cache. [C1]

Evidence:
  [C1] page 3 - 3 Hardware
```

Other commands:

```bash
python -m src.cli inspect --pdf paper.pdf --offline --show-chunks 3
python -m src.cli ask --pdf paper.pdf --question "..." --strategy fixed --top-k 8 --json
pip install -r requirements-dev.txt && python -m pytest    # 748 tests
```

In Python:

```python
from src.config import RAGConfig
from src.pipeline import DenseRAGPipeline

pipeline = DenseRAGPipeline(RAGConfig.default())   # or .low_memory() / .offline()
pipeline.index("paper.pdf")
result = pipeline.ask("What is the main contribution?")

print(result.answer)
print(result.cited_pages)      # (4, 9) -- resolved by code, not written by the model
print(result.abstained)
```

Every model name, chunk size, top-k and threshold lives in `src/config.py`
(DD-017). `RAGConfig.validate()` refuses a configuration that exceeds the
free-tier T4 budget or that makes a research-licensed model the default
(DD-024).

---

## 20. License

**Apache-2.0.** See [LICENSE](LICENSE); third-party and model licences are
recorded in [NOTICE](NOTICE).

Chosen for its explicit patent grant, and because it matches the licence of the
default model stack. It was unblocked by DD-033, which moved the default PDF
backend from PyMuPDF (AGPL-3.0) to pdfplumber (MIT); the whole runtime
dependency chain is now permissive — pdfplumber MIT, pdfminer.six MIT,
pypdfium2 BSD-3-Clause/Apache-2.0, Pillow MIT-CMU, numpy BSD-3-Clause,
faiss-cpu MIT. The Colab extras in `requirements-colab.txt` are permissive too:
gradio (chat interface), pytesseract and the Tesseract engine (OCR), all
Apache-2.0.

Two licence facts worth stating plainly:

* **PyMuPDF remains an optional test-only dependency** (AGPL-3.0). It writes the
  synthetic PDFs the test suite builds, and is the second parser backend. It is
  in the `dev` extra, never in the runtime dependencies.
* **The default generator must be Apache-2.0** (DD-024). `Qwen2.5-3B-Instruct`
  is `qwen-research`, research/non-commercial only; `RAGConfig.validate()`
  raises rather than let it become the default.

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
