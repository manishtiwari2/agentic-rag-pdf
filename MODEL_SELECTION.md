# Model Selection

## 1. Objective

Select a fully local model stack that provides strong PDF question-answering quality while remaining practical on the free Google Colab GPU environment.

The stack contains three model roles:

1. Generation / agent reasoning
2. Document and query embeddings
3. Candidate reranking

Model selection must be benchmark-driven.

No model should be declared the final choice solely because it is larger or newer.

---

## 2. Hard Constraints

Every selected model must satisfy:

* Open-weight/local inference.
* No API key required for inference.
* No paid inference service.
* Practical execution on free Google Colab GPU.
* Compatible with the chosen Python inference stack.
* License permits the intended open-source project usage.
* Model weights can be downloaded during notebook setup.
* Quantized inference should be considered when useful.

The notebook must report:

* model identifier
* model revision/version where practical
* parameter count
* quantization
* approximate model memory
* peak GPU memory during inference
* inference latency

---

# 3. Generation Model

The generation model is responsible for:

* query planning
* retrieval decisions where required
* evidence assessment
* answer generation
* verification where applicable

The initial candidate pool should remain small.

### Candidate A

```text
Qwen3-4B-Instruct-2507
```

Primary candidate because it is small enough to investigate for Colab while providing useful instruction-following and reasoning capability.

### Candidate B

```text
Qwen2.5-3B-Instruct
```

Smaller alternative for memory and latency comparisons.

### Candidate C

```text
Qwen2.5-1.5B-Instruct
```

Lightweight fallback and efficiency baseline.

Additional models should only be introduced if there is a clear experimental reason.

---

# 4. Generation Model Selection Criteria

Evaluate candidate models on:

| Metric             | Purpose                              |
| ------------------ | ------------------------------------ |
| Answer correctness | Does the model answer correctly?     |
| Faithfulness       | Is the answer supported by evidence? |
| Citation quality   | Are citations appropriate?           |
| Abstention         | Does it avoid unsupported answers?   |
| Latency            | Practical Colab performance          |
| Peak VRAM          | Hardware feasibility                 |
| Stability          | Consistent structured output         |

The final model should be selected using the combined benchmark results rather than parameter count.

---

# 5. Embedding Model

The embedding model converts:

```text
document chunks → vectors
user query → vector
```

Initial candidate:

```text
BAAI/bge-m3
```

It should be evaluated for:

* retrieval recall
* retrieval ranking
* latency
* memory
* input length
* language coverage
* ease of local deployment

A smaller embedding model may be tested if BGE-M3 is impractical for the target Colab runtime.

---

# 6. Reranker

The reranker receives:

```text
query + retrieved candidate chunk
```

and produces a relevance score.

Initial candidate:

```text
BAAI/bge-reranker-v2-m3
```

Evaluate whether reranking improves:

```text
Recall@k
MRR
nDCG
Answer correctness
Faithfulness
```

while adding acceptable latency.

Reranking must remain optional so its contribution can be measured.

---

# 7. Model Stack

Initial experimental stack:

```text
Generation:
Qwen3-4B-Instruct-2507

Embeddings:
BAAI/bge-m3

Reranking:
BAAI/bge-reranker-v2-m3
```

This is the **starting configuration**, not the final configuration.

---

# 8. Quantization

Generation models should be tested with an appropriate low-memory configuration when required.

Potential options include:

```text
4-bit
8-bit
FP16/BF16 where practical
```

Selection should consider:

* memory reduction
* generation speed
* output quality
* numerical stability

Do not assume that the lowest-memory configuration is automatically best.

Record the exact configuration used in benchmarks.

---

# 9. Colab Resource Budget

The system should target a conservative memory budget.

The implementation should assume that GPU availability can vary between Colab sessions.

At runtime record:

```text
GPU name
total GPU memory
allocated GPU memory
reserved GPU memory
peak GPU memory
system RAM
```

The pipeline should fail gracefully if the selected model does not fit.

---

# 10. Model Loading Principles

Models should be loaded lazily where practical.

Avoid simultaneously keeping unnecessary copies of:

* generation model
* embedding model
* reranker

in GPU memory.

Potential strategy:

```text
PDF ingestion
    ↓
embedding/indexing
    ↓
release unnecessary resources
    ↓
generation / agent inference
```

However, model unloading/reloading should only be used if the latency trade-off is acceptable.

---

# 11. Licensing

Before final selection, verify the license of every model and relevant dependency.

Record:

```text
Model
License
Source
Commercial-use restrictions
Redistribution considerations
```

The final repository should clearly document model licenses.

Do not describe a model as suitable for the project until its licensing terms have been checked.

---

# 12. Evaluation Procedure

Model selection should happen in stages.

### Stage 1 — Feasibility

For each candidate:

```text
Can it load?
Can it run inference?
Does it fit the target GPU?
```

Reject models that fail basic hardware requirements.

### Stage 2 — Quality

Run the same benchmark questions against each viable model.

Measure:

```text
answer correctness
faithfulness
citation quality
abstention behaviour
```

### Stage 3 — Performance

Measure:

```text
model loading time
first-token latency where available
generation latency
peak VRAM
```

### Stage 4 — Robustness

Test:

* short PDFs
* long PDFs
* technical terminology
* numerical questions
* multi-hop questions
* unanswerable questions

---

# 13. Controlled Variables

When comparing generation models, keep constant:

```text
PDFs
benchmark questions
retrieval results
prompt structure
temperature
maximum output tokens
evaluation criteria
```

When comparing embedding models, keep constant:

```text
documents
chunking
retrieval algorithm
top-k
reranker
generation model
```

When comparing rerankers, keep constant:

```text
documents
chunks
embedding model
candidate pool
generation model
```

This prevents misleading comparisons.

---

# 14. Selection Rule

The final model stack should be selected using a practical multi-objective decision:

```text
quality
+
grounding
+
retrieval performance
+
latency
+
memory
+
reliability
+
licensing
```

Do not optimize one metric in isolation.

For example, a model that improves answer quality slightly but requires substantially more memory or latency may not be appropriate for the target environment.

The final choice and its rationale must be recorded in:

```text
DESIGN_DECISIONS.md
```

---

# 15. Reproducibility

Every benchmark result must record the exact:

```text
model identifier
model revision
quantization
device
generation parameters
embedding model
reranker model
```

A fresh Colab session should be able to reproduce the selected configuration using the documented setup.

---

# 16. Important Constraint

Do not increase the model count unnecessarily.

The goal is not to benchmark dozens of models.

The initial experiment should establish whether a small, practical local stack can provide strong Agentic RAG performance.

Only introduce additional candidates when benchmark results reveal a specific weakness that another model could plausibly address.
