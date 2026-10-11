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

## 9.1 The budget, stated as a number

"Conservative" is not actionable. The target is the free-tier NVIDIA T4:

```text
Total VRAM              ~15 GB usable (16 GB nominal)
Budget for all models   <= 10 GB resident
Headroom reserved       >= 5 GB for activations, KV cache and fragmentation
System RAM              ~12 GB
```

The headroom is not padding. Peak VRAM during generation is driven by the KV
cache and activations, not by the weights, and a stack that fits at rest can
still fail on a long-context query.

Three models must fit inside the 10 GB at once, or be loaded and released in
sequence:

| Component | Approximate resident VRAM |
| --- | --- |
| Generation, 4B @ 4-bit NF4 | ~2.5 GB |
| Generation, 4B @ bf16 | ~8.0 GB |
| Generation, 1.5B @ bf16 | ~3.1 GB |
| `bge-m3` embeddings @ fp16 | ~1.2 GB |
| `bge-reranker-v2-m3` @ fp16 | ~1.2 GB |

These are estimates from parameter counts, not measurements. They exist to set
expectations before the first run; section 3 of `EVALUATION_PROTOCOL.md`
requires the measured figures to be recorded and they supersede this table.

The practical consequence: a 4B generator in bf16 plus both retrieval models
leaves too little headroom to be safe on a T4, so either quantisation or
sequential loading is required rather than optional.

## 9.2 Session constraints beyond memory

The free tier also imposes limits that affect experiment design:

* Sessions are reclaimed after a period of inactivity, and have a wall-clock
  ceiling. A benchmark run that cannot finish inside one session must
  checkpoint per-question results to disk as it goes, not only at the end.
* The assigned GPU varies between sessions (T4, sometimes L4). Results are not
  comparable across sessions unless the GPU is recorded — which section 2 of
  `EVALUATION_PROTOCOL.md` already requires.
* Local disk does not persist. Model weights are re-downloaded on every fresh
  runtime unless a mounted Drive cache is used; that download time must be
  reported separately from inference latency.

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

Do not describe a model as suitable for the project until its licensing terms have been checked.

## 11.1 Verified model licenses

Checked against the Hugging Face model cards on 2026-09-20. Re-verify before
release: model cards are edited, and a license can change between revisions.

| Model | Role | License | Commercial use | Notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | generation | Apache-2.0 | Yes | 4.0B params (3.6B excl. embeddings), 262k native context |
| `Qwen/Qwen2.5-3B-Instruct` | generation | **`qwen-research`** | **No — research only** | See 11.2 |
| `Qwen/Qwen2.5-1.5B-Instruct` | generation | Apache-2.0 | Yes | |
| `BAAI/bge-m3` | embeddings | MIT | Yes | |
| `BAAI/bge-reranker-v2-m3` | reranking | Apache-2.0 | Yes | |

## 11.2 Finding: Qwen2.5-3B-Instruct is not openly licensed

Candidate B is the only model in the pool released under the Qwen Research
License rather than Apache-2.0. That license restricts use to research and
non-commercial purposes.

This matters because the project describes itself as open source and intends
the notebook to be reusable by others. Shipping a default configuration that
silently pulls a research-only model would hand every downstream user a
licensing problem they did not choose.

Consequences for model selection:

* Candidate B may still be **benchmarked**, as a research comparison point.
* Candidate B must **not** become the default configuration.
* If Candidate B wins on quality, the recommended stack must still default to
  an Apache-2.0 model, and the notebook must state the licensing trade-off
  rather than quietly selecting the better-scoring model.

This restriction is specific to the 3B checkpoint. Other sizes in the Qwen2.5
family are Apache-2.0, so the restriction does not generalise across the
family and must be checked per checkpoint rather than per vendor.

Recorded as DD-024 in `DESIGN_DECISIONS.md`.

## 11.3 Licenses still to record

The license audit is not complete. Before release, also record:

* **Benchmark documents** — source URL and license for every PDF
  (`BENCHMARK_SPEC.md` section 4). Blocked until the documents are chosen.
* **Python dependencies** — the transitive set, in particular any GPL/AGPL
  component that would constrain the project's own license.
* **The project's own license** — `README.md` section 20 says only that one
  will be chosen. Until it is named, contributors cannot know what terms they
  are contributing under.

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
