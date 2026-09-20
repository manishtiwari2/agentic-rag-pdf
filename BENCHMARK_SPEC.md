# Benchmark Specification

## 1. Objective

The benchmark measures whether the proposed local Agentic RAG system can answer questions from PDFs accurately, faithfully, and efficiently.

The benchmark must compare:

1. Dense RAG baseline
2. Hybrid/Reranked RAG
3. Agentic RAG

The benchmark must measure both **retrieval quality** and **end-to-end answer quality**.

---

# 2. Benchmark Principles

The benchmark must be:

* reproducible
* document-grounded
* independent of external web knowledge
* representative of real PDF questions
* sufficiently difficult to expose retrieval failures
* small enough to run repeatedly in Colab

The same documents and questions must be used when comparing system variants.

---

# 3. Benchmark Documents

The benchmark should contain approximately **5–10 PDFs** initially.

Documents should vary in:

* length
* domain
* writing style
* formatting
* number of sections
* presence of tables
* presence of numerical information
* information density

Recommended document categories:

```text
research paper
technical documentation
annual/financial report
instruction/manual document
long-form report
```

Avoid constructing the benchmark entirely from one type of PDF.

---

# 4. Document Requirements

Each benchmark document should have:

```text
document_id
filename
domain/category
approximate page count
source/license information
```

Documents must be legally usable for the project.

The benchmark metadata should record the source and license where applicable.

---

# 5. Question Dataset

Questions should be stored in:

```text
benchmark/questions.json
```

Each question should contain at minimum:

```json
{
  "id": "q001",
  "document": "document_01.pdf",
  "question": "...",
  "answer": "...",
  "evidence_pages": [3, 4],
  "question_type": "factual"
}
```

Optional fields:

```text
evidence_chunk_ids
difficulty
reasoning_steps
alternative_answers
notes
```

---

# 6. Question Categories

The initial benchmark should contain multiple question types.

### 6.1 Factual

Directly stated information.

Example:

> What dataset was used in the experiment?

---

### 6.2 Definition

Tests understanding of terminology.

Example:

> What does the document mean by “X”?

---

### 6.3 Why / Explanation

Requires connecting a statement with its explanation.

Example:

> Why does the proposed method use X?

---

### 6.4 Comparison

Requires evidence about multiple concepts.

Example:

> How does method A differ from method B?

---

### 6.5 Numerical

Tests accurate retrieval of numbers.

Examples:

* scores
* percentages
* dates
* quantities
* costs

Numerical questions should receive special attention because small generation errors can change the answer.

---

### 6.6 Table-Based

Questions requiring information contained in a table.

Example:

> Which method achieved the highest value in Table 3?

---

### 6.7 Multi-Hop

Requires combining evidence from different parts of the document.

Example:

> Based on the methodology and results sections, why did the authors prefer approach X?

These questions are particularly important for evaluating the agentic retrieval loop.

---

### 6.8 Unanswerable

The required information is absent from the document.

Expected behaviour:

```text
The document does not provide sufficient information to answer this question.
```

The system should not hallucinate an answer.

---

### 6.9 Ambiguous / Reference Resolution

Tests questions involving terms such as:

```text
this method
the proposed approach
the second experiment
they
it
the previous model
```

The system must resolve the reference using document context.

---

# 7. Difficulty Levels

Each question should optionally have:

```text
easy
medium
hard
```

Suggested distribution:

```text
Easy       30%
Medium     50%
Hard       20%
```

Hard questions should include multi-hop, comparison, numerical, or ambiguous cases.

---

# 8. Dataset Size

Initial target:

```text
50–100 questions
```

Recommended starting point:

```text
~75 questions
```

Example distribution:

```text
Factual          15
Definition        8
Explanation      10
Comparison        8
Numerical         8
Table             6
Multi-hop        10
Unanswerable      6
Ambiguous         4
```

The exact distribution may change based on available documents.

---

# 9. Ground Truth

Every answerable question must have:

```text
reference answer
supporting page(s)
```

Where possible, record the specific evidence chunk(s).

For multi-hop questions, all necessary evidence locations should be recorded.

Ground truth should be manually reviewed.

---

# 10. Retrieval Metrics

## Recall@K

Measures whether relevant evidence appears within the top K retrieved chunks.

Report:

```text
Recall@1
Recall@3
Recall@5
Recall@10
```

Primary retrieval metric:

```text
Recall@5
```

---

## Mean Reciprocal Rank

MRR measures how highly the first relevant result appears.

Report:

```text
MRR@10
```

---

## nDCG

Use nDCG when graded relevance labels are available.

Report:

```text
nDCG@10
```

---

# 11. Answer Metrics

The benchmark should evaluate:

### Correctness

Does the answer agree with the reference answer and document evidence?

### Faithfulness

Are the claims supported by retrieved document evidence?

### Citation correctness

Do cited pages/chunks actually support the claims?

### Citation completeness

Are important factual claims supported by citations?

---

# 12. LLM-as-Judge

LLM-based evaluation may be used for:

* answer correctness
* faithfulness
* citation quality

The evaluator must be explicitly documented.

The evaluation prompt must receive:

```text
question
reference answer
system answer
retrieved evidence
citations
```

The evaluator must not have access to external information.

LLM-as-judge scores should not be treated as ground truth.

Where practical, manually inspect a sample of results.

---

# 13. Abstention Metrics

For unanswerable questions measure:

```text
Abstention accuracy
False-answer rate
Unsupported-answer rate
```

Important failure:

```text
Document does not contain answer
          ↓
System invents answer
```

This should count as a serious error.

---

# 14. End-to-End Metrics

For every system configuration report:

```text
Answer correctness
Faithfulness
Citation correctness
Citation completeness
Abstention accuracy
```

Also report:

```text
Average latency
Median latency
P95 latency
```

---

# 15. Resource Metrics

Record:

```text
PDF ingestion time
Index construction time
Embedding time
Retrieval latency
Reranking latency
Generation latency
Total query latency
Peak GPU memory
Peak system RAM
```

This is necessary to establish practical Colab feasibility.

---

# 16. Baseline Experiments

### Experiment A

```text
Dense retrieval
+
LLM generation
```

### Experiment B

```text
Hybrid retrieval
+
reranker
+
LLM generation
```

### Experiment C

```text
Query planning
+
hybrid retrieval
+
reranker
+
evidence control
+
LLM generation
+
verification
```

All experiments must use the same benchmark dataset.

---

# 17. Ablation Experiments

The project should measure the contribution of important components.

At minimum:

```text
Without reranker
Without query planner
Without retrieval refinement
Without evidence controller
Without verification
```

Example:

| Configuration      | Recall@5 | Faithfulness | Accuracy | Latency |
| ------------------ | -------: | -----------: | -------: | ------: |
| Dense baseline     |          |              |          |         |
| + Hybrid retrieval |          |              |          |         |
| + Reranker         |          |              |          |         |
| + Agent            |          |              |          |         |
| + Verification     |          |              |          |         |

---

# 18. Agentic Value

The agentic architecture should be considered useful only if experiments show measurable improvement in one or more important dimensions.

Potential improvements include:

```text
higher answer accuracy
higher faithfulness
better citation accuracy
better multi-hop performance
better abstention
```

The additional cost must also be reported:

```text
latency
GPU usage
number of model calls
retrieval iterations
```

Do not claim that agentic RAG is better merely because the architecture is more complex.

---

# 19. Reproducibility

Every benchmark run must record:

```text
timestamp
system configuration
generation model
embedding model
reranker
quantization
chunk size
chunk overlap
top-k
retrieval configuration
generation parameters
random seed where applicable
```

Results should be saved under:

```text
results/
├── baseline/
├── experiments/
└── final/
```

---

# 20. Benchmark Output

The benchmark should generate:

### Summary table

```text
System
Answer Accuracy
Faithfulness
Citation Accuracy
Recall@5
MRR
Latency
Peak VRAM
```

### Per-question results

For each question:

```text
question_id
system
answer
reference_answer
retrieved_chunks
citations
metrics
latency
verification_status
```

### Error analysis

Group failures into:

```text
retrieval failure
reranking failure
context selection failure
generation failure
citation failure
verification failure
abstention failure
```

---

# 21. Success Criteria

The final system should demonstrate:

1. Reliable document retrieval.
2. Grounded answers.
3. Accurate citations.
4. Appropriate abstention.
5. Improved performance over the simplest baseline on relevant benchmark categories.
6. Acceptable latency for free-Colab execution.
7. Reproducible results.

The benchmark is successful only if it reveals **where the system works, where it fails, and whether each architectural component provides measurable value**.
