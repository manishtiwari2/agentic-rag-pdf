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

## 4.1 Sourcing, and why it is the blocking task

No documents have been chosen yet. Nothing downstream can proceed without
them: the questions, the ground truth, every metric and every experiment all
depend on having the PDFs in hand. This is the critical-path item, not the
retrieval code.

Licensing rules out most of the obvious choices. Corporate annual reports,
commercial technical manuals and most published books cannot be redistributed
in the repository, even though they are exactly the document types section 3
asks for.

Categories that are safe to redistribute:

| Category | Source | Typical license |
| --- | --- | --- |
| Research paper | arXiv | per-paper; many are CC-BY, some are arXiv's non-exclusive licence — check each |
| Technical standard | NIST, IETF RFCs | US Government works: public domain |
| Government report | GAO, national statistics offices | public domain in the US |
| Financial report | central bank / public-agency annual reports | public domain in the US |
| Manual / instructional | government agency handbooks | public domain in the US |
| Long-form report | intergovernmental organisation reports | often CC-BY |

Practical guidance:

* Check the licence of **each individual paper**. "arXiv" is not a licence,
  and the per-paper terms vary; a CC-BY paper may be redistributed, one under
  arXiv's non-exclusive licence may not.
* "US Government work" means produced *by* the agency. A contractor report
  hosted on a government site may still be under copyright.
* Where redistribution is not permitted, ship a **manifest** — URL plus SHA-256
  checksum — and download at notebook run time instead of committing the PDF.
  The checksum matters: a silently updated source document invalidates every
  evidence page in the ground truth, and without a checksum that failure is
  invisible.

## 4.2 Document metadata

Record per document, in `benchmark/documents_metadata.json`:

```json
{
  "document_01.pdf": {
    "title": "...",
    "category": "research_paper",
    "pages": 15,
    "source_url": "https://...",
    "license": "CC-BY-4.0",
    "redistributable": true,
    "sha256": "...",
    "retrieved": "2026-09-20"
  }
}
```

`sha256` and `retrieved` are not bureaucracy. Ground truth is annotated
against a specific rendering of a specific file; if the file changes, the
page numbers in `questions.json` become wrong and the benchmark degrades
silently rather than failing.

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
  "question_type": "factual",
  "difficulty": "easy",
  "split": "eval"
}
```

Field rules:

| Field | Rule |
| --- | --- |
| `id` | unique across the whole file |
| `document` | filename, matching a file in `benchmark/documents/` |
| `answer` | `null` for `unanswerable`, a non-empty string otherwise |
| `evidence_pages` | **1-based** page numbers; empty exactly when `answer` is `null` |
| `question_type` | one of the ten categories in section 6 |
| `difficulty` | `easy`, `medium` or `hard` |
| `split` | `dev` or `eval`; fixed, stratified, never recomputed (see `EVALUATION_PROTOCOL.md` section 5.1) |

`evidence_pages` being 1-based is worth stating explicitly because PDF
libraries index from 0. An off-by-one here is invisible in review, depresses
recall uniformly across every system, and looks exactly like a retrieval
problem.

Optional fields:

```text
evidence_chunk_ids     secondary annotation; page-level is authoritative
reasoning_steps        for multi_hop: what must be combined
alternative_answers    other phrasings a correct answer may take
notes                  anything a reviewer needs to know
```

## 5.1 Validation before use

A malformed benchmark produces plausible-looking numbers, which is the worst
failure mode available to an evaluation harness. The dataset must be checked
before any run, and the run must refuse to start on failure:

* no duplicate `id`
* no placeholder text (`REPLACE_WITH_...`) left in any field
* every answerable question has at least one `evidence_pages` entry
* every unanswerable question has `answer: null` **and** empty `evidence_pages`
* every `evidence_pages` value is >= 1 and <= that document's page count
* every `document` names a file that exists
* every `question_type` is in the section 6 taxonomy

The current `benchmark/questions.json` is a 10-question template that fails
several of these checks by design. It is a schema example, not a dataset, and
it must not be mistaken for one.

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
Factual          14
Definition        7
Explanation       9
Comparison        7
Numerical         8
Table             6
Multi-hop        10
Summary           4
Unanswerable      6
Ambiguous         4
                ---
                 75
```

This covers all ten categories listed in section 6 and in
`benchmark/README.md` section 8. An earlier version of this table omitted
`summary`, which left a category that the taxonomy allows, the question schema
accepts, and `questions.json` already uses, but that the dataset plan never
budgeted for.

The exact distribution may change based on available documents. Two
constraints should hold whatever it becomes:

* Every category in section 6 gets a non-zero allocation, or is removed from
  the taxonomy. A category with no questions cannot be reported on, and
  leaving it in the taxonomy implies coverage that does not exist.
* `multi_hop`, `comparison` and `unanswerable` keep their share. These three
  carry most of the discriminative power between the baseline and agentic
  systems (research questions RQ3 and RQ4); trimming them to make room makes
  the benchmark easier and less informative at the same time.

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

## Unit of relevance

Ground-truth evidence is annotated and scored at **page** level, not chunk
level. A retrieved chunk counts as relevant if any page it covers is in the
question's `evidence_pages`.

This is a deliberate choice, and the reason matters:

* A chunk ID is a function of the chunking configuration. Scoring against
  chunk IDs would mean the ground truth changes when the chunker changes,
  which makes the structure-aware vs. fixed-size comparison in DD-009
  meaningless — the two systems would be graded against different answer keys.
* Page numbers are stable across every configuration, and they are the unit
  the system cites, so retrieval metrics and citation metrics share a scale.

`evidence_chunk_ids` stays in the schema as optional documentation, and
chunk-level scoring may be reported as a secondary diagnostic. The page-level
figures are the ones compared across systems.

A chunk that spans a page break is credited for **every** page it covers. A
chunk running across pages 7-8 is a hit for a question whose evidence is on
either page.

---

## Recall@K

Measures whether relevant evidence appears within the top K retrieved chunks.

Two definitions are reported, because they answer different questions and
disagree sharply on multi-hop items:

```text
Recall@K        1.0 if ANY gold evidence page is in the top K
Full-Recall@K   1.0 if EVERY gold evidence page is in the top K
```

For a single-evidence question the two are identical. For a question whose
`evidence_pages` is `[7, 12]`, retrieving only page 7 scores 1.0 on Recall@5
and 0.0 on Full-Recall@5.

Reporting only the first would overstate multi-hop retrieval, since a system
that reliably finds one hop and never the second would look complete.
Reporting only the second would understate ordinary factual retrieval. Both
are therefore required.

Report:

```text
Recall@1
Recall@3
Recall@5
Recall@10
Full-Recall@5
Full-Recall@10
```

Primary retrieval metric:

```text
Recall@5        overall
Full-Recall@5   for multi_hop and comparison questions
```

Recall is undefined for `unanswerable` questions, which have no gold evidence.
Those questions are excluded from every retrieval metric rather than scored as
0.0 — scoring them as failures would make a correctly-abstaining system look
like a retrieval failure.

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

Abstention is a two-sided problem and must be measured on both sides. A system
that abstains on everything scores perfectly on the unanswerable set, so
metrics restricted to that set cannot detect the failure.

The four outcomes:

| | System answered | System abstained |
| --- | --- | --- |
| **Document supports an answer** | correct behaviour | **over-abstention** |
| **Document does not** | **false answer** | correct behaviour |

Metrics:

```text
Abstention accuracy    over UNANSWERABLE questions:  abstained / total
False-answer rate      over UNANSWERABLE questions:  answered / total
Over-abstention rate   over ANSWERABLE questions:    abstained / total
Unsupported-answer rate  over ALL questions: answers whose claims are not
                         supported by the retrieved evidence
```

`Over-abstention rate` was missing from earlier versions of this list even
though `EVALUATION_PROTOCOL.md` section 21 already called incorrect refusal a
failure. Without it the specification defined a metric set that a
refuse-everything system would win.

Note that `Unsupported-answer rate` is not restricted to unanswerable
questions: a system can answer an answerable question with claims its
retrieved evidence does not support, and that is the same failure.

## 13.1 Severity

These failures are not equally bad and should not be averaged into one number.

```text
Document does not contain answer
          ↓
System invents answer
```

A confident false answer is the most serious failure in this project. It is
the one a user cannot detect without re-reading the source document, which
defeats the purpose of the system.

Over-abstention is a real failure and must be reported, but it is a *safe*
failure: the user learns nothing, rather than learning something false.

Report the two separately and never as a combined "abstention error rate".

## 13.2 Detecting abstention

Abstention is detected from the answer text, not from system self-report
alone. The generator is instructed to emit one canonical refusal sentence, and
the evaluator also matches a small fixed set of paraphrases.

Two rules keep this honest:

* The detector's patterns must be fixed before the run and recorded with the
  results, for the same reason the judge prompt is fixed.
* A long answer that mentions a gap in passing ("the document does not state
  the exact date, but the experiment ran in March") is an **answer**, not an
  abstention. A detector that classifies it as abstention will inflate
  abstention accuracy and hide false answers.

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

Results should be saved under the layout defined in `EVALUATION_PROTOCOL.md`
section 27, which is authoritative:

```text
results/
├── baseline/
├── hybrid/
├── agentic/
├── ablations/
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
