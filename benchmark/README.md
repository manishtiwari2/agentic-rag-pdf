# Benchmark Dataset

## 1. Purpose

This directory contains the documents and question-answer pairs used to evaluate the PDF RAG system.

The benchmark must be independent of the implementation.

Changing the RAG implementation must not require changing the benchmark.

---

## 2. Directory Structure

```text
benchmark/
├── README.md
├── documents/
│   ├── document_01.pdf
│   ├── document_02.pdf
│   └── ...
│
├── questions.json
└── ground_truth.json
```

For a small initial benchmark, `ground_truth.json` may be omitted if all ground-truth information is stored directly in `questions.json`.

---

## 3. Benchmark Documents

Use approximately:

```text
5–10 PDFs
```

The documents should represent different document types.

Recommended categories:

```text
research paper
technical documentation
technical/report document
financial or annual report
long-form report
instruction/manual document
```

Avoid using only short, clean research papers.

---

## 4. Document Selection Criteria

Prefer documents containing a mixture of:

* normal paragraphs
* headings
* lists
* tables
* numerical information
* cross-references
* multiple sections

Include both relatively short and long documents.

The benchmark should expose retrieval problems rather than only testing easy questions.

---

## 5. Document Metadata

Maintain a simple metadata file if useful:

```text
documents_metadata.json
```

Example:

```json
{
  "document_01.pdf": {
    "title": "Example Document",
    "category": "research_paper",
    "pages": 20,
    "source": "...",
    "license": "..."
  }
}
```

The source and license should be recorded for documents where applicable.

---

# 6. Question Dataset

Store benchmark questions in:

```text
questions.json
```

Recommended schema:

```json
[
  {
    "id": "q001",
    "document": "document_01.pdf",
    "question": "What is the main contribution of the document?",
    "answer": "...",
    "evidence_pages": [2, 3],
    "question_type": "summary",
    "difficulty": "medium"
  }
]
```

---

# 7. Optional Ground-Truth Fields

Questions may additionally contain:

```json
{
  "evidence_chunk_ids": [],
  "reasoning_steps": [],
  "alternative_answers": [],
  "notes": ""
}
```

These fields should only be included when useful.

Do not create unnecessary annotation work.

---

# 8. Question Categories

Every question should have one primary category.

Allowed categories:

```text
factual
definition
explanation
comparison
numerical
table
multi_hop
unanswerable
ambiguous
summary
```

---

# 9. Difficulty

Use:

```text
easy
medium
hard
```

Difficulty should describe the difficulty of answering from the document, not the difficulty of reading the question.

Example:

A simple fact buried on page 35 may be harder than a conceptual question clearly stated on page 2.

---

# 10. Ground-Truth Answer

For answerable questions:

```text
answer
+
evidence_pages
```

must be manually reviewed.

Answers should be concise and factual.

Do not write unnecessarily verbose reference answers.

---

# 11. Evidence Annotation

Whenever practical, identify the exact supporting chunk(s).

Example:

```json
{
  "id": "q014",
  "document": "document_03.pdf",
  "question": "Why was method A preferred?",
  "answer": "...",
  "evidence_pages": [7, 12],
  "evidence_chunk_ids": [
    "document_03_p7_c2",
    "document_03_p12_c1"
  ],
  "question_type": "multi_hop",
  "difficulty": "hard"
}
```

For multi-hop questions, include all evidence needed to construct the answer.

---

# 12. Unanswerable Questions

Some questions must deliberately have no answer in the document.

Example:

```json
{
  "id": "q031",
  "document": "document_02.pdf",
  "question": "What was the author's salary?",
  "answer": null,
  "evidence_pages": [],
  "question_type": "unanswerable",
  "difficulty": "medium"
}
```

The expected system behaviour is abstention.

Do not create unanswerable questions that are merely ambiguous unless they are intentionally testing ambiguity handling.

---

# 13. Numerical Questions

Include questions involving:

* percentages
* dates
* quantities
* scores
* measurements
* financial values
* experimental results

Ground-truth answers must preserve important numerical precision.

Example:

```text
87.4%
```

should not casually be treated as equivalent to:

```text
80%
```

---

# 14. Table Questions

Where documents contain tables, include questions that require retrieving table information.

Record the relevant page.

Example:

```text
Which model achieved the highest accuracy in Table 4?
```

The evaluation should verify that the generated answer agrees with the table evidence.

---

# 15. Multi-Hop Questions

Multi-hop questions should require evidence from more than one location.

Example:

```text
Page 5 → methodology
Page 13 → results
```

The answer should require combining those pieces of information.

These questions are especially useful for evaluating the agentic retrieval loop.

---

# 16. Benchmark Size

Initial target:

```text
50–100 questions
```

A practical first version:

```text
~75 questions
```

Suggested distribution:

```text
15 factual
8 definition
10 explanation
8 comparison
8 numerical
6 table
10 multi-hop
6 unanswerable
4 ambiguous
```

The distribution may change depending on the available PDFs.

---

# 17. Quality Control

Before using the benchmark:

* verify every document opens correctly
* verify page numbers
* verify every reference answer
* verify supporting evidence
* verify unanswerable questions are genuinely unsupported
* remove duplicate questions
* check that questions are understandable without additional context
* ensure question categories are correct

At least a sample of questions should be independently reviewed.

---

# 18. Preventing Benchmark Leakage

Do not include benchmark answers in prompts, code, or retrieval indexes.

The system receives only:

```text
PDF
+
user question
```

Ground-truth answers are used only by the evaluation pipeline.

The benchmark should not be used to fine-tune the generation model.

---

# 19. Versioning

Assign a benchmark version.

Example:

```text
benchmark_version = "1.0"
```

If documents or questions change substantially, create a new version.

Do not silently modify questions after final evaluation.

---

# 20. Expected Benchmark Result

The evaluation system should produce per-question records containing:

```text
question_id
system
generated_answer
reference_answer
retrieved_evidence
citations
retrieval_metrics
answer_metrics
latency
verification_status
```

Aggregate results should be generated automatically from these records.

---

# 21. Important Principle

The benchmark is not intended to make the system look good.

It exists to expose weaknesses in:

```text
retrieval
reranking
context selection
reasoning
generation
citation
abstention
```

A strong benchmark should contain questions the system can fail.

That makes improvements measurable and credible.
