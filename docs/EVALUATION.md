# Evaluation

How the numbers in this repository are produced, and why each choice was made.
The normative rules live in [EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md)
and [BENCHMARK_SPEC.md](../BENCHMARK_SPEC.md). This page is the guided tour.

> **Every stored result is from the offline stand-in stack:** a hashing
> embedder, a scripted extractive generator, a term-overlap reranker and
> rule-based agents. The GPU model-stack run is pending.

## 1. The dataset

| | |
| --- | --- |
| Documents | 5 PDFs, 56 pages (`benchmark/documents/`, checksums in `documents_metadata.json`) |
| Questions | 76: 33 dev, 43 eval (`benchmark/questions.json`) |
| Types | factual, numerical, table, definition, explanation, comparison, multi-hop, summary, ambiguous, unanswerable |
| Unanswerable | 6 (3 dev, 3 eval) |
| Verification | Every evidence page was checked by a person against the rendered PDF (`inspect-page`). Three page errors in the AI-drafted set were corrected |

**Why it was built first.** EXPERIMENT_PLAN.md section 1 names the benchmark,
not the retrieval code, as the critical path. Without it, nothing downstream
can be verified. `validate-benchmark` enforces every structural rule in one
pass.

## 2. Dev/eval discipline

* **Tuning happens on dev only** (`--split dev`).
* **Every tuning choice has a rule committed before the run.** Git history
  shows the rule commit before the results commit, in DD-058, DD-063, DD-069,
  DD-071 and DD-072.
* **Eval has been looked at twice:**
  * DD-059: threshold 0.3, on the pre-fix system;
  * DD-069: threshold 0.6, after the DD-061 to DD-063 fixes.
  Each look is disclosed where its numbers appear. A third look happens only
  if a dev study changes a default, and it will be disclosed the same way.
* **Some defects were found by reading eval failures** (q065 and q069 in
  DD-060). That is disclosed, and it is the reason eval is not a clean holdout
  any more.

## 3. Metrics, and why each one

| Metric | Why |
| --- | --- |
| Recall@1/3/5/10, page level (DD-025) | Page-level relevance keeps chunking strategies comparable. A gold page is found or not, however the text was cut |
| MRR@10, nDCG@10 (binary, DD-040) | Ranking quality, separate from whether the page was found at all |
| Full-Recall@5/10 | A multi-hop question needs *every* gold page, not one of them |
| Deterministic correctness (DD-036) | Token overlap plus exact numeric agreement. No LLM grades the answers, so scores are reproducible and cannot flatter the generator |
| Faithfulness (DD-037) | Token support of the answer in its evidence, plus a numeric check that is undefined (null, never 0) when the answer has no number |
| Citation any-correct, precision, completeness (DD-042) | Whether cited pages are gold pages, and whether every sentence is cited |
| Abstention accuracy, false-answer rate, over-abstention (DD-043) | All four cells of the answer/refuse outcome matrix, each over its own denominator |
| p95 latency, nearest rank | Never the mean alone (EVALUATION_PROTOCOL.md section 15) |
| Model calls per question (DD-055) | The agentic loop's cost. Absent (null) for systems that make none |

Unanswerable questions are excluded from every retrieval metric rather than
scored 0, and a metric with an empty denominator is null.

## 4. Statistics

* **Paired bootstrap over questions** (DD-028, DD-047): 2,000 resamples,
  seed 42, with a percentile CI of `mean(system − baseline)`. The p-value is
  the one that same distribution implies, so the CI and the p-value never
  disagree.
* **Primary metrics are named before each run:**
  * RQ1-RQ4 in `statistics.PRIMARY_METRICS`;
  * the Phase 6 rules in DD-058.
* **Every contrast is counted** (273 by DD-069), so one significant result
  among many is read as a hypothesis, not a finding (section 26.2).
* **Section 10 fair comparison:** generator, embedder, chunking and top-k
  must match, or `compare-runs` refuses. The dev studies declare their varied
  factor explicitly.

## 5. Failure taxonomy

Nine ordered, versioned, first-match categories (DD-038). Every failing record
gets exactly one. The order matters: a question whose gold page was never
retrieved is a retrieval failure, even if the generator also erred. The
counterfactual rules (DD-044, DD-054) blame the reranker or the verifier only
when the run without that component kept the page or the answer.

## 6. Reproducing a number

```bash
python -m src.cli validate-benchmark
python -m src.cli run-benchmark --system dense  --offline --out results/baseline
python -m src.cli run-benchmark --system hybrid --offline --out results/hybrid
python -m src.cli compare-runs --baseline results/baseline --system results/hybrid
python -m src.cli final-table          # results/final/REPORT.md
python -m src.cli readme-table         # copies its eval headline into README.md
python -m src.cli sweep-table --study chunking   # a dev study's table (DD-071)
```

The offline stack is deterministic. The test suite re-runs the 76 questions
and checks them record by record against `results/`.
