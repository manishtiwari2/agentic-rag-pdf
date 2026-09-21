# Evaluation Protocol

## 1. Purpose

This document defines the procedure used to evaluate and compare the PDF RAG systems.

The objective is to make benchmark results:

* reproducible
* comparable
* statistically meaningful where practical
* independent of accidental configuration changes

The protocol applies to:

```text id="6t1vlz"
Dense RAG
Hybrid/Reranked RAG
Agentic RAG
Ablation experiments
Model comparisons
```

---

# 2. Evaluation Environment

Every final benchmark run must record:

```text id="6evq7c"
Python version
PyTorch version
Transformers version
embedding library/version
reranker library/version
GPU name
GPU memory
system RAM
CUDA version
```

The final notebook should display this information automatically.

---

# 3. Configuration Snapshot

Every benchmark result must store the complete experiment configuration.

Minimum configuration:

```yaml id="y07l7s"
generation_model:
generation_model_revision:
quantization:

embedding_model:
embedding_model_revision:

reranker_model:
reranker_enabled:

chunking_strategy:
chunk_size:
chunk_overlap:

retrieval_strategy:
top_k:
rerank_k:

max_retrieval_iterations:

temperature:
max_new_tokens:

random_seed:
```

No benchmark result should exist without a corresponding configuration.

---

# 4. Dataset Version

The benchmark dataset must have an explicit version.

Example:

```text id="r2yn0n"
benchmark_version: 1.0
```

If questions, answers, evidence pages, or documents change, increment the benchmark version.

Do not compare results from different benchmark versions without explicitly noting the difference.

---

# 5. Dataset Split

Where practical, maintain:

```text id="2j9vqs"
development set
evaluation set
```

The development set may be used during implementation.

The evaluation set should remain unchanged during final tuning.

This reduces the risk of optimizing directly against the final benchmark.

## 5.1 The split, concretely

```text
development set   ~1/3 of questions   (~25 of 75)
evaluation set    ~2/3 of questions   (~50 of 75)
```

Assignment rules:

* The split is recorded in `questions.json` as a `split` field per question,
  with values `dev` or `eval`. It is not computed at run time — a split that
  can be recomputed can be recomputed differently, and then the held-out set
  was never held out.
* The split is **stratified by question type and by document**. A dev set
  containing no multi-hop questions, or drawing only from one PDF, cannot
  detect the failures it exists to catch.
* Once fixed, questions do not move between splits. Moving a question after
  seeing a result is how a held-out set stops being held out.

## 5.2 What each set is for

| | Development set | Evaluation set |
| --- | --- | --- |
| Prompt wording | tune freely | do not tune against |
| Thresholds, top-k, chunk size | tune freely | do not tune against |
| Debugging individual failures | yes | only after final numbers are recorded |
| Reported headline results | no | yes |
| Times it may be run | unlimited | ideally once, at the end |

Running the evaluation set repeatedly during development does not corrupt it
mechanically, but it corrupts it in practice: each look informs the next
change, and the final number drifts upward without the system improving.

If the evaluation set does get used during tuning, say so in the write-up. A
disclosed leak is a caveat; an undisclosed one invalidates the result.

Development-set numbers may be reported, but always labelled as such and never
in the same table as evaluation-set results.

---

# 6. Evaluation Modes

The system must support two modes.

## Interactive Mode

Used for:

* demonstrations
* debugging
* manual inspection

## Benchmark Mode

Used for:

* automated evaluation
* latency measurement
* metric calculation
* result storage

Benchmark mode must avoid manual intervention.

---

# 7. Baseline Configuration

The first baseline is:

```text id="8q2qj7"
PDF
 ↓
chunking
 ↓
dense retrieval
 ↓
top-k context
 ↓
generation
```

No query planner or agentic retrieval loop should be used.

This establishes the simplest useful reference point.

---

# 8. Hybrid Baseline

The second configuration is:

```text id="z6j5qv"
PDF
 ↓
chunking
 ↓
dense retrieval + lexical retrieval
 ↓
fusion
 ↓
reranking
 ↓
generation
```

This measures the value of retrieval engineering before introducing agents.

---

# 9. Agentic Configuration

The proposed system is:

```text id="4jv7jh"
query
 ↓
query planner
 ↓
retrieval
 ↓
fusion
 ↓
reranking
 ↓
evidence assessment
 ↓
optional refinement
 ↓
context selection
 ↓
generation
 ↓
verification
 ↓
final answer
```

All other benchmark conditions should remain equivalent.

---

# 10. Fair Comparison

When comparing systems, keep constant wherever possible:

```text id="9k3tuj"
documents
questions
ground truth
chunking
generation model
generation parameters
evaluation criteria
```

When testing a specific component, change only that component.

Example:

```text id="ebquzt"
Test reranker
→ keep embedding model and generator unchanged.
```

---

# 11. Randomness

Set deterministic seeds where supported.

At minimum:

```python id="j8jv1h"
SEED = 42
```

Record the seed in every experiment.

Some GPU/model operations may remain nondeterministic.

The notebook should document this rather than claiming perfect determinism.

---

# 12. Generation Parameters

The benchmark should use fixed generation parameters.

Initial defaults:

```text id="16w3at"
temperature = 0.0
```

Use deterministic or near-deterministic generation for benchmark evaluation where supported.

Set a fixed maximum output length.

The exact values must be recorded in the configuration.

---

# 13. Retrieval Parameters

For each benchmark run record:

```text id="svmqgf"
retrieval method
initial top-k
reranking candidate count
final context count
maximum retrieval iterations
```

Do not silently change these values between experiments.

---

# 14. Warm-Up Runs

Before measuring latency:

1. Load all required models.
2. Run one or more warm-up queries.
3. Clear stale timing measurements.
4. Begin benchmark timing.

Model download and model loading time should not be included in per-query inference latency.

Report model loading time separately.

---

# 15. Latency Measurement

Measure at least:

```text id="ezk5a3"
retrieval latency
reranking latency
generation latency
verification latency
total query latency
```

Use a monotonic timer.

For GPU operations, synchronize the GPU before measuring boundaries where necessary.

Report:

```text id="zq8o0x"
mean
median
P95
```

Avoid relying only on average latency.

---

# 16. Resource Measurement

Record:

```text id="8u8hlp"
peak GPU memory
peak system RAM
```

Measure during:

* indexing
* retrieval
* generation
* full query execution

Where supported, reset GPU memory statistics before each measured experiment.

---

# 17. Retrieval Evaluation

For every answerable question, compare retrieved chunks with ground-truth evidence.

Calculate:

```text id="20drk9"
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
```

Where graded relevance is available:

```text id="3k6pby"
nDCG@10
```

Ground-truth evidence may be defined by page or chunk.

The benchmark implementation must clearly document which representation is used.

---

# 18. Answer Evaluation

For every question, evaluate:

```text id="zqj1p2"
correctness
faithfulness
citation correctness
citation completeness
```

Use a combination of:

* deterministic checks where possible
* reference-based comparison
* model-based evaluation
* manual inspection of a sample

Do not rely entirely on an LLM judge.

---

# 19. LLM Judge Protocol

If an LLM judge is used, the judge receives only:

```text id="7fq2cd"
question
reference answer
generated answer
retrieved evidence
generated citations
```

The judge must not use web search or external retrieval.

The judging prompt must be fixed for the experiment.

The judge model and version must be recorded.

## 19.1 The judge is the same model being graded

This project's local-only constraint creates a problem the protocol must name
rather than inherit silently.

The only model available to judge is a small local one, and the only such
model already loaded is the generator. Under the Colab memory budget
(`MODEL_SELECTION.md` section 9.1) a second independent judge of comparable
capability does not fit alongside the generator, the embedder and the
reranker.

So by default the judge and the system under test are the same weights. Models
are known to score their own outputs more favourably than a third party
would, which biases every judged metric **upward**, and biases it upward
*unevenly* — the systems whose answers look most like the judge's own style
benefit most.

This is a real limitation, not a formality. Three mitigations, all required:

1. **Never report a judged score alone.** Every judged metric is reported
   beside a deterministic one measuring the same thing: correctness beside
   token-overlap and numeric agreement, faithfulness beside the fraction of
   the answer's numbers that appear in the retrieved evidence.
2. **Report judge/deterministic disagreement as a number.** The rate at which
   the two disagree is itself a result. A high rate means the judged figures
   should not be trusted, and the write-up must say so.
3. **Inspect the disagreements by hand.** Sampling at random wastes effort on
   cases where both measures already agree. Reading the cases where they
   conflict is where a scoring bug or a judge failure is actually found.

Where a second GPU or a larger session is available, run the judge as a
separate model and report both sets of scores. Any change in the conclusions
between the two is the measured size of the self-preference effect.

## 19.2 What the judge is not used for

The judge does not decide:

* **Abstention.** Detected deterministically from the answer text
  (`BENCHMARK_SPEC.md` section 13.2). A model asked whether another model
  refused will over-interpret hedging as refusal.
* **Numeric agreement.** Compared exactly. Small models are unreliable at
  precisely the comparison that matters most here — telling 86.7 from 87.6 —
  and a judge that gets this wrong corrupts the metric the numerical question
  category exists to measure.
* **Retrieval metrics.** Computed from page overlap, with no model involved.

The judge is used for what regular expressions genuinely cannot do: whether a
paraphrase means the same thing as the reference answer, and whether a claim
is supported by a passage that does not restate it word for word.

---

# 20. Judge Output

Prefer structured output:

```json id="v0n4u1"
{
  "correctness": 0,
  "faithfulness": 0,
  "citation_correctness": 0,
  "reason": "..."
}
```

Use a clearly defined scale.

Example:

```text id="k8m5yn"
0 = incorrect
1 = partially correct
2 = correct
```

The exact scale must remain consistent across experiments.

---

# 21. Abstention Evaluation

For unanswerable questions:

```text id="y0j3h7"
Correct abstention → success
Unsupported answer → failure
Incorrect refusal when evidence exists → failure
```

Calculate:

```text id="mmfl6m"
abstention accuracy      over unanswerable questions
false-answer rate        over unanswerable questions
over-abstention rate     over ANSWERABLE questions
```

Note the third metric's denominator. The first two are computed over the
unanswerable set only, so a system that refuses every question scores
perfectly on both. `over-abstention rate` is the metric that catches it, and
it can only be computed over the answerable set.

See `BENCHMARK_SPEC.md` section 13 for the full outcome matrix.

A confident unsupported answer should be treated as a significant failure.

## 21.1 How unanswerable questions enter the headline accuracy

An unanswerable question has `answer: null`, so there is no reference string
to compare against and the usual correctness measures do not apply.

The rule:

```text
unanswerable question:  correct = 1 if the system abstained, else 0
answerable question:    correct = judged/deterministic correctness as usual
```

Unanswerable questions **are** included in the headline accuracy figure, and
the composition of the set is reported alongside it. Excluding them would let
a system with a serious hallucination problem post a good headline number,
since the questions that expose the problem would not be counted.

Because the headline figure therefore mixes two different scoring rules,
always report it with its breakdown:

```text
accuracy (all)            n = 75
accuracy (answerable)     n = 69
abstention accuracy       n = 6
```

Six unanswerable questions out of 75 is a small denominator. A single
additional false answer moves abstention accuracy by 17 percentage points, so
that figure must be read with its confidence interval (section 26) and not
quoted as a precise value.

---

# 22. Citation Evaluation

For each generated citation determine:

```text id="1phw9h"
Does the cited page/chunk exist?
Does it support the associated claim?
Is an important claim missing a citation?
```

Report:

```text id="qmxzgd"
citation precision
citation completeness
```

Where exact claim-level evaluation is impractical, perform citation evaluation on a manually reviewed sample.

---

# 23. Error Analysis

Every benchmark run should preserve per-question results.

Failures should be categorized as:

```text id="t8b1tw"
retrieval failure
reranking failure
context selection failure
generation error
hallucination
citation error
abstention error
verification error
system/runtime failure
```

The purpose is diagnosis, not only producing a single score.

---

# 24. Ablation Protocol

Each major component should be removed independently.

Required ablations:

```text id="e0c5wv"
Agent planner OFF
Hybrid retrieval OFF
Reranker OFF
Retrieval refinement OFF
Evidence controller OFF
Verification OFF
```

Use the same evaluation dataset and configuration wherever possible.

---

# 25. Repeated Runs

For deterministic configurations, one complete run may be sufficient for initial development.

For final results, repeat experiments where practical.

Recommended:

```text id="5jvlv3"
3 runs
```

for configurations where model/runtime nondeterminism is significant.

Report mean and standard deviation when repeated runs are performed.

## 25.1 What repetition actually measures here

Section 12 mandates greedy decoding at `temperature = 0`, so repeated runs do
**not** sample different answers. Repetition under this protocol measures only
residual hardware nondeterminism: non-associative floating-point reductions in
cuBLAS and in some attention kernels, which can flip a token when two logits
are nearly tied, and which occasionally cascades into a different answer.

That effect is real but small. It is not the dominant source of uncertainty in
these results — **question sampling is**. With 75 questions, the sampling error
on an accuracy figure is on the order of ±5 percentage points, which is far
larger than run-to-run drift on fixed inputs.

The practical consequence:

* Repeating a greedy run three times and reporting a tight standard deviation
  is misleading. It reports the stability of the hardware, then invites the
  reader to interpret it as the reliability of the result.
* Effort is better spent on confidence intervals over questions (section 26)
  than on repeated identical runs.

Repeat runs when: the configuration samples (`temperature > 0`), the run
crashed partway, or a result is surprising enough to warrant checking that it
reproduces at all. Otherwise one run plus interval estimates is the more
honest use of the compute budget.

---

# 26. Statistical Reporting

Do not overstate small differences.

If two systems differ only slightly, report the actual values and discuss the uncertainty.

Example:

```text id="q6x6xj"
System A: 78.4%
System B: 79.1%
```

Do not describe this as a major improvement without supporting evidence.

## 26.1 The method

"Discuss the uncertainty" is not enforceable. The specific procedure:

**Every headline figure carries a 95% confidence interval**, computed by
percentile bootstrap over questions (resample the 75 per-question scores with
replacement, 2,000 times, take the 2.5th and 97.5th percentiles of the
resampled means).

**Every system-vs-baseline claim uses a paired test.** All systems answer the
same questions, so the comparison is paired and the per-question differences
are the right unit:

```text
1. For each question i, compute d_i = score_system(i) - score_baseline(i)
2. Bootstrap the mean of d over questions, 2,000 resamples
3. Report: mean difference, its 95% CI, and the two-sided p-value
```

A paired test is substantially more sensitive than comparing two independent
intervals, because it removes the variance caused by some questions simply
being harder than others.

**The reporting rule:**

> If the 95% CI of the difference includes zero, the result is reported as
> "no significant difference", regardless of the sign of the point estimate.

Applied to the example above: with 75 questions, a 0.7-point difference has a
CI comfortably spanning zero and must be written up as indistinguishable, not
as a 0.7-point gain.

## 26.2 Multiple comparisons

The ablation study (section 24) compares seven or more arms against the same
baseline. At a 5% threshold, roughly one in twenty such comparisons looks
significant by chance alone, so a sweep of this size is expected to produce a
spurious "significant" result even if every component is useless.

This does not need a formal correction, but it does need honesty in the
write-up:

* State the number of comparisons performed.
* Treat a single isolated significant result among many as a hypothesis, not a
  finding — particularly when the component has no mechanism that would
  explain the effect.
* Give more weight to components that improve a metric they plausibly *should*
  improve. A reranker improving Recall@5 is evidence; a reranker improving
  abstention accuracy and nothing else is probably noise.

## 26.3 Reporting failure honestly

If the agentic pipeline does not beat the baselines, that is a publishable
result and the write-up states it plainly. Section 18 of `BENCHMARK_SPEC.md`
already forbids claiming value from complexity alone; this is the same rule
applied to the final narrative.

A negative result with a clean methodology is worth considerably more than a
positive result obtained by adjusting the benchmark until the numbers improve.

---

# 27. Result Storage

Store experiment outputs under:

```text id="p3j19h"
results/
├── baseline/
├── hybrid/
├── agentic/
├── ablations/
└── final/
```

Each experiment should have:

```text id="1v4y4b"
config.json
results.json
per_question.json
summary.csv
```

Exact formats may be simplified if the implementation has a better consistent structure.

---

# 28. Final Comparison Table

The final notebook should produce a table similar to:

| System      | Recall@5 | MRR | Accuracy | Faithfulness | Citation | Abstention | P95 Latency |
| ----------- | -------: | --: | -------: | -----------: | -------: | ---------: | ----------: |
| Dense RAG   |          |     |          |              |          |            |             |
| Hybrid RAG  |          |     |          |              |          |            |             |
| Agentic RAG |          |     |          |              |          |            |             |

Also report:

```text
Peak GPU memory
Average retrieval iterations
Average model calls
```

for the agentic system.

---

# 29. Final Evaluation Checklist

Before submission verify:

* [ ] Same benchmark documents used.
* [ ] Same benchmark questions used.
* [ ] Ground truth manually reviewed.
* [ ] Configuration saved.
* [ ] Model versions recorded.
* [ ] Retrieval metrics calculated.
* [ ] Answer metrics calculated.
* [ ] Citation metrics calculated.
* [ ] Abstention tested.
* [ ] Latency measured.
* [ ] GPU memory measured.
* [ ] Ablations completed.
* [ ] Per-question failures inspected.
* [ ] Results reproducible from a fresh Colab runtime.

---

# 30. Interpretation Rule

Benchmark results must be reported descriptively.

The project should explain:

* which components improved which metrics,
* where performance degraded,
* what computational cost was introduced,
* which question types remain difficult,
* and what limitations remain.

Do not hide failures behind a single aggregate score.
