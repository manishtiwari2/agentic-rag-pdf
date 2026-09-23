# Design Decisions

## Purpose

This document records important architectural and engineering decisions for the Local Agentic RAG project.

Each decision should capture:

* what was chosen
* why it was chosen
* alternatives considered
* trade-offs
* how the decision will be validated

Decisions should be updated when benchmark evidence justifies a change.

---

# DD-001 — Modular Architecture

**Status:** Accepted

### Decision

Separate the system into independent modules:

```text
ingestion
chunking
embeddings
retrieval
reranking
agents
generation
evaluation
```

### Reason

The project needs to compare components independently and perform ablation experiments.

### Alternative

One monolithic notebook implementation.

### Trade-off

More files and interfaces, but significantly easier testing and experimentation.

---

# DD-002 — Notebook as Entry Point

**Status:** Accepted

### Decision

The Google Colab notebook is the primary user-facing entry point.

Core implementation should live under `src/`.

### Reason

The deliverable must be easy to run in Colab while remaining maintainable.

### Trade-off

Requires some duplication between notebook demonstration and library code.

---

# DD-003 — Local-Only Inference

**Status:** Accepted

### Decision

All generation, embedding, and reranking inference must run locally.

### Reason

The project explicitly targets free Colab and prohibits paid/cloud LLM inference.

### Alternative

Cloud APIs.

### Rejected because

They violate the project constraints.

---

# DD-004 — Small Local Generation Model

**Status:** Accepted

### Decision

Prefer a small instruction-tuned model capable of practical local inference.

Initial candidates:

```text
Qwen3-4B-Instruct-2507
Qwen2.5-3B-Instruct
Qwen2.5-1.5B-Instruct
```

### Reason

Free-Colab compatibility is a hard constraint.

### Validation

Compare answer quality, faithfulness, latency and peak VRAM.

---

# DD-005 — Embedding Model

**Status:** Experimental

### Initial candidate

```text
BAAI/bge-m3
```

### Reason

Designed for retrieval and supports broad text retrieval use cases.

### Validation

Compare retrieval recall and latency against any smaller candidate introduced later.

The final embedding model must be selected using benchmark evidence.

---

# DD-006 — Reranking

**Status:** Experimental

### Initial candidate

```text
BAAI/bge-reranker-v2-m3
```

### Reason

Reranking may improve retrieval precision after initial candidate generation.

### Validation

Compare:

```text
retrieval without reranking
vs.
retrieval with reranking
```

Measure both quality and latency.

---

# DD-007 — Dense + Lexical Retrieval

**Status:** Accepted for experimentation

### Decision

Test both semantic and lexical retrieval.

### Reason

Dense retrieval is strong for semantic similarity, while lexical retrieval can help with:

* exact terms
* names
* identifiers
* numbers
* technical terminology

### Validation

Compare dense-only, lexical-only where practical, and hybrid retrieval.

---

# DD-008 — Reciprocal Rank Fusion

**Status:** Initial candidate

### Decision

Use Reciprocal Rank Fusion as the initial deterministic method for combining retrieval results.

### Reason

It is simple, model-independent and easy to benchmark.

### Alternative

Learned score fusion.

### Trade-off

RRF may not be optimal but provides a strong, interpretable baseline.

---

# DD-009 — Structure-Aware Chunking

**Status:** Accepted

### Decision

Prefer chunks based on document structure and semantic boundaries instead of blindly splitting every N characters.

### Reason

PDFs contain:

* headings
* paragraphs
* sections
* lists
* tables

Preserving these boundaries can improve retrieval context.

### Trade-off

Parsing is more complicated than fixed-size chunking.

### Validation

Compare retrieval performance against a simple fixed-size baseline.

---

# DD-010 — Page Metadata Preservation

**Status:** Accepted

### Decision

Every chunk must retain its source page number.

### Reason

The final answer must provide useful citations.

### Required metadata

```text
document_id
page_number
chunk_id
section
```

This metadata must survive the complete retrieval pipeline.

---

# DD-011 — Explicit Agent State

**Status:** Accepted

### Decision

Represent agent execution using explicit state rather than hidden conversational state.

### Reason

Explicit state improves:

* debugging
* reproducibility
* observability
* testing

Example:

```text
query
query_type
retrieval_strategy
retrieved_chunks
evidence_status
iteration
draft_answer
verification_status
```

---

# DD-012 — Bounded Agent Loop

**Status:** Accepted

### Decision

Agentic retrieval must have a hard maximum number of iterations.

Initial value:

```text
MAX_RETRIEVAL_ITERATIONS = 2
```

### Reason

Prevents:

* infinite loops
* unpredictable latency
* excessive model calls

The value may be changed based on benchmark results.

---

# DD-013 — Baseline Before Agentic RAG

**Status:** Accepted

### Decision

Implement a simple dense RAG baseline before implementing the agentic system.

### Reason

Without a baseline, the value of agentic complexity cannot be measured.

---

# DD-014 — Evidence-First Generation

**Status:** Accepted

### Decision

The generator receives retrieved evidence explicitly and is instructed to answer from that evidence.

### Reason

The primary goal is grounded document QA rather than general chatbot behaviour.

### Failure behaviour

If evidence is insufficient, the system should abstain.

---

# DD-015 — Verification After Generation

**Status:** Experimental

### Decision

Test a verification stage that checks whether generated claims are supported by retrieved evidence.

### Reason

Generation quality alone does not guarantee grounding.

### Validation

Compare:

```text
generation
vs.
generation + verification
```

Measure:

* faithfulness
* hallucination rate
* citation correctness
* latency

---

# DD-016 — No External Knowledge During QA

**Status:** Accepted

### Decision

The answer-generation pipeline must not use web search or external knowledge retrieval.

### Reason

The benchmark evaluates understanding of the uploaded document.

### Exception

External resources may be used during development to obtain libraries, model documentation or benchmark metadata.

They must not supply answer content during evaluation.

---

# DD-017 — Configuration-Driven System

**Status:** Accepted

### Decision

Model names and important parameters must be configurable.

Configuration includes:

```text
models
chunk size
chunk overlap
top-k
rerank-k
retrieval strategy
temperature
max tokens
agent iteration limit
```

### Reason

Enables controlled experiments without modifying implementation code.

---

# DD-018 — Deterministic Components Where Possible

**Status:** Accepted

### Decision

Use normal deterministic code for deterministic operations.

Examples:

```text
ranking
metadata management
score fusion
citation formatting
configuration
result storage
```

Use LLMs only for tasks requiring semantic reasoning.

### Reason

Improves reliability, speed and debuggability.

---

# DD-019 — Evaluation Before Optimization

**Status:** Accepted

### Decision

Do not optimize components based only on intuition.

Changes should be evaluated against the benchmark.

### Reason

RAG systems often have non-obvious interactions between:

```text
chunking
retrieval
reranking
context size
generation
```

---

# DD-020 — Resource-Constrained Design

**Status:** Accepted

### Decision

Treat GPU memory and latency as first-class metrics.

### Reason

A theoretically stronger system that cannot reliably run on free Colab does not satisfy the project objective.

Every final experiment should report:

```text
peak VRAM
latency
model configuration
```

---

# DD-021 — Error Analysis

**Status:** Accepted

### Decision

Benchmarking must preserve per-question results.

### Reason

Aggregate scores cannot identify whether a failure came from:

```text
retrieval
reranking
context selection
generation
citation
verification
```

Error categories will guide future improvements.

---

# DD-022 — No Unnecessary Frameworks

**Status:** Accepted

### Decision

Use frameworks only when they provide clear value.

Prefer lightweight Python components where possible.

### Reason

The project should remain:

* understandable
* reproducible
* easy to run in Colab
* easy to debug

The implementation must not depend on a framework merely to call an LLM or perform simple retrieval.

---

# DD-023 — Final Decisions Must Be Evidence-Based

**Status:** Accepted

Initial choices are hypotheses.

After experiments, update this document with:

```text
Decision
Benchmark evidence
Observed trade-off
Final configuration
```

No component should remain in the final architecture solely because it was part of the original design.

---

# DD-024 — Qwen2.5-3B Cannot Be the Default Generator

**Status:** Accepted

### Decision

`Qwen2.5-3B-Instruct` may be benchmarked but must not be the default model.
The default generator must be Apache-2.0: `Qwen3-4B-Instruct-2507`, or
`Qwen2.5-1.5B-Instruct` in the low-memory configuration.

### Reason

License verification (`MODEL_SELECTION.md` 11.1, checked 2026-09-20) found
that `Qwen2.5-3B-Instruct` is released under `qwen-research`, which is
research/non-commercial only. Every other candidate in the pool is Apache-2.0
or MIT.

The project presents itself as open source. A default configuration that
downloads a research-only model hands every downstream user a licensing
problem they did not choose and probably will not notice.

### Trade-off

If the 3B model turns out to score best, the default is deliberately not the
highest-scoring option. The notebook must state that trade-off rather than
quietly selecting on score alone.

### Note

The restriction is per-checkpoint, not per-vendor. Other Qwen2.5 sizes are
Apache-2.0. Licences must be checked per model, not inferred from the family.

---

# DD-025 — Retrieval Metrics Are Scored at Page Level

**Status:** Accepted

### Decision

Ground-truth evidence is annotated and scored by **page**, not chunk ID.
`evidence_chunk_ids` remains an optional secondary annotation.

### Reason

Chunk IDs are a function of the chunking configuration. Scoring against them
would change the answer key whenever the chunker changes, making the
structure-aware vs. fixed-size comparison in DD-009 meaningless.

Page numbers are stable across configurations and are the unit the system
cites, so retrieval metrics and citation metrics share a scale.

### Consequence

A chunk spanning a page break counts as a hit for every page it covers.

---

# DD-026 — Both "Any" and "All" Recall Are Reported

**Status:** Accepted

### Decision

Report `Recall@K` (any gold page retrieved) and `Full-Recall@K` (all gold
pages retrieved). `Full-Recall@5` is the primary metric for `multi_hop` and
`comparison` questions.

### Reason

They are identical for single-evidence questions and diverge sharply on
multi-hop ones. A system that reliably finds one hop and never the second
scores 1.0 on the first metric and 0.0 on the second. Reporting only the
former would overstate exactly the capability the agentic loop is supposed to
provide.

---

# DD-027 — The Judge Is the Generator, and This Is Disclosed

**Status:** Accepted, with known limitation

### Decision

The LLM judge defaults to the generation model, because a second independent
judge does not fit in the Colab memory budget. Every judged metric is reported
beside a deterministic one, judge/deterministic disagreement is reported as a
number, and the disagreements are manually inspected.

### Reason

Self-grading biases judged scores upward and unevenly. The bias cannot be
removed under the local-only constraint, so it is measured and disclosed
instead of ignored.

### Alternative

A separate judge model, where hardware allows. The change in conclusions
between the two is the measured size of the effect.

---

# DD-028 — Confidence Intervals and Paired Tests Are Mandatory

**Status:** Accepted

### Decision

Every headline figure carries a bootstrap 95% CI. Every system-vs-baseline
claim uses a paired bootstrap over per-question differences. If the CI of the
difference includes zero, the result is reported as "no significant
difference", whatever the point estimate.

### Reason

EVALUATION_PROTOCOL section 26 already forbade overstating small differences
but named no method, which made the rule unenforceable. With 75 questions the
sampling error is roughly ±5 points — larger than most differences the
ablation study will produce.

---

# DD-029 — Unanswerable Questions Count Toward Headline Accuracy

**Status:** Accepted

### Decision

Unanswerable questions score 1 for abstaining and 0 for answering, and are
included in the headline accuracy figure. Accuracy is always reported with its
breakdown (all / answerable / abstention) and with the denominator for each.

### Reason

Excluding them would let a system with a serious hallucination problem post a
good headline number, since the questions that expose the problem would not be
counted.

### Trade-off

The headline figure mixes two scoring rules, which is why the breakdown is
mandatory rather than optional.

---

# DD-030 — Chunk Sizes Are Measured in Characters

**Status:** Accepted

### Decision

`chunk_size` and `chunk_overlap` are character counts, not token counts.
Roughly four characters per token.

### Reason

A token-based size binds the chunker to one specific tokenizer. `ARCHITECTURE.md`
section 4 requires the layers to be independently replaceable, and a chunker
that must load the generation model's tokenizer to decide where a paragraph ends
couples the two in a way that would have to be undone before any model
comparison in `MODEL_SELECTION.md` section 12.

### Trade-off

The character budget corresponds to slightly different token counts for
different tokenizers, so a context-window calculation needs the ~4x conversion
rather than being exact. That is the cheaper error: it is visible and constant,
where the coupling would be invisible and structural.

### Validation

`EXPERIMENT_PLAN.md` section 5.1 already requires a chunk-size sweep on the dev
set. It sweeps characters.

---

# DD-031 — The Generator Is Never Shown a Page Number

**Status:** Accepted

### Decision

Evidence blocks in the prompt carry a marker and text only. No page number, no
chunk id, no section title, no document name. The model emits markers such as
`[C1]`; `generation/citations.py` maps each marker back to the pages stored on
the chunk.

### Reason

DD-018 says deterministic operations stay deterministic code, and mapping a
marker to a page is a dictionary lookup. The stronger reason is structural: a
model that has never seen a page number cannot copy one incorrectly. Instructing
a 4B model not to write page numbers is a request it will mostly honour;
omitting them from its input is a guarantee.

Any page reference that does appear in generated text is therefore known to be
fabricated, and is counted and reported rather than trusted.

### Consequence

Out-of-range markers are **dropped, not repaired**. If the model writes `[C7]`
against a five-item list, no rule maps it to a real page without inventing
support the model never claimed. Dropped markers are counted so the rate is
visible.

### Validation

`tests/test_generation.py` asserts the rendered prompt contains no page
information, and that out-of-range markers are dropped rather than resolved.

---

# DD-032 — The Fixed-Size Baseline Does Not Record Sections

**Status:** Accepted

### Decision

`FixedSizeChunker` leaves `section` as `None`. Every other field DD-010
requires — `document_id`, `page_number`, `pages`, `page_span`, `chunk_id` — is
populated as normal.

### Reason

DD-009 claims structure-aware chunking beats fixed-size splitting, and the
fixed-size chunker exists to test that claim. A baseline that consulted headings
in order to label its chunks would already be partly structure-aware, and the
comparison would understate the difference it was built to measure.

### Trade-off

Citations from the fixed-size configuration name a page but not a section. That
is a real reduction in citation quality, and it is part of what the comparison
is measuring rather than a defect to be patched.

---

# DD-033 — The Default PDF Backend Is pdfplumber, Not PyMuPDF

**Status:** Accepted — resolved 2026-09-21

### Finding

PyMuPDF is AGPL-3.0 unless a commercial licence is purchased. It is the only
copyleft dependency in the Phase 1–2 stack; `numpy`, `faiss-cpu` and `pytest`
are BSD/MIT. `MODEL_SELECTION.md` section 11.3 lists the dependency audit as
outstanding, and this is its first result.

### Consequence

The project cannot describe itself as permissively licensed while depending on
PyMuPDF. Either the project accepts AGPL terms, or the parser is replaced.

### Mitigation already in place

All PDF-library use is confined to `src/ingestion/`, enforced by
`tests/test_architecture.py`, which fails if any other layer imports a PDF
library. Switching to `pypdf` (BSD-3-Clause) or `pdfplumber` (MIT) means adding
one parser class behind the existing `PdfParser` protocol and changing one
configuration value. Table detection and font-size metadata would need
reimplementing, which would weaken heading classification.

### Decision

`pdfplumber` (MIT) is the default parser. PyMuPDF remains available behind the
same `PdfParser` protocol as `ingestion.parser='pymupdf'`, for anyone who
accepts AGPL terms and wants its speed.

Chosen over `pypdf` (BSD-3-Clause) because pdfplumber preserves both
capabilities the structure-aware chunker depends on: table detection, so tables
still get their own chunk, and per-character font sizes, so heading
classification still works. pypdf would have cost both.

The full dependency chain is permissive: pdfminer.six (MIT), pypdfium2
(BSD-3-Clause / Apache-2.0), Pillow (MIT-CMU). The project is therefore free to
choose a permissive licence for itself.

### What it cost

pdfplumber reports words and lines rather than layout blocks, so
`pdfplumber_parser.py` groups lines into blocks itself, splitting where vertical
spacing or font size changes. That grouping is what lets the chunker tell a
heading from a paragraph, so it is a deliberate component rather than a detail
of the library.

Shared post-processing — header stripping, hyphen joining, page assembly, the
scanned and blank checks — moved into `parser_base.BaseParser`. It was inside
the PyMuPDF class, where a backend swap would have meant reimplementing the part
of ingestion that the page-attribution guarantees actually rest on.

### Validation

The ingestion test suite is parametrized over every installed backend, so the
two implementations cannot drift apart. On the demo corpus both produce
identical block structure, identical chunk counts and zero mis-attributed pages.
`tests/test_architecture.py` fails if any layer outside `src/ingestion/` imports
a PDF library.

### Residual wart

PyMuPDF remains a **test-only** dependency: `tests/pdf_fixtures.py` writes the
synthetic PDFs, and pdfplumber cannot write PDFs. It is in the `dev` extra, not
in the runtime dependencies, so it does not reach anyone who installs the
package. Removing it would mean generating fixtures with reportlab and dropping
the AES-256 encrypted fixture.

### Reason this was decided now

The choice was cheap today and would have been expensive after the benchmark had
been run against one parser's output.


---

# DD-034 — The Notebook Obtains `src/` by Cloning at a Pinned Ref

**Status:** Accepted — resolves `EXPERIMENT_PLAN.md` section 5.4

### Decision

The Colab notebook's setup cell clones this repository at a pinned tag or commit
and puts it on `sys.path`:

```python
!git clone --depth 1 --branch <tag> <repo-url> /content/agentic-pdf-rag
import sys; sys.path.insert(0, "/content/agentic-pdf-rag")
```

The pinned ref is recorded with every benchmark run, alongside the config
fingerprint that `RAGConfig.fingerprint()` already produces.

### Reason

`PROJECT_SPEC.md` section 12 requires the notebook to be a reproducible entry
point rather than the implementation. Three options were considered:

* **Clone at a pinned ref** — one source of truth; the code stays readable and
  editable inside the session, which matters while the project is still being
  developed; pinning makes a run reproducible.
* **`pip install git+...`** — cleaner import semantics, but reinstalls on every
  fresh runtime and makes editing a file mid-session awkward.
* **Write the files inline from notebook cells** — self-contained, but
  duplicates four thousand lines into the notebook and guarantees drift.

### Consequence

The notebook is not self-contained: it needs network access to the repository.
That is accepted, because it already needs network access to download model
weights from Hugging Face, so no new failure mode is introduced.

Unpinned cloning of the default branch is explicitly rejected. A benchmark run
that cannot say which commit produced it is not reproducible, which
`MODEL_SELECTION.md` section 15 requires.


---

# DD-035 — The Project Is Licensed Apache-2.0

**Status:** Accepted — resolves `README.md` section 20 and
`EXPERIMENT_PLAN.md` section 5.5

### Decision

Apache-2.0. `LICENSE` holds the licence text; `NOTICE` records the licences of
dependencies, model weights and (once Phase 0 chooses them) benchmark documents.

### Reason

Chosen over MIT for the explicit patent grant in section 3, which MIT does not
provide. It also matches the licence of the default model stack —
`Qwen3-4B-Instruct-2507` and `bge-reranker-v2-m3` are Apache-2.0 — so a user
reading the repository sees one set of terms rather than two.

The choice was only available because DD-033 replaced the AGPL-3.0 PDF backend.
Had PyMuPDF remained a runtime dependency, this decision would have been made
for the project rather than by it.

### Scope, stated because it is routinely confused

The project licence covers **this repository's code**. It does not cover:

* **Model weights**, downloaded at runtime under their own terms. `NOTICE`
  records them, and DD-024 keeps a non-commercially-licensed checkpoint out of
  the default configuration.
* **Benchmark documents**, not yet chosen. `BENCHMARK_SPEC.md` section 4
  requires a source URL and licence for each before release.
* **PyMuPDF**, which is AGPL-3.0 and remains available as an optional parser
  backend. Selecting it means accepting copyleft terms for the result. It is
  not installed by default.

### Copyright holder

The notice currently reads `The agentic-pdf-rag authors`, which is valid and
avoids asserting a legal name that has not been stated. Replace it with a
personal or organisational name if individual attribution is wanted; it appears
in `LICENSE` and `NOTICE`.

---

# DD-036 — Deterministic Correctness Is Token Overlap Plus Exact Numeric Agreement

**Status:** Accepted

### Decision

An answerable question is scored correct when **both** hold against the
reference answer (or any `alternative_answers` entry):

```text
token overlap  >= 0.6    share of the reference's content tokens present in the answer
numeric agreement        every number in the reference appears in the answer, exactly
```

`CORRECTNESS_OVERLAP_THRESHOLD` and the tokenizer live in
`src/evaluation/metrics.py` and are recorded in every result set.

### Reason

`EVALUATION_PROTOCOL.md` section 18 requires deterministic checks and forbids
relying entirely on a judge. Section 19.2 puts numeric agreement out of the
judge's reach specifically because telling 86.7 from 87.6 is the comparison
that matters most and the one a small model is least reliable at, so numbers
are a separate gate rather than tokens among tokens.

Overlap is **recall-oriented, not F1**. Reference answers here are a phrase and
system answers are extracted sentences; an F1 would mark a correct answer down
for the sentence it was extracted from.

The threshold was fixed before any results existed, per `EXPERIMENT_PLAN.md`
section 3: a threshold chosen after seeing the numbers is not a threshold.

### Known weakness

A system that emits the whole document scores well on overlap. That is why
`answer_chars` is recorded per question, why citation and faithfulness metrics
are reported beside correctness rather than folded into it, and why the judge
(DD-027) exists for the paraphrase case at all. It is a measured weakness, not
an unnoticed one.

### Consequence

The evaluation layer keeps its **own** stopword list and tokenizer, deliberately
separate from the one in `src/generation/backends.py`. A scorer sharing the
system's tokenizer flatters the system, and the scripted backend is one of the
systems this scorer grades.

---

# DD-037 — Faithfulness Is Two Numbers, and the Numeric One May Be Undefined

**Status:** Accepted

### Decision

```text
faithfulness_token     share of the answer's content tokens present in the retrieved evidence
faithfulness_numeric   share of the answer's numbers present in the retrieved evidence
```

`faithfulness_token` is the headline figure because it is always defined.
`faithfulness_numeric` is `null` when the answer contains no numbers.

### Reason

`EVALUATION_PROTOCOL.md` section 19.1 names the numeric measure explicitly as
the deterministic counterpart the judged faithfulness score must be reported
beside. But most answers contain no numbers, and returning 1.0 for those would
credit every number-free answer with perfect grounding — inflating the metric
exactly where it has no evidence. `BENCHMARK_SPEC.md` section 14 still requires
a faithfulness figure for every configuration, so a second, always-defined
measure is needed alongside it.

### Consequence

`unsupported_answer_rate` (section 13, over **all** questions) is derived from
these: an answer is unsupported when it carries no citation at all, or contains
a number absent from the evidence it was generated from.

---

# DD-038 — The Error Taxonomy Is an Ordered, Versioned, First-Match Rule Set

**Status:** Accepted

### Decision

Nine categories, tried in this order, first match wins:

```text
1  system-runtime      the run raised
2  retrieval           no gold page on the answer path
3  reranking           [reserved] a reranker had it and lost it
4  context-selection   retrieved, then dropped before the generator saw it
5  abstention          refused an answerable question, or answered an unanswerable one
6  hallucination       answered from evidence, with something the evidence does not say
7  citation            right answer, wrong pages
8  verification        [reserved] a verifier ran and did not pass
9  generation          terminal: had what it needed, still wrong
```

`ERROR_TAXONOMY_VERSION` is recorded with every result set, like
`ABSTENTION_PATTERNS_VERSION`.

### Reason

`EVALUATION_PROTOCOL.md` section 23 and `BENCHMARK_SPEC.md` section 20 both
require every failure to be categorised, and DD-021 says the point is diagnosis
rather than a score. A failure with two categories diagnoses nothing.

Ordering makes exclusivity **structural** rather than a property the predicates
happen to have: one `return`, so one category. The order is
most-upstream-cause-first, because a question whose evidence was never retrieved
is a retrieval failure whatever the generator then did with the wrong passages —
blaming the generator would point Phase 4 at the wrong component.

`generation` is terminal rather than a peer, so a failing record can never fall
through to no category at all.

### Reserved categories

`reranking` and `verification` describe components DD-013 deliberately leaves
out of the dense baseline. Rather than dropping them (both specifications name
them) or building the components early, each fires only for a system that
**declares the stage** in the record's `stages`. One taxonomy therefore serves
Phases 3, 4 and 5, and a test exercises the reserved rules against a synthetic
record without either component existing.

### Consequence

Blame follows the **answer path**, not the deeper probe of DD-039: a gold page
ranked eighth never reached the generator, so it is a retrieval failure and not
a context-selection failure.

---

# DD-039 — Retrieval Is Scored at Depth 10 by a Read-Only Probe

**Status:** Accepted

### Decision

`DenseRAGPipeline.retrieve(question, k)` ranks chunks without generating. The
harness calls it at depth 10 for the retrieval metrics, while the answer comes
from `ask()` at the configured `top_k` (5), untouched.

### Reason

`BENCHMARK_SPEC.md` section 10 requires Recall@10 and MRR@10, and defines them
over "the top K **retrieved chunks**" — the retriever's ranking, not the
generation context. With `top_k = 5` and no probe, Recall@10 would be a copy of
Recall@5 under a bigger name, and Phase 4 could not compare a hybrid system that
ranks deeper.

`ARCHITECTURE.md` section 26 requires benchmark instrumentation not to change
answer behaviour. The probe reads the index and returns; it mutates nothing. A
test asserts the stronger property directly: a harness-run `RAGResult` equals a
bare `pipeline.ask()` `RAGResult` on every field but the timings.

### Alternative rejected

Running the benchmark at `top_k = 10`. That changes what the generator sees,
which changes Baseline A into a different system.

### Consequence

`QueryableSystem` makes `retrieve` optional. A system without one is scored on
what its `ask` returned, and the shallower depth is recorded in the per-question
record's `capped_ks` rather than hidden.

---

# DD-040 — nDCG Is Reported Under Binary Relevance, and Labelled As Such

**Status:** Accepted

### Decision

`ndcg_at_10` is computed with binary gains and every record carries
`ndcg_relevance_scale: "binary"`.

### Reason

Both specifications condition nDCG on graded labels — "Use nDCG when graded
relevance labels are available" (`BENCHMARK_SPEC.md` section 10), "Where graded
relevance is available" (`EVALUATION_PROTOCOL.md` section 17). This dataset has
none: `evidence_pages` is a set, not a ranking. Reporting the figure silently
would pass a binary computation off as the graded one the specs describe;
omitting it entirely would discard a usable secondary diagnostic. Labelling it
does neither.

---

# DD-041 — The Judge Is Optional, Off by Default, and a Skip Is Never a Zero

**Status:** Accepted — implements DD-027

### Decision

`--judge` enables it. Nothing the Phase 3 exit criterion asks for depends on it.
Every failure path — no model, no `transformers`, a refusal to load, an
unparseable reply, the offline scripted backend — records `judge_skipped: true`
with its reason and leaves the judged fields `null`. The run continues.

### Reason

DD-027 already established that the only judge fitting the memory budget is the
generator grading itself. A harness whose headline figures depended on that
would be reporting a bias as a result. And a judged metric defaulting to 0 on a
skip is worse than no metric: it is a number that looks like a measurement.

### What is recorded

The fixed prompt, the scale, the model id, the parse-failure rate
(`EXPERIMENT_PLAN.md` section 4 names unreliable structured output from a small
model as a likely risk and asks for it to be measured), and — section 19.1's
second mitigation — the **judge/deterministic disagreement rate**.

### Scope

The judge never decides abstention, numeric agreement or retrieval metrics
(section 19.2). A test asserts that `src/evaluation/judge.py` does not so much
as import the abstention detector.

---

# DD-042 — Citation Completeness Is Sentence-Level Marker Coverage

**Status:** Accepted

### Decision

```text
citation precision          cited pages that are gold / cited pages
citation completeness       answer sentences carrying a marker / substantive sentences
citation any-correct        1.0 when at least one cited page is gold
citation gold-page recall   gold pages cited / gold pages
```

Pages come from `Citation.pages` and are never re-read out of the answer text.
Precision is `null`, not 0.0, when nothing was cited.

### Reason

`EVALUATION_PROTOCOL.md` section 22 asks whether an important claim is missing a
citation and then concedes that claim-level evaluation needs a manually reviewed
sample. Sentence-level marker coverage is the deterministic proxy, and is
labelled as a proxy rather than presented as claim-level scoring.

Reading pages from the answer text would measure a fabrication: DD-031 means the
generator has never been shown a page number, so any it writes is invented.
Those mentions are counted separately as `fabricated_page_mentions`.

Precision over an empty set is undefined, not zero; averaging it as zero would
conflate "cited badly" with "did not cite". `citation any-correct` **is** 0.0 in
that case, because `EXPERIMENT_PLAN.md` section 3 sets a stopping rule on it and
an uncited answer fails that rule.

---

# DD-043 — Abstention Is Scored From the Answer Text, With the Self-Report Beside It

**Status:** Accepted

### Decision

The harness calls `src/generation/abstention.py::is_abstention` on the answer
text. `RAGResult.abstained` — the system's own flag — is recorded beside it as
`self_reported_abstention`, with `abstention_agrees_with_self_report`.

### Reason

`BENCHMARK_SPEC.md` section 13.2 requires abstention to be detected from the
answer text rather than from system self-report. Grading a system on its own
flag lets the system decide its own abstention accuracy.

Reusing the existing detector rather than writing a second one is not
convenience: the patterns are versioned and must be fixed before the run, and
two copies would be two things to keep in step, with the copy further from the
generator drifting. A test in `tests/test_architecture.py` enforces that the
harness imports the detector and hard-codes no refusal phrase of its own.

### Consequence

A disagreement between the two is visible per question rather than silently
resolved in the system's favour.
