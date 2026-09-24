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

---

# DD-044 — A Gold Page Is Lost "by the Reranker" Only If Reranker-OFF Would Have Kept It

**Status:** Accepted — changes DD-038's rule 2

### Decision

`reranking` fires when a gold page was in the top-k the reranker was handed —
what the generator would have received with the reranker off — and no gold
page is in the top-k it returned. The system reports that counterfactual itself
(`metadata["pre_rerank_pages"]`, beside `metadata["stages"]`), and the harness
reads it without knowing which system produced it.

A gold page that was only deeper in the pool and never promoted, or that fusion
dropped before the reranker saw it, is a `retrieval` failure. The taxonomy has
no fusion category, and fusion is part of the retrieval stage.

### Reason

DD-038 reserved `reranking` ("a reranker had it and lost it") but ordered it
*after* `retrieval`, whose trigger was "no gold page on the answer path". A gold
page the reranker pushes out is by definition missing from the answer path, so
rule 2 always fired first and rule 3 was unreachable. Phase 3 could not notice:
no system declared the stage and no caller ever passed the flag. Rule 2 now
excludes the reranker-lost case.

The counterfactual boundary is the one an ablation can check. "Reranker OFF"
(EVALUATION_PROTOCOL.md section 24) sends exactly the pre-rerank top-k, so every
question filed as `reranking` is one the OFF arm answers from a gold page.

### Consequence

`ERROR_TAXONOMY_VERSION` and `SCORING_RULES_VERSION` become `2026-09-23.1`. No
formula or threshold changed, and a record declaring no reranking stage is
classified exactly as before: `TestBaselineAIsUnchanged` re-runs Baseline A and
requires its stored per-question records, categories included.
`metrics.SCORE_COMPATIBLE_VERSIONS` records `2026-09-21.1` as score-compatible
on that evidence, so `results/baseline/` is compared as published rather than
regenerated.

---

# DD-045 — RRF With k = 60, Exact Arithmetic and a Fixed Tie-Break

**Status:** Accepted — implements DD-008

### Decision

`rrf(d) = sum 1/(60 + rank)` over the rankings containing `d`, summed as
`fractions.Fraction`. Scores are never read. Ties are broken by, in order: the
best single rank; the higher-priority ranking among those giving that rank
(`retrieval.retrievers` order, dense first); document position; chunk id.

### Reason

k = 60 is the value from Cormack et al. (2009) and is not tuned here: open
question 2 in `EXPERIMENT_PLAN.md` section 5 stays open until a dev-set sweep.
Exact arithmetic makes ties exact — ranks (1, 3) and (3, 1) are mathematically
equal, and with three or more rankings float addition would order them by
summation order. Every tie-break key is a function of the input rankings, so a
test can require the fused order to survive any rescaling, or replacement, of
the input scores.

### Alternative

Normalized score fusion, which needs dense and BM25 scores calibrated against
each other. `retrieval.fusion` is a field, so a second method is a
configuration change.

---

# DD-046 — Each Retriever Ranks to 20, and the Reranker Sees the Fused Top 20

**Status:** Accepted, untuned

### Decision

`retrieval.candidate_k = 20` and `reranking.candidates = 20`. With the reranker
off, the fused top 20 is the ranking, unchanged. The ranking never depends on
the `k` a caller asks for.

### Reason

The pool must be at least the harness's probe depth (10, DD-039), or Recall@10
would be capped. At twice that depth the reranker can promote from beyond depth
10. The pool is identical with the reranker on and off, so the ablation changes
one thing. Cost is linear in pool size: 20 cross-encoder pairs per question.

---

# DD-047 — The Paired Test's p-Value Is the One Its Percentile Interval Implies

**Status:** Accepted — implements DD-028

### Decision

2,000 resamples, seed 42, percentile CI (the 2.5th and 97.5th percentiles of
the resampled mean differences), exactly as EVALUATION_PROTOCOL.md 26.1 states.
The two-sided p-value is
`min(1, 2 * min(#{mean* <= 0} + 1, #{mean* >= 0} + 1) / 2001)`.

### Reason

Section 26.1 asks for a p-value and does not name a method. This one comes from
the same bootstrap distribution the CI is read from, so the two never disagree
about which side of zero a result is on. The +1 correction means a finite
resample cannot claim p = 0. Identical systems give exactly 1.0 and CI [0, 0];
the smallest reportable value is 2/2001. Both are tested as known answers.

The verdict follows the CI alone: zero inside the interval, including at its
boundary, means "no significant difference".

### Consequence

Per-question vectors are read from stored `per_question.json` records, so
Baseline A is compared as published. Records round to 4 dp, so a vector's mean
matches the summary figure to within 5e-5; a test holds it there.

---

# DD-048 — The Offline Reranker Is Term Coverage, and the Primary Metrics Were Named Before the Run

**Status:** Accepted, with known limitation

### Decision

`RAGConfig.offline()` reranks with `local/term-overlap-reranker`, which scores
each (query, passage) pair on the share of the query's distinct content words
the passage contains, plus half the share of its adjacent word pairs.

Declared before the Phase 4 run, per EVALUATION_PROTOCOL.md 26.2:

| | Primary | Secondary |
| --- | --- | --- |
| RQ1 (hybrid vs dense) | Recall@5 | MRR@10, Full-Recall@5 (multi) |
| RQ2 (reranker on vs off) | accuracy (all), faithfulness; latency as cost | Recall@5, MRR@10 |

### Reason

`--offline` has to run Baseline B end to end with no weights, just as it runs
Baseline A. The fallback is deliberately a different signal from BM25 (no IDF,
no TF saturation, no length normalization), so it actually reorders the fused
pool rather than restating one of its inputs.

### Limitation

It is not a cross-encoder. Any RQ2 finding measured with it is a finding about
it, not about `bge-reranker-v2-m3`, and its latency says nothing about the T4.

---

# DD-049 — BM25 With Whole Numbers, No Stemming, and No Zero-Score Hits

**Status:** Accepted

### Decision

Okapi BM25, k1 = 1.2 and b = 0.75, with Lucene's non-negative IDF. The tokenizer
lowercases, keeps `9.4` as one token and folds `1,200` to `1200`. There is no
stemmer and no stopword list. A chunk scoring 0 is not returned.

### Reason

Numbers and identifiers are what DD-007 adds a lexical retriever for, and
splitting `3.14` into `3` and `14` would give a lexical retriever the dense
retriever's weakness. IDF already drives a ubiquitous word towards zero, and a
stemmer would blur the exact-match signal. Ranking zero-score chunks would hand
fusion ranks decided by document order.

### Validation

`tests/test_hybrid.py` builds a parts-catalogue fixture: three distractors
carry identifiers one digit away and repeat every query word, and the target
states `884211` once. BM25 ranks the target first. The offline dense retriever
ranks it fifth, behind the near-identical identifiers.

---

# DD-050 — The Agentic System Is Baseline B's Retrieval Driven by an Explicit, Bounded Loop

**Status:** Accepted — implements EVALUATION_PROTOCOL.md section 9

### Decision

`AgenticRAGPipeline` (`retrieval.strategy = "agentic"`) subclasses Baseline B.
Retrieval, fusion and reranking are Baseline B's `_pools`, called once per query
the system issues. Chunking, evidence assembly, generation, citations and
abstention are the shared layers, unchanged. What it adds is the loop
planner → retrieval → assessment → refinement → context → generation →
verification, and the four components in `src/agents/`.

Each component is switched by one configuration field: `agents.planner_enabled`,
`evidence_controller_enabled`, `refinement_enabled` and `verification_enabled`.
A switched-off component is never built. Refinement is triggered only by the
controller, so switching the controller off also means a single round. With all
four off, the system answers exactly as Baseline B does: one query's ranking is
never re-scored.

Every loop has a cap from configuration: retrieval rounds
(`max_retrieval_iterations`, default 2), sub-queries (`max_sub_queries`, 3) and
regenerations (`max_regenerations`, 1). **The loop enforces each cap, not the
component.** A controller that asks for another round forever still stops. The
cap and whether it was hit are written into every trace.

The probe (DD-039) re-runs the retrieval loop without generating and returns
the final merged ranking. It is stateless, and `ask`'s context is the first
`top_k` of that ranking.

### Reason

EVALUATION_PROTOCOL.md section 10: change only the component under test. Reusing
B's retrieval makes agentic-vs-B the comparison that isolates the agent loop,
and one field per component makes the section 24 ablations configuration.

---

# DD-051 — The Planner Decomposes Multi-Hop and Comparison Questions by Clause, Original First

**Status:** Accepted, untuned

### Decision

The rule-based planner assigns one question type from ordered cue patterns:
comparison, multi_hop, numerical, definition, summary, else factual. It never
emits `unanswerable`, because rules cannot tell. Only `multi_hop` and
`comparison` questions are decomposed. Such a question is split on generic
clause boundaries (`;`, sentence breaks, `, and`, `and what/how/which…`,
`versus`, `compared to`, `differ from`), and lead-ins ("Combining", "Using
both") are stripped. The original question is always query 0, duplicates are
dropped, and the list is capped at 3.

Sub-query rankings, each Baseline B's reranked pool, are fused by the same RRF
the retrievers use (DD-045), with the original question first in priority. The
retrieval strategy is always executed as `hybrid`. An LLM planner's suggested
strategy is recorded and not acted on.

The LLM planner asks for a JSON plan. Invalid JSON or schema falls back to the
rules, with the failure counted (DD-055).

### Reason

Keeping the original first means decomposition can add evidence but cannot
lose the un-decomposed ranking. RRF across sub-queries puts each hop's best
chunks into the top k, which a single ranking by the whole question does not
guarantee. Fixing the strategy keeps RQ3 about decomposition and refinement,
not a retriever switch.

### Disclosure

The benchmark questions are in the repository and were visible while these
rules were written. The rules are generic English clause structure, not fitted
to any question, but that is a statement of intent, not a validation.

---

# DD-052 — Evidence Is Sufficient When Every Query Is Half-Covered and Every Question Number Is Present

**Status:** Accepted, untuned

### Decision

The rule-based controller calls the top-k context sufficient iff:

* for every query issued, at least `agents.sufficiency_threshold = 0.5` of its
  distinct content terms appear in the union of the context text, compared as
  crude suffix-stripped word forms; and
* every number in the original question appears in the context.

Sufficient → answer. Insufficient with budget left → retrieve again.
Insufficient at the cap, or with no new query to issue → **abstain** without
calling the generator (DD-014, ARCHITECTURE.md section 24).

The LLM controller returns section 12's JSON. Its sufficiency verdict decides,
and the budget rule is the same for both controllers.

### Reason

A question whose terms and figures are largely absent from the best evidence
the system can find is the case DD-014 says to refuse. 0.5 is a round number
set before any agentic result existed. The stemming was added while
smoke-testing on the synthetic fixture PDF, before any benchmark run, because
"embeds" did not cover "embedded". The threshold is to be swept only on
`--split dev` in Phase 6.

### Consequence

This is the only component that can move abstention on the offline stack, and
it can equally cause over-abstention. Both are measured over their own
denominators.

---

# DD-053 — Refinement Rewrites Each Under-Covered Query From Its Missing Terms

**Status:** Accepted

### Decision

Refinement is deterministic in every configuration. For each query below the
coverage threshold, plus the original question when a number was missing, the
refined query is up to two covered terms as topic anchors, then the missing
terms, then the missing numbers. A refined query whose term set matches one
already issued is dropped. If none remain, the loop stops with `no_new_query`
and abstains. New rankings join the RRF merge; earlier ones are kept.

`max_retrieval_iterations` counts total rounds, the first included.

When a second round ran, the pipeline also generates, diagnostically, from the
first round's context. It records `refinement_changed_answer`: whether that
draft differs from the draft generated from the final context. The diagnostic
call is counted in `diagnostic_model_calls`, never in `model_calls`, and never
reaches the answer.

### Reason

Section 13's first option, aimed at the gap the controller measured.
Re-running a query already issued cannot retrieve anything new. The
counterfactual is how open question 3 ("does iteration 2 ever change an
answer?") is answered, independently of what the controller or verifier then
did with the answer.

### Clarification (2026-09-24, issue #12)

A missing question number does not always produce a new query. The refined
query is at most two covered terms, then the missing terms, then the missing
numbers. For a question with two or fewer covered terms, that is the
question's own term set, so the dedup rule above drops it and the loop stops
with `no_new_query`. The code has always done this. A test in
`tests/test_agentic.py` expected `["latency 2019"]` for such a question, and a
malformed `or` clause hid the mismatch. The test was wrong, not the code, so
no stored result changes. The test now asserts `[]` for that case, and a
second test covers a question with enough anchors to yield a new query.

---

# DD-054 — Verification: Claim-Level Rules, One Bounded Regeneration, and a Counterfactual Taxonomy Rule

**Status:** Accepted — changes DD-038 rules 5 and 8

### Decision

**Rule-based verifier.** Each substantive sentence is a claim, and a claim is
supported when all three hold:

* it carries a resolvable marker, its own or the one closing its run of
  unmarked sentences;
* every number in it appears in the evidence it cites; and
* at least 0.5 of its word forms appear in that cited evidence.

`SUPPORTED` means every claim passes.

**Status mapping.** SUPPORTED → `passed`. UNSUPPORTED or UNCERTAIN → `failed`.
A verifier that raised, or an LLM verifier whose output did not parse, gives
`unavailable`. It is never `passed`, and there is no fallback to the rules,
because a substituted weaker check would report a verification that did not
happen.

**Action.** On `failed`, regenerate up to `max_regenerations = 1` times,
dropping the evidence the unsupported claims cited, and re-verify. If the
answer still fails, abstain. On `unavailable`, keep the answer, marked
unverified.

**Taxonomy.** The system reports its first draft as `metadata["draft_answer"]`,
which is what Verification OFF would return. The harness scores the draft with
the same rule as the answer. When a declared verifier turned a correct draft
into a wrong final answer, the record is filed as `verification`, not
`abstention` (rule 5 excludes it and rule 8 takes it). Separately, a
failed/unavailable verification makes a record a failure only when an answer
was returned: a verifier that refused an unanswerable question has done its job.

The harness now reads `verification_status` from the result's metadata.
Phase 3 hard-wired `not_run`, which no verifying system could have survived.

### Reason

Without the counterfactual, rule 8 is unreachable for the verifier's most
consequential mistake. A rejected correct draft becomes a refusal of an
answerable question, and rule 5 fires first. This is the same defect DD-044
fixed for `reranking`, fixed the same way.

### Consequence

`SCORING_RULES_VERSION` and `ERROR_TAXONOMY_VERSION` → `2026-09-23.2`. No
formula or threshold changed. For a record declaring no verification stage
every rule is as before, and `2026-09-23.1` is recorded as score-compatible.

### Clarification (2026-09-24, issue #12)

"Re-verify" applies to an answer, not to a refusal. A regeneration whose draft
abstains counts toward `regenerations_used`: it was a generator call. The loop
then stops without calling the verifier, because a refusal has no claims to
check. So the verifier is called once per generation that did not abstain,
which is not always `regenerations_used + 1`. The code has always done this. A
test assumed the `+ 1` formula and failed at a cap of 2, where the second
regeneration abstains. The test was wrong, not the code, so no stored result
changes.

---

# DD-055 — Offline Rule-Based Stand-Ins, LLM Fallbacks, Parse-Failure Rate and Model-Call Accounting

**Status:** Accepted, with known limitation

### Decision

`RAGConfig.offline()` sets the planner, controller and verifier to `"rules"`:
`local/rule-based-planner`, `local/rule-based-evidence-controller` and
`local/rule-based-verifier`. These names appear wherever a record names the
component that decided. `"llm"` components share the generator's backend, so
no second model is loaded, and they are refused on the scripted backend.

An unparseable LLM decision falls back to the rules for the planner and the
controller, and becomes `unavailable` for the verifier (DD-054). Every LLM
decision and parse failure is counted per question. The summary reports
`llm_parse_failure_rate` over all LLM decisions (EXPERIMENT_PLAN.md section 4).

`model_calls` counts generation-model invocations made by the system: planner,
controller, generator and verifier. Reranker passes are `reranker_calls`, and
diagnostic calls are `diagnostic_model_calls`. The summary's
`retrieval_iterations_mean` and `model_calls_mean` (EVALUATION_PROTOCOL.md
section 28) are read from any record that reports them. They are not an agentic
special case, and they are null for systems that report none.

### Limitation

Offline figures measure the stand-ins, not an LLM planner, controller or
verifier, just as DD-048's reranker finding is about the term-overlap
stand-in. The LLM paths are exercised only against canned outputs. Their
behaviour with a real model is unmeasured, and the parse-failure rate offline
is undefined (zero LLM decisions).

---

# DD-056 — RQ3, RQ4 and the Keep-or-Drop Metrics Were Named Before the Agentic Run

**Status:** Accepted — declared before `results/agentic/` existed

### Decision

Declared per EVALUATION_PROTOCOL.md 26.2, and recorded in
`statistics.PRIMARY_METRICS`:

| | Primary | Secondary | Cost |
| --- | --- | --- | --- |
| RQ3 (agentic retrieval on hard / multi-hop) | Full-Recall@5 (multi_hop + comparison), accuracy (multi_hop + comparison) — n = 15 | accuracy (all), Recall@5, MRR@10 | latency, model calls, iterations |
| RQ4 (verification vs unsupported answers) | unsupported-answer rate | false-answer rate, faithfulness, citation precision, `verification` category count | — |

**Keep-or-drop (EXPERIMENT_PLAN.md section 3).** The agentic system is kept
only if at least one of accuracy (all), faithfulness, citation any-correct or
abstention accuracy improves over **Baseline B** by a margin whose 95% CI
excludes zero. Baseline B is the decisive comparison because the agentic
system reuses B's retrieval, so agentic-vs-B isolates the agent loop.
Agentic-vs-A is reported beside it. Otherwise the report says the added
complexity did not pay for itself. The number of comparisons is reported.

"Difficult" questions are not a separate subset. `difficulty` is not in the
stored baseline records, so a difficulty subset could not be paired against
`results/baseline/` or `results/hybrid/` without regenerating them, and those
are fixed. multi_hop + comparison covers 7 of the 11 `hard` questions.

### Floors, stated in advance

* **RQ4 is expected to be unanswerable offline.** Both baselines'
  unsupported-answer rate is already 0.000 on the scripted backend, which
  copies its sentences verbatim out of the evidence it cites. A rate cannot
  fall below zero, and the rule-based verifier is expected to pass essentially
  every draft. An offline null on RQ4 says nothing about an LLM verifier.
* **Abstention has 6 questions.** One question moves abstention accuracy by 17
  points. Any offline movement there comes from the evidence controller (DD-052),
  not the verifier. It can be attributed to the controller only by Phase 6's
  controller-OFF arm, not by this run.
* **RQ3's subset is 15 questions.** A CI on 15 paired binary differences is
  wide; a real effect of a few questions may not be detectable.
* The generator is the scripted extractive backend throughout. It answers from
  the single best-matching block, so decomposition can change which blocks it
  sees but cannot make it combine two hops into one answer.

### Reason

A metric chosen after the result is not a test of the result. The floors are
written down now so that a null is read as what the offline stack can and
cannot show, not explained after the fact.

---

# DD-057 — The Seven Ablations Are Section 24's Six Components on the Agentic System, Plus the Existing Reranker-OFF Arm on Baseline B

**Status:** Accepted — reconciles EXPERIMENT_PLAN.md section 2 with EVALUATION_PROTOCOL.md section 24

### Decision

The Phase 6 exit criterion names "the seven ablations in
`EVALUATION_PROTOCOL.md` section 24". Section 24 lists six components, and
section 26.2 says "seven or more arms". The specifications name no seventh
component anywhere, so none is invented to reach the number. The seven are
seven *arms*, meaning measured configurations:

| # | Arm | Removed from | Reference | Directory |
| --- | --- | --- | --- | --- |
| 1 | Agent planner OFF | agentic | `results/agentic/` | `results/ablations/planner_off/` |
| 2 | Hybrid retrieval OFF | agentic | `results/agentic/` | `results/ablations/hybrid_off/` |
| 3 | Reranker OFF | agentic | `results/agentic/` | `results/ablations/reranker_off_agentic/` |
| 4 | Retrieval refinement OFF | agentic | `results/agentic/` | `results/ablations/refinement_off/` |
| 5 | Evidence controller OFF | agentic | `results/agentic/` | `results/ablations/evidence_controller_off/` |
| 6 | Verification OFF | agentic | `results/agentic/` | `results/ablations/verification_off/` |
| 7 | Reranker OFF | Baseline B | `results/hybrid/` | `results/ablations/reranker_off/` (Phase 4, exists) |

Section 24 says "each major component should be removed independently". That
means removing it from the full system, and arms 1-6 do exactly that.

Arm 3 is needed even though arm 7 exists:
* Arm 7 removes the reranker from Baseline B, where it answers RQ2 (DD-048).
* Arm 3 asks whether the reranker still earns its place once the planner merges
  several reranked rankings by RRF.

Arm 7 is kept and counted for two reasons. It is a measured configuration under
`results/ablations/`, and the reranker verdict uses both arms.

Not counted as ablations:

* **Agent loop OFF** (all four switches off). DD-050 and
  `test_all_four_off_is_baseline_b` make it Baseline B, record for record. It is
  a baseline, and Phase 5 already compared against it.
* **Hybrid OFF at the baseline level.** That is Baseline A against
  `reranker_off`, which is RQ1's comparison (STATUS.md section 9.2). It is not a
  new arm.
* **`--max-iterations 1`.** With the rule controller it behaves exactly like
  refinement OFF. `can_retrieve_again` is false in both, so the controller
  answers or abstains the same way, and only `stop_reason` differs. It belongs to
  open question 3's sweep (DD-058), not to the ablations.
* **The threshold sweep.** It tunes a value on `--split dev` and removes
  nothing.

### Reason

A count that does not match its source list is a documentation defect. There
are two ways to fix it: add a component, or read "seven" as arms.

Adding a component would mean inventing an experiment to satisfy a number,
which is the opposite of pre-declaration. Every arm above maps to a switch the
code already has, and six of them map to a line of section 24.

### Consequence

`--no-hybrid` is added to the CLI. It sets `retrieval.retrievers = ("dense",)`
and nothing else. `tests/test_ablations.py` checks that every arm differs from
the reference in exactly one configuration field.

In the agentic system, hybrid OFF still reranks the dense top 20. It is
therefore not Baseline A.

---

# DD-058 — Phase 6's Metrics, Floors, Threshold Rule and "Earned Its Cost" Rule Were Named Before Any Run

**Status:** Accepted — declared before any Phase 6 result existed. Recorded in
`statistics.PRIMARY_METRICS` under the `ablation:*` and `threshold_selection`
keys.

### Ablation arms

Each arm runs over all 76 questions, the same set as `results/agentic/`.

Each arm is compared with `compare-runs` against the full agentic system,
`results/agentic/`, as **arm − reference**. The one exception is arm 7, whose
reference remains Baseline B (DD-048).

A component **helps** on a metric when removing it makes that metric
significantly worse: the 95% CI of arm − reference excludes zero in the
harmful direction.

| Arm | Metrics the component should plausibly move (n) | Why |
| --- | --- | --- |
| planner OFF | Full-Recall@5, multi-hop + comparison (15); accuracy, multi-hop + comparison (15) | decomposition exists to retrieve each hop |
| hybrid OFF | Recall@5 (70); MRR@10 (70) | BM25 adds candidates the dense retriever misses |
| reranker OFF (agentic) | MRR@10 (70); accuracy (all) (76) | it reorders the pool; accuracy is RQ2's answer metric |
| refinement OFF | over-abstention rate (70); Full-Recall@5 (70) | a second round should turn a refusal into an answer from new evidence |
| evidence controller OFF | abstention accuracy (6); over-abstention rate (70) | the only component that refuses before generation |
| verification OFF | unsupported-answer rate (76); faithfulness (76) | RQ4 |
| reranker OFF (Baseline B, exists) | accuracy (all); faithfulness (DD-048) | unchanged from Phase 4 |

**Regression attribution.** Phase 5's two significant regressions against
Baseline B were in faithfulness and citation any-correct. Both metrics are read
on every arm against the reference.

The hypothesis, stated now: **only the evidence-controller-OFF arm**
significantly improves both, because the controller's refusals caused them
(STATUS.md section 9.3). If no arm separates the cause, or more than one does,
that is recorded as a follow-up. No unplanned run is added to settle it.

**Cost** is read on every arm:
* mean latency, as a paired contrast;
* mean generation-model calls, also paired. `model_calls_mean` is added to
  `statistics.METRICS` so it gets a paired contrast;
* reranker passes (retrieval queries issued), reported descriptively.

### "Earned its cost", per component

Three things are read for each component:

1. **Benefit:** removing it significantly worsens at least one of its own
   metrics above.
2. **Harm:** removing it significantly *improves* any of the four keep-or-drop
   metrics: accuracy (all), faithfulness, citation any-correct, abstention
   accuracy. A harm means the component causes a regression.
3. **Cost:** the paired latency and model-call differences, reported whatever
   their size.

The verdict follows from them:

* **Earned its cost:** benefit, and no harm.
* **Did not earn its cost:** no benefit. A null buys nothing, so even a
  negligible cost is unjustified (BENCHMARK_SPEC.md section 18). If the arm's
  answers are identical to the reference's, the component is called **inert on
  this stack**.
* **Not shown (trade-off):** benefit and harm both. It is reported as the
  trade-off it is, not as earned.

Offline, cost cannot disqualify a component that shows benefit. Every agent is a
rule, makes zero model calls and adds milliseconds. So the cost column is
reported but is not decisive. It does not transfer to an LLM agent, which would
add a model call per decision on a T4.

For **hybrid retrieval** and **the reranker**, the agentic arm is section 24's
test and decides the verdict. The Phase 4 baseline-level contrasts (RQ1, RQ2)
are reported beside it, and any disagreement between the two is stated.

A significant result on a metric *not* declared above is a hypothesis, not a
benefit (EVALUATION_PROTOCOL.md 26.2).

### Evidence-controller threshold (DD-052's 0.5)

* **Values:** `agents.sufficiency_threshold` ∈ {0.2, 0.3, 0.4, 0.5, 0.6}, on
  `--split dev` only (33 questions). Every other field is the reference
  configuration.
* **Selection rule:** the highest dev accuracy (all). Ties are broken in order:
  1. higher dev citation any-correct;
  2. lower dev over-abstention rate;
  3. the value nearest 0.5, the untuned default, so nothing changes without
     evidence;
  4. if two values are equally near 0.5, the higher, because it abstains more.

  Accuracy (all) is the criterion because DD-029 counts unanswerable questions
  in it. It therefore penalises both over-abstention and false answers.
* **Order of work:** the dev results are committed, and the chosen value
  recorded in a DD, before the eval run.
* **The eval run:** the chosen value is run **once** on `--split eval`
  (43 questions), into `results/experiments/threshold/eval_<value>/`. It is
  compared, on the eval split, against the reference and against Baseline B.
  Both are restricted to eval from their stored all-question runs
  (`compare-runs --split eval`).
* **Keep-or-drop:** the rule is applied to the tuned arm against Baseline B,
  exactly as DD-056 applied it to the reference.
* If the rule selects 0.5, the eval run is still made once, and the report says
  that tuning did not move the default.

### `MAX_RETRIEVAL_ITERATIONS` (open question 3)

* **Values:** `--max-iterations` ∈ {1, 2, 3}, on `--split dev`, at the reference
  threshold of 0.5. That keeps this sweep independent of the threshold choice.
* **Descriptive only.** Reported per value: questions refined, answers changed
  by the extra round(s), mean iterations, accuracy and over-abstention.
* The cap stays at 2 in every reported system. No second tuned value goes to
  the eval set, so eval is looked at for one tuned parameter only.

### Final table

The headline table (EVALUATION_PROTOCOL.md 28) is on the **eval split**,
43 questions. It has four rows: dense, hybrid, agentic and the tuned arm, each
figure with a 95% CI. Section 5.2 forbids headline results from anything else.

The stored all-question runs are restricted to eval by filtering their records,
not by re-running them. The stack is deterministic, so a re-run would be another
look at eval that adds no information.

Two other tables are kept separate from the headline:
* an all-76-question table, labelled, for continuity with STATUS.md sections
  9.2-9.3;
* the dev sweep numbers, in their own table only.

Peak VRAM is `null` offline, and is written as null, never as 0.

### Floors, stated in advance

* **Offline stack throughout:** hashing embedder, scripted extractive generator,
  term-overlap reranker, and rule-based planner, controller and verifier. No
  finding is about Qwen3-4B, bge-m3 or bge-reranker-v2-m3.
* **Unanswerable questions:** 6 of 76, 3 in each split. One question moves
  abstention accuracy by 17 points over all questions, and by 33 over one
  split. The controller-OFF arm's abstention CI will almost certainly include
  zero.
* **Multi-hop + comparison:** 15 of 76, 6 dev and 9 eval. The planner's metrics
  are bounded by this. The scripted generator also answers from a single block,
  so it cannot combine hops (DD-056).
* **Unsupported-answer rate is already 0.000** in every system. Removing the
  verifier cannot show it lowering that rate, so the verifier's primary metric
  is expected to be null by construction.
* **Refinement ran on 4 of 76 questions** in Phase 5 (DD-053). Refinement OFF can
  change at most those 4 records. That is far too few for a CI over 70 questions
  to exclude zero unless all four move.
* **Model calls:** rule agents make none. Arms differ in model calls only
  through refusals (fewer generator calls) and regenerations (more).
* **Latency** is milliseconds on a stack that loads no weights. A paired latency
  contrast is a real measurement of the stand-ins and nothing more.

### Multiple comparisons

* **Planned Phase 6 comparisons:** 6 arm files plus 2 eval files, at 21
  contrasts each, **168** in total.
* **Pre-declared among them:**
  * 23 distinct ablation contrasts: 12 per-arm and 12 attribution, with the
    verifier's faithfulness counted in both lists;
  * 4 keep-or-drop contrasts for the tuned arm against Baseline B.
* **Earlier phases:** the 97 comparisons of Phases 4-5 are reported beside
  these in the write-up.
* **Not added:** eval-split pairwise contrasts among dense, hybrid and agentic.
  Those comparisons were already made over all questions.

### Disclosure

Phase 5's diagnosis, which named the 0.5 threshold as the suspect, read failures
from both splits. The two lost answers it cited, q065 and q069, are eval
questions. So the decision *to tune this threshold* was informed by eval-set
failures, even though the *value* is chosen on dev alone. Under section 5.2 this
is a disclosed leak, and the write-up repeats it.

---

# DD-059 — The Sufficiency Threshold Is 0.3, Chosen on Dev by DD-058's Rule; the Iteration Cap Stays 2

**Status:** Accepted. Recorded and committed before the eval run.

### Dev sweep (`--split dev`, 33 questions; offline stand-in stack)

`results/experiments/threshold/dev_<value>/`. Every other field is the reference
configuration. These are dev-set numbers only (EVALUATION_PROTOCOL.md 5.2), and
they must never share a table with eval results.

| threshold | accuracy (all) | citation any-correct | over-abstention | abstention acc. (n=3) | faithfulness | controller refusals |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.212 | 0.567 | 0.100 | 0.333 | 0.879 | 4 |
| **0.3** | **0.242** | **0.533** | **0.133** | 0.667 | 0.818 | 6 |
| 0.4 | 0.242 | 0.533 | 0.167 | 0.667 | 0.788 | 7 |
| 0.5 (reference) | 0.242 | 0.533 | 0.167 | 0.667 | 0.788 | 7 |
| 0.6 | 0.242 | 0.467 | 0.233 | 0.667 | 0.727 | 9 |

The rule, applied mechanically:

1. **Accuracy (all):** 0.3, 0.4, 0.5 and 0.6 tie at 0.242.
2. **Citation any-correct:** among those four, 0.3, 0.4 and 0.5 tie at 0.533.
   0.6 drops out.
3. **Over-abstention:** 0.3 has the lowest of the three, 0.133 against 0.167.

**Chosen: 0.3.**

On 33 questions, the decisive difference is one answerable question that is
no longer refused. That is a tuning choice, not a finding. Its only test is
the single eval run that follows.

What the dev sweep does show is that the threshold trades coverage against
refusals monotonically:
* Controller refusals rise from 4 to 9 across the grid.
* Over-abstention and faithfulness move with them.
* At 0.2, the controller stops refusing an unanswerable question and dev
  accuracy falls.

### `MAX_RETRIEVAL_ITERATIONS` (open question 3), dev, threshold 0.5

`results/experiments/max_iterations/dev_<n>/`

| cap | accuracy (all) | over-abstention | mean iterations | questions refined | answers changed by refinement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.242 | 0.200 | 1.000 | 0 | 0 |
| 2 | 0.242 | 0.167 | 1.030 | 1 | 0 |
| 3 | 0.242 | 0.167 | 1.030 | 1 | 0 |

* **Cap 3 is identical to cap 2.** No dev question reaches a third round,
  because the rule refiner runs out of new queries first (`no_new_query`).
* **Cap 2 differs from cap 1 on one question.** The second round turned a
  refusal into an answer. The answer is still wrong, so accuracy does not move.
* `refinement_changed_answer` is 0. That field compares drafts, and at cap 1
  this question produced no draft, only a refusal.

**Closed for the offline stack:** the cap is inert above 2, and iteration 2
changes one refusal in 33 into a (wrong) answer. As DD-058 declared, the cap
stays at 2. The question stays open for an LLM controller and refiner, which
could keep issuing new queries.

---

# DD-060 — Phase 6 Verdicts: No Component Earned Its Cost Offline, and the Controller's Refusals Are Two Rules, One of Them Fed by a Chunking Defect

**Status:** Accepted — Phase 6 findings, all from the offline stand-in stack

### Decision

DD-058's rule is applied mechanically by `src/evaluation/report.py`
(`python -m src.cli final-table`, which writes `results/final/`). Each verdict
comes from that component's agentic arm.

| Component | Its declared metrics, arm − reference | Verdict |
| --- | --- | --- |
| hybrid retrieval | Recall@5 0.000 [−0.043, +0.043]; MRR@10 +0.003 [−0.010, +0.016] | did not earn its cost |
| reranker | MRR@10 −0.011 [−0.067, +0.039]; accuracy +0.013 [0.000, +0.039] | did not earn its cost |
| planner | Full-Recall@5 (multi) +0.067 [0.000, +0.200]; accuracy (multi) 0.000 | did not earn its cost |
| refinement | over-abstention +0.014 [0.000, +0.043]; Full-Recall@5 0.000 | did not earn its cost |
| evidence controller | abstention acc. −0.167 [−0.500, 0.000]; over-abstention −0.100 [−0.172, −0.029] | did not earn its cost; **removing it improves faithfulness and citation** |
| verifier | unsupported rate 0.000 (floor); faithfulness +0.026 [0.000, +0.066] | did not earn its cost |

Phase 4's baseline-level contrasts disagree with the agentic arms in one
place:
* **Hybrid retrieval.** Without the agent loop, hybrid retrieval significantly
  improves ranking over dense: MRR@10 +0.074, and Full-Recall@5 is also
  significant. Those were RQ1 *secondaries*, and its primary, Recall@5, was
  n.s. Inside the agentic system, even the ranking gain disappears. The
  planner's RRF merge over sub-queries evidently re-ranks as much as the
  retriever pair does.
* **Reranker.** Both levels agree: no answer gain.

### Findings recorded with the verdicts

1. **Pre-declared attribution confirmed.** Only controller OFF significantly
   improves both faithfulness (+0.105 [+0.039, +0.171]) and citation
   any-correct (+0.086 [+0.029, +0.157]). No other arm moves either one. Phase
   5's regression is the controller's.
2. **The controller has two refusal rules, and the threshold governs only one
   of them.** Of its 13 refusals in `results/agentic/`:
   * 8 are coverage-only;
   * 2 are number-only (q005, q069);
   * 3 fail both.

   The eval run at threshold 0.3 still refuses 6 eval questions, 3 of them
   number-only. No threshold can release those. This is why tuning 0.5 → 0.3
   changed no eval answer.
3. **The structure chunker hides heading text from every component that
   decides.** `doc1.pdf` has two pages, and both begin with "GATE 2027 IIT
   Madras | Organizing Institute". Parsing keeps the line: the page text
   contains it, and running-header removal does not fire on a 2-page document
   (`header_footer_min_pages = 3`). The structure chunker then takes the line as
   a heading and stores it in `chunk.section` instead of `chunk.text`. `section`
   is used only to label citations. The embedder, BM25, reranker, generator,
   controller and verifier all read `chunk.text`, so none of them ever sees the
   line. The consequences:
   * q001's answer ("IIT Madras") is unrecoverable by every system;
   * "2027" is missing from every GATE chunk, which triggers the number rule
     on q001, q002 and q005.

   This is a chunking defect, upstream of every system measured. It is recorded
   rather than fixed here, because fixing it would change every stored
   baseline.

   *Correction, made before this branch was merged:* the first draft of this
   DD blamed running-header removal. Tracing the line through
   `PdfParser` and the chunker showed it is the heading-to-`section` move.
4. **q065, the first `verification` record, is a marker-parsing defect, not a
   verifier-threshold problem.** The draft is the correct passage, copied
   verbatim from block [C1]. That passage contains the paper's own
   bibliography reference "shrinking[3]". `citations.py` accepts a bare `[3]`
   as a marker, so the reference became `[C3]`. The verifier then checked the
   preceding claims against block C3, a licence notice, found term support of
   0.07-0.17 and missing numbers, and rejected a correct answer.
5. **Open question 3 is closed for the offline stack** (DD-059). A cap of 3
   is identical to a cap of 2, and a second round changes one refusal in 33
   dev questions into a (wrong) answer.

### Multiple comparisons

Phase 6 made 8 comparison files of 21 contrasts each, **168 contrasts**:

| Source | Contrasts | Significant |
| --- | ---: | ---: |
| 6 ablation arms | 126 | 5 |
| 2 eval files for the tuned arm | 42 | 5 |

The significant ablation contrasts are:
* 4 in the controller arm: faithfulness, citation, over-abstention and model
  calls;
* 1 isolated nDCG@10 result in the planner arm. It is undeclared, so it is
  treated as a hypothesis.

The significant eval contrasts are:
* the latency difference in both files, measured across sessions and not a
  finding;
* faithfulness, citation and over-abstention against Baseline B, which
  replicate Phase 5 on the eval split.

With Phases 4-5, the project has made **265** paired contrasts.

### Reason

EVALUATION_PROTOCOL.md 26.3: a null is a finding. Every component's null here
is bounded by the floors DD-058 stated in advance, so these verdicts are about
the rule-based and hashing stand-ins. They are not verdicts on the components
as designed for the model stack.

---

# DD-061 — Every Chunk Carries Its Section Heading, and No Table Is Cut Off

**Status:** Accepted — changes what every system sees; every stored offline
result is regenerated under it (DD-069). Issue #13.

### Decision

The structure chunker now guarantees that every line a page shows reaches
`chunk.text`. Two changes do this:

1. **Headings.**
   * Every chunk starts with its section heading, not only the chunk the
     heading opened.
   * When a heading is followed only by tables, as on `doc1.pdf`, the headings
     waiting in the buffer lead the first table instead of being dropped at
     the end of the document.
   * A heading still waiting when the document ends becomes its own chunk.
   * `chunk.section` is unchanged and still labels citations.
   * The repeated heading counts toward `chunk_size` (DD-030): the room left
     for content is `chunk_size − len(heading) − 2`, never less than half the
     chunk size.
2. **Tables.**
   * A table over `max_table_chars` is split into row groups, each repeating
     the header rows (every line up to the `| --- |` rule).
   * It is no longer cut off with "[table truncated]".
   * A single row longer than the cap is kept whole.

The repeated heading is a label, not evidence. It adds no page to the chunk's
`pages`, so a chunk on page 4 under a heading from page 3 still cites page 4.

The fixed-size chunker is untouched and records no section (DD-032), so the
DD-009 comparison still sets structure against no structure.

### Reason

`chunk.section` is read only by citation labels. The embedder, BM25, the
reranker, the generator, the evidence controller and the verifier all read
`chunk.text`. DD-060 traced q001, q002 and q005 to this.

On `doc1.pdf`, "GATE 2027 IIT Madras | Organizing Institute" is followed only
by tables. Before the fix:
* it sat in the heading buffer;
* it labelled the tables' `section`;
* the final `flush()` discarded it.

So "2027" and "IIT Madras" were invisible to every component.

The invariant written for this fix exposed a second loss, which no DD had
recorded. Three `doc2.pdf` tables exceed 4,000 characters and lost 20 lines to
truncation.

### Test

`tests/test_chunking.py::TestNoPageTextIsLost` asserts that every non-blank
line of every page of the five benchmark PDFs appears in at least one chunk.
Whitespace is compared collapsed, because segmentation joins a paragraph's
lines with spaces.
* Before the fix, it failed on `doc1.pdf` (the title line, twice) and
  `doc2.pdf` (20 table lines).
* After the fix, it passes on all five.

Unit tests cover:
* a heading followed only by tables;
* a long section whose every chunk carries the heading within `chunk_size`;
* a 200-row table split with its header.

### Trade-off

* Repeating a heading costs up to about 120 characters of each chunk's 1,200.
* Retrieval now scores the heading once per chunk under it, which can tilt BM25
  towards heading terms. This is measured, not assumed, by the regenerated
  results.

### Consequence

Chunk text changes on every benchmark document. Every result in `results/` was
measured under the old chunker and is regenerated under DD-069.

---

# DD-062 — A Bare `[n]` Is a Citation Only If the Evidence Does Not Already Contain It

**Status:** Accepted. Issue #14.

### Decision

**The resolver.** `resolve_citations` still accepts a bare `[n]`, because
small models drop the `C` prefix. It now treats one as source text when that
exact bracketed token already occurs in the evidence text. It leaves such a
token in place and cites nothing.

**Unchanged:**
* A prefixed `[C3]`, which never occurs in source text, is always a marker.
* A bare `[1]` that the evidence does not contain still resolves to block 1.

**The readers.** Two readers of a *resolved* answer now count only the
canonical `[Cn]` form the resolver emits:
* the verifier's `cited_numbers` (`src/agents/text.py`);
* citation completeness (`src/evaluation/metrics.py`).

Leaving the bare token in the answer is not enough on its own. Both readers
used the resolver's old permissive pattern, so the verifier would still have
read "shrinking[3]" as a citation of block 3.

**Scope limit.** Stripping markers before counting terms or numbers
(`content_terms`, `numbers_in`, `content_tokens`) is unchanged.

**Version.** `SCORING_RULES_VERSION` → `2026-09-24.1`.
* It is declared score-compatible with `2026-09-23.2`. Before this change the
  resolver rewrote or removed every bare group, so no stored answer contains
  one, and completeness scores every stored record as before. A test checks
  this claim against every stored `per_question.json`.
* `scores_comparable` now accepts any two versions that are each
  score-compatible with the current one, because they all score identically.

### Reason

DD-060 finding 4 is q065. The verifier rejected a correct draft that was a
verbatim passage from block C1:
1. The passage carries the paper's own reference, "shrinking[3]".
2. The resolver rewrote that reference to `[C3]`.
3. The verifier checked the claim against block C3, a licence notice, and
   failed it.

### Alternative rejected

Stop accepting a bare `[n]` altogether. That breaks citation for any model
that drops the prefix, which small models do. The evidence-text test separates
the two cases directly.

### Limitation

A bare `[n]` that the model meant as a citation, and that also happens to
occur verbatim in the evidence, is now left uncited. Quoting is the far more
likely reading of such a token.

---

# DD-063 — The Evidence Controller's Number Rule: Decided on Dev, by a Rule Written First

**Status:** Rule declared 2026-09-24, before any run under DD-061 and DD-062.
The outcome is appended below it. Issue #15.

### Background

The rule-based evidence controller refuses unless every number in the question
appears in the context (DD-052). In Phase 6, 5 of its 13 refusals involved this
rule. Three of the five (q001, q002, q005) trace to the DD-061 chunking defect,
because "2027" never reached the context. The GPU run uses the LLM controller,
which does not have this rule, so a change affects the offline stack only.

### Rule, declared before looking

1. **Run.** `python -m src.cli run-benchmark --system agentic --offline --split
   dev` on the DD-061 and DD-062 code, at the default threshold 0.5, into a
   scratch directory. Eval is not run and not read for this decision.
2. **Count.** A refusal is *number-only* when all of the following hold:
   * `abstention_source` is `evidence_controller`;
   * the question is answerable;
   * the final round's `evidence_decision` has a non-empty `missing_numbers`;
   * every query's coverage is at or above the threshold.
3. **Decide.**
   * **0 number-only refusals:** no change. The rule is recorded and kept.
   * **1 or more:** make exactly one change: numbers of a single digit (0-9)
     are exempt from the rule. Years, quantities and multi-digit identifiers
     stay checked. Then re-run dev and record the before-and-after counts. No
     second change is tried.

### Disclosure

The candidate change was chosen with q069 ("iPhone 6") in mind, and q069 is an
eval question. DD-060 already inspected it. Choosing the candidate is therefore
informed by one eval failure. Applying the candidate is decided by dev counts
alone.

### Outcome (appended after the dev run)

**Offline stand-in stack, `--split dev` (33 questions), threshold 0.5.**

| Controller refusals on dev | Phase 6 reference | Under DD-061 and DD-062 |
| --- | ---: | ---: |
| total | 7 | 5 |
| number-only, answerable | 0 | **0** |
| both rules (number and coverage) | 2 (q001 answerable, q007 unanswerable), both missing "2027" | 0 |
| coverage-only | 5 (q051, q053, q055, q057, q061) | 5 (the same) |

**Decision: no change.** No number-only refusal of an answerable dev question
remains, so the declared rule keeps the number rule as it is.

What moved:
* **"2027" now reaches the context.** Neither q001 nor q007 trips the number
  rule any more.
* **q007 is no longer refused.** It is an unanswerable dev question that the
  reference refused only because "2027" was missing. That was a right refusal
  for a wrong reason, and it is gone. It is reported here rather than hidden,
  because it will show up in the regenerated abstention numbers (DD-069).

The number-only refusals DD-060 listed, q005 and q069, are eval questions. They
were neither counted nor looked at for this decision.

---

# DD-064 — Resumable Benchmark Runs, Refused Across Configurations

**Status:** Accepted. Issue #16.

### Decision

**The checkpoint now runs from the command line.** `run_benchmark_cli` passes
its output directory to `run_benchmark` as `checkpoint_dir`, so
`per_question.json` is rewritten after every question. Before this, only a
caller that passed `out_dir` got checkpoints, and the CLI did not pass it.
`EXPERIMENT_PLAN.md` section 4's "checkpoint per-question results" was
therefore true of the library but not of the command people actually run. The
other three files are still written once, at the end.

**`run-benchmark --resume`** works as follows:
1. It loads the records already in `--out`.
2. It refuses to start if any record's `config_fingerprint` differs from the
   current configuration's, or if a record answers a question outside the
   current `--questions` / `--split` / `--limit` selection.
3. A record whose run raised (`system-runtime`) is dropped, so its question is
   asked again. A crash is usually the session that died, not the question.
4. Kept records stay in the position their question takes in a normal run.
   Their scores are rebuilt with `QuestionScore.from_record`.
5. A document whose questions are all done is not indexed again.
6. The four files are written as for any run.

`--resume` with `--judge` is refused, because the judge's summary would cover
only this session's questions. With nothing stored, `--resume` is a fresh run.

### Guarantee and its test

`tests/test_resume.py` interrupts an offline agentic run after 9 of 10
questions and resumes it. That leaves one document finished and one half done.
The resumed `per_question.json` equals an uninterrupted run's record for
record, excluding the timing fields (`latency`, `*_s`).

Headline means agree to 1e-4. Kept records carry their stored scores, rounded
to 4 decimal places as written. That is the same view the confidence intervals
and `compare-runs` already read.

A changed chunk size is refused, and so is an out-of-selection record.

### Limitation

In a resumed run's `results.json`, two figures cover only the resumed session:
* `index_seconds_total`;
* `documents_indexed`.

The summary says so with `resumed_records`, which appears only in a resumed
run. The records' own timings are kept as they were measured.

### Consequence

The fingerprint hashes every configuration field (`RAGConfig.fingerprint`), so
a resume across code that added a field is refused too. That is the right
default for a run meant to be one experiment.

---

# DD-065 — Follow-Up Questions: A Rewriting Layer Above the Pipeline

**Status:** Accepted. Issue #17.

### Decision

`src/chat/session.py` adds `ChatSession(pipeline, rewriter=None, max_turns=5)`:

1. It keeps the last `max_turns` turns: question, standalone question, answer.
2. Before calling `pipeline.ask`, it rewrites the new question into a
   standalone one. The pipeline sees only that standalone question, never the
   history.
3. It records `metadata["conversation"]` on the result. The record holds:
   * the turn number;
   * the original and standalone questions;
   * whether the question was rewritten;
   * which rewriter ran;
   * whether it fell back;
   * the rewriter's model calls.

Two rewriters, chosen by `build_rewriter` from what the pipeline has loaded:

* **`LLMRewriter`** asks the pipeline's own generation backend, through the new
  read-only `pipeline.backend` property. No second model is loaded
  (ARCHITECTURE.md section 21).
  * A reply that is empty, the refusal sentence, or more than
    `3 × len(question) + 200` characters is unusable. The rules then answer,
    and the rewrite is marked `fallback`, in the spirit of DD-055.
  * The first turn makes no model call.
* **`RuleBasedRewriter`**, offline. A question is a follow-up when it:
  * contains a referring word ("it", "that", "those", ...);
  * opens with a continuation ("and", "what about", ...);
  * or has fewer than three content terms.

  A follow-up becomes `"<question> (<previous standalone question>)"`.

### Why above the pipeline

The benchmark measures one self-contained question at a time, and the stored
results depend on that. A layer that only calls `pipeline.ask` cannot change
what a single question retrieves or how it is answered:
* the first turn of a chat is passed through unchanged;
* the harness never constructs a `ChatSession`.

`tests/test_architecture.py::TestChatLayer` asserts that ingestion, chunking,
retrieval, reranking, generation, agents, evaluation, benchmark and
`pipeline.py` never import `src/chat`. `tests/test_chat.py` checks that a first
turn equals a bare `pipeline.ask`.

### Alternatives rejected

* **Passing the history into the prompt.** It changes the generator's input for
  every question, so single-turn answers would no longer be the benchmarked
  ones.
* **A second, small rewriting model.** ARCHITECTURE.md section 21 forbids
  loading several generation models, and a T4 has no room for one.

### Limitation

The rule rewriter is crude:
* it cannot resolve "the second one";
* it treats a short standalone question ("What is RAG?") as a follow-up.

Appending the previous question only widens retrieval, so the cost is a
noisier query, not a wrong answer. It exists so the chat runs and is tested
offline. It is not a measured component: no benchmark question is multi-turn,
so no follow-up rewrite has a number attached, on either stack.

---

# DD-066 — The Chat Interface: Gradio, Imported Lazily, Over Plain Functions

**Status:** Accepted. Issue #18.

### Decision

`src/chat/ui.py` builds the notebook's chat app as a Gradio `Blocks`. The app
lets the user:
* upload one or more PDFs;
* choose dense, hybrid or agentic;
* press **Index**;
* chat.

Each answer shows its page citations, and the source file when more than one
PDF is loaded. A refusal is shown as **Refused:** followed by the canonical
sentence. A collapsible "Agent trace" panel shows:
* the follow-up rewrite, if any (DD-065);
* the question type and sub-queries;
* each round's queries and evidence-controller verdict;
* the stop reason, the verifier status, and who refused.

The behaviour lives in plain functions over a `ChatState`: `load_documents`,
`respond`, `format_answer`, `format_trace`, `format_error` and `format_status`.
`build_app` only wires them to components, and it is the one place gradio is
imported. `tests/test_architecture.py` checks that.

`tests/test_chat.py` covers indexing, multiple PDFs, errors, answering, traces
and refusals without gradio. One test builds the real app, and it is skipped
when gradio is absent.

**Multiple PDFs.** A new `index_documents(paths, on_error=None)` on the
pipeline parses and chunks every file into one index. Chunk ids carry the
document id, so they cannot collide. `index` is untouched, so the benchmark
still indexes one document at a time.

**Errors.** A file that fails to parse (encrypted, corrupt, image-only without
OCR) is reported by name with its whole message, which already carries its
remedy (`src/errors.py`), and skipped. The remaining files are still indexed.

**Status table.** After indexing, the status table shows pages, chunks and
OCR'd pages per file. Offline mode says "offline stand-in stack" in the table.

**Dependencies.** `requirements-colab.txt` is `-r requirements-models.txt` plus
gradio (`>=5,<7`) and pytesseract. The Tesseract binary is a system package the
notebook installs with `apt-get`.
* gradio, pytesseract and Tesseract are all Apache-2.0. They are recorded like
  every other licence: a trailing comment in the requirements file, a NOTICE
  table and README section 20.
* A new architecture test requires every requirement line to carry a licence
  comment and to appear in NOTICE.

### Reason

The brief asks for a chatbot, and Gradio is the standard way to serve one from
Colab: `launch(share=True)` gives a public link with no server to run. Keeping
the logic outside the library means the suite does not need a web framework,
and the interface cannot drift from what is tested.

### Limitation

Gradio's own dependency tree was not audited licence by licence. It is an
optional, Colab-only extra and is never imported by the library outside
`ui.py`. NOTICE says so.

---

# DD-067 — OCR Is On by Default, for Pages With No Text Layer Only

**Status:** Accepted. Issue #19.

### Decision

The brief says "any uploaded PDF", and scanned PDFs are PDFs. So OCR is a core
feature, not an option.

**Configuration.** `IngestionConfig` gains three fields:

| Field | Default |
| --- | --- |
| `ocr` | `True` |
| `ocr_languages` | `"eng"` |
| `ocr_resolution` | 300 DPI |

Every preset, `offline()` included, inherits `ocr=True`, and a test checks all
three presets. `describe()` records `ocr`, so it appears in every run's
`config.json`. The CLI's `--no-ocr` turns it off, for speed or to reproduce
the old refusal.

**Which pages.**
* A page is OCR'd when its extracted text is empty and it carries at least one
  image. That covers fully scanned PDFs and the scanned pages of an otherwise
  digital one.
* Pages with a text layer are never OCR'd, however short that layer is.

**How.**
* The page is rendered with the existing pdfplumber → pypdfium2 path
  (`render_page_images` in `src/ingestion/render.py`). There is no PyMuPDF,
  which is AGPL and test-only (DD-033).
* The image is read by `TesseractOcr` (`src/ingestion/ocr.py`, pytesseract over
  the Tesseract binary).
* The recognised text goes through the same character normalization and
  hyphen joining as extracted text. It is split into paragraph blocks with no
  font size, so the chunker treats it as body text.

**Recording.**
* An OCR'd page records `metadata["ocr"] = True` and the engine's name.
* The document records `ocr_pages`.
* A chunk drawn from an OCR'd page records `ocr: True`, which is `meta_ocr` in
  records.
* The keys are added only when OCR ran, so a text PDF's pages, document and
  chunks are exactly what they were.

**Missing engine.** If a page needs OCR and pytesseract or the binary is
missing, `OCRUnavailableError` is raised. Its message gives:
* `apt-get install -y tesseract-ocr` for Colab/Debian, plus brew and Windows;
* `pip install pytesseract`;
* `--no-ocr`.

A document is never returned with the page silently empty. The check runs only
when a page needs OCR, so text PDFs never import pytesseract.

**`ScannedPDFError` stays.** It now has two messages:
* **OCR on:** OCR ran but the document is still near-empty. The message names
  the likely causes: faint, handwritten, or another language, with
  `ocr_languages`.
* **OCR off:** OCR was not tried, and the message says how to turn it on.

**Confinement.** pytesseract and pypdfium2 join the libraries
`tests/test_architecture.py` confines to `src/ingestion/`.

### Stored results are unaffected by this DD alone

`tests/test_ocr.py::TestTextPagesAreNeverOcrd` parses each of the five
benchmark PDFs twice:
* once with OCR on and an engine that fails the test if called;
* once with OCR off.

The two parses have identical pages, blocks, metadata and document metadata.
No benchmark page lacks a text layer, so OCR never runs on the benchmark.

The configuration fingerprint does change, because it hashes every field.
Results are regenerated under DD-069 regardless.

### Cost, measured offline on this machine

The rendering half of OCR at 300 DPI, 5 pages × 3 repeats, on this CPU
(Intel64 Family 6 Model 154, Python 3.13):

| Page | Median per page | Image |
| --- | ---: | --- |
| an image-only A4 page (test fixture) | 0.127 s | 2480 × 3509 px |
| a text A4 page (`doc5.pdf`), for scale | 0.221 s | 2482 × 3508 px |

**The Tesseract half was not measured here,** because this machine has no
Tesseract binary. The notebook's scanned-PDF demo prints the measured seconds
per OCR'd page on Colab. That figure, not an estimate, is what the Colab run
reports. For a text PDF the cost is zero: nothing is rendered.

### Tests

`tests/test_ocr.py` uses a fake engine, so the suite runs without Tesseract.
It covers:
* a fully scanned PDF;
* a mixed PDF, where only page 2 is OCR'd and pages 1 and 3 are identical to an
  OCR-off parse;
* chunk `meta_ocr`;
* OCR that reads nothing;
* a scanned PDF answered with page citations through the pipeline;
* missing pytesseract;
* a missing binary.

One test runs the real Tesseract on a rendered page, and it is skipped where
the binary is absent, as it is here. The old scanned-PDF tests now use
`--no-ocr` / `ocr=False`.

### Limitations

* **A page with a short text layer** (a stamped page number over a scanned
  image) is not OCR'd, because pages with a text layer never are. Such a
  document may still be refused as scanned, with the remedy in the message.
* **Text drawn as vector outlines** (no image, no text layer) is not OCR'd. It
  fails as `EmptyDocumentError`, as before.
* **OCR text has no font sizes,** so headings on scanned pages are found only by
  numbering or capitals.

---

# DD-068 — Two Notebooks, Generated by a Script, Tested by Running One

**Status:** Accepted. Issue #20.

### Decision

`notebooks/build_notebooks.py --ref <tag>` generates both notebooks with
nbformat. The default ref is `v0.7-gpu-run`. Neither is edited by hand.

**`agentic_pdf_rag.ipynb`, the deliverable** (DD-002). It follows PROJECT_SPEC
section 12's fifteen items in order, and every code cell has a markdown cell
before it. The cells cover:
1. **Setup.** It finds the checkout, or clones the pinned tag (DD-034) and
   prints the commit. In Colab it runs `apt-get install -y tesseract-ocr`
   (always, because OCR is on by default) and installs
   `requirements-colab.txt`.
2. **Mode switch.** `OFFLINE = not torch.cuda.is_available()`, announced in a
   banner. The model-stack configuration is validated here, and weights load
   on the first question.
3. **Config.** `config.describe()`.
4. **Upload.** `UPLOAD = True` for your own file. Otherwise it uses bundled
   `doc5.pdf` (CC-BY-4.0), so *Run all* is unattended.
5. **Parse, chunk and index,** with page, chunk and OCR'd-page counts and
   timings.
6. **A scanned-PDF demo.** It draws text into an image, saves an image-only PDF
   with Pillow and parses it. It prints the OCR'd page's text and seconds per
   page, or the install remedy where Tesseract is missing.
7. **Dense and agentic, side by side,** on the benchmark's own questions for
   that PDF, read from `questions.json`. Refusals are marked.
8. **The agent trace.**
9. **The chat UI.** `share=True` in Colab; built but not launched elsewhere.
10. **An optional benchmark cell,** `--limit 5` by default, into a temporary
    directory, never `results/`.
11. **Metrics and ablations,** rendered from `results/final/REPORT.md` and from
    `model_stack/results/final/REPORT.md` when present, each with its stack
    caveat.
12. **Latency, and peak VRAM,** printed as null offline.
13. **A three-turn `ChatSession` demo.**

Pipelines are built one at a time and freed in between (`del`, `gc`,
`torch.cuda.empty_cache`), so a T4 never holds two copies of the weights
(ARCHITECTURE.md section 21).

**`gpu_benchmark_run.ipynb`, the runner for the Colab T4.** In order:
1. A GPU check.
2. Drive mounted, `HF_HOME` on Drive, a clone at `REF`, Tesseract, the install.
3. `COMMIT.txt`, which asserts that HEAD is the tag.
4. A smoke test: one `ask`, then `run-benchmark --limit 3`. It prints a
   paste-back block: the answer and pages, `peak_vram_gb`,
   `llm_parse_failure_rate`, seconds per question, and an estimated full run
   (agentic latency × 859 question-runs, an upper bound).
5. One cell per run, each with `--resume` and output under
   `<Drive>/model_stack/results/`:
   * dense, hybrid, hybrid `--no-rerank`;
   * agentic and its six arms (DD-057);
   * `--max-iterations 1/2/3` on `--split dev`.
6. The Phase 4-6 `compare-runs` pairs.
7. `final-table --root <Drive>/model_stack`.
8. A zip of `model_stack/` to bring home.

**No threshold sweep, and a corrected reason.** Earlier documents said
`sufficiency_threshold` is read only by the rule controller. That is
incomplete. It also decides which queries `MissingTermsRefiner` rewrites
(`src/agents/refinement.py`). It also acts inside the LLM controller, which
always computes the rule result and falls back to it on a parse failure
(`src/agents/evidence.py`).

The sweep is still left out, as the run plan specifies. The LLM controller
makes the sufficiency decision itself. The threshold's remaining effect on the
GPU run is limited to refinement targets and to parse-failure fallbacks, and
the smoke test's parse-failure rate bounds the fallbacks. The default 0.5 is
used, unchanged.

### Tests (`tests/test_notebooks.py`)

* **Drift.** Each committed notebook equals the script's output, compared with
  line endings normalized.
* **Validity.** Both notebooks are valid nbformat 4.
* **Explained cells.** Every deliverable code cell is preceded by markdown.
* **Pinned tag.** REF is a tag, and `main`, `master` or a branch name is
  refused by the builder.
* **Imports.** Every `src.` import in either notebook exists.
* **The runner's structure.**
  * every run resumes under the model-stack root, and none writes to
    `results/`;
  * every arm and the dev sweep are present;
  * no `--sufficiency-threshold`;
  * every comparison, `final-table --root`, COMMIT.txt, `HF_HOME`, Tesseract;
  * the fields of the paste-back block.
* **No typed-in numbers.** No three-decimal number appears in the deliverable.
* **Execution.** The whole deliverable is executed with nbclient in offline
  mode, with no errors, the OFFLINE banner, and a null peak VRAM. It takes
  about 30 s on this machine, so it stays in the default suite. The CI job's
  20-minute budget has room for it. `SKIP_NOTEBOOK_EXECUTION=1` skips it, and
  it skips itself on a machine with a GPU, where it would not be testing the
  offline path.

nbformat, nbclient and ipykernel (all BSD-3-Clause) join `requirements-dev.txt`
and NOTICE. gradio and pytesseract become the `colab` extra in
`pyproject.toml`.

### Limitations

* **Colab itself is not tested here.** The Colab-only branches (the clone,
  `apt-get`, `files.upload`, `share=True`, Drive) cannot run on this machine.
  "Run all in a fresh Colab runtime" remains a manual check, in `NEXT_STEPS.md`.
* **The tag does not exist yet.** The notebooks pin `v0.7-gpu-run`. Until that
  tag is pushed, a fresh Colab clone fails at step 1 with git's "Remote branch
  not found".

---

# DD-069 — Offline Results Regenerated After DD-061 to DD-063; Threshold Re-Chosen on Dev, Eval Looked at a Second Time

**Status:** Part 1 was written and committed with the dev results, before the eval
run. Part 2, the eval outcome, is appended after it. Issue #21.

> **Offline stand-in stack throughout**, as in STATUS.md 9.1-9.4: hashing
> embedder, scripted extractive generator, term-overlap reranker and rule-based
> agents. No figure here is about Qwen3-4B, bge-m3 or bge-reranker-v2-m3.

### Why everything was regenerated

DD-061 changes chunk text on every benchmark PDF, and DD-062 changes citation
resolution. Every stored result therefore measured a system that no longer
exists. DD-063 changed no code, and DD-067's OCR provably changes nothing for
the benchmark PDFs (`tests/test_ocr.py`).

The following were re-run with the commands in the previous `NEXT_STEPS.md`
step 1e:
* `results/baseline`, `hybrid` and `agentic`;
* all seven ablation arms (DD-057);
* all eleven comparisons;
* the dev threshold sweep and the dev iteration-cap sweep.

**One arm was resumed.** `ablations/refinement_off` stopped at question 54 of
76 when Windows refused a checkpoint rewrite (fixed: result files are now
written atomically, with retries, DD-064). It was finished with
`run-benchmark --resume`, which carried over the 54 stored records under the
same configuration fingerprint. Its `results.json` records
`resumed_records: 54`.

### Part 1 — the threshold, re-chosen on dev by DD-058's rule, unchanged

`--split dev`, 33 questions, dev-only numbers:

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy (all) | 0.242 | 0.273 | 0.273 | 0.273 | **0.303** |
| citation any-correct | 0.667 | 0.633 | 0.633 | 0.633 | **0.533** |
| over-abstention | 0.067 | 0.100 | 0.133 | 0.133 | **0.233** |

**The rule chooses 0.6.** DD-058's first criterion is the highest dev accuracy
(all), and 0.6 has it alone, 0.303. No tie-break applies.

This choice is recorded as the rule makes it, not as it looks. 0.6 also has the
lowest dev citation any-correct and the highest over-abstention of the five
values. The accuracy lead is one question in 33. A rule that ranks accuracy
first chose a threshold that refuses more answerable questions. The eval run
below will show whether that holds up. Changing the rule after seeing this
table would be the tuning-on-the-result that DD-058 exists to prevent.

The previous choice, 0.3 (DD-059), was made on the pre-DD-061 system and is
superseded. The code default stays 0.5, and 0.6 exists only as the tuned arm.

**Iteration cap (descriptive, DD-058):**

| `--max-iterations` | 1 | 2 | 3 |
| --- | ---: | ---: | ---: |
| accuracy (all) | 0.273 | 0.273 | 0.273 |
| citation any-correct | 0.600 | 0.633 | 0.633 |
| over-abstention | 0.167 | 0.133 | 0.133 |

A cap of 3 is identical to 2 on these figures, and the cap stays 2, as in
DD-059.

### Disclosure (EVALUATION_PROTOCOL.md 5.2)

**Eval is about to be looked at a second time.** The first time was DD-059's
0.3 run on the old system. Two consequences follow:
* this eval run is not a clean first look;
* the DD-061 to DD-063 defects it measures were themselves found partly by
  inspecting eval failures (q065, q069 in DD-060).

The threshold value was chosen on dev alone. The eval run is made once, at 0.6,
and it is reported whatever it shows.
