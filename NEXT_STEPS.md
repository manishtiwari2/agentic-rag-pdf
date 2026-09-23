# Next Steps

Phase 6 is complete on the offline stand-in stack (`STATUS.md` section 9.4).
Three steps remain. **Do them in this order.** Step 1 changes what every system
sees, so it must land before the numbers that are meant to last.

1. Decide and fix the three defects found by the error analysis. Local, no GPU.
2. The GPU model-stack run. Colab T4.
3. The notebook.

Every step ends with a checkpoint. Do not start the next step until the
checkpoint holds.

---

## Before anything: freeze the Phase 6 results

The offline results are the reference for everything after them. Tag them so
they can always be recovered exactly:

```bash
git checkout main && git pull
git tag phase6-offline-v1
git push origin phase6-offline-v1
```

---

## Step 1: Decide the three defects (local, no GPU)

Each fix changes the system that every stored result measured. So each fix
needs a DD, a test, and a full regeneration of the offline results. Work on a
new branch:

```bash
git checkout -b fix/phase6-defects
```

### 1a. The structure chunker hides heading text (fix: yes)

**What happens.** The structure chunker takes a line such as "GATE 2027 IIT
Madras | Organizing Institute" as a heading and stores it in `chunk.section`
instead of `chunk.text`. Only citation labels read `section`. The embedder,
BM25, reranker, generator, controller and verifier all read `chunk.text`, so
the line is invisible to all of them. On `doc1.pdf` this makes q001's answer
unrecoverable, and it removes "2027" from every GATE chunk.

**Recommended fix.** Prepend the section heading to the text of each chunk the
structure chunker emits under it, so retrieval and generation both see it.
Keep `chunk.section` as it is for the citation label.

* Where: `src/chunking/structure.py`.
* Test first, as an invariant in `tests/test_chunking.py`: *every non-blank line
  of every page's text appears in the text of at least one chunk.* It fails
  today on `doc1.pdf`, and it is the rule the fix has to satisfy.
* Watch for:
  * the heading now counts toward `chunk_size` (characters, DD-030);
  * DD-032 says the fixed-size chunker records no section, and it should
    stay that way.
* Record it as DD-061.

### 1b. Bare `[n]` bibliography references read as citations (fix: yes)

**What happens.** `src/generation/citations.py` accepts a bare `[3]` as a
citation marker, because small models drop the `C` prefix. But source text
also contains bibliography references like "shrinking[3]". An answer that
copies that text gets `[3]` rewritten to `[C3]`, which means evidence block 3.
That is how the verifier came to reject q065's correct answer.

**Recommended fix.** Keep accepting bare `[n]`, *unless that exact bracketed
token already appears in the evidence text*. If it does, it is source text, so
leave it alone.

* Where: `resolve_citations` in `src/generation/citations.py`.
* Test: an answer copying "shrinking[3]." from evidence that contains
  "shrinking[3]." produces no citation to block 3. A model's bare `[1]` that is
  not in the evidence still resolves to block 1.
* Record it as DD-062.

### 1c. The controller's number rule (decide on dev; may be "no change")

**What happens.** The rule-based controller refuses unless every number in the
question appears in the context. 5 of its 13 refusals involve this rule.

**Why decide it after 1a.** Fixing 1a puts "2027" back in the GATE chunks,
which removes 3 of those 5 cases (q001, q002, q005). The remaining case in the
sample is q069, "iPhone 6".

**It affects the offline stack only.** The GPU run uses the LLM controller,
which does not have this rule. So "record it and change nothing" is a
legitimate decision.

If you do change it, decide on `--split dev` only, with a rule written in a DD
before you look. Example: "ignore single-digit numbers attached to a product
name". Record the decision either way as DD-063.

### 1d. Optional but recommended before step 2: resumable runs

`run-benchmark` writes `per_question.json` after every question, but it cannot
*resume*. If a Colab session dies 50 minutes into an agentic arm, that arm
restarts from question 1.

A `--resume` flag would load the existing records from `--out`, skip their
question ids, and keep appending. That is about 20 lines in
`run_benchmark_cli` plus a test, and it pays for itself on the first
disconnect. It is optional, because every arm can simply be re-run.

### 1e. Regenerate the offline results

After 1a-1c, every stored offline result is stale. The stability tests will
fail until the results are regenerated, because they re-run the offline stack
and compare it record by record against the stored results.

```bash
python -m src.cli run-benchmark --system dense --offline --out results/baseline
python -m src.cli run-benchmark --system hybrid --offline
python -m src.cli run-benchmark --system hybrid --no-rerank --offline --out results/ablations/reranker_off
python -m src.cli run-benchmark --system agentic --offline
python -m src.cli run-benchmark --system agentic --no-planner             --offline --out results/ablations/planner_off
python -m src.cli run-benchmark --system agentic --no-hybrid              --offline --out results/ablations/hybrid_off
python -m src.cli run-benchmark --system agentic --no-rerank              --offline --out results/ablations/reranker_off_agentic
python -m src.cli run-benchmark --system agentic --no-refinement          --offline --out results/ablations/refinement_off
python -m src.cli run-benchmark --system agentic --no-evidence-controller --offline --out results/ablations/evidence_controller_off
python -m src.cli run-benchmark --system agentic --no-verification        --offline --out results/ablations/verification_off
```

Then the comparisons, each to the path Phase 4-6 used:

```bash
python -m src.cli compare-runs --baseline results/baseline --system results/hybrid
python -m src.cli compare-runs --baseline results/baseline --system results/ablations/reranker_off
python -m src.cli compare-runs --baseline results/ablations/reranker_off --system results/hybrid --out results/hybrid/comparison_vs_reranker_off.json
python -m src.cli compare-runs --baseline results/baseline --system results/agentic
python -m src.cli compare-runs --baseline results/hybrid   --system results/agentic --out results/agentic/comparison_vs_hybrid.json
python -m src.cli compare-runs --baseline results/agentic  --system results/ablations/planner_off
# ... likewise for the other five agentic arms
python -m src.cli final-table
```

The dev sweeps in `results/experiments/` and the tuned eval arm were chosen
under the old system. There are two honest options:
* re-run the dev sweep, re-apply DD-058's rule and run eval once more; or
* delete the tuned row and say why.

Either way, **disclose that eval has now been looked at a second time**
(EVALUATION_PROTOCOL.md 5.2).

### Checkpoint 1

- [ ] DD-061, DD-062, DD-063 written.
- [ ] The line-preservation invariant passes.
- [ ] `python -m pytest` is green, apart from the two known failures in
  `tests/test_agentic.py`, unless you fixed those too.
- [ ] `results/` regenerated and `results/final/REPORT.md` re-read. Update
  STATUS.md 9.4 wherever a number moved.
- [ ] Merged to `main`, then tagged:
  ```bash
  git tag v0.7-gpu-run && git push origin v0.7-gpu-run
  ```
  DD-034 requires the GPU run to clone a pinned ref, never a branch.

---

## Step 2: The GPU model-stack run (Colab, T4)

The model stack is:
* generator `Qwen/Qwen3-4B-Instruct-2507`, 4-bit;
* embedder `BAAI/bge-m3`;
* reranker `BAAI/bge-reranker-v2-m3`;
* LLM planner, controller and verifier. `RAGConfig.default()` sets them all to
  `"llm"`, and they share the generator's weights (ARCHITECTURE.md 21).

Leaving out `--offline` selects all of this. No other flag is needed.

### 2.1 Where the results go: a separate root, never `results/`

`results/` holds the offline runs, and the stability tests compare against
them. Put the GPU runs under their own root, which has its own `results/`
inside it:

```text
model_stack/results/baseline/
model_stack/results/hybrid/
model_stack/results/agentic/
model_stack/results/ablations/...
model_stack/results/final/        <- final-table --root model_stack
```

`final-table --root model_stack` reads and writes only under that root, so the
report code needs no change. `compare-runs` refuses to compare a GPU run with
an offline one (EVALUATION_PROTOCOL.md 10), which catches a wrong path.

### 2.2 Colab setup cells

Set **Runtime → Change runtime type → T4 GPU**, then run:

```python
!nvidia-smi                                   # confirm a T4, ~15 GB
from google.colab import drive
drive.mount('/content/drive')                 # results must survive a disconnect

REF = "v0.7-gpu-run"                          # the pinned tag from checkpoint 1
!git clone --depth 1 --branch {REF} https://github.com/manishtiwari2/agentic-rag-pdf /content/agentic-pdf-rag
%cd /content/agentic-pdf-rag
!pip install -q -r requirements-models.txt

ROOT = "/content/drive/MyDrive/agentic-pdf-rag/model_stack"   # on Drive
!mkdir -p {ROOT}/results
!git rev-parse HEAD > {ROOT}/COMMIT.txt       # record which commit produced the run
```

**Model downloads.** The weights are several GB, and every fresh runtime
downloads them again. Setting `HF_HOME` to a Drive folder before the first run
keeps them across sessions:

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### 2.3 Smoke test before any long run

```python
!python -m src.cli ask --pdf benchmark/documents/doc5.pdf --question "What problem did FinFET face at 5nm?" --system agentic
!python -m src.cli run-benchmark --system agentic --limit 3 --out /tmp/smoke
```

Check three things:
* The answer carries `[C…]` citations and pages.
* `/tmp/smoke/results.json` reports `peak_vram_gb` under the 10 GB stopping
  rule and `llm_parse_failure_rate` below roughly 0.2. A high parse-failure rate
  means the LLM agents are mostly falling back to the rules (DD-055), and the
  run would not measure what it claims to.
* The per-question latency, from which you estimate the full run. Estimate it
  here rather than guessing: multiply by 76 questions per arm and 10 runs.

If VRAM is too high, the config refuses to start with a remedy. The fallback is
`--low-memory` (Qwen2.5-1.5B). Recording that as the stack is a DD, not a
quiet swap.

### 2.4 The runs

Run the systems in this order. One cell per run, so a disconnect loses one arm,
not all of them:

```python
R = f"{ROOT}/results"
!python -m src.cli run-benchmark --system dense  --out {R}/baseline
!python -m src.cli run-benchmark --system hybrid --out {R}/hybrid
!python -m src.cli run-benchmark --system hybrid --no-rerank --out {R}/ablations/reranker_off
!python -m src.cli run-benchmark --system agentic --out {R}/agentic
for flag, arm in [("--no-planner", "planner_off"), ("--no-hybrid", "hybrid_off"),
                  ("--no-rerank", "reranker_off_agentic"), ("--no-refinement", "refinement_off"),
                  ("--no-evidence-controller", "evidence_controller_off"),
                  ("--no-verification", "verification_off")]:
    !python -m src.cli run-benchmark --system agentic {flag} --out {R}/ablations/{arm}
```

**Dev sweep (open question 3, LLM version).** Run the iteration-cap sweep on dev
only:

```python
for n in (1, 2, 3):
    !python -m src.cli run-benchmark --system agentic --max-iterations {n} --split dev --out {R}/experiments/max_iterations/dev_{n}
```

**No threshold sweep.** `sufficiency_threshold` is read only by the rule-based
controller, and the LLM controller decides sufficiency itself.
`final-table` omits the tuned-threshold row automatically when that run does
not exist.

### 2.5 Comparisons and the report

These are the same comparisons as step 1e, with `{R}` in front of every path.
Then:

```python
!python -m src.cli final-table --root {ROOT}
!cat {ROOT}/results/final/REPORT.md
```

### 2.6 Bring the results home

Download `model_stack/` from Drive into the repository root, on a new branch:

```bash
git checkout -b feat/model-stack-run
# copy model_stack/ in, then:
git add model_stack && git commit
```

Then write up:
* **DD-064:** what the model-stack run found, per RQ and per component, read
  against DD-056 and DD-058's pre-declared metrics. Those declarations were
  made for this stack too, so do not add new metrics now.
* **STATUS.md 9.6:** the table and verdicts. Remove the "offline stand-in
  stack" caveat only from these new tables.
* The new tables must carry their own honesty line: "4-bit Qwen3-4B on a T4;
  one greedy run (EVALUATION_PROTOCOL.md 25.1)".

### Checkpoint 2

- [ ] `COMMIT.txt` matches the pinned tag.
- [ ] Every `config.json` under `model_stack/` names the real models, not
  `local/...`.
- [ ] No `compare-runs` call needed `--allow-confounded`.
- [ ] `peak_vram_gb` is a number, not null. Parse-failure rate is reported.
- [ ] RQ2 (reranker), RQ3 (agentic on multi-hop) and RQ4 (verifier) are
  answered or stated as null, with CIs.

---

## Step 3: The notebook (`notebooks/agentic_pdf_rag.ipynb`, currently empty)

The notebook is the user-facing entry point (DD-002, README.md section 12). It
must **call** `src/`, never copy it (DD-034). It should also present the step 2
results, not recompute them.

### Cell plan

Follows README.md section 12:

| # | Cell | Content |
| --- | --- | --- |
| 1 | Setup | `nvidia-smi`; clone at the pinned `REF`; `pip install`; `sys.path.insert` |
| 2 | Mode switch | `OFFLINE = not torch.cuda.is_available()`: offline runs anywhere, the model stack only on a GPU. Print which one is active, loudly |
| 3 | Config | `RAGConfig.offline()`, `RAGConfig.default()` or `RAGConfig.low_memory()`. Print `config.describe()` |
| 4 | Upload | `google.colab.files.upload()`, or a bundled benchmark PDF |
| 5 | Parse and chunk | `build_pipeline(config)`, `pipeline.index(pdf)`, then `pipeline.index_stats` and the first few chunks with their pages |
| 6 | Ask: dense vs agentic | The same question to `--system dense` and `agentic`, side by side: answer, citations with pages, abstention |
| 7 | Agent trace | For the agentic answer: planner sub-queries, rounds, controller verdict, verifier status, from `result.metadata["agent_trace"]` |
| 8 | Benchmark (optional, slow) | A clearly marked cell running `run-benchmark` with `--limit` by default |
| 9 | Results | Render `results/final/REPORT.md`, plus `model_stack/results/final/REPORT.md` if present, each with its stack caveat |
| 10 | Resources | Peak VRAM (`torch.cuda.max_memory_allocated`), per-question latency |

### Keeping it honest and testable

* **Build it from a script, not by hand.** Write `notebooks/build_notebook.py`,
  which uses `nbformat` to assemble the cells. A hand-edited `.ipynb` drifts,
  and a diff of one is unreadable.
* **Add a test** (`tests/test_notebook.py`):
  * the notebook is valid JSON;
  * it clones a tag, never `main` (DD-034);
  * every `src.` import in it exists.
* **Offline smoke run.** Execute cells 1-7 in offline mode with `nbclient`, in
  a test or by hand. That proves it runs top to bottom without a GPU.

### Checkpoint 3

- [ ] "Runtime → Run all" works in a fresh Colab runtime, in both offline and
  GPU modes.
- [ ] No result table is typed into the notebook. Every one is read from a
  `REPORT.md`.
- [ ] README.md section 13 (Quick Start) points at the notebook.
- [ ] EVALUATION_PROTOCOL.md 29's last checkbox, "reproducible from a fresh
  Colab runtime", can be ticked.

---

## What Claude can and cannot do here

| Step | Who |
| --- | --- |
| 1a-1e (fixes, DDs, tests, regeneration) | Claude can do all of it locally; ask for it |
| 2 (GPU run) | **You**, in Colab: this machine has no GPU. Claude can prepare the cells, review your smoke-test output, and write DD-064 and STATUS 9.6 from the downloaded results |
| 3 (notebook) | Claude can write `build_notebook.py`, the notebook and its test. The "Run all in Colab" check at the end needs you |
