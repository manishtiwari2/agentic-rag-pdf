# Next Steps: the GPU Run

Everything up to the GPU run is done, on `feat/colab-chatbot-release`:
* the three Phase 6 defect fixes (DD-061 to DD-063);
* resumable runs (DD-064);
* follow-up questions (DD-065) and the chat UI (DD-066);
* OCR (DD-067);
* the notebooks (DD-068);
* regenerated offline results (DD-069).

Every number in the repository is still from the **offline stand-in stack**.
What remains needs a GPU, and it is yours to run.

Each step ends with a checkpoint. Do not start the next step until the
checkpoint holds.

---

## Step 1: Merge and tag

The notebooks clone a pinned tag, never a branch (DD-034). They are built for
`v0.7-gpu-run`, and until that tag exists a fresh Colab clone fails at its
first cell.

```bash
git checkout main && git pull
git tag v0.7-gpu-run
git push origin v0.7-gpu-run
```

If you tag something else, rebuild the notebooks for it and commit them before
tagging, so the tag contains notebooks that clone it:

```bash
python notebooks/build_notebooks.py --ref v0.7.1-gpu-run
```

### Checkpoint 1

- [ ] `git ls-remote --tags origin v0.7-gpu-run` prints the tag.
- [ ] The tag's `notebooks/gpu_benchmark_run.ipynb` has `REF = "v0.7-gpu-run"`.

---

## Step 2: The smoke test (Colab, T4, about 15 minutes)

1. Open `notebooks/gpu_benchmark_run.ipynb` in Colab: *File → Open notebook →
   GitHub*, paste the repository URL, and pick the tag.
2. *Runtime → Change runtime type → T4 GPU.*
3. Run cells 1 to 4:
   * GPU check;
   * Drive, clone, Tesseract and install;
   * `COMMIT.txt`;
   * smoke test.

   Drive asks for permission once. Weights download into `HF_HOME` on Drive, so
   only the first session pays for them.

### What to paste back

Cell 4 ends with a block between `PASTE FROM HERE` and `PASTE TO HERE`. Paste
it into the conversation:

```text
answer (q1):            ...
citations (q1):         [...]
peak_vram_gb:           ...
llm_parse_failure_rate: ...
seconds per question:   ...
estimated full run:     ... h
```

What each line should show:
* **answer:** carries `[C…]` citations, and `citations` lists pages.
* **peak_vram_gb:** a number under 10 (the stopping rule), not null.
* **llm_parse_failure_rate:** below roughly 0.2. Higher means the LLM agents
  are mostly falling back to the rules (DD-055), and the run would not measure
  what it claims to.
* **estimated full run:** an upper bound. It prices every run at agentic
  latency.

If VRAM is too high, the configuration refuses to start and names a remedy.
The fallback is `--low-memory` (Qwen2.5-1.5B). Recording that as the stack is a
DD, not a quiet swap.

### Checkpoint 2

- [ ] The pasted block is reviewed, and the full run is expected to fit the
  time you have.

---

## Step 3: The full run (Colab, several sessions)

Run cells 5 onwards, one cell per system or arm. Every run passes `--resume`.
After a disconnect:
1. re-run cell 2 (setup);
2. re-run cell 5 (the `run` helper);
3. re-run the interrupted cell.

It continues from the last finished question. It refuses if the configuration
changed, so a changed flag cannot silently mix two experiments (DD-064).

The notebook runs, in order:
* dense, hybrid, and hybrid `--no-rerank`;
* agentic and its six arms (DD-057);
* `--max-iterations 1`, `2` and `3` on `--split dev`;
* the Phase 4-6 `compare-runs` pairs;
* `final-table --root <Drive>/model_stack`.

It runs **no threshold sweep.** The LLM controller decides sufficiency itself.
The threshold also chooses which queries refinement rewrites, and it applies
when an LLM decision fails to parse (DD-068). Both stay at the default 0.5.

All results go under `MyDrive/agentic-pdf-rag/model_stack/results/`, **never
`results/`**. `results/` holds the offline runs, and the stability tests
compare against it.

### Checkpoint 3

- [ ] `model_stack/COMMIT.txt` names the tag.
- [ ] Every `config.json` under `model_stack/` names the real models, not
  `local/...`.
- [ ] No `compare-runs` call needed `--allow-confounded`.
- [ ] `peak_vram_gb` is a number in every `results.json`, and the parse-failure
  rate is reported.

---

## Step 4: Bring `model_stack/` home

The last cell zips `model_stack/` on Drive. Download it and unzip it into the
repository root, on a new branch:

```bash
git checkout main && git pull
git checkout -b feat/model-stack-run
```

Then copy `model_stack/` in, and:

```bash
python -m src.cli final-table --root model_stack
```

```bash
git add model_stack
```

```bash
git commit -m "Add the model-stack benchmark run"
```

`final-table --root model_stack` reads and writes only under `model_stack/`.
It regenerates the report the notebook already wrote, as a check. The
deliverable notebook shows `model_stack/results/final/REPORT.md` automatically
once it exists.

### The write-up (Claude can do this from the downloaded files)

* **DD-070.** What the model-stack run found, per RQ and per component, read
  against DD-056 and DD-058's pre-declared metrics. Those declarations were
  made for this stack too, so no new metrics are added now.
* **STATUS.md.** A new section with the table and verdicts. Its tables carry
  their own honesty line: "4-bit Qwen3-4B on a T4; one greedy run
  (EVALUATION_PROTOCOL.md 25.1)". The offline caveat stays on every offline
  table.
* **README.md.** Replace "offline numbers only" in the limitations.

### Checkpoint 4

- [ ] RQ2 (reranker), RQ3 (agentic on multi-hop) and RQ4 (verifier) are
  answered or stated as null, with CIs.
- [ ] *Runtime → Run all* on `notebooks/agentic_pdf_rag.ipynb` works in a fresh
  Colab runtime with a GPU. That is the last box in EVALUATION_PROTOCOL.md 29.

---

## Who does what

| Step | Who |
| --- | --- |
| 1 merge and tag | **You** |
| 2-3 the Colab run | **You**, on a T4. Claude reviews the pasted smoke-test block |
| 4 bring home, write up | You copy `model_stack/` in; Claude can write DD-070 and the STATUS section from it |
