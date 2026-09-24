"""Build the two Colab notebooks from source (DD-068).

    python notebooks/build_notebooks.py                 # REF = v0.7-gpu-run
    python notebooks/build_notebooks.py --ref v0.8

A hand-edited ``.ipynb`` drifts, and its diff is unreadable. Both notebooks are
generated here instead, and ``tests/test_notebooks.py`` fails if a committed
notebook differs from what this script produces. Edit this file, re-run it,
commit both.

* ``agentic_pdf_rag.ipynb`` is the deliverable (DD-002, PROJECT_SPEC.md
  section 12): the full pipeline, explained cell by cell, on any PDF. With no
  GPU it runs the offline stand-in stack top to bottom on a CPU.
* ``gpu_benchmark_run.ipynb`` is the runner for the model-stack benchmark on a
  Colab T4: every system and ablation, resumable, with results on Drive.

Both clone the repository at a pinned tag, never a branch (DD-034), and call
``src/``; neither copies it. Neither contains a result number: metrics are read
from the ``REPORT.md`` files ``final-table`` writes.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/manishtiwari2/agentic-rag-pdf"
DEFAULT_REF = "v0.7-gpu-run"
HERE = pathlib.Path(__file__).resolve().parent
DELIVERABLE = HERE / "agentic_pdf_rag.ipynb"
GPU_RUNNER = HERE / "gpu_benchmark_run.ipynb"

#: A tag, never a branch: DD-034 pins the code a notebook runs.
_TAG = re.compile(r"^v\d+(?:\.\d+)*(?:[-.][0-9A-Za-z.-]+)?$")

_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
}


def _notebook(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    notebook = new_notebook(metadata=dict(_METADATA))
    for index, (kind, source) in enumerate(cells):
        source = source.strip("\n")
        cell = new_markdown_cell(source) if kind == "md" else new_code_cell(source)
        # Fixed ids keep the output byte-identical between builds.
        cell["id"] = f"cell-{index:02d}"
        notebook.cells.append(cell)
    return notebook


def _md(text: str) -> tuple[str, str]:
    return ("md", text)


def _code(text: str) -> tuple[str, str]:
    return ("code", text)


SETUP = '''
import os, pathlib, subprocess, sys

REF = "{ref}"   # a pinned tag (DD-034): the exact code this notebook was built for
REPO_URL = "{repo}"
IN_COLAB = "google.colab" in sys.modules


def sh(*command):
    """Run a command, stream its output, and stop the notebook if it fails."""
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True)


def find_checkout():
    for base in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        if (base / "src" / "pipeline.py").exists():
            return base
    return None


ROOT = find_checkout()
if ROOT is None:
    ROOT = pathlib.Path("/content/agentic-pdf-rag" if IN_COLAB else "agentic-pdf-rag").resolve()
    if not ROOT.exists():
        sh("git", "clone", "--depth", "1", "--branch", REF, REPO_URL, ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Code:", ROOT)
print("Commit:", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown")
'''


def deliverable(ref: str) -> nbformat.NotebookNode:
    return _notebook([
        _md(f"""
# Agentic RAG over any PDF, with local models only

This notebook is the project's entry point (DD-002). It runs the whole pipeline on a PDF you
upload: parsing (with OCR for scanned pages), chunking, indexing, a dense baseline, the agentic
system, a chat interface, and the benchmark metrics. The code lives in the repository's `src/`
package; this notebook calls it and copies none of it (DD-034).

**Two modes, chosen automatically in cell 2:**

* **Model stack** (a GPU is present, e.g. Colab's free T4): Qwen3-4B-Instruct-2507 in 4-bit,
  bge-m3 embeddings, the bge-reranker-v2-m3 cross-encoder, and LLM planner, evidence controller
  and verifier sharing the generator's weights.
* **Offline stand-in stack** (no GPU): a hashing embedder, a scripted extractive generator, a
  term-overlap reranker and rule-based agents. It runs anywhere on a CPU with no downloads, and
  its answers are far weaker. It exists so every cell can run and be tested without a GPU.

In Colab: *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
Pinned code version: `{ref}`.
"""),
        _md("""
## 1. Environment setup

Finds the code, or clones the repository at the pinned tag. A tag, never a branch, so this
notebook always runs the code it was written for (DD-034). The commit printed at the end is the
one every answer below came from.
"""),
        _code(SETUP.format(ref=ref, repo=REPO_URL)),
        _md("""
## 2. Dependencies

In Colab this installs the Tesseract OCR engine (a system package; OCR is on by default, DD-067)
and `requirements-colab.txt`: the model stack, Gradio for the chat interface and pytesseract.
Every licence is permissive; they are listed in `NOTICE`. Outside Colab nothing is installed:
install `requirements-colab.txt` yourself, or rely on the offline stack.
"""),
        _code('''
if IN_COLAB:
    sh("apt-get", "install", "-y", "-qq", "tesseract-ocr")
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt")
else:
    print("Not in Colab: skipping installs. For everything: pip install -r requirements-colab.txt")
'''),
        _md("""
## 3. Mode switch and model loading

No CUDA device means the offline stand-in stack, announced loudly, because its numbers are not
the model stack's. With a GPU, the model-stack configuration is validated here: it refuses to
start if it would not fit the T4 budget (MODEL_SELECTION.md 9.1) or if it would make a
non-commercially licensed model the default generator (DD-024). The weights themselves load
lazily, on the first question.
"""),
        _code('''
import json

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:
    torch, CUDA = None, False
OFFLINE = not CUDA

banner = "=" * 78
if OFFLINE:
    print(banner)
    print("OFFLINE MODE: no GPU found. Running the dependency-free STAND-IN stack.")
    print("Hashing embedder, scripted extractive generator, rule-based agents.")
    print("Answers and numbers below are NOT the model stack's (Qwen3-4B + bge-m3).")
    print(banner)
else:
    print(banner)
    print("MODEL STACK on", torch.cuda.get_device_name(0))
    print(banner)

from src.chat.ui import make_config

config = make_config("agentic", offline=OFFLINE)
print(json.dumps(config.describe(), indent=2))
'''),
        _md("""
## 4. PDF upload

Set `UPLOAD = True` to upload your own PDF in Colab (text or scanned). Otherwise the notebook uses
`benchmark/documents/doc5.pdf`, a CC-BY-4.0 paper on semiconductor scaling bundled with the
benchmark, so *Run all* works unattended.
"""),
        _code('''
UPLOAD = False
PDF = "benchmark/documents/doc5.pdf"

if UPLOAD and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    PDF = next(iter(uploaded))
print("PDF:", PDF)
'''),
        _md("""
## 5. Parsing, chunking and index construction

`pipeline.index` parses the PDF into 1-based pages, OCRs any page that has no text layer,
chunks it along headings, paragraphs and tables (DD-009, DD-061), embeds the chunks and builds
the index. Every chunk carries the pages it came from; citations are resolved from those pages
by code, never written by the model (DD-031).
"""),
        _code('''
import gc
import time

from src.pipeline import build_pipeline


def free_memory():
    """Return a deleted pipeline's GPU memory before the next is built.

    One T4, one set of weights at a time (ARCHITECTURE.md section 21): call it
    after deleting the pipeline.
    """
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


dense = build_pipeline(make_config("dense", offline=OFFLINE))
document = dense.index(PDF)
stats = dense.index_stats
ocr_pages = [p.page_number for p in document.pages if p.metadata.get("ocr")]
print(f"{document.source_name}: {document.page_count} pages, {stats['chunks']} chunks, "
      f"{len(ocr_pages)} OCR'd pages")
print(f"parse {stats['parse_s']} s, chunk {stats['chunk_s']} s, embed + index {stats['embed_index_s']} s")
for chunk in dense.chunks[:3]:
    print(f"\\n[{chunk.citation_label()}] section: {chunk.section}")
    print(chunk.text[:300])
'''),
        _md("""
### 5b. A scanned PDF, read by OCR

A scanned page is a picture with no text layer. This cell draws a page of text into an image,
saves it as an image-only PDF, and parses it: the parser renders the page and hands it to
Tesseract (DD-067). Without Tesseract installed, the error says how to install it rather than
indexing an empty page.
"""),
        _code('''
from PIL import Image, ImageDraw, ImageFont

from src.errors import OCRUnavailableError
from src.ingestion.parser import build_parser

lines = ["Scanned report, page 1", "",
         "Median latency on the T4 GPU was 4.2 seconds per question.",
         "Peak memory stayed below ten gigabytes."]
image = Image.new("L", (2480, 3508), 255)
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 64)
except OSError:
    font = ImageFont.load_default()
for row, line in enumerate(lines):
    draw.text((200, 300 + row * 110), line, fill=0, font=font)
import tempfile

SCANNED = str(pathlib.Path(tempfile.gettempdir(), "scanned_demo.pdf"))
image.save(SCANNED, "PDF", resolution=300)

try:
    started = time.perf_counter()
    scanned = build_parser(config.ingestion).parse(SCANNED)
    seconds = time.perf_counter() - started
    print(f"OCR'd pages: {scanned.metadata.get('ocr_pages')}, {seconds:.2f} s per page")
    print(scanned.page(1).text)
except OCRUnavailableError as exc:
    print("OCR is unavailable here:", exc)
'''),
        _md("""
## 6. Baseline RAG: dense retrieval, then generation

Baseline A (DD-013): embed the question, take the top-5 chunks, generate an answer that cites
them with `[C1]`-style markers, which code maps to pages. A question the evidence cannot answer
gets a fixed refusal sentence. The questions are the benchmark's own for this PDF, read from
`benchmark/questions.json`, not typed in here.
"""),
        _code('''
from src.benchmark.schema import load_questions

questions = [q for q in load_questions("benchmark/questions.json") if q.document == pathlib.Path(PDF).name]
EXAMPLES = [q.question for q in questions[:3]] or [
    "What is this document about?", "What are its main findings?", "What limitations does it state?"]
unanswerable = [q.question for q in questions if q.is_unanswerable]
EXAMPLES += unanswerable[:1]

dense_results = []
for question in EXAMPLES:
    result = dense.ask(question)
    dense_results.append(result)
    print("Q:", question)
    print(result.format(), "\\n")
del dense
free_memory()
'''),
        _md("""
## 7. Agentic RAG

The agentic system (DD-050) drives hybrid retrieval (BM25 + dense, fused by RRF, then reranked)
through a bounded loop: a planner may split the question into sub-queries, an evidence controller
judges whether the retrieved evidence suffices, refinement re-queries for what is missing (at most
two rounds), and a verifier checks each claim against the evidence it cites, regenerating once
or refusing. Every loop has a hard cap, recorded in the trace.
"""),
        _code('''
agentic = build_pipeline(make_config("agentic", offline=OFFLINE))
agentic.index(PDF)
agentic_results = [agentic.ask(question) for question in EXAMPLES]
'''),
        _md("""
## 8. Example queries: dense and agentic side by side

The same questions, both systems. **Refused** marks an abstention. Pages come from the chunks
each marker points at.
"""),
        _code('''
from IPython.display import Markdown, display

from src.chat.ui import format_answer


def cell(result):
    return format_answer(result).replace("\\n", "<br>").replace("|", "\\\\|")


rows = ["| Question | Dense RAG | Agentic RAG |", "| --- | --- | --- |"]
for question, d, a in zip(EXAMPLES, dense_results, agentic_results):
    rows.append(f"| {question} | {cell(d)} | {cell(a)} |")
display(Markdown("\\n".join(rows)))
'''),
        _md("""
## 9. The agent trace

What the agentic system did for the first question: the planner's sub-queries, each retrieval
round with the evidence controller's verdict, why the loop stopped, and the verifier's status.
The full trace is in `result.metadata["agent_trace"]`.
"""),
        _code('''
from src.chat.ui import format_trace

display(Markdown(format_trace(agentic_results[0])))
bounds = agentic_results[0].metadata["agent_trace"]["bounds"]
print(json.dumps(bounds, indent=2))
'''),
        _md("""
## 10. The chat interface

A Gradio app over the same pipeline (DD-066): upload one or more PDFs (text or scanned), choose
dense, hybrid or agentic, and chat. Follow-up questions ("what about its latency?") are rewritten
into standalone ones before retrieval (DD-065). In Colab, `share=True` prints a public link.
Outside Colab the app is built but not launched.
"""),
        _code('''
del agentic
free_memory()
try:
    from src.chat.ui import build_app
    app = build_app(offline=OFFLINE)
    if IN_COLAB:
        app.launch(share=True, debug=False)
    else:
        print("Chat app built; call app.launch() to open it.")
except ImportError:
    print("Gradio is not installed: pip install -r requirements-colab.txt")
'''),
        _md("""
## 11. Benchmark execution (optional, slow)

The benchmark is 76 questions over 5 PDFs, each with verified evidence pages (BENCHMARK_SPEC.md).
This cell runs the agentic system on the first `LIMIT` questions only, into a scratch directory,
to show the harness working. A full model-stack run of every system takes hours: use
`notebooks/gpu_benchmark_run.ipynb` for that. Set `RUN_BENCHMARK = False` to skip.
"""),
        _code('''
RUN_BENCHMARK = True
LIMIT = 5
OUT = str(pathlib.Path(tempfile.gettempdir(), "benchmark_demo"))  # never results/

if RUN_BENCHMARK:
    command = [sys.executable, "-m", "src.cli", "run-benchmark", "--system", "agentic",
               "--limit", str(LIMIT), "--out", OUT]
    if OFFLINE:
        command.append("--offline")
    sh(*command)
    summary = json.loads(pathlib.Path(OUT, "results.json").read_text())["summary"]
    print({k: summary[k] for k in ("n_questions", "accuracy_all", "recall_at_5", "latency_median_s", "peak_vram_gb")})
'''),
        _md("""
## 12. Evaluation metrics

The results tables are **read from the reports `final-table` generates**, never typed into this
notebook. `results/final/REPORT.md` is the offline stand-in stack, measured on a machine with no
GPU. `model_stack/results/final/REPORT.md` is the model stack, present once the GPU run has been
brought home. Each is shown with the stack it measured.
"""),
        _code('''
import re

REPORTS = [
    ("results/final/REPORT.md",
     "**Offline stand-in stack** (hashing embedder, scripted generator, rule-based agents; "
     "no GPU). These numbers are not the model stack's."),
    ("model_stack/results/final/REPORT.md",
     "**Model stack**: 4-bit Qwen3-4B-Instruct-2507, bge-m3, bge-reranker-v2-m3 and LLM agents, "
     "on a Colab T4; one greedy run (EVALUATION_PROTOCOL.md 25.1)."),
]


def section(text, heading):
    """One `###` section of a report, by the start of its heading."""
    match = re.search(rf"^### {re.escape(heading)}.*?(?=^### |\\Z)", text, flags=re.M | re.S)
    return match.group(0) if match else ""


reports = {}
for path, caveat in REPORTS:
    if pathlib.Path(path).exists():
        reports[path] = pathlib.Path(path).read_text(encoding="utf-8")
        display(Markdown(f"> {caveat}\\n\\n" + section(reports[path], "Eval split")))
    else:
        display(Markdown(f"`{path}` is not present yet."))
'''),
        _md("""
## 13. Ablation results

Each agentic component removed in turn (DD-057), judged by the rule declared before any run
(DD-058): a component earns its cost only if its declared metric improves significantly and
nothing it should protect gets worse. Read from the same reports.
"""),
        _code('''
for path, caveat in REPORTS:
    if path in reports:
        display(Markdown(f"> {caveat}\\n\\n" + section(reports[path], "Ablations")))
'''),
        _md("""
## 14. Latency and memory

Per-question latency of the answers above, and peak GPU memory. Offline, peak VRAM is `null`:
there is no GPU, so nothing was measured, and reporting 0 would claim a measurement.
"""),
        _code('''
import statistics

for name, results in (("dense", dense_results), ("agentic", agentic_results)):
    latencies = [r.metadata["latency_s"] for r in results]
    print(f"{name:8s} median {statistics.median(latencies):.3f} s, max {max(latencies):.3f} s "
          f"over {len(latencies)} questions")
peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if CUDA else None
print("peak VRAM (GB):", peak_vram_gb if peak_vram_gb is not None else "null (no GPU, not measured)")
'''),
        _md("""
## 15. Final demonstration: a short conversation

A three-turn chat with the agentic system through `ChatSession`: the second and third questions
lean on the first, and the trace shows how each was rewritten before retrieval.
"""),
        _code('''
from src.chat.session import ChatSession

pipeline = build_pipeline(make_config("agentic", offline=OFFLINE))
pipeline.index(PDF)
session = ChatSession(pipeline)
for question in (EXAMPLES[0], "What about its limitations?", "Why does that matter?"):
    result = session.ask(question)
    conversation = result.metadata["conversation"]
    print("You:      ", question)
    if conversation["rewritten"]:
        print("Rewritten:", conversation["standalone_question"])
    print("Answer:   ", format_answer(result).replace("\\n", "\\n           "), "\\n")
'''),
    ])


GPU_SETUP = '''
import os, pathlib, subprocess, sys

REF = "{ref}"   # the pinned tag this run measures (DD-034)
REPO_URL = "{repo}"

from google.colab import drive
drive.mount("/content/drive")
DRIVE = pathlib.Path("/content/drive/MyDrive/agentic-pdf-rag")
ROOT = DRIVE / "model_stack"                    # final-table --root; never results/
R = ROOT / "results"
R.mkdir(parents=True, exist_ok=True)
# Model weights on Drive: several GB, downloaded once instead of every session.
os.environ["HF_HOME"] = str(DRIVE / "hf_cache")


def sh(*command):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True)


CODE = pathlib.Path("/content/agentic-pdf-rag")
if not CODE.exists():
    sh("git", "clone", "--depth", "1", "--branch", REF, REPO_URL, CODE)
os.chdir(CODE)
sys.path.insert(0, str(CODE))
sh("apt-get", "install", "-y", "-qq", "tesseract-ocr")
sh(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt")
'''

#: (label, run-benchmark flags, output directory under R). The same arms and
#: paths as Phases 4-6 (DD-057), with the dev iteration sweep last.
GPU_RUNS = [
    ("Baseline A: dense", ["--system", "dense"], "baseline"),
    ("Baseline B: hybrid", ["--system", "hybrid"], "hybrid"),
    ("Reranker OFF on Baseline B", ["--system", "hybrid", "--no-rerank"], "ablations/reranker_off"),
    ("Agentic", ["--system", "agentic"], "agentic"),
    ("Agentic, planner OFF", ["--system", "agentic", "--no-planner"], "ablations/planner_off"),
    ("Agentic, hybrid OFF", ["--system", "agentic", "--no-hybrid"], "ablations/hybrid_off"),
    ("Agentic, reranker OFF", ["--system", "agentic", "--no-rerank"], "ablations/reranker_off_agentic"),
    ("Agentic, refinement OFF", ["--system", "agentic", "--no-refinement"], "ablations/refinement_off"),
    ("Agentic, evidence controller OFF", ["--system", "agentic", "--no-evidence-controller"],
     "ablations/evidence_controller_off"),
    ("Agentic, verification OFF", ["--system", "agentic", "--no-verification"], "ablations/verification_off"),
] + [
    (f"Dev sweep: --max-iterations {n}",
     ["--system", "agentic", "--max-iterations", str(n), "--split", "dev"],
     f"experiments/max_iterations/dev_{n}")
    for n in (1, 2, 3)
]

#: (baseline, system, out or None): the Phase 4-6 comparison pairs.
GPU_COMPARISONS = [
    ("baseline", "hybrid", None),
    ("baseline", "ablations/reranker_off", None),
    ("ablations/reranker_off", "hybrid", "hybrid/comparison_vs_reranker_off.json"),
    ("baseline", "agentic", None),
    ("hybrid", "agentic", "agentic/comparison_vs_hybrid.json"),
] + [
    ("agentic", f"ablations/{arm}", None)
    for arm in ("planner_off", "hybrid_off", "reranker_off_agentic", "refinement_off",
                "evidence_controller_off", "verification_off")
]


def gpu_runner(ref: str) -> nbformat.NotebookNode:
    cells = [
        _md(f"""
# Model-stack benchmark run (Colab T4)

Runs every system and ablation of Phases 4-6 against the model stack: 4-bit
Qwen3-4B-Instruct-2507, bge-m3, bge-reranker-v2-m3 and LLM agents. Results go to Drive under
`model_stack/results/`, **never** `results/`, which holds the offline runs the stability tests
compare against.

1. *Runtime → Change runtime type → T4 GPU.*
2. Run the cells in order. Every run cell passes `--resume`: after a disconnect, re-run the
   setup cell and then the cell that was interrupted. It continues from the last finished
   question and refuses if the configuration changed (DD-064).
3. Paste the smoke-test block (cell 4) back before starting the long runs.

Pinned code version: `{ref}`. No threshold sweep: the LLM controller decides sufficiency itself.
"""),
        _md("## 1. GPU check"),
        _code('''
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU"
print(torch.cuda.get_device_name(0))
'''),
        _md("## 2. Drive, code at the pinned tag, Tesseract and dependencies"),
        _code(GPU_SETUP.format(ref=ref, repo=REPO_URL)),
        _md("## 3. Record which commit produced the run"),
        _code('''
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
tag = subprocess.run(["git", "describe", "--tags", "--exact-match"], capture_output=True, text=True).stdout.strip()
(ROOT / "COMMIT.txt").write_text(f"{commit}\\n{tag or 'no tag'}\\n")
print("commit", commit, "tag", tag or "NONE")
assert tag == REF, f"HEAD is not the pinned tag {REF}"
'''),
        _md("""
## 4. Smoke test: one question, then three benchmark questions

Before hours of runs, check that the stack loads, fits and parses its own output. The block
printed at the end is what to paste back. `llm_parse_failure_rate` above about 0.2 means the LLM
agents are mostly falling back to the rules (DD-055), and the run would not measure what it
claims to.
"""),
        _code('''
import json

sh(sys.executable, "-m", "src.cli", "ask", "--pdf", "benchmark/documents/doc5.pdf",
   "--question", "What problem did FinFET face at 5nm?", "--system", "agentic")
SMOKE = pathlib.Path("/content/smoke")
sh(sys.executable, "-m", "src.cli", "run-benchmark", "--system", "agentic", "--limit", "3",
   "--out", SMOKE)

summary = json.loads((SMOKE / "results.json").read_text())["summary"]
records = json.loads((SMOKE / "per_question.json").read_text())
per_question = summary["latency_median_s"]
# Upper bound: agentic latency for every run (dense and hybrid are faster).
question_runs = 10 * 76 + 3 * 33
print("=" * 30, "PASTE FROM HERE", "=" * 30)
print("answer (q1):           ", records[0]["answer"][:300])
print("citations (q1):        ", records[0]["cited_pages"])
print("peak_vram_gb:          ", summary["peak_vram_gb"])
print("llm_parse_failure_rate:", summary["llm_parse_failure_rate"])
print("seconds per question:  ", per_question)
print(f"estimated full run:     {per_question * question_runs / 3600:.1f} h "
      f"({question_runs} question-runs, agentic latency as an upper bound)")
print("=" * 30, "PASTE TO HERE", "=" * 32)
'''),
        _md("""
## 5. The runs

One cell per run, so a disconnect costs one arm at most, and `--resume` recovers even that.
"""),
        _code('''
def run(*flags, out):
    sh(sys.executable, "-m", "src.cli", "run-benchmark", *flags, "--resume", "--out", R / out)
'''),
    ]
    for label, flags, out in GPU_RUNS:
        args = ", ".join(f'"{f}"' for f in flags)
        cells.append(_md(f"### {label}"))
        cells.append(_code(f'run({args}, out="{out}")'))
    compare_lines = ["def compare(baseline, system, out=None):",
                     '    command = [sys.executable, "-m", "src.cli", "compare-runs",',
                     '               "--baseline", R / baseline, "--system", R / system]',
                     "    if out:",
                     '        command += ["--out", R / out]',
                     "    sh(*command)",
                     ""]
    for baseline, system, out in GPU_COMPARISONS:
        suffix = f', out="{out}"' if out else ""
        compare_lines.append(f'compare("{baseline}", "{system}"{suffix})')
    cells += [
        _md("""
## 6. Paired comparisons

The same pairs as Phases 4-6. `compare-runs` refuses a pair that differs in a held-constant
field (EVALUATION_PROTOCOL.md 10); none of these should need `--allow-confounded`.
"""),
        _code("\n".join(compare_lines)),
        _md("## 7. The final table, ablation verdicts and error analysis"),
        _code('''
sh(sys.executable, "-m", "src.cli", "final-table", "--root", ROOT)
from IPython.display import Markdown, display
display(Markdown((R / "final" / "REPORT.md").read_text()))
'''),
        _md("""
## 8. Bring the results home

Download `MyDrive/agentic-pdf-rag/model_stack/` (or the zip this cell writes) into the
repository root, on a new branch. Follow `NEXT_STEPS.md` from there.
"""),
        _code('''
import shutil
archive = shutil.make_archive(str(DRIVE / "model_stack"), "zip", root_dir=DRIVE, base_dir="model_stack")
print("Wrote", archive)
'''),
    ]
    return _notebook(cells)


def build(ref: str = DEFAULT_REF) -> dict[pathlib.Path, str]:
    """The notebooks' contents, keyed by path, without writing them."""
    if not _TAG.match(ref) or ref in ("main", "master"):
        raise ValueError(f"REF must be a release tag such as v0.7-gpu-run, not {ref!r} (DD-034)")
    return {
        DELIVERABLE: nbformat.writes(deliverable(ref)) + "\n",
        GPU_RUNNER: nbformat.writes(gpu_runner(ref)) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default=DEFAULT_REF, help="The pinned tag to clone (DD-034).")
    args = parser.parse_args(argv)
    for path, text in build(args.ref).items():
        nbformat.validate(nbformat.reads(text, as_version=4))
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote", path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
