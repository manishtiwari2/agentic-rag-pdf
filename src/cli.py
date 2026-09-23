"""Command-line entry point for the dense baseline.

    python -m src.cli ask --pdf paper.pdf --question "What was measured?"
    python -m src.cli inspect --pdf paper.pdf --offline
    python -m src.cli run-benchmark --out results/baseline --offline

The notebook is the deliverable (DD-002); this exists so a single PDF can be run
end to end without one, which is what the Phase 1 and Phase 2 exit criteria ask
for.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from .benchmark.inspect import inspect_page
from .benchmark.manifest import DEFAULT_MANIFEST_PATH
from .benchmark.schema import DEFAULT_DOCUMENTS_DIR, DEFAULT_QUESTIONS_PATH
from .benchmark.validate import validate_benchmark
from .config import ChunkingConfig, IngestionConfig, RAGConfig, RetrievalConfig
from .errors import RAGError
from .pipeline import DenseRAGPipeline


def _build_config(args: argparse.Namespace) -> RAGConfig:
    config = RAGConfig.offline() if args.offline else RAGConfig.default()
    if args.low_memory:
        config = RAGConfig.low_memory()

    replace = dataclasses.replace
    chunking: ChunkingConfig = config.chunking
    if args.strategy:
        chunking = replace(chunking, strategy=args.strategy)
    if args.chunk_size:
        chunking = replace(chunking, chunk_size=args.chunk_size)
    if args.chunk_overlap is not None:
        chunking = replace(chunking, chunk_overlap=args.chunk_overlap)

    retrieval: RetrievalConfig = config.retrieval
    if args.top_k:
        retrieval = replace(retrieval, top_k=args.top_k)

    return replace(config, chunking=chunking, retrieval=retrieval)


def _report(question_run) -> None:
    """One line per question, so a 76-question run is not a silent wait."""
    score = question_run.score
    mark = "." if not score.failed else (score.error_category or "?")
    print(
        f"{score.question_id} {mark:<16} "
        f"correct={score.answer.correct:.0f} "
        f"recall@5={score.retrieval.recall.get(5)} "
        f"abstained={score.abstention.abstained}",
        flush=True,
    )


def _format_summary(summary: dict) -> str:
    """The headline figures, with the denominators section 21.1 requires."""
    def show(value):
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"System: {summary['system']}    n = {summary['n_questions']}",
        "",
        f"  accuracy (all)          {show(summary['accuracy_all'])}"
        f"   n={summary['n_questions']}",
        f"  accuracy (answerable)   {show(summary['accuracy_answerable'])}"
        f"   n={summary['n_answerable']}",
        f"  abstention accuracy     {show(summary['abstention_accuracy'])}"
        f"   n={summary['n_unanswerable']}",
        "",
        f"  Recall@1/3/5/10         {show(summary['recall_at_1'])} / "
        f"{show(summary['recall_at_3'])} / {show(summary['recall_at_5'])} / "
        f"{show(summary['recall_at_10'])}   n={summary['n_retrieval_scored']}",
        f"  Full-Recall@5/10        {show(summary['full_recall_at_5'])} / "
        f"{show(summary['full_recall_at_10'])}",
        f"  Full-Recall@5 (multi)   "
        f"{show(summary['full_recall_at_5_multi_hop_comparison'])}"
        f"   n={summary['n_multi_evidence_types']}",
        f"  MRR@10                  {show(summary['mrr_at_10'])}",
        "",
        f"  faithfulness            {show(summary['faithfulness'])}",
        f"  citation any-correct    {show(summary['citation_any_correct'])}",
        f"  citation precision      {show(summary['citation_precision'])}",
        "",
        f"  false-answer rate       {show(summary['false_answer_rate'])}"
        f"   n={summary['n_unanswerable']}",
        f"  over-abstention rate    {show(summary['over_abstention_rate'])}"
        f"   n={summary['n_answerable']}",
        f"  unsupported-answer rate {show(summary['unsupported_answer_rate'])}"
        f"   n={summary['n_questions']}",
        "",
        f"  latency median / p95    {show(summary['latency_median_s'])} s / "
        f"{show(summary['latency_p95_s'])} s",
        "",
        f"  failures: {summary['n_failures']}",
    ]
    for category, count in summary["error_categories"].items():
        if count:
            lines.append(f"    {category:<18} {count}")
    return "\n".join(lines)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", required=True, help="Path to the PDF.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the hashing embedder and scripted backend: no weights, no GPU.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use the smaller Apache-2.0 generator preset.",
    )
    parser.add_argument("--strategy", choices=["structure", "fixed"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--json", action="store_true", help="Emit a JSON record.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer a question about a PDF.")
    _add_common(ask)
    ask.add_argument("--question", required=True)

    inspect = sub.add_parser("inspect", help="Show how a PDF was parsed and chunked.")
    _add_common(inspect)
    inspect.add_argument(
        "--show-chunks", type=int, default=0, help="Print the first N chunks."
    )

    inspect_page_cmd = sub.add_parser(
        "inspect-page", help="Render one PDF page to PNG and print its text."
    )
    inspect_page_cmd.add_argument("--pdf", required=True)
    inspect_page_cmd.add_argument("--page", required=True, type=int)
    inspect_page_cmd.add_argument("--password")
    inspect_page_cmd.add_argument("--out")

    validate_cmd = sub.add_parser(
        "validate-benchmark",
        help="Check benchmark/questions.json against BENCHMARK_SPEC.md section 5.1.",
    )
    validate_cmd.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    validate_cmd.add_argument("--documents-dir", default=str(DEFAULT_DOCUMENTS_DIR))
    validate_cmd.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))

    bench = sub.add_parser(
        "run-benchmark",
        help="Answer every benchmark question and score it (Phase 3).",
    )
    bench.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    bench.add_argument("--pdf-dir", default=str(DEFAULT_DOCUMENTS_DIR))
    bench.add_argument("--out", default="results/baseline")
    bench.add_argument(
        "--offline",
        action="store_true",
        help="Use the hashing embedder and scripted backend: no weights, no GPU.",
    )
    bench.add_argument("--low-memory", action="store_true")
    bench.add_argument("--strategy", choices=["structure", "fixed"])
    bench.add_argument("--chunk-size", type=int)
    bench.add_argument("--chunk-overlap", type=int)
    bench.add_argument("--top-k", type=int)
    bench.add_argument(
        "--split", choices=["dev", "eval"], help="Score one split only."
    )
    bench.add_argument("--limit", type=int, help="Stop after N questions.")
    bench.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Also run the optional LLM judge (DD-027: it is the generator "
            "grading itself, so its scores are reported beside the "
            "deterministic ones and never instead of them)."
        ),
    )
    bench.add_argument(
        "--skip-validation",
        action="store_true",
        help="Run against a dataset that fails BENCHMARK_SPEC.md 5.1. Off by default.",
    )

    args = parser.parse_args(argv)

    if args.command == "run-benchmark":
        from .evaluation.benchmark import run_benchmark_cli  # noqa: PLC0415
        from .evaluation.judge import LLMJudge  # noqa: PLC0415

        config = _build_config(args)
        judge = LLMJudge(config.generation) if args.judge else None
        try:
            run = run_benchmark_cli(
                questions_path=args.questions,
                documents_dir=args.pdf_dir,
                out_dir=args.out,
                config=config,
                split=args.split,
                limit=args.limit,
                judge=judge,
                skip_validation=args.skip_validation,
                progress=_report,
            )
        except RAGError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print()
        print(_format_summary(run.summary()))
        print()
        print(
            f"Wrote {args.out}/"
            + ", ".join(
                ["config.json", "results.json", "per_question.json", "summary.csv"]
            )
        )
        return 0

    if args.command == "inspect-page":
        try:
            out_path = args.out or (
                os.path.splitext(os.path.basename(args.pdf))[0] + f"_p{args.page}.png"
            )
            png, text = inspect_page(
                args.pdf, args.page, config=IngestionConfig(password=args.password)
            )
            with open(out_path, "wb") as handle:
                handle.write(png)
            print(f"Rendered page {args.page} of {args.pdf} to {out_path}")
            print()
            print(text)
            return 0
        except RAGError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if args.command == "validate-benchmark":
        result = validate_benchmark(args.questions, args.documents_dir, args.manifest)
        print(result.report())
        return 0 if result.ok else 1

    try:
        pipeline = DenseRAGPipeline(_build_config(args))
        pipeline.index(args.pdf)

        if args.command == "inspect":
            if args.json:
                print(json.dumps(pipeline.index_stats, indent=2, default=str))
            else:
                for key, value in pipeline.index_stats.items():
                    print(f"{key}: {value}")
            for chunk in pipeline.chunks[: args.show_chunks]:
                print(
                    f"\n--- {chunk.chunk_id} | {chunk.citation_label()} | "
                    f"section={chunk.section!r}"
                )
                print(chunk.text[:400])
            return 0

        result = pipeline.ask(args.question)
        if args.json:
            print(json.dumps(result.to_record(), indent=2, default=str))
        else:
            print(result.format())
        return 0

    except RAGError as exc:
        # These carry a remedy; print it plainly rather than as a traceback.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
