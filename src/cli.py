"""Command-line entry point for both baselines and the agentic system.

    python -m src.cli ask --pdf paper.pdf --question "What was measured?"
    python -m src.cli ask --pdf paper.pdf --question "..." --system hybrid
    python -m src.cli inspect --pdf paper.pdf --offline
    python -m src.cli run-benchmark --offline                      # Baseline A
    python -m src.cli run-benchmark --system hybrid --offline      # Baseline B
    python -m src.cli run-benchmark --system agentic --offline     # agentic
    python -m src.cli run-benchmark --system agentic --no-verification --offline         --out results/ablations/verification_off                   # an ablation
    python -m src.cli compare-runs --baseline results/baseline --system results/hybrid
    python -m src.cli final-table                                  # results/final/

``--system`` selects the system through configuration (``retrieval.strategy``)
rather than through a second code path, so every other flag means the same
thing for both.

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
from .config import AgentConfig, ChunkingConfig, IngestionConfig, RAGConfig, RetrievalConfig
from .errors import ConfigurationError, RAGError
from .pipeline import build_pipeline

#: Where each system's results go by default (EVALUATION_PROTOCOL.md 27).
DEFAULT_OUT = {
    "dense": "results/baseline",
    "hybrid": "results/hybrid",
    "agentic": "results/agentic",
}

#: The agentic ablation switches of EVALUATION_PROTOCOL.md section 24, each one
#: configuration field (DD-050).
AGENT_SWITCHES: dict[str, str] = {
    "no_planner": "planner_enabled",
    "no_evidence_controller": "evidence_controller_enabled",
    "no_refinement": "refinement_enabled",
    "no_verification": "verification_enabled",
}


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
    system = getattr(args, "system", "dense") or "dense"
    retrieval = replace(retrieval, strategy=system)

    reranking = config.reranking
    if getattr(args, "no_rerank", False):
        if system not in ("hybrid", "agentic"):
            raise ConfigurationError(
                "--no-rerank applies to --system hybrid or agentic; the dense "
                "baseline has no reranker to switch off."
            )
        reranking = replace(reranking, enabled=False)
    if getattr(args, "no_hybrid", False):
        if system not in ("hybrid", "agentic"):
            raise ConfigurationError(
                "--no-hybrid applies to --system hybrid or agentic; the dense "
                "baseline already retrieves with the dense retriever alone."
            )
        # The "Hybrid retrieval OFF" ablation: this one field, nothing else.
        retrieval = replace(retrieval, retrievers=("dense",))

    agents: AgentConfig = config.agents
    requested = [flag for flag in AGENT_SWITCHES if getattr(args, flag, False)]
    iterations = getattr(args, "max_iterations", None)
    threshold = getattr(args, "sufficiency_threshold", None)
    if (requested or iterations is not None or threshold is not None) and system != "agentic":
        flags = [f"--{f.replace('_', '-')}" for f in requested]
        if iterations is not None:
            flags.append("--max-iterations")
        if threshold is not None:
            flags.append("--sufficiency-threshold")
        raise ConfigurationError(
            f"{', '.join(flags)} apply to --system agentic; the {system} "
            "baseline has no agent components to switch off."
        )
    for flag in requested:
        agents = replace(agents, **{AGENT_SWITCHES[flag]: False})
    if iterations is not None:
        agents = replace(agents, max_retrieval_iterations=iterations)
    if threshold is not None:
        agents = replace(agents, sufficiency_threshold=threshold)

    return replace(
        config, chunking=chunking, retrieval=retrieval, reranking=reranking, agents=agents
    )


def _add_system(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system",
        choices=["dense", "hybrid", "agentic"],
        default="dense",
        help=(
            "dense = Baseline A; hybrid = Baseline B (dense + BM25, RRF, rerank); "
            "agentic = Baseline B's retrieval driven by planner, evidence "
            "controller, refinement and verifier."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Hybrid/agentic: the 'Reranker OFF' ablation (EVALUATION_PROTOCOL.md 24).",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help=(
            "Hybrid/agentic: the 'Hybrid retrieval OFF' ablation "
            "(EVALUATION_PROTOCOL.md 24): retrieval.retrievers = ('dense',)."
        ),
    )
    for flag, label in (
        ("--no-planner", "Agent planner OFF"),
        ("--no-evidence-controller", "Evidence controller OFF"),
        ("--no-refinement", "Retrieval refinement OFF"),
        ("--no-verification", "Verification OFF"),
    ):
        parser.add_argument(
            flag,
            action="store_true",
            help=f"Agentic only: the '{label}' ablation (EVALUATION_PROTOCOL.md 24).",
        )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Agentic only: MAX_RETRIEVAL_ITERATIONS (default 2; open question 3).",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        help=(
            "Agentic only: agents.sufficiency_threshold, the rule-based evidence "
            "controller's coverage threshold (default 0.5, DD-052). Tune on "
            "--split dev only."
        ),
    )


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
    ]
    if summary.get("n_reporting_model_calls"):
        # EVALUATION_PROTOCOL.md section 28's extras, for a system reporting them.
        lines += [
            f"  retrieval iterations    {show(summary['retrieval_iterations_mean'])} mean,"
            f" {summary['retrieval_iterations_max']:.0f} max",
            f"  model calls             {show(summary['model_calls_mean'])} mean,"
            f" {summary['model_calls_max']:.0f} max",
            f"  LLM parse failures      {summary['llm_parse_failures_total']}"
            f" of {summary['llm_decisions_total']} decisions",
            f"  refined / changed       {summary['n_refined']} /"
            f" {summary['n_refinement_changed_answer']}",
            f"  verification            {summary['verification_status_counts']}",
            "",
        ]
    lines += [
        f"  failures: {summary['n_failures']}",
    ]
    for category, count in summary["error_categories"].items():
        if count:
            lines.append(f"    {category:<18} {count}")
    return "\n".join(lines)


def _format_comparison(comparison: dict) -> str:
    """One line per metric: both means, the paired difference, its CI, the verdict."""

    def signed(value):
        return "n/a" if value is None else f"{value:+.3f}"

    def plain(value):
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"{comparison['system_system_name']} vs {comparison['baseline_system_name']}"
        f"    n = {comparison['n_questions']}    "
        f"paired bootstrap, {comparison['method']['resamples']} resamples",
    ]
    if comparison["fair_comparison"]["confounded"]:
        mismatched = sorted(comparison["fair_comparison"]["mismatched"])
        lines.append("  CONFOUNDED: " + ", ".join(mismatched))
    lines.append("")
    lines.append(
        f"  {'metric':<38} {'base':>6} {'sys':>6} {'diff':>7}  {'95% CI':<17} "
        f"{'p':>6} {'n':>3}  verdict"
    )
    for name, c in comparison["contrasts"].items():
        if c["ci_low"] is None:
            ci = "n/a"
        else:
            ci = f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        p = "n/a" if c["p_value"] is None else f"{c['p_value']:.3f}"
        lines.append(
            f"  {name:<38} {plain(c['baseline_mean']):>6} {plain(c['system_mean']):>6} "
            f"{signed(c['mean_difference']):>7}  {ci:<17} {p:>6} {c['n']:>3}  "
            f"{c['verdict']}"
        )
    lines.append("")
    multiple = comparison["multiple_comparisons"]
    lines.append(
        f"  {multiple['n_comparisons']} comparisons, {multiple['n_significant']} "
        "significant (EVALUATION_PROTOCOL.md 26.2: read the primary metrics first)"
    )
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
    _add_system(parser)


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
    bench.add_argument(
        "--out", help="Output directory. Default: results/baseline or results/hybrid."
    )
    _add_system(bench)
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

    compare_cmd = sub.add_parser(
        "compare-runs",
        help="Paired bootstrap comparison of two stored runs (DD-028).",
    )
    compare_cmd.add_argument("--baseline", default="results/baseline")
    compare_cmd.add_argument("--system", default="results/hybrid")
    compare_cmd.add_argument("--out", help="Default: <system>/comparison.json")
    compare_cmd.add_argument(
        "--split",
        choices=["dev", "eval"],
        help=(
            "Compare only this split's questions. A run over every question is "
            "filtered to it; a run over the other split is refused."
        ),
    )
    compare_cmd.add_argument(
        "--allow-confounded",
        action="store_true",
        help="Compare runs differing in a held-constant field (EVALUATION_PROTOCOL.md 10).",
    )

    final_cmd = sub.add_parser(
        "final-table",
        help=(
            "Regenerate results/final/ from stored results: the section 28 table, "
            "the ablation verdicts (DD-058) and the error analysis."
        ),
    )
    final_cmd.add_argument("--root", default=".", help="Repository root holding results/.")
    final_cmd.add_argument("--out", default="results/final")

    args = parser.parse_args(argv)

    if args.command == "final-table":
        from .evaluation.report import write_report  # noqa: PLC0415

        try:
            paths = write_report(args.root, args.out)
        except (OSError, RAGError, ValueError) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if hasattr(sys.stdout, "reconfigure"):
            # A Windows console's code page cannot print every character in it.
            sys.stdout.reconfigure(errors="replace")
        print(paths["markdown"].read_text(encoding="utf-8"))
        print("Wrote " + ", ".join(str(p) for p in paths.values()))
        return 0

    if args.command == "compare-runs":
        from .evaluation.benchmark import compare_runs  # noqa: PLC0415
        from .evaluation.statistics import ComparisonError  # noqa: PLC0415

        try:
            comparison = compare_runs(
                args.baseline,
                args.system,
                out_path=args.out,
                allow_confounded=args.allow_confounded,
                split=args.split,
            )
        except (ComparisonError, OSError, RAGError) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(_format_comparison(comparison))
        print()
        print(f"Wrote {comparison['written_to']}")
        return 0

    if args.command == "run-benchmark":
        from .evaluation.benchmark import run_benchmark_cli  # noqa: PLC0415
        from .evaluation.judge import LLMJudge  # noqa: PLC0415

        try:
            config = _build_config(args)
        except RAGError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        args.out = args.out or DEFAULT_OUT[args.system]
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
        pipeline = build_pipeline(_build_config(args))
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
