"""Command-line entry point for the dense baseline.

    python -m src.cli ask --pdf paper.pdf --question "What was measured?"
    python -m src.cli inspect --pdf paper.pdf --offline

The notebook is the deliverable (DD-002); this exists so a single PDF can be run
end to end without one, which is what the Phase 1 and Phase 2 exit criteria ask
for.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .config import ChunkingConfig, RAGConfig, RetrievalConfig
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

    args = parser.parse_args(argv)

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
