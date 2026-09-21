"""Render one PDF page and print its extracted text (EXPERIMENT_PLAN.md section 2).

    python -m src.benchmark.inspect --pdf benchmark/documents/document_01.pdf --page 4

This is the tool that makes "verify every evidence page" a thing a human can
actually do: it turns "open the PDF, scroll to page 4, read it" into one
command that prints the same text the pipeline would index next to a PNG of
the page, so a reviewer can confirm a `questions.json` entry without leaving
the terminal.
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import IngestionConfig
from ..errors import RAGError
from ..ingestion.parser import build_parser
from ..ingestion.render import render_page_png


def inspect_page(
    pdf_path: str | os.PathLike[str],
    page_number: int,
    *,
    config: IngestionConfig | None = None,
) -> tuple[bytes, str]:
    """Return `(png_bytes, extracted_text)` for one 1-based page of `pdf_path`."""
    config = config or IngestionConfig()
    document = build_parser(config).parse(pdf_path)
    if page_number < 1 or page_number > document.page_count:
        raise RAGError(
            f"Page {page_number} is out of range for {document.source_name} "
            f"(pages 1..{document.page_count})."
        )
    text = document.page(page_number).text
    png = render_page_png(pdf_path, page_number, password=config.password)
    return png, text


def _default_out_path(pdf_path: str, page_number: int) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return f"{stem}_p{page_number}.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.benchmark.inspect", description=__doc__)
    parser.add_argument("--pdf", required=True, help="Path to the PDF.")
    parser.add_argument("--page", required=True, type=int, help="1-based page number.")
    parser.add_argument("--password", help="Password, if the PDF is encrypted.")
    parser.add_argument(
        "--out", help="PNG output path. Defaults to <pdf-stem>_p<page>.png."
    )
    args = parser.parse_args(argv)

    out_path = args.out or _default_out_path(args.pdf, args.page)
    try:
        png, text = inspect_page(
            args.pdf,
            args.page,
            config=IngestionConfig(password=args.password),
        )
    except RAGError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    with open(out_path, "wb") as handle:
        handle.write(png)

    print(f"Rendered page {args.page} of {args.pdf} to {out_path}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
