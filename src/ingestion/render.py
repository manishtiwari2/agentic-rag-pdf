"""Rendering a single page for human verification (EXPERIMENT_PLAN.md section 2).

Verifying an `evidence_pages` entry means opening the PDF and looking, for
every question. This turns one 1-based page into a PNG so
`src/benchmark/inspect.py` can hand a reviewer something to look at from the
terminal, without a second PDF library: pdfplumber (DD-033's default parser)
already pulls in pypdfium2 and Pillow, and can rasterize a page through them.

Kept in `src/ingestion/` rather than `src/benchmark/` because
`tests/test_architecture.py` confines every `import pdfplumber` to this
layer -- rendering is PDF-library work like parsing is, even though it is
used only by the benchmark tooling.
"""

from __future__ import annotations

import io
import os

from ..errors import PDFReadError


def render_page_images(
    data: bytes,
    page_numbers: list[int],
    *,
    resolution: int = 300,
    password: str | None = None,
) -> dict[int, object]:
    """Render 1-based pages of an in-memory PDF to Pillow images, for OCR (DD-067).

    Opens the file once for all pages. Uses the same pdfplumber -> pypdfium2
    path as ``render_page_png``, so OCR adds no PDF library.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PDFReadError(
            "pdfplumber is required to render a page for OCR but is not "
            "installed. Install it with `pip install pdfplumber`."
        ) from exc

    with pdfplumber.open(io.BytesIO(data), password=password or "") as pdf:
        return {
            number: pdf.pages[number - 1].to_image(resolution=resolution).original
            for number in page_numbers
        }


def render_page_png(
    path: str | os.PathLike[str],
    page_number: int,
    *,
    resolution: int = 150,
    password: str | None = None,
) -> bytes:
    """Render 1-based `page_number` of the PDF at `path` to PNG bytes."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PDFReadError(
            "pdfplumber is required to render a page but is not installed. "
            "Install it with `pip install pdfplumber`."
        ) from exc

    source_name = os.path.basename(os.fspath(path))
    try:
        pdf = pdfplumber.open(os.fspath(path), password=password or "")
    except Exception as exc:
        raise PDFReadError(f"{source_name} could not be opened: {exc}.") from exc

    try:
        page_count = len(pdf.pages)
        if page_number < 1 or page_number > page_count:
            raise PDFReadError(
                f"Page {page_number} is out of range for {source_name} "
                f"(pages 1..{page_count})."
            )
        image = pdf.pages[page_number - 1].to_image(resolution=resolution)
        buffer = io.BytesIO()
        image.original.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        pdf.close()
