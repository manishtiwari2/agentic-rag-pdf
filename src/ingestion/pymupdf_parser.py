"""PDF extraction via PyMuPDF.

Kept as an alternative backend, not the default. DD-033: PyMuPDF is AGPL-3.0,
which would impose copyleft on downstream users of this project, so
``pdfplumber`` is the default. This backend remains available for comparison,
and because its block-level extraction is faster and reports layout directly
rather than requiring lines to be grouped.

Selecting it means accepting AGPL terms for whatever is built on top.
"""

from __future__ import annotations

from ..config import IngestionConfig
from ..errors import EncryptedPDFError, PDFReadError
from .parser_base import (
    BaseParser,
    RawBlock,
    RawPage,
    mostly_inside,
    normalize_lines,
    render_table,
    sort_blocks_into_reading_order,
)

#: PyMuPDF span flag bit for a bold face.
_BOLD_FLAG = 1 << 4


class PyMuPDFParser(BaseParser):
    """Text and layout extraction via PyMuPDF (AGPL-3.0)."""

    name = "pymupdf"

    def __init__(self, config: IngestionConfig | None = None) -> None:
        super().__init__(config)

    # -- backend hook -------------------------------------------------------

    def _extract(self, data: bytes, source_name: str) -> tuple[list[RawPage], dict]:
        pymupdf = _import_pymupdf()

        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise PDFReadError(
                f"{source_name} could not be opened as a PDF: {exc}. Check that "
                "the file is a valid, uncorrupted PDF."
            ) from exc

        try:
            self._authenticate(pdf, source_name)
            pages = [self._extract_page(pdf[i], i + 1) for i in range(pdf.page_count)]
            metadata = dict(pdf.metadata or {})
        finally:
            pdf.close()
        return pages, metadata

    def _authenticate(self, pdf, source_name: str) -> None:
        if not pdf.needs_pass:
            return
        password = self.config.password
        if password is None or not pdf.authenticate(password):
            detail = (
                "no password was supplied"
                if password is None
                else "the supplied password was rejected"
            )
            raise EncryptedPDFError(
                f"{source_name} is password protected and {detail}. Pass the "
                "password via IngestionConfig(password=...), or decrypt the file "
                "first (for example with `qpdf --decrypt in.pdf out.pdf`)."
            )

    # -- per page -----------------------------------------------------------

    def _extract_page(self, page, page_number: int) -> RawPage:
        raw = RawPage(page_number=page_number, height=float(page.rect.height))
        raw.image_count = len(page.get_images(full=True))

        table_bboxes: list[tuple[float, float, float, float]] = []
        order = 0
        for table_text, bbox in self._extract_tables(page):
            table_bboxes.append(bbox)
            raw.blocks.append(
                RawBlock(
                    lines=table_text.split("\n"),
                    bbox=bbox,
                    is_table=True,
                    order=order,
                )
            )
            order += 1

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 == text, 1 == image
                continue
            bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))
            if any(mostly_inside(bbox, tb) for tb in table_bboxes):
                continue

            lines: list[str] = []
            sizes: list[float] = []
            bold_chars = 0
            total_chars = 0
            for line in block.get("lines", []):
                parts: list[str] = []
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    parts.append(text)
                    sizes.append(float(span.get("size", 0.0)))
                    stripped = len(text.strip())
                    total_chars += stripped
                    if int(span.get("flags", 0)) & _BOLD_FLAG:
                        bold_chars += stripped
                if parts:
                    lines.append("".join(parts))

            cleaned = normalize_lines(lines, self.config)
            if not cleaned:
                continue

            raw.blocks.append(
                RawBlock(
                    lines=cleaned,
                    bbox=bbox,  # type: ignore[arg-type]
                    font_size=max(sizes) if sizes else None,
                    is_bold=bool(total_chars) and bold_chars / total_chars > 0.6,
                    order=order,
                )
            )
            order += 1

        sort_blocks_into_reading_order(raw.blocks)
        return raw

    def _extract_tables(self, page) -> list[tuple[str, tuple[float, float, float, float]]]:
        finder = getattr(page, "find_tables", None)
        if finder is None:
            return []
        try:
            found = finder()
        except Exception:
            return []

        results = []
        for table in getattr(found, "tables", []):
            try:
                rows = table.extract()
                bbox = tuple(float(v) for v in table.bbox)
            except Exception:
                continue
            rendered = render_table(rows)
            if rendered:
                results.append((rendered, bbox))  # type: ignore[arg-type]
        return results


def _import_pymupdf():
    try:
        import pymupdf  # noqa: PLC0415

        return pymupdf
    except ImportError:  # pragma: no cover - environment dependent
        try:
            import fitz as pymupdf  # noqa: PLC0415

            return pymupdf
        except ImportError as exc:
            raise PDFReadError(
                "PyMuPDF is not installed. Install it with `pip install pymupdf` "
                "(note: AGPL-3.0, see DD-033), or use the default pdfplumber "
                "parser by setting ingestion.parser='pdfplumber'."
            ) from exc
