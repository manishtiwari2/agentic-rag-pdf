"""PDF extraction via pdfplumber (MIT).

The default backend. DD-033 records why: PyMuPDF is AGPL-3.0 and would impose
copyleft on every downstream user of this project, which is the same class of
problem DD-024 refused to accept from a research-licensed model. pdfplumber and
its dependency chain -- pdfminer.six, pypdfium2, Pillow -- are MIT, BSD-3-Clause
and Apache-2.0.

pdfplumber reports words and lines rather than blocks, so this backend builds
blocks itself: lines are grouped while their vertical spacing and font size stay
consistent, and split where either changes. That grouping is what lets the
chunker tell a heading from a paragraph, so it is deliberate rather than a
detail of the library.
"""

from __future__ import annotations

import io
import statistics

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

#: A vertical gap larger than this many times the line height starts a new
#: block. Paragraph spacing in a typical document is 1.2-1.6x leading.
_BLOCK_GAP_RATIO = 1.6
#: A font-size change of more than this fraction also starts a new block, so a
#: heading never lands in the same block as the paragraph beneath it.
_FONT_CHANGE_RATIO = 0.12
#: Substrings that mark a bold face in a PostScript font name.
_BOLD_MARKERS = ("bold", "black", "heavy", "semib", "-bd", "bd")


class PdfPlumberParser(BaseParser):
    """Text and layout extraction via pdfplumber."""

    name = "pdfplumber"

    def __init__(self, config: IngestionConfig | None = None) -> None:
        super().__init__(config)

    # -- backend hook -------------------------------------------------------

    def _extract(self, data: bytes, source_name: str) -> tuple[list[RawPage], dict]:
        pdfplumber = _import_pdfplumber()

        try:
            pdf = pdfplumber.open(
                io.BytesIO(data), password=self.config.password or ""
            )
        except Exception as exc:
            raise _open_error(exc, source_name, self.config.password) from exc

        try:
            pages = [
                self._extract_page(page, number)
                for number, page in enumerate(pdf.pages, start=1)
            ]
            metadata = dict(pdf.metadata or {})
        except Exception as exc:
            if isinstance(exc, (EncryptedPDFError, PDFReadError)):
                raise
            raise PDFReadError(
                f"{source_name} could not be read: {exc}. The file may be "
                "corrupt or use an unsupported PDF feature."
            ) from exc
        finally:
            pdf.close()

        # pdfplumber lowercases nothing in metadata keys; normalize the one we use.
        if "Title" in metadata and "title" not in metadata:
            metadata["title"] = metadata["Title"]
        return pages, metadata

    # -- per page -----------------------------------------------------------

    def _extract_page(self, page, page_number: int) -> RawPage:
        raw = RawPage(page_number=page_number, height=float(page.height or 0.0))
        raw.image_count = len(page.images or [])

        table_bboxes: list[tuple[float, float, float, float]] = []
        order = 0
        for text, bbox in self._extract_tables(page):
            table_bboxes.append(bbox)
            raw.blocks.append(
                RawBlock(
                    lines=text.split("\n"),
                    bbox=bbox,
                    is_table=True,
                    order=order,
                )
            )
            order += 1

        lines = self._text_lines(page, table_bboxes)

        for group in _group_lines_into_blocks(lines):
            texts = normalize_lines([item["text"] for item in group], self.config)
            if not texts:
                continue
            sizes = [item["size"] for item in group if item["size"]]
            bold_share = sum(1 for item in group if item["bold"]) / len(group)
            raw.blocks.append(
                RawBlock(
                    lines=texts,
                    bbox=(
                        min(i["x0"] for i in group),
                        min(i["top"] for i in group),
                        max(i["x1"] for i in group),
                        max(i["bottom"] for i in group),
                    ),
                    font_size=max(sizes) if sizes else None,
                    is_bold=bold_share > 0.6,
                    order=order,
                )
            )
            order += 1

        sort_blocks_into_reading_order(raw.blocks)
        return raw

    def _text_lines(
        self, page, table_bboxes: list[tuple[float, float, float, float]]
    ) -> list[dict]:
        """Lines of prose, with the geometry and font facts blocks need."""
        try:
            extracted = page.extract_text_lines(
                layout=False, strip=True, return_chars=True
            )
        except Exception:
            return []

        lines: list[dict] = []
        for line in extracted:
            text = line.get("text", "")
            if not text.strip():
                continue
            bbox = (
                float(line["x0"]),
                float(line["top"]),
                float(line["x1"]),
                float(line["bottom"]),
            )
            # A table's cells are also ordinary characters. Emitting both would
            # duplicate every table in the index and let one sentence be
            # retrieved twice as if it were two independent pieces of evidence.
            if any(mostly_inside(bbox, tb) for tb in table_bboxes):
                continue
            chars = line.get("chars") or []
            sizes = [float(c["size"]) for c in chars if c.get("size")]
            fonts = [str(c.get("fontname", "")).lower() for c in chars]
            bold_chars = sum(
                1 for f in fonts if any(marker in f for marker in _BOLD_MARKERS)
            )
            lines.append(
                {
                    "text": text,
                    "x0": bbox[0],
                    "top": bbox[1],
                    "x1": bbox[2],
                    "bottom": bbox[3],
                    "size": statistics.median(sizes) if sizes else None,
                    "bold": bool(fonts) and bold_chars / len(fonts) > 0.6,
                }
            )
        lines.sort(key=lambda item: (round(item["top"], 1), item["x0"]))
        return lines

    def _extract_tables(self, page) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Rendered tables and their bounding boxes.

        Best-effort: table detection is imperfect in every library. A failure
        here degrades a table to ordinary prose rather than failing ingestion.
        """
        try:
            found = page.find_tables()
        except Exception:
            return []

        results: list[tuple[str, tuple[float, float, float, float]]] = []
        for table in found or []:
            try:
                rows = table.extract()
                bbox = tuple(float(v) for v in table.bbox)
            except Exception:
                continue
            rendered = render_table(rows)
            if rendered:
                results.append((rendered, bbox))  # type: ignore[arg-type]
        return results


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------


def _group_lines_into_blocks(lines: list[dict]) -> list[list[dict]]:
    """Group consecutive lines into blocks.

    A new block starts where the vertical gap widens past normal leading, or
    where the font size changes materially. The second rule is what keeps a
    heading out of the paragraph below it, which the chunker depends on.
    """
    if not lines:
        return []

    heights = [line["bottom"] - line["top"] for line in lines]
    typical = statistics.median(heights) if heights else 12.0

    groups: list[list[dict]] = [[lines[0]]]
    for previous, current in zip(lines, lines[1:]):
        gap = current["top"] - previous["bottom"]
        size_changed = False
        if previous["size"] and current["size"]:
            larger = max(previous["size"], current["size"])
            size_changed = (
                abs(previous["size"] - current["size"]) / larger > _FONT_CHANGE_RATIO
            )
        if gap > typical * _BLOCK_GAP_RATIO or size_changed:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _error_chain(exc: BaseException, limit: int = 5) -> list[BaseException]:
    """The exception and the causes behind it.

    pdfplumber wraps pdfminer's errors in a ``PdfminerException`` carrying an
    empty message, so the only way to tell "wrong password" from "corrupt file"
    is to look at what it wrapped.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < limit:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _open_error(exc: Exception, source_name: str, password: str | None) -> Exception:
    """Map a pdfminer failure onto this project's error taxonomy."""
    chain = _error_chain(exc)
    name = " ".join(type(e).__name__ for e in chain).lower()
    message = " ".join(str(e) for e in chain).lower()
    if "password" in name or "password" in message or "encrypt" in name or "encrypt" in message:
        detail = (
            "no password was supplied"
            if not password
            else "the supplied password was rejected"
        )
        return EncryptedPDFError(
            f"{source_name} is password protected and {detail}. Pass the "
            "password via IngestionConfig(password=...), or decrypt the file "
            "first (for example with `qpdf --decrypt in.pdf out.pdf`)."
        )
    return PDFReadError(
        f"{source_name} could not be opened as a PDF: {exc}. Check that the "
        "file is a valid, uncorrupted PDF."
    )


def _import_pdfplumber():
    try:
        import pdfplumber  # noqa: PLC0415

        return pdfplumber
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PDFReadError(
            "pdfplumber is required to parse PDFs but is not installed. "
            "Install it with `pip install pdfplumber`."
        ) from exc
