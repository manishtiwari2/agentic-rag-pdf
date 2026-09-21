"""The parser interface and backend selection (ARCHITECTURE.md section 5).

A parser's only job is to turn a PDF into a ``Document``. It does not chunk,
embed, retrieve or call a model, and nothing downstream imports it -- the
retrieval layer sees only the document model, which is what lets the extraction
backend be replaced.

That replaceability is not theoretical. DD-033 required exactly this swap:
PyMuPDF is AGPL-3.0, so ``pdfplumber`` (MIT) is the default and PyMuPDF remains
available behind the same protocol.

Failure is explicit. A password-protected file, an image-only scan and a blank
PDF each raise a distinct error carrying a remedy, because the alternative --
returning a document with almost no text -- produces a confident wrong answer
several layers away from the actual cause.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from ..config import IngestionConfig
from ..errors import PDFReadError
from .document import Document
from .parser_base import BaseParser, RawBlock, RawPage
from .pdfplumber_parser import PdfPlumberParser
from .pymupdf_parser import PyMuPDFParser

__all__ = [
    "PdfParser",
    "BaseParser",
    "PdfPlumberParser",
    "PyMuPDFParser",
    "RawBlock",
    "RawPage",
    "build_parser",
]


@runtime_checkable
class PdfParser(Protocol):
    """The contract the rest of the system depends on."""

    name: str

    def parse(self, path: str | os.PathLike[str]) -> Document: ...

    def parse_bytes(self, data: bytes, source_name: str) -> Document: ...


#: Backends, by the name used in ``IngestionConfig.parser``.
PARSERS: dict[str, type[BaseParser]] = {
    "pdfplumber": PdfPlumberParser,
    "pymupdf": PyMuPDFParser,
}


def build_parser(config: IngestionConfig) -> PdfParser:
    """Construct the parser named by the configuration (DD-017)."""
    try:
        parser_class = PARSERS[config.parser]
    except KeyError:
        raise PDFReadError(
            f"Unknown parser {config.parser!r}. Available: "
            f"{', '.join(sorted(PARSERS))}."
        ) from None
    return parser_class(config)
