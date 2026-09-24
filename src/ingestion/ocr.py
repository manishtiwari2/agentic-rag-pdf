"""OCR for pages that have no text layer (DD-067).

A scanned page is an image. The parser renders it through the path pdfplumber
already provides (pypdfium2 and Pillow, DD-033 -- no PyMuPDF, which is AGPL)
and hands the image to an ``OcrEngine``. ``TesseractOcr`` wraps pytesseract,
which calls the Tesseract binary; both are Apache-2.0.

The engine is a protocol so tests can pass a fake one and run without
Tesseract installed. The real engine checks for pytesseract and the binary only
when a page actually needs OCR, so a text PDF never touches either.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..config import IngestionConfig
from ..errors import OCRUnavailableError

INSTALL_REMEDY = (
    "Install both: the Tesseract binary (Colab/Debian/Ubuntu: "
    "`apt-get install -y tesseract-ocr`; macOS: `brew install tesseract`; "
    "Windows: the UB-Mannheim installer) and the Python wrapper "
    "(`pip install pytesseract`). Or pass --no-ocr / IngestionConfig(ocr=False) "
    "to skip OCR, in which case a scanned PDF is refused."
)


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def image_to_text(self, image: Any) -> str: ...


class TesseractOcr:
    """Tesseract through pytesseract, loaded on first use."""

    name = "tesseract"

    def __init__(self, languages: str = "eng") -> None:
        self.languages = languages
        self._pytesseract: Any = None

    def _load(self) -> Any:
        if self._pytesseract is not None:
            return self._pytesseract
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRUnavailableError(
                "This PDF has pages with no text layer, which need OCR, but the "
                f"pytesseract package is not installed. {INSTALL_REMEDY}"
            ) from exc
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # TesseractNotFoundError, or a broken install
            raise OCRUnavailableError(
                "This PDF has pages with no text layer, which need OCR, but the "
                f"Tesseract binary was not found ({exc}). {INSTALL_REMEDY}"
            ) from exc
        self._pytesseract = pytesseract
        return pytesseract

    def image_to_text(self, image: Any) -> str:
        pytesseract = self._load()
        try:
            return pytesseract.image_to_string(image, lang=self.languages)
        except pytesseract.TesseractError as exc:
            raise OCRUnavailableError(
                f"Tesseract failed on a page ({exc}). If the message names a "
                f"language, install its data (for example `apt-get install -y "
                f"tesseract-ocr-deu`) or change IngestionConfig.ocr_languages "
                f"(currently {self.languages!r})."
            ) from exc


def build_ocr_engine(config: IngestionConfig) -> OcrEngine:
    return TesseractOcr(config.ocr_languages)
