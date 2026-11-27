"""Render a bounded PDF page with rectangles over the stored cited passage."""

import io
import logging
import math
import re
import unicodedata
from difflib import SequenceMatcher

from ..errors import PDFReadError

_WORD = re.compile(r"[^\W_]+")
_FIX_DASH = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def _tokens(text):
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _matched_words(passage, words):
    expected = _tokens(_FIX_DASH.sub("", passage))[:1800]
    if not expected:
        return []
    observed, word_ids = [], []
    for index, word in enumerate(words[:6000]):
        tokens = _tokens(word["text"])
        observed.extend(tokens)
        word_ids.extend([index] * len(tokens))
    matcher = SequenceMatcher(None, expected, observed)
    selected = set()
    for block in matcher.get_matching_blocks():
        if block.size >= min(4, len(expected)):
            selected.update(word_ids[block.b:block.b + block.size])
    return [words[index] for index in sorted(selected)]


def render_highlight(data: bytes, number: int, passage: str, *, ocr: bool = False) -> tuple[bytes, bool]:
    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(data), pages=[number]) as pdf:
            page = pdf.pages[0]
            x0, top, x1, bottom = page.mediabox
            extent = max(x1 - x0, bottom - top)
            if not math.isfinite(extent) or extent <= 0:
                raise PDFReadError("This PDF page has invalid dimensions.")
            image = page.to_image(resolution=min(144, 72 * 1800 / extent), antialias=True)
            words = page.extract_words()
            matches = _matched_words(passage, words)
            for word in matches:
                image.draw_rect(word, fill=(255, 189, 35, 95), stroke=None)
            if ocr and not words:
                matches = _ocr_matches(image.original, passage)
                for word in matches:
                    image.draw.rectangle(word["bbox"], fill=(255, 189, 35, 95))
            buffer = io.BytesIO()
            image.annotated.save(buffer, format="PNG")
            return buffer.getvalue(), bool(matches)
    except Exception as exc:
        raise PDFReadError("This cited page could not be rendered.") from exc


def _ocr_matches(image, passage):
    try:
        import pytesseract

        result = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT, timeout=8)
    except (ImportError, OSError, RuntimeError):
        logging.getLogger(__name__).warning("Could not locate OCR text for citation")
        return []
    words = []
    for text, x, y, width, height in zip(
        result["text"], result["left"], result["top"], result["width"], result["height"],
    ):
        if text.strip():
            words.append({"text": text, "bbox": (x, y, x + width, y + height)})
    return _matched_words(passage, words)
