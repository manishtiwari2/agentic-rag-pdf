"""Text normalization for PDF extraction artifacts.

PDF text extraction introduces damage that looks harmless and quietly degrades
retrieval:

* ligature glyphs, so ``efficient`` is extracted as ``ef<U+FB01>cient`` and
  never matches the query token ``efficient``;
* soft hyphens and zero-width characters embedded mid-word;
* words split by a hyphen at a line break, so ``retrie-\\nval`` is two tokens;
* a running header or footer repeated on every page, which the chunker would
  otherwise sprinkle through every chunk as high-frequency noise.

Each transformation is individually switchable through ``IngestionConfig`` so
its contribution can be ablated later rather than assumed.

Nothing here changes page boundaries. Header stripping removes lines *within* a
page; it never removes a page, because that would renumber everything after it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ..config import IngestionConfig

# Ligatures. NFKC would handle most of these, but it also rewrites a lot more
# than we want (superscripts, fractions, full-width forms), so the substitution
# is explicit.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
    "Ĳ": "IJ",
    "ĳ": "ij",
    "Œ": "OE",
    "œ": "oe",
    "Æ": "AE",
    "æ": "ae",
}

#: Characters that carry no meaning but break token matching.
INVISIBLE = dict.fromkeys(
    [
        "­",  # soft hyphen
        "​",  # zero-width space
        "‌",
        "‍",
        "﻿",  # BOM
    ],
    "",
)

#: Space-like characters mapped to a plain space.
SPACES = dict.fromkeys(
    [" ", " ", " ", " ", " ", " ", "\t"], " "
)

QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
}

_LIGATURE_TABLE = str.maketrans({**LIGATURES, **INVISIBLE, **SPACES})
_INVISIBLE_ONLY_TABLE = str.maketrans({**INVISIBLE, **SPACES})
_QUOTE_TABLE = str.maketrans(QUOTES)

# A hyphen at end of line, where the next line starts a word.
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
_MULTISPACE = re.compile(r"[ ]{2,}")
_BLANKLINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


def normalize_text(text: str, config: IngestionConfig) -> str:
    """Apply character-level normalization, preserving line structure."""
    if not text:
        return ""
    # Normal form C first: decomposed accents would otherwise survive as
    # separate code points and break exact matching.
    out = unicodedata.normalize("NFC", text)
    out = out.translate(_LIGATURE_TABLE if config.fix_ligatures else _INVISIBLE_ONLY_TABLE)
    if config.normalize_quotes:
        out = out.translate(_QUOTE_TABLE)
    # Strip remaining control characters except newline.
    out = "".join(ch for ch in out if ch == "\n" or unicodedata.category(ch)[0] != "C")
    out = _MULTISPACE.sub(" ", out)
    return "\n".join(line.rstrip() for line in out.split("\n"))


def join_hyphenated_linebreaks(text: str) -> str:
    """Rejoin words split by a hyphen at a line break.

    The hyphen is dropped when the continuation is lowercase (``retrie-\\nval``
    is one word) and kept when it is uppercase or a digit (``Mixed-\\nCase`` and
    ``UTF-\\n8`` are genuine hyphenated forms, and deleting the hyphen there
    would invent a word that does not appear in the document).
    """

    def _join(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if right.islower():
            return left + right
        return f"{left}-{right}"

    previous = None
    # Repeat: overlapping matches (consecutive hyphenated lines) are not all
    # replaced in a single pass because each match consumes its boundary chars.
    while previous != text:
        previous = text
        text = _HYPHEN_BREAK.sub(_join, text)
    return text


def collapse_blank_lines(text: str) -> str:
    return _BLANKLINES.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Running headers and footers
# ---------------------------------------------------------------------------

#: Lines longer than this are body text, never a running header.
_MAX_HEADER_LINE_CHARS = 120


def line_key(line: str) -> str:
    """Comparison key for header detection.

    Digits are masked so that ``Page 3`` and ``Page 4`` compare equal: the page
    number is exactly the part that differs between pages, and matching on the
    literal text would therefore never detect a numbered footer.
    """
    collapsed = " ".join(line.split()).lower()
    return _DIGITS.sub("#", collapsed)


def count_running_keys(
    candidates_per_page: list[list[str]], page_count: int, config: IngestionConfig
) -> set[str]:
    """Return the keys that repeat across pages among the supplied candidates.

    Which lines are candidates is decided by the caller, because it depends on
    information this module does not have: the parser knows each line's position
    on the page, which is a much better signal than its position in the line
    order. A phrase repeated in the middle of many pages is a stylistic tic of
    the document, not page furniture, and removing it would delete real content.
    """
    if page_count < max(2, config.header_footer_min_pages):
        return set()

    counts: Counter[str] = Counter()
    for lines in candidates_per_page:
        seen_on_page: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if not stripped or len(stripped) > _MAX_HEADER_LINE_CHARS:
                continue
            seen_on_page.add(line_key(stripped))
        counts.update(seen_on_page)

    threshold = max(
        config.header_footer_min_pages,
        int(round(config.header_footer_min_page_fraction * page_count)),
    )
    return {key for key, count in counts.items() if count >= threshold}


def edge_lines(lines: list[str], config: IngestionConfig) -> list[str]:
    """Positional fallback: the first and last few lines of a page.

    Used only when the parser reports no geometry.
    """
    n = max(1, config.header_footer_scan_lines)
    if len(lines) <= n:
        return list(lines)
    return lines[:n] + lines[-n:]


def keep_mask(
    lines: list[str],
    running: set[str],
    config: IngestionConfig,
    is_candidate: list[bool] | None = None,
) -> list[bool]:
    """Per-line keep/drop decision for one page.

    Returned as a mask rather than as filtered text so that a caller holding
    extra per-line state -- the parser, which needs to know which block each
    line came from -- can apply the same decision without reimplementing it.

    ``is_candidate`` marks the lines eligible for removal. Without it, the
    positional fallback applies: only the first and last few lines can be
    furniture. A line is dropped only if it is both a candidate and a key that
    was found to repeat, so body text that happens to match a header is left
    alone wherever it appears mid-page.
    """
    if not running:
        return [True] * len(lines)
    if is_candidate is None:
        n = max(1, config.header_footer_scan_lines)
        last = len(lines) - 1
        is_candidate = [i < n or i > last - n for i in range(len(lines))]

    mask: list[bool] = []
    for line, candidate in zip(lines, is_candidate):
        stripped = line.strip()
        drop = candidate and bool(stripped) and line_key(stripped) in running
        mask.append(not drop)
    return mask

