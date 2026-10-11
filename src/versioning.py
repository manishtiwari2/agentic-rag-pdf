"""Which version of a document this is: a date, and where the date came from.

DD-074 section 1. The first source that yields a date wins, in this order:

1. **user** -- a date (and optional label) supplied at index time;
2. **pdf_metadata** -- the PDF's ``CreationDate`` (not ``ModDate``: a re-save
   changes ModDate without changing the content);
3. **filename** -- a year, optionally followed by a month, standing as its own
   token in the file name;
4. **unknown**.

``source`` is always recorded, so a date read from an export timestamp or a
file name is never presented as if a person had stated it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

USER = "user"
PDF_METADATA = "pdf_metadata"
FILENAME = "filename"
UNKNOWN = "unknown"

#: PDF date syntax: D:YYYYMMDDHHmmSS+HH'mm', every field after the year optional.
_PDF_DATE = re.compile(r"^(?:D:)?(\d{4})(\d{2})?(\d{2})?")
#: An ISO-like date a user types: 2024, 2024-03 or 2024-03-15 (/ or . also accepted).
_USER_DATE = re.compile(r"^\s*(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?\s*$")
#: A year (optionally -month) bounded by non-digits, so v20240.pdf yields nothing.
_FILENAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:[-_.]?(0[1-9]|1[0-2]))?(?!\d)")


@dataclass(frozen=True)
class DocumentVersion:
    """A document's date at the precision it was stated, and its source."""

    date: str | None = None
    precision: str | None = None  # "day" | "month" | "year"
    label: str | None = None
    source: str = UNKNOWN
    raw: str | None = None

    @property
    def known(self) -> bool:
        return self.date is not None

    def key(self, precision: str) -> str | None:
        """The date cut to ``precision``, for comparing two versions fairly."""
        if self.date is None:
            return None
        return self.date[: {"year": 4, "month": 7, "day": 10}[precision]]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


_PRECISION_ORDER = {"year": 0, "month": 1, "day": 2}


def _iso(year: str, month: str | None, day: str | None) -> tuple[str, str] | None:
    y = int(year)
    if not 1900 <= y <= 2100:
        return None
    if month is None:
        return f"{y:04d}", "year"
    m = int(month)
    if not 1 <= m <= 12:
        return None
    if day is None:
        return f"{y:04d}-{m:02d}", "month"
    d = int(day)
    if not 1 <= d <= 31:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}", "day"


def parse_user_date(value: str) -> tuple[str, str] | None:
    """``"2024-03-01"`` -> ``("2024-03-01", "day")``; None if not a date."""
    match = _USER_DATE.match(value or "")
    return _iso(*match.groups()) if match else None


def parse_pdf_date(value: Any) -> tuple[str, str] | None:
    """``"D:20240215093000+05'30'"`` -> ``("2024-02-15", "day")``."""
    match = _PDF_DATE.match(str(value or "").strip())
    return _iso(*match.groups()) if match else None


def parse_filename_date(name: str) -> tuple[str, str] | None:
    """``"report_2024-06.pdf"`` -> ``("2024-06", "month")``."""
    match = _FILENAME_DATE.search(name or "")
    return _iso(match.group(1), match.group(2), None) if match else None


def resolve_version(
    source_name: str,
    pdf_metadata: Mapping[str, Any] | None = None,
    user_value: str | None = None,
) -> DocumentVersion:
    """Apply DD-074's order: user, then PDF CreationDate, then file name.

    ``user_value`` is a date, a label, or ``"<date> <label>"``. A label alone
    is kept but orders nothing, so the next source still supplies the date.
    """
    label = None
    if user_value:
        text = user_value.strip()
        head, _, tail = text.partition(" ")
        parsed = parse_user_date(head)
        if parsed:
            return DocumentVersion(parsed[0], parsed[1], tail.strip() or None, USER, text)
        label = text
    meta = pdf_metadata or {}
    creation = meta.get("pdf_creation_date") or meta.get("CreationDate") or meta.get("creationDate")
    parsed = parse_pdf_date(creation)
    if parsed:
        return DocumentVersion(parsed[0], parsed[1], label, PDF_METADATA, str(creation))
    parsed = parse_filename_date(source_name)
    if parsed:
        return DocumentVersion(parsed[0], parsed[1], label, FILENAME, source_name)
    return DocumentVersion(label=label, source=UNKNOWN)


def compare_versions(a: DocumentVersion, b: DocumentVersion) -> int | None:
    """-1 if ``a`` is older, 1 if newer, 0 if equal at their shared precision.

    None when either date is unknown: DD-074's policy refuses rather than guess.
    """
    if not (a.known and b.known):
        return None
    shared = min(a.precision, b.precision, key=lambda p: _PRECISION_ORDER[p])
    ka, kb = a.key(shared), b.key(shared)
    return (ka > kb) - (ka < kb)
