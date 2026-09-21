"""Document manifest with checksum verification (BENCHMARK_SPEC.md section 4.1-4.2).

Where a benchmark PDF cannot be redistributed, section 4.1 requires shipping a
manifest -- source URL plus SHA-256 -- instead of the file, and downloading it
at run time. The checksum is not paperwork: ground truth is annotated against
one specific rendering of one specific file, and a silently updated source
document invalidates every evidence page in `questions.json` for it. Without a
checksum that failure is invisible; with one, it is a single comparison.

Page counts recorded here are informational only. `validate.py` never trusts
`DocumentEntry.pages` for the range check in BENCHMARK_SPEC.md section 5.1 --
it reopens the PDF through `PdfParser` instead, because a manifest can go
stale in exactly the same way a source document can.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import IngestionConfig
from ..ingestion.parser import PdfParser, build_parser
from .schema import DocumentEntry

DEFAULT_MANIFEST_PATH = Path("benchmark/documents_metadata.json")

#: Read in chunks so verifying a manifest never loads a whole PDF into memory
#: at once.
_HASH_CHUNK_SIZE = 1 << 20


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file's bytes, as hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ChecksumMismatch:
    """A manifest entry whose recorded hash no longer matches the file on disk."""

    filename: str
    expected_sha256: str
    actual_sha256: str

    def __str__(self) -> str:
        return (
            f"{self.filename}: manifest records sha256 {self.expected_sha256}, "
            f"but the file on disk hashes to {self.actual_sha256}. The source "
            "document changed since the ground truth in questions.json was "
            "annotated against it; every evidence page for this document must "
            "be re-verified before the benchmark can be trusted."
        )


class Manifest:
    """The document manifest, keyed by filename."""

    def __init__(self, entries: dict[str, DocumentEntry] | None = None) -> None:
        self.entries: dict[str, DocumentEntry] = dict(entries or {})

    def __contains__(self, filename: str) -> bool:
        return filename in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MANIFEST_PATH) -> Manifest:
        """Load the manifest. A missing file loads as empty, not an error:
        the manifest is optional infrastructure for documents that cannot be
        redistributed, and most Phase 0 corpora will have none yet.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            {
                filename: DocumentEntry.from_dict(filename, data)
                for filename, data in raw.items()
            }
        )

    def save(self, path: str | Path = DEFAULT_MANIFEST_PATH) -> None:
        path = Path(path)
        ordered = {name: self.entries[name].to_dict() for name in sorted(self.entries)}
        path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def add(self, entry: DocumentEntry) -> None:
        self.entries[entry.filename] = entry

    def verify_checksums(
        self, documents_dir: str | Path = Path("benchmark/documents")
    ) -> list[ChecksumMismatch]:
        """Recompute SHA-256 for every entry whose file is present on disk.

        An entry whose file is missing is not reported here -- that is a
        sourcing/redistribution question `validate.py` reports separately --
        only entries whose file exists but no longer matches what was
        recorded.
        """
        documents_dir = Path(documents_dir)
        mismatches: list[ChecksumMismatch] = []
        for filename, entry in self.entries.items():
            if not entry.sha256:
                continue
            file_path = documents_dir / filename
            if not file_path.is_file():
                continue
            actual = sha256_of(file_path)
            if actual != entry.sha256:
                mismatches.append(ChecksumMismatch(filename, entry.sha256, actual))
        return mismatches

    @staticmethod
    def build_entry(
        file_path: str | Path,
        *,
        title: str,
        category: str,
        source_url: str,
        license: str,
        redistributable: bool,
        retrieved: str,
        parser: PdfParser | None = None,
    ) -> DocumentEntry:
        """Build a manifest entry from an actual file: a real page count from
        `PdfParser` (never a hand-typed guess) and a freshly computed checksum.
        """
        file_path = Path(file_path)
        parser = parser or build_parser(IngestionConfig())
        document = parser.parse(file_path)
        return DocumentEntry(
            filename=file_path.name,
            title=title,
            category=category,
            pages=document.page_count,
            source_url=source_url,
            license=license,
            redistributable=redistributable,
            sha256=sha256_of(file_path),
            retrieved=retrieved,
        )
