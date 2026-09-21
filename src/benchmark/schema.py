"""Typed models for the benchmark dataset (BENCHMARK_SPEC.md sections 4.2 and 5).

Two things live here: :class:`Question`, one row of ``benchmark/questions.json``,
and :class:`DocumentEntry`, one row of the document manifest
(``benchmark/documents_metadata.json``, section 4.2). Neither validates
itself -- a placeholder row with ``evidence_pages: [0]`` must load cleanly so
that :mod:`.validate` can report it as one failure among many, rather than the
loader raising on the first bad row and hiding the rest (DD-018: this module
does the deterministic bookkeeping, ``validate.py`` does the checking).

``load_questions``/``save_questions`` round-trip ``questions.json`` without
reordering fields or expanding its inline arrays onto multiple lines. That
matters because the file is hand-edited: a human fixing one question's answer
should see a one-line diff, not a whole-file reformat from a generic
``json.dump(..., indent=2)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: benchmark/README.md section 8 and BENCHMARK_SPEC.md section 6/8: the ten
#: categories a `question_type` must be one of. Section 6 documents nine of
#: them as numbered subsections; `summary` is the tenth, used by
#: `questions.json` and budgeted for in section 8's distribution table even
#: though section 8 notes an earlier version of that table omitted it.
QUESTION_TYPES: frozenset[str] = frozenset(
    {
        "factual",
        "definition",
        "explanation",
        "comparison",
        "numerical",
        "table",
        "multi_hop",
        "unanswerable",
        "ambiguous",
        "summary",
    }
)

#: BENCHMARK_SPEC.md section 5: `split` is fixed, stratified, and never
#: recomputed (EVALUATION_PROTOCOL.md section 5.1).
SPLITS: frozenset[str] = frozenset({"dev", "eval"})

DEFAULT_QUESTIONS_PATH = Path("benchmark/questions.json")
DEFAULT_DOCUMENTS_DIR = Path("benchmark/documents")

#: Fields with a schema-defined default. Anything not in here (id, document,
#: question, answer, evidence_pages, question_type, difficulty) is always
#: written back out, even when empty, because BENCHMARK_SPEC.md section 5
#: requires all of them present.
#: Defaults as they appear in `to_dict`'s `values` (lists, not tuples) so the
#: "differs from default" comparison below is comparing like with like.
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "split": None,
    "evidence_chunk_ids": [],
    "reasoning_steps": [],
    "alternative_answers": [],
    "notes": "",
}
_REQUIRED_FIELDS = (
    "id",
    "document",
    "question",
    "answer",
    "evidence_pages",
    "question_type",
    "difficulty",
)
_KNOWN_FIELDS = frozenset(_REQUIRED_FIELDS) | frozenset(_OPTIONAL_DEFAULTS)


@dataclass(frozen=True)
class Question:
    """One row of ``benchmark/questions.json`` (BENCHMARK_SPEC.md section 5).

    ``evidence_pages`` is **1-based**, matching ``Page.page_number`` in
    ``src/ingestion/document.py`` -- the same convention, not a coincidence.
    """

    id: str
    document: str
    question: str
    answer: str | None
    evidence_pages: tuple[int, ...]
    question_type: str
    difficulty: str
    split: str | None = None
    evidence_chunk_ids: tuple[str, ...] = ()
    reasoning_steps: tuple[Any, ...] = ()
    alternative_answers: tuple[str, ...] = ()
    notes: str = ""
    #: Fields present in the source JSON that this schema does not name,
    #: kept so a save never silently drops data a human added.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Key order as read from the source file. Empty for a `Question` built
    #: directly rather than loaded, in which case `to_dict` falls back to
    #: the canonical order above.
    source_order: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @property
    def is_unanswerable(self) -> bool:
        return self.question_type == "unanswerable"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        extra = {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}
        return cls(
            id=data.get("id", ""),
            document=data.get("document", ""),
            question=data.get("question", ""),
            answer=data.get("answer"),
            evidence_pages=tuple(data.get("evidence_pages") or ()),
            question_type=data.get("question_type", ""),
            difficulty=data.get("difficulty", ""),
            split=data.get("split"),
            evidence_chunk_ids=tuple(data.get("evidence_chunk_ids") or ()),
            reasoning_steps=tuple(data.get("reasoning_steps") or ()),
            alternative_answers=tuple(data.get("alternative_answers") or ()),
            notes=data.get("notes", ""),
            extra=extra,
            source_order=tuple(data.keys()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Inverse of :meth:`from_dict`, preserving the source key order.

        A key that was present in the source file, or that always belongs in
        the schema, or whose value differs from its default, is written out
        in its original position. Everything else -- an optional field a
        loaded question never had, still at its default -- is left out
        rather than written as an explicit ``null``.
        """
        values: dict[str, Any] = {
            "id": self.id,
            "document": self.document,
            "question": self.question,
            "answer": self.answer,
            "evidence_pages": list(self.evidence_pages),
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "split": self.split,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "reasoning_steps": list(self.reasoning_steps),
            "alternative_answers": list(self.alternative_answers),
            "notes": self.notes,
            **self.extra,
        }
        present = (
            set(_REQUIRED_FIELDS)
            | set(self.source_order)
            | set(self.extra)
            | {
                key
                for key, default in _OPTIONAL_DEFAULTS.items()
                if values[key] != default
            }
        )
        order = list(self.source_order) or list(values)
        ordered_keys = [k for k in order if k in values] + [
            k for k in values if k not in order
        ]
        return {k: values[k] for k in ordered_keys if k in present}


@dataclass(frozen=True)
class DocumentEntry:
    """One row of the document manifest (BENCHMARK_SPEC.md section 4.2)."""

    filename: str
    title: str
    category: str
    pages: int
    source_url: str
    license: str
    redistributable: bool
    sha256: str
    retrieved: str
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = (
        "title",
        "category",
        "pages",
        "source_url",
        "license",
        "redistributable",
        "sha256",
        "retrieved",
    )

    @classmethod
    def from_dict(cls, filename: str, data: dict[str, Any]) -> DocumentEntry:
        extra = {k: v for k, v in data.items() if k not in cls._KNOWN}
        return cls(
            filename=filename,
            title=data.get("title", ""),
            category=data.get("category", ""),
            pages=int(data.get("pages") or 0),
            source_url=data.get("source_url", ""),
            license=data.get("license", ""),
            redistributable=bool(data.get("redistributable", False)),
            sha256=data.get("sha256", ""),
            retrieved=data.get("retrieved", ""),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "pages": self.pages,
            "source_url": self.source_url,
            "license": self.license,
            "redistributable": self.redistributable,
            "sha256": self.sha256,
            "retrieved": self.retrieved,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# questions.json I/O
# ---------------------------------------------------------------------------


def load_questions(path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    """Load every question, preserving the file's own field order."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON array of questions, got "
            f"{type(data).__name__}."
        )
    return [Question.from_dict(item) for item in data]


def save_questions(path: str | Path, questions: list[Question]) -> None:
    """Write questions back in the file's house style: one field per line,
    scalar/short arrays inline, no reordering beyond what `Question.to_dict`
    already decided.

    Line endings are always ``\\n``. This is a deliberate normalization, not
    a formatting choice left to chance: it makes a round-trip idempotent
    regardless of what a previous editor's line-ending setting produced.
    """
    path = Path(path)
    records = [q.to_dict() for q in questions]
    path.write_text(_dumps_records(records), encoding="utf-8", newline="\n")


def _dumps_records(records: list[dict[str, Any]]) -> str:
    """Render a JSON array of flat objects with one field per line.

    Each field's value is rendered with a single `json.dumps` call rather
    than recursively indented, which is what keeps a short array like
    `"evidence_pages": [3, 4]` on one line instead of one line per element.
    """
    if not records:
        return "[]\n"
    return "[\n" + ",\n".join(_dumps_object(record) for record in records) + "\n]\n"


def _dumps_object(record: dict[str, Any]) -> str:
    items = list(record.items())
    lines = ["{"]
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append(f"{json.dumps(key)}: {json.dumps(value)}{comma}")
    lines.append("}")
    return "\n".join(lines)
