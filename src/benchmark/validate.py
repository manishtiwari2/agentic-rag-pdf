"""The benchmark validator (BENCHMARK_SPEC.md section 5.1).

A malformed benchmark produces plausible-looking numbers, which is the worst
failure mode an evaluation harness has. Every rule below is checked for every
question before anything is reported, so fixing a 75-question file is one
round of edits instead of 75: a validator that stops at the first failure
turns dataset repair into a loop bounded only by patience.

Nothing here is an LLM call (DD-018): every rule is a deterministic check
against the loaded questions, the files in `documents_dir`, and real page
counts read through `PdfParser` -- never a recorded count, because a stale
recorded count is exactly the kind of error this validator exists to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import IngestionConfig
from ..errors import BenchmarkValidationError, RAGError
from ..ingestion.parser import PdfParser, build_parser
from .manifest import DEFAULT_MANIFEST_PATH, Manifest
from .schema import (
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_QUESTIONS_PATH,
    QUESTION_TYPES,
    SPLITS,
    Question,
    load_questions,
)

_PLACEHOLDER = "REPLACE_WITH"


@dataclass(frozen=True)
class ValidationFailure:
    """One rule violation, precise enough to fix without re-deriving what broke."""

    question_id: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.question_id} [{self.field}]: {self.message}"


@dataclass(frozen=True)
class ValidationResult:
    """Every failure found in one validation pass, never just the first."""

    failures: tuple[ValidationFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def __bool__(self) -> bool:
        return self.ok

    def report(self) -> str:
        if self.ok:
            return "Benchmark OK."
        header = f"{len(self.failures)} problem(s) found:"
        return "\n".join([header, *(f"  {f}" for f in self.failures)])


def _label(question: Question, index: int) -> str:
    return question.id or f"<question at index {index}>"


def _placeholder_strings(value: Any) -> list[str]:
    """Every string reachable from `value`, for the placeholder scan."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _placeholder_strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _placeholder_strings(item)]
    return []


def _check_placeholders(question: Question, label: str) -> list[ValidationFailure]:
    failures = []
    fields = {
        "id": question.id,
        "document": question.document,
        "question": question.question,
        "answer": question.answer,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "split": question.split,
        "evidence_chunk_ids": question.evidence_chunk_ids,
        "reasoning_steps": question.reasoning_steps,
        "alternative_answers": question.alternative_answers,
        "notes": question.notes,
        "extra": question.extra,
    }
    for field_name, value in fields.items():
        for string in _placeholder_strings(value):
            if _PLACEHOLDER in string:
                failures.append(
                    ValidationFailure(
                        label,
                        field_name,
                        f"contains an unreplaced placeholder: {string!r}.",
                    )
                )
    return failures


def _check_answerability(question: Question, label: str) -> list[ValidationFailure]:
    failures = []
    if question.is_unanswerable:
        if question.answer is not None:
            failures.append(
                ValidationFailure(
                    label,
                    "answer",
                    "question_type is 'unanswerable', so answer must be null, "
                    f"got {question.answer!r}.",
                )
            )
        if question.evidence_pages:
            failures.append(
                ValidationFailure(
                    label,
                    "evidence_pages",
                    "question_type is 'unanswerable', so evidence_pages must "
                    f"be empty, got {list(question.evidence_pages)}.",
                )
            )
    else:
        if not question.answer or not question.answer.strip():
            failures.append(
                ValidationFailure(
                    label,
                    "answer",
                    "answerable questions need a non-empty answer.",
                )
            )
        if not question.evidence_pages:
            failures.append(
                ValidationFailure(
                    label,
                    "evidence_pages",
                    "answerable questions need at least one evidence page.",
                )
            )
    return failures


def _check_evidence_pages(
    question: Question, label: str, page_count: int | None
) -> list[ValidationFailure]:
    failures = []
    for page in question.evidence_pages:
        if not isinstance(page, int) or isinstance(page, bool):
            failures.append(
                ValidationFailure(
                    label,
                    "evidence_pages",
                    f"{page!r} is not an integer page number.",
                )
            )
            continue
        if page < 1:
            failures.append(
                ValidationFailure(
                    label,
                    "evidence_pages",
                    f"page {page} is not 1-based; evidence_pages must be >= 1.",
                )
            )
        elif page_count is not None and page > page_count:
            failures.append(
                ValidationFailure(
                    label,
                    "evidence_pages",
                    f"page {page} exceeds {question.document}'s real page "
                    f"count of {page_count} (valid range 1..{page_count}).",
                )
            )
    return failures


def _resolve_page_count(
    document: str,
    documents_dir: Path,
    parser: PdfParser,
    cache: dict[str, int | None | str],
) -> int | None | str:
    """The real page count for `document`, or an error message string.

    Cached per filename: a 75-question benchmark over 5-10 documents would
    otherwise reparse the same PDF dozens of times.
    """
    if document in cache:
        return cache[document]

    file_path = documents_dir / document
    if not file_path.is_file():
        cache[document] = "file not found"
        return cache[document]

    try:
        parsed = parser.parse(file_path)
    except RAGError as exc:
        cache[document] = f"could not be parsed: {exc}"
        return cache[document]

    cache[document] = parsed.page_count
    return cache[document]


def validate_questions(
    questions: list[Question],
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    *,
    parser: PdfParser | None = None,
    manifest: Manifest | None = None,
) -> ValidationResult:
    """Check every BENCHMARK_SPEC.md section 5.1 rule against `questions`.

    `manifest`, if given, adds its checksum verification to the same report
    (section 4.1): a document whose bytes changed since annotation is exactly
    as fatal to the ground truth as a bad page number.
    """
    documents_dir = Path(documents_dir)
    parser = parser or build_parser(IngestionConfig())
    failures: list[ValidationFailure] = []

    seen_ids: dict[str, int] = {}
    for question in questions:
        seen_ids[question.id] = seen_ids.get(question.id, 0) + 1

    page_count_cache: dict[str, int | None | str] = {}
    checked_documents: set[str] = set()

    for index, question in enumerate(questions):
        label = _label(question, index)

        count = seen_ids.get(question.id, 0)
        if question.id and count > 1:
            failures.append(
                ValidationFailure(
                    label, "id", f"duplicated: appears {count} times."
                )
            )
        elif not question.id:
            failures.append(ValidationFailure(label, "id", "missing or empty."))

        failures.extend(_check_placeholders(question, label))

        if question.question_type not in QUESTION_TYPES:
            failures.append(
                ValidationFailure(
                    label,
                    "question_type",
                    f"{question.question_type!r} is not one of "
                    f"{sorted(QUESTION_TYPES)}.",
                )
            )

        if question.split not in SPLITS:
            failures.append(
                ValidationFailure(
                    label,
                    "split",
                    f"{question.split!r} is not one of {sorted(SPLITS)}.",
                )
            )

        failures.extend(_check_answerability(question, label))

        if not question.document:
            failures.append(
                ValidationFailure(label, "document", "missing or empty.")
            )
        else:
            resolved = _resolve_page_count(
                question.document, documents_dir, parser, page_count_cache
            )
            if isinstance(resolved, str) and question.document not in checked_documents:
                checked_documents.add(question.document)
                failures.append(
                    ValidationFailure(question.document, "document", f"{resolved}.")
                )
            page_count = resolved if isinstance(resolved, int) else None
            failures.extend(_check_evidence_pages(question, label, page_count))

    if manifest is not None:
        for mismatch in manifest.verify_checksums(documents_dir):
            failures.append(
                ValidationFailure(mismatch.filename, "sha256", str(mismatch))
            )

    return ValidationResult(tuple(failures))


def validate_benchmark(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    *,
    parser: PdfParser | None = None,
) -> ValidationResult:
    """Load `questions_path` and validate it. The entry point the CLI uses."""
    try:
        questions = load_questions(questions_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ValidationResult(
            (ValidationFailure(str(questions_path), "file", str(exc)),)
        )

    manifest = Manifest.load(manifest_path) if manifest_path is not None else None
    return validate_questions(
        questions, documents_dir, parser=parser, manifest=manifest
    )


def require_valid_benchmark(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    *,
    parser: PdfParser | None = None,
) -> ValidationResult:
    """Validate, and raise if invalid.

    This is the "refuse to start" guard BENCHMARK_SPEC.md section 5.1
    requires: whatever eventually runs the benchmark (the CLI now, the Phase 3
    metrics harness later) calls this before touching a single question.
    """
    result = validate_benchmark(
        questions_path, documents_dir, manifest_path, parser=parser
    )
    if not result.ok:
        raise BenchmarkValidationError(result.report())
    return result
