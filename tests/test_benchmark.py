"""Phase 0 benchmark tooling: schema round-trip, the manifest, and the
validator that enforces BENCHMARK_SPEC.md section 5.1.
"""

from __future__ import annotations

import json

import pytest

from src.benchmark.inspect import inspect_page
from src.benchmark.manifest import DocumentEntry, Manifest, sha256_of
from src.benchmark.schema import (
    QUESTION_TYPES,
    Question,
    load_questions,
    save_questions,
)
from src.benchmark.validate import (
    require_valid_benchmark,
    validate_benchmark,
    validate_questions,
)
from src.errors import BenchmarkValidationError

from . import pdf_fixtures as pdfs


def _question(**overrides) -> Question:
    base = dict(
        id="q1",
        document="doc.pdf",
        question="What is on page 1?",
        answer="Page 1 content.",
        evidence_pages=(1,),
        question_type="factual",
        difficulty="easy",
        split="dev",
    )
    base.update(overrides)
    return Question(**base)


@pytest.fixture
def documents_dir(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "doc.pdf").write_bytes(pdfs.multipage_pdf(3))
    return docs


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


class TestQuestionRoundTrip:
    def test_from_dict_to_dict_preserves_source_key_order(self):
        data = {
            "id": "q001",
            "document": "d.pdf",
            "question": "Q?",
            "answer": "A.",
            "evidence_pages": [3, 4],
            "evidence_chunk_ids": [],
            "question_type": "factual",
            "difficulty": "easy",
            "notes": "",
        }
        question = Question.from_dict(data)
        assert list(question.to_dict().keys()) == list(data.keys())

    def test_to_dict_omits_unset_optional_fields(self):
        question = _question(split=None)
        # split was never in source_order and is still at its default (None):
        # it must not appear as an explicit "split": null.
        assert "split" not in question.to_dict()

    def test_to_dict_keeps_a_value_that_was_actually_set(self):
        question = _question()  # split="dev", not the default
        assert question.to_dict()["split"] == "dev"

    def test_extra_fields_survive_a_round_trip(self):
        data = {
            "id": "q001",
            "document": "d.pdf",
            "question": "Q?",
            "answer": "A.",
            "evidence_pages": [1],
            "question_type": "factual",
            "difficulty": "easy",
            "reviewer": "a human",
        }
        question = Question.from_dict(data)
        assert question.to_dict()["reviewer"] == "a human"

    def test_save_then_load_is_idempotent(self, tmp_path):
        path = tmp_path / "questions.json"
        original = [
            {
                "id": "q001",
                "document": "document_01.pdf",
                "question": "What was measured?",
                "answer": "Latency and recall.",
                "evidence_pages": [3, 4],
                "evidence_chunk_ids": [],
                "question_type": "factual",
                "difficulty": "easy",
                "split": "dev",
                "notes": "",
            }
        ]
        path.write_text(json.dumps(original), encoding="utf-8")

        questions = load_questions(path)
        save_questions(path, questions)
        first_save = path.read_text(encoding="utf-8")

        save_questions(path, load_questions(path))
        assert path.read_text(encoding="utf-8") == first_save

    def test_save_keeps_short_arrays_inline(self, tmp_path):
        path = tmp_path / "questions.json"
        save_questions(path, [_question(evidence_pages=(3, 4))])
        text = path.read_text(encoding="utf-8")
        assert '"evidence_pages": [3, 4]' in text
        assert text.count("\n{") <= 1  # one object, not one line per field-array item


class TestDocumentEntry:
    def test_round_trips_known_and_extra_fields(self):
        data = {
            "title": "Example",
            "category": "research_paper",
            "pages": 12,
            "source_url": "https://example.org/paper.pdf",
            "license": "CC-BY-4.0",
            "redistributable": True,
            "sha256": "abc123",
            "retrieved": "2026-09-20",
            "reviewer_note": "checked twice",
        }
        entry = DocumentEntry.from_dict("document_01.pdf", data)
        assert entry.to_dict() == data


# ---------------------------------------------------------------------------
# validate.py -- one rule at a time
# ---------------------------------------------------------------------------


class TestValidationRules:
    def test_valid_fixture_passes(self, documents_dir):
        questions = [
            _question(id="q1", evidence_pages=(1,)),
            _question(id="q2", evidence_pages=(2,), split="eval"),
            _question(
                id="q3",
                answer=None,
                evidence_pages=(),
                question_type="unanswerable",
                difficulty="medium",
            ),
        ]
        result = validate_questions(questions, documents_dir)
        assert result.ok, result.report()

    def test_duplicate_id_is_reported(self, documents_dir):
        result = validate_questions(
            [_question(id="dup"), _question(id="dup", evidence_pages=(2,))],
            documents_dir,
        )
        assert not result.ok
        assert any(f.field == "id" for f in result.failures)

    def test_placeholder_text_is_reported(self, documents_dir):
        result = validate_questions(
            [_question(question="REPLACE_WITH_REAL_QUESTION")], documents_dir
        )
        assert any(f.field == "question" for f in result.failures)

    def test_answerable_question_needs_evidence_pages(self, documents_dir):
        result = validate_questions([_question(evidence_pages=())], documents_dir)
        assert any(f.field == "evidence_pages" for f in result.failures)

    def test_answerable_question_needs_a_non_null_answer(self, documents_dir):
        result = validate_questions([_question(answer=None)], documents_dir)
        assert any(f.field == "answer" for f in result.failures)

    def test_unanswerable_with_an_answer_is_rejected(self, documents_dir):
        result = validate_questions(
            [
                _question(
                    question_type="unanswerable",
                    answer="Should be null.",
                    evidence_pages=(),
                )
            ],
            documents_dir,
        )
        assert any(f.field == "answer" for f in result.failures)

    def test_unanswerable_with_evidence_pages_is_rejected(self, documents_dir):
        result = validate_questions(
            [
                _question(
                    question_type="unanswerable", answer=None, evidence_pages=(1,)
                )
            ],
            documents_dir,
        )
        assert any(f.field == "evidence_pages" for f in result.failures)

    def test_genuinely_unanswerable_question_passes(self, documents_dir):
        result = validate_questions(
            [
                _question(
                    question_type="unanswerable", answer=None, evidence_pages=()
                )
            ],
            documents_dir,
        )
        assert result.ok, result.report()

    def test_page_below_one_is_rejected(self, documents_dir):
        result = validate_questions([_question(evidence_pages=(0,))], documents_dir)
        assert any(
            f.field == "evidence_pages" and "1-based" in f.message
            for f in result.failures
        )

    def test_page_count_plus_one_is_rejected(self, documents_dir):
        """The off-by-one case: doc.pdf has 3 pages, page 4 does not exist."""
        result = validate_questions([_question(evidence_pages=(4,))], documents_dir)
        assert any(
            f.field == "evidence_pages" and "exceeds" in f.message
            for f in result.failures
        )

    def test_last_real_page_is_accepted(self, documents_dir):
        result = validate_questions([_question(evidence_pages=(3,))], documents_dir)
        assert result.ok, result.report()

    def test_missing_document_file_is_reported(self, documents_dir):
        result = validate_questions(
            [_question(document="missing.pdf")], documents_dir
        )
        assert any(f.field == "document" for f in result.failures)

    def test_unknown_question_type_is_reported(self, documents_dir):
        result = validate_questions(
            [_question(question_type="not_a_real_category")], documents_dir
        )
        assert any(f.field == "question_type" for f in result.failures)

    def test_every_taxonomy_category_is_accepted(self, documents_dir):
        for category in QUESTION_TYPES:
            if category == "unanswerable":
                q = _question(
                    question_type=category, answer=None, evidence_pages=()
                )
            else:
                q = _question(question_type=category)
            result = validate_questions([q], documents_dir)
            assert result.ok, f"{category}: {result.report()}"

    def test_split_must_be_dev_or_eval(self, documents_dir):
        result = validate_questions([_question(split="train")], documents_dir)
        assert any(f.field == "split" for f in result.failures)

    def test_multiple_failures_on_one_question_are_all_reported(self, documents_dir):
        """Several rules broken at once -- none of them may hide the others."""
        result = validate_questions(
            [
                _question(
                    question="REPLACE_WITH_REAL_QUESTION",
                    evidence_pages=(0,),
                    question_type="not_real",
                    split="train",
                )
            ],
            documents_dir,
        )
        fields = {f.field for f in result.failures}
        assert {"question", "evidence_pages", "question_type", "split"} <= fields

    def test_failures_across_many_questions_are_all_reported(self, documents_dir):
        questions = [
            _question(id="a", question_type="bogus"),
            _question(id="b", split="train"),
            _question(id="c", evidence_pages=(0,)),
        ]
        result = validate_questions(questions, documents_dir)
        assert {f.question_id for f in result.failures} >= {"a", "b", "c"}


class TestValidateBenchmarkEntryPoint:
    def test_a_placeholder_dataset_is_rejected(self, tmp_path, documents_dir):
        """A REPLACE_WITH-style template fails several BENCHMARK_SPEC.md section
        5.1 checks at once, by design -- this is what `questions.json` looked
        like before the real dataset replaced it."""
        path = tmp_path / "questions.json"
        save_questions(
            path,
            [
                Question(
                    id="q001",
                    document="document_01.pdf",
                    question="REPLACE_WITH_REAL_QUESTION",
                    answer="REPLACE_WITH_MANUALLY_VERIFIED_REFERENCE_ANSWER",
                    evidence_pages=(0,),
                    question_type="factual",
                    difficulty="easy",
                    split=None,
                )
            ],
        )
        result = validate_benchmark(path, documents_dir, manifest_path=None)
        assert not result.ok

    def test_the_shipped_dataset_validates_cleanly(self):
        """benchmark/questions.json now holds the real ~75-question dataset,
        not the placeholder template, and should pass every section 5.1 rule."""
        result = validate_benchmark()
        assert result.ok, result.report()

    def test_a_hand_built_valid_fixture_passes(self, tmp_path, documents_dir):
        path = tmp_path / "questions.json"
        save_questions(
            path,
            [
                _question(id="q1", evidence_pages=(1,)),
                _question(id="q2", evidence_pages=(2,), split="eval"),
                _question(
                    id="q3",
                    answer=None,
                    evidence_pages=(),
                    question_type="unanswerable",
                ),
            ],
        )
        result = validate_benchmark(path, documents_dir, manifest_path=None)
        assert result.ok, result.report()

    def test_require_valid_benchmark_raises_on_failure(self, tmp_path, documents_dir):
        path = tmp_path / "questions.json"
        save_questions(path, [_question(id="q1", evidence_pages=(0,))])
        with pytest.raises(BenchmarkValidationError):
            require_valid_benchmark(path, documents_dir, manifest_path=None)

    def test_require_valid_benchmark_passes_a_good_dataset(self, tmp_path, documents_dir):
        path = tmp_path / "questions.json"
        save_questions(path, [_question(id="q1", evidence_pages=(1,))])
        require_valid_benchmark(path, documents_dir, manifest_path=None)

    def test_require_valid_benchmark_passes_the_shipped_dataset(self):
        require_valid_benchmark()


class TestManifestChecksumVerification:
    def test_matching_checksum_is_not_reported(self, documents_dir):
        manifest = Manifest(
            {
                "doc.pdf": DocumentEntry(
                    filename="doc.pdf",
                    title="t",
                    category="c",
                    pages=3,
                    source_url="https://example.org/doc.pdf",
                    license="CC-BY-4.0",
                    redistributable=True,
                    sha256=sha256_of(documents_dir / "doc.pdf"),
                    retrieved="2026-09-20",
                )
            }
        )
        assert manifest.verify_checksums(documents_dir) == []

    def test_mismatched_checksum_is_detected(self, documents_dir):
        manifest = Manifest(
            {
                "doc.pdf": DocumentEntry(
                    filename="doc.pdf",
                    title="t",
                    category="c",
                    pages=3,
                    source_url="https://example.org/doc.pdf",
                    license="CC-BY-4.0",
                    redistributable=True,
                    sha256="0" * 64,
                    retrieved="2026-09-20",
                )
            }
        )
        mismatches = manifest.verify_checksums(documents_dir)
        assert len(mismatches) == 1
        assert mismatches[0].filename == "doc.pdf"

    def test_checksum_mismatch_fails_full_validation(self, tmp_path, documents_dir):
        questions_path = tmp_path / "questions.json"
        save_questions(questions_path, [_question(id="q1", evidence_pages=(1,))])

        manifest_path = tmp_path / "documents_metadata.json"
        Manifest(
            {
                "doc.pdf": DocumentEntry(
                    filename="doc.pdf",
                    title="t",
                    category="c",
                    pages=3,
                    source_url="https://example.org/doc.pdf",
                    license="CC-BY-4.0",
                    redistributable=True,
                    sha256="0" * 64,
                    retrieved="2026-09-20",
                )
            }
        ).save(manifest_path)

        result = validate_benchmark(questions_path, documents_dir, manifest_path)
        assert not result.ok
        assert any(f.field == "sha256" for f in result.failures)

    def test_build_entry_reads_a_real_page_count_and_checksum(self, documents_dir):
        entry = Manifest.build_entry(
            documents_dir / "doc.pdf",
            title="t",
            category="c",
            source_url="https://example.org/doc.pdf",
            license="CC-BY-4.0",
            redistributable=True,
            retrieved="2026-09-20",
        )
        assert entry.pages == 3
        assert entry.sha256 == sha256_of(documents_dir / "doc.pdf")


# ---------------------------------------------------------------------------
# inspect.py
# ---------------------------------------------------------------------------


class TestInspectPage:
    def test_renders_a_png_and_the_matching_page_text(self, documents_dir):
        png, text = inspect_page(documents_dir / "doc.pdf", 2)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert pdfs.PAGE_MARKER.format(n=2) in text

    def test_out_of_range_page_is_rejected(self, documents_dir):
        with pytest.raises(Exception, match="out of range"):
            inspect_page(documents_dir / "doc.pdf", 99)
