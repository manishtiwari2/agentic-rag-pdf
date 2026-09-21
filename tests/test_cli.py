"""The command-line entry point, which is how the exit criteria are checked by
hand on a user-supplied PDF."""

from __future__ import annotations

import json

from src.cli import main

from . import pdf_fixtures as pdfs


def _pdf(tmp_path, data: bytes = None):
    path = tmp_path / "doc.pdf"
    path.write_bytes(data if data is not None else pdfs.structured_pdf())
    return str(path)


class TestAsk:
    def test_answers_and_prints_pages(self, tmp_path, capsys):
        code = main(
            [
                "ask",
                "--pdf",
                _pdf(tmp_path),
                "--question",
                "What recall did the hybrid system achieve?",
                "--offline",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "0.83" in out
        assert "page 2" in out

    def test_json_output_is_a_complete_record(self, tmp_path, capsys):
        main(
            [
                "ask",
                "--pdf",
                _pdf(tmp_path),
                "--question",
                "How are chunks embedded?",
                "--offline",
                "--json",
            ]
        )
        record = json.loads(capsys.readouterr().out)
        assert record["question"]
        assert record["citations"]
        assert record["retrieved"]
        assert "abstained" in record

    def test_scanned_pdf_reports_the_error_and_exits_nonzero(self, tmp_path, capsys):
        code = main(
            [
                "ask",
                "--pdf",
                _pdf(tmp_path, pdfs.scanned_pdf(3)),
                "--question",
                "anything",
                "--offline",
            ]
        )
        assert code == 1
        error = capsys.readouterr().err
        assert "ScannedPDFError" in error
        assert "ocr" in error.lower()


class TestValidateBenchmark:
    def test_the_shipped_dataset_exits_zero(self, capsys):
        code = main(["validate-benchmark"])
        assert code == 0
        assert "OK" in capsys.readouterr().out

    def test_a_valid_dataset_exits_zero(self, tmp_path, capsys):
        from src.benchmark.schema import Question, save_questions

        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "doc.pdf").write_bytes(pdfs.multipage_pdf(2))
        questions_path = tmp_path / "questions.json"
        save_questions(
            questions_path,
            [
                Question(
                    id="q1",
                    document="doc.pdf",
                    question="What is on page 1?",
                    answer="Page 1 content.",
                    evidence_pages=(1,),
                    question_type="factual",
                    difficulty="easy",
                    split="dev",
                )
            ],
        )
        code = main(
            [
                "validate-benchmark",
                "--questions",
                str(questions_path),
                "--documents-dir",
                str(docs),
                "--manifest",
                str(tmp_path / "no_such_manifest.json"),
            ]
        )
        assert code == 0
        assert "OK" in capsys.readouterr().out


class TestInspectPage:
    def test_renders_a_page_and_prints_its_text(self, tmp_path, capsys):
        code = main(
            [
                "inspect-page",
                "--pdf",
                _pdf(tmp_path, pdfs.multipage_pdf(2)),
                "--page",
                "1",
                "--out",
                str(tmp_path / "page.png"),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert pdfs.PAGE_MARKER.format(n=1) in out
        assert (tmp_path / "page.png").exists()


class TestInspect:
    def test_reports_chunking_statistics(self, tmp_path, capsys):
        code = main(["inspect", "--pdf", _pdf(tmp_path), "--offline", "--json"])
        assert code == 0
        stats = json.loads(capsys.readouterr().out)
        assert stats["pages"] == 2
        assert stats["chunks"] > 0

    def test_strategy_override_reaches_the_chunker(self, tmp_path, capsys):
        main(
            [
                "inspect",
                "--pdf",
                _pdf(tmp_path),
                "--offline",
                "--strategy",
                "fixed",
                "--json",
            ]
        )
        assert json.loads(capsys.readouterr().out)["chunk_strategy"] == "fixed"
