"""Conversation layer (DD-065) and chat interface logic (DD-066).

Everything here runs without gradio: the UI's behaviour lives in plain
functions, and only ``build_app`` touches the library.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.chat import ui
from src.chat.session import (
    ChatSession,
    LLMRewriter,
    RuleBasedRewriter,
    Turn,
    build_rewriter,
    is_follow_up,
)
from src.generation.prompts import ABSTENTION_SENTENCE
from src.pipeline import RAGResult, build_pipeline

from . import pdf_fixtures as pdfs


class Canned:
    """An LLM backend returning fixed text."""

    name = "test/canned"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        self.prompt = user_prompt
        return self.text


@pytest.fixture
def pipeline(offline_config, structured_pdf_path):
    pipeline = build_pipeline(offline_config)
    pipeline.index(structured_pdf_path)
    return pipeline


FIRST = "What recall did the hybrid system achieve?"
HISTORY = [Turn(question=FIRST, standalone=FIRST, answer="Recall reached 0.86. [C1]")]


class TestFollowUpDetection:
    @pytest.mark.parametrize(
        "question",
        ["What about its latency?", "And the dense one?", "Why is that?", "How long?"],
    )
    def test_follow_ups_are_detected(self, question):
        assert is_follow_up(question)

    def test_a_standalone_question_is_not(self):
        assert not is_follow_up("How are chunks embedded by the retrieval method?")


class TestRuleBasedRewriter:
    def test_the_first_turn_is_never_rewritten(self):
        rewrite = RuleBasedRewriter().rewrite("What about its latency?", [])
        assert rewrite.standalone == "What about its latency?"

    def test_a_follow_up_carries_the_previous_question(self):
        rewrite = RuleBasedRewriter().rewrite("What about its latency?", HISTORY)
        assert rewrite.standalone == f"What about its latency? ({FIRST})"
        assert rewrite.rewriter == "rules" and not rewrite.fallback

    def test_a_standalone_question_is_left_alone(self):
        question = "How are chunks embedded by the retrieval method?"
        assert RuleBasedRewriter().rewrite(question, HISTORY).standalone == question


class TestLLMRewriter:
    def test_the_models_rewrite_is_used(self):
        backend = Canned("What latency did the hybrid system achieve?\n")
        rewrite = LLMRewriter(backend).rewrite("What about its latency?", HISTORY)
        assert rewrite.standalone == "What latency did the hybrid system achieve?"
        assert rewrite.model_calls == 1 and not rewrite.fallback
        # The history reaches the prompt, without the answer's evidence markers.
        assert FIRST in backend.prompt and "[C1]" not in backend.prompt

    @pytest.mark.parametrize("reply", ["", ABSTENTION_SENTENCE, "x" * 1000])
    def test_an_unusable_reply_falls_back_to_the_rules(self, reply):
        rewrite = LLMRewriter(Canned(reply)).rewrite("What about its latency?", HISTORY)
        assert rewrite.fallback and rewrite.rewriter == "llm"
        assert rewrite.standalone == f"What about its latency? ({FIRST})"

    def test_no_model_call_on_the_first_turn(self):
        backend = Canned("anything")
        LLMRewriter(backend).rewrite("What about its latency?", [])
        assert backend.calls == 0


class TestChatSession:
    def test_offline_uses_the_rule_rewriter_and_no_second_model(self, pipeline):
        assert isinstance(build_rewriter(pipeline), RuleBasedRewriter)
        session = ChatSession(pipeline)
        assert session.pipeline.backend is pipeline.backend

    def test_the_first_turn_answers_exactly_as_the_pipeline_does(self, pipeline):
        direct = pipeline.ask(FIRST)
        chatted = ChatSession(pipeline).ask(FIRST)
        timing = {"retrieval_s", "generation_s", "latency_s", "rerank_s"}
        assert chatted.answer == direct.answer
        assert chatted.citations == direct.citations
        assert {k: v for k, v in chatted.metadata.items() if k not in timing | {"conversation"}} == {
            k: v for k, v in direct.metadata.items() if k not in timing
        }
        assert chatted.metadata["conversation"]["rewritten"] is False

    def test_a_follow_up_is_rewritten_and_recorded(self, pipeline):
        session = ChatSession(pipeline)
        session.ask(FIRST)
        result = session.ask("What about its latency?")
        conversation = result.metadata["conversation"]
        assert conversation["turn"] == 2
        assert conversation["original_question"] == "What about its latency?"
        assert conversation["standalone_question"] == f"What about its latency? ({FIRST})"
        assert conversation["rewritten"] is True
        assert result.question == conversation["standalone_question"]

    def test_history_is_capped(self, pipeline):
        session = ChatSession(pipeline, max_turns=2)
        for question in ("How are chunks embedded?", "What recall was reached?", "Why?"):
            session.ask(question)
        assert [t.question for t in session.history] == ["What recall was reached?", "Why?"]
        session.reset()
        assert session.history == [] and session.turns == 0

    def test_the_llm_rewriter_shares_the_pipeline_backend(self, pipeline):
        shared = Canned("What latency did the hybrid system achieve?")
        rewriter = LLMRewriter(shared)
        session = ChatSession(pipeline, rewriter=rewriter)
        session.ask(FIRST)
        result = session.ask("What about its latency?")
        assert shared.calls == 1
        assert result.metadata["conversation"]["standalone_question"] == (
            "What latency did the hybrid system achieve?"
        )

    def test_max_turns_must_be_positive(self, pipeline):
        with pytest.raises(ValueError):
            ChatSession(pipeline, max_turns=0)


def test_results_stay_frozen(pipeline):
    result = ChatSession(pipeline).ask(FIRST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "changed"


# ---------------------------------------------------------------------------
# Chat interface logic (DD-066), without gradio
# ---------------------------------------------------------------------------

@pytest.fixture
def uploads(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return write


class TestIndexing:
    def test_nothing_uploaded_asks_for_a_file(self):
        state, status = ui.load_documents([], "agentic", ui.ChatState())
        assert not state.ready and "Upload" in status

    def test_a_pdf_is_indexed_with_its_counts(self, uploads):
        path = uploads("structured.pdf", pdfs.structured_pdf())
        state, status = ui.load_documents([path], "agentic", ui.ChatState())
        assert state.ready and state.system == "agentic"
        (doc,) = state.documents
        assert doc["source_name"] == "structured.pdf" and doc["pages"] == 2
        assert doc["chunks"] >= 1 and doc["ocr_pages"] == []
        assert "| structured.pdf | 2 | 0 |" in status
        assert "offline stand-in stack" in status

    def test_several_pdfs_share_one_index(self, uploads):
        paths = [
            uploads("structured.pdf", pdfs.structured_pdf()),
            uploads("multipage.pdf", pdfs.multipage_pdf(3)),
        ]
        state, _ = ui.load_documents(paths, "hybrid", ui.ChatState())
        assert [d["source_name"] for d in state.documents] == ["structured.pdf", "multipage.pdf"]
        assert len(state.pipeline.chunks) == sum(d["chunks"] for d in state.documents)

    def test_an_encrypted_pdf_is_reported_with_its_remedy_and_skipped(self, uploads):
        paths = [
            uploads("locked.pdf", pdfs.encrypted_pdf()),
            uploads("structured.pdf", pdfs.structured_pdf()),
        ]
        state, status = ui.load_documents(paths, "dense", ui.ChatState())
        assert [d["source_name"] for d in state.documents] == ["structured.pdf"]
        assert "locked.pdf: EncryptedPDFError" in status and "password" in status

    def test_a_scanned_pdf_is_ocrd_and_its_pages_reported(self, uploads, monkeypatch):
        class FakeOcr:
            name = "test/fake-ocr"

            def image_to_text(self, image):
                return "The scanned page reports a median latency of 4.2 seconds."

        monkeypatch.setattr(
            "src.ingestion.parser_base.build_ocr_engine", lambda config: FakeOcr()
        )
        path = uploads("scan.pdf", pdfs.scanned_pdf(2))
        state, status = ui.load_documents([path], "dense", ui.ChatState())
        assert state.documents[0]["ocr_pages"] == [1, 2]
        assert "| scan.pdf | 2 | 2 |" in status

    def test_only_unreadable_pdfs_leave_nothing_to_ask(self, uploads):
        path = uploads("locked.pdf", pdfs.encrypted_pdf())
        state, status = ui.load_documents([path], "dense", ui.ChatState())
        assert not state.ready and "Nothing was indexed" in status
        history, _, _ = ui.respond("Anything?", [], state)
        assert "Index" in history[-1]["content"]


class TestResponding:
    def test_asking_before_indexing_says_what_to_do(self):
        history, _, trace = ui.respond("What recall?", [], ui.ChatState())
        assert history[0] == {"role": "user", "content": "What recall?"}
        assert "Upload a PDF" in history[1]["content"] and trace == ""

    def test_an_empty_message_changes_nothing(self):
        history, _, _ = ui.respond("   ", [], ui.ChatState())
        assert history == []

    def test_an_answer_carries_page_citations_and_a_trace(self, uploads):
        path = uploads("structured.pdf", pdfs.structured_pdf())
        state, _ = ui.load_documents([path], "agentic", ui.ChatState())
        history, state, trace = ui.respond(FIRST, [], state)
        answer = history[-1]["content"]
        assert history[-1]["role"] == "assistant"
        assert "**Sources**" in answer and "page" in answer
        assert "**Round 1:**" in trace and "**Verifier:**" in trace
        history, state, trace = ui.respond("What about its latency?", history, state)
        assert len(history) == 4 and "Follow-up rewritten" in trace

    def test_a_baseline_trace_says_there_is_no_agent_loop(self, uploads):
        path = uploads("structured.pdf", pdfs.structured_pdf())
        state, _ = ui.load_documents([path], "dense", ui.ChatState())
        _, _, trace = ui.respond(FIRST, [], state)
        assert "no agent loop" in trace


class TestFormatting:
    def test_a_refusal_is_shown_as_one(self):
        result = RAGResult(
            question="q", answer=ABSTENTION_SENTENCE, citations=(), evidence=(),
            retrieved=(), abstained=True,
        )
        assert ui.format_answer(result) == f"**Refused:** {ABSTENTION_SENTENCE}"

    def test_an_error_names_the_file_and_keeps_the_remedy(self):
        text = ui.format_error("/tmp/x/scan.pdf", RuntimeError("Run OCR first: install it."))
        assert text == "**scan.pdf: RuntimeError.** Run OCR first: install it."

    def test_an_unknown_system_is_refused(self):
        with pytest.raises(ValueError):
            ui.make_config("magic", offline=True)


def test_the_gradio_app_builds_when_gradio_is_installed():
    gradio = pytest.importorskip("gradio")
    app = ui.build_app(offline=True)
    assert isinstance(app, gradio.Blocks)
