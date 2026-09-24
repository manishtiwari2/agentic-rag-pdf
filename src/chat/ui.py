"""The notebook's chat interface (DD-066).

Everything the interface does is a plain function over a ``ChatState``:
indexing uploads, answering a message, and formatting answers, traces and
errors. ``build_app`` only wires those functions to Gradio components, and it
is the one place gradio is imported. The tests therefore exercise the whole
interface without gradio installed, and the library stays importable on a
machine that has never heard of it.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..config import RAGConfig
from ..errors import RAGError
from ..pipeline import RAGResult, build_pipeline
from .session import ChatSession

SYSTEMS = ("dense", "hybrid", "agentic")
SYSTEM_LABELS = {
    "dense": "Dense RAG (Baseline A)",
    "hybrid": "Hybrid RAG (Baseline B)",
    "agentic": "Agentic RAG",
}


def make_config(system: str, offline: bool) -> RAGConfig:
    """The configuration the interface runs: a preset with the chosen system.

    ``offline`` selects the dependency-free stand-ins (hashing embedder,
    scripted generator, rule-based agents), which run on any CPU.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, not {system!r}")
    config = RAGConfig.offline() if offline else RAGConfig.default()
    return dataclasses.replace(
        config, retrieval=dataclasses.replace(config.retrieval, strategy=system)
    )


@dataclass
class ChatState:
    """One user's interface state: the indexed pipeline and its conversation."""

    system: str = "agentic"
    offline: bool = True
    pipeline: Any = None
    session: ChatSession | None = None
    #: One entry per indexed file: name, pages, chunks and OCR'd pages.
    documents: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.session is not None


def _path(upload: Any) -> str:
    """A filesystem path from whatever the upload widget handed over."""
    if isinstance(upload, (str, os.PathLike)):
        return os.fspath(upload)
    return getattr(upload, "name", None) or getattr(upload, "path")


def format_error(source: str, exc: BaseException) -> str:
    """An ingestion error as the user should see it: which file, what, and the remedy.

    Every ``RAGError`` message already carries its remedy (``src/errors.py``),
    so the message is shown whole rather than summarised away.
    """
    name = os.path.basename(source)
    return f"**{name}: {type(exc).__name__}.** {exc}"


def load_documents(
    uploads: Sequence[Any] | None,
    system: str,
    state: ChatState,
    config_factory: Callable[[str, bool], RAGConfig] = make_config,
) -> tuple[ChatState, str]:
    """Index the uploaded PDFs with the chosen system; return the new state and a status.

    A file that cannot be read (encrypted, corrupt, image-only with OCR
    unavailable) is reported with its remedy and skipped; the others are still
    indexed. Re-indexing starts a new conversation.
    """
    state = dataclasses.replace(state, system=system, documents=[], errors=[], session=None)
    paths = [_path(u) for u in uploads or []]
    if not paths:
        return state, "Upload one or more PDFs, then press **Index**."

    pipeline = build_pipeline(config_factory(system, state.offline))
    errors: list[str] = []
    documents = pipeline.index_documents(
        paths, on_error=lambda path, exc: errors.append(format_error(os.fspath(path), exc))
    )
    state.errors = errors
    if documents:
        state.pipeline = pipeline
        state.session = ChatSession(pipeline)
        state.documents = list(pipeline.index_stats["documents"])
    else:
        state.pipeline = None
    return state, format_status(state)


def format_status(state: ChatState) -> str:
    lines: list[str] = []
    if state.documents:
        lines.append(f"Indexed for **{SYSTEM_LABELS[state.system]}**"
                     + (" (offline stand-in stack)" if state.offline else "") + ":")
        lines.append("")
        lines.append("| File | Pages | OCR'd pages | Chunks |")
        lines.append("| --- | ---: | ---: | ---: |")
        for doc in state.documents:
            lines.append(
                f"| {doc['source_name']} | {doc['pages']} | {len(doc['ocr_pages'])} | {doc['chunks']} |"
            )
    if state.errors:
        if lines:
            lines.append("")
        lines.append("Not indexed:")
        lines.extend(f"* {error}" for error in state.errors)
    if not state.documents:
        lines.append("")
        lines.append("Nothing was indexed, so there is nothing to ask about yet.")
    return "\n".join(lines).strip()


def format_answer(result: RAGResult, documents: Sequence[dict[str, Any]] = ()) -> str:
    """The answer with its page citations, or a refusal shown as one."""
    names = {d["document_id"]: d["source_name"] for d in documents}
    several = len(names) > 1
    if result.abstained:
        text = f"**Refused:** {result.answer}"
    else:
        text = result.answer
    if result.citations:
        lines = [text, "", "**Sources**"]
        for citation in result.citations:
            where = citation.label
            if several:
                where += f", {names.get(citation.document_id, citation.document_id)}"
            section = f" ({citation.section})" if citation.section else ""
            lines.append(f"* [{citation.marker}] {where}{section}")
        text = "\n".join(lines)
    return text


def format_trace(result: RAGResult) -> str:
    """What happened on the way to the answer, for the collapsible trace panel."""
    meta = result.metadata
    lines: list[str] = []
    conversation = meta.get("conversation")
    if conversation and conversation.get("rewritten"):
        fallback = ", fell back to the rules" if conversation.get("rewriter_fallback") else ""
        lines.append(
            f"**Follow-up rewritten** ({conversation['rewriter']}{fallback}): "
            f"{conversation['standalone_question']}"
        )
    trace = meta.get("agent_trace")
    if not trace:
        lines.append(f"**System:** {meta.get('pipeline', '?')}, no agent loop.")
        lines.append(f"**Retrieved chunks:** {len(result.retrieved)}")
        return "\n\n".join(lines)

    sub_queries = meta.get("sub_queries") or []
    lines.append(f"**Question type:** {meta.get('query_type', '?')}")
    if len(sub_queries) > 1:
        lines.append("**Sub-queries:**\n" + "\n".join(f"* {q}" for q in sub_queries))
    for round_ in trace.get("rounds", []):
        decision = round_.get("evidence_decision") or {}
        verdict = "sufficient" if decision.get("sufficient") else "insufficient"
        detail = f" ({decision['reason']})" if decision.get("reason") else ""
        queries = "; ".join(round_.get("queries", []))
        lines.append(
            f"**Round {round_.get('iteration')}:** {queries}\n\n"
            f"Evidence controller: {verdict if decision else 'not run'}{detail}"
            + (f", next: {decision['next_action']}" if decision.get("next_action") else "")
        )
    lines.append(f"**Stopped:** {trace.get('stop_reason', '?')}")
    lines.append(f"**Verifier:** {trace.get('verification_status', 'not_run')}")
    if meta.get("abstention_source"):
        lines.append(f"**Refused by:** {meta['abstention_source']}")
    return "\n\n".join(lines)


def respond(
    message: str, history: list[dict[str, str]] | None, state: ChatState
) -> tuple[list[dict[str, str]], ChatState, str]:
    """Answer one chat message; return the new history, state and trace."""
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        return history, state, ""
    history.append({"role": "user", "content": message})
    if not state.ready:
        history.append({"role": "assistant",
                        "content": "Upload a PDF and press **Index** first."})
        return history, state, ""
    try:
        result = state.session.ask(message)
    except RAGError as exc:
        history.append({"role": "assistant", "content": f"**{type(exc).__name__}.** {exc}"})
        return history, state, ""
    history.append({"role": "assistant", "content": format_answer(result, state.documents)})
    return history, state, format_trace(result)


def build_app(offline: bool = True, system: str = "agentic") -> Any:
    """The Gradio Blocks app. Gradio is imported here and nowhere else."""
    import gradio as gr  # noqa: PLC0415 - optional, Colab-only dependency

    chatbot_kwargs: dict[str, Any] = {"label": "Chat", "height": 480}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        chatbot_kwargs["type"] = "messages"  # Gradio 5; 6 uses messages only

    stack = "offline stand-in stack (CPU, no model weights)" if offline else "model stack"
    with gr.Blocks(title="Agentic PDF RAG") as app:
        state = gr.State(ChatState(system=system, offline=offline))
        gr.Markdown(
            "# Agentic PDF RAG\n"
            f"Running the **{stack}**. Upload PDFs (text or scanned), choose a "
            "system, press **Index**, then ask questions. Answers cite pages; "
            "a question the documents cannot answer is refused."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"])
                choice = gr.Radio(
                    choices=[(SYSTEM_LABELS[s], s) for s in SYSTEMS], value=system,
                    label="System",
                )
                index_button = gr.Button("Index", variant="primary")
                status = gr.Markdown("Upload one or more PDFs, then press **Index**.")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(**chatbot_kwargs)
                message = gr.Textbox(label="Question", placeholder="Ask about the documents")
                with gr.Accordion("Agent trace", open=False):
                    trace = gr.Markdown()

        def on_index(uploads, chosen, current):
            new_state, text = load_documents(uploads, chosen, current)
            return new_state, text, [], ""

        index_button.click(
            on_index, [files, choice, state], [state, status, chatbot, trace]
        )

        def on_message(text, history, current):
            new_history, new_state, trace_text = respond(text, history, current)
            return new_history, new_state, trace_text, ""

        message.submit(
            on_message, [message, chatbot, state], [chatbot, state, trace, message]
        )
    return app


def launch(offline: bool = True, share: bool = False, **kwargs: Any) -> Any:
    """Build and launch the app. In Colab, pass ``share=True`` for a public link."""
    return build_app(offline=offline).launch(share=share, **kwargs)
