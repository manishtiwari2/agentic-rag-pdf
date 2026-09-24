"""Multi-turn conversation over one indexed pipeline (DD-065).

The pipeline answers one self-contained question at a time, and that is what
the benchmark measures. A chat user asks "and what about its latency?", which
retrieves nothing useful on its own. ``ChatSession`` sits above the pipeline:
it keeps the last few turns, rewrites a follow-up into a standalone question,
and asks the pipeline that. The pipeline never sees the history, so a
conversation cannot change what a single question retrieves or how it is
answered.

Two rewriters, chosen by what is loaded:

* ``LLMRewriter`` asks the generation model the pipeline already loaded, via
  ``pipeline.backend``. No second model is loaded (ARCHITECTURE.md section 21).
* ``RuleBasedRewriter`` is the deterministic offline stand-in, the same role
  the rule-based agents play for DD-055. The scripted backend cannot rewrite:
  any prompt not in its evidence format gets the refusal sentence back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ..agents.text import content_terms
from ..generation.backends import LLMBackend, ScriptedBackend
from ..generation.prompts import ABSTENTION_SENTENCE
from ..pipeline import RAGResult

#: Turns kept for rewriting. Older turns rarely resolve a pronoun, and every
#: kept turn lengthens the LLM rewriter's prompt.
DEFAULT_MAX_TURNS = 5

#: Words that point back at an earlier turn.
_REFERRING = frozenset(
    """it its it's they them their theirs this that these those he him his she
    her there such same former latter one ones""".split()
)
#: Openings that continue the previous question rather than start a new one.
_CONTINUATIONS = ("and ", "also ", "what about", "how about", "then ", "so ", "but ")
#: A question with fewer content terms than this leans on the conversation.
_MIN_STANDALONE_TERMS = 3
_WORD = re.compile(r"[a-z']+")
_MARKERS = re.compile(r"\s*\[C\d+(?:, C\d+)*\]")


@dataclass(frozen=True)
class Turn:
    question: str
    standalone: str
    answer: str


@dataclass(frozen=True)
class Rewrite:
    standalone: str
    rewriter: str
    #: The LLM rewriter's output was unusable and the rules answered instead.
    fallback: bool = False
    model_calls: int = 0


class Rewriter(Protocol):
    name: str

    def rewrite(self, question: str, history: list[Turn]) -> Rewrite: ...


def is_follow_up(question: str) -> bool:
    """Whether ``question`` leans on an earlier turn to be understood."""
    lowered = question.strip().lower()
    words = set(_WORD.findall(lowered))
    return (
        bool(words & _REFERRING)
        or lowered.startswith(_CONTINUATIONS)
        or len(set(content_terms(question))) < _MIN_STANDALONE_TERMS
    )


class RuleBasedRewriter:
    """Append the previous standalone question to a follow-up.

    "What about its latency?" after "What recall did the hybrid system
    achieve?" becomes "What about its latency? (What recall did the hybrid
    system achieve?)". Crude, but retrieval then sees the topic's terms, and the
    output is deterministic, so offline tests can pin it.
    """

    name = "rules"

    def rewrite(self, question: str, history: list[Turn]) -> Rewrite:
        if not history or not is_follow_up(question):
            return Rewrite(standalone=question, rewriter=self.name)
        return Rewrite(
            standalone=f"{question.strip()} ({history[-1].standalone})",
            rewriter=self.name,
        )


REWRITE_SYSTEM_PROMPT = """You rewrite the user's latest question from a conversation about a document into one standalone question.

Rules:
1. Replace pronouns and references ("it", "that method", "the second one") with what they refer to in the conversation.
2. Do not answer the question, and do not add facts that are not in the conversation.
3. If the question is already standalone, return it unchanged.
4. Output only the rewritten question, on one line."""


def _rewrite_prompt(question: str, history: list[Turn]) -> str:
    lines = ["CONVERSATION"]
    for turn in history:
        answer = _MARKERS.sub("", turn.answer).strip()
        if len(answer) > 300:
            answer = answer[:300].rstrip() + " ..."
        lines.append(f"User: {turn.standalone}")
        lines.append(f"Assistant: {answer}")
    lines.extend(["", "LATEST QUESTION", question.strip(), "", "Standalone question:"])
    return "\n".join(lines)


class LLMRewriter:
    """Rewrite with the generation model, falling back to the rules.

    A reply that is empty, the refusal sentence, or far longer than the
    question is not a question, and the rule-based rewrite is used instead,
    marked ``fallback``. As with the LLM agents (DD-055), a failure is recorded
    rather than hidden.
    """

    name = "llm"

    def __init__(self, backend: LLMBackend, fallback: Rewriter | None = None) -> None:
        self.backend = backend
        self.fallback = fallback or RuleBasedRewriter()

    def rewrite(self, question: str, history: list[Turn]) -> Rewrite:
        if not history:
            return Rewrite(standalone=question, rewriter=self.name)
        raw = self.backend.generate(REWRITE_SYSTEM_PROMPT, _rewrite_prompt(question, history))
        candidate = _first_line(raw)
        if (
            not candidate
            or candidate == ABSTENTION_SENTENCE
            or len(candidate) > 3 * len(question) + 200
        ):
            rewrite = self.fallback.rewrite(question, history)
            return replace(rewrite, rewriter=self.name, fallback=True, model_calls=1)
        return Rewrite(standalone=candidate, rewriter=self.name, model_calls=1)


def _first_line(raw: str) -> str:
    for line in (raw or "").splitlines():
        line = line.strip().strip('"').strip()
        if line.lower().startswith("standalone question:"):
            line = line.split(":", 1)[1].strip()
        if line:
            return line
    return ""


def build_rewriter(pipeline: Any) -> Rewriter:
    """The LLM rewriter when a generation model is loaded, the rules otherwise."""
    backend = pipeline.backend
    if isinstance(backend, ScriptedBackend):
        return RuleBasedRewriter()
    return LLMRewriter(backend)


class ChatSession:
    """A conversation with one indexed pipeline.

    The first turn is passed to ``pipeline.ask`` unchanged, so a one-question
    chat answers exactly as the benchmark would. The result's metadata records
    both questions under ``conversation``.
    """

    def __init__(
        self, pipeline: Any, rewriter: Rewriter | None = None, max_turns: int = DEFAULT_MAX_TURNS
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.pipeline = pipeline
        self.rewriter = rewriter or build_rewriter(pipeline)
        self.max_turns = max_turns
        self.history: list[Turn] = []
        self.turns = 0

    def ask(self, question: str) -> RAGResult:
        rewrite = self.rewriter.rewrite(question, list(self.history))
        result = self.pipeline.ask(rewrite.standalone)
        self.turns += 1
        self.history.append(
            Turn(question=question, standalone=rewrite.standalone, answer=result.answer)
        )
        del self.history[: -self.max_turns]
        conversation = {
            "turn": self.turns,
            "original_question": question,
            "standalone_question": rewrite.standalone,
            "rewritten": rewrite.standalone != question,
            "rewriter": rewrite.rewriter,
            "rewriter_fallback": rewrite.fallback,
            "rewriter_model_calls": rewrite.model_calls,
        }
        return replace(result, metadata={**result.metadata, "conversation": conversation})

    def reset(self) -> None:
        self.history.clear()
        self.turns = 0
