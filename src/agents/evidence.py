"""Evidence controller (ARCHITECTURE.md section 12, DD-052).

Decides whether the selected context is sufficient to answer from, and so
whether to answer, retrieve again, or abstain. Its output is section 12's
``sufficient / confidence / reason / next_action``, plus the per-query coverage
and missing terms the refinement step builds its queries from.

The controller *proposes* ``next_action``. The loop in the pipeline disposes:
``retrieve_again`` is honoured only while the retrieval budget lasts, and turns
into ``abstain`` at the cap whatever the controller said (DD-012, DD-052). An
LLM controller that asks for another round forever therefore still terminates.

Rule-based sufficiency (the offline stand-in, DD-055), stated in full because
it is the decision the abstention figures depend on:

* for every query the planner issued -- the original question and each
  sub-query -- the union of the selected chunks' text must contain at least
  ``agents.sufficiency_threshold`` of the query's distinct content terms,
  compared as crude suffix-stripped word forms (``text.stem``); and
* every number in the original question must appear in that text.

Insufficient with budget left means ``retrieve_again``; insufficient with none
means ``abstain`` (DD-014: "If evidence is insufficient, the system should
abstain"). The threshold is a round number chosen before any agentic result
existed and is not tuned (EXPERIMENT_PLAN.md section 5); a sweep belongs on
``--split dev`` in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence, runtime_checkable

from ..config import RULE_EVIDENCE_CONTROLLER, AgentConfig
from ..retrieval.retriever import RetrievedChunk
from .text import JSONParseError, content_terms, extract_json, numbers_in, stem, stems, truncate

ANSWER = "answer"
RETRIEVE_AGAIN = "retrieve_again"
ABSTAIN = "abstain"
NEXT_ACTIONS: tuple[str, ...] = (ANSWER, RETRIEVE_AGAIN, ABSTAIN)


@dataclass(frozen=True)
class EvidenceDecision:
    sufficient: bool
    confidence: float
    reason: str
    next_action: str
    #: ``{query: share of its content terms the context contains}``.
    coverage: dict[str, float] = field(default_factory=dict)
    #: ``{query: its content terms the context lacks}``, what refinement uses.
    missing_terms: dict[str, list[str]] = field(default_factory=dict)
    missing_numbers: tuple[str, ...] = ()
    backend: str = RULE_EVIDENCE_CONTROLLER
    parse_failed: bool = False
    parse_error: str | None = None
    raw: str | None = None
    model_calls: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sufficient": self.sufficient,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "next_action": self.next_action,
            "coverage": {q: round(c, 4) for q, c in self.coverage.items()},
            "missing_terms": self.missing_terms,
            "missing_numbers": list(self.missing_numbers),
            "parse_failed": self.parse_failed,
            "parse_error": self.parse_error,
            "raw": self.raw,
            "model_calls": self.model_calls,
        }


@runtime_checkable
class EvidenceController(Protocol):
    name: str

    def assess(
        self,
        question: str,
        queries: Sequence[str],
        context: Sequence[RetrievedChunk],
        can_retrieve_again: bool,
    ) -> EvidenceDecision: ...


def _propose(sufficient: bool, can_retrieve_again: bool) -> str:
    if sufficient:
        return ANSWER
    return RETRIEVE_AGAIN if can_retrieve_again else ABSTAIN


class RuleBasedEvidenceController:
    """Term coverage of every issued query, plus exact numbers (DD-052)."""

    name = RULE_EVIDENCE_CONTROLLER

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def assess(
        self,
        question: str,
        queries: Sequence[str],
        context: Sequence[RetrievedChunk],
        can_retrieve_again: bool,
    ) -> EvidenceDecision:
        text = "\n".join(item.chunk.text for item in context)
        present = stems(text)
        coverage: dict[str, float] = {}
        missing: dict[str, list[str]] = {}
        for query in dict.fromkeys(queries or [question]):
            terms = list(dict.fromkeys(content_terms(query)))
            if not terms:
                coverage[query] = 1.0
                continue
            lacking = [t for t in terms if stem(t) not in present]
            coverage[query] = (len(terms) - len(lacking)) / len(terms)
            if lacking:
                missing[query] = lacking
        missing_numbers = tuple(sorted(numbers_in(question) - numbers_in(text)))

        threshold = self.config.sufficiency_threshold
        short = [q for q, c in coverage.items() if c < threshold]
        sufficient = bool(context) and not short and not missing_numbers
        if not context:
            reason = "no context was retrieved"
        elif sufficient:
            reason = f"every query has coverage >= {threshold} and every question number is present"
        else:
            parts = []
            if short:
                parts.append(f"{len(short)} of {len(coverage)} queries below coverage {threshold}")
            if missing_numbers:
                parts.append(f"question numbers absent from context: {list(missing_numbers)}")
            reason = "; ".join(parts)
        return EvidenceDecision(
            sufficient=sufficient,
            confidence=min(coverage.values()) if coverage and context else 0.0,
            reason=reason,
            next_action=_propose(sufficient, can_retrieve_again),
            coverage=coverage,
            missing_terms=missing,
            missing_numbers=missing_numbers,
            backend=self.name,
        )


# ---------------------------------------------------------------------------
# Model-backed
# ---------------------------------------------------------------------------

EVIDENCE_SYSTEM_PROMPT = f"""You judge whether numbered passages from a document contain enough information to answer a question. You never answer the question.

Reply with one JSON object and nothing else, with exactly these keys:
  "sufficient": true or false
  "confidence": a number from 0 to 1
  "reason": one short sentence
  "next_action": one of {list(NEXT_ACTIONS)}"""


def _evidence_prompt(question: str, context: Sequence[RetrievedChunk]) -> str:
    blocks = "\n\n".join(
        f"[{number}]\n{item.chunk.text.strip()}" for number, item in enumerate(context, 1)
    )
    return f"PASSAGES\n{blocks}\n\nQUESTION\n{question.strip()}\n\nJSON:"


class LLMEvidenceController:
    """Asks the generation model; falls back to the rules on unusable output.

    The rule-based coverage is always computed as well: refinement needs the
    missing terms whichever controller decided, and the rules are the fallback.
    """

    def __init__(self, backend, config: AgentConfig | None = None, label: str = "llm") -> None:
        self.backend = backend
        self.config = config or AgentConfig()
        self.name = label
        self._rules = RuleBasedEvidenceController(self.config)

    def assess(
        self,
        question: str,
        queries: Sequence[str],
        context: Sequence[RetrievedChunk],
        can_retrieve_again: bool,
    ) -> EvidenceDecision:
        rules = self._rules.assess(question, queries, context, can_retrieve_again)
        if not context:
            # Nothing to show the model; the rules' "no context" is the answer.
            return rules
        raw = ""
        try:
            raw = self.backend.generate(EVIDENCE_SYSTEM_PROMPT, _evidence_prompt(question, context))
            data = extract_json(raw)
            sufficient = data.get("sufficient")
            confidence = data.get("confidence")
            reason = data.get("reason", "")
            if not isinstance(sufficient, bool):
                raise JSONParseError("sufficient is not a boolean")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
                raise JSONParseError("confidence is not a number in [0, 1]")
            if not isinstance(reason, str):
                raise JSONParseError("reason is not a string")
            if data.get("next_action") not in NEXT_ACTIONS:
                raise JSONParseError(f"next_action {data.get('next_action')!r} is not one of {NEXT_ACTIONS}")
        except (JSONParseError, ValueError, TypeError) as exc:
            return replace(
                rules,
                backend=f"{self._rules.name} (fallback from {self.name})",
                parse_failed=True,
                parse_error=str(exc),
                raw=truncate(raw),
                model_calls=1,
            )
        return replace(
            rules,
            sufficient=sufficient,
            confidence=float(confidence),
            reason=reason,
            # The model's own next_action is recorded in `raw`; the proposal is
            # derived from its sufficiency verdict so the budget rule is the
            # same for both controllers.
            next_action=_propose(sufficient, can_retrieve_again),
            backend=self.name,
            raw=truncate(raw),
            model_calls=1,
        )


def build_evidence_controller(config: AgentConfig, backend=None, label: str = "llm") -> EvidenceController:
    if config.evidence_controller == "llm":
        return LLMEvidenceController(backend, config, label=label)
    return RuleBasedEvidenceController(config)
