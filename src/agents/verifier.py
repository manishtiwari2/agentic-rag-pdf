"""Answer verification (ARCHITECTURE.md section 16, DD-054).

Checks a generated answer against the evidence it cites and returns section
16's ``SUPPORTED / UNSUPPORTED / UNCERTAIN``, with the unsupported claims named.

The recorded status follows ARCHITECTURE.md section 24 to the letter:

* ``passed`` only for ``SUPPORTED``;
* ``failed`` for ``UNSUPPORTED`` or ``UNCERTAIN`` -- a verifier that cannot
  confirm support has not passed the answer;
* ``unavailable`` when the verifier itself failed -- it raised, or an LLM
  verifier's output did not parse. It is **never** reported as ``passed``, and
  there is deliberately no fallback to the rules here: substituting a weaker
  check for the one that failed would report a verification that did not
  happen (DD-054, DD-055).

Rule-based verification (the offline stand-in, DD-055). Each substantive
sentence of the answer is a claim, and a claim is supported when:

1. it carries at least one marker resolving to an evidence item -- its own,
   or the marker closing the run of unmarked sentences it belongs to, the
   usual convention for a citation at the end of a passage (the scripted
   backend writes exactly that: two sentences, one marker);
2. every number in it appears in the evidence it cites -- exact figures are
   the claims a small model gets wrong most often, and EVALUATION_PROTOCOL.md
   section 19.2 keeps them out of any model's hands; and
3. at least ``agents.verifier_support_threshold`` of its content terms appear
   in the evidence it cites.

Checked against the *cited* evidence, not the whole context: a claim grounded
in a passage it does not cite is still a claim pointing its reader at the wrong
page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from ..config import RULE_VERIFIER, AgentConfig
from ..generation.evidence import EvidenceItem
from .text import (
    JSONParseError,
    cited_numbers,
    content_terms,
    extract_json,
    numbers_in,
    sentences,
    stems,
    truncate,
)

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
UNCERTAIN = "UNCERTAIN"
VERDICTS: tuple[str, ...] = (SUPPORTED, UNSUPPORTED, UNCERTAIN)

#: The record's ``verification_status`` values -- the same strings the
#: evaluation layer's ``VERIFICATION_*`` constants hold, which a test checks.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NOT_RUN = "not_run"


@dataclass(frozen=True)
class Verification:
    verdict: str | None
    status: str
    unsupported_claims: tuple[str, ...] = ()
    #: Evidence numbers the unsupported claims cite; what a regeneration drops.
    blamed_evidence: tuple[int, ...] = ()
    reason: str = ""
    backend: str = RULE_VERIFIER
    raw: str | None = None
    error: str | None = None
    model_calls: int = 0
    claims: list[dict[str, object]] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "verdict": self.verdict,
            "status": self.status,
            "reason": self.reason,
            "unsupported_claims": list(self.unsupported_claims),
            "blamed_evidence": list(self.blamed_evidence),
            "claims": self.claims,
            "raw": self.raw,
            "error": self.error,
            "model_calls": self.model_calls,
        }


def status_for(verdict: str) -> str:
    return STATUS_PASSED if verdict == SUPPORTED else STATUS_FAILED


@runtime_checkable
class Verifier(Protocol):
    name: str

    def verify(
        self, question: str, answer: str, evidence: Sequence[EvidenceItem]
    ) -> Verification: ...


class RuleBasedVerifier:
    """Per-claim marker, number and term support against cited evidence."""

    name = RULE_VERIFIER

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def verify(
        self, question: str, answer: str, evidence: Sequence[EvidenceItem]
    ) -> Verification:
        by_number = {item.number: item for item in evidence}
        threshold = self.config.verifier_support_threshold
        claims: list[dict[str, object]] = []
        unsupported: list[str] = []
        blamed: list[int] = []

        for sentence, markers in _claims_with_markers(answer):
            terms = stems(sentence)
            cited = [n for n in markers if n in by_number]
            cited_text = "\n".join(by_number[n].text for n in cited)
            missing_numbers = sorted(numbers_in(sentence) - numbers_in(cited_text))
            support = (
                len(terms & stems(cited_text)) / len(terms) if cited else 0.0
            )
            problems = []
            if not cited:
                problems.append("no resolvable citation")
            if cited and missing_numbers:
                problems.append(f"numbers not in cited evidence: {missing_numbers}")
            if cited and support < threshold:
                problems.append(f"term support {support:.2f} < {threshold}")
            claims.append(
                {
                    "claim": sentence,
                    "cited": cited,
                    "term_support": round(support, 4),
                    "missing_numbers": missing_numbers,
                    "supported": not problems,
                    "problems": problems,
                }
            )
            if problems:
                unsupported.append(sentence)
                blamed.extend(n for n in cited if n not in blamed)

        if not claims:
            return Verification(
                verdict=UNCERTAIN,
                status=status_for(UNCERTAIN),
                reason="the answer contains no substantive claim to check",
                backend=self.name,
            )
        verdict = UNSUPPORTED if unsupported else SUPPORTED
        return Verification(
            verdict=verdict,
            status=status_for(verdict),
            unsupported_claims=tuple(unsupported),
            blamed_evidence=tuple(blamed),
            reason=(
                f"{len(unsupported)} of {len(claims)} claims unsupported"
                if unsupported
                else f"all {len(claims)} claims supported by their cited evidence"
            ),
            backend=self.name,
            claims=claims,
        )


# ---------------------------------------------------------------------------
# Model-backed
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = f"""You check whether an answer is supported by numbered passages from a document. Judge only against the passages, never against outside knowledge.

Reply with one JSON object and nothing else, with exactly these keys:
  "verdict": one of {list(VERDICTS)}
  "unsupported_claims": a list of the sentences of the answer the passages do not support ([] if none)"""


def _verifier_prompt(question: str, answer: str, evidence: Sequence[EvidenceItem]) -> str:
    blocks = "\n\n".join(f"[C{item.number}]\n{item.text.strip()}" for item in evidence)
    return (
        f"PASSAGES\n{blocks}\n\nQUESTION\n{question.strip()}\n\n"
        f"ANSWER\n{answer.strip()}\n\nJSON:"
    )


class LLMVerifier:
    """Asks the generation model. Unusable output is ``unavailable``, never ``passed``."""

    def __init__(self, backend, config: AgentConfig | None = None, label: str = "llm") -> None:
        self.backend = backend
        self.config = config or AgentConfig()
        self.name = label

    def verify(
        self, question: str, answer: str, evidence: Sequence[EvidenceItem]
    ) -> Verification:
        raw = ""
        try:
            raw = self.backend.generate(
                VERIFIER_SYSTEM_PROMPT, _verifier_prompt(question, answer, evidence)
            )
            data = extract_json(raw)
            verdict = data.get("verdict")
            if verdict not in VERDICTS:
                raise JSONParseError(f"verdict {verdict!r} is not one of {VERDICTS}")
            claims = data.get("unsupported_claims", [])
            if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
                raise JSONParseError("unsupported_claims is not a list of strings")
        except (JSONParseError, ValueError, TypeError) as exc:
            return unavailable(self.name, exc, raw=raw, model_calls=1)
        blamed = [
            n
            for claim in claims
            for n in cited_numbers(_matching_sentence(answer, claim))
        ]
        return Verification(
            verdict=verdict,
            status=status_for(verdict),
            unsupported_claims=tuple(claims),
            blamed_evidence=tuple(dict.fromkeys(blamed)),
            reason=f"model verdict {verdict}",
            backend=self.name,
            raw=truncate(raw),
            model_calls=1,
        )


def _claims_with_markers(answer: str) -> list[tuple[str, list[int]]]:
    """Substantive sentences with the markers that cover them.

    A sentence without a marker is covered by the next marker in the answer,
    so ``"A. B. [C1]"`` cites C1 for both. Unmarked sentences after the last
    marker stay uncited.
    """
    claims: list[tuple[str, list[int]]] = []
    waiting: list[int] = []
    for sentence in sentences(answer):
        markers = cited_numbers(sentence)
        if content_terms(sentence):
            claims.append((sentence, markers))
            if not markers:
                waiting.append(len(claims) - 1)
                continue
        if markers:
            # A marker on its own ("... index. [C1]") closes the run too.
            for index in waiting:
                claims[index] = (claims[index][0], markers)
            waiting = []
    return claims


def _matching_sentence(answer: str, claim: str) -> str:
    """The answer sentence a model-quoted claim came from, markers included."""
    wanted = set(content_terms(claim))
    best, best_overlap = claim, 0
    for sentence in sentences(answer):
        overlap = len(wanted & set(content_terms(sentence)))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best


def unavailable(backend: str, exc: BaseException, raw: str = "", model_calls: int = 0) -> Verification:
    """ARCHITECTURE.md section 24: a verifier failure is never a pass."""
    return Verification(
        verdict=None,
        status=STATUS_UNAVAILABLE,
        reason="the verifier could not produce a verdict",
        backend=backend,
        raw=truncate(raw) if raw else None,
        error=f"{type(exc).__name__}: {exc}",
        model_calls=model_calls,
    )


def build_verifier(config: AgentConfig, backend=None, label: str = "llm") -> Verifier:
    if config.verifier == "llm":
        return LLMVerifier(backend, config, label=label)
    return RuleBasedVerifier(config)
