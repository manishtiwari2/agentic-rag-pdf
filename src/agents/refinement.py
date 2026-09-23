"""Retrieval refinement (ARCHITECTURE.md section 13, DD-053).

When the evidence controller finds the context insufficient and the retrieval
budget allows another round, this builds the queries for that round. It is
deterministic in every configuration: turning "these terms are missing" into a
query is a construction, not a judgement (DD-018), and keeping it rule-based
means an LLM parse failure can never leave the loop without a next query.

Of section 13's four options it takes the first -- rewrite the query -- aimed
at the gap the controller measured. The targets are the queries whose coverage
fell below ``agents.sufficiency_threshold``, plus the original question when
the context lacked one of its numbers. Each target's refined query is its
missing content terms (and the missing numbers, for the original question),
led by up to two of the terms the context *did* cover, so the rewrite stays
anchored to the topic rather than retrieving the missing words in isolation.

A refined query whose terms match a query already issued is dropped. If
nothing new is left, the loop stops with ``no_new_query`` rather than spend a
round re-running a query it has already run.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..config import AgentConfig
from .evidence import EvidenceDecision
from .text import content_terms

#: Covered terms kept at the head of a refined query, as topic anchors.
ANCHOR_TERMS = 2


@runtime_checkable
class Refiner(Protocol):
    name: str

    def refine(
        self, question: str, decision: EvidenceDecision, issued: Sequence[str]
    ) -> list[str]: ...


class MissingTermsRefiner:
    name = "local/missing-terms-refiner"

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def refine(
        self, question: str, decision: EvidenceDecision, issued: Sequence[str]
    ) -> list[str]:
        threshold = self.config.sufficiency_threshold
        targets = [q for q, c in decision.coverage.items() if c < threshold]
        if decision.missing_numbers and question not in targets:
            targets.insert(0, question)

        seen = {_key(q) for q in issued}
        refined: list[str] = []
        for query in targets:
            lacking = decision.missing_terms.get(query, [])
            numbers = list(decision.missing_numbers) if query == question else []
            covered = [t for t in dict.fromkeys(content_terms(query)) if t not in lacking]
            terms = covered[:ANCHOR_TERMS] + lacking + numbers
            candidate = " ".join(dict.fromkeys(terms))
            if candidate and _key(candidate) not in seen:
                seen.add(_key(candidate))
                refined.append(candidate)
        return refined


def _key(query: str) -> tuple[str, ...]:
    return tuple(sorted(set(content_terms(query))))
