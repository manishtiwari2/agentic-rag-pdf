"""Explicit agent state (ARCHITECTURE.md section 17, DD-011).

One ``AgentState`` per question, created by ``ask`` and discarded after it: no
state survives on the pipeline between questions, and none hides inside a
prompt. Every decision a component makes is appended here as it is made, and
``to_record`` renders the section 23 trace --

    query -> planner decision -> retrieval strategy -> candidates -> reranking
          -> evidence decision -> generation -> verification -> final response

-- in that order, into the per-question record, so a reader of
``per_question.json`` can replay what the system did and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalRound:
    """One iteration of the retrieval loop."""

    iteration: int
    queries: list[str]
    #: ``{query: fused (pre-rerank) chunk ids}`` and ``{query: reranked ids}``.
    candidates: dict[str, list[str]] = field(default_factory=dict)
    reranked: dict[str, list[str]] = field(default_factory=dict)
    merged_chunk_ids: list[str] = field(default_factory=list)
    evidence_decision: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "queries": list(self.queries),
            "candidates": self.candidates,
            "reranked": self.reranked,
            "merged_chunk_ids": self.merged_chunk_ids,
            "evidence_decision": self.evidence_decision,
        }


@dataclass
class AgentState:
    """Section 17's conceptual state, made concrete."""

    query: str
    query_type: str | None = None
    retrieval_strategy: str = "hybrid"
    query_variants: list[str] = field(default_factory=list)
    iteration: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    reranked_chunk_ids: list[str] = field(default_factory=list)
    selected_context: list[str] = field(default_factory=list)
    evidence_status: str = "not_assessed"
    draft_answer: str | None = None
    verification_status: str = "not_run"
    citations: list[dict[str, Any]] = field(default_factory=list)

    planner: dict[str, Any] | None = None
    rounds: list[RetrievalRound] = field(default_factory=list)
    stop_reason: str | None = None
    generations: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] | None = None
    bounds: dict[str, Any] = field(default_factory=dict)

    #: Calls to the generation model made by the system, by component. The
    #: diagnostic counterfactual is counted apart: it is instrumentation.
    model_calls: dict[str, int] = field(
        default_factory=lambda: {"planner": 0, "evidence_controller": 0, "generator": 0, "verifier": 0}
    )
    diagnostic_model_calls: int = 0
    reranker_calls: int = 0
    llm_decisions: int = 0
    llm_parse_failures: int = 0

    def count(self, component: str, calls: int, llm: bool, parse_failed: bool = False) -> None:
        """Record one decision's model calls and, for an LLM one, its parse outcome."""
        self.model_calls[component] = self.model_calls.get(component, 0) + calls
        if llm:
            self.llm_decisions += 1
            self.llm_parse_failures += int(parse_failed)

    @property
    def total_model_calls(self) -> int:
        return sum(self.model_calls.values())

    def to_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "planner": self.planner,
            "query_type": self.query_type,
            "retrieval_strategy": self.retrieval_strategy,
            "query_variants": list(self.query_variants),
            "rounds": [r.to_record() for r in self.rounds],
            "iterations": self.iteration,
            "stop_reason": self.stop_reason,
            "evidence_status": self.evidence_status,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "reranked_chunk_ids": self.reranked_chunk_ids,
            "selected_context": self.selected_context,
            "generations": self.generations,
            "draft_answer": self.draft_answer,
            "verifications": self.verifications,
            "verification_status": self.verification_status,
            "citations": self.citations,
            "final": self.final,
            "refinement_counterfactual": self.counterfactual,
            "model_calls": dict(self.model_calls),
            "diagnostic_model_calls": self.diagnostic_model_calls,
            "reranker_calls": self.reranker_calls,
            "llm_decisions": self.llm_decisions,
            "llm_parse_failures": self.llm_parse_failures,
            "bounds": self.bounds,
        }
