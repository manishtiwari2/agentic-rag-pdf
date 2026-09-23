"""Query planner (ARCHITECTURE.md section 11, DD-051).

The planner decides how a question is handled -- its type, and for a
``multi_hop`` or ``comparison`` question the sub-queries it is decomposed into
-- and never answers it. Its output is structured (``PlannerDecision``), which
is what makes a planner decision inspectable in the per-question trace.

Two implementations behind one interface:

* ``RuleBasedPlanner`` -- deterministic cue patterns and clause splitting. It is
  the offline stand-in (DD-055) and the fallback for the LLM planner.
* ``LLMPlanner`` -- asks the generation model for a JSON plan. Any output that
  does not parse into the schema is replaced by the rule-based plan, with the
  failure recorded rather than hidden: EXPERIMENT_PLAN.md section 4 expects a
  small model to get structured output wrong often enough to measure.

Three rules hold for both (DD-051):

* The original question is always ``sub_queries[0]``. Decomposition can only
  add queries, so it can add evidence but cannot lose the ranking the
  un-decomposed question would have produced.
* The number of queries is capped by ``agents.max_sub_queries``, and a
  truncation is recorded.
* The retrieval strategy is always executed as ``hybrid``. ARCHITECTURE.md
  section 11 lets the planner choose one; this phase keeps Baseline B's
  retrieval fixed so RQ3 measures decomposition and refinement rather than a
  strategy switch. An LLM's suggestion is recorded, not acted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..config import RULE_PLANNER, AgentConfig
from .text import JSONParseError, content_terms, extract_json, truncate

QUESTION_TYPES: tuple[str, ...] = (
    "factual",
    "definition",
    "comparison",
    "numerical",
    "multi_hop",
    "summary",
    "unanswerable",
)
#: The types decomposed into sub-queries: the two BENCHMARK_SPEC.md section 10
#: says need every gold page, so the two a single query is most likely to miss.
DECOMPOSED_TYPES: frozenset[str] = frozenset({"multi_hop", "comparison"})
RETRIEVAL_STRATEGIES: tuple[str, ...] = ("dense", "lexical", "hybrid")
#: What this phase executes, whatever a planner suggests (DD-051).
EXECUTED_STRATEGY = "hybrid"

#: A fragment needs this many content terms to be worth retrieving for alone:
#: one, so the far side of "X differ from Y" survives when Y is a single name.
MIN_FRAGMENT_TERMS = 1


@dataclass(frozen=True)
class PlannerDecision:
    """ARCHITECTURE.md section 11's structured output, plus its provenance."""

    query_type: str
    sub_queries: tuple[str, ...]
    requires_iteration: bool
    retrieval_strategy: str = EXECUTED_STRATEGY
    suggested_strategy: str | None = None
    backend: str = RULE_PLANNER
    parse_failed: bool = False
    parse_error: str | None = None
    raw: str | None = None
    truncated_sub_queries: int = 0
    model_calls: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "query_type": self.query_type,
            "retrieval_strategy": self.retrieval_strategy,
            "suggested_strategy": self.suggested_strategy,
            "sub_queries": list(self.sub_queries),
            "requires_iteration": self.requires_iteration,
            "truncated_sub_queries": self.truncated_sub_queries,
            "parse_failed": self.parse_failed,
            "parse_error": self.parse_error,
            "raw": self.raw,
            "model_calls": self.model_calls,
        }


@runtime_checkable
class Planner(Protocol):
    name: str

    def plan(self, question: str) -> PlannerDecision: ...


# ---------------------------------------------------------------------------
# Rule-based
# ---------------------------------------------------------------------------

#: Tried in order; first match wins. Comparison and multi-hop first, because a
#: comparison question often also asks "what", and the decomposition is the
#: decision that changes retrieval.
_TYPE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "comparison",
        re.compile(
            r"\b(?:compare[sd]?|comparison|differ(?:s|ed|ence|ences|ent)?|versus|vs\.?|"
            r"contrast|distinguish|better than|worse than|more than|less than|"
            r"similarit(?:y|ies)|in common)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "multi_hop",
        re.compile(
            r"\b(?:combining|combined with|using both|taken together|together with|"
            r"both the|and then|in turn|as well as)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "numerical",
        re.compile(
            r"\b(?:how many|how much|how long|how often|what (?:percentage|proportion|"
            r"fraction|number|share|rate|value)|how (?:large|big|small|fast|high|low))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "definition",
        re.compile(
            r"^(?:what (?:is|are|does) (?:meant by|an?|the)?|define|what does .* mean)\b|"
            r"\b(?:definition of|meaning of|refers to)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "summary",
        re.compile(
            r"\b(?:summari[sz]e|summary|overview|main (?:contribution|point|finding|idea)s?|"
            r"key (?:points|findings|takeaways))\b",
            re.IGNORECASE,
        ),
    ),
)

#: Clause boundaries a decomposable question is split on. Deliberately generic
#: English conjunction structure, not fitted to any benchmark question (DD-051).
_SPLIT = re.compile(
    r"(?:;|\?\s+|\.\s+|,\s+and\s+|\s+and\s+(?=(?:what|how|which|why|where|when|who)\b)|"
    r",\s+(?=(?:what|how|which|why|where|when|who)\b)|"
    r"\s+(?:versus|vs\.?|compared (?:to|with)|as opposed to|differ(?:s)? from|"
    r"in contrast to)\s+)",
    re.IGNORECASE,
)
_LEAD_IN = re.compile(
    r"^\s*(?:combining|using both|using|taken together|together|based on|"
    r"according to|given)\b[\s,]*",
    re.IGNORECASE,
)


def classify(question: str) -> str:
    """The rule-based question type. Never ``unanswerable``: rules cannot tell."""
    for question_type, cue in _TYPE_CUES:
        if cue.search(question):
            return question_type
    return "factual"


def decompose(question: str) -> list[str]:
    """Clause fragments of ``question`` with enough content to retrieve for."""
    fragments: list[str] = []
    seen: set[tuple[str, ...]] = {tuple(sorted(set(content_terms(question))))}
    for piece in _SPLIT.split(question):
        fragment = _LEAD_IN.sub("", piece or "").strip(" ,;:.?")
        terms = content_terms(fragment)
        if len(set(terms)) < MIN_FRAGMENT_TERMS:
            continue
        key = tuple(sorted(set(terms)))
        if key in seen:
            continue
        seen.add(key)
        fragments.append(fragment)
    return fragments


class RuleBasedPlanner:
    """Cue-pattern classification and clause splitting (DD-051)."""

    name = RULE_PLANNER

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def plan(self, question: str) -> PlannerDecision:
        question_type = classify(question)
        queries = [question.strip()]
        if question_type in DECOMPOSED_TYPES:
            queries += decompose(question)
        return _capped(
            PlannerDecision(
                query_type=question_type,
                sub_queries=tuple(queries),
                requires_iteration=question_type in DECOMPOSED_TYPES,
                backend=self.name,
            ),
            self.config.max_sub_queries,
        )


def _capped(decision: PlannerDecision, cap: int) -> PlannerDecision:
    if len(decision.sub_queries) <= cap:
        return decision
    return replace(
        decision,
        sub_queries=decision.sub_queries[:cap],
        truncated_sub_queries=len(decision.sub_queries) - cap,
    )


# ---------------------------------------------------------------------------
# Model-backed
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You plan how to search a single document for the answer to a question. You never answer the question.

Reply with one JSON object and nothing else, with exactly these keys:
  "question_type": one of {list(QUESTION_TYPES)}
  "sub_queries": a list of short search queries. Use [] unless the question needs two or more separate pieces of evidence (multi_hop or comparison); then give one query per piece.
  "retrieval_strategy": one of {list(RETRIEVAL_STRATEGIES)}
  "requires_iteration": true or false"""


def _planner_prompt(question: str) -> str:
    return f"QUESTION\n{question.strip()}\n\nJSON:"


class LLMPlanner:
    """Asks the generation model for a JSON plan; falls back to the rules."""

    def __init__(self, backend, config: AgentConfig | None = None, label: str = "llm") -> None:
        self.backend = backend
        self.config = config or AgentConfig()
        self.name = label
        self._fallback = RuleBasedPlanner(self.config)

    def plan(self, question: str) -> PlannerDecision:
        raw = ""
        try:
            raw = self.backend.generate(PLANNER_SYSTEM_PROMPT, _planner_prompt(question))
            decision = _parse_plan(question, raw, self.name)
        except (JSONParseError, ValueError, TypeError) as exc:
            fallback = self._fallback.plan(question)
            return replace(
                fallback,
                backend=f"{self._fallback.name} (fallback from {self.name})",
                parse_failed=True,
                parse_error=str(exc),
                raw=truncate(raw),
                model_calls=1,
            )
        return _capped(replace(decision, model_calls=1), self.config.max_sub_queries)


def _parse_plan(question: str, raw: str, backend: str) -> PlannerDecision:
    data = extract_json(raw)
    question_type = data.get("question_type")
    if question_type not in QUESTION_TYPES:
        raise JSONParseError(f"question_type {question_type!r} is not one of {QUESTION_TYPES}")
    sub_queries = data.get("sub_queries", [])
    if not isinstance(sub_queries, list) or not all(isinstance(q, str) for q in sub_queries):
        raise JSONParseError("sub_queries is not a list of strings")
    strategy = data.get("retrieval_strategy", EXECUTED_STRATEGY)
    if strategy not in RETRIEVAL_STRATEGIES:
        raise JSONParseError(f"retrieval_strategy {strategy!r} is not one of {RETRIEVAL_STRATEGIES}")
    requires = data.get("requires_iteration", False)
    if not isinstance(requires, bool):
        raise JSONParseError("requires_iteration is not a boolean")

    queries = [question.strip()]
    seen = {question.strip().lower()}
    for query in sub_queries:
        query = query.strip()
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)
    return PlannerDecision(
        query_type=question_type,
        sub_queries=tuple(queries),
        requires_iteration=requires,
        suggested_strategy=strategy,
        backend=backend,
        raw=truncate(raw),
    )


def build_planner(config: AgentConfig, backend=None, label: str = "llm") -> Planner:
    if config.planner == "llm":
        return LLMPlanner(backend, config, label=label)
    return RuleBasedPlanner(config)


__all__ = [
    "DECOMPOSED_TYPES",
    "LLMPlanner",
    "Planner",
    "PlannerDecision",
    "RuleBasedPlanner",
    "build_planner",
    "classify",
    "decompose",
]

