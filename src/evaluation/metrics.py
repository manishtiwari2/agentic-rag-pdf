"""Benchmark scoring (BENCHMARK_SPEC.md sections 10-14, EVALUATION_PROTOCOL.md 17-23).

Every number this module produces is computed by deterministic code. No model is
consulted here, which is DD-018 applied to measurement: a metric decided by the
system being measured is not a measurement. The optional LLM judge lives in
``judge.py`` and is reported *beside* these figures, never instead of them
(EVALUATION_PROTOCOL.md section 19.1).

Three rules in here are easy to get subtly wrong and expensive to discover late,
so each is stated where it is implemented:

* **Relevance is page-level** (DD-025). A chunk is relevant when any page it
  covers is in ``evidence_pages``; a chunk spanning a page break is credited for
  every page it covers.
* **Unanswerable questions are excluded from retrieval metrics**, never scored
  0.0. They have no gold evidence, so scoring them as retrieval failures would
  make a correctly-abstaining system look broken. The exclusion is recorded in
  the per-question record (``scored: false``) rather than being invisible in an
  aggregate.
* **Over-abstention is measured over the ANSWERABLE set.** Its denominator is
  the whole point: every other abstention metric is computed over the
  unanswerable set, where a system that refuses everything scores perfectly.

The scoring rule set is versioned for the same reason the abstention patterns
are (BENCHMARK_SPEC.md section 13.2): two runs scored under different rules must
never be compared as though they were not.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..benchmark.schema import Question
from ..generation.abstention import ABSTENTION_PATTERNS_VERSION, is_abstention

#: Bump when any formula, threshold or taxonomy rule below changes. Recorded
#: with every result set.
#:
#: 2026-09-23.1: taxonomy rule 2 no longer claims a gold page a declared
#: reranker pushed out (DD-044). No formula or threshold changed, and for a
#: record that declares no reranking stage every rule is as before -- a test
#: re-runs Baseline A and requires its stored per-question metrics exactly.
SCORING_RULES_VERSION = "2026-09-23.1"

#: Earlier versions whose per-question *scores* (not error categories) are
#: computed identically to this one, so a paired comparison across them is
#: valid. Each entry needs the reason and the test that establishes it.
SCORE_COMPATIBLE_VERSIONS: dict[str, str] = {
    "2026-09-21.1": (
        "taxonomy-only change (DD-044); tests/test_hybrid.py::"
        "TestBaselineAIsUnchanged reproduces results/baseline/ metrics exactly"
    ),
}

# ---------------------------------------------------------------------------
# Retrieval (BENCHMARK_SPEC.md section 10)
# ---------------------------------------------------------------------------

#: "1.0 if ANY gold evidence page is in the top K."
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)
#: "1.0 if EVERY gold evidence page is in the top K."
FULL_RECALL_KS: tuple[int, ...] = (5, 10)
MRR_K = 10
NDCG_K = 10

#: BENCHMARK_SPEC.md section 10: Full-Recall@5 is the *primary* metric for these
#: two categories, because a system that finds one hop and never the second
#: scores 1.0 on plain recall and 0.0 here.
FULL_RECALL_REQUIRED_TYPES: frozenset[str] = frozenset({"multi_hop", "comparison"})

# ---------------------------------------------------------------------------
# Answer scoring thresholds
# ---------------------------------------------------------------------------

#: Share of the reference answer's content tokens that must appear in the system
#: answer before deterministic correctness counts it as correct. Chosen before
#: any results existed, for the reason EXPERIMENT_PLAN.md section 3 gives about
#: thresholds: one picked after seeing the numbers is not a threshold. It is
#: recorded in every result set so a later change is visible as a change.
CORRECTNESS_OVERLAP_THRESHOLD = 0.6

#: Tokens carrying no discriminative content. Deliberately a *separate* list
#: from the one in ``src/generation/backends.py``: a scorer that shares the
#: system's tokenizer flatters the system, and the scripted backend is one of
#: the systems this module scores.
_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being by can could did do does doing done for from
    had has have having he her hers him his how i if in into is it its me my no nor not
    of on or our ours out over own she should so some such than that the their theirs
    them then there these they this those to too under up us was we were what when
    where which while who whom why will with would you your yours""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
#: Evidence markers are the harness's own annotation, not answer content.
_MARKER = re.compile(r"\[[Cc]?\s*\d+(?:\s*[,;]\s*[Cc]?\s*\d+)*\]")
#: Numbers as a reader would see them: 1,200  86.7  9.4%  -3
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def content_tokens(text: str) -> list[str]:
    """Lower-cased content tokens, markers and stopwords removed."""
    stripped = _MARKER.sub(" ", text or "")
    return [
        token
        for token in _TOKEN.findall(stripped.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


def numbers(text: str) -> set[str]:
    """Distinct numbers in ``text``, normalized so ``1,200`` == ``1200``.

    Normalization stops short of arithmetic. EVALUATION_PROTOCOL.md section 19.2
    puts numeric agreement out of the judge's reach precisely because telling
    86.7 from 87.6 is the comparison that matters most and the one a small model
    is least reliable at, so it is a string comparison over a canonical form and
    nothing cleverer.
    """
    found: set[str] = set()
    for raw in _NUMBER.findall(_MARKER.sub(" ", text or "")):
        value = raw.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        if value in {"", "-"}:
            continue
        found.add(value)
    return found


def token_overlap(answer: str, reference: str) -> float:
    """Share of the reference's content tokens that appear in the answer.

    Recall-oriented rather than F1: reference answers in this benchmark are a
    phrase and system answers are extracted sentences, so an F1 would penalise
    a correct answer for the sentence it was extracted from. The cost is that a
    system dumping the whole document scores well, which is why
    ``answer_chars`` is recorded per question and why citation and faithfulness
    metrics are reported beside this one rather than folded into it.
    """
    wanted = set(content_tokens(reference))
    if not wanted:
        return 0.0
    got = set(content_tokens(answer))
    return len(wanted & got) / len(wanted)


# ---------------------------------------------------------------------------
# Error taxonomy (BENCHMARK_SPEC.md section 20, EVALUATION_PROTOCOL.md 23)
# ---------------------------------------------------------------------------

#: Bump when a trigger rule below changes, for the same reason
#: ABSTENTION_PATTERNS_VERSION exists: a category assigned under different rules
#: is not the same category.
ERROR_TAXONOMY_VERSION = "2026-09-23.1"

RETRIEVAL_FAILURE = "retrieval"
RERANKING_FAILURE = "reranking"
CONTEXT_SELECTION_FAILURE = "context-selection"
GENERATION_FAILURE = "generation"
HALLUCINATION = "hallucination"
CITATION_FAILURE = "citation"
ABSTENTION_FAILURE = "abstention"
VERIFICATION_FAILURE = "verification"
SYSTEM_RUNTIME_FAILURE = "system-runtime"

#: The nine categories the two specifications name between them, in the order
#: the classifier tries them. First match wins, so exclusivity is structural
#: rather than a property the rules happen to have.
ERROR_CATEGORIES: tuple[str, ...] = (
    SYSTEM_RUNTIME_FAILURE,
    RETRIEVAL_FAILURE,
    RERANKING_FAILURE,
    CONTEXT_SELECTION_FAILURE,
    ABSTENTION_FAILURE,
    HALLUCINATION,
    CITATION_FAILURE,
    VERIFICATION_FAILURE,
    GENERATION_FAILURE,
)

#: Stages a system declares it ran. Two categories -- reranking and
#: verification -- describe components DD-013 deliberately leaves out of the
#: dense baseline, so they can only fire for a system that declares the stage.
#: That is what lets this taxonomy be written once and reused by Phases 4 and 5
#: without inventing the missing components now.
STAGE_RETRIEVAL = "retrieval"
STAGE_RERANKING = "reranking"
STAGE_CONTEXT_SELECTION = "context-selection"
STAGE_GENERATION = "generation"
STAGE_VERIFICATION = "verification"

#: ARCHITECTURE.md section 24: "Verifier failure -> mark verification as
#: unavailable rather than claiming success." A system without a verifier
#: reports ``not_run`` and never ``passed``.
VERIFICATION_NOT_RUN = "not_run"
VERIFICATION_PASSED = "passed"
VERIFICATION_FAILED = "failed"
VERIFICATION_UNAVAILABLE = "unavailable"
VERIFICATION_STATUSES = (
    VERIFICATION_NOT_RUN,
    VERIFICATION_PASSED,
    VERIFICATION_FAILED,
    VERIFICATION_UNAVAILABLE,
)


# ---------------------------------------------------------------------------
# Per-question scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalScore:
    """Page-level retrieval metrics for one question, or the reason there are none.

    ``scored`` is false for every ``unanswerable`` question and the metric
    fields stay ``None``. A test asserts the exclusion here, not in the
    aggregate: an unanswerable question silently contributing 0.0 would be
    invisible in a mean but is obvious in a record.
    """

    scored: bool
    excluded_reason: str = ""
    gold_pages: tuple[int, ...] = ()
    retrieved_count: int = 0
    retrieved_pages: tuple[int, ...] = ()
    recall: Mapping[int, float] = field(default_factory=dict)
    full_recall: Mapping[int, float] = field(default_factory=dict)
    mrr_at_10: float | None = None
    ndcg_at_10: float | None = None
    first_relevant_rank: int | None = None
    #: K values larger than the number of chunks the system actually returned.
    #: Recall@10 on a five-chunk retrieval is a real figure -- five is what the
    #: system surfaced -- but it is the same figure as Recall@5, and a reader
    #: comparing systems with different depths needs to see that.
    capped_ks: tuple[int, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "scored": self.scored,
            "gold_pages": list(self.gold_pages),
            "retrieved_count": self.retrieved_count,
            "retrieved_pages": list(self.retrieved_pages),
            "first_relevant_rank": self.first_relevant_rank,
            "mrr_at_10": self.mrr_at_10,
            "ndcg_at_10": self.ndcg_at_10,
            "ndcg_relevance_scale": "binary",
            "capped_ks": list(self.capped_ks),
        }
        if self.excluded_reason:
            record["excluded_reason"] = self.excluded_reason
        for k in RECALL_KS:
            record[f"recall_at_{k}"] = self.recall.get(k)
        for k in FULL_RECALL_KS:
            record[f"full_recall_at_{k}"] = self.full_recall.get(k)
        return record


@dataclass(frozen=True)
class AnswerScore:
    """Deterministic answer metrics (BENCHMARK_SPEC.md section 11)."""

    #: BENCHMARK_SPEC.md section 11 / EVALUATION_PROTOCOL.md 21.1. For an
    #: answerable question this is the token-overlap-plus-numeric verdict; for
    #: an unanswerable one it is 1.0 exactly when the system abstained.
    correct: float
    overlap: float
    numeric_agreement: bool
    #: None when the answer contains no numbers: the metric
    #: EVALUATION_PROTOCOL.md 19.1 names is undefined there, and returning 1.0
    #: would credit every number-free answer with perfect grounding.
    faithfulness_numeric: float | None
    #: Share of the answer's content tokens present in the retrieved evidence.
    #: Always defined for a non-empty answer, so this is the headline
    #: deterministic faithfulness figure and the numeric one is its cross-check.
    faithfulness_token: float
    unsupported: bool
    matched_reference: str | None = None
    answer_chars: int = 0

    @property
    def faithfulness(self) -> float:
        return self.faithfulness_token

    def to_record(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "token_overlap": round(self.overlap, 4),
            "overlap_threshold": CORRECTNESS_OVERLAP_THRESHOLD,
            "numeric_agreement": self.numeric_agreement,
            "faithfulness": round(self.faithfulness_token, 4),
            "faithfulness_token": round(self.faithfulness_token, 4),
            "faithfulness_numeric": (
                None
                if self.faithfulness_numeric is None
                else round(self.faithfulness_numeric, 4)
            ),
            "unsupported": self.unsupported,
            "matched_reference": self.matched_reference,
            "answer_chars": self.answer_chars,
        }


@dataclass(frozen=True)
class CitationScore:
    """Citation metrics (BENCHMARK_SPEC.md section 11, EVALUATION_PROTOCOL.md 22).

    Pages come from ``Citation.pages`` -- resolved by code from the chunk the
    marker points at -- and are never re-read out of the answer text. DD-031 is
    the reason: the generator has never been shown a page number, so any page
    number appearing in its output is fabricated and scoring it would measure
    the fabrication.
    """

    scored: bool
    cited_pages: tuple[int, ...] = ()
    #: None when the system cited nothing: precision over an empty set is not
    #: 0.0, it is undefined, and averaging it as 0.0 would conflate "cited
    #: badly" with "did not cite".
    precision: float | None = None
    #: Share of the question's gold pages that were cited.
    gold_page_recall: float | None = None
    #: 1.0 when at least one cited page is a gold page. Zero when the system
    #: cited nothing on an answerable question, which is the honest reading:
    #: EXPERIMENT_PLAN.md section 3 sets a stopping rule on this figure.
    any_correct: float | None = None
    #: Share of the answer's substantive sentences carrying at least one marker.
    #: EVALUATION_PROTOCOL.md section 22 asks whether an important claim is
    #: missing a citation; claim-level scoring needs a human, so this is the
    #: deterministic proxy and is labelled as one.
    completeness: float | None = None
    resolved: int = 0
    dropped_markers: int = 0
    fabricated_page_mentions: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "scored": self.scored,
            "cited_pages": list(self.cited_pages),
            "precision": None if self.precision is None else round(self.precision, 4),
            "gold_page_recall": (
                None if self.gold_page_recall is None else round(self.gold_page_recall, 4)
            ),
            "any_correct": self.any_correct,
            "completeness": (
                None if self.completeness is None else round(self.completeness, 4)
            ),
            "resolved": self.resolved,
            "dropped_markers": self.dropped_markers,
            "fabricated_page_mentions": self.fabricated_page_mentions,
        }


@dataclass(frozen=True)
class AbstentionScore:
    """One cell of the BENCHMARK_SPEC.md section 13 outcome matrix.

    The two booleans are kept separate all the way to the aggregate because the
    four outcomes have four different denominators, and section 13.1 forbids
    averaging the two failure modes into one number.
    """

    answerable: bool
    abstained: bool

    @property
    def correct_abstention(self) -> bool:
        return not self.answerable and self.abstained

    @property
    def false_answer(self) -> bool:
        """The most serious failure in the project (section 13.1)."""
        return not self.answerable and not self.abstained

    @property
    def over_abstention(self) -> bool:
        """A safe failure, but a failure. Measured over the answerable set."""
        return self.answerable and self.abstained

    def to_record(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "abstained": self.abstained,
            "correct_abstention": self.correct_abstention,
            "false_answer": self.false_answer,
            "over_abstention": self.over_abstention,
            "patterns_version": ABSTENTION_PATTERNS_VERSION,
        }


@dataclass(frozen=True)
class QuestionScore:
    """Everything measured about one question."""

    question_id: str
    question_type: str
    split: str | None
    answerable: bool
    retrieval: RetrievalScore
    answer: AnswerScore
    citation: CitationScore
    abstention: AbstentionScore
    verification_status: str = VERIFICATION_NOT_RUN
    error: str | None = None
    error_category: str | None = None
    failed: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "split": self.split,
            "answerable": self.answerable,
            "correct": self.answer.correct,
            "failed": self.failed,
            "error_category": self.error_category,
            "error": self.error,
            "retrieval": self.retrieval.to_record(),
            "answer": self.answer.to_record(),
            "citation": self.citation.to_record(),
            "abstention": self.abstention.to_record(),
            "scoring_rules_version": SCORING_RULES_VERSION,
            "error_taxonomy_version": ERROR_TAXONOMY_VERSION,
        }


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


def _pages_through_k(retrieved: Sequence[Any], k: int) -> set[int]:
    """Every page covered by the top ``k`` retrieved chunks.

    DD-025: a chunk spanning a page break is credited for every page it covers,
    so this is a union over ``pages`` rather than over ``page_number``.
    """
    pages: set[int] = set()
    for item in retrieved[:k]:
        pages.update(_pages_of(item))
    return pages


def _pages_of(item: Any) -> tuple[int, ...]:
    pages = getattr(item, "pages", None)
    if pages is None:
        chunk = getattr(item, "chunk", None)
        pages = getattr(chunk, "pages", ())
    return tuple(pages or ())


def score_retrieval(question: Question, retrieved: Sequence[Any]) -> RetrievalScore:
    """Page-level retrieval metrics, or an explicit exclusion.

    ``unanswerable`` questions have no gold evidence, so every retrieval metric
    is undefined for them. BENCHMARK_SPEC.md section 10 is explicit that they
    are excluded rather than scored 0.0, because scoring them as failures would
    make a correctly-abstaining system look like a retrieval failure.
    """
    gold = tuple(sorted(set(question.evidence_pages)))
    if question.is_unanswerable or not gold:
        reason = "unanswerable" if question.is_unanswerable else "no_evidence_pages"
        return RetrievalScore(
            scored=False,
            excluded_reason=reason,
            gold_pages=gold,
            retrieved_count=len(retrieved),
            retrieved_pages=tuple(sorted(_pages_through_k(retrieved, len(retrieved)))),
        )

    gold_set = set(gold)
    recall: dict[int, float] = {}
    for k in RECALL_KS:
        recall[k] = 1.0 if gold_set & _pages_through_k(retrieved, k) else 0.0
    full_recall: dict[int, float] = {}
    for k in FULL_RECALL_KS:
        full_recall[k] = 1.0 if gold_set <= _pages_through_k(retrieved, k) else 0.0

    first_rank: int | None = None
    gains: list[int] = []
    for position, item in enumerate(retrieved[:NDCG_K], start=1):
        hit = bool(gold_set & set(_pages_of(item)))
        gains.append(1 if hit else 0)
        if hit and first_rank is None:
            first_rank = position

    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    return RetrievalScore(
        scored=True,
        gold_pages=gold,
        retrieved_count=len(retrieved),
        retrieved_pages=tuple(sorted(_pages_through_k(retrieved, len(retrieved)))),
        recall=recall,
        full_recall=full_recall,
        mrr_at_10=mrr,
        ndcg_at_10=_ndcg(gains),
        first_relevant_rank=first_rank,
        capped_ks=tuple(
            k for k in sorted(set(RECALL_KS) | set(FULL_RECALL_KS)) if k > len(retrieved)
        ),
    )


def _ndcg(gains: Sequence[int]) -> float:
    """nDCG over binary page-level relevance.

    Both specifications condition nDCG on graded relevance labels, which this
    dataset does not have: ``evidence_pages`` is a set, not a ranking. It is
    reported anyway as a secondary diagnostic, with the scale recorded as
    ``binary`` in the record so it is never read as the graded figure the specs
    describe.
    """
    if not gains:
        return 0.0
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, 1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Answer and citation scoring
# ---------------------------------------------------------------------------


def score_answer(
    question: Question,
    answer: str,
    abstained: bool,
    evidence_text: str,
    has_citations: bool,
) -> AnswerScore:
    """Deterministic correctness and faithfulness.

    EVALUATION_PROTOCOL.md section 21.1 fixes the two scoring rules this mixes:
    an unanswerable question is correct exactly when the system abstained, and
    an answerable one is scored against its reference. Both feed the headline
    accuracy figure, which is why that figure is never reported without its
    breakdown.
    """
    chars = len(answer or "")
    faith_token = _supported_token_fraction(answer, evidence_text)
    faith_numeric = _numeric_faithfulness(answer, evidence_text)
    unsupported = (not abstained) and (
        not has_citations or (faith_numeric is not None and faith_numeric < 1.0)
    )

    if question.is_unanswerable:
        return AnswerScore(
            correct=1.0 if abstained else 0.0,
            overlap=0.0,
            numeric_agreement=True,
            faithfulness_numeric=faith_numeric,
            faithfulness_token=faith_token,
            unsupported=unsupported,
            matched_reference=None,
            answer_chars=chars,
        )

    references = [r for r in (question.answer, *question.alternative_answers) if r]
    best_overlap = 0.0
    best_reference: str | None = None
    best_numeric = False
    for reference in references:
        overlap = token_overlap(answer, reference)
        numeric_ok = numbers(reference) <= numbers(answer)
        # Rank by the pair, so a reference that agrees numerically wins over one
        # with a slightly higher word overlap that does not.
        if (numeric_ok, overlap) > (best_numeric, best_overlap):
            best_numeric, best_overlap, best_reference = numeric_ok, overlap, reference

    correct = (
        1.0
        if best_overlap >= CORRECTNESS_OVERLAP_THRESHOLD and best_numeric and not abstained
        else 0.0
    )
    return AnswerScore(
        correct=correct,
        overlap=best_overlap,
        numeric_agreement=best_numeric,
        faithfulness_numeric=faith_numeric,
        faithfulness_token=faith_token,
        unsupported=unsupported,
        matched_reference=best_reference,
        answer_chars=chars,
    )


def _supported_token_fraction(answer: str, evidence_text: str) -> float:
    tokens = set(content_tokens(answer))
    if not tokens:
        return 0.0
    supported = set(content_tokens(evidence_text))
    return len(tokens & supported) / len(tokens)


def _numeric_faithfulness(answer: str, evidence_text: str) -> float | None:
    """"The fraction of the answer's numbers that appear in the retrieved
    evidence" (EVALUATION_PROTOCOL.md section 19.1), or None when it has none."""
    answer_numbers = numbers(answer)
    if not answer_numbers:
        return None
    evidence_numbers = numbers(evidence_text)
    return len(answer_numbers & evidence_numbers) / len(answer_numbers)


def score_citations(
    question: Question, answer: str, cited_pages: Sequence[int], **counts: int
) -> CitationScore:
    """Citation precision and completeness from resolved pages only."""
    pages = tuple(sorted(set(cited_pages)))
    gold = set(question.evidence_pages)
    resolved = int(counts.get("resolved", len(pages)))
    dropped = int(counts.get("dropped_markers", 0))
    fabricated = int(counts.get("fabricated_page_mentions", 0))

    if question.is_unanswerable or not gold:
        # No gold pages, so "does the cited page support the claim" has no
        # answer. Precision stays undefined rather than being scored against an
        # empty key.
        return CitationScore(
            scored=False,
            cited_pages=pages,
            completeness=_citation_completeness(answer),
            resolved=resolved,
            dropped_markers=dropped,
            fabricated_page_mentions=fabricated,
        )

    hits = len(set(pages) & gold)
    return CitationScore(
        scored=True,
        cited_pages=pages,
        precision=(hits / len(pages)) if pages else None,
        gold_page_recall=hits / len(gold),
        any_correct=1.0 if hits else 0.0,
        completeness=_citation_completeness(answer),
        resolved=resolved,
        dropped_markers=dropped,
        fabricated_page_mentions=fabricated,
    )


def _citation_completeness(answer: str) -> float | None:
    """Share of the answer's substantive sentences that carry a marker."""
    sentences = [
        s
        for s in _SENTENCE_SPLIT.split((answer or "").strip())
        if content_tokens(s)
    ]
    if not sentences:
        return None
    cited = sum(1 for s in sentences if _MARKER.search(s))
    return cited / len(sentences)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


def is_failure(score: QuestionScore) -> bool:
    """Does this record count as a failure that needs a category?

    Four ways to fail, and each is a failure the specifications name:
    the run raised (ARCHITECTURE.md section 24), the headline correctness rule
    scored it 0 (EVALUATION_PROTOCOL.md 21.1), the answer asserts something the
    evidence does not support (BENCHMARK_SPEC.md 13), or an answerable question
    was answered with no correct citation (section 22 -- citations are a
    headline feature, so a right answer pointing at the wrong page is not a
    pass).
    """
    if score.error:
        return True
    if score.verification_status in (VERIFICATION_FAILED, VERIFICATION_UNAVAILABLE):
        return True
    if score.answer.correct < 1.0:
        return True
    if score.answer.unsupported:
        return True
    if score.citation.scored and score.citation.any_correct == 0.0:
        return True
    return False


def classify_error(
    score: QuestionScore,
    stages: Sequence[str] = (),
    reranking_lost_gold: bool = False,
    evidence_pages: Sequence[int] = (),
    retrieved_pages: Sequence[int] | None = None,
) -> str | None:
    """Assign exactly one category to a failing record, or None to a passing one.

    The rules are tried in order and the first match wins, so exclusivity is a
    property of the structure rather than of the predicates happening not to
    overlap. The order is most-upstream-cause-first: a question whose evidence
    was never retrieved is a retrieval failure whatever the generator then did
    with the wrong passages, and blaming the generator for it would point future
    work at the wrong component -- which is the whole purpose DD-021 gives for
    keeping per-question records at all.

    ``reranking`` and ``verification`` describe components DD-013 deliberately
    leaves out of the dense baseline. They can only fire for a system that
    declares the stage in ``stages``, which is how one taxonomy serves Phases 3,
    4 and 5 without inventing the missing components now.

    ``reranking_lost_gold`` is the counterfactual DD-044 defines: a gold page
    was in the top-k the reranker was handed -- what "Reranker OFF" would have
    sent the generator -- and is not in the top-k it returned. A gold page that
    was only deeper in the pool and never promoted, or that fusion dropped, is a
    ``retrieval`` failure: the taxonomy has no fusion category, and fusion is
    part of the retrieval stage.

    ``retrieved_pages`` is the depth the system actually answered from, which is
    not always the depth its retrieval was *scored* at: the harness ranks to
    depth 10 for Recall@10 while the baseline generates from 5. Blame follows
    the answer path. A gold page ranked eighth never reached the generator, so
    it is a retrieval failure, and calling it a context-selection failure would
    point Phase 4 at the wrong component.
    """
    if not is_failure(score):
        return None

    # 1. The run itself broke. Nothing downstream of it means anything.
    if score.error:
        return SYSTEM_RUNTIME_FAILURE

    retrieval = score.retrieval
    answer_path_pages = set(
        retrieval.retrieved_pages if retrieved_pages is None else retrieved_pages
    )
    gold_retrieved = bool(set(retrieval.gold_pages) & answer_path_pages)

    # A reranker "lost" gold when the ranking it was handed would have put a
    # gold page on the answer path and its own ranking did not (DD-044). Only a
    # system that declares the stage can suffer one.
    lost_by_reranker = STAGE_RERANKING in stages and reranking_lost_gold

    # 2. The evidence was never retrieved -- not onto the answer path, and not
    #    into the top-k the reranker was handed either. A gold page the reranker
    #    pushed out *was* retrieved, so it falls through to rule 3; without this
    #    exclusion rule 3 could never fire, because a page the reranker pushed
    #    out is by definition missing from the answer path.
    if retrieval.scored and not gold_retrieved and not lost_by_reranker:
        return RETRIEVAL_FAILURE

    # 3. Retrieval found it and the reranker pushed it out.
    if lost_by_reranker:
        return RERANKING_FAILURE

    # 4. It survived ranking but never reached the generator, because evidence
    #    assembly deduplicated or budget-trimmed it away.
    if (
        retrieval.scored
        and gold_retrieved
        and evidence_pages
        and not (set(retrieval.gold_pages) & set(evidence_pages))
    ):
        return CONTEXT_SELECTION_FAILURE

    # 5. The abstention decision itself was wrong, in either direction. Checked
    #    before hallucination so that a false answer on an unanswerable question
    #    is filed as the abstention failure it is, rather than as a generation
    #    problem that better prompting would fix.
    if score.abstention.false_answer or score.abstention.over_abstention:
        return ABSTENTION_FAILURE

    # 6. It answered, from evidence it had, with something the evidence does not
    #    say.
    if score.answer.unsupported or score.citation.fabricated_page_mentions:
        return HALLUCINATION

    # 7. The answer is right and the citations are not. Reached only when
    #    correctness passed, so this never competes with a wrong answer.
    if (
        score.answer.correct >= 1.0
        and score.citation.scored
        and score.citation.any_correct == 0.0
    ):
        return CITATION_FAILURE

    # 8. Reserved, like reranking: a verifier that ran and did not pass.
    if STAGE_VERIFICATION in stages and score.verification_status in (
        VERIFICATION_FAILED,
        VERIFICATION_UNAVAILABLE,
    ):
        return VERIFICATION_FAILURE

    # 9. Terminal. The system had what it needed and still produced the wrong
    #    answer, which is what a generation failure is.
    return GENERATION_FAILURE


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float | None:
    collected = [v for v in values if v is not None]
    return sum(collected) / len(collected) if collected else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(scores: Sequence[QuestionScore]) -> dict[str, Any]:
    """Aggregate per-question scores into the reported figures.

    Every rate here carries its denominator in the output. That is not padding:
    EVALUATION_PROTOCOL.md section 21.1 requires accuracy to be reported as
    three distinct numbers over three different denominators, and a reader
    cannot check an abstention figure computed over six questions without being
    told it was six.
    """
    total = len(scores)
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    retrieval_scored = [s for s in scores if s.retrieval.scored]

    summary: dict[str, Any] = {
        "n_questions": total,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_retrieval_scored": len(retrieval_scored),
        "scoring_rules_version": SCORING_RULES_VERSION,
        "error_taxonomy_version": ERROR_TAXONOMY_VERSION,
        "abstention_patterns_version": ABSTENTION_PATTERNS_VERSION,
        "correctness_overlap_threshold": CORRECTNESS_OVERLAP_THRESHOLD,
    }

    # -- retrieval, over answerable questions only --------------------------
    for k in RECALL_KS:
        summary[f"recall_at_{k}"] = _mean(s.retrieval.recall.get(k) for s in retrieval_scored)
    for k in FULL_RECALL_KS:
        summary[f"full_recall_at_{k}"] = _mean(
            s.retrieval.full_recall.get(k) for s in retrieval_scored
        )
    summary["mrr_at_10"] = _mean(s.retrieval.mrr_at_10 for s in retrieval_scored)
    summary["ndcg_at_10"] = _mean(s.retrieval.ndcg_at_10 for s in retrieval_scored)
    summary["ndcg_relevance_scale"] = "binary"

    # BENCHMARK_SPEC.md section 10: Full-Recall@5 is the primary figure for
    # these categories, so it is broken out rather than left inside the mean.
    multi = [
        s for s in retrieval_scored if s.question_type in FULL_RECALL_REQUIRED_TYPES
    ]
    summary["n_multi_evidence_types"] = len(multi)
    summary["full_recall_at_5_multi_hop_comparison"] = _mean(
        s.retrieval.full_recall.get(5) for s in multi
    )

    # -- answer -------------------------------------------------------------
    # EVALUATION_PROTOCOL.md 21.1: three numbers, three denominators, never
    # collapsed into one.
    summary["accuracy_all"] = _mean(s.answer.correct for s in scores)
    summary["accuracy_answerable"] = _mean(s.answer.correct for s in answerable)
    summary["faithfulness"] = _mean(s.answer.faithfulness_token for s in scores)
    summary["faithfulness_numeric"] = _mean(
        s.answer.faithfulness_numeric for s in scores
    )
    summary["n_faithfulness_numeric_defined"] = sum(
        1 for s in scores if s.answer.faithfulness_numeric is not None
    )

    # -- abstention (BENCHMARK_SPEC.md section 13) --------------------------
    # Three denominators. The third is the one that matters: a system that
    # refuses everything scores 1.0 on the first two, and only over-abstention
    # -- computed over the ANSWERABLE set -- can detect it.
    summary["abstention_accuracy"] = _rate(
        sum(1 for s in unanswerable if s.abstention.abstained), len(unanswerable)
    )
    summary["false_answer_rate"] = _rate(
        sum(1 for s in unanswerable if s.abstention.false_answer), len(unanswerable)
    )
    summary["over_abstention_rate"] = _rate(
        sum(1 for s in answerable if s.abstention.over_abstention), len(answerable)
    )
    summary["unsupported_answer_rate"] = _rate(
        sum(1 for s in scores if s.answer.unsupported), total
    )
    summary["abstention_denominators"] = {
        "abstention_accuracy": len(unanswerable),
        "false_answer_rate": len(unanswerable),
        "over_abstention_rate": len(answerable),
        "unsupported_answer_rate": total,
    }

    # -- citation -----------------------------------------------------------
    cited = [s for s in scores if s.citation.scored]
    summary["citation_precision"] = _mean(s.citation.precision for s in cited)
    summary["citation_any_correct"] = _mean(s.citation.any_correct for s in cited)
    summary["citation_gold_page_recall"] = _mean(
        s.citation.gold_page_recall for s in cited
    )
    summary["citation_completeness"] = _mean(s.citation.completeness for s in scores)
    summary["dropped_marker_rate"] = _rate(
        sum(1 for s in scores if s.citation.dropped_markers), total
    )
    summary["fabricated_page_mention_rate"] = _rate(
        sum(1 for s in scores if s.citation.fabricated_page_mentions), total
    )

    # -- errors (DD-021) ----------------------------------------------------
    failures = [s for s in scores if s.failed]
    counts = {category: 0 for category in ERROR_CATEGORIES}
    for score in failures:
        if score.error_category:
            counts[score.error_category] = counts.get(score.error_category, 0) + 1
    summary["n_failures"] = len(failures)
    summary["error_categories"] = counts

    return summary
