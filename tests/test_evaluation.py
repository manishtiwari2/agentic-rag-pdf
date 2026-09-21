"""Tests for the Phase 3 metrics harness.

Three of these exist because the metric they check is one a plausible
implementation gets wrong in a way that flatters the system:

* ``Recall@K`` and ``Full-Recall@K`` must disagree on a multi-evidence question.
  An implementation that computed one and reported both would pass every
  single-evidence test in this file.
* Unanswerable questions must be *excluded* from retrieval metrics, not scored
  0.0. The assertion is on the per-question record, because a wrong exclusion is
  invisible in an aggregate.
* ``over_abstention_rate``'s denominator is the answerable set. A refuse-
  everything system scores perfectly on every other abstention metric, so the
  test builds that system and requires it to score badly.
"""

from __future__ import annotations

import csv
import json
import pathlib

import pytest

from src.benchmark.schema import Question, load_questions
from src.chunking.base import Chunk
from src.config import RAGConfig
from src.evaluation.benchmark import (
    build_config_snapshot,
    run_benchmark_cli,
    latency_stats,
    load_results,
    run_benchmark,
    score_result,
    write_results,
    BenchmarkRun,
)
from src.evaluation.judge import JudgeVerdict, LLMJudge, parse_judge_reply
from src.evaluation.metrics import (
    CONTEXT_SELECTION_FAILURE,
    ERROR_CATEGORIES,
    GENERATION_FAILURE,
    HALLUCINATION,
    ABSTENTION_FAILURE,
    CITATION_FAILURE,
    RERANKING_FAILURE,
    RETRIEVAL_FAILURE,
    STAGE_RERANKING,
    STAGE_VERIFICATION,
    SYSTEM_RUNTIME_FAILURE,
    VERIFICATION_FAILED,
    VERIFICATION_FAILURE,
    VERIFICATION_NOT_RUN,
    AbstentionScore,
    AnswerScore,
    CitationScore,
    QuestionScore,
    RetrievalScore,
    classify_error,
    is_failure,
    numbers,
    score_answer,
    score_citations,
    score_retrieval,
    summarize,
)
from src.generation.citations import Citation
from src.generation.evidence import EvidenceItem
from src.generation.prompts import ABSTENTION_SENTENCE
from src.pipeline import DenseRAGPipeline, RAGResult
from src.retrieval.retriever import RetrievedChunk

# ---------------------------------------------------------------------------
# Builders: a scored record without a PDF
# ---------------------------------------------------------------------------


def make_question(
    id: str = "q001",
    pages: tuple[int, ...] = (3,),
    answer: str | None = "The peak memory was 9.4 gigabytes.",
    question_type: str = "factual",
    **extra,
) -> Question:
    return Question(
        id=id,
        document="doc1.pdf",
        question="What was the peak memory?",
        answer=answer,
        evidence_pages=pages,
        question_type=question_type,
        difficulty="easy",
        split="dev",
        **extra,
    )


def make_chunk(pages: tuple[int, ...], text: str = "body text", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"doc:chunk:{index:05d}",
        document_id="doc",
        text=text,
        pages=pages,
        index=index,
    )


def make_retrieved(*page_groups: tuple[int, ...], text: str = "body text"):
    return [
        RetrievedChunk(chunk=make_chunk(pages, text, index), score=1.0 - index / 100, rank=index + 1)
        for index, pages in enumerate(page_groups)
    ]


def make_result(
    answer: str = "The peak memory was 9.4 gigabytes. [C1]",
    retrieved=None,
    cited_pages: tuple[int, ...] = (3,),
    evidence_text: str = "Peak memory during generation reached 9.4 gigabytes.",
    abstained: bool = False,
    **metadata,
) -> RAGResult:
    retrieved = retrieved if retrieved is not None else make_retrieved((3,), (4,))
    evidence = tuple(
        EvidenceItem(number=i + 1, text=evidence_text, retrieved=item)
        for i, item in enumerate(retrieved)
    )
    citations = tuple(
        Citation(
            marker=f"C{i + 1}",
            evidence_number=i + 1,
            chunk_id=f"doc:chunk:{i:05d}",
            document_id="doc",
            pages=(page,),
        )
        for i, page in enumerate(cited_pages)
    )
    return RAGResult(
        question="What was the peak memory?",
        answer=answer,
        citations=() if abstained else citations,
        evidence=evidence,
        retrieved=tuple(retrieved),
        abstained=abstained,
        metadata={"pipeline": "test", "latency_s": 0.01, **metadata},
    )


def make_score(
    *,
    answerable: bool = True,
    abstained: bool = False,
    correct: float = 1.0,
    unsupported: bool = False,
    gold_pages: tuple[int, ...] = (3,),
    retrieved_pages: tuple[int, ...] = (3, 4),
    citation_any_correct: float | None = 1.0,
    citation_scored: bool = True,
    fabricated: int = 0,
    verification_status: str = VERIFICATION_NOT_RUN,
    error: str | None = None,
) -> QuestionScore:
    return QuestionScore(
        question_id="q001",
        question_type="factual" if answerable else "unanswerable",
        split="dev",
        answerable=answerable,
        retrieval=RetrievalScore(
            scored=answerable,
            excluded_reason="" if answerable else "unanswerable",
            gold_pages=gold_pages if answerable else (),
            retrieved_count=len(retrieved_pages),
            retrieved_pages=retrieved_pages,
        ),
        answer=AnswerScore(
            correct=correct,
            overlap=1.0 if correct else 0.0,
            numeric_agreement=True,
            faithfulness_numeric=None,
            faithfulness_token=1.0,
            unsupported=unsupported,
        ),
        citation=CitationScore(
            scored=citation_scored,
            cited_pages=(3,),
            any_correct=citation_any_correct,
            fabricated_page_mentions=fabricated,
        ),
        abstention=AbstentionScore(answerable=answerable, abstained=abstained),
        verification_status=verification_status,
        error=error,
    )


def classify(score: QuestionScore, **kwargs) -> str | None:
    return classify_error(score, **kwargs)


# ---------------------------------------------------------------------------


class TestPageLevelRelevance:
    """DD-025: a chunk is relevant when ANY page it covers is a gold page."""

    def test_a_chunk_spanning_a_page_break_is_credited_for_every_page(self):
        question = make_question(pages=(8,))
        score = score_retrieval(question, make_retrieved((7, 8)))
        assert score.recall[1] == 1.0

    def test_a_chunk_on_another_page_is_not_relevant(self):
        score = score_retrieval(make_question(pages=(3,)), make_retrieved((9,), (10,)))
        assert score.recall[5] == 0.0
        assert score.mrr_at_10 == 0.0
        assert score.first_relevant_rank is None


class TestRecallVersusFullRecall:
    """BENCHMARK_SPEC.md section 10: both are required because they disagree."""

    def test_they_disagree_when_only_one_gold_page_is_retrieved(self):
        """The spec's own example: evidence [7, 12], only page 7 retrieved."""
        question = make_question(pages=(7, 12), question_type="multi_hop")
        score = score_retrieval(question, make_retrieved((7,), (3,), (4,), (5,), (6,)))

        assert score.recall[5] == 1.0, "page 7 is in the top 5, so ANY-recall is a hit"
        assert score.full_recall[5] == 0.0, "page 12 is missing, so ALL-recall is a miss"

    def test_they_agree_when_every_gold_page_is_retrieved(self):
        question = make_question(pages=(7, 12), question_type="multi_hop")
        score = score_retrieval(question, make_retrieved((7,), (12,)))
        assert score.recall[5] == score.full_recall[5] == 1.0

    def test_they_are_identical_for_a_single_evidence_question(self):
        question = make_question(pages=(4,))
        score = score_retrieval(question, make_retrieved((4,), (9,)))
        assert score.recall[5] == score.full_recall[5]

    def test_k_bounds_the_hit(self):
        question = make_question(pages=(9,))
        score = score_retrieval(
            question, make_retrieved((1,), (2,), (3,), (4,), (9,))
        )
        assert score.recall[1] == 0.0
        assert score.recall[3] == 0.0
        assert score.recall[5] == 1.0
        assert score.mrr_at_10 == pytest.approx(1 / 5)

    def test_full_recall_at_5_is_reported_for_the_categories_that_need_it(self):
        scores = [
            _scored(make_question(id="a", pages=(1, 9), question_type="multi_hop"),
                    make_retrieved((1,))),
            _scored(make_question(id="b", pages=(1, 9), question_type="comparison"),
                    make_retrieved((1,), (9,))),
            _scored(make_question(id="c", pages=(1,), question_type="factual"),
                    make_retrieved((1,))),
        ]
        summary = summarize(scores)
        assert summary["n_multi_evidence_types"] == 2
        assert summary["full_recall_at_5_multi_hop_comparison"] == pytest.approx(0.5)


def _scored(question: Question, retrieved) -> QuestionScore:
    """A QuestionScore carrying only retrieval, for aggregate tests."""
    return QuestionScore(
        question_id=question.id,
        question_type=question.question_type,
        split=question.split,
        answerable=not question.is_unanswerable,
        retrieval=score_retrieval(question, retrieved),
        answer=AnswerScore(1.0, 1.0, True, None, 1.0, False),
        citation=CitationScore(scored=False),
        abstention=AbstentionScore(
            answerable=not question.is_unanswerable, abstained=question.is_unanswerable
        ),
    )


class TestUnanswerableQuestionsAreExcluded:
    """BENCHMARK_SPEC.md section 10: excluded, never scored 0.0.

    Scoring them as retrieval failures would make a correctly-abstaining system
    look like a retrieval failure, so the exclusion is asserted on the record
    itself. A wrong exclusion is invisible in a mean.
    """

    def test_the_record_says_excluded_rather_than_zero(self):
        question = make_question(id="q007", pages=(), answer=None, question_type="unanswerable")
        score = score_retrieval(question, make_retrieved((1,), (2,)))

        assert score.scored is False
        assert score.excluded_reason == "unanswerable"
        record = score.to_record()
        assert record["scored"] is False
        assert record["excluded_reason"] == "unanswerable"
        for key, value in record.items():
            if key.startswith(("recall_at_", "full_recall_at_")) or key == "mrr_at_10":
                assert value is None, f"{key} must be None for an unanswerable question"

    def test_an_excluded_question_does_not_enter_the_retrieval_denominator(self):
        answerable = _scored(make_question(id="a", pages=(1,)), make_retrieved((1,)))
        unanswerable = _scored(
            make_question(id="b", pages=(), answer=None, question_type="unanswerable"),
            make_retrieved((9,)),
        )
        summary = summarize([answerable, unanswerable])

        assert summary["n_questions"] == 2
        assert summary["n_retrieval_scored"] == 1
        # Scored as 0.0 this would be 0.5, and the abstaining system would look
        # like it had a retrieval problem.
        assert summary["recall_at_5"] == 1.0

    def test_citation_metrics_are_excluded_too(self):
        question = make_question(pages=(), answer=None, question_type="unanswerable")
        score = score_citations(question, answer=ABSTENTION_SENTENCE, cited_pages=())
        assert score.scored is False
        assert score.precision is None
        assert score.any_correct is None


class TestAbstentionOutcomeMatrix:
    """BENCHMARK_SPEC.md section 13's four cells, each on its own."""

    def test_unanswerable_and_abstained_is_correct_behaviour(self):
        outcome = AbstentionScore(answerable=False, abstained=True)
        assert outcome.correct_abstention
        assert not outcome.false_answer
        assert not outcome.over_abstention

    def test_unanswerable_and_answered_is_a_false_answer(self):
        outcome = AbstentionScore(answerable=False, abstained=False)
        assert outcome.false_answer
        assert not outcome.over_abstention
        assert not outcome.correct_abstention

    def test_answerable_and_abstained_is_over_abstention(self):
        outcome = AbstentionScore(answerable=True, abstained=True)
        assert outcome.over_abstention
        assert not outcome.false_answer

    def test_answerable_and_answered_is_correct_behaviour(self):
        outcome = AbstentionScore(answerable=True, abstained=False)
        assert not outcome.over_abstention
        assert not outcome.false_answer
        assert not outcome.correct_abstention


class TestAbstentionDenominators:
    """EVALUATION_PROTOCOL.md section 21: note the third metric's denominator."""

    @staticmethod
    def _population(abstain_on_answerable: bool, abstain_on_unanswerable: bool):
        answerable = [
            make_score(answerable=True, abstained=abstain_on_answerable,
                       correct=0.0 if abstain_on_answerable else 1.0)
            for _ in range(8)
        ]
        unanswerable = [
            make_score(answerable=False, abstained=abstain_on_unanswerable,
                       correct=1.0 if abstain_on_unanswerable else 0.0,
                       citation_scored=False, citation_any_correct=None)
            for _ in range(2)
        ]
        return answerable + unanswerable

    def test_a_refuse_everything_system_does_not_score_perfectly(self):
        """The failure the over-abstention metric exists to catch."""
        summary = summarize(self._population(True, True))

        assert summary["abstention_accuracy"] == 1.0
        assert summary["false_answer_rate"] == 0.0
        # ...and the metric that sees through it:
        assert summary["over_abstention_rate"] == 1.0
        assert summary["accuracy_answerable"] == 0.0

    def test_an_answer_everything_system_posts_the_opposite_failure(self):
        summary = summarize(self._population(False, False))
        assert summary["abstention_accuracy"] == 0.0
        assert summary["false_answer_rate"] == 1.0
        assert summary["over_abstention_rate"] == 0.0

    def test_each_rate_carries_its_own_denominator(self):
        summary = summarize(self._population(True, True))
        assert summary["abstention_denominators"] == {
            "abstention_accuracy": 2,
            "false_answer_rate": 2,
            "over_abstention_rate": 8,
            "unsupported_answer_rate": 10,
        }

    def test_the_two_failure_modes_are_never_averaged(self):
        """Section 13.1: report them separately, never as one error rate."""
        summary = summarize(self._population(True, False))
        assert summary["false_answer_rate"] == 1.0
        assert summary["over_abstention_rate"] == 1.0
        assert "abstention_error_rate" not in summary

    def test_accuracy_is_three_distinct_numbers(self):
        """EVALUATION_PROTOCOL.md 21.1: all / answerable / abstention."""
        summary = summarize(self._population(False, True))
        assert summary["accuracy_all"] == pytest.approx(1.0)
        assert summary["accuracy_answerable"] == pytest.approx(1.0)
        assert summary["abstention_accuracy"] == pytest.approx(1.0)
        assert summary["n_answerable"] == 8
        assert summary["n_unanswerable"] == 2


class TestDeterministicAnswerScoring:
    """EVALUATION_PROTOCOL.md 18 and 19.2: overlap plus exact numerics."""

    def test_an_answer_repeating_the_reference_is_correct(self):
        question = make_question(answer="Peak memory was 9.4 gigabytes.")
        score = score_answer(
            question,
            answer="Peak memory was 9.4 gigabytes. [C1]",
            abstained=False,
            evidence_text="Peak memory during generation reached 9.4 gigabytes.",
            has_citations=True,
        )
        assert score.correct == 1.0
        assert score.numeric_agreement

    def test_a_near_miss_number_is_not_correct(self):
        """86.7 vs 87.6 is the comparison section 19.2 keeps away from a judge."""
        question = make_question(answer="Recall was 86.7 percent.")
        score = score_answer(
            question,
            answer="Recall was 87.6 percent. [C1]",
            abstained=False,
            evidence_text="Recall was 87.6 percent.",
            has_citations=True,
        )
        assert score.numeric_agreement is False
        assert score.correct == 0.0

    def test_thousands_separators_do_not_change_a_number(self):
        assert numbers("1,200 items") == numbers("1200 items")
        assert numbers("86.70") == numbers("86.7")
        assert numbers("86.7") != numbers("87.6")

    def test_an_alternative_answer_also_counts(self):
        question = make_question(
            answer="IIT Madras.", alternative_answers=("Indian Institute of Technology Madras.",)
        )
        score = score_answer(
            question,
            answer="The Indian Institute of Technology Madras organises it. [C1]",
            abstained=False,
            evidence_text="Indian Institute of Technology Madras",
            has_citations=True,
        )
        assert score.correct == 1.0
        assert score.matched_reference == "Indian Institute of Technology Madras."

    def test_abstaining_on_an_answerable_question_is_never_correct(self):
        question = make_question(answer="IIT Madras.")
        score = score_answer(
            question,
            answer=ABSTENTION_SENTENCE,
            abstained=True,
            evidence_text="IIT Madras organises it.",
            has_citations=False,
        )
        assert score.correct == 0.0

    def test_an_unanswerable_question_is_correct_exactly_when_abstained(self):
        question = make_question(pages=(), answer=None, question_type="unanswerable")
        abstained = score_answer(question, ABSTENTION_SENTENCE, True, "evidence", False)
        answered = score_answer(question, "It was 9.4 GB. [C1]", False, "evidence", True)
        assert abstained.correct == 1.0
        assert answered.correct == 0.0

    def test_faithfulness_is_undefined_rather_than_perfect_without_numbers(self):
        question = make_question(answer="It is a transformer.")
        score = score_answer(
            question, "It is a transformer. [C1]", False, "It is a transformer.", True
        )
        assert score.faithfulness_numeric is None
        assert score.faithfulness_token == pytest.approx(1.0)

    def test_a_number_absent_from_the_evidence_is_unsupported(self):
        question = make_question(answer="Peak memory was 9.4 gigabytes.")
        score = score_answer(
            question,
            answer="Peak memory was 12.8 gigabytes. [C1]",
            abstained=False,
            evidence_text="Peak memory reached 9.4 gigabytes.",
            has_citations=True,
        )
        assert score.faithfulness_numeric == 0.0
        assert score.unsupported is True

    def test_an_uncited_answer_is_unsupported(self):
        question = make_question(answer="It is a transformer.")
        score = score_answer(question, "It is a transformer.", False, "evidence", False)
        assert score.unsupported is True


class TestCitationScoring:
    """EVALUATION_PROTOCOL.md section 22, from resolved pages only."""

    def test_precision_counts_cited_pages_against_gold(self):
        question = make_question(pages=(3, 4))
        score = score_citations(question, "Answer. [C1] More. [C2]", cited_pages=(3, 9))
        assert score.precision == pytest.approx(0.5)
        assert score.gold_page_recall == pytest.approx(0.5)
        assert score.any_correct == 1.0

    def test_precision_is_undefined_rather_than_zero_when_nothing_was_cited(self):
        question = make_question(pages=(3,))
        score = score_citations(question, ABSTENTION_SENTENCE, cited_pages=())
        assert score.precision is None
        assert score.any_correct == 0.0

    def test_completeness_is_the_share_of_sentences_carrying_a_marker(self):
        question = make_question(pages=(3,))
        score = score_citations(
            question, "First claim. [C1] Second claim with no marker.", cited_pages=(3,)
        )
        assert score.completeness == pytest.approx(0.5)


class TestErrorTaxonomy:
    """EVALUATION_PROTOCOL.md section 23: one category per failure, and only one."""

    def test_a_passing_record_gets_no_category(self):
        assert classify(make_score()) is None

    def test_system_runtime_failure(self):
        score = make_score(error="PDFReadError: the file is not a PDF")
        assert classify(score) == SYSTEM_RUNTIME_FAILURE

    def test_retrieval_failure(self):
        """The gold page never reached the answer path."""
        score = make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(9, 10))
        assert classify(score, evidence_pages=(9, 10)) == RETRIEVAL_FAILURE

    def test_reranking_failure_needs_a_system_that_ran_a_reranker(self):
        """DD-013 leaves the reranker out of the baseline, so the category is
        reserved: it can only fire for a system declaring the stage."""
        score = make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(3, 4))
        assert (
            classify(score, stages=(STAGE_RERANKING,), reranking_lost_gold=True)
            == RERANKING_FAILURE
        )
        # Same record, a system without a reranker: never blamed on one.
        assert classify(score, stages=(), reranking_lost_gold=True) != RERANKING_FAILURE

    def test_context_selection_failure(self):
        """Retrieved, then dropped before the generator saw it."""
        score = make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(3, 4))
        assert classify(score, evidence_pages=(4,)) == CONTEXT_SELECTION_FAILURE

    def test_abstention_failure_covers_over_abstention(self):
        score = make_score(correct=0.0, abstained=True, citation_any_correct=0.0)
        assert classify(score, evidence_pages=(3,)) == ABSTENTION_FAILURE

    def test_abstention_failure_covers_a_false_answer(self):
        score = make_score(answerable=False, abstained=False, correct=0.0,
                           citation_scored=False, citation_any_correct=None)
        assert classify(score, evidence_pages=(3,)) == ABSTENTION_FAILURE

    def test_hallucination(self):
        score = make_score(correct=0.0, unsupported=True)
        assert classify(score, evidence_pages=(3,)) == HALLUCINATION

    def test_citation_failure_is_a_right_answer_pointing_at_the_wrong_page(self):
        score = make_score(correct=1.0, citation_any_correct=0.0)
        assert classify(score, evidence_pages=(3,)) == CITATION_FAILURE

    def test_verification_failure_needs_a_system_that_ran_a_verifier(self):
        score = make_score(correct=1.0, verification_status=VERIFICATION_FAILED)
        assert (
            classify(score, stages=(STAGE_VERIFICATION,), evidence_pages=(3,))
            == VERIFICATION_FAILURE
        )

    def test_generation_failure_is_the_terminal_category(self):
        """Everything it needed, and still the wrong answer."""
        score = make_score(correct=0.0)
        assert classify(score, evidence_pages=(3,)) == GENERATION_FAILURE

    def test_every_failing_record_gets_exactly_one_category(self):
        population = [
            make_score(error="boom"),
            make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(9,)),
            make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(3, 4)),
            make_score(correct=0.0, abstained=True, citation_any_correct=0.0),
            make_score(answerable=False, abstained=False, correct=0.0,
                       citation_scored=False, citation_any_correct=None),
            make_score(correct=0.0, unsupported=True),
            make_score(correct=1.0, citation_any_correct=0.0),
            make_score(correct=1.0, verification_status=VERIFICATION_FAILED),
            make_score(correct=0.0),
            make_score(),  # passes
        ]
        assigned = [
            classify(s, stages=(STAGE_VERIFICATION,), evidence_pages=(4,))
            for s in population
        ]
        for score, category in zip(population, assigned):
            if is_failure(score):
                assert category is not None, "a failing record must get a category"
                assert category in ERROR_CATEGORIES
            else:
                assert category is None, "a passing record must get no category"
        # One string per record is one category per record: the classifier
        # returns a single value, so exclusivity is structural.
        assert all(isinstance(c, (str, type(None))) for c in assigned)

    def test_every_category_is_reachable(self):
        """A category no rule can ever fire is documentation, not a taxonomy."""
        reached = {
            classify(make_score(error="boom")),
            classify(make_score(correct=0.0, gold_pages=(3,), retrieved_pages=(9,))),
            classify(make_score(correct=0.0), stages=(STAGE_RERANKING,),
                     reranking_lost_gold=True),
            classify(make_score(correct=0.0), evidence_pages=(4,)),
            classify(make_score(correct=0.0, abstained=True, citation_any_correct=0.0),
                     evidence_pages=(3,)),
            classify(make_score(correct=0.0, unsupported=True), evidence_pages=(3,)),
            classify(make_score(correct=1.0, citation_any_correct=0.0),
                     evidence_pages=(3,)),
            classify(make_score(correct=1.0, verification_status=VERIFICATION_FAILED),
                     stages=(STAGE_VERIFICATION,), evidence_pages=(3,)),
            classify(make_score(correct=0.0), evidence_pages=(3,)),
        }
        assert set(ERROR_CATEGORIES) <= reached


class TestPerQuestionRecord:
    """BENCHMARK_SPEC.md section 20's required fields."""

    def test_the_record_carries_every_required_field(self):
        run = score_result(make_question(), make_result(), system="baseline_dense")
        record = run.record
        for field in (
            "question_id",
            "system",
            "answer",
            "reference_answer",
            "retrieved_chunks",
            "citations",
            "metrics",
            "latency",
            "verification_status",
        ):
            assert field in record, f"section 20 requires {field}"

    def test_a_baseline_without_a_verifier_never_claims_one_passed(self):
        """ARCHITECTURE.md section 24: unavailable, not a claimed success."""
        run = score_result(make_question(), make_result(), system="baseline_dense")
        assert run.record["verification_status"] == VERIFICATION_NOT_RUN

    def test_abstention_is_detected_from_the_text_not_the_self_report(self):
        """BENCHMARK_SPEC.md 13.2, with the disagreement recorded."""
        result = make_result(answer=ABSTENTION_SENTENCE, abstained=False)
        run = score_result(make_question(), result, system="s")
        assert run.score.abstention.abstained is True
        assert run.record["metrics"]["self_reported_abstention"] is False
        assert run.record["metrics"]["abstention_agrees_with_self_report"] is False

    def test_the_bare_record_is_unchanged_for_the_cli(self):
        record = make_result().to_record()
        assert "question_id" not in record
        assert record["question"] and record["answer"]


class TestResultStorage:
    """EVALUATION_PROTOCOL.md section 27's four files, round-tripped."""

    @pytest.fixture
    def written(self, tmp_path):
        runs = [
            score_result(make_question(id="q001"), make_result(), system="baseline_dense"),
            score_result(
                make_question(id="q002", pages=(), answer=None, question_type="unanswerable"),
                make_result(answer=ABSTENTION_SENTENCE, abstained=True),
                system="baseline_dense",
            ),
        ]
        run = BenchmarkRun(system="baseline_dense", runs=list(runs), latencies=[0.1, 0.3])
        run.config = build_config_snapshot(
            RAGConfig.offline(), "baseline_dense", tmp_path / "q.json", tmp_path, 2, None
        )
        paths = write_results(tmp_path / "out", run)
        return tmp_path / "out", run, paths

    def test_all_four_files_are_written(self, written):
        directory, _, paths = written
        assert set(paths) == {"config", "results", "per_question", "summary"}
        for name in ("config.json", "results.json", "per_question.json", "summary.csv"):
            assert (directory / name).exists(), name

    def test_every_file_round_trips(self, written):
        directory, run, _ = written
        loaded = load_results(directory)

        assert loaded["config"]["config_fingerprint"] == run.config["config_fingerprint"]
        assert loaded["results"]["summary"]["n_questions"] == 2
        assert [r["question_id"] for r in loaded["per_question"]] == ["q001", "q002"]
        assert loaded["per_question"] == json.loads(
            json.dumps(run.records, default=str)
        )
        assert len(loaded["summary_csv"]) == 1
        assert loaded["summary_csv"][0]["system"] == "baseline_dense"

    def test_the_config_records_what_reproduces_the_run(self, written):
        directory, _, _ = written
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for key in (
            "generation_model",
            "embedding_model",
            "quantization",
            "chunking_strategy",
            "chunk_size",
            "chunk_overlap",
            "top_k",
            "temperature",
            "max_new_tokens",
            "random_seed",
            "benchmark_version",
            "scoring_rules_version",
            "abstention_patterns_version",
        ):
            assert key in config, f"EVALUATION_PROTOCOL.md section 3 requires {key}"
        assert config["reranker_enabled"] is False

    def test_the_summary_csv_has_the_section_20_columns(self, written):
        directory, _, _ = written
        with (directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        for column in ("accuracy_all", "faithfulness", "recall_at_5", "mrr_at_10"):
            assert column in rows[0]

    def test_results_json_breaks_the_summary_down(self, written):
        directory, _, _ = written
        results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        assert "by_question_type" in results
        assert "by_split" in results
        assert isinstance(results["failures"], list)


class TestLatencyStats:
    """EVALUATION_PROTOCOL.md section 15: never the average alone."""

    def test_mean_median_and_p95_are_all_reported(self):
        stats = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
        assert stats["latency_mean_s"] == pytest.approx(22.0)
        assert stats["latency_median_s"] == pytest.approx(3.0)
        assert stats["latency_p95_s"] == pytest.approx(100.0)

    def test_an_empty_run_reports_nothing_rather_than_zero(self):
        assert latency_stats([])["latency_median_s"] is None


class TestJudgeIsOptional:
    """EVALUATION_PROTOCOL.md 19: optional, non-blocking, never a silent zero."""

    def test_the_offline_backend_declines_to_judge(self):
        judge = LLMJudge(RAGConfig.offline().generation)
        verdict = judge.judge(make_question(), make_result())
        assert verdict.skipped
        assert "judge" in verdict.skipped_reason

    def test_a_skipped_judgement_is_null_not_zero(self):
        record = {"metrics": {"answer": {"correct": 1.0}}}
        LLMJudge(RAGConfig.offline().generation).annotate(
            make_question(), make_result(), record
        )
        metrics = record["metrics"]
        assert metrics["judge_correctness"] is None
        assert metrics["judge_skipped"] is True
        assert metrics["judge_agrees_with_deterministic"] is None

    def test_a_failing_model_disables_the_judge_instead_of_raising(self):
        class Exploding:
            name = "exploding"

            def generate(self, system_prompt, user_prompt):
                raise RuntimeError("out of memory")

        judge = LLMJudge(backend=Exploding())
        verdict = judge.judge(make_question(), make_result())
        assert verdict.skipped
        assert "out of memory" in verdict.skipped_reason

    def test_a_well_formed_reply_is_parsed(self):
        verdict = parse_judge_reply(
            'Sure! {"correctness": 2, "faithfulness": 1, '
            '"citation_correctness": 0, "reason": "paraphrase"}'
        )
        assert verdict.correctness == 2
        assert verdict.faithfulness == 1
        assert verdict.reason == "paraphrase"

    @pytest.mark.parametrize(
        "reply",
        ["no json here", "{not json}", '{"correctness": 7, "faithfulness": 1, '
         '"citation_correctness": 0}', '{"correctness": 2}'],
    )
    def test_an_unparseable_reply_is_a_recorded_skip(self, reply):
        assert parse_judge_reply(reply).skipped

    def test_the_prompt_and_model_are_recorded(self):
        described = LLMJudge(RAGConfig.default().generation).describe()
        assert described["system_prompt"] == described["system_prompt"].strip()
        assert described["model"]
        assert described["protocol_version"]
        assert described["is_the_model_under_test"] is True

    def test_the_disagreement_rate_is_reported(self):
        class Fixed:
            name = "fixed"

            def generate(self, system_prompt, user_prompt):
                return '{"correctness": 2, "faithfulness": 2, "citation_correctness": 2}'

        judge = LLMJudge(backend=Fixed())
        judge.annotate(make_question(), make_result(), {"metrics": {"answer": {"correct": 0.0}}})
        judge.annotate(make_question(), make_result(), {"metrics": {"answer": {"correct": 1.0}}})
        summary = judge.summary()
        assert summary["judge_comparable_questions"] == 2
        assert summary["judge_deterministic_disagreement_rate"] == pytest.approx(0.5)
        assert summary["judge_parse_failure_rate"] == 0.0

    def test_the_judge_never_decides_abstention(self):
        """Section 19.2 keeps it out of this decision entirely."""
        import src.evaluation.judge as judge_module

        source = judge_module.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "is_abstention" not in text
        assert "abstained" not in text.split('"""')[-1]


class TestHarnessDoesNotChangeAnswers:
    """ARCHITECTURE.md section 26: benchmark instrumentation must not change
    answer behaviour. Asserted as the stronger property: the answer the harness
    scores is the answer a bare ``ask`` produces."""

    #: Timings legitimately differ between two runs of the same query, and a
    #: latency that never varied would mean the clock was not being read.
    TIMING_KEYS = frozenset({"retrieval_s", "generation_s", "latency_s"})

    def test_a_harness_run_result_equals_a_bare_ask(self, offline_config, structured_pdf_path):
        pipeline = DenseRAGPipeline(offline_config)
        pipeline.index(structured_pdf_path)
        question = make_question(id="q001")
        question = Question(
            id="q001",
            document="structured.pdf",
            question="What recall did the hybrid system achieve?",
            answer="It reached 0.87 recall.",
            evidence_pages=(2,),
            question_type="numerical",
            difficulty="easy",
            split="dev",
        )

        bare = pipeline.ask(question.question)
        probed = pipeline.retrieve(question.question, 10)
        instrumented = pipeline.ask(question.question)
        scored = score_result(
            question, instrumented, system="baseline_dense", retrieved_for_metrics=probed
        )

        assert instrumented.answer == bare.answer
        assert instrumented.abstained == bare.abstained
        assert instrumented.cited_pages == bare.cited_pages
        assert [c.to_record() for c in instrumented.citations] == [
            c.to_record() for c in bare.citations
        ]
        assert [r.chunk_id for r in instrumented.retrieved] == [
            r.chunk_id for r in bare.retrieved
        ]
        assert [e.marker for e in instrumented.evidence] == [
            e.marker for e in bare.evidence
        ]
        non_timing = {
            k: v for k, v in instrumented.metadata.items() if k not in self.TIMING_KEYS
        }
        assert non_timing == {
            k: v for k, v in bare.metadata.items() if k not in self.TIMING_KEYS
        }
        # ...and scoring it changed nothing either.
        assert scored.record["answer"] == bare.answer

    def test_the_retrieval_probe_does_not_disturb_the_index(
        self, offline_config, structured_pdf_path
    ):
        pipeline = DenseRAGPipeline(offline_config)
        pipeline.index(structured_pdf_path)
        before = pipeline.ask("How are chunks embedded?")
        pipeline.retrieve("something else entirely", 10)
        after = pipeline.ask("How are chunks embedded?")
        assert before.answer == after.answer
        assert before.cited_pages == after.cited_pages


class TestRunFailuresAreRecordedNotFatal:
    def test_an_unreadable_document_does_not_abort_the_run(self, tmp_path):
        questions = [
            make_question(id="q001"),
            Question(
                id="q002",
                document="missing.pdf",
                question="anything",
                answer="something",
                evidence_pages=(1,),
                question_type="factual",
                difficulty="easy",
            ),
        ]
        run = run_benchmark(
            [questions[1]], documents_dir=tmp_path, config=RAGConfig.offline()
        )
        assert len(run.runs) == 1
        assert run.scores[0].error
        assert run.scores[0].error_category == SYSTEM_RUNTIME_FAILURE
        assert run.records[0]["question_id"] == "q002"


@pytest.mark.parametrize("path", [pathlib.Path("benchmark/questions.json")])
class TestFullBenchmarkSmokeRun:
    """The whole 76-question dataset, offline, as one run.

    Slower than the rest of the suite (the 42-page document dominates it), and
    kept in the default suite anyway: every unit test above uses a synthetic
    record, and this is the only one that proves the four output files can be
    produced from the real dataset, the real PDFs and the real pipeline. The
    weak offline components make the *numbers* meaningless, which is fine --
    what is being checked is that every question is scored, every failure is
    categorised, and nothing raises.
    """

    def test_every_question_is_answered_scored_and_categorised(self, path, tmp_path):
        if not path.exists():  # pragma: no cover - a checkout without the dataset
            pytest.skip("benchmark/questions.json is not present")

        run = run_benchmark_cli(
            questions_path=path,
            documents_dir=pathlib.Path("benchmark/documents"),
            out_dir=tmp_path / "baseline",
            config=RAGConfig.offline(),
        )

        questions = load_questions(path)
        assert len(run.runs) == len(questions)
        assert {s.question_id for s in run.scores} == {q.id for q in questions}

        for score in run.scores:
            assert score.error is None, f"{score.question_id}: {score.error}"
            if score.failed:
                assert score.error_category in ERROR_CATEGORIES, score.question_id
            else:
                assert score.error_category is None

        summary = run.summary()
        assert summary["n_questions"] == len(questions)
        assert summary["n_unanswerable"] == sum(1 for q in questions if q.is_unanswerable)
        assert summary["n_retrieval_scored"] == summary["n_answerable"]
        for key in ("recall_at_5", "full_recall_at_5", "mrr_at_10", "accuracy_all"):
            assert summary[key] is not None, key

        loaded = load_results(tmp_path / "baseline")
        assert len(loaded["per_question"]) == len(questions)
        assert loaded["summary_csv"][0]["system"] == "baseline_dense"

    def test_unanswerable_questions_are_excluded_from_every_retrieval_metric(
        self, path, tmp_path
    ):
        if not path.exists():  # pragma: no cover
            pytest.skip("benchmark/questions.json is not present")
        run = run_benchmark_cli(
            questions_path=path,
            documents_dir=pathlib.Path("benchmark/documents"),
            out_dir=tmp_path / "baseline",
            config=RAGConfig.offline(),
            split="dev",
        )
        excluded = [s for s in run.scores if not s.answerable]
        assert excluded, "the dev split has unanswerable questions"
        for score in excluded:
            assert score.retrieval.scored is False
            assert score.retrieval.excluded_reason == "unanswerable"
            assert score.retrieval.recall == {}
            assert score.retrieval.mrr_at_10 is None
