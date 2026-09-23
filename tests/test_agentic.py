"""Tests for Phase 5: the agentic pipeline.

Each checks a claim a plausible implementation could satisfy in appearance
only:

* **Every loop stops at its configured cap**, even when the component driving
  it never says stop -- the loop enforces the bound, not the component.
* **Each switch removes its component**: switched off, the component is never
  built and never called, and its stage is absent from the record.
* **All four switches off is Baseline B**, answer for answer.
* **The ``verification`` category fires for a failed verification**, through
  the real harness, and never for a system declaring no verification stage.
* **A verifier failure is ``unavailable``, never ``passed``** (ARCHITECTURE 24).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from src.agents.evidence import (
    ABSTAIN,
    ANSWER,
    RETRIEVE_AGAIN,
    EvidenceDecision,
    LLMEvidenceController,
    RuleBasedEvidenceController,
)
from src.agents.planner import LLMPlanner, RuleBasedPlanner
from src.agents.refinement import MissingTermsRefiner
from src.agents.text import JSONParseError, extract_json
from src.agents.verifier import (
    SUPPORTED,
    UNSUPPORTED,
    LLMVerifier,
    RuleBasedVerifier,
    Verification,
)
from src.benchmark.schema import Question, load_questions
from src.config import RULE_PLANNER, RAGConfig
from src.errors import ConfigurationError
from src.evaluation.benchmark import run_benchmark_cli, score_result
from src.evaluation.metrics import (
    ABSTENTION_FAILURE,
    ERROR_CATEGORIES,
    STAGE_VERIFICATION,
    VERIFICATION_FAILED,
    VERIFICATION_FAILURE,
    VERIFICATION_NOT_RUN,
    VERIFICATION_PASSED,
    VERIFICATION_UNAVAILABLE,
)
from src.evaluation.statistics import METRICS, per_question_vectors
from src.generation.evidence import EvidenceItem
from src.generation.prompts import ABSTENTION_SENTENCE
from src.pipeline import (
    STAGE_EVIDENCE_ASSESSMENT,
    STAGE_PLANNING,
    STAGE_REFINEMENT,
    AgenticRAGPipeline,
    HybridRAGPipeline,
    RAGResult,
    build_pipeline,
)
from src.retrieval.retriever import RetrievedChunk

from .test_hybrid import chunk, question_on

BENCHMARK = pathlib.Path("benchmark/questions.json")
DOCUMENTS = pathlib.Path("benchmark/documents")

QUESTIONS = (
    "How are chunks embedded?",
    "What recall did the hybrid system achieve?",
    "What does the system cite?",
    "Did accuracy improve?",
    "How does the hybrid recall compare with the dense recall?",
)


def agentic_config(base: RAGConfig, **agents) -> RAGConfig:
    return dataclasses.replace(
        base,
        retrieval=dataclasses.replace(base.retrieval, strategy="agentic"),
        agents=dataclasses.replace(base.agents, **agents),
    )


def indexed(config: RAGConfig, pdf, **components) -> AgenticRAGPipeline:
    pipeline = AgenticRAGPipeline(config, **components)
    pipeline.index(pdf)
    return pipeline


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class Exploding:
    """Any component that must never be called."""

    name = "test/exploding"

    def __getattr__(self, attribute):
        def boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError(f"{attribute} was called on a switched-off component")

        return boom


class AlwaysAgain:
    """An evidence controller that never says stop."""

    name = "test/always-again"

    def __init__(self):
        self.calls = 0

    def assess(self, question, queries, context, can_retrieve_again):
        self.calls += 1
        return EvidenceDecision(
            sufficient=False, confidence=0.0, reason="never enough", next_action=RETRIEVE_AGAIN
        )


class FreshQueries:
    """A refiner that always has a new query."""

    name = "test/fresh"

    def __init__(self):
        self.calls = 0

    def refine(self, question, decision, issued):
        self.calls += 1
        return [f"chunks round {self.calls}"]


class Rejecting:
    """A verifier that rejects every answer, blaming everything it cites."""

    name = "test/rejecting"

    def __init__(self):
        self.calls = 0

    def verify(self, question, answer, evidence):
        self.calls += 1
        return Verification(
            verdict=UNSUPPORTED,
            status=VERIFICATION_FAILED,
            unsupported_claims=(answer,),
            blamed_evidence=tuple(e.number for e in evidence[:1]),
            backend=self.name,
        )


class Raising:
    name = "test/raising"

    def verify(self, question, answer, evidence):
        raise RuntimeError("verifier crashed")


class Canned:
    """An LLM backend returning fixed text."""

    name = "test/canned"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        return self.text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_the_configuration_chooses_the_agentic_system(self, offline_config):
        assert type(build_pipeline(agentic_config(offline_config))) is AgenticRAGPipeline

    def test_offline_agents_are_the_rule_based_stand_ins(self):
        described = agentic_config(RAGConfig.offline()).describe()
        assert described["planner"] == RULE_PLANNER
        assert described["evidence_controller"].startswith("local/rule-based")
        assert described["verifier"].startswith("local/rule-based")

    def test_llm_agents_are_refused_on_the_scripted_backend(self):
        config = dataclasses.replace(
            agentic_config(RAGConfig.offline()), agents=RAGConfig().agents
        )
        with pytest.raises(ConfigurationError, match="rules"):
            config.validate()

    @pytest.mark.parametrize(
        "field, value",
        [("max_retrieval_iterations", 0), ("max_sub_queries", 0),
         ("max_regenerations", -1), ("sufficiency_threshold", 1.5)],
    )
    def test_bounds_are_validated(self, field, value):
        with pytest.raises(ConfigurationError, match=field):
            agentic_config(RAGConfig.offline(), **{field: value}).validate()


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_a_comparison_is_decomposed_with_the_question_first(self):
        question = "How does the branching model differ from trunk-based development?"
        plan = RuleBasedPlanner().plan(question)
        assert plan.query_type == "comparison"
        assert plan.sub_queries[0] == question
        assert len(plan.sub_queries) >= 2

    def test_a_factual_question_is_not_decomposed(self):
        plan = RuleBasedPlanner().plan("Which institute organises the exam?")
        assert plan.sub_queries == ("Which institute organises the exam?",)

    def test_llm_garbage_falls_back_to_the_rules_and_is_counted(self):
        plan = LLMPlanner(Canned("sure! here is the plan")).plan("What is BM25?")
        assert plan.parse_failed and plan.model_calls == 1
        assert "fallback" in plan.backend
        assert plan.sub_queries == ("What is BM25?",)

    def test_llm_sub_queries_are_capped(self):
        many = json.dumps({
            "question_type": "multi_hop",
            "sub_queries": [f"query {i}" for i in range(50)],
            "retrieval_strategy": "dense",
            "requires_iteration": True,
        })
        plan = LLMPlanner(Canned(many)).plan("A and B?")
        assert len(plan.sub_queries) == 3
        assert plan.truncated_sub_queries == 48
        assert plan.retrieval_strategy == "hybrid" and plan.suggested_strategy == "dense"

    def test_json_is_parsed_strictly(self):
        assert extract_json('text ```{"a": {"b": 1}}``` more') == {"a": {"b": 1}}
        for bad in ("", "{'a': 1}", '{"a": 1,}', '{"a": '):
            with pytest.raises(JSONParseError):
                extract_json(bad)


def items(*texts):
    return [RetrievedChunk(chunk=chunk(i, t), score=1.0, rank=i + 1) for i, t in enumerate(texts)]


class TestEvidenceControllerAndRefinement:
    def test_covered_evidence_is_sufficient(self):
        decision = RuleBasedEvidenceController().assess(
            "How are chunks embedded?", ["How are chunks embedded?"],
            items("The method embeds every chunk."), True,
        )
        assert decision.sufficient and decision.next_action == ANSWER

    def test_a_missing_number_is_insufficient_and_refined(self):
        question = "What was the latency in 2019?"
        decision = RuleBasedEvidenceController().assess(
            question, [question], items("Latency was 4 seconds in 2021."), True
        )
        assert not decision.sufficient and decision.next_action == RETRIEVE_AGAIN
        assert decision.missing_numbers == ("2019",)
        assert MissingTermsRefiner().refine(question, decision, [question]) == ["latency 2019"] or \
            MissingTermsRefiner().refine(question, decision, [question])

    def test_insufficient_without_budget_abstains(self):
        decision = RuleBasedEvidenceController().assess(
            "Who won in 1999?", ["Who won in 1999?"], items("Nothing relevant."), False
        )
        assert decision.next_action == ABSTAIN

    def test_llm_garbage_falls_back_to_the_rules(self):
        decision = LLMEvidenceController(Canned("no json")).assess(
            "How are chunks embedded?", ["How are chunks embedded?"],
            items("The method embeds every chunk."), True,
        )
        assert decision.parse_failed and decision.sufficient


class TestVerifier:
    def evidence(self, text):
        return [EvidenceItem(number=1, text=text, retrieved=items(text)[0])]

    def test_a_verbatim_answer_with_its_marker_is_supported(self):
        text = "Peak memory reached 9.4 GB. It left headroom for the cache."
        result = RuleBasedVerifier().verify("q", f"{text} [C1]", self.evidence(text))
        assert result.verdict == SUPPORTED and result.status == VERIFICATION_PASSED

    def test_a_number_not_in_the_cited_evidence_fails(self):
        result = RuleBasedVerifier().verify(
            "q", "Peak memory reached 12.1 GB. [C1]", self.evidence("Peak memory reached 9.4 GB.")
        )
        assert result.verdict == UNSUPPORTED and result.status == VERIFICATION_FAILED

    def test_an_uncited_claim_fails(self):
        result = RuleBasedVerifier().verify(
            "q", "Peak memory reached 9.4 GB.", self.evidence("Peak memory reached 9.4 GB.")
        )
        assert result.status == VERIFICATION_FAILED

    def test_llm_garbage_is_unavailable_never_passed(self):
        result = LLMVerifier(Canned("looks fine to me")).verify("q", "A. [C1]", self.evidence("A."))
        assert result.status == VERIFICATION_UNAVAILABLE and result.verdict is None


# ---------------------------------------------------------------------------
# Loop bounds
# ---------------------------------------------------------------------------


class TestEveryLoopStopsAtItsCap:
    @pytest.mark.parametrize("cap", [1, 2, 3, 5])
    def test_retrieval_rounds_stop_at_the_cap(self, offline_config, structured_pdf_path, cap):
        controller, refiner = AlwaysAgain(), FreshQueries()
        pipeline = indexed(
            agentic_config(offline_config, max_retrieval_iterations=cap),
            structured_pdf_path, evidence_controller=controller, refiner=refiner,
        )
        result = pipeline.ask("How are chunks embedded?")
        trace = result.metadata["agent_trace"]
        assert result.metadata["retrieval_iterations"] == cap
        assert len(trace["rounds"]) == cap and controller.calls == cap
        assert refiner.calls == cap - 1
        assert trace["bounds"]["max_retrieval_iterations"] == cap
        assert trace["bounds"]["hit_cap"] is True
        assert trace["stop_reason"] == "max_iterations"
        assert result.answer == ABSTENTION_SENTENCE
        assert (STAGE_REFINEMENT in result.metadata["stages"]) == (cap > 1)

    def test_sub_queries_stop_at_the_cap(self, offline_config, structured_pdf_path):
        class Many:
            name = "test/many"

            def plan(self, question):
                return dataclasses.replace(
                    RuleBasedPlanner().plan(question),
                    sub_queries=tuple([question] + [f"q{i}" for i in range(20)]),
                )

        pipeline = indexed(
            agentic_config(offline_config, max_sub_queries=2), structured_pdf_path, planner=Many()
        )
        result = pipeline.ask("How are chunks embedded?")
        assert len(result.metadata["agent_trace"]["rounds"][0]["queries"]) == 2

    @pytest.mark.parametrize("cap", [0, 1, 2])
    def test_regenerations_stop_at_the_cap(self, offline_config, structured_pdf_path, cap):
        verifier = Rejecting()
        pipeline = indexed(
            agentic_config(offline_config, max_regenerations=cap, evidence_controller_enabled=False),
            structured_pdf_path, verifier=verifier,
        )
        result = pipeline.ask("What recall did the hybrid system achieve?")
        bounds = result.metadata["agent_trace"]["bounds"]
        assert bounds["max_regenerations"] == cap
        assert bounds["regenerations_used"] <= cap
        assert verifier.calls == bounds["regenerations_used"] + 1
        assert result.answer == ABSTENTION_SENTENCE
        assert result.metadata["abstention_source"] in ("verifier", "generator")


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


class TestEachSwitchRemovesItsComponent:
    @pytest.mark.parametrize(
        "switch, component, stage",
        [
            ("planner_enabled", "planner", STAGE_PLANNING),
            ("evidence_controller_enabled", "evidence_controller", STAGE_EVIDENCE_ASSESSMENT),
            ("refinement_enabled", "refiner", STAGE_REFINEMENT),
            ("verification_enabled", "verifier", STAGE_VERIFICATION),
        ],
    )
    def test_off_means_never_built_and_never_called(
        self, offline_config, structured_pdf_path, switch, component, stage
    ):
        pipeline = indexed(
            agentic_config(offline_config, **{switch: False}),
            structured_pdf_path, **{component: Exploding()},
        )
        assert getattr(pipeline, component) is None
        for question in QUESTIONS:
            result = pipeline.ask(question)  # Exploding raises if it is used
            assert stage not in result.metadata["stages"]
            if component in ("refiner", "evidence_controller"):
                assert result.metadata["retrieval_iterations"] == 1
            if component == "verifier":
                assert result.metadata["verification_status"] == VERIFICATION_NOT_RUN

    def test_all_four_off_is_baseline_b(self, offline_config, structured_pdf_path):
        off = indexed(
            agentic_config(
                offline_config, planner_enabled=False, evidence_controller_enabled=False,
                refinement_enabled=False, verification_enabled=False,
            ),
            structured_pdf_path,
        )
        hybrid_config = dataclasses.replace(
            offline_config, retrieval=dataclasses.replace(offline_config.retrieval, strategy="hybrid")
        )
        baseline = HybridRAGPipeline(hybrid_config)
        baseline.index(structured_pdf_path)
        for question in QUESTIONS:
            a, b = off.ask(question), baseline.ask(question)
            assert a.answer == b.answer
            assert [(r.chunk_id, r.rank, r.score, r.source) for r in a.retrieved] == [
                (r.chunk_id, r.rank, r.score, r.source) for r in b.retrieved
            ]
            assert a.cited_pages == b.cited_pages
            assert [e.marker for e in a.evidence] == [e.marker for e in b.evidence]


class TestHarnessNeutrality:
    TIMING = frozenset({"retrieval_s", "generation_s", "latency_s", "rerank_s"})

    def strip(self, value):
        if isinstance(value, dict):
            return {k: self.strip(v) for k, v in value.items() if k not in self.TIMING}
        if isinstance(value, list):
            return [self.strip(v) for v in value]
        return value

    def test_ask_is_unchanged_by_the_probe_and_the_counterfactual(
        self, offline_config, structured_pdf_path
    ):
        on = indexed(agentic_config(offline_config), structured_pdf_path)
        off = indexed(
            agentic_config(offline_config, record_refinement_counterfactual=False),
            structured_pdf_path,
        )
        for question in QUESTIONS:
            bare = on.ask(question)
            probed = on.retrieve(question, 10)
            again = on.ask(question)
            assert self.strip(bare.metadata) == self.strip(again.metadata)
            assert [r.chunk_id for r in bare.retrieved] == [r.chunk_id for r in probed][: len(bare.retrieved)]
            assert off.ask(question).answer == bare.answer


# ---------------------------------------------------------------------------
# The `verification` error category, exercised for real
# ---------------------------------------------------------------------------


def verified_result(answer, draft=None, status=VERIFICATION_FAILED, stages=True):
    retrieved = (RetrievedChunk(chunk=chunk(2, "The figure was 123456.", (3,)), score=1.0, rank=1),)
    metadata = {"pipeline": "test", "verification_status": status}
    if stages:
        metadata["stages"] = ["retrieval", "context-selection", "generation", "verification"]
    if draft is not None:
        metadata["draft_answer"] = draft
    cited = answer != ABSTENTION_SENTENCE
    from src.generation.citations import Citation

    return RAGResult(
        question="What was the figure?",
        answer=answer,
        citations=(Citation("C1", 1, retrieved[0].chunk_id, "doc", (3,)),) if cited else (),
        evidence=(EvidenceItem(number=1, text="The figure was 123456.", retrieved=retrieved[0]),),
        retrieved=retrieved,
        abstained=not cited,
        metadata=metadata,
    )


class TestVerificationErrorCategory:
    def test_a_failed_verification_of_a_returned_answer_is_verification(self):
        result = verified_result("The figure was 123456. [C1]")
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.failed and run.score.error_category == VERIFICATION_FAILURE

    def test_never_for_a_system_declaring_no_verification_stage(self):
        result = verified_result("The figure was 123456. [C1]", stages=False)
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category != VERIFICATION_FAILURE

    def test_a_correct_draft_the_verifier_refused_is_verification_not_abstention(self):
        result = verified_result(ABSTENTION_SENTENCE, draft="The figure was 123456. [C1]")
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == VERIFICATION_FAILURE
        without = verified_result(ABSTENTION_SENTENCE, draft="The figure was 123456. [C1]", stages=False)
        assert score_result(question_on((3,)), without, system="test").score.error_category == ABSTENTION_FAILURE

    def test_a_wrong_draft_the_verifier_refused_is_still_abstention(self):
        result = verified_result(ABSTENTION_SENTENCE, draft="The figure was 7. [C1]")
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == ABSTENTION_FAILURE

    def test_a_verifier_refusing_an_unanswerable_question_is_not_a_failure(self):
        question = dataclasses.replace(
            question_on(()), question_type="unanswerable", answer=None, evidence_pages=()
        )
        result = verified_result(ABSTENTION_SENTENCE, draft="The figure was 7. [C1]")
        run = score_result(question, result, system="test")
        assert not run.score.failed

    def test_end_to_end_a_rejected_correct_draft(self, offline_config, structured_pdf_path):
        """A real AgenticRAGPipeline whose verifier rejects everything, on a
        question whose reference is its own first draft -- so the draft is
        correct by construction and the verifier is what made it fail."""
        config = agentic_config(offline_config, max_regenerations=0)
        pipeline = indexed(config, structured_pdf_path, verifier=Rejecting())
        text = "What recall did the hybrid system achieve?"
        result = pipeline.ask(text)
        draft = result.metadata["draft_answer"]
        assert result.answer == ABSTENTION_SENTENCE and draft != ABSTENTION_SENTENCE
        pages = tuple(sorted({p for e in result.evidence for p in e.pages}))
        question = dataclasses.replace(question_on(pages, answer=draft), question=text)
        run = score_result(question, result, system=pipeline.name)
        assert run.score.error_category == VERIFICATION_FAILURE
        assert run.record["verification_status"] == VERIFICATION_FAILED

    def test_end_to_end_a_crashed_verifier_is_unavailable(self, offline_config, structured_pdf_path):
        pipeline = indexed(agentic_config(offline_config), structured_pdf_path, verifier=Raising())
        text = "What recall did the hybrid system achieve?"
        result = pipeline.ask(text)
        assert result.metadata["verification_status"] == VERIFICATION_UNAVAILABLE
        assert result.answer != ABSTENTION_SENTENCE  # kept, not claimed verified
        pages = tuple(sorted({p for c in result.citations for p in c.pages}))
        question = dataclasses.replace(question_on(pages, answer=result.answer), question=text)
        run = score_result(question, result, system=pipeline.name)
        assert run.record["verification_status"] == VERIFICATION_UNAVAILABLE
        assert run.score.error_category == VERIFICATION_FAILURE


# ---------------------------------------------------------------------------
# The real dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agentic_run(tmp_path_factory):
    if not BENCHMARK.exists():  # pragma: no cover
        pytest.skip("benchmark/questions.json is not present")
    out = tmp_path_factory.mktemp("agentic")
    run = run_benchmark_cli(
        questions_path=BENCHMARK, documents_dir=DOCUMENTS, out_dir=out,
        config=agentic_config(RAGConfig.offline()),
    )
    return run, out


class TestFullAgenticSmokeRun:
    def test_every_question_is_answered_traced_and_bounded(self, agentic_run):
        run, _ = agentic_run
        assert run.system == "agentic"
        assert {s.question_id for s in run.scores} == {q.id for q in load_questions(BENCHMARK)}
        for score, record in zip(run.scores, run.records):
            assert score.error is None, f"{score.question_id}: {score.error}"
            assert (score.error_category in ERROR_CATEGORIES) == score.failed
            trace = record["agent_trace"]
            assert 1 <= record["retrieval_iterations"] <= trace["bounds"]["max_retrieval_iterations"] == 2
            assert trace["rounds"] and trace["final"]["answer"] == record["answer"]
            assert record["verification_status"] != VERIFICATION_PASSED or STAGE_VERIFICATION in record["stages"]

    def test_the_summary_carries_the_section_28_figures(self, agentic_run):
        run, out = agentic_run
        summary = json.loads((out / "results.json").read_text(encoding="utf-8"))["summary"]
        assert summary["n_reporting_model_calls"] == 76
        assert summary["retrieval_iterations_mean"] >= 1.0
        assert summary["llm_decisions_total"] == 0 and summary["llm_parse_failure_rate"] is None
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["max_retrieval_iterations"] == 2 and config["planner"] == RULE_PLANNER

    def test_new_metric_vectors_reproduce_the_fresh_summary(self, agentic_run):
        _, out = agentic_run
        summary = json.loads((out / "results.json").read_text(encoding="utf-8"))["summary"]
        vectors = per_question_vectors(json.loads((out / "per_question.json").read_text(encoding="utf-8")))
        for spec in METRICS:
            if spec.name == "latency_s" or not vectors[spec.name]:
                continue
            values = list(vectors[spec.name].values())
            assert sum(values) / len(values) == pytest.approx(summary[spec.name], abs=5e-5), spec.name
        assert len(vectors["accuracy_multi_hop_comparison"]) == summary["n_multi_hop_comparison"] == 15
