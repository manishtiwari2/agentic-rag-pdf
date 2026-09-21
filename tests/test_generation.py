"""Citation resolution, abstention detection and grounded generation.

These are the deterministic guarantees the system's credibility rests on: a
citation that points at the right page, an out-of-range marker that disappears
instead of being guessed, and a refusal that is recognised as a refusal.
"""

from __future__ import annotations

import pytest

from src.chunking.base import Chunk
from src.config import GenerationConfig
from src.generation.abstention import (
    ABSTENTION_PATTERNS_VERSION,
    detect_abstention,
    is_abstention,
)
from src.generation.backends import ScriptedBackend
from src.generation.citations import resolve_citations
from src.generation.evidence import build_evidence, evidence_blocks
from src.generation.generator import GroundedGenerator
from src.generation.prompts import (
    ABSTENTION_SENTENCE,
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_user_prompt,
)
from src.retrieval.retriever import RetrievedChunk

PAGES = [(4,), (9,), (11, 12)]


def _retrieved() -> list[RetrievedChunk]:
    texts = [
        "The proposed method improves recall by reranking the candidate set.",
        "Peak GPU memory reached 9.4 GB during generation.",
        "The evaluation used seventy-five questions across eight documents.",
    ]
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"doc:structure:{i:05d}",
                document_id="doc",
                text=text,
                pages=PAGES[i],
                section=f"Section {i}",
                index=i,
            ),
            score=1.0 - i * 0.1,
            rank=i + 1,
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def evidence():
    return build_evidence(_retrieved(), GenerationConfig())


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_evidence_blocks_contain_no_page_information(self, evidence):
        """The model cannot copy a page number it was never shown."""
        prompt = build_user_prompt("What improved?", evidence_blocks(evidence))
        assert "page" not in prompt.lower()
        assert "9" not in prompt.replace("9.4 GB", "")  # no page 9 leaked
        for item in evidence:
            assert item.chunk_id not in prompt
            assert str(item.chunk.section) not in prompt

    def test_markers_are_numbered_from_one_in_rank_order(self, evidence):
        assert [item.marker for item in evidence] == ["C1", "C2", "C3"]
        assert evidence[0].chunk.pages == (4,)

    def test_system_prompt_names_the_canonical_refusal(self):
        assert ABSTENTION_SENTENCE in SYSTEM_PROMPT

    def test_prompt_round_trips(self, evidence):
        prompt = build_user_prompt("What improved?", evidence_blocks(evidence))
        question, blocks = parse_user_prompt(prompt)
        assert question == "What improved?"
        assert [n for n, _ in blocks] == [1, 2, 3]
        assert blocks[1][1].startswith("Peak GPU memory")


class TestEvidenceBudget:
    def test_duplicate_passages_are_dropped(self):
        retrieved = _retrieved()
        duplicate = RetrievedChunk(
            chunk=Chunk(
                chunk_id="doc:structure:00099",
                document_id="doc",
                text=retrieved[0].chunk.text,
                pages=(4,),
                index=99,
            ),
            score=0.5,
            rank=4,
        )
        items = build_evidence(retrieved + [duplicate], GenerationConfig())
        assert len(items) == 3

    def test_context_budget_is_respected(self):
        config = GenerationConfig(max_context_chars=80, max_evidence_chars=80)
        items = build_evidence(_retrieved(), config)
        assert len(items) == 1  # the top hit always survives

    def test_oversized_block_is_truncated_not_dropped(self):
        config = GenerationConfig(max_context_chars=200, max_evidence_chars=30)
        items = build_evidence(_retrieved(), config)
        assert items[0].truncated
        assert items[0].text.endswith("[...]")


# ---------------------------------------------------------------------------
# Citation resolution
# ---------------------------------------------------------------------------


class TestMarkerSyntax:
    """Marker forms, checked through the resolver that actually runs.

    Tested here rather than against a separate extraction helper: two
    implementations of the same syntax can drift apart, and only one of them
    decides what gets cited.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Claim [C1].", [1]),
            ("Claim [C1] and [C2].", [1, 2]),
            ("Claim [C1, C2].", [1, 2]),
            ("Claim [C1;C3].", [1, 3]),
            ("Claim [c2].", [2]),
            ("Claim [2].", [2]),  # small models drop the prefix
            ("Claim [ C1 ].", [1]),
        ],
    )
    def test_marker_forms(self, text, expected, evidence):
        resolution = resolve_citations(text, evidence)
        assert [c.evidence_number for c in resolution.citations] == expected
        assert resolution.dropped_markers == ()

    def test_bracketed_prose_is_not_a_marker(self, evidence):
        resolution = resolve_citations("See [the appendix] for details.", evidence)
        assert resolution.citations == ()
        assert resolution.dropped_markers == ()
        assert "[the appendix]" in resolution.answer  # and is left in place


class TestCitationResolution:
    def test_markers_resolve_to_the_pages_of_their_chunks(self, evidence):
        resolution = resolve_citations("Recall improved [C1].", evidence)
        assert len(resolution.citations) == 1
        citation = resolution.citations[0]
        assert citation.marker == "C1"
        assert citation.pages == (4,)
        assert citation.label == "page 4"
        assert citation.section == "Section 0"
        assert resolution.cited_pages == (4,)

    def test_page_spanning_chunk_cites_both_pages(self, evidence):
        resolution = resolve_citations("Seventy-five questions [C3].", evidence)
        assert resolution.citations[0].pages == (11, 12)
        assert resolution.citations[0].label == "pages 11-12"
        assert resolution.cited_pages == (11, 12)

    def test_multiple_markers_resolve_independently(self, evidence):
        resolution = resolve_citations("A [C1]. B [C2].", evidence)
        assert [c.evidence_number for c in resolution.citations] == [1, 2]
        assert resolution.cited_pages == (4, 9)

    def test_repeated_marker_is_cited_once(self, evidence):
        resolution = resolve_citations("A [C1]. B [C1].", evidence)
        assert len(resolution.citations) == 1

    def test_out_of_range_marker_is_dropped_not_guessed(self, evidence):
        """There is no defensible way to guess which of three it meant."""
        resolution = resolve_citations("A claim [C7].", evidence)
        assert resolution.citations == ()
        assert resolution.dropped_markers == (7,)
        assert "[C7]" not in resolution.answer
        assert resolution.answer == "A claim."

    def test_zero_marker_is_dropped(self, evidence):
        resolution = resolve_citations("A claim [C0].", evidence)
        assert resolution.citations == ()
        assert resolution.dropped_markers == (0,)

    def test_mixed_group_keeps_the_valid_marker_only(self, evidence):
        resolution = resolve_citations("A claim [C1, C9].", evidence)
        assert [c.evidence_number for c in resolution.citations] == [1]
        assert resolution.dropped_markers == (9,)
        assert resolution.answer == "A claim [C1]."

    def test_dropping_leaves_no_double_spaces(self, evidence):
        resolution = resolve_citations("Alpha [C9] beta [C8] gamma.", evidence)
        assert "  " not in resolution.answer
        assert resolution.answer == "Alpha beta gamma."

    def test_raw_answer_is_preserved_for_inspection(self, evidence):
        resolution = resolve_citations("A claim [C7].", evidence)
        assert resolution.raw_answer == "A claim [C7]."

    def test_fabricated_page_mention_is_flagged(self, evidence):
        """The model saw no pages, so any page it writes is invented."""
        resolution = resolve_citations("As stated on page 3 [C1].", evidence)
        assert resolution.fabricated_page_mentions
        assert resolution.citations[0].pages == (4,)  # the real page wins

    def test_answer_without_markers_resolves_to_no_citations(self, evidence):
        resolution = resolve_citations("An unsupported claim.", evidence)
        assert resolution.citations == ()
        assert resolution.dropped_markers == ()


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


class TestAbstention:
    def test_canonical_sentence_is_detected(self):
        verdict = detect_abstention(ABSTENTION_SENTENCE)
        assert verdict.abstained
        assert verdict.reason == "canonical"
        assert verdict.patterns_version == ABSTENTION_PATTERNS_VERSION

    def test_canonical_sentence_with_whitespace_and_markers(self):
        assert is_abstention(f"  {ABSTENTION_SENTENCE} [C1] ")

    @pytest.mark.parametrize(
        "text",
        [
            "I could not find that information in the document.",
            "The document does not contain this information.",
            "The provided evidence does not mention the release date.",
            "There is no information about pricing in the evidence.",
            "Insufficient evidence to answer.",
            "Unable to determine from the supplied passages.",
            "This question cannot be answered from the document.",
        ],
    )
    def test_paraphrases_are_detected(self, text):
        assert is_abstention(text)
        assert detect_abstention(text).reason == "paraphrase"

    def test_a_gap_mentioned_in_passing_is_an_answer(self):
        """BENCHMARK_SPEC 13.2: this is the case that must not count."""
        text = (
            "The document does not state the exact date, but the experiment ran "
            "in March."
        )
        assert not is_abstention(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The document does not specify the vendor; however, it lists three "
            "suppliers by name.",
            "There is no figure for 2023, although the 2022 figure was 41%.",
        ],
    )
    def test_contrastive_continuations_are_answers(self, text):
        assert not is_abstention(text)

    def test_a_long_answer_is_never_an_abstention(self):
        text = (
            "The document does not contain a single summary figure. "
        ) + "It does report results for each configuration in turn. " * 6
        assert not is_abstention(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Recall improved to 0.83 [C1].",
            "The peak memory was 9.4 GB.",
            "Yes.",
        ],
    )
    def test_ordinary_answers_are_not_abstentions(self, text):
        assert not is_abstention(text)

    def test_empty_output_counts_as_abstention(self):
        verdict = detect_abstention("   ")
        assert verdict.abstained
        assert verdict.reason == "empty"


# ---------------------------------------------------------------------------
# The scripted backend and the generator
# ---------------------------------------------------------------------------


class TestScriptedBackend:
    def test_answers_from_the_best_matching_evidence(self, evidence):
        prompt = build_user_prompt(
            "How much GPU memory was used?", evidence_blocks(evidence)
        )
        answer = ScriptedBackend().generate(SYSTEM_PROMPT, prompt)
        assert "9.4 GB" in answer
        assert "[C2]" in answer

    def test_refuses_when_nothing_matches(self, evidence):
        prompt = build_user_prompt(
            "What is the capital of Portugal?", evidence_blocks(evidence)
        )
        assert ScriptedBackend().generate(SYSTEM_PROMPT, prompt) == ABSTENTION_SENTENCE

    def test_refuses_with_no_evidence(self):
        prompt = build_user_prompt("Anything?", [])
        assert ScriptedBackend().generate(SYSTEM_PROMPT, prompt) == ABSTENTION_SENTENCE


class TestGroundedGenerator:
    def test_answer_carries_resolved_citations(self):
        generator = GroundedGenerator(GenerationConfig(backend="scripted"))
        result = generator.generate("How much GPU memory was used?", _retrieved())
        assert not result.abstained
        assert result.cited_pages == (9,)
        assert result.citations[0].chunk_id == "doc:structure:00001"

    def test_no_retrieval_means_no_model_call(self):
        """With nothing retrieved, answering could only come from memory."""

        class ExplodingBackend:
            name = "exploding"

            def generate(self, system_prompt: str, user_prompt: str) -> str:
                raise AssertionError("the backend must not be called")

        generator = GroundedGenerator(GenerationConfig(), backend=ExplodingBackend())
        result = generator.generate("anything", [])
        assert result.abstained
        assert result.abstention_reason == "no_evidence"
        assert result.answer == ABSTENTION_SENTENCE

    def test_refusal_carries_no_citations(self):
        generator = GroundedGenerator(GenerationConfig(backend="scripted"))
        result = generator.generate("What is the capital of Portugal?", _retrieved())
        assert result.abstained
        assert result.citations == ()
        assert result.cited_pages == ()

    def test_abstention_normalises_to_the_canonical_sentence(self):
        class ParaphrasingBackend:
            name = "paraphrasing"

            def generate(self, system_prompt: str, user_prompt: str) -> str:
                return "The document does not contain this information."

        generator = GroundedGenerator(GenerationConfig(), backend=ParaphrasingBackend())
        result = generator.generate("anything", _retrieved())
        assert result.abstained
        assert result.answer == ABSTENTION_SENTENCE
        assert result.raw_answer == "The document does not contain this information."

    def test_out_of_range_markers_are_reported_on_the_result(self):
        class WildBackend:
            name = "wild"

            def generate(self, system_prompt: str, user_prompt: str) -> str:
                return "Recall improved [C42]."

        generator = GroundedGenerator(GenerationConfig(), backend=WildBackend())
        result = generator.generate("anything", _retrieved())
        assert result.dropped_markers == (42,)
        assert result.citations == ()
        assert result.metadata["uncited"] is True
