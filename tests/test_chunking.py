"""Chunk boundaries, page spans and metadata survival.

The page-attribution tests are the important ones. A chunker that silently
shifts a page number produces citations that look right and point at the wrong
page, which is the failure EXPERIMENT_PLAN.md section 4 flags as "a silent
off-by-one looks like a retrieval bug".
"""

from __future__ import annotations

import pytest

from src.chunking.base import Chunk, build_chunker
from src.chunking.fixed import FixedSizeChunker
from src.chunking.segmentation import (
    SegmentKind,
    segment_document,
    slice_page_offsets,
)
from src.chunking.structure import StructureAwareChunker
from src.config import ChunkingConfig
from src.errors import ChunkingError
from src.ingestion.document import Block, Document, Page

from . import pdf_fixtures as pdfs


def _chunkers(config: ChunkingConfig | None = None):
    config = config or ChunkingConfig(chunk_size=400, chunk_overlap=80)
    from dataclasses import replace

    return [
        StructureAwareChunker(replace(config, strategy="structure")),
        FixedSizeChunker(replace(config, strategy="fixed")),
    ]


# ---------------------------------------------------------------------------
# Properties both strategies must satisfy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker", _chunkers(), ids=lambda c: c.name)
class TestBothStrategies:
    def test_required_metadata_is_present_on_every_chunk(
        self, chunker, structured_document
    ):
        """DD-010: this metadata must survive to the citation."""
        chunks = chunker.chunk(structured_document)
        assert chunks
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.document_id == structured_document.document_id
            assert chunk.pages and all(p >= 1 for p in chunk.pages)
            assert chunk.page_number == chunk.pages[0]
            assert chunk.page_span == (chunk.pages[0], chunk.pages[-1])
            # `section` is nullable, but the attribute must exist on every chunk.
            assert hasattr(chunk, "section")

    def test_chunk_ids_are_unique_and_deterministic(self, chunker, structured_document):
        first = chunker.chunk(structured_document)
        second = chunker.chunk(structured_document)
        ids = [c.chunk_id for c in first]
        assert len(ids) == len(set(ids))
        assert ids == [c.chunk_id for c in second]

    def test_indices_are_contiguous(self, chunker, structured_document):
        chunks = chunker.chunk(structured_document)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_pages_are_within_the_document(self, chunker, structured_document):
        chunks = chunker.chunk(structured_document)
        valid = {p.page_number for p in structured_document.pages}
        for chunk in chunks:
            assert set(chunk.pages) <= valid

    def test_no_chunk_is_empty(self, chunker, structured_document):
        assert all(c.text.strip() for c in chunker.chunk(structured_document))


# ---------------------------------------------------------------------------
# Page attribution
# ---------------------------------------------------------------------------


class TestPageAttribution:
    """The off-by-one tests."""

    @pytest.mark.parametrize("strategy", ["structure", "fixed"])
    def test_chunk_pages_match_the_markers_in_its_text(self, parser, strategy):
        """Every marker in a chunk's text must name a page the chunk claims."""
        document = parser.parse_bytes(pdfs.multipage_pdf(6), "m.pdf")
        chunker = build_chunker(
            ChunkingConfig(strategy=strategy, chunk_size=300, chunk_overlap=60)
        )
        for chunk in chunker.chunk(document):
            for page_number in range(1, 7):
                marker = pdfs.PAGE_MARKER.format(n=page_number)
                if marker in chunk.text:
                    assert page_number in chunk.pages, (
                        f"{chunk.chunk_id} contains {marker} but reports pages "
                        f"{chunk.pages}"
                    )

    @pytest.mark.parametrize("strategy", ["structure", "fixed"])
    def test_every_page_is_covered_by_some_chunk(self, parser, strategy):
        document = parser.parse_bytes(pdfs.multipage_pdf(6), "m.pdf")
        chunker = build_chunker(
            ChunkingConfig(strategy=strategy, chunk_size=300, chunk_overlap=60)
        )
        covered = {p for c in chunker.chunk(document) for p in c.pages}
        assert covered == {1, 2, 3, 4, 5, 6}

    @pytest.mark.parametrize("strategy", ["structure", "fixed"])
    def test_page_spanning_chunk_reports_both_pages(self, structured_document, strategy):
        """The paragraph broken across the page break must claim pages 1 and 2."""
        chunker = build_chunker(
            ChunkingConfig(strategy=strategy, chunk_size=600, chunk_overlap=80)
        )
        chunks = chunker.chunk(structured_document)
        spanning = [c for c in chunks if c.spans_pages]
        assert spanning, "expected at least one chunk to span the page break"
        for chunk in spanning:
            assert chunk.page_span == (1, 2)
            assert chunk.page_number == 1  # the citation anchor is where it starts

    def test_empty_pages_do_not_shift_attribution(self, parser):
        document = parser.parse_bytes(pdfs.pdf_with_empty_page(3, 2), "e.pdf")
        chunks = build_chunker(ChunkingConfig(chunk_size=400)).chunk(document)
        for chunk in chunks:
            if pdfs.PAGE_MARKER.format(n=3) in chunk.text:
                assert 3 in chunk.pages
                break
        else:
            pytest.fail("no chunk contained page 3's marker")

    def test_slice_page_offsets_narrows_to_the_pages_in_range(self):
        offsets = ((0, 4), (50, 5), (120, 6))
        assert slice_page_offsets(offsets, 0, 40) == ((0, 4),)
        assert slice_page_offsets(offsets, 60, 110) == ((0, 5),)
        assert slice_page_offsets(offsets, 40, 130) == ((0, 4), (10, 5), (80, 6))


# ---------------------------------------------------------------------------
# Structure-aware behaviour
# ---------------------------------------------------------------------------


class TestStructureAware:
    def test_table_gets_its_own_chunk(self, structured_document):
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=4000, chunk_overlap=0)
        ).chunk(structured_document)
        tables = [c for c in chunks if c.metadata.get("kind") == "table"]
        assert len(tables) == 1
        table = tables[0]
        assert table.text.startswith("| System | Recall | Latency |")
        # Even with a chunk size large enough to swallow the document, the table
        # is not merged with the prose around it.
        assert "Accuracy improved" not in table.text

    def test_table_rows_stay_with_their_header(self, structured_document):
        chunks = StructureAwareChunker(ChunkingConfig()).chunk(structured_document)
        table = next(c for c in chunks if c.metadata.get("kind") == "table")
        assert "| System | Recall | Latency |" in table.text
        assert "| Hybrid | 0.83 | 5.1 s |" in table.text

    def test_heading_starts_a_new_chunk_and_leads_it(self, structured_document):
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=2000, chunk_overlap=0)
        ).chunk(structured_document)
        text_chunks = [c for c in chunks if c.metadata.get("kind") != "table"]
        # Each heading opens a chunk rather than landing in the middle of one.
        assert text_chunks[0].text.startswith("1 Introduction")
        starts = {c.text.split("\n", 1)[0] for c in text_chunks}
        assert {"1 Introduction", "2 Method", "3 Results"} <= starts

    def test_section_is_recorded_on_every_chunk(self, structured_document):
        chunks = StructureAwareChunker(ChunkingConfig()).chunk(structured_document)
        assert all(c.section for c in chunks)
        assert {c.section for c in chunks} == {"1 Introduction", "2 Method", "3 Results"}

    def test_section_path_is_tracked_for_nested_headings(self):
        document = _document_from_blocks(
            [
                (1, "2 Method", 18.0),
                (1, "2.1 Data", 14.0),
                (1, "We collected documents from a public archive over two years.", 11.0),
            ]
        )
        chunks = StructureAwareChunker(ChunkingConfig()).chunk(document)
        assert chunks[-1].metadata["section_path"] == "2 Method > 2.1 Data"

    def test_paragraph_is_not_split_when_it_fits(self):
        paragraph = "This paragraph fits comfortably inside the configured size."
        document = _document_from_blocks([(1, paragraph, 11.0)])
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=400, min_chunk_chars=0)
        ).chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text == paragraph

    def test_oversized_paragraph_is_split_at_sentence_boundaries(self):
        sentences = [f"Sentence number {i} says something about retrieval." for i in range(30)]
        document = _document_from_blocks([(1, " ".join(sentences), 11.0)])
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=300, chunk_overlap=0)
        ).chunk(document)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.strip().endswith(".")

    def test_unbreakable_run_still_terminates(self):
        document = _document_from_blocks([(1, "x" * 5000, 11.0)])
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=200, chunk_overlap=0)
        ).chunk(document)
        assert len(chunks) == 25
        assert sum(len(c.text) for c in chunks) == 5000

    def test_overlap_repeats_text_only_on_size_triggered_flushes(self):
        sentences = [f"Alpha {i} beta gamma delta epsilon zeta." for i in range(20)]
        document = _document_from_blocks([(1, " ".join(sentences), 11.0)])
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=300, chunk_overlap=100)
        ).chunk(document)
        assert len(chunks) > 2
        tail = chunks[0].text[-40:]
        assert tail in chunks[1].text or chunks[1].text.startswith(tail.split()[-1])

    def test_no_overlap_across_a_heading_boundary(self):
        """Carrying a tail past a heading would mislabel its section."""
        document = _document_from_blocks(
            [
                (1, "1 First", 18.0),
                (1, "Content of the first section, which is reasonably long.", 11.0),
                (1, "2 Second", 18.0),
                (1, "Content of the second section, entirely unrelated.", 11.0),
            ]
        )
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=200, chunk_overlap=100, min_chunk_chars=0)
        ).chunk(document)
        second = next(c for c in chunks if c.section == "2 Second")
        assert "first section" not in second.text

    def test_short_chunk_is_folded_into_its_neighbour_not_dropped(self):
        document = _document_from_blocks(
            [
                (1, "A long opening paragraph that comfortably exceeds the minimum "
                    "chunk length configured for this test.", 11.0),
                (1, "Tiny.", 11.0),
            ]
        )
        chunks = StructureAwareChunker(
            ChunkingConfig(chunk_size=2000, min_chunk_chars=40)
        ).chunk(document)
        assert any("Tiny." in c.text for c in chunks)


# ---------------------------------------------------------------------------
# Fixed-size behaviour
# ---------------------------------------------------------------------------


class TestFixedSize:
    def test_chunks_respect_the_configured_size(self, structured_document):
        config = ChunkingConfig(strategy="fixed", chunk_size=300, chunk_overlap=50)
        for chunk in FixedSizeChunker(config).chunk(structured_document):
            assert len(chunk.text) <= config.chunk_size

    def test_section_is_deliberately_absent(self, structured_document):
        """The size baseline is structure-blind, so DD-009 stays measurable."""
        chunks = FixedSizeChunker(
            ChunkingConfig(strategy="fixed", chunk_size=300)
        ).chunk(structured_document)
        assert all(c.section is None for c in chunks)

    def test_windows_overlap_by_roughly_the_configured_amount(self, parser):
        document = parser.parse_bytes(pdfs.multipage_pdf(3), "m.pdf")
        config = ChunkingConfig(strategy="fixed", chunk_size=300, chunk_overlap=80)
        chunks = FixedSizeChunker(config).chunk(document)
        assert len(chunks) > 1
        first_end = chunks[0].metadata["char_span"][1]
        second_start = chunks[1].metadata["char_span"][0]
        assert 0 < first_end - second_start <= 80

    def test_words_are_not_cut_in_half(self, parser):
        document = parser.parse_bytes(pdfs.multipage_pdf(3), "m.pdf")
        chunks = FixedSizeChunker(
            ChunkingConfig(strategy="fixed", chunk_size=250, chunk_overlap=0)
        ).chunk(document)
        for chunk in chunks[:-1]:
            assert not chunk.text.endswith("-")
            assert chunk.text == chunk.text.strip()

    def test_zero_overlap_does_not_stall(self, parser):
        document = parser.parse_bytes(pdfs.multipage_pdf(2), "m.pdf")
        chunks = FixedSizeChunker(
            ChunkingConfig(strategy="fixed", chunk_size=200, chunk_overlap=0)
        ).chunk(document)
        spans = [c.metadata["char_span"] for c in chunks]
        assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))


# ---------------------------------------------------------------------------
# Segmentation and the chunk model
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_kinds_are_assigned(self, structured_document):
        segments = segment_document(structured_document, ChunkingConfig())
        kinds = {s.kind for s in segments}
        assert SegmentKind.HEADING in kinds
        assert SegmentKind.PARAGRAPH in kinds
        assert SegmentKind.TABLE in kinds

    def test_page_break_paragraph_is_rejoined_with_both_pages(
        self, structured_document
    ):
        segments = segment_document(structured_document, ChunkingConfig())
        merged = [s for s in segments if len(s.pages) > 1]
        assert len(merged) == 1
        assert merged[0].pages == (1, 2)
        assert "finished before the page break" in merged[0].text

    def test_joining_can_be_disabled(self, structured_document):
        segments = segment_document(
            structured_document, ChunkingConfig(join_paragraphs_across_pages=False)
        )
        assert all(len(s.pages) == 1 for s in segments)


class TestChunkModel:
    def test_chunk_requires_page_attribution(self):
        with pytest.raises(ChunkingError, match="page attribution"):
            Chunk(chunk_id="c", document_id="d", text="t", pages=())

    def test_chunk_rejects_zero_based_pages(self):
        with pytest.raises(ChunkingError, match="1-based"):
            Chunk(chunk_id="c", document_id="d", text="t", pages=(0,))

    def test_chunk_rejects_unsorted_pages(self):
        with pytest.raises(ChunkingError, match="out of order"):
            Chunk(chunk_id="c", document_id="d", text="t", pages=(3, 1))

    def test_citation_label_reads_naturally(self):
        single = Chunk(chunk_id="c", document_id="d", text="t", pages=(4,))
        spanning = Chunk(chunk_id="c", document_id="d", text="t", pages=(4, 5))
        assert single.citation_label() == "page 4"
        assert spanning.citation_label() == "pages 4-5"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _document_from_blocks(specs: list[tuple[int, str, float]]) -> Document:
    """Build a Document directly, bypassing the PDF layer.

    Lets a chunking test state exactly the layout it is about, instead of
    reverse-engineering one out of a generated PDF.
    """
    by_page: dict[int, list[Block]] = {}
    for page_number, text, font_size in specs:
        blocks = by_page.setdefault(page_number, [])
        blocks.append(
            Block(text=text, font_size=font_size, order=len(blocks))
        )
    pages = tuple(
        Page(
            page_number=number,
            text="\n\n".join(b.text for b in blocks),
            blocks=tuple(blocks),
        )
        for number, blocks in sorted(by_page.items())
    )
    return Document(document_id="doc_test", source_name="test.pdf", pages=pages)
