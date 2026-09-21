"""Unit tests for the text normalizers.

Ligatures are tested here on strings rather than through a PDF: the base-14
fonts the fixture builder can use have no ligature glyph, so a ligature written
into a fixture comes back as a replacement character and the test would be
measuring the fixture.
"""

from __future__ import annotations

import pytest

from src.config import IngestionConfig
from src.ingestion.normalization import (
    collapse_blank_lines,
    count_running_keys,
    find_running_lines,
    join_hyphenated_linebreaks,
    keep_mask,
    line_key,
    normalize_text,
    strip_running_lines,
)


@pytest.fixture
def config() -> IngestionConfig:
    return IngestionConfig()


class TestCharacterNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("efﬁcient", "efficient"),
            ("ﬂow", "flow"),
            ("diﬀerence", "difference"),
            ("ﬃliation", "ffiliation"),
        ],
    )
    def test_ligatures_are_expanded(self, raw, expected, config):
        assert normalize_text(raw, config) == expected

    def test_soft_hyphen_is_removed(self, config):
        assert normalize_text("re­trieval", config) == "retrieval"

    def test_zero_width_and_nbsp(self, config):
        assert normalize_text("a​b c", config) == "ab c"

    def test_curly_quotes_are_straightened(self, config):
        assert normalize_text("“the model’s”", config) == '"the model\'s"'

    def test_quote_normalization_can_be_disabled(self):
        config = IngestionConfig(normalize_quotes=False)
        assert "’" in normalize_text("the model’s", config)

    def test_ligature_expansion_can_be_disabled(self):
        config = IngestionConfig(fix_ligatures=False)
        assert "ﬁ" in normalize_text("efﬁcient", config)
        # Invisible characters are still removed: they are never meaningful.
        assert normalize_text("re­trieval", config) == "retrieval"

    def test_line_structure_is_preserved(self, config):
        assert normalize_text("a  b\nc   d", config) == "a b\nc d"


class TestHyphenJoining:
    def test_lowercase_continuation_joins_and_drops_the_hyphen(self):
        assert join_hyphenated_linebreaks("retrie-\nval") == "retrieval"

    def test_uppercase_continuation_keeps_the_hyphen(self):
        assert join_hyphenated_linebreaks("Mixed-\nCase") == "Mixed-Case"

    def test_digit_continuation_keeps_the_hyphen(self):
        assert join_hyphenated_linebreaks("UTF-\n8") == "UTF-8"

    def test_consecutive_breaks_are_all_joined(self):
        text = "seg-\nment and doc-\nument and re-\ntrieval"
        assert join_hyphenated_linebreaks(text) == "segment and document and retrieval"

    def test_hyphen_not_at_a_line_break_is_untouched(self):
        assert join_hyphenated_linebreaks("state-of-the-art") == "state-of-the-art"

    def test_blank_line_between_halves_still_joins(self):
        """Extractors often emit the two halves as separate blocks."""
        assert join_hyphenated_linebreaks("retrie-\n\nval") == "retrieval"


class TestRunningLines:
    def test_digits_are_masked_in_the_key(self):
        assert line_key("Page 3 of 10") == line_key("Page 7 of 10")
        assert line_key("  THE   Report ") == "the report"

    def test_repeated_edge_line_is_detected(self, config):
        pages = [["Annual Report", f"body {i}", f"{i}"] for i in range(1, 6)]
        running = find_running_lines(pages, config)
        assert "annual report" in running
        assert "#" in running  # the bare page number

    def test_mid_page_repetition_is_not_a_header(self, config):
        """A phrase repeated in the body is style, not page furniture."""
        pages = [
            ["title", "intro", f"body {i}", "the same line", "more", "tail"]
            for i in range(1, 6)
        ]
        running = find_running_lines(pages, config)
        assert "the same line" not in running
        assert "title" in running  # the genuine edge line still is

    def test_short_documents_are_left_alone(self, config):
        pages = [["Annual Report", "body"], ["Annual Report", "body"]]
        assert find_running_lines(pages, config) == set()

    def test_long_lines_are_never_furniture(self, config):
        """A 200-character line is a paragraph, however often it repeats."""
        long_line = "x" * 200
        pages = [[long_line, f"body {i}"] for i in range(1, 6)]
        assert line_key(long_line) not in count_running_keys(pages, 5, config)

    def test_stripping_removes_only_edge_occurrences(self, config):
        pages = [["Annual Report", "Annual Report", "body"] for _ in range(5)]
        # scan_lines defaults to 2, so both leading copies are at the edge.
        stripped = strip_running_lines(pages, {"annual report"}, config)
        assert stripped[0] == ["body"]

    def test_candidate_flags_override_position(self, config):
        lines = ["Annual Report", "body", "Annual Report"]
        # Only the last line is flagged as sitting in a margin.
        mask = keep_mask(lines, {"annual report"}, config, [False, False, True])
        assert mask == [True, True, False]


class TestBlankLines:
    def test_runs_of_blank_lines_collapse(self):
        assert collapse_blank_lines("a\n\n\n\n\nb") == "a\n\nb"
