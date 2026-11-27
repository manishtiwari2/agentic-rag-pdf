"""Citation previews use matching page geometry and preserve unmatched pages."""

from tests.pdf_fixtures import PAGE_MARKER, multipage_pdf

from src.ingestion.highlight import render_highlight


def test_highlight_changes_the_actual_page_pixels():
    data = multipage_pdf(2)
    plain, unmatched = render_highlight(data, 2, "A passage that is not in the document.")
    highlighted, located = render_highlight(data, 2, PAGE_MARKER.format(n=2))
    assert plain.startswith(b"\x89PNG") and highlighted.startswith(b"\x89PNG")
    assert not unmatched and located
    assert highlighted != plain
