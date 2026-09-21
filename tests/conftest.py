"""Shared fixtures.

The ``parser`` fixture is parametrized over every backend, so the ingestion
tests run twice and the two implementations cannot drift apart. DD-033 made the
parser genuinely swappable; this is what keeps it swappable.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.config import ChunkingConfig, IngestionConfig, RAGConfig
from src.ingestion.parser import PARSERS, build_parser

from . import pdf_fixtures


def _installed(name: str) -> bool:
    """Is this backend's library importable?

    PyMuPDF is an optional, AGPL-3.0, test-only dependency (DD-033), so its
    half of the parser-equivalence tests is skipped rather than failed when it
    is absent.
    """
    module = {"pdfplumber": "pdfplumber", "pymupdf": "pymupdf"}.get(name, name)
    return importlib.util.find_spec(module) is not None


PARSER_NAMES = [name for name in sorted(PARSERS) if _installed(name)]


@pytest.fixture(params=PARSER_NAMES)
def parser_name(request) -> str:
    return request.param


@pytest.fixture
def parser(parser_name: str):
    return build_parser(IngestionConfig(parser=parser_name))


@pytest.fixture
def make_parser(parser_name: str):
    """Build a parser of the same backend with a custom configuration."""

    def factory(**overrides):
        return build_parser(IngestionConfig(parser=parser_name, **overrides))

    return factory


@pytest.fixture
def offline_config() -> RAGConfig:
    """Small chunks so a short fixture still produces several of them."""
    config = RAGConfig.offline()
    return RAGConfig(
        ingestion=config.ingestion,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=80),
        embedding=config.embedding,
        retrieval=config.retrieval,
        generation=config.generation,
    )


@pytest.fixture
def structured_document(parser):
    return parser.parse_bytes(pdf_fixtures.structured_pdf(), "structured.pdf")


@pytest.fixture
def structured_pdf_path(tmp_path):
    path = tmp_path / "structured.pdf"
    path.write_bytes(pdf_fixtures.structured_pdf())
    return path
