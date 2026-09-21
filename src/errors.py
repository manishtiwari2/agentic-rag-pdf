"""Explicit failure taxonomy.

ARCHITECTURE.md section 24 requires every component to fail explicitly with an
understandable message rather than returning degraded output. Each error below
carries an actionable remedy in its message, not just a description of what
went wrong.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for every error raised by this project."""


# --- ingestion -------------------------------------------------------------


class IngestionError(RAGError):
    """PDF could not be turned into a usable Document."""


class PDFReadError(IngestionError):
    """The file is missing, unreadable, or not a PDF at all."""


class EncryptedPDFError(IngestionError):
    """The PDF is password protected and no usable password was supplied."""


class ScannedPDFError(IngestionError):
    """The PDF carries images but no extractable text layer.

    Raised instead of returning a near-empty Document, because an empty
    document silently produces confident nonsense several layers later.
    """


class EmptyDocumentError(IngestionError):
    """The PDF parsed successfully but contains no usable text anywhere."""


# --- chunking --------------------------------------------------------------


class ChunkingError(RAGError):
    """The document could not be turned into retrieval units."""


# --- retrieval -------------------------------------------------------------


class RetrievalError(RAGError):
    """Indexing or search failed."""


class IndexNotBuiltError(RetrievalError):
    """A search was attempted before any document was indexed."""


class EmbeddingError(RetrievalError):
    """The embedding backend could not produce vectors."""


# --- generation ------------------------------------------------------------


class GenerationError(RAGError):
    """The generation backend failed or produced unusable output."""


class ModelLoadError(RAGError):
    """A model could not be loaded, typically for memory or licensing reasons."""


class ConfigurationError(RAGError):
    """The configuration is internally inconsistent or violates a hard constraint."""


# --- benchmark ---------------------------------------------------------------


class BenchmarkValidationError(RAGError):
    """The benchmark dataset fails a BENCHMARK_SPEC.md section 5.1 rule.

    Raised by callers that must refuse to start a run on an invalid dataset
    rather than produce plausible-looking numbers from a malformed one.
    """
