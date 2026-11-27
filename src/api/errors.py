"""Stable error names and messages for HTTP and SSE."""

from ..errors import DeadlineExceeded, GenerationError, IngestionError, LibraryItemNotFound, StorageError


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def public_error(exc: Exception) -> ApiError:
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, LibraryItemNotFound):
        return ApiError(404, "not_found", "The requested item does not exist.")
    if isinstance(exc, StorageError):
        return ApiError(503, "storage_unavailable", "The document library is temporarily unavailable.")
    if isinstance(exc, DeadlineExceeded):
        return ApiError(504, "question_timeout", str(exc))
    if isinstance(exc, GenerationError):
        return ApiError(503, "model_unavailable", "The model service could not answer.")
    if isinstance(exc, IngestionError):
        return ApiError(422, "invalid_pdf", "The PDF could not be accepted.")
    return ApiError(500, "internal_error", "The request could not be completed.")


def error_body(error: ApiError, request_id: str) -> dict:
    return {"error": {"code": error.code, "message": str(error), "request_id": request_id}}
