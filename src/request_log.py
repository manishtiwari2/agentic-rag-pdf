"""Request-scoped logging for the CLI and chat app."""

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

request_id = ContextVar("request_id", default="")
logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "time": record.created,
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "message": record.getMessage(),
        })


def configure_logging() -> None:
    app = logging.getLogger("src")
    if not any(isinstance(h.formatter, JsonFormatter) for h in app.handlers):
        handler = logging.StreamHandler(sys.__stderr__ or sys.stderr)
        handler.setFormatter(JsonFormatter())
        app.addHandler(handler)
    app.setLevel(logging.INFO)
    app.propagate = False


@contextmanager
def request_scope(operation: str):
    if request_id.get():
        yield
        return
    token = request_id.set(uuid4().hex)
    try:
        yield
    except Exception as exc:
        logger.warning("%s failed (%s)", operation, type(exc).__name__)
        raise
    else:
        logger.info("%s completed", operation)
    finally:
        request_id.reset(token)
