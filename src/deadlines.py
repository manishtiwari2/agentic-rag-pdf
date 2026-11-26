"""Bound caller latency even when a native model or retrieval call is blocked."""

from concurrent.futures import Future
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import Callable, TypeVar

from .errors import DeadlineExceeded
from .request_log import request_scope

T = TypeVar("T")
_deadline: ContextVar[tuple[float, str] | None] = ContextVar("deadline", default=None)
# A stuck native call keeps its slot until it exits, including after caller timeout.
_questions = BoundedSemaphore(4)


def remaining_seconds() -> float:
    deadline = _deadline.get()
    if deadline is None:
        return float("inf")
    end, operation = deadline
    remaining = end - monotonic()
    if remaining <= 0:
        raise DeadlineExceeded(f"{operation} timed out. Please try again.")
    return remaining


def run_bounded(operation: str, seconds: float, slots: BoundedSemaphore, work: Callable[[], T]) -> T:
    parent = _deadline.get()
    end = monotonic() + seconds
    deadline = parent if parent and parent[0] <= end else (end, operation)
    token = _deadline.set(deadline)
    try:
        future: Future[T] = Future()

        def run() -> None:
            try:
                remaining_seconds()
                future.set_result(work())
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                slots.release()

        thread = Thread(target=copy_context().run, args=(run,), daemon=True)
        if not slots.acquire(timeout=remaining_seconds()):
            raise DeadlineExceeded(f"{deadline[1]} timed out. Please try again.")
        try:
            thread.start()
        except BaseException:
            slots.release()
            raise
        thread.join(remaining_seconds())
        if thread.is_alive():
            raise DeadlineExceeded(f"{deadline[1]} timed out. Please try again.")
        remaining_seconds()
        return future.result()
    finally:
        _deadline.reset(token)


def run_question(seconds: float, work: Callable[[], T]) -> T:
    with request_scope("question"):
        if _deadline.get() is not None:
            remaining_seconds()
            return work()
        return run_bounded("Question", seconds, _questions, work)
