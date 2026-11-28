"""Latency metrics for the question stream."""

from collections import deque
from math import ceil
from threading import Lock

from prometheus_client import Histogram
from prometheus_client.core import GaugeMetricFamily

BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 180)


class QuestionTimings:
    def __init__(self, registry) -> None:
        self.first_token = Histogram(
            "rag_ask_first_token_seconds", "Time from ask to the first answer text event",
            buckets=BUCKETS, registry=registry,
        )
        self._latency = Histogram(
            "rag_ask_latency_seconds", "Time from ask to the final answer event",
            buckets=BUCKETS, registry=registry,
        )
        self._recent = deque(maxlen=1024)
        self._lock = Lock()
        registry.register(self)

    def observe_answer(self, elapsed: float) -> None:
        self._latency.observe(elapsed)
        with self._lock:
            self._recent.append(elapsed)

    def collect(self):
        with self._lock:
            samples = sorted(self._recent)
        gauge = GaugeMetricFamily(
            "rag_ask_latency_quantile_seconds", "Latency over the latest 1024 completed asks in this process",
            labels=["quantile"],
        )
        if samples:
            for quantile in (0.5, 0.95):
                gauge.add_metric([str(quantile)], samples[ceil(quantile * len(samples)) - 1])
        yield gauge
