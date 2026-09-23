"""The agentic system's decision components (ARCHITECTURE.md sections 11-17).

Planner, evidence controller, refinement and verifier, each behind a small
protocol with a model-backed implementation and a rule-based stand-in
(DD-055), plus the explicit per-query state they write into (DD-011). The loop
that drives them lives in ``src/pipeline.py`` (``AgenticRAGPipeline``), because
the pipeline owns the wiring; nothing here imports the pipeline, the parser or
the evaluation layer.

Neither baseline imports this package. DD-013 requires systems that lack every
part of it, and ``tests/test_architecture.py`` checks that they still do.
"""
