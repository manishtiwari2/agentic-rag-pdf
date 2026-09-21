"""Phase 0 benchmark tooling (BENCHMARK_SPEC.md, EXPERIMENT_PLAN.md section 2).

Choosing documents, writing questions and eyeballing evidence pages is manual
work that this package does not do. What it provides is the part that makes
that manual work checkable: typed access to ``benchmark/questions.json``
(:mod:`.schema`), a checksum-verified document manifest (:mod:`.manifest`), a
validator that enforces BENCHMARK_SPEC.md section 5.1 (:mod:`.validate`), and
a page-inspection tool so a human can verify an evidence page from the
terminal (:mod:`.inspect`).
"""
