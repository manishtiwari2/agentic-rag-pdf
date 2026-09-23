"""Reciprocal Rank Fusion (ARCHITECTURE.md section 9, DD-008, DD-045).

    rrf(d) = sum over rankings r containing d of  1 / (k + rank_r(d))

RRF reads **ranks only**. Dense cosine similarities and BM25 scores live on
unrelated scales -- one is bounded by 1, the other grows with query length --
and any method that adds them has to calibrate one against the other first.
RRF needs no calibration, which is why DD-008 makes it the default: not because
it is known to be better here, but because it is deterministic, model-free and
has nothing to tune but ``k``.

Two implementation details make it deterministic rather than merely
usually-deterministic (DD-045):

* Scores are summed as exact ``Fraction`` values. Two chunks at ranks (1, 3)
  and (3, 1) tie exactly, and with three or more rankings float addition is not
  associative, so a float sum could order a mathematical tie by summation
  order.
* Ties are broken by a fixed key, in this order: the better single rank; the
  earlier ranking *among those that gave it that rank* (retriever priority, as
  configured -- dense first by default); the chunk's position in the document;
  its id. Every key is a function of the input rankings, so the output is too.
  A chunk ranked (1, 3) and one ranked (3, 1) tie exactly on score and on best
  rank; the first wins because its rank 1 came from the higher-priority list.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .retriever import RetrievedChunk

#: Cormack, Clarke and Buettcher (2009), the value RRF is usually run at.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse several rankings into one.

    Each ranking's order is taken as given; ``RetrievedChunk.rank`` and
    ``score`` are ignored, so a ranking whose scores were rescaled, or replaced
    outright, fuses identically. The fused chunk's ``source`` names every
    retriever that ranked it, joined with ``+`` in ranking order.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive.")

    totals: dict[str, Fraction] = {}
    #: (best rank, index of the highest-priority list giving that rank).
    best: dict[str, tuple[int, int]] = {}
    sources: dict[str, list[str]] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for list_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        for position, item in enumerate(ranking, start=1):
            chunk_id = item.chunk_id
            if chunk_id in seen:
                # A retriever listing a chunk twice is a bug upstream; counting
                # it twice would let that bug decide the fused order.
                continue
            seen.add(chunk_id)
            totals[chunk_id] = totals.get(chunk_id, Fraction(0)) + Fraction(1, k + position)
            best[chunk_id] = min(best.get(chunk_id, (position, list_index)), (position, list_index))
            chunks.setdefault(chunk_id, item)
            names = sources.setdefault(chunk_id, [])
            if item.source not in names:
                names.append(item.source)

    ordered = sorted(
        totals,
        key=lambda cid: (
            -totals[cid],
            best[cid],
            chunks[cid].chunk.index,
            cid,
        ),
    )
    return [
        RetrievedChunk(
            chunk=chunks[cid].chunk,
            score=float(totals[cid]),
            rank=rank,
            source="+".join(sources[cid]),
        )
        for rank, cid in enumerate(ordered, start=1)
    ]
