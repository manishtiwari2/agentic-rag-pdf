"""Which retriever wins on which questions (DD-072).

Two pieces, kept apart on purpose:

* :func:`question_buckets` sorts the questions into ``numeric``, ``lexical``,
  ``mixed`` and ``paraphrase`` from the questions and their gold pages alone.
  It reads no system output, so the buckets and their cut points can be written
  into DD-072 Part 1 before any run.
* :func:`retriever_wins` reads one stored hybrid run and, per question, finds
  the rank of the first gold page in the dense list, the BM25 list and the
  fused list, all *before* reranking. Reranking would reorder every list the
  same way and hide which retriever put the page there.

Overlap uses the harness's own tokenizer (``metrics.content_tokens``), not the
system's, for the reason ``metrics`` gives: a scorer sharing the system's
tokenizer flatters it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..benchmark.schema import Question
from ..chunking.base import build_chunker
from ..config import RAGConfig
from ..ingestion.parser import build_parser
from .metrics import content_tokens

#: Question types whose answer is a number or a table cell (DD-072).
NUMERIC_TYPES = frozenset({"numerical", "table"})
WINS_FILENAME = "WINS.md"
WINS_JSON_FILENAME = "wins.json"
RETRIEVERS = ("dense", "lexical", "fused")


# ---------------------------------------------------------------------------
# Buckets: from the questions and gold pages only
# ---------------------------------------------------------------------------


def _page_texts(documents_dir: Path, names: set[str], config: RAGConfig) -> dict[str, dict[int, str]]:
    parser = build_parser(config.ingestion)
    texts = {}
    for name in sorted(names):
        document = parser.parse(documents_dir / name)
        texts[name] = {page.page_number: page.text for page in document.pages}
    return texts


def overlap(question: str, evidence: str) -> float:
    """Share of the question's content terms that appear on its gold pages."""
    terms = set(content_tokens(question))
    if not terms:
        return 0.0
    return len(terms & set(content_tokens(evidence))) / len(terms)


def _tercile_cuts(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[(n - 1) // 3], ordered[(2 * (n - 1)) // 3]


def question_buckets(
    questions: Sequence[Question],
    documents_dir: Path | str,
    config: RAGConfig | None = None,
) -> dict[str, Any]:
    """Assign every answerable question one bucket; unanswerable ones none.

    ``numeric``: a numerical or table question. (A digit anywhere in the
    question was the first draft; it swept in definition questions that merely
    name a year or section, 18 of 30 on dev. Changed before any run, DD-072.)
    The rest are cut into terciles of :func:`overlap` with their gold pages:
    at or above the upper cut is ``lexical``, at or below the lower cut is
    ``paraphrase``, between is ``mixed``. The cuts are computed over the
    questions passed in, so passing the dev questions gives dev cut points.
    """
    config = config or RAGConfig.offline()
    answerable = [q for q in questions if not q.is_unanswerable]
    texts = _page_texts(Path(documents_dir), {q.document for q in answerable}, config)

    rows: dict[str, dict[str, Any]] = {}
    for q in answerable:
        evidence = " ".join(texts[q.document].get(p, "") for p in q.evidence_pages)
        numeric = q.question_type in NUMERIC_TYPES
        rows[q.id] = {
            "question_type": q.question_type,
            "overlap": round(overlap(q.question, evidence), 4),
            "bucket": "numeric" if numeric else None,
        }
    rest = [r for r in rows.values() if r["bucket"] is None]
    low, high = _tercile_cuts([r["overlap"] for r in rest]) if rest else (0.0, 0.0)
    for r in rest:
        r["bucket"] = (
            "lexical" if r["overlap"] >= high else "paraphrase" if r["overlap"] <= low else "mixed"
        )
    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    return {
        "cuts": {"paraphrase_at_or_below": low, "lexical_at_or_above": high},
        "counts": dict(sorted(counts.items())),
        "questions": dict(sorted(rows.items())),
    }


# ---------------------------------------------------------------------------
# Wins: from one stored hybrid run
# ---------------------------------------------------------------------------


def chunk_pages(
    documents: Sequence[str], documents_dir: Path | str, config: RAGConfig
) -> dict[str, tuple[int, ...]]:
    """Chunk id -> pages, rebuilt with the run's chunking (deterministic)."""
    parser = build_parser(config.ingestion)
    chunker = build_chunker(config.chunking)
    pages: dict[str, tuple[int, ...]] = {}
    for name in sorted(set(documents)):
        for chunk in chunker.chunk(parser.parse(Path(documents_dir) / name)):
            pages[chunk.chunk_id] = tuple(chunk.pages)
    return pages


def first_gold_rank(
    chunk_ids: Sequence[str], gold: set[int], pages: Mapping[str, tuple[int, ...]]
) -> int | None:
    for rank, chunk_id in enumerate(chunk_ids or (), start=1):
        if gold & set(pages.get(chunk_id, ())):
            return rank
    return None


def retriever_wins(
    records: Sequence[Mapping[str, Any]],
    buckets: Mapping[str, Any],
    pages: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Per question, each retriever's first gold rank; tallied per group.

    Dense "wins" a question when its first gold page ranks above BM25's (a
    page the other never found counts as ranked below everything). Groups are
    the buckets and the question types.
    """
    per_question = {}
    for record in records:
        qid = str(record["question_id"])
        info = (buckets.get("questions") or {}).get(qid)
        gold = set(((record.get("metrics") or {}).get("retrieval") or {}).get("gold_pages") or ())
        if info is None or not gold:
            continue
        ranks = {
            "dense": first_gold_rank(record.get("dense_chunk_ids"), gold, pages),
            "lexical": first_gold_rank(record.get("lexical_chunk_ids"), gold, pages),
            "fused": first_gold_rank(record.get("pre_rerank_chunk_ids"), gold, pages),
        }
        d, lx = (ranks[k] if ranks[k] is not None else float("inf") for k in ("dense", "lexical"))
        winner = "dense" if d < lx else "lexical" if lx < d else "tie"
        per_question[qid] = {
            "bucket": info["bucket"],
            "question_type": info["question_type"],
            "ranks": ranks,
            "winner": winner,
        }

    def tally(key: str) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for row in per_question.values():
            g = groups.setdefault(
                row[key],
                {"n": 0, "dense": 0, "tie": 0, "lexical": 0,
                 **{f"mrr_{r}": 0.0 for r in RETRIEVERS}},
            )
            g["n"] += 1
            g[row["winner"]] += 1
            for r in RETRIEVERS:
                rank = row["ranks"][r]
                g[f"mrr_{r}"] += (1.0 / rank) if rank else 0.0
        for g in groups.values():
            for r in RETRIEVERS:
                g[f"mrr_{r}"] = round(g[f"mrr_{r}"] / g["n"], 3)
        return dict(sorted(groups.items()))

    return {
        "by_bucket": tally("bucket"),
        "by_question_type": tally("question_type"),
        "questions": dict(sorted(per_question.items())),
    }


def render_wins(result: Mapping[str, Any], stack: str, arm: str) -> str:
    cuts = result["buckets"]["cuts"]
    lines = [
        f"# Which retriever wins (DD-072), dev split, arm `{arm}`",
        "",
        "Generated by `python -m src.cli fusion-wins`; do not edit by hand.",
        "",
        f"> {stack}",
        "",
        "Rank of the first gold page in each list **before reranking**. Dense wins a "
        "question when its gold page ranks above BM25's (a page never found ranks "
        "below everything). MRR here is over the pre-rerank lists, not the final "
        "ranking. Descriptive only: no test, and no part of DD-072's rule.",
        "",
        f"Buckets (DD-072 Part 1): `numeric` = a numerical or table question; "
        f"otherwise question-to-gold-page term overlap "
        f"≥ {cuts['lexical_at_or_above']:.3f} is `lexical`, ≤ "
        f"{cuts['paraphrase_at_or_below']:.3f} is `paraphrase`, between is `mixed`.",
    ]
    for title, key in (("By bucket", "by_bucket"), ("By question type", "by_question_type")):
        lines += [
            "",
            f"### {title}",
            "",
            "| group | n | dense wins | ties | BM25 wins | MRR dense | MRR BM25 | MRR fused |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, g in result[key].items():
            lines.append(
                f"| {name} | {g['n']} | {g['dense']} | {g['tie']} | {g['lexical']} | "
                f"{g['mrr_dense']:.3f} | {g['mrr_lexical']:.3f} | {g['mrr_fused']:.3f} |"
            )
    decided = [(q, r) for q, r in result["questions"].items() if r["winner"] != "tie"]
    if decided:
        lines += ["", "Questions one retriever wins (ranks as dense / BM25 / fused; – = not found):", ""]
        for qid, r in decided:
            ranks = " / ".join(str(r["ranks"][k] or "–") for k in RETRIEVERS)
            lines.append(f"* {qid} ({r['bucket']}, {r['question_type']}): {r['winner']}, {ranks}")
    return "\n".join(lines) + "\n"


def write_wins(result: Mapping[str, Any], markdown: str, out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"table": out / WINS_FILENAME, "json": out / WINS_JSON_FILENAME}
    paths["table"].write_text(markdown, encoding="utf-8", newline="\n")
    paths["json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return paths
