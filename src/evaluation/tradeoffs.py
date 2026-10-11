"""The reranker's cost/benefit, from the stored ablations (A3, docs/TRADEOFFS.md).

No new run and no tuning. Two existing pairs differ in ``reranking.enabled``
alone:

* Baseline B ``results/hybrid`` vs ``results/ablations/reranker_off``;
* the agentic system ``results/agentic`` vs
  ``results/ablations/reranker_off_agentic``.

For each pair this reports reranker ON − OFF: the overall paired bootstrap
(DD-028) and a per-question-type breakdown, beside what the reranker costs.
Costs are read from the records, not assumed:

* ``rerank_s`` — seconds spent reranking per question (measured, this stack);
* reranker invocations per question (``reranker_calls`` where the system
  reports it, else 1 for a system that reranks once);
* candidate pairs scored per question: the candidate pool times the
  invocations. On the model stack each pair is one cross-encoder forward pass
  (batched), so this count carries over to bge-reranker-v2-m3 even though the
  offline seconds do not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import statistics
from .benchmark import load_results
from .report import note_for_config

PAIRS = (
    ("Baseline B (hybrid)", "results/ablations/reranker_off", "results/hybrid"),
    ("Agentic", "results/ablations/reranker_off_agentic", "results/agentic"),
)
QUALITY = ("recall_at_5", "mrr_at_10", "accuracy_all", "faithfulness")
TABLE_FILENAME = "TABLE.md"
JSON_FILENAME = "table.json"


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))], 4)


def costs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What reranking cost per question in one stored run."""
    seconds, calls, pairs = [], [], []
    for record in records:
        if record.get("rerank_s") is not None:
            seconds.append(float(record["rerank_s"]))
        invocations = record.get("reranker_calls")
        if invocations is None:
            invocations = 1 if record.get("reranker") else 0
        pool = record.get("candidate_pool")
        calls.append(float(invocations))
        if isinstance(pool, (int, float)) and not isinstance(pool, bool):
            pairs.append(float(pool) * float(invocations))
    return {
        "rerank_s_mean": _mean(seconds),
        "rerank_s_p95": _p95(seconds),
        "reranker_invocations_mean": _mean(calls),
        "pairs_scored_mean": _mean(pairs),
    }


def by_type(
    off: Sequence[Mapping[str, Any]], on: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per question type, mean ON − OFF and how many questions changed."""
    types = {str(r["question_id"]): (r.get("metrics") or {}).get("question_type") for r in on}
    a, b = statistics.per_question_vectors(off), statistics.per_question_vectors(on)
    out: dict[str, dict[str, Any]] = {}
    for metric in QUALITY:
        for qid in sorted(set(a[metric]) & set(b[metric])):
            cell = out.setdefault(str(types.get(qid)), {}).setdefault(
                metric, {"n": 0, "sum": 0.0, "up": 0, "down": 0}
            )
            delta = b[metric][qid] - a[metric][qid]
            cell["n"] += 1
            cell["sum"] += delta
            cell["up"] += delta > 0
            cell["down"] += delta < 0
    for cells in out.values():
        for cell in cells.values():
            cell["mean_difference"] = round(cell.pop("sum") / cell["n"], 4)
    return dict(sorted(out.items()))


def reranker_tradeoff(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    pairs = []
    stack = None
    for label, off_dir, on_dir in PAIRS:
        off, on = load_results(root / off_dir), load_results(root / on_dir)
        stack = stack or note_for_config(on.get("config") or {})
        comparison = statistics.compare(off, on, off_dir, on_dir, allow_confounded=True)
        pairs.append(
            {
                "system": label,
                "reranker_off": off_dir,
                "reranker_on": on_dir,
                "n_questions": comparison["n_questions"],
                "varied": comparison["fair_comparison"]["varied"],
                "overall": {
                    m: {k: comparison["contrasts"][m].get(k)
                        for k in ("mean_difference", "ci_low", "ci_high", "questions_changed", "verdict")}
                    for m in (*QUALITY, "latency_s")
                },
                "by_question_type": by_type(off["per_question"], on["per_question"]),
                "costs_on": costs(on["per_question"]),
                "costs_off": costs(off["per_question"]),
            }
        )
    return {"stack": stack, "pairs": pairs}


def _ci(c: Mapping[str, Any]) -> str:
    if c.get("mean_difference") is None:
        return "n/a"
    mark = {statistics.SIGNIFICANT_IMPROVEMENT: " ▲", statistics.SIGNIFICANT_REGRESSION: " ▼"}.get(
        c.get("verdict"), ""
    )
    return f"{c['mean_difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]{mark}"


def _num(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def render(result: Mapping[str, Any]) -> str:
    lines = [
        "# Reranker cost/benefit (A3), all 76 questions",
        "",
        "Generated by `python -m src.cli reranker-table`; do not edit by hand.",
        "",
        f"> {result['stack']}",
        "",
        "Every figure is reranker ON − OFF on the same questions, from stored "
        "ablations; no new run. ▲/▼ mark a 95% paired-bootstrap CI excluding 0 "
        "(for latency, ▼ means slower).",
    ]
    for pair in result["pairs"]:
        on, off = pair["costs_on"], pair["costs_off"]
        lines += [
            "",
            f"## {pair['system']}: `{pair['reranker_on']}` − `{pair['reranker_off']}`",
            "",
            "| metric | ON − OFF, 95% CI | questions changed |",
            "| --- | --- | ---: |",
        ]
        for metric, c in pair["overall"].items():
            lines.append(f"| {metric} | {_ci(c)} | {_num(c.get('questions_changed'))} |")
        lines += [
            "",
            "| cost per question | reranker ON | reranker OFF |",
            "| --- | ---: | ---: |",
            f"| rerank seconds, mean (this stack) | {_num(on['rerank_s_mean'])} | {_num(off['rerank_s_mean'])} |",
            f"| rerank seconds, p95 (this stack) | {_num(on['rerank_s_p95'])} | {_num(off['rerank_s_p95'])} |",
            f"| reranker invocations | {_num(on['reranker_invocations_mean'])} | {_num(off['reranker_invocations_mean'])} |",
            f"| candidate pairs scored (= cross-encoder passes on the model stack) | {_num(on['pairs_scored_mean'])} | {_num(off['pairs_scored_mean'])} |",
            "",
            "Per question type, mean ON − OFF (questions up / down):",
            "",
            "| question type | n | " + " | ".join(QUALITY) + " |",
            "| --- | ---: |" + " ---: |" * len(QUALITY),
        ]
        for qtype, cells in pair["by_question_type"].items():
            n = max((c["n"] for c in cells.values()), default=0)
            row = [
                f"{cells[m]['mean_difference']:+.3f} ({cells[m]['up']}/{cells[m]['down']})"
                if m in cells else "n/a"
                for m in QUALITY
            ]
            lines.append(f"| {qtype} | {n} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write(result: Mapping[str, Any], out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"table": out / TABLE_FILENAME, "json": out / JSON_FILENAME}
    paths["table"].write_text(render(result), encoding="utf-8", newline="\n")
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return paths
