"""Phase 6 reports, generated from stored results only.

    python -m src.cli final-table            # writes results/final/

Nothing here runs a system. Every figure is read from the ``per_question.json``,
``results.json`` and ``comparison.json`` files that ``run-benchmark`` and
``compare-runs`` wrote, so the tables can be regenerated from the repository
and cannot drift from the runs they describe:

* :func:`final_table` -- EVALUATION_PROTOCOL.md section 28, each figure with a
  95% CI (DD-028), headline on the eval split (DD-058, section 5.2).
* :func:`ablation_report` -- DD-058's "earned its cost" rule, applied
  mechanically to each arm's comparison against its reference.
* :func:`error_analysis` -- sections 23 and 30: categories across every system
  and arm, accuracy by question type, the question types that stay hard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import statistics
from .benchmark import load_results, restrict_to_split
from .metrics import ERROR_CATEGORIES

#: Printed beside every table (the Phase 6 honesty rule).
OFFLINE_NOTE = (
    "Offline stand-in stack: hashing embedder, scripted extractive generator, "
    "term-overlap reranker, rule-based planner / evidence controller / verifier. "
    "No figure here is about Qwen3-4B, bge-m3 or bge-reranker-v2-m3."
)

#: The rows of section 28, in order. The tuned arm holds only the eval split.
FINAL_SYSTEMS: tuple[tuple[str, str], ...] = (
    ("Dense RAG (Baseline A)", "results/baseline"),
    ("Hybrid RAG (Baseline B)", "results/hybrid"),
    ("Agentic RAG (threshold 0.5)", "results/agentic"),
    ("Agentic RAG (threshold 0.3, tuned on dev)", "results/experiments/threshold/eval_0.3"),
)

#: Section 28's columns, as (metric, label).
TABLE_METRICS: tuple[tuple[str, str], ...] = (
    ("recall_at_5", "Recall@5"),
    ("mrr_at_10", "MRR@10"),
    ("accuracy_all", "Accuracy"),
    ("faithfulness", "Faithfulness"),
    ("citation_any_correct", "Citation any-correct"),
    ("abstention_accuracy", "Abstention accuracy"),
)

#: The four keep-or-drop metrics (EXPERIMENT_PLAN.md section 3, DD-056); a
#: component whose removal significantly improves one of them causes a harm.
KEEP_OR_DROP: tuple[str, ...] = tuple(statistics.PRIMARY_METRICS["keep_or_drop"]["primary"])

#: DD-057's seven arms. ``comparison`` is the stored compare-runs output;
#: ``sign`` is +1 when it holds arm - reference and -1 when it holds
#: reference - arm (Phase 4 compared reranker-on against reranker-off).
ABLATION_ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "planner_off", "component": "planner", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/planner_off",
     "comparison": "results/ablations/planner_off/comparison.json", "sign": 1,
     "declared": "ablation:planner_off"},
    {"arm": "hybrid_off", "component": "hybrid retrieval", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/hybrid_off",
     "comparison": "results/ablations/hybrid_off/comparison.json", "sign": 1,
     "declared": "ablation:hybrid_off"},
    {"arm": "reranker_off_agentic", "component": "reranker", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/reranker_off_agentic",
     "comparison": "results/ablations/reranker_off_agentic/comparison.json", "sign": 1,
     "declared": "ablation:reranker_off_agentic"},
    {"arm": "refinement_off", "component": "refinement", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/refinement_off",
     "comparison": "results/ablations/refinement_off/comparison.json", "sign": 1,
     "declared": "ablation:refinement_off"},
    {"arm": "evidence_controller_off", "component": "evidence controller", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/evidence_controller_off",
     "comparison": "results/ablations/evidence_controller_off/comparison.json", "sign": 1,
     "declared": "ablation:evidence_controller_off"},
    {"arm": "verification_off", "component": "verifier", "decides": True,
     "reference": "results/agentic", "directory": "results/ablations/verification_off",
     "comparison": "results/ablations/verification_off/comparison.json", "sign": 1,
     "declared": "ablation:verification_off"},
    # Arm 7 (Phase 4): the reranker removed from Baseline B, RQ2's metrics.
    {"arm": "reranker_off", "component": "reranker", "decides": False,
     "reference": "results/hybrid", "directory": "results/ablations/reranker_off",
     "comparison": "results/hybrid/comparison_vs_reranker_off.json", "sign": -1,
     "declared": "RQ2"},
    # Not an arm (DD-057): hybrid OFF at the baseline level is Baseline A
    # against reranker-off B, RQ1's comparison. Corroboration only.
    {"arm": "baseline_a_vs_reranker_off", "component": "hybrid retrieval", "decides": False,
     "reference": "results/ablations/reranker_off", "directory": "results/baseline",
     "comparison": "results/ablations/reranker_off/comparison.json", "sign": -1,
     "declared": "RQ1", "counted_as_arm": False},
)

#: Every stored run the error analysis reads, as (label, directory).
ERROR_ANALYSIS_RUNS: tuple[tuple[str, str], ...] = (
    ("dense", "results/baseline"),
    ("hybrid", "results/hybrid"),
    ("agentic", "results/agentic"),
    ("reranker_off (B)", "results/ablations/reranker_off"),
    ("planner_off", "results/ablations/planner_off"),
    ("hybrid_off", "results/ablations/hybrid_off"),
    ("reranker_off_agentic", "results/ablations/reranker_off_agentic"),
    ("refinement_off", "results/ablations/refinement_off"),
    ("evidence_controller_off", "results/ablations/evidence_controller_off"),
    ("verification_off", "results/ablations/verification_off"),
    ("tuned 0.3 (eval only)", "results/experiments/threshold/eval_0.3"),
)

#: The systems whose per-type accuracy decides which question types stay hard.
HEADLINE_LABELS: tuple[str, ...] = ("dense", "hybrid", "agentic")


def _resolve(root: Path, path: str) -> Path:
    return root / path


def _load(
    root: Path, path: str, split: str | None = None, all_questions_only: bool = False
) -> dict[str, Any] | None:
    directory = _resolve(root, path)
    if not (directory / "per_question.json").exists():
        return None
    loaded = load_results(directory)
    stored = (loaded.get("config") or {}).get("split")
    if split:
        if stored not in (None, split):
            return None
        loaded = restrict_to_split(loaded, split)
    elif stored is not None and all_questions_only:
        # An every-question table cannot hold a one-split run.
        return None
    return loaded


def _reported(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(r[key])
        for r in records
        if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
    ]


# ---------------------------------------------------------------------------
# Section 28: the final comparison table
# ---------------------------------------------------------------------------


def system_row(label: str, loaded: Mapping[str, Any]) -> dict[str, Any]:
    """One row: each section 28 figure with its 95% CI, plus the extras."""
    records = loaded["per_question"]
    vectors = statistics.per_question_vectors(records)
    summary = (loaded.get("results") or {}).get("summary") or {}
    config = loaded.get("config") or {}
    iterations = _reported(records, "retrieval_iterations")
    calls = _reported(records, "model_calls")
    return {
        "label": label,
        "system": config.get("system"),
        "split": config.get("split"),
        "restricted_from_split": config.get("restricted_from_split", config.get("split")),
        "n_questions": len(records),
        "metrics": {
            name: statistics.bootstrap_ci(list(vectors[name].values()))
            for name, _ in TABLE_METRICS
        },
        "latency_p95_s": statistics.bootstrap_quantile_ci(list(vectors["latency_s"].values())),
        # Absent for the baselines (DD-055): null, never an invented 1.
        "retrieval_iterations_mean": statistics.bootstrap_ci(iterations),
        "model_calls_mean": statistics.bootstrap_ci(calls),
        # Run-level, and null without a GPU: written as null, never as 0.
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "sufficiency_threshold": (config.get("config") or {}).get("sufficiency_threshold"),
    }


def final_table(
    root: Path | str = ".",
    split: str | None = "eval",
    systems: Sequence[tuple[str, str]] = FINAL_SYSTEMS,
) -> dict[str, Any]:
    """Section 28 over one split (or every question, with ``split=None``).

    A system whose stored run cannot supply that split -- the tuned arm holds
    only eval -- is left out rather than approximated.
    """
    root = Path(root)
    rows, omitted = [], []
    for label, path in systems:
        loaded = _load(root, path, split, all_questions_only=split is None)
        if loaded is None:
            omitted.append({"label": label, "directory": path})
            continue
        row = system_row(label, loaded)
        row["directory"] = path
        rows.append(row)
    return {
        "split": split or "all",
        "note": OFFLINE_NOTE,
        "method": statistics.method_description(),
        "rows": rows,
        "omitted": omitted,
    }


# ---------------------------------------------------------------------------
# DD-058: did each component earn its cost?
# ---------------------------------------------------------------------------


def _flip(value: float | None) -> float | None:
    # ``0.0 - x`` rather than ``-x``: a flipped zero stays 0.0, never -0.0.
    return None if value is None else 0.0 - value


def _oriented(contrast: Mapping[str, Any], sign: int) -> dict[str, Any]:
    """A stored contrast as arm - reference, whichever way it was written."""
    diff, low, high = contrast["mean_difference"], contrast["ci_low"], contrast["ci_high"]
    if sign < 0 and diff is not None:
        diff, low, high = _flip(diff), _flip(high), _flip(low)
    ref, arm = contrast["baseline_mean"], contrast["system_mean"]
    if sign < 0:
        ref, arm = arm, ref
    return {
        "reference": ref,
        "arm": arm,
        "difference": diff,
        "ci_low": low,
        "ci_high": high,
        "n": contrast["n"],
        "questions_changed": contrast.get("questions_changed"),
        "higher_is_better": contrast["higher_is_better"],
    }


def _worse_without(c: Mapping[str, Any]) -> bool:
    """Removing the component made this metric significantly worse."""
    if c["ci_low"] is None:
        return False
    return c["ci_high"] < 0 if c["higher_is_better"] else c["ci_low"] > 0


def _better_without(c: Mapping[str, Any]) -> bool:
    """Removing the component made this metric significantly better."""
    if c["ci_low"] is None:
        return False
    return c["ci_low"] > 0 if c["higher_is_better"] else c["ci_high"] < 0


def _answers(loaded: Mapping[str, Any]) -> dict[str, str]:
    return {str(r["question_id"]): str(r.get("answer")) for r in loaded["per_question"]}


def assess_arm(root: Path, spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """DD-058's rule for one arm, read from its stored comparison."""
    path = _resolve(root, spec["comparison"])
    if not path.exists():
        return None
    comparison = json.loads(path.read_text(encoding="utf-8"))
    contrasts = {
        name: _oriented(c, spec["sign"]) for name, c in comparison["contrasts"].items()
    }
    declared = statistics.PRIMARY_METRICS[spec["declared"]]["primary"]
    attribution = statistics.PRIMARY_METRICS["ablation:attribution"]["primary"]
    benefit = [m for m in declared if _worse_without(contrasts[m])]
    harm = [m for m in KEEP_OR_DROP if _better_without(contrasts[m])]
    other = [
        m for m, c in contrasts.items()
        if m not in declared and m not in KEEP_OR_DROP and m != "latency_s"
        and m != "model_calls_mean" and (_worse_without(c) or _better_without(c))
    ]

    reference, arm = _load(root, spec["reference"]), _load(root, spec["directory"])
    inert = None
    reranker_calls = None
    if reference is not None and arm is not None:
        ref_answers, arm_answers = _answers(reference), _answers(arm)
        inert = ref_answers == arm_answers
        ref_calls = _reported(reference["per_question"], "reranker_calls")
        arm_calls = _reported(arm["per_question"], "reranker_calls")
        reranker_calls = {
            "reference_mean": round(sum(ref_calls) / len(ref_calls), 4) if ref_calls else None,
            "arm_mean": round(sum(arm_calls) / len(arm_calls), 4) if arm_calls else None,
        }
        answers_changed = sum(1 for q in ref_answers if ref_answers[q] != arm_answers.get(q))
    else:
        answers_changed = None

    if benefit and harm:
        verdict = "not shown (trade-off)"
    elif benefit:
        verdict = "earned its cost"
    elif harm:
        verdict = "did not earn its cost (removing it improves " + ", ".join(harm) + ")"
    elif inert:
        verdict = "did not earn its cost (inert on this stack)"
    else:
        verdict = "did not earn its cost"

    return {
        **{k: spec[k] for k in ("arm", "component", "reference", "directory", "decides")},
        "counted_as_arm": spec.get("counted_as_arm", True),
        "comparison": spec["comparison"],
        "declared_metrics": {m: contrasts[m] for m in declared},
        "attribution_metrics": {m: contrasts[m] for m in attribution},
        "keep_or_drop_metrics": {m: contrasts[m] for m in KEEP_OR_DROP},
        # What the component adds: reference - arm, i.e. minus arm - reference.
        "cost_added": {
            m: {
                "difference": _flip(contrasts[m]["difference"]),
                "ci_low": _flip(contrasts[m]["ci_high"]),
                "ci_high": _flip(contrasts[m]["ci_low"]),
                "n": contrasts[m]["n"],
            }
            for m in ("latency_s", "model_calls_mean")
            if m in contrasts
        },
        "reranker_calls": reranker_calls,
        "benefit": benefit,
        "harm": harm,
        "undeclared_significant": other,
        "answers_changed": answers_changed,
        "inert": inert,
        "n_comparisons": comparison["multiple_comparisons"]["n_comparisons"],
        "n_significant": comparison["multiple_comparisons"]["n_significant"],
        "verdict": verdict,
    }


def ablation_report(
    root: Path | str = ".", arms: Sequence[Mapping[str, Any]] = ABLATION_ARMS
) -> dict[str, Any]:
    root = Path(root)
    assessed = [a for a in (assess_arm(root, spec) for spec in arms) if a is not None]
    components: dict[str, dict[str, Any]] = {}
    for item in assessed:
        entry = components.setdefault(item["component"], {"decided_by": None, "corroboration": []})
        if item["decides"]:
            entry["decided_by"] = item["arm"]
            entry["verdict"] = item["verdict"]
        else:
            entry["corroboration"].append({"arm": item["arm"], "verdict_if_deciding": item["verdict"]})
    return {
        "note": OFFLINE_NOTE,
        "rule": "DD-058",
        "arms": assessed,
        "components": components,
        "n_arms_counted": sum(1 for a in assessed if a["counted_as_arm"]),
    }


# ---------------------------------------------------------------------------
# Sections 23 and 30: error analysis
# ---------------------------------------------------------------------------


def _run_errors(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = {name: 0 for name in ERROR_CATEGORIES}
    by_category: dict[str, list[str]] = {name: [] for name in ERROR_CATEGORIES}
    by_type: dict[str, dict[str, Any]] = {}
    for record in records:
        m = record["metrics"]
        qtype = str(m["question_type"])
        group = by_type.setdefault(qtype, {"n": 0, "correct": 0, "failed": 0, "categories": {}})
        group["n"] += 1
        group["correct"] += int(bool(m["correct"]))
        if m.get("failed"):
            category = m.get("error_category") or "uncategorised"
            categories[category] = categories.get(category, 0) + 1
            by_category.setdefault(category, []).append(str(record["question_id"]))
            group["failed"] += 1
            group["categories"][category] = group["categories"].get(category, 0) + 1
    for group in by_type.values():
        group["accuracy"] = round(group["correct"] / group["n"], 4)
    return {
        "n_questions": len(records),
        "n_failures": sum(categories.values()),
        "error_categories": categories,
        "failing_questions": {k: sorted(v) for k, v in by_category.items() if v},
        "by_question_type": dict(sorted(by_type.items())),
    }


def error_analysis(
    root: Path | str = ".",
    runs: Sequence[tuple[str, str]] = ERROR_ANALYSIS_RUNS,
    headline: Sequence[str] = HEADLINE_LABELS,
    sample_per_category: int = 3,
) -> dict[str, Any]:
    """Categories for every run; which question types stay hard; a sample.

    The sample is deterministic -- the first ``sample_per_category`` failing
    question ids of the agentic reference in each category, sorted -- so the
    manual inspection it drives can be repeated on the same questions.
    """
    root = Path(root)
    per_run: dict[str, Any] = {}
    loaded_runs: dict[str, dict[str, Any]] = {}
    for label, path in runs:
        loaded = _load(root, path)
        if loaded is None:
            continue
        loaded_runs[label] = loaded
        per_run[label] = {"directory": path, "split": loaded["config"].get("split"),
                          **_run_errors(loaded["per_question"])}

    types = sorted({t for label in headline if label in per_run
                    for t in per_run[label]["by_question_type"]})
    hardness = []
    for qtype in types:
        accuracies = {
            label: per_run[label]["by_question_type"].get(qtype, {}).get("accuracy")
            for label in headline if label in per_run
        }
        n = next(
            per_run[label]["by_question_type"][qtype]["n"]
            for label in headline if label in per_run and qtype in per_run[label]["by_question_type"]
        )
        best = max(v for v in accuracies.values() if v is not None)
        hardness.append({"question_type": qtype, "n": n, "accuracy": accuracies, "best": best})
    hardness.sort(key=lambda row: (row["best"], row["question_type"]))

    correct_anywhere: dict[str, bool] = {}
    for label in headline:
        for record in loaded_runs.get(label, {}).get("per_question", []):
            qid = str(record["question_id"])
            correct_anywhere[qid] = correct_anywhere.get(qid, False) or bool(record["metrics"]["correct"])
    never = sorted(q for q, ok in correct_anywhere.items() if not ok)

    sample: dict[str, list[str]] = {}
    reference = per_run.get("agentic")
    if reference:
        for category, ids in reference["failing_questions"].items():
            sample[category] = ids[:sample_per_category]

    return {
        "note": OFFLINE_NOTE,
        "runs": per_run,
        "question_type_difficulty": hardness,
        "never_correct_in_headline_systems": never,
        "n_never_correct": len(never),
        "n_headline_questions": len(correct_anywhere),
        "inspection_sample": sample,
    }


# ---------------------------------------------------------------------------
# Writing results/final/
# ---------------------------------------------------------------------------


def _ci(entry: Mapping[str, Any] | None, key: str = "mean") -> str:
    if not entry or entry.get(key) is None:
        return "null"
    return f"{entry[key]:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def _signed(c: Mapping[str, Any]) -> str:
    if c.get("difference") is None:
        return "n/a"
    return f"{c['difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def render_final_table(table: Mapping[str, Any]) -> str:
    split = table["split"]
    heading = {
        "eval": "Eval split — headline (EVALUATION_PROTOCOL.md 5.2, 28)",
        "all": "All 76 questions, both splits — continuity with STATUS.md 9.2-9.3, not a headline",
        "dev": "Dev split — tuning numbers, never a headline",
    }.get(split, split)
    lines = [f"### {heading}", "", f"> {table['note']}", ""]
    header = ["System", "n"] + [label for _, label in TABLE_METRICS] + ["P95 latency (s)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")
    for row in table["rows"]:
        cells = [row["label"], str(row["n_questions"])]
        cells += [_ci(row["metrics"][name]) for name, _ in TABLE_METRICS]
        cells.append(_ci(row["latency_p95_s"], "value"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| System | Avg retrieval iterations | Avg model calls | Peak VRAM (GB) |",
              "| --- | ---: | ---: | ---: |"]
    for row in table["rows"]:
        vram = row["peak_vram_gb"]
        lines.append(
            f"| {row['label']} | {_ci(row['retrieval_iterations_mean'])} | "
            f"{_ci(row['model_calls_mean'])} | "
            f"{'null (no GPU; not measured)' if vram is None else vram} |"
        )
    lines += [
        "",
        "Each cell is the value and its 95% percentile-bootstrap CI over questions "
        "(2,000 resamples, seed 42). P95 latency is nearest-rank, with a bootstrap "
        "CI of the same statistic. Iterations and model calls are null for the "
        "baselines, which report neither (DD-055). Latencies are milliseconds on a "
        "stack that loads no weights, and were measured in different sessions.",
    ]
    if table["omitted"]:
        lines.append("")
        lines.append("Omitted (no stored run for this split): " + ", ".join(
            o["label"] for o in table["omitted"]))
    return "\n".join(lines)


def render_ablations(report: Mapping[str, Any]) -> str:
    lines = ["### Ablations (DD-057 arms, DD-058 rule)", "", f"> {report['note']}", "",
             "Differences are arm − reference with 95% CIs; a component *helps* when "
             "removing it makes one of its declared metrics significantly worse.", "",
             "| Arm | Reference | Declared metrics (arm − ref) | Faithfulness / citation (attribution) "
             "| Cost added (latency s; model calls) | Answers changed | Verdict |",
             "| --- | --- | --- | --- | --- | ---: | --- |"]
    for a in report["arms"]:
        declared = "; ".join(f"{m} {_signed(c)}" for m, c in a["declared_metrics"].items())
        attribution = "; ".join(f"{m} {_signed(c)}" for m, c in a["attribution_metrics"].items())
        cost = "; ".join(
            f"{_signed(c) if c['difference'] is not None else 'n/a'}"
            for c in a["cost_added"].values()
        )
        if not a["counted_as_arm"]:
            tag = " (RQ1 corroboration, not an arm)"
        elif not a["decides"]:
            tag = " (arm 7, Baseline B; corroboration)"
        else:
            tag = ""
        verdict = a["verdict"] if a["decides"] else f"({a['verdict']}, if it decided)"
        lines.append(
            f"| {a['arm']}{tag} | {a['reference']} | {declared} | {attribution} | {cost} | "
            f"{a['answers_changed']} | {verdict} |"
        )
    lines += ["", "Verdict per component (decided by its agentic arm, DD-058):", ""]
    for component, entry in report["components"].items():
        lines.append(f"* **{component}** — {entry.get('verdict', 'no deciding arm')}"
                     f" (arm: {entry['decided_by']})")
    return "\n".join(lines)


def render_errors(analysis: Mapping[str, Any]) -> str:
    lines = ["### Error categories (section 23)", "", f"> {analysis['note']}", ""]
    labels = list(analysis["runs"])
    lines.append("| Category | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(labels)) + " |")
    for category in ERROR_CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(
            str(analysis["runs"][label]["error_categories"].get(category, 0)) for label in labels) + " |")
    lines.append("| **failures / n** | " + " | ".join(
        f"{analysis['runs'][l]['n_failures']} / {analysis['runs'][l]['n_questions']}" for l in labels) + " |")
    lines += ["", "### Accuracy by question type, hardest first (all 76 questions)", "",
              "| Question type | n | " + " | ".join(HEADLINE_LABELS) + " | best |",
              "| --- | ---: | " + " | ".join(["---:"] * len(HEADLINE_LABELS)) + " | ---: |"]
    for row in analysis["question_type_difficulty"]:
        lines.append(f"| {row['question_type']} | {row['n']} | " + " | ".join(
            "n/a" if row["accuracy"].get(l) is None else f"{row['accuracy'][l]:.3f}"
            for l in HEADLINE_LABELS) + f" | {row['best']:.3f} |")
    lines += ["", f"Never answered correctly by dense, hybrid or agentic: "
                  f"{analysis['n_never_correct']} of {analysis['n_headline_questions']} questions.",
              "", "Inspection sample (agentic reference, first 3 per category): " + "; ".join(
                  f"{k}: {', '.join(v)}" for k, v in analysis["inspection_sample"].items())]
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


def write_report(root: Path | str = ".", out_dir: Path | str = "results/final") -> dict[str, Path]:
    """Regenerate every Phase 6 table under ``out_dir`` from stored results."""
    root = Path(root)
    out = root / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_table = final_table(root, split="eval")
    all_table = final_table(root, split=None)
    ablations = ablation_report(root)
    errors = error_analysis(root)
    paths = {
        "final_table": out / "final_table.json",
        "ablations": out / "ablations.json",
        "error_analysis": out / "error_analysis.json",
        "markdown": out / "REPORT.md",
    }
    _write_json(paths["final_table"], {"eval": eval_table, "all": all_table})
    _write_json(paths["ablations"], ablations)
    _write_json(paths["error_analysis"], errors)
    markdown = "\n\n".join([
        "# Phase 6 report (generated by `python -m src.cli final-table`; do not edit by hand)",
        f"> {OFFLINE_NOTE}",
        render_final_table(eval_table),
        render_final_table(all_table),
        render_ablations(ablations),
        render_errors(errors),
    ]) + "\n"
    paths["markdown"].write_text(markdown, encoding="utf-8", newline="\n")
    return paths
