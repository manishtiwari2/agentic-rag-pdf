"""Dev-split parameter studies: a generated table and a rule applied in code.

DD-071 (chunking) and DD-072 (fusion) each sweep one family of settings on
``--split dev`` and choose by a rule written down before the runs. This module
turns the stored arms into the study's table and applies that rule, so no
number in a study is typed by hand and the choice cannot drift from the rule.

Every arm lives in ``<root>/<arm name>/`` as an ordinary ``run-benchmark``
output. Each challenger is compared with the incumbent arm by the same paired
bootstrap every other comparison uses (DD-028), with the study's own setting
declared as the varied factor: a chunking arm differs from the incumbent in
exactly the chunking fields EVALUATION_PROTOCOL.md section 10 otherwise holds
constant, and nothing else.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import statistics
from .benchmark import load_results
from .report import note_for_config

TABLE_FILENAME = "TABLE.md"
TABLE_JSON_FILENAME = "table.json"


@dataclass(frozen=True)
class StudySpec:
    """A study's arms and its decision rule, as declared before the runs."""

    name: str
    dd: str
    incumbent: str
    #: A challenger qualifies when one of these improves with a CI excluding 0 ...
    primary: tuple[str, ...]
    #: ... and none of these, nor the other primaries, significantly regresses.
    guard: tuple[str, ...]
    #: Ranking among qualifiers: each (metric, higher_is_better), in order.
    tiebreak: tuple[tuple[str, bool], ...]
    #: Columns of the table, in order (summary keys or metric names).
    columns: tuple[str, ...]
    #: Arms reported beside the grid but never eligible to be chosen.
    references: tuple[str, ...] = ()
    #: Config fields the study varies, shown per arm.
    factors: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


STUDIES: dict[str, StudySpec] = {
    # DD-071 Part 1, committed before any chunking arm was run.
    "chunking": StudySpec(
        name="chunking",
        dd="DD-071",
        incumbent="structure_1200_150",
        primary=("recall_at_5", "accuracy_all"),
        guard=("mrr_at_10",),
        tiebreak=(("recall_at_5", True), ("mrr_at_10", True), ("chunks_total", False)),
        columns=(
            "chunks_total",
            "indexed_chars_total",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "full_recall_at_5_multi_hop",
            "full_recall_at_5_multi_hop_comparison",
            "accuracy_all",
            "latency_p95_s",
        ),
        factors=("chunking_strategy", "chunk_size", "chunk_overlap"),
        notes=(
            "Full-Recall@5 on multi-hop has n = 3 on dev (n = 6 with comparison "
            "questions): descriptive only, never part of the rule.",
            "p95 latency is milliseconds on the offline stack and moves between "
            "sessions (DD-069): descriptive only.",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_arms(root: Path | str) -> dict[str, dict[str, Any]]:
    """Every stored run directly under ``root``, by directory name."""
    root = Path(root)
    arms = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if (directory / "results.json").exists():
            arms[directory.name] = load_results(directory)
    return arms


def _multi_hop_full_recall(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Full-Recall@5 over ``multi_hop`` questions only (no comparison)."""
    values = []
    for record in records:
        metrics = record.get("metrics") or {}
        if metrics.get("question_type") != "multi_hop":
            continue
        value = (metrics.get("retrieval") or {}).get("full_recall_at_5")
        if value is not None:
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else None


def arm_values(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """The table's figures for one arm, read from what the run stored."""
    summary = (loaded.get("results") or {}).get("summary") or {}
    vectors = statistics.per_question_vectors(loaded.get("per_question") or [])
    values: dict[str, Any] = dict(summary)
    for spec in statistics.METRICS:
        vector = vectors[spec.name]
        values[spec.name] = (
            round(sum(vector.values()) / len(vector), 4) if vector else None
        )
    values["full_recall_at_5_multi_hop"] = _multi_hop_full_recall(
        loaded.get("per_question") or []
    )
    return values


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _significant(contrast: Mapping[str, Any] | None, verdict: str) -> bool:
    return bool(contrast) and contrast.get("verdict") == verdict


def qualifies(spec: StudySpec, contrasts: Mapping[str, Mapping[str, Any]]) -> bool:
    """Spec's rule for one challenger, from its contrasts against the incumbent."""
    improves = any(
        _significant(contrasts.get(m), statistics.SIGNIFICANT_IMPROVEMENT)
        for m in spec.primary
    )
    regresses = any(
        _significant(contrasts.get(m), statistics.SIGNIFICANT_REGRESSION)
        for m in spec.primary + spec.guard
    )
    return improves and not regresses


def choose(
    spec: StudySpec,
    values: Mapping[str, Mapping[str, Any]],
    qualified: Sequence[str],
) -> str:
    """The incumbent, unless challengers qualify; then the best by tiebreak."""
    if not qualified:
        return spec.incumbent

    def key(name: str) -> tuple:
        parts = []
        for metric, higher in spec.tiebreak:
            value = values[name].get(metric)
            if value is None:
                value = -math.inf if higher else math.inf
            parts.append(-value if higher else value)
        return (*parts, name)

    return sorted(qualified, key=key)[0]


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


def run_study(spec: StudySpec, root: Path | str) -> dict[str, Any]:
    """Load the arms under ``root``, compare each with the incumbent, choose."""
    arms = load_arms(root)
    if spec.incumbent not in arms:
        raise statistics.ComparisonError(
            f"The incumbent arm {spec.incumbent!r} is not under {root}."
        )
    incumbent = arms[spec.incumbent]
    splits = {name: (arm.get("config") or {}).get("split") for name, arm in arms.items()}
    if set(splits.values()) != {"dev"}:
        raise statistics.ComparisonError(
            f"A {spec.dd} study is run on --split dev only; found {splits}."
        )

    values = {name: arm_values(arm) for name, arm in arms.items()}
    rows = []
    qualified = []
    for name, arm in arms.items():
        config = arm.get("config") or {}
        row: dict[str, Any] = {
            "arm": name,
            "factors": {f: config.get(f) for f in spec.factors},
            "reference": name in spec.references,
            "incumbent": name == spec.incumbent,
            "values": {c: values[name].get(c) for c in spec.columns},
            "contrasts": {},
        }
        if name != spec.incumbent:
            # The study's factor is the treatment, so the section 10 check is
            # expected to see it vary; it is recorded in the comparison.
            comparison = statistics.compare(
                incumbent, arm, spec.incumbent, name, allow_confounded=True
            )
            row["confounded_by"] = sorted(comparison["fair_comparison"]["mismatched"])
            row["contrasts"] = {
                m: _contrast_summary(comparison["contrasts"][m])
                for m in spec.primary + spec.guard
            }
            row["qualifies"] = (not row["reference"]) and qualifies(spec, row["contrasts"])
            if row["qualifies"]:
                qualified.append(name)
        rows.append(row)

    chosen = choose(spec, values, qualified)
    extra = {"strategy_by_type": strategy_by_type(arms)} if spec.name == "chunking" else {}
    return {
        "study": spec.name,
        "dd": spec.dd,
        "stack": note_for_config(incumbent.get("config") or {}),
        "split": "dev",
        "n_questions": len(incumbent.get("per_question") or []),
        "rule": {
            "incumbent": spec.incumbent,
            "primary": list(spec.primary),
            "guard": list(spec.guard),
            "tiebreak": [list(t) for t in spec.tiebreak],
            "references": list(spec.references),
        },
        "qualified": qualified,
        "chosen": chosen,
        "rows": rows,
        "notes": list(spec.notes),
        **extra,
    }


def _contrast_summary(contrast: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("n", "mean_difference", "ci_low", "ci_high", "questions_changed", "verdict")
    return {k: contrast.get(k) for k in keys}


# ---------------------------------------------------------------------------
# Chunking only: where structure-aware chunking loses
# ---------------------------------------------------------------------------

#: Per-question figures compared between matched structure/fixed arms.
BY_TYPE_METRICS = ("recall_at_5", "accuracy_all")


def strategy_by_type(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Structure-aware vs fixed, matched on size and overlap, per question type.

    For every (chunk_size, chunk_overlap) run under both strategies, each
    question's figure under ``structure`` is compared with the same question
    under ``fixed``. Counted per question type: how many (question, setting)
    pairs structure wins, ties and loses. Descriptive: with 33 dev questions
    these are counts to read, not tests.
    """
    by_setting: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    for arm in arms.values():
        config = arm.get("config") or {}
        key = (config.get("chunk_size"), config.get("chunk_overlap"))
        by_setting.setdefault(key, {})[str(config.get("chunking_strategy"))] = arm

    tally: dict[str, dict[str, dict[str, int]]] = {}
    losses: list[dict[str, Any]] = []
    settings = []
    for (size, overlap), pair in sorted(by_setting.items()):
        if not {"structure", "fixed"} <= set(pair):
            continue
        settings.append([size, overlap])
        types = {
            str(r["question_id"]): (r.get("metrics") or {}).get("question_type")
            for r in pair["structure"].get("per_question") or []
        }
        structure = statistics.per_question_vectors(pair["structure"].get("per_question") or [])
        fixed = statistics.per_question_vectors(pair["fixed"].get("per_question") or [])
        for metric in BY_TYPE_METRICS:
            for qid in sorted(set(structure[metric]) & set(fixed[metric])):
                qtype = str(types.get(qid))
                cell = tally.setdefault(metric, {}).setdefault(
                    qtype, {"wins": 0, "ties": 0, "losses": 0}
                )
                delta = structure[metric][qid] - fixed[metric][qid]
                if delta > 0:
                    cell["wins"] += 1
                elif delta < 0:
                    cell["losses"] += 1
                    losses.append(
                        {"question_id": qid, "question_type": qtype, "metric": metric,
                         "chunk_size": size, "chunk_overlap": overlap,
                         "structure": structure[metric][qid], "fixed": fixed[metric][qid]}
                    )
                else:
                    cell["ties"] += 1
    return {"settings": settings, "tally": tally, "losses": losses}


def _render_by_type(data: Mapping[str, Any]) -> list[str]:
    lines = [
        "### Where structure-aware chunking loses: structure vs fixed, matched on "
        "size and overlap",
        "",
        f"Per (question, setting) pair over {len(data['settings'])} settings, counted "
        "as structure wins / ties / losses. Descriptive only.",
        "",
    ]
    tally = data["tally"]
    types = sorted({t for metric in tally.values() for t in metric})
    lines.append("| question type | " + " | ".join(tally) + " |")
    lines.append("| --- |" + " ---: |" * len(tally))
    for qtype in types:
        cells = []
        for metric in tally:
            c = tally[metric].get(qtype)
            cells.append(f"{c['wins']} / {c['ties']} / {c['losses']}" if c else "n/a")
        lines.append(f"| {qtype} | " + " | ".join(cells) + " |")
    lost: dict[tuple, list[str]] = {}
    for loss in data["losses"]:
        lost.setdefault((loss["question_id"], loss["question_type"], loss["metric"]), []).append(
            f"{loss['chunk_size']}/{loss['chunk_overlap']}"
        )
    if lost:
        lines += ["", "Questions structure loses on (settings as size/overlap):", ""]
        for (qid, qtype, metric), where in sorted(lost.items()):
            lines.append(f"* {qid} ({qtype}), {metric}: {', '.join(where)}")
    return lines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_contrast(c: Mapping[str, Any] | None) -> str:
    if not c or c.get("mean_difference") is None:
        return "n/a"
    mark = {
        statistics.SIGNIFICANT_IMPROVEMENT: " ▲",
        statistics.SIGNIFICANT_REGRESSION: " ▼",
    }.get(c.get("verdict"), "")
    return f"{c['mean_difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]{mark}"


def render(study: Mapping[str, Any]) -> str:
    rule = study["rule"]
    lines = [
        f"# {study['study'].capitalize()} study ({study['dd']}), dev split, "
        f"{study['n_questions']} questions",
        "",
        "Generated by `python -m src.cli sweep-table`; do not edit by hand.",
        "",
        f"> {study['stack']}",
        "",
        f"**Rule (declared in {study['dd']} Part 1, before any run).** The incumbent "
        f"`{rule['incumbent']}` stays unless a challenger improves "
        f"{' or '.join(rule['primary'])} with a paired-bootstrap 95% CI excluding 0 "
        f"and significantly regresses none of {', '.join(rule['primary'] + rule['guard'])}. "
        "Among qualifiers: "
        + ", then ".join(f"{'highest' if h else 'lowest'} {m}" for m, h in rule["tiebreak"])
        + ".",
        "",
        f"**Qualified:** {', '.join(study['qualified']) or 'none'}. "
        f"**Chosen:** `{study['chosen']}`.",
        "",
    ]
    columns = list(study["rows"][0]["values"]) if study["rows"] else []
    header = ["arm", *columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- |" + " ---: |" * len(columns))
    for row in study["rows"]:
        name = f"**{row['arm']}** (incumbent)" if row["incumbent"] else row["arm"]
        if row["reference"]:
            name += " (reference)"
        lines.append(
            "| " + " | ".join([name, *(_fmt(row["values"][c]) for c in columns)]) + " |"
        )

    metrics = rule["primary"] + rule["guard"]
    lines += [
        "",
        "### Each arm − incumbent, 95% paired-bootstrap CI (▲/▼ significant)",
        "",
        "| arm | " + " | ".join(metrics) + " | qualifies |",
        "| --- |" + " --- |" * len(metrics) + " :---: |",
    ]
    for row in study["rows"]:
        if row["incumbent"]:
            continue
        cells = [_fmt_contrast(row["contrasts"].get(m)) for m in metrics]
        verdict = "reference" if row["reference"] else ("yes" if row.get("qualifies") else "no")
        lines.append(f"| {row['arm']} | " + " | ".join(cells) + f" | {verdict} |")

    if study.get("strategy_by_type"):
        lines += [""] + _render_by_type(study["strategy_by_type"])
    if study["notes"]:
        lines += [""] + [f"* {note}" for note in study["notes"]]
    return "\n".join(lines) + "\n"


def write_study(study: Mapping[str, Any], out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"table": out / TABLE_FILENAME, "json": out / TABLE_JSON_FILENAME}
    paths["table"].write_text(render(study), encoding="utf-8", newline="\n")
    paths["json"].write_text(
        json.dumps(study, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return paths
