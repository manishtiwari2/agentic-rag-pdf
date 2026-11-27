"""Select and calculate with cells from the prompt's table evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext

from .evidence import EvidenceItem

_OPERATIONS = {
    "sum": r"\b(?:sum of|combined total|add together)\b",
    "difference": r"\b(?:(?:absolute|numeric(?:al)?) difference|(?:calculate|compute) (?:the )?difference)\b",
    "subtract": r"\b(?:minus|subtract|how (?:much|many) more)\b",
    "multiply": r"\b(?:multiply|multiplied by|product of)\b",
    "divide": r"\b(?:divide|divided by|ratio of)\b",
    "average": r"\b(?:average of|mean of)\b",
    "percent_change": r"\b(?:percent(?:age)? change)\b",
}
_CELL = re.compile(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)\s*(%|[A-Za-z][A-Za-z0-9]*)?")
_HEADER_UNIT = re.compile(r"\((%|[A-Za-z][A-Za-z0-9]*)\)\s*$")


def operation_for(question: str) -> str | None:
    matches = [op for op, pattern in _OPERATIONS.items() if re.search(pattern, question, re.IGNORECASE)]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else "ambiguous"


def _pipe_row(line: str) -> tuple[str, ...]:
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|")):
        return ()
    return tuple(cell.strip() for cell in line[1:-1].split("|"))


@dataclass(frozen=True)
class Table:
    marker: int
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_record(self) -> dict[str, object]:
        return {"evidence": self.marker, "columns": self.columns, "rows": self.rows}


def prompt_tables(evidence: list[EvidenceItem]) -> list[Table]:
    tables = []
    for item in evidence:
        if item.chunk.metadata.get("kind") != "table":
            continue
        lines = [_pipe_row(line) for line in item.text.splitlines()]
        for index, rule in enumerate(lines[1:], start=1):
            header = lines[index - 1]
            if not rule or len(rule) != len(header) or any(cell != "---" for cell in rule):
                continue
            rows = []
            for row in lines[index + 1:]:
                if len(row) != len(header):
                    break
                rows.append(row)
            if rows:
                tables.append(Table(item.number, header, tuple(rows)))
            break
    return tables


@dataclass(frozen=True)
class TableCalculation:
    operation: str
    cells: tuple[tuple[int, int, int], ...]
    values: tuple[str, ...]
    result: str

    def answer(self) -> str:
        markers = ", ".join(f"C{number}" for number in dict.fromkeys(cell[0] for cell in self.cells))
        return f"{self.result} [{markers}]"

    def to_record(self) -> dict[str, object]:
        return {"operation": self.operation, "cells": self.cells, "values": self.values, "result": self.result}


def _number(cell: str) -> tuple[Decimal, str] | None:
    match = _CELL.fullmatch(cell)
    if not match or len(match[1]) > 24:
        return None
    return Decimal(match[1].replace(",", "")), match[2] or ""


def calculate(operation: str, raw: str, tables: list[Table]) -> TableCalculation | None:
    try:
        plan = json.loads(raw) if len(raw) <= 4096 else None
    except (ValueError, RecursionError):
        return None
    if not isinstance(plan, dict) or set(plan) != {"cells"} or not isinstance(plan["cells"], list):
        return None
    cells = plan["cells"]
    if not 2 <= len(cells) <= 8:
        return None
    if operation in {"difference", "subtract", "divide", "percent_change"} and len(cells) != 2:
        return None

    by_marker = {table.marker: table for table in tables}
    references, values, numbers, units = [], [], [], []
    for cell in cells:
        if not isinstance(cell, list) or len(cell) != 3 or any(type(part) is not int or part < 1 for part in cell):
            return None
        marker, row, column = cell
        table = by_marker.get(marker)
        if table is None or row > len(table.rows) or column > len(table.columns):
            return None
        coordinate = (marker, row, column)
        value = table.rows[row - 1][column - 1]
        parsed = _number(value)
        if coordinate in references or parsed is None:
            return None
        references.append(coordinate)
        values.append(value)
        number, unit = parsed
        header_unit = _HEADER_UNIT.search(table.columns[column - 1])
        if header_unit and unit and unit != header_unit[1]:
            return None
        numbers.append(number)
        units.append(unit or (header_unit[1] if header_unit else ""))

    if len(set(units)) != 1 or operation == "multiply" and units[0]:
        return None
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        result = _arithmetic(operation, numbers, units[0])
    return TableCalculation(operation, tuple(references), tuple(values), result) if result is not None else None


def _arithmetic(operation: str, numbers: list[Decimal], unit: str) -> str | None:
    first, second = numbers[:2]
    if operation == "sum":
        result = sum(numbers, Decimal(0))
    elif operation in {"difference", "subtract"}:
        result = abs(first - second) if operation == "difference" else first - second
        if unit == "%":
            unit = "percentage points"
    elif operation == "multiply":
        result = Decimal(1)
        for number in numbers:
            result *= number
    elif operation == "average":
        result = sum(numbers, Decimal(0)) / len(numbers)
    elif operation == "divide" and second:
        result, unit = first / second, ""
    elif operation == "percent_change" and first > 0:
        result, unit = (second - first) / first * 100, "%"
    else:
        return None
    value = format(result.normalize() if result else Decimal(0), "f")
    separator = " " if unit and unit != "%" else ""
    return f"{value}{separator}{unit}"


def scripted_cell_selection(prompt: str) -> str:
    """The offline stand-in can select rows and a column named in the question."""
    data = json.loads(prompt)
    question = data["question"]

    def position(label: str) -> int | None:
        match = re.search(rf"(?<!\w){re.escape(label)}(?!\w)", question, re.IGNORECASE) if label else None
        return match.start() if match else None

    selected = []
    for table in data["tables"]:
        columns = [
            i for i, label in enumerate(table["columns"])
            if position(_HEADER_UNIT.sub("", label).strip()) is not None
        ]
        if len(columns) != 1:
            continue
        column = columns[0]
        for row_number, row in enumerate(table["rows"], start=1):
            matched = position(row[0])
            if matched is not None and _number(row[column]) is not None:
                selected.append((matched, [table["evidence"], row_number, column + 1]))
    if len({matched for matched, _ in selected}) != len(selected):
        return json.dumps({"cells": []})
    subtract_from = data["operation"] == "subtract" and bool(
        re.search(r"\bsubtract\b.*\bfrom\b", question, re.IGNORECASE)
    )
    return json.dumps({"cells": [cell for _, cell in sorted(selected, reverse=subtract_from)]})
