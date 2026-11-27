"""Arithmetic uses only numeric cells available in the answer prompt."""

from dataclasses import replace
import json

import pytest

from src.chunking.base import Chunk
from src.config import GenerationConfig, RAGConfig
from src.generation.generator import GroundedGenerator
from src.generation.prompts import ABSTENTION_SENTENCE, DOCUMENT_TEXT_IS_DATA
from src.generation.table_calculator import Table, calculate
from src.pipeline import build_pipeline
from src.retrieval.retriever import RetrievedChunk


def table_chunk(text, page, *, kind="table"):
    chunk = Chunk(f"table-{page}", "doc", text, (page,), metadata={"kind": kind})
    return RetrievedChunk(chunk, score=1.0, rank=page)


class Selector:
    name = "test/selector"

    def __init__(self, response):
        self.response = response
        self.prompt = None

    def generate(self, system_prompt, user_prompt):
        assert DOCUMENT_TEXT_IS_DATA in system_prompt
        self.prompt = json.loads(user_prompt)
        return self.response


@pytest.mark.parametrize(
    ("operation", "values", "result"),
    [
        ("sum", ("1,200", "2.5"), "1202.5"),
        ("difference", ("0.71", "0.83"), "0.12"),
        ("subtract", ("40%", "65%"), "-25 percentage points"),
        ("average", ("4 s", "6 s"), "5 s"),
        ("multiply", ("-2", ".3"), "-0.6"),
        ("divide", ("8 s", "4 s"), "2"),
        ("percent_change", ("40", "50"), "25%"),
    ],
)
def test_decimal_operations(operation, values, result):
    table = Table(1, ("Value",), ((values[0],), (values[1],)))
    calculation = calculate(operation, '{"cells":[[1,1,1],[1,2,1]]}', [table])
    assert calculation is not None and calculation.result == result


@pytest.mark.parametrize(
    ("response", "question"),
    [
        ('{"cells":[[1,1,2],[2,1,2]]}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,3,2]]}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,1,2]]}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,2,1]]}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,true,2]]}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,2,2]],"result":12}', "What is the combined total?"),
        ('{"cells":[[1,1,2],[1,2,2]]}', "What is the ratio of the two rows?"),
        ("The total is 12. [C1]", "What is the combined total?"),
    ],
)
def test_unsafe_plans_refuse(response, question):
    table = table_chunk("| Label | Value |\n| --- | --- |\n| North | 12 |\n| South | 0 |", 4)
    result = GroundedGenerator(backend=Selector(response), table_calculator=True).generate(question, [table])
    assert result.answer == ABSTENTION_SENTENCE
    assert result.abstention_reason == "invalid_table_operands"
    assert result.citations == ()


def test_grounded_calculation_cites_each_source_page():
    north = table_chunk("| Label | Time (s) |\n| --- | --- |\n| North | 12 |", 44)
    south = table_chunk("| Label | Time (s) |\n| --- | --- |\n| South | 4 |", 57)
    backend = Selector('{"cells":[[1,1,2],[2,1,2]]}')
    result = GroundedGenerator(backend=backend, table_calculator=True).generate(
        "What is the sum of North and South Time?", [north, south],
    )
    assert result.answer == "16 s [C1, C2]"
    assert result.cited_pages == (44, 57)
    assert result.metadata["table_calculation"]["values"] == ("12", "4")
    assert backend.prompt == {
        "question": "What is the sum of North and South Time?",
        "operation": "sum",
        "tables": [
            {"evidence": 1, "columns": ["Label", "Time (s)"], "rows": [["North", "12"]]},
            {"evidence": 2, "columns": ["Label", "Time (s)"], "rows": [["South", "4"]]},
        ],
    }


def test_truncated_rows_and_prose_are_unavailable():
    prefix = "| Label | Time |\n| --- | --- |\n| North | 12 s |"
    table = table_chunk(prefix + "\n| South | 4 s |", 4)
    prose = table_chunk(prefix, 9, kind="text")
    backend = Selector('{"cells":[[1,1,2],[1,2,2]]}')
    generator = GroundedGenerator(
        GenerationConfig(max_evidence_chars=len(prefix) + 10), backend, table_calculator=True,
    )
    question = "What is the combined total?"
    assert generator.generate(question, [prose]).model_calls == 0
    assert backend.prompt is None
    result = generator.generate(question, [table])
    assert result.abstained and result.model_calls == 1
    assert backend.prompt["tables"][0]["rows"] == [["North", "12 s"]]


def test_mismatched_units_refuse():
    table = Table(1, ("Value",), (("1 s",), ("2 ms",)))
    assert calculate("sum", '{"cells":[[1,1,1],[1,2,1]]}', [table]) is None


def test_offline_selector_uses_named_column_and_subtraction_order():
    table = table_chunk("| Label | Time (s) |\n| --- | --- |\n| North | 12 |\n| South | 4 |", 4)
    generator = GroundedGenerator(GenerationConfig(backend="scripted"), table_calculator=True)
    result = generator.generate("Subtract South Time from North Time.", [table])
    assert result.answer == "8 s [C1]"


@pytest.mark.parametrize("strategy", ["agentic", "adaptive"])
def test_calculator_runs_in_the_pipeline_and_verifies_derived_numbers(structured_pdf_path, strategy):
    config = RAGConfig.offline()
    pipeline = build_pipeline(replace(config, retrieval=replace(config.retrieval, strategy=strategy)))
    pipeline.index(structured_pdf_path)

    result = pipeline.ask("What is the sum of Dense and Hybrid Recall?")

    assert result.answer.startswith("1.54 [C")
    assert result.cited_pages == (2,)
    assert result.metadata["verification_status"] == "passed"
    assert result.metadata["agent_trace"]["verifications"][0]["backend"] == "table_calculator"
