"""Prompt construction for grounded answering.

The important property of this format is what it leaves out. Evidence blocks
carry a marker and text and **no page numbers, chunk ids or section titles**.
The model is therefore structurally incapable of writing a page number that came
from the document's metadata: it has never seen one. Pages are attached
afterwards by ``citations.py``, which reads them from the stored chunk.

That is DD-018 applied to citations. Mapping a marker to a page is a lookup, and
a lookup is deterministic code; asking a 4B model to carry page numbers through
a generation step is asking it to do arithmetic it has no reason to get right.

The format is also machine-parseable in both directions, so the scripted
fallback backend can read the same prompt the transformer backend receives
instead of needing a private interface.
"""

from __future__ import annotations

import re

#: The one sentence the generator must emit when the evidence does not support
#: an answer. Fixed here and nowhere else: the abstention detector, the prompt
#: and the pipeline all refer to this constant, so they cannot drift apart.
ABSTENTION_SENTENCE = (
    "I could not find sufficient evidence in the uploaded document to answer "
    "this question."
)

EVIDENCE_HEADER = "EVIDENCE"
QUESTION_HEADER = "QUESTION"
MARKER_FORMAT = "[C{n}]"

SYSTEM_PROMPT = f"""You answer questions about a single document, using only the evidence supplied to you.

Rules:
1. Use only the numbered evidence below. Do not use outside knowledge, and do not guess.
2. Support each factual statement with the marker of the evidence it came from, written exactly as [C1], [C2], and so on. Place the marker at the end of the sentence it supports.
3. Only use markers that appear in the evidence list. Never invent a marker.
4. Never write a page number, a section number or a document name. You have not been shown them, and they are added automatically from the markers you use.
5. Be concise and factual. Do not restate the question or add a preamble.
6. If the evidence does not contain enough information to answer, reply with exactly this sentence and nothing else:
{ABSTENTION_SENTENCE}"""


def format_evidence_block(number: int, text: str) -> str:
    return f"{MARKER_FORMAT.format(n=number)}\n{text.strip()}"


def build_user_prompt(question: str, blocks: list[tuple[int, str]]) -> str:
    """Render the evidence list and the question."""
    rendered = "\n\n".join(format_evidence_block(n, text) for n, text in blocks)
    return (
        f"{EVIDENCE_HEADER}\n{rendered}\n\n"
        f"{QUESTION_HEADER}\n{question.strip()}\n\n"
        "Answer:"
    )


_BLOCK_SPLIT = re.compile(r"^\[C(\d+)\]\s*$", re.MULTILINE)


def parse_user_prompt(prompt: str) -> tuple[str, list[tuple[int, str]]]:
    """Recover the question and evidence blocks from a rendered prompt.

    Used by the scripted backend, which is a text-in/text-out backend like any
    other and so must read its input the way a model would.
    """
    question = ""
    body = prompt
    marker = f"\n{QUESTION_HEADER}\n"
    if marker in prompt:
        body, _, tail = prompt.partition(marker)
        question = tail.split("\n\nAnswer:")[0].strip()

    body = body.split(f"{EVIDENCE_HEADER}\n", 1)[-1]
    parts = _BLOCK_SPLIT.split(body)
    blocks: list[tuple[int, str]] = []
    # split() yields [prefix, num, text, num, text, ...]
    for index in range(1, len(parts) - 1, 2):
        number = int(parts[index])
        text = parts[index + 1].strip()
        if text:
            blocks.append((number, text))
    return question, blocks
