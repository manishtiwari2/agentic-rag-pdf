"""Text utilities the rule-based agents share, and the JSON reader the LLM ones do.

The tokenizer is the lexical retriever's (DD-049): numbers stay whole, so
``3.14`` is one token and an evidence check on figures compares figures. On top
of it sits a question-word stopword list, because "how", "which" and "does"
carry no information about whether a passage covers a question.

``extract_json`` is the only place an LLM agent's output is parsed. It is
strict on purpose -- EXPERIMENT_PLAN.md section 4 expects a small model to
produce unreliable structured output, and a lenient parser would turn that
unreliability into silent wrong decisions instead of a counted parse failure
(DD-055).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..retrieval.lexical import tokenize

_STOPWORDS: frozenset[str] = frozenset(
    """a an and any are as at be been being both but by can could did do does doing
    each either for from had has have how i if in into is it its itself me more most
    my no nor not of on or other our out over own same she should so some such than
    that the their them then there these they this those through to too under up us
    very was we were what when where which while who whom why will with within would
    you your according described describe document paper booklet text say says said
    state states stated mention mentions give given""".split()
)

_NUMBER = re.compile(r"^\d+(?:\.\d+)*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKER = re.compile(r"\[[Cc]?\s*\d+(?:\s*[,;]\s*[Cc]?\s*\d+)*\]")
#: The only form ``resolve_citations`` emits. A bare ``[3]`` still in a resolved
#: answer is source text the answer quoted, not a citation (DD-062).
_CANONICAL_MARKER = re.compile(r"\[C\d+(?:, C\d+)*\]")


def content_terms(text: str) -> list[str]:
    """Lower-cased content tokens, in order, stopwords and 1-letter words removed."""
    stripped = _MARKER.sub(" ", text or "")
    return [
        token
        for token in tokenize(stripped)
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


#: Suffixes stripped so "embeds", "embedded" and "embedding" count as one term
#: when coverage and support are checked. Crude on purpose, like the scripted
#: backend's: a word-form match, not morphology. Numbers are never stemmed.
_SUFFIXES = ("ations", "ation", "ingly", "ings", "ing", "edly", "ed", "es", "s", "ly")


def stem(token: str) -> str:
    if _NUMBER.match(token):
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            root = token[: -len(suffix)]
            if len(root) > 4 and root[-1] == root[-2] and root[-1] not in "aeiou":
                root = root[:-1]
            return root
    return token


def stems(text: str) -> set[str]:
    """The stemmed content terms of ``text``, as a set."""
    return {stem(t) for t in content_terms(text)}


def numbers_in(text: str) -> set[str]:
    """The numbers in ``text`` as the lexical tokenizer sees them (``1,200`` -> ``1200``)."""
    return {token for token in tokenize(_MARKER.sub(" ", text or "")) if _NUMBER.match(token)}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]


def cited_numbers(sentence: str) -> list[int]:
    """The evidence numbers a sentence's markers point at, in order."""
    found: list[int] = []
    for group in _CANONICAL_MARKER.findall(sentence):
        for number in re.findall(r"\d+", group):
            if int(number) not in found:
                found.append(int(number))
    return found


class JSONParseError(ValueError):
    """An LLM agent's output was not the JSON object its prompt asked for."""


def extract_json(raw: str) -> dict[str, Any]:
    """The first balanced ``{...}`` in ``raw``, parsed, or ``JSONParseError``.

    Tolerates prose or a code fence around the object -- instruct models add
    both -- and nothing else: no trailing-comma repair, no single quotes, no
    guessing at a truncated object.
    """
    text = raw or ""
    start = text.find("{")
    if start < 0:
        raise JSONParseError("no JSON object in the output")
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : position + 1])
                except json.JSONDecodeError as exc:
                    raise JSONParseError(f"invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise JSONParseError("the JSON value is not an object")
                return value
    raise JSONParseError("unterminated JSON object")


def truncate(text: str, limit: int = 400) -> str:
    """Raw model output as recorded in a trace: enough to diagnose, bounded."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " [...]"
