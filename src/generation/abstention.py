"""Abstention detection (BENCHMARK_SPEC.md section 13.2).

Abstention is read off the answer text by fixed rules, never by asking a model
whether another model refused: EVALUATION_PROTOCOL.md section 19.2 excludes the
judge from this decision because a model asked that question over-reads hedging
as refusal, which would inflate abstention accuracy and hide false answers.

The rule that does the real work is the second one. A long answer that mentions
a gap in passing --

    "the document does not state the exact date, but the experiment ran in March"

-- is an **answer**, and a detector that calls it an abstention credits the
system for refusing while it was in fact asserting something. So a paraphrase
only counts when the refusal is the whole of the answer: short, and not followed
by a contrastive clause that supplies content anyway.

The pattern set is versioned. BENCHMARK_SPEC.md requires it to be fixed before a
run and recorded with the results, for the same reason the judge prompt is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import ABSTENTION_SENTENCE

#: Bump when the patterns below change. Recorded alongside every result set, so
#: two runs scored under different rules are never compared as if they were not.
ABSTENTION_PATTERNS_VERSION = "2026-09-21.1"

#: A refusal is a sentence. Past this length the answer is saying something.
MAX_ABSTENTION_CHARS = 320
MAX_ABSTENTION_SENTENCES = 2

_PARAPHRASES: tuple[str, ...] = (
    r"^(?:i|we)\s+(?:could not|cannot|can't|couldn't|am unable to|are unable to)\s+"
    r"(?:find|locate|determine|identify|establish|answer)",
    r"^(?:the\s+)?(?:document|text|pdf|passage|excerpt|evidence|context|provided\s+\w+)\s+"
    r"(?:does not|doesn't|do not|don't)\s+"
    r"(?:contain|provide|include|state|mention|specify|say|indicate|address)",
    r"^(?:there\s+(?:is|are)|there's)\s+(?:no|not enough|insufficient)\s+"
    r"(?:information|evidence|detail|mention)",
    r"^(?:insufficient|not enough|no)\s+(?:evidence|information)\b",
    r"^(?:unable|not possible)\s+to\s+(?:answer|determine|establish)",
    r"^(?:this|the)\s+question\s+cannot\s+be\s+answered",
)

_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _PARAPHRASES)

#: A refusal followed by one of these is not a refusal: the clause after it is
#: where the system says the thing it will be judged on.
_CONTRAST = re.compile(
    r"\b(?:but|however|although|though|nevertheless|nonetheless|that said|"
    r"instead|whereas)\b",
    re.IGNORECASE,
)

_MARKER = re.compile(r"\[[^\[\]]{1,60}\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PUNCT = re.compile(r"[\s.!?,;:\"']+$")


@dataclass(frozen=True)
class AbstentionVerdict:
    abstained: bool
    #: "canonical", "paraphrase", "empty", or "" when the system answered.
    reason: str = ""
    patterns_version: str = ABSTENTION_PATTERNS_VERSION

    def __bool__(self) -> bool:
        return self.abstained


def _normalize(text: str) -> str:
    stripped = _MARKER.sub("", text)
    collapsed = " ".join(stripped.split()).lower()
    return _PUNCT.sub("", collapsed)


def detect_abstention(answer: str) -> AbstentionVerdict:
    """Classify an answer as a refusal or an answer."""
    if not answer or not answer.strip():
        # An empty generation is a failure, but it asserts nothing, so it is
        # counted on the safe side rather than as a false answer.
        return AbstentionVerdict(True, "empty")

    normalized = _normalize(answer)
    if not normalized:
        return AbstentionVerdict(True, "empty")

    if normalized == _normalize(ABSTENTION_SENTENCE):
        return AbstentionVerdict(True, "canonical")

    # The canonical sentence on its own line, with nothing else of substance.
    if normalized.startswith(_normalize(ABSTENTION_SENTENCE)) and len(
        normalized
    ) <= len(_normalize(ABSTENTION_SENTENCE)) + 40:
        return AbstentionVerdict(True, "canonical")

    if len(normalized) > MAX_ABSTENTION_CHARS:
        return AbstentionVerdict(False)

    sentences = [s for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]
    if len(sentences) > MAX_ABSTENTION_SENTENCES:
        return AbstentionVerdict(False)

    for pattern in _COMPILED:
        match = pattern.search(normalized)
        if not match:
            continue
        remainder = normalized[match.end() :]
        if _CONTRAST.search(remainder):
            # "...does not state the date, but the experiment ran in March"
            return AbstentionVerdict(False)
        return AbstentionVerdict(True, "paraphrase")

    return AbstentionVerdict(False)


def is_abstention(answer: str) -> bool:
    return detect_abstention(answer).abstained
