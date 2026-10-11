"""Stale and conflicting sources: detection and the declared policy (DD-074).

Detection runs on the evidence the generator would see, and only when that
evidence comes from two or more documents. Two detectors, as for every other
agent decision (DD-055):

* :class:`RuleBasedConflictDetector` -- offline. Sentences from different
  documents make the *same claim* when the Jaccard overlap of their anchors
  (content terms with the values removed) is at least ``anchor_threshold``,
  and *conflict* when their values of one kind -- number, date or entity --
  differ.
* :class:`LLMConflictDetector` -- the model stack. The model is shown the
  candidate pairs and names the conflicting ones; values are re-extracted by
  code, never taken from the model. Unusable output is ``unavailable`` and
  asserts no conflict.

:func:`apply_policy` is DD-074 section 3, applied by code:

* no conflict: the answer is unchanged;
* both dates known and different: answer from the newer source (the older
  side's chunk is withheld from the generator) and append one sentence, built
  by code, that names both and cites both;
* either date unknown, or equal at their shared precision: refuse with the
  canonical sentence, followed by both claims, each cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..config import RULE_CONFLICT_DETECTOR, ConflictConfig
from ..generation.citations import Citation
from ..generation.evidence import EvidenceItem, build_evidence
from ..generation.generator import GeneratedAnswer
from ..generation.prompts import ABSTENTION_SENTENCE
from ..versioning import DocumentVersion, compare_versions
from ..retrieval.retriever import RetrievedChunk
from .text import JSONParseError, content_terms, extract_json, sentences, stem, truncate

NUMBER, DATE, ENTITY = "number", "date", "entity"
#: When two sentences differ in more than one kind, the first kind here names it.
KINDS = (NUMBER, DATE, ENTITY)
POLICY_NONE, POLICY_NEWER, POLICY_REFUSED = "none", "prefer_newer", "refused"
ABSTENTION_REASON = "conflicting_sources"

_MONTHS = (
    "january february march april may june july august september october november "
    "december jan feb mar apr jun jul aug sep sept oct nov dec"
).split()
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE = re.compile(
    rf"\b(?:(?:\d{{1,2}}\s+)?(?:{_MONTH_RE})\.?\s+)?(19\d{{2}}|20\d{{2}})\b", re.IGNORECASE
)
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:\s?%)?")
#: Two or more capitalised words in a row: a name, a place, an organisation.
_ENTITY = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    """One document's statement of a claim."""

    document_id: str
    source_name: str
    version: DocumentVersion
    sentence: str
    values: tuple[str, ...]
    chunk_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_name": self.source_name,
            "version": self.version.to_record(),
            "sentence": self.sentence,
            "values": list(self.values),
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class Conflict:
    claim: tuple[str, ...]
    kind: str
    sides: tuple[Side, Side]
    overlap: float

    def to_record(self) -> dict[str, Any]:
        return {
            "claim": list(self.claim),
            "kind": self.kind,
            "overlap": round(self.overlap, 4),
            "sides": [s.to_record() for s in self.sides],
        }


@dataclass
class ConflictReport:
    conflicts: list[Conflict] = field(default_factory=list)
    detector: str = RULE_CONFLICT_DETECTOR
    status: str = "ok"  # "ok" | "unavailable" | "single_document"
    model_calls: int = 0
    raw: str = ""
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "detector": self.detector,
            "status": self.status,
            "model_calls": self.model_calls,
            "conflicts": [c.to_record() for c in self.conflicts],
        }
        if self.error:
            record["error"] = self.error
        if self.raw:
            record["raw"] = self.raw
        return record


# ---------------------------------------------------------------------------
# Claims: values and anchors
# ---------------------------------------------------------------------------


def _normalise_number(text: str) -> str:
    return text.replace(",", "").replace(" ", "")


def extract_values(sentence: str) -> dict[str, dict[str, str]]:
    """``{kind: {normalised: as written}}`` for numbers, dates and entities.

    A year (1900-2099) counts as a date, not a number, so "in 2024" against
    "in 2023" is a date disagreement and "412,000" against "398,500" a number one.
    """
    found: dict[str, dict[str, str]] = {NUMBER: {}, DATE: {}, ENTITY: {}}
    date_spans = []
    for match in _DATE.finditer(sentence):
        found[DATE][" ".join(match.group(0).lower().split())] = match.group(0)
        date_spans.append(match.span(1))
    for match in _NUM.finditer(sentence):
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        text = match.group(0).strip()
        found[NUMBER][_normalise_number(text)] = text
    first_word = sentence.lstrip().split(" ", 1)[0] if sentence.strip() else ""
    for match in _ENTITY.finditer(sentence):
        words = match.group(0).split()
        if match.start() == len(sentence) - len(sentence.lstrip()) and words[0] == first_word:
            # A capitalised sentence opener is not a name: drop that word.
            words = words[1:]
        if len(words) >= 2 and words[0].lower() not in _MONTHS:
            found[ENTITY][" ".join(words).lower()] = " ".join(words)
    return found


def anchor(sentence: str, values: Mapping[str, Mapping[str, str]]) -> frozenset[str]:
    """Content stems of the sentence with every value's words removed."""
    value_words = {
        word.lower().strip(",.%")
        for kind in values.values()
        for written in kind.values()
        for word in written.split()
    }
    return frozenset(
        stem(term)
        for term in content_terms(sentence)
        if term not in value_words and term not in _MONTHS and not term[0].isdigit()
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


@dataclass(frozen=True)
class _Sentence:
    retrieved: RetrievedChunk
    text: str
    values: dict[str, dict[str, str]]
    anchor: frozenset[str]

    @property
    def document_id(self) -> str:
        return self.retrieved.chunk.document_id


def _candidate_sentences(question: str, retrieved: Sequence[RetrievedChunk]) -> list[_Sentence]:
    """Evidence sentences sharing a content term with the question, with a value."""
    asked = {stem(t) for t in content_terms(question)}
    out = []
    for item in retrieved:
        body = item.chunk.text
        for text in sentences(body.replace("\n", " ")):
            values = extract_values(text)
            if not any(values.values()):
                continue
            anchored = anchor(text, values)
            if asked & anchored:
                out.append(_Sentence(item, text, values, anchored))
    return out


def _differing_kind(a: _Sentence, b: _Sentence) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    for kind in KINDS:
        va, vb = a.values[kind], b.values[kind]
        if va and vb and set(va) - set(vb) and set(vb) - set(va):
            return (
                kind,
                tuple(va[k] for k in va if k not in vb),
                tuple(vb[k] for k in vb if k not in va),
            )
    return None


def _cross_document_pairs(found: list[_Sentence]):
    for i, a in enumerate(found):
        for b in found[i + 1 :]:
            if a.document_id != b.document_id:
                yield a, b


def _side(s: _Sentence, values: tuple[str, ...], versions, names) -> Side:
    doc = s.document_id
    return Side(
        document_id=doc,
        source_name=names.get(doc, doc),
        version=versions.get(doc, DocumentVersion()),
        sentence=s.text,
        values=values,
        chunk_id=s.retrieved.chunk_id,
    )


def _dedupe(conflicts: list[Conflict]) -> list[Conflict]:
    """One conflict per (document pair, kind, differing values)."""
    seen = set()
    kept = []
    for c in sorted(conflicts, key=lambda c: -c.overlap):
        key = (
            frozenset((s.document_id, s.values) for s in c.sides),
            c.kind,
        )
        if key not in seen:
            seen.add(key)
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class ConflictDetector(Protocol):
    name: str

    def detect(
        self,
        question: str,
        retrieved: Sequence[RetrievedChunk],
        versions: Mapping[str, DocumentVersion],
        names: Mapping[str, str],
    ) -> ConflictReport: ...


def _single_document(retrieved: Sequence[RetrievedChunk]) -> bool:
    return len({r.chunk.document_id for r in retrieved}) < 2


class RuleBasedConflictDetector:
    """Anchor overlap for "same claim", value disagreement for "conflict"."""

    def __init__(self, config: ConflictConfig | None = None) -> None:
        self.config = config or ConflictConfig(detector="rules")
        self.name = RULE_CONFLICT_DETECTOR

    def detect(self, question, retrieved, versions, names) -> ConflictReport:
        if _single_document(retrieved):
            return ConflictReport(detector=self.name, status="single_document")
        found = _candidate_sentences(question, retrieved)
        conflicts = []
        for a, b in _cross_document_pairs(found):
            overlap = jaccard(a.anchor, b.anchor)
            if overlap < self.config.anchor_threshold:
                continue
            differing = _differing_kind(a, b)
            if differing is None:
                continue
            kind, va, vb = differing
            conflicts.append(
                Conflict(
                    claim=tuple(sorted(a.anchor & b.anchor)),
                    kind=kind,
                    sides=(_side(a, va, versions, names), _side(b, vb, versions, names)),
                    overlap=overlap,
                )
            )
        return ConflictReport(conflicts=_dedupe(conflicts), detector=self.name)


CONFLICT_SYSTEM_PROMPT = (
    "You compare statements taken from different documents. For each numbered "
    "pair, decide whether both statements make the same claim and disagree about "
    "it (a different number, date or name for the same thing). Reply with JSON "
    'only: {"conflicts": [{"pair": <number>, "claim": "<short description>", '
    '"kind": "number" | "date" | "entity"}]}. Return an empty list if none disagree.'
)


class LLMConflictDetector:
    """Asks the generation model which candidate pairs conflict."""

    def __init__(self, backend, config: ConflictConfig | None = None, label: str = "llm") -> None:
        self.backend = backend
        self.config = config or ConflictConfig()
        self.name = label

    def detect(self, question, retrieved, versions, names) -> ConflictReport:
        if _single_document(retrieved):
            return ConflictReport(detector=self.name, status="single_document")
        found = _candidate_sentences(question, retrieved)
        pairs = sorted(
            _cross_document_pairs(found), key=lambda p: -jaccard(p[0].anchor, p[1].anchor)
        )[: self.config.max_llm_pairs]
        if not pairs:
            return ConflictReport(detector=self.name)
        prompt = f"Question: {question}\n\n" + "\n\n".join(
            f"Pair {i}:\nA: {a.text}\nB: {b.text}" for i, (a, b) in enumerate(pairs, start=1)
        )
        raw = ""
        try:
            raw = self.backend.generate(CONFLICT_SYSTEM_PROMPT, prompt)
            data = extract_json(raw)
            items = data.get("conflicts")
            if not isinstance(items, list):
                raise JSONParseError("conflicts is not a list")
            conflicts = []
            for item in items:
                index = int(item["pair"]) - 1
                if not 0 <= index < len(pairs):
                    raise JSONParseError(f"pair {index + 1} does not exist")
                a, b = pairs[index]
                differing = _differing_kind(a, b)
                kind, va, vb = differing or (
                    str(item.get("kind") or NUMBER),
                    tuple(v for vals in a.values.values() for v in vals.values()),
                    tuple(v for vals in b.values.values() for v in vals.values()),
                )
                conflicts.append(
                    Conflict(
                        claim=tuple(sorted(a.anchor & b.anchor)) or (str(item.get("claim", "")),),
                        kind=kind,
                        sides=(_side(a, va, versions, names), _side(b, vb, versions, names)),
                        overlap=jaccard(a.anchor, b.anchor),
                    )
                )
        except (JSONParseError, KeyError, TypeError, ValueError) as exc:
            return ConflictReport(
                detector=self.name, status="unavailable", model_calls=1,
                raw=truncate(raw), error=str(exc),
            )
        return ConflictReport(
            conflicts=_dedupe(conflicts), detector=self.name, model_calls=1, raw=truncate(raw)
        )


def build_conflict_detector(config: ConflictConfig, backend=None, label: str = "llm") -> ConflictDetector:
    if config.detector == "rules" or backend is None:
        return RuleBasedConflictDetector(config)
    return LLMConflictDetector(backend, config, label)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def _date_text(version: DocumentVersion) -> str:
    if not version.known:
        return "undated"
    return f"{version.date}, from {version.source.replace('_', ' ')}"


def _evidence_number(
    evidence: list[EvidenceItem], retrieved: RetrievedChunk
) -> int:
    """The evidence number of ``retrieved``, appending it if absent."""
    for item in evidence:
        if item.chunk_id == retrieved.chunk_id:
            return item.number
    number = len(evidence) + 1
    evidence.append(EvidenceItem(number=number, text=retrieved.chunk.text, retrieved=retrieved))
    return number


def _citation(evidence: list[EvidenceItem], number: int) -> Citation:
    item = next(e for e in evidence if e.number == number)
    chunk = item.chunk
    return Citation(
        marker=f"C{number}",
        evidence_number=number,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        pages=tuple(chunk.pages),
        section=chunk.section,
    )


def _markers(numbers: Sequence[int]) -> str:
    return "[" + ", ".join(f"C{n}" for n in numbers) + "]"


def newer_side(conflict: Conflict) -> tuple[Side, Side] | None:
    """(newer, older), or None when the dates cannot order them (DD-074)."""
    a, b = conflict.sides
    order = compare_versions(a.version, b.version)
    if not order:  # None (unknown) or 0 (equal at shared precision)
        return None
    return (a, b) if order > 0 else (b, a)


def apply_policy(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    report: ConflictReport,
    generate: Callable[[list[RetrievedChunk]], GeneratedAnswer],
    generation_config: Any,
    backend_name: str,
) -> tuple[GeneratedAnswer, str]:
    """DD-074 section 3. Returns the answer and the policy that produced it."""
    if not report.conflicts:
        return generate(list(retrieved)), POLICY_NONE

    by_id = {r.chunk_id: r for r in retrieved}
    ordered = [newer_side(c) for c in report.conflicts]

    if any(o is None for o in ordered):
        # Refuse, and show every claim that could not be ordered.
        chunks = []
        for conflict, o in zip(report.conflicts, ordered):
            if o is None:
                chunks += [by_id[s.chunk_id] for s in conflict.sides if s.chunk_id in by_id]
        unique = list({r.chunk_id: r for r in chunks}.values())
        evidence = list(build_evidence(unique, generation_config))
        parts, citations = [], []
        for conflict, o in zip(report.conflicts, ordered):
            if o is not None:
                continue
            for side in conflict.sides:
                number = _evidence_number(evidence, by_id[side.chunk_id])
                citations.append(_citation(evidence, number))
                parts.append(
                    f'{side.source_name} ({_date_text(side.version)}) says "{side.sentence}" '
                    f"{_markers([number])}"
                )
        answer = f"{ABSTENTION_SENTENCE} The sources disagree: " + "; ".join(parts) + "."
        return (
            GeneratedAnswer(
                answer=answer,
                raw_answer=answer,
                citations=tuple(dict.fromkeys(citations)),
                evidence=tuple(evidence),
                abstained=True,
                abstention_reason=ABSTENTION_REASON,
                backend=backend_name,
                metadata={"evidence_count": len(evidence)},
            ),
            POLICY_REFUSED,
        )

    withheld = {older.chunk_id for _, older in ordered} - {newer.chunk_id for newer, _ in ordered}
    generated = generate([r for r in retrieved if r.chunk_id not in withheld])
    if generated.abstained:
        return generated, POLICY_NEWER
    evidence = list(generated.evidence)
    notes, citations = [], list(generated.citations)
    for newer, older in ordered:
        n_new = _evidence_number(evidence, by_id[newer.chunk_id])
        n_old = _evidence_number(evidence, by_id[older.chunk_id])
        citations += [_citation(evidence, n_new), _citation(evidence, n_old)]
        notes.append(
            f"Note: {older.source_name} ({_date_text(older.version)}) reports "
            f"{', '.join(older.values)}; the newer {newer.source_name} "
            f"({_date_text(newer.version)}) supersedes it. {_markers([n_new, n_old])}"
        )
    answer = generated.answer + "\n\n" + "\n".join(dict.fromkeys(notes))
    return (
        GeneratedAnswer(
            answer=answer,
            raw_answer=generated.raw_answer,
            citations=tuple(dict.fromkeys(citations)),
            evidence=tuple(evidence),
            abstained=False,
            abstention_reason=generated.abstention_reason,
            dropped_markers=generated.dropped_markers,
            fabricated_page_mentions=generated.fabricated_page_mentions,
            backend=generated.backend,
            latency_s=generated.latency_s,
            metadata={**generated.metadata, "evidence_count": len(evidence)},
        ),
        POLICY_NEWER,
    )
