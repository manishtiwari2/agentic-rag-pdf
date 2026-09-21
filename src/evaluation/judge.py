"""The optional LLM judge (EVALUATION_PROTOCOL.md sections 19, 20; DD-027).

Optional, and off by default. Everything the Phase 3 exit criterion asks for is
computed without it, which is deliberate: EVALUATION_PROTOCOL.md section 19.1
says the only judge that fits the memory budget is the generator itself, so a
judged number is a model grading its own work. A harness whose headline figures
depended on that would be reporting a bias as a result.

What the judge is therefore for is the narrow thing regular expressions cannot
do (section 19.2): whether a paraphrase means the same as the reference, and
whether a claim is supported by a passage that does not restate it. It never
decides abstention, numeric agreement or retrieval metrics.

Four properties the rest of the harness relies on:

* It **cannot block**. No model, no `transformers`, a refusal to load, an
  unparseable reply -- each records a skip with its reason and the run
  continues.
* A **skip is not a zero**. Judged fields are ``None`` with a ``judge_skipped``
  reason beside them, so no aggregate can average a missing judgment as 0.
* The **prompt is fixed** and stored with the results, for the reason section 19
  gives: a judging prompt that changes between runs makes two runs
  incomparable.
* Every judged metric is reported beside its deterministic counterpart, and the
  **disagreement rate between them is itself reported** (section 19.1's second
  mitigation). A high rate means the judged figures are not to be trusted, and
  the write-up has to say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..benchmark.schema import Question
from ..config import SCRIPTED_BACKEND, GenerationConfig
from ..pipeline import RAGResult

#: Bump when the prompt or the scale below changes. Two runs judged under
#: different prompts are not comparable, and section 19 requires the prompt to
#: be fixed for the experiment.
JUDGE_PROTOCOL_VERSION = "2026-09-21.1"

#: EVALUATION_PROTOCOL.md section 20's scale, stated once so it cannot drift.
JUDGE_SCALE = {0: "incorrect", 1: "partially correct", 2: "correct"}
JUDGE_MAX = 2

JUDGE_SYSTEM_PROMPT = """You are grading one answer produced by a document question-answering system.

You are given the question, the reference answer, the system's answer, the evidence passages the system retrieved, and the citations it produced. Use nothing else. You have no web access and no outside knowledge, and you must not use any.

Grade three things on this scale:
0 = incorrect
1 = partially correct
2 = correct

correctness          Does the system answer agree with the reference answer?
faithfulness         Is every claim in the system answer supported by the evidence passages?
citation_correctness Do the cited passages actually support the claims they are attached to?

Reply with one JSON object and nothing else:
{"correctness": 0, "faithfulness": 0, "citation_correctness": 0, "reason": "one short sentence"}"""

JUDGE_USER_TEMPLATE = """QUESTION
{question}

REFERENCE ANSWER
{reference_answer}

SYSTEM ANSWER
{answer}

RETRIEVED EVIDENCE
{evidence}

CITATIONS
{citations}

JSON:"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeVerdict:
    """One judgment, or the reason there is not one."""

    correctness: int | None = None
    faithfulness: int | None = None
    citation_correctness: int | None = None
    reason: str = ""
    skipped_reason: str = ""

    @property
    def skipped(self) -> bool:
        return self.correctness is None

    def to_record(self) -> dict[str, Any]:
        return {
            "judge_correctness": self.correctness,
            "judge_faithfulness": self.faithfulness,
            "judge_citation_correctness": self.citation_correctness,
            "judge_reason": self.reason,
            "judge_skipped": self.skipped,
            "judge_skipped_reason": self.skipped_reason,
            "judge_scale_max": JUDGE_MAX,
        }


SKIPPED = JudgeVerdict()


class LLMJudge:
    """Grades answers with a local model, and degrades to skipping.

    Construction never loads weights. The backend is built lazily on the first
    judgment and a failure there disables the judge for the rest of the run
    rather than raising once per question.
    """

    def __init__(self, config: GenerationConfig | None = None, backend: Any = None) -> None:
        self.config = config or GenerationConfig()
        self._backend = backend
        self._disabled_reason = ""
        self._tried = backend is not None
        self.judged = 0
        self.skipped = 0
        self.parse_failures = 0
        self.disagreements = 0
        self.comparable = 0

    # -- availability -------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def _ensure_backend(self) -> Any:
        if self._backend is not None or self._disabled_reason:
            return self._backend
        if self._tried:
            return self._backend
        self._tried = True
        if self.config.model_id == SCRIPTED_BACKEND or self.config.backend == "scripted":
            # The scripted backend extracts sentences by term overlap. It cannot
            # judge whether a paraphrase means the same thing, and pretending it
            # can would put a number in the results that measures nothing.
            self._disabled_reason = (
                "the offline scripted backend cannot judge semantic equivalence"
            )
            return None
        try:
            from ..generation.backends import build_backend  # noqa: PLC0415

            self._backend = build_backend(self.config)
        except Exception as exc:  # noqa: BLE001 - a judge must never break a run
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
        return self._backend

    def describe(self) -> dict[str, Any]:
        """What goes in ``config.json`` (section 19: record the judge)."""
        return {
            "enabled": True,
            "model": self.config.model_id,
            "model_revision": None,
            "protocol_version": JUDGE_PROTOCOL_VERSION,
            "scale": {str(k): v for k, v in JUDGE_SCALE.items()},
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "user_template": JUDGE_USER_TEMPLATE,
            "is_the_model_under_test": True,
            "known_bias": (
                "DD-027: the judge is the generator, so judged scores are "
                "biased upward and unevenly. Every judged metric is reported "
                "beside a deterministic one and the disagreement rate is "
                "reported with them."
            ),
        }

    # -- judging ------------------------------------------------------------

    def judge(self, question: Question, result: RAGResult) -> JudgeVerdict:
        backend = self._ensure_backend()
        if backend is None:
            self.skipped += 1
            return JudgeVerdict(skipped_reason=self._disabled_reason or "no judge model")

        prompt = JUDGE_USER_TEMPLATE.format(
            question=question.question,
            reference_answer=question.answer if question.answer else "(none: this question is unanswerable from the document)",
            answer=result.answer,
            evidence=result.evidence_text or "(none)",
            citations=", ".join(
                f"[{c.marker}] {c.label}" for c in result.citations
            )
            or "(none)",
        )
        try:
            raw = backend.generate(JUDGE_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001
            self.skipped += 1
            return JudgeVerdict(skipped_reason=f"{type(exc).__name__}: {exc}")

        verdict = parse_judge_reply(raw)
        if verdict.skipped:
            # EXPERIMENT_PLAN.md section 4 names unreliable structured output
            # from a small model as a likely risk and asks for the parse failure
            # rate to be measured rather than worked around.
            self.parse_failures += 1
            self.skipped += 1
        else:
            self.judged += 1
        return verdict

    def annotate(
        self, question: Question, result: RAGResult, record: dict[str, Any]
    ) -> dict[str, Any]:
        """Add the judgment to a per-question record, in place."""
        verdict = self.judge(question, result)
        metrics = record.setdefault("metrics", {})
        metrics.update(verdict.to_record())

        deterministic = ((metrics.get("answer") or {}).get("correct"))
        if verdict.correctness is not None and deterministic is not None:
            judged_correct = verdict.correctness >= JUDGE_MAX
            self.comparable += 1
            if judged_correct != (deterministic >= 1.0):
                self.disagreements += 1
            metrics["judge_agrees_with_deterministic"] = judged_correct == (
                deterministic >= 1.0
            )
        else:
            metrics["judge_agrees_with_deterministic"] = None
        return record

    def summary(self) -> dict[str, Any]:
        """The figures section 19.1 requires beside the judged metrics."""
        attempted = self.judged + self.skipped
        return {
            "judge_enabled": True,
            "judge_model": self.config.model_id,
            "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
            "judge_questions_judged": self.judged,
            "judge_questions_skipped": self.skipped,
            "judge_skipped_reason": self._disabled_reason,
            "judge_parse_failure_rate": (
                self.parse_failures / attempted if attempted else None
            ),
            "judge_deterministic_disagreement_rate": (
                self.disagreements / self.comparable if self.comparable else None
            ),
            "judge_comparable_questions": self.comparable,
        }


def parse_judge_reply(raw: str) -> JudgeVerdict:
    """Read the JSON object out of a small model's reply.

    Tolerant about what surrounds the object -- a 1.5B model will wrap it in
    prose -- and strict about what is inside it. A score outside the scale is a
    parse failure, not something to clamp: silently rounding 7 down to 2 would
    turn an unreliable judge into a confident-looking one.
    """
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return JudgeVerdict(skipped_reason="no JSON object in the judge reply")
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return JudgeVerdict(skipped_reason="the judge reply was not valid JSON")
    if not isinstance(payload, dict):
        return JudgeVerdict(skipped_reason="the judge reply was not a JSON object")

    scores: dict[str, int] = {}
    for field_name in ("correctness", "faithfulness", "citation_correctness"):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return JudgeVerdict(skipped_reason=f"{field_name} missing or not a number")
        if int(value) != value or not 0 <= int(value) <= JUDGE_MAX:
            return JudgeVerdict(
                skipped_reason=f"{field_name}={value!r} is outside the 0-{JUDGE_MAX} scale"
            )
        scores[field_name] = int(value)

    reason = payload.get("reason")
    return JudgeVerdict(
        correctness=scores["correctness"],
        faithfulness=scores["faithfulness"],
        citation_correctness=scores["citation_correctness"],
        reason=str(reason) if isinstance(reason, str) else "",
    )
