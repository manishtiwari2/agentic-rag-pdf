"""Generation backends.

``HuggingFaceBackend`` runs an open-weight instruct model locally. Greedy
decoding is the default and ``temperature == 0`` means ``do_sample=False`` --
not a very small temperature, which still samples and still makes a benchmark
irreproducible.

``ScriptedBackend`` needs no weights at all. It is a real extractive QA system,
weak but honest: it scores each evidence block against the question by
IDF-weighted term overlap, returns the best-matching sentences, and emits the
canonical refusal when nothing matches well enough. It reads the same rendered
prompt the transformer backend receives rather than getting a private channel to
the evidence, so exercising it exercises the real prompt format.

It exists so that every deterministic part of the pipeline -- retrieval,
evidence numbering, citation resolution, abstention, the end-to-end path -- can
be tested on CPU without downloading four gigabytes of weights.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from ..config import GenerationConfig
from ..errors import GenerationError, ModelLoadError
from .prompts import ABSTENTION_SENTENCE, parse_user_prompt


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Dependency-free backend
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Below this share of the question's weighted terms, the scripted backend
#: refuses. Tuned to be cautious rather than accurate: this backend's job is to
#: exercise the pipeline, and a weak reader that answers anyway would make the
#: abstention path untestable.
DEFAULT_MATCH_THRESHOLD = 0.22

#: Question words carry no retrieval signal and would dominate a short question.
_STOPWORDS = frozenset(
    """a an and are as at be been by do does for from has have how i in is it its of on
    or that the this to was were what when where which who why will with does did can
    could should would about into their there these those they you your""".split()
)


class ScriptedBackend:
    """Extractive answering by weighted term overlap. No model, no downloads."""

    name = "scripted"

    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.match_threshold = match_threshold

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        question, blocks = parse_user_prompt(user_prompt)
        if not blocks:
            return ABSTENTION_SENTENCE

        query_terms = _content_terms(question)
        if not query_terms:
            return ABSTENTION_SENTENCE

        idf = _idf([text for _, text in blocks])
        scored = [
            (_coverage(query_terms, _content_terms(text), idf), number, text)
            for number, text in blocks
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_number, best_text = scored[0]

        if best_score < self.match_threshold:
            return ABSTENTION_SENTENCE

        sentences = _best_sentences(query_terms, best_text, idf)
        if not sentences:
            return ABSTENTION_SENTENCE
        return f"{' '.join(sentences)} [C{best_number}]"


#: Suffixes stripped to match word forms. Crude on purpose: enough that
#: "embedded", "embedding" and "embeds" compare equal, and short enough that it
#: cannot be mistaken for real morphology.
_SUFFIXES = ("ations", "ation", "ingly", "ings", "ing", "edly", "ed", "es", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            stem = token[: -len(suffix)]
            # "embedd" -> "embed": undo the doubled consonant English adds
            # before -ed and -ing.
            if len(stem) > 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem
    return token


def _content_terms(text: str) -> list[str]:
    return [
        _stem(token)
        for token in (t.lower() for t in _TOKEN.findall(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


def _idf(documents: list[str]) -> dict[str, float]:
    total = len(documents)
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(set(_content_terms(document)))
    return {
        term: math.log((total + 1) / (count + 1)) + 1.0
        for term, count in frequency.items()
    }


def _coverage(
    query_terms: list[str], candidate_terms: list[str], idf: dict[str, float]
) -> float:
    """Share of the question's weighted terms present in the candidate."""
    present = set(candidate_terms)
    total = sum(idf.get(term, 1.0) for term in set(query_terms))
    if total == 0:
        return 0.0
    matched = sum(idf.get(term, 1.0) for term in set(query_terms) if term in present)
    return matched / total


def _best_sentences(
    query_terms: list[str], text: str, idf: dict[str, float], limit: int = 2
) -> list[str]:
    """The sentences of ``text`` that best cover the question, in reading order."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    scored = [
        (_coverage(query_terms, _content_terms(sentence), idf), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [item for item in scored[:limit] if item[0] > 0]
    if not chosen:
        return []
    chosen.sort(key=lambda item: item[1])
    return [sentence for _, _, sentence in chosen]


# ---------------------------------------------------------------------------
# Model-backed backend
# ---------------------------------------------------------------------------


class HuggingFaceBackend:
    """A local instruct model via ``transformers``."""

    name = "transformers"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                f"Generation model {self.config.model_id!r} needs `transformers` "
                "and `torch`, which are not installed. Install them, or use "
                "RAGConfig.offline() to run the pipeline without model weights."
            ) from exc

        kwargs: dict[str, object] = {}
        if self.config.quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415

                if self.config.quantization == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(
                    f"{self.config.quantization} quantization needs "
                    "`bitsandbytes`. Install it, or set "
                    "generation.quantization='none' -- note that an unquantized "
                    "4B model does not leave enough headroom on a free-tier T4 "
                    "(MODEL_SELECTION.md 9.1)."
                ) from exc
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        if self.config.device == "auto":
            kwargs["device_map"] = "auto"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, **kwargs
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                f"Could not load {self.config.model_id!r}: {exc}. If this is an "
                "out-of-memory failure, try RAGConfig.low_memory(), or "
                "generation.quantization='4bit'. If the weights could not be "
                "downloaded, check the model id and network access."
            ) from exc

        if self.config.device != "auto":
            self._model.to(self.config.device)
        self._model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(  # type: ignore[misc]
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover - model without a chat template
            text = f"{system_prompt}\n\n{user_prompt}"

        inputs = self._tokenizer(text, return_tensors="pt").to(  # type: ignore[misc]
            self._model.device  # type: ignore[union-attr]
        )

        greedy = self.config.temperature == 0
        kwargs: dict[str, object] = {
            "max_new_tokens": self.config.max_new_tokens,
            # Greedy decoding: temperature 0 means no sampling at all, not a
            # temperature close to zero, which would still be stochastic.
            "do_sample": not greedy,
            "pad_token_id": self._tokenizer.pad_token_id  # type: ignore[union-attr]
            or self._tokenizer.eos_token_id,  # type: ignore[union-attr]
        }
        if not greedy:
            kwargs["temperature"] = self.config.temperature

        try:
            with torch.no_grad():
                output = self._model.generate(**inputs, **kwargs)  # type: ignore[misc]
        except Exception as exc:
            raise GenerationError(
                f"Generation failed: {exc}. On a CUDA out-of-memory error, "
                "reduce generation.max_new_tokens or retrieval.top_k, or switch "
                "to RAGConfig.low_memory()."
            ) from exc

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(  # type: ignore[union-attr]
            generated, skip_special_tokens=True
        ).strip()


def build_backend(config: GenerationConfig) -> LLMBackend:
    """Construct the backend named by the configuration (DD-017)."""
    from ..config import SCRIPTED_BACKEND  # noqa: PLC0415

    if config.backend == "scripted" or config.model_id == SCRIPTED_BACKEND:
        return ScriptedBackend()
    return HuggingFaceBackend(config)
