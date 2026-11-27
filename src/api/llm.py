"""Shared client for an OpenAI-compatible inference server."""

from threading import BoundedSemaphore
from time import sleep

from openai import APIConnectionError, APIStatusError, OpenAI

from ..config import GenerationConfig
from ..deadlines import remaining_seconds, run_bounded
from ..errors import GenerationError


class RemoteBackend:
    name = "openai-compatible"

    def __init__(self, config: GenerationConfig, base_url: str, key: str) -> None:
        self.config = config
        self.client = OpenAI(base_url=base_url, api_key=key or "local", max_retries=0)
        self._slots = BoundedSemaphore(4)

    def ready(self) -> bool:
        models = self.client.with_options(timeout=3).models.list()
        return any(model.id == self.config.model_id for model in models.data)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return run_bounded(
            "Model call", self.config.call_timeout_s, self._slots,
            lambda: self._generate(system_prompt, user_prompt),
        )

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.with_options(timeout=min(20, remaining_seconds())).chat.completions.create(
                    model=self.config.model_id,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=self.config.temperature, max_tokens=self.config.max_new_tokens,
                )
                if not response.choices or not response.choices[0].message.content:
                    raise GenerationError("The model returned no answer.")
                return response.choices[0].message.content
            except (APIConnectionError, APIStatusError) as exc:
                retryable = isinstance(exc, APIConnectionError) or exc.status_code in {408, 429} or exc.status_code >= 500
                if attempt == 2 or not retryable:
                    raise GenerationError("The model service could not answer.") from exc
                sleep(min(0.25 * 2**attempt, remaining_seconds()))
        raise AssertionError("unreachable")
