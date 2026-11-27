"""Environment settings shared by API processes and ingestion jobs."""

import os
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from ..config import RAGConfig
from ..errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    code_version: str
    offline: bool = False
    llm_base_url: str = "http://ollama:11434/v1"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_api_key: str = ""
    cors_origins: tuple[str, ...] = ()
    session_seconds: int = 12 * 60 * 60
    storage_bytes: int = 256 * 1024**2
    daily_questions: int = 100
    max_question_chars: int = 2000
    job_timeout_s: int = 240
    job_memory_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        for origin in self.cors_origins:
            url = urlsplit(origin)
            if url.scheme not in {"https", "http"} or not url.netloc or origin != f"{url.scheme}://{url.netloc}":
                raise ConfigurationError("CORS_ORIGINS must contain explicit origins without paths.")
        for name in ("session_seconds", "storage_bytes", "daily_questions", "max_question_chars",
                     "job_timeout_s", "job_memory_bytes"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name.upper()} must be positive.")

    @classmethod
    def from_env(cls) -> "Settings":
        required = {name: os.environ.get(name, "") for name in ("DATABASE_URL", "REDIS_URL", "RAG_CODE_VERSION")}
        if not all(value.strip() for value in required.values()):
            raise ConfigurationError("Set DATABASE_URL, REDIS_URL, and RAG_CODE_VERSION.")
        optional = {}
        for name in ("session_seconds", "storage_bytes", "daily_questions", "max_question_chars",
                     "job_timeout_s", "job_memory_bytes"):
            if name.upper() in os.environ:
                try:
                    optional[name] = int(os.environ[name.upper()])
                except ValueError as exc:
                    raise ConfigurationError(f"{name.upper()} must be an integer.") from exc
        return cls(
            database_url=required["DATABASE_URL"], redis_url=required["REDIS_URL"],
            code_version=required["RAG_CODE_VERSION"], offline=os.environ.get("RAG_OFFLINE") == "1",
            llm_base_url=os.environ.get("LLM_BASE_URL", cls.llm_base_url),
            llm_model=os.environ.get("LLM_MODEL", cls.llm_model),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            cors_origins=tuple(v.strip() for v in os.environ.get("CORS_ORIGINS", "").split(",") if v.strip()),
            **optional,
        )

    def rag_config(self) -> RAGConfig:
        config = RAGConfig.offline() if self.offline else RAGConfig.default()
        return replace(
            config, retrieval=replace(config.retrieval, strategy="agentic"),
            generation=config.generation if self.offline else replace(config.generation, model_id=self.llm_model),
            conflicts=replace(config.conflicts, enabled=True),
        )
