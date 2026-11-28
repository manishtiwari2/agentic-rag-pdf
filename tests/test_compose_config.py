"""Validate the deployment modes without starting Docker services."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def compose_config(tmp_path):
    docker = shutil.which("docker")
    if not docker or subprocess.run([docker, "compose", "version"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker Compose is unavailable")
    env_file = tmp_path / "compose.env"
    env_file.touch()
    env = {
        **os.environ, "COMPOSE_PROFILES": "", "POSTGRES_PASSWORD": "admin@:/test",
        "RAG_DB_PASSWORD": "app/:@test", "RAG_CODE_VERSION": "compose-tests", "RAG_OFFLINE": "0", "HF_TOKEN": "",
        "API_HOST": "127.0.0.1", "API_PORT": "8000", "LLM_MODEL": "test-model", "VLLM_MODEL": "org/test-model",
        "RAG_DB_NAME": "rag_test", "VLLM_MODEL_REVISION": "0" * 40,
    }

    def config(*options, omit=None):
        variables = env if omit is None else {key: value for key, value in env.items() if key != omit}
        result = subprocess.run(
            [docker, "compose", "--env-file", str(env_file), "-f", "docker-compose.yml",
             *options, "config", "--format", "json"],
            cwd=ROOT, env=variables, capture_output=True, encoding="utf-8", timeout=20,
        )
        if omit is not None:
            assert result.returncode != 0
            return result.stderr
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["services"]

    return config


def test_default_profile_uses_ollama_and_a_restricted_runtime_login(compose_config):
    services = compose_config()
    assert {"api", "worker", "migrate", "postgres", "redis", "ollama", "ollama-model"} <= services.keys()
    assert [name for name, service in services.items() if service.get("ports")] == ["api"]
    assert services["api"]["ports"][0]["host_ip"] == "127.0.0.1"
    for name in ("api", "worker"):
        environment = services[name]["environment"]
        assert environment["DATABASE_URL"] == "postgresql+psycopg://rag_app@postgres:5432/rag_test"
        assert environment["PGPASSWORD"] == "app/:@test"
        assert "POSTGRES_PASSWORD" not in environment
        assert services[name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert services["migrate"]["environment"]["PGPASSWORD"] == "admin@:/test"
    assert services["postgres"]["environment"]["POSTGRES_DB"] == "rag_test"
    assert services["api"]["depends_on"]["prefetch-models"]["condition"] == "service_completed_successfully"
    assert services["api"]["environment"]["LLM_MODEL"] == services["ollama-model"]["command"][0]
    assert services["api"]["depends_on"]["ollama-model"]["condition"] == "service_completed_successfully"
    assert services["ollama"]["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 1


def test_cpu_override_runs_offline_without_model_servers(compose_config):
    services = compose_config("-f", "docker-compose.cpu.yml")
    assert {"ollama", "ollama-model", "vllm"}.isdisjoint(services)
    assert all(services[name]["environment"]["RAG_OFFLINE"] == "1" for name in ("api", "worker", "prefetch-models"))
    assert set(services["api"]["depends_on"]) == {"migrate", "redis", "prefetch-models"}


def test_vllm_profile_replaces_ollama_with_the_same_served_alias(compose_config):
    services = compose_config("-f", "docker-compose.vllm.yml", "--profile", "vllm")
    assert {"ollama", "ollama-model"}.isdisjoint(services)
    assert services["api"]["depends_on"]["vllm"]["condition"] == "service_healthy"
    model_args = services["vllm"]["command"]
    assert model_args[model_args.index("--model") + 1] == "org/test-model"
    assert model_args[model_args.index("--revision") + 1] == model_args[model_args.index("--tokenizer-revision") + 1]
    for name in ("api", "worker"):
        environment = services[name]["environment"]
        assert environment["LLM_BASE_URL"] == "http://vllm:8000/v1"
        assert environment["LLM_MODEL"] == model_args[model_args.index("--served-model-name") + 1]
        assert environment["LLM_MODEL_REVISION"] == model_args[model_args.index("--revision") + 1]


def test_compose_requires_an_app_password(compose_config):
    assert "Set RAG_DB_PASSWORD for the app role" in compose_config(omit="RAG_DB_PASSWORD")
