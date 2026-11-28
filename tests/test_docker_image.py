"""Smoke-test a locally built image with RAG_TEST_DOCKER_IMAGE set."""

import os
import subprocess
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

IMAGE = os.environ.get("RAG_TEST_DOCKER_IMAGE")


def docker(*args):
    result = subprocess.run(
        ["docker", *args], check=True, capture_output=True, encoding="utf-8", errors="replace", timeout=45,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not IMAGE, reason="set RAG_TEST_DOCKER_IMAGE to a locally built app image")
def test_image_runs_nonroot_with_liveness_and_dependency_checks():
    docker("info", "--format", "{{.ServerVersion}}")
    name = f"rag-image-smoke-{uuid4().hex}"
    start = datetime.now(timezone.utc)
    docker(
        "run", "--detach", "--rm", "--name", name, "--health-interval", "1s", "--health-start-interval", "1s",
        "--env", "DATABASE_URL=postgresql+psycopg://unused@127.0.0.1:1/unused",
        "--env", "REDIS_URL=redis://127.0.0.1:1/0", "--env", "RAG_CODE_VERSION=image-smoke",
        "--env", "RAG_OFFLINE=1", IMAGE,
    )
    try:
        healthy = docker(
            "events", "--since", start.isoformat(),
            "--until", (datetime.now(timezone.utc) + timedelta(seconds=15)).isoformat(),
            "--filter", f"container={name}", "--filter", "event=health_status: healthy", "--format", "{{.Action}}",
        )
        assert healthy == "health_status: healthy", docker("logs", name)
        assert docker("exec", name, "id", "-u") == "10001"
        assert "eng" in docker("exec", name, "tesseract", "--list-langs").splitlines()
        assert docker("exec", name, "alembic", "heads").endswith("(head)")
        docker("exec", name, "python", "-c", """
import json
import os
from importlib.util import find_spec
from shutil import which
from urllib.error import HTTPError
from urllib.request import urlopen

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

assert os.access(os.environ['HF_HOME'], os.W_OK)
assert not os.access('/opt/venv', os.W_OK)
assert find_spec('pymupdf') is None
assert find_spec('bitsandbytes') is None
assert which('uv') is None
assert torch.version.cuda is None
assert all(model.from_pretrained for model in (AutoModel, AutoModelForSequenceClassification, AutoTokenizer))
with urlopen('http://127.0.0.1:8000/assets/app.js', timeout=3) as response:
    assert b'readEvents' in response.read()
try:
    urlopen('http://127.0.0.1:8000/ready', timeout=7)
except HTTPError as exc:
    with exc:
        assert exc.code == 503
        assert json.load(exc)['checks'] == {'database': False, 'redis': False, 'llm': True}
else:
    raise AssertionError('Readiness unexpectedly passed without services')
""")
    finally:
        docker("stop", name)
