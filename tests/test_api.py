"""Integration tests against disposable PostgreSQL and Redis instances."""

import asyncio
import json
import os
import socket
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL")
REDIS_URL = os.environ.get("RAG_TEST_REDIS_URL")
if not DATABASE_URL or not REDIS_URL:
    pytest.skip("set both RAG_TEST_DATABASE_URL and RAG_TEST_REDIS_URL", allow_module_level=True)

import httpx
from arq.worker import Worker
from arq.connections import RedisSettings
from fastapi.testclient import TestClient
from openai import OpenAI
from sqlalchemy import select

from src.api.app import create_app
from src.api.auth import SESSION_COOKIE
from src.api.jobs import QUEUE, deserialize_job, serialize_job
from src.api.llm import RemoteBackend
from src.api.settings import Settings
from src.api import worker as ingestion_worker
from src.api.worker import ingest, shutdown, startup
from src.config import GenerationConfig
from src.generation.prompts import ABSTENTION_SENTENCE
from src.storage.schema import users

from . import pdf_fixtures
from .test_postgres_library import engine

PASSWORD = "a strong test password"
PDF = pdf_fixtures.structured_pdf()
QUESTION = "What recall did the hybrid system achieve?"


@pytest.fixture
def settings(engine, monkeypatch):
    db_url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("RAG_CODE_VERSION", "api-tests")
    monkeypatch.setenv("RAG_OFFLINE", "1")
    return Settings(db_url, REDIS_URL, "api-tests", offline=True, daily_questions=1)


def client(settings, engine=None):
    return TestClient(create_app(settings, engine=engine), base_url="https://testserver",
                      client=(f"2001:db8::{uuid4().hex[:4]}", 50000))


def sign_in(browser):
    email = f"{uuid4().hex}@example.test"
    credentials = {"email": email, "password": PASSWORD}
    assert browser.post("/auth/register", json=credentials).status_code == 201
    result = browser.post("/auth/login", json=credentials)
    assert result.status_code == 200, result.text
    assert result.json()["role"] == "user"
    cookies = result.headers.get_list("set-cookie")
    session = next(cookie for cookie in cookies if cookie.startswith(SESSION_COOKIE))
    assert all(attribute in session.lower() for attribute in ("httponly", "secure", "samesite=lax", "path=/"))
    return {"X-CSRF-Token": result.json()["csrf_token"]}


def upload(browser, csrf, **params):
    return browser.post("/documents", params={"filename": "report.pdf", **params}, content=PDF,
                        headers={**csrf, "Content-Type": "application/pdf"})


async def finish_jobs():
    worker = Worker(
        [ingest], redis_settings=RedisSettings.from_dsn(REDIS_URL), queue_name=QUEUE, burst=True,
        handle_signals=False, on_startup=startup, on_shutdown=shutdown, max_jobs=1,
        keep_result=0, job_serializer=serialize_job, job_deserializer=deserialize_job,
    )
    try:
        await worker.async_run()
    finally:
        # arq's signal-based close path uses SIGUSR1, which Windows lacks.
        await shutdown(worker.ctx)
        await worker.pool.aclose(close_connection_pool=True)


def test_upload_answer_history_and_cross_user_denials(settings, engine):
    with client(settings, engine) as alice, client(settings, engine) as bob:
        alice_token, bob_token = sign_in(alice), sign_in(bob)
        first = upload(alice, alice_token, version="2025-02 approved")
        assert first.status_code == 202, first.text
        job = first.json()["job_id"]
        assert upload(alice, alice_token, version="2025-02 approved").json()["job_id"] == job
        assert alice.get(f"/jobs/{job}").json()["status"] == "queued"
        assert bob.get(f"/jobs/{job}").status_code == 404
        asyncio.run(finish_jobs())

        completed = alice.get(f"/jobs/{job}").json()
        assert completed["status"] == "succeeded", completed
        document, version = completed["document_id"], completed["result_version_id"]
        assert alice.get("/documents").json()[0]["id"] == document
        assert alice.get(f"/documents/{document}/versions").json()[0]["version"]["date"] == "2025-02"
        pdf_url = f"/documents/{document}/versions/{version}/pdf"
        assert alice.get(pdf_url).content == PDF
        assert bob.get("/documents").json() == []
        assert bob.get(pdf_url).status_code == 404
        assert bob.delete(f"/documents/{document}", headers=bob_token).status_code == 404
        assert upload(bob, bob_token, document_id=document).status_code == 404

        payload = {"question": QUESTION, "version_ids": [version]}
        denied = bob.post("/ask", headers=bob_token, json=payload)
        assert denied.status_code == 404
        response = alice.post("/ask", headers=alice_token, json=payload)
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        assert response.text.startswith("event: start")
        answers = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        final = answers[-1]
        assert final["citations"] and not final["abstained"]
        assert all(citation["document_id"] == version and citation["pages"] for citation in final["citations"])
        chat = final["chat_id"]
        assert alice.get("/chats").json()[0]["id"] == chat
        history = alice.get(f"/chats/{chat}/messages").json()
        assert [message["role"] for message in history] == ["user", "assistant"]
        assert history[-1]["citations"] == final["citations"]
        assert bob.get(f"/chats/{chat}/messages").status_code == 404
        limited = alice.post("/ask", headers=alice_token, json=payload)
        assert limited.status_code == 429 and limited.json()["error"]["code"] == "daily_quota"
        assert alice.delete(f"/documents/{document}", headers=alice_token).status_code == 204
        assert alice.get(pdf_url).status_code == 404
        bad = alice.post("/documents?filename=broken.pdf", headers={**alice_token, "Content-Type": "application/pdf"},
                         content=b"%PDF-1.7\nbroken")
        assert bad.status_code == 202
        asyncio.run(finish_jobs())
        failed = alice.get(f"/jobs/{bad.json()['job_id']}").json()
        assert failed["status"] == "failed" and failed["error_code"] == "invalid_pdf"


def test_csrf_origin_password_hash_and_session_revocation(settings, engine):
    settings = replace(settings, cors_origins=("https://frontend.example",))
    with client(settings, engine) as browser:
        assert browser.get("/auth/me").status_code == 401
        denied = browser.post("/auth/login", headers={"Origin": "https://foreign.example"}, json={})
        assert denied.status_code == 403
        csrf = sign_in(browser)
        user = browser.get("/auth/me").json()
        with engine.connect() as conn:
            stored = conn.scalar(select(users.c.password_hash).where(users.c.id == user["user_id"]))
        assert stored.startswith("$argon2id$")
        assert "Invalid email or password." in browser.post(
            "/auth/login", json={"email": f"{uuid4().hex}@example.test", "password": PASSWORD},
        ).text
        assert browser.post("/auth/logout").status_code == 403
        assert browser.post("/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
        original = browser.cookies.get(SESSION_COOKIE)
        assert browser.post("/auth/logout", headers=csrf).status_code == 204
        assert browser.get("/auth/me", headers={"Cookie": f"{SESSION_COOKIE}={original}"}).status_code == 401

        preflight = browser.options("/auth/login", headers={
            "Origin": "https://frontend.example", "Access-Control-Request-Method": "POST",
        })
        assert preflight.headers["access-control-allow-origin"] == "https://frontend.example"
        rejected = browser.options("/auth/login", headers={
            "Origin": "https://foreign.example", "Access-Control-Request-Method": "POST",
        })
        assert rejected.status_code == 400 and rejected.json()["error"]["code"] == "cors_denied"
        assert browser.get("/health").status_code == 200
        assert browser.get("/ready").json()["checks"] == {"database": True, "redis": True, "llm": True}
        assert "rag_http_requests_total" in browser.get("/metrics").text


def test_upload_quotas_types_and_question_length(settings, engine):
    with client(replace(settings, storage_bytes=len(PDF) - 1, max_question_chars=10), engine) as browser:
        csrf = sign_in(browser)
        assert upload(browser, csrf).json()["error"]["code"] == "storage_quota"
        assert browser.post("/documents?filename=junk.pdf", headers=csrf, content=PDF).status_code == 415
        invalid = browser.post("/documents?filename=junk.pdf", content=b"not a PDF",
                               headers={**csrf, "Content-Type": "application/pdf"})
        assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "invalid_pdf"
        question = {"question": QUESTION, "version_ids": [str(uuid4())]}
        result = browser.post("/ask", headers=csrf, json=question)
        assert result.status_code == 422 and result.json()["error"]["code"] == "invalid_question"


def test_login_rate_limit_counts_failed_attempts(settings, engine):
    with client(settings, engine) as browser:
        credentials = {"email": f"{uuid4().hex}@example.test", "password": PASSWORD}
        for _ in range(8):
            assert browser.post("/auth/login", json=credentials).status_code == 401
        blocked = browser.post("/auth/login", json=credentials)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "rate_limited"


def test_dead_database_has_clean_errors(settings):
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        url = f"postgresql+psycopg://test:unused@127.0.0.1:{unavailable.getsockname()[1]}/postgres"
        with client(replace(settings, database_url=url)) as browser:
            assert browser.get("/health").status_code == 200
            ready = browser.get("/ready")
            assert ready.status_code == 503 and ready.json()["checks"]["database"] is False
            response = browser.post("/auth/login", json={"email": f"{uuid4().hex}@example.test", "password": PASSWORD})
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["code"] == "storage_unavailable"
            assert error["request_id"] == response.headers["X-Request-ID"]
            assert "127.0.0.1" not in response.text


def test_remote_backend_retries_transient_error():
    calls = []

    def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local-model", "object": "model"}]})
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 1, "model": "local-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
        })

    backend = RemoteBackend(GenerationConfig(model_id="local-model", call_timeout_s=5), "http://model/v1", "local")
    backend.client.close()
    with OpenAI(base_url="http://model/v1", api_key="local", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as mock:
        backend.client = mock
        assert backend.ready()
        assert backend.generate("system", "prompt") == "answer"
    assert len(calls) == 2 and calls[-1]["messages"][0]["content"] == "system"


def test_worker_terminates_a_job_that_exceeds_its_deadline(monkeypatch):
    owner, job = uuid4(), uuid4()
    failed, stopped = [], []

    async def scenario():
        process = SimpleNamespace(wait=asyncio.Event().wait)

        async def spawn(*args, **kwargs):
            return process

        async def stop(child):
            stopped.append(child)

        monkeypatch.setattr(ingestion_worker.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(ingestion_worker, "_stop", stop)
        uploads = SimpleNamespace(
            claim=lambda *_: True, fail=lambda *args: failed.append(args), job_timeout_s=0.01,
        )
        await ingest({"uploads": uploads}, str(owner), str(job))
        assert stopped == [process]

    asyncio.run(scenario())
    assert failed == [(owner, job, "job_timeout")]


def test_web_pages_use_saved_passages_and_escape_stored_content(settings, engine):
    with client(settings, engine) as browser, client(settings, engine) as other:
        login = browser.get("/")
        assert str(login.url).endswith("/login")
        assert "script-src 'self'" in login.headers["content-security-policy"]
        assert browser.get("/assets/htmx-2.0.11.min.js").status_code == 200
        assert browser.get("/register").status_code == 200
        csrf = sign_in(browser)
        sign_in(other)
        owner = UUID(browser.get("/auth/me").json()["user_id"])
        dangerous = "<img src=x onerror=alert(1)>"
        version = browser.app.state.library.ingest(owner, PDF, dangerous + ".pdf")
        library = browser.get("/library")
        assert library.status_code == 200 and "Upload progress" in library.text
        assert "&lt;img src=x onerror=alert(1)&gt;.pdf" in library.text
        assert dangerous not in library.text and dangerous not in other.get("/library").text
        assert browser.get("/fragments/uploads", headers={"HX-Request": "true"}).status_code == 200
        response = browser.post("/ask", headers=csrf, json={"question": QUESTION, "version_ids": [str(version.id)]})
        answer = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
        citation = answer["citations"][0]
        page = citation["pages"][0]
        url = f"/citations/{version.id}/{citation['chunk_id']}/pages/{page}"
        viewer = browser.get(url)
        assert viewer.status_code == 200
        assert "with the cited text highlighted" in viewer.text
        assert "data:image/png;base64," in viewer.text and f"#page={page}" in viewer.text
        assert other.get(url).status_code == 404
        assert browser.get(url.rsplit("/", 1)[0] + "/999").status_code == 404
        assert browser.get(f"/citations/{version.id}/foreign/pages/{page}").status_code == 404

        chat_id = UUID(answer["chat_id"])
        browser.app.state.history.save(owner, chat_id, "Another question", dangerous, [], new=False)
        saved = browser.get(f"/chat?chat_id={chat_id}")
        assert "&lt;img src=x onerror=alert(1)&gt;" in saved.text and dangerous not in saved.text
        assert f"Page {page}</a>" in saved.text
        assert other.get(f"/chat?chat_id={chat_id}").status_code == 404


def test_configured_reranker_floor_applies_to_an_existing_index(settings, engine, monkeypatch):
    with client(settings, engine) as browser:
        csrf = sign_in(browser)
        owner = UUID(browser.get("/auth/me").json()["user_id"])
        version = browser.app.state.library.ingest(owner, PDF, "report.pdf")

        monkeypatch.setenv("RAG_RERANKER_MIN_SCORE", "2")
        with client(Settings.from_env(), engine) as strict:
            strict.cookies.update(browser.cookies)
            response = strict.post("/ask", headers=csrf, json={"question": QUESTION, "version_ids": [str(version.id)]})
            assert response.status_code == 200 and "event: answer" in response.text
            reply = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
            assert reply["abstained"] and reply["answer"] == ABSTENTION_SENTENCE
            assert reply["citations"] == []
