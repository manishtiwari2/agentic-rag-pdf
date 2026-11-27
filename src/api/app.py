"""FastAPI entry point. Run with uvicorn src.api.app:create_app --factory."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from arq.connections import ArqRedis
from fastapi import Depends, FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError
from sqlalchemy import select
from starlette.exceptions import HTTPException

from ..request_log import configure_logging, request_id, request_scope
from ..reranking.reranker import build_reranker
from ..storage.database import build_engine
from ..storage.library import DocumentLibrary
from ..storage.schema import users
from ..web.router import router as web_router, static_dir
from .auth import CSRF_COOKIE, SESSION_COOKIE, Accounts, Identity, email_address
from .errors import ApiError, error_body, public_error
from .history import History
from .jobs import UploadJobs, deserialize_job, enqueue, serialize_job
from .limits import take, today
from .llm import RemoteBackend
from .settings import Settings

logger = logging.getLogger(__name__)


async def _body(request: Request, limit: int) -> bytes:
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > limit:
            raise ApiError(413, "body_too_large", "The request exceeds the size limit.")
        content.extend(chunk)
    return bytes(content)


async def _json(request: Request, limit: int) -> dict:
    if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json":
        raise ApiError(415, "unsupported_media_type", "Send an application/json body.")
    try:
        value = json.loads(await _body(request, limit))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ApiError(422, "invalid_request", "Send a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ApiError(422, "invalid_request", "Send a JSON object.")
    return value


async def _credentials(request: Request) -> tuple[str, str]:
    body = await _json(request, 8192)
    email, password = body.get("email"), body.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        raise ApiError(422, "invalid_request", "Email and password are required.")
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in password):
        raise ApiError(422, "invalid_password", "Use a valid text password.")
    return email_address(email), password


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data), separators=(',', ':'))}\n\n"


def create_app(settings: Settings | None = None, *, engine=None) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = CollectorRegistry()
    requests = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        db = engine if engine is not None else build_engine(settings.database_url)
        redis = ArqRedis(
            pool_or_conn=ConnectionPool.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3),
            job_serializer=serialize_job, job_deserializer=deserialize_job,
        )
        config = settings.rag_config()
        backend = None if settings.offline else RemoteBackend(config.generation, settings.llm_base_url, settings.llm_api_key)
        try:
            reranker = build_reranker(config.reranking) if config.uses_reranker else None
            library = DocumentLibrary(db, config, settings.code_version, backend=backend, reranker=reranker)
            await run_in_threadpool(library.embedder.embed_query, "startup")
            if reranker is not None:
                await run_in_threadpool(reranker.score, "startup", ["startup"])
            app.state.db, app.state.redis, app.state.backend, app.state.library = db, redis, backend, library
            app.state.accounts = Accounts(db, settings.session_seconds)
            app.state.jobs = UploadJobs(library, settings.storage_bytes, settings.job_timeout_s)
            app.state.history = History(db)
            app.state.max_question_chars = settings.max_question_chars
            yield
        finally:
            await redis.aclose(close_connection_pool=True)
            if backend is not None:
                backend.client.close()
            if engine is None:
                db.dispose()

    app = FastAPI(lifespan=lifespan, debug=False)
    app.mount("/assets", StaticFiles(directory=static_dir), name="web_assets")
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-CSRF-Token"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        error = ApiError(422, "invalid_request", "A request field is missing or invalid.")
        return JSONResponse(error_body(error, request_id.get()), status_code=error.status)

    @app.exception_handler(HTTPException)
    async def http_handler(request, exc):
        error = ApiError(exc.status_code, "http_error", "This HTTP request cannot be served.")
        return JSONResponse(error_body(error, request_id.get()), status_code=error.status, headers=exc.headers)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        with request_scope("HTTP request"):
            try:
                origin = request.headers.get("origin")
                if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
                    local = f"{request.url.scheme}://{request.url.netloc}"
                    if origin != local and origin not in settings.cors_origins:
                        raise ApiError(403, "origin_denied", "This request origin is not allowed.")
                response = await call_next(request)
                if response.status_code == 400 and request.method == "OPTIONS" and request.headers.get(
                    "access-control-request-method",
                ):
                    error = ApiError(400, "cors_denied", "This cross-origin request is not allowed.")
                    headers = {key: value for key, value in response.headers.items()
                               if key == "vary" or key.startswith("access-control-")}
                    response = JSONResponse(error_body(error, request_id.get()), status_code=400, headers=headers)
            except Exception as exc:
                error = public_error(exc)
                if error.status >= 500:
                    logger.error("Request failed (%s)", type(exc).__name__)
                response = JSONResponse(error_body(error, request_id.get()), status_code=error.status)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            method = request.method if request.method in {"GET", "HEAD", "POST", "DELETE", "OPTIONS"} else "OTHER"
            requests.labels(method, route, str(response.status_code)).inc()
            response.headers["X-Request-ID"] = request_id.get()
            response.headers.setdefault("Cache-Control", "no-store")
            if response.headers.get("content-type", "").startswith("text/html") and not request.url.path.startswith(("/docs", "/redoc")):
                response.headers["Content-Security-Policy"] = (
                    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                    "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
                )
            return response

    def identity(request: Request) -> Identity:
        return request.app.state.accounts.authenticate(request.cookies.get(SESSION_COOKIE))

    def verified(request: Request, current: Identity = Depends(identity)) -> Identity:
        current.check_csrf(request.headers.get("x-csrf-token"))
        return current

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        state = request.app.state

        def database_ready():
            with state.db.connect() as conn:
                conn.execute(select(users.c.id).limit(0))
                return True

        async def probe(operation):
            try:
                return bool(await asyncio.wait_for(operation, timeout=5))
            except Exception as exc:
                logger.warning("Readiness check failed (%s)", type(exc).__name__)
                return False

        async def llm_ready():
            return True if state.backend is None else await run_in_threadpool(state.backend.ready)

        values = await asyncio.gather(
            probe(run_in_threadpool(database_ready)), probe(state.redis.ping()), probe(llm_ready()),
        )
        checks = dict(zip(("database", "redis", "llm"), values))
        ok = all(values)
        return JSONResponse({"status": "ready" if ok else "unavailable", "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post("/auth/register", status_code=201)
    async def register(request: Request):
        redis = request.app.state.redis
        await take(redis, ("register-ip", _client_ip(request), 10, 300))
        email, password = await _credentials(request)
        await run_in_threadpool(request.app.state.accounts.register, email, password)
        return {"status": "registered"}

    @app.post("/auth/login")
    async def login(request: Request):
        email, password = await _credentials(request)
        state = request.app.state
        await take(state.redis, ("login-ip", _client_ip(request), 30, 300), ("login-user", email, 8, 300))
        session, csrf, role = await run_in_threadpool(state.accounts.login, email, password)
        response = JSONResponse({"role": role, "csrf_token": csrf})
        options = {"path": "/", "secure": True, "samesite": "lax", "max_age": settings.session_seconds}
        response.set_cookie(SESSION_COOKIE, session, httponly=True, **options)
        response.set_cookie(CSRF_COOKIE, csrf, **options)
        return response

    @app.get("/auth/me")
    def me(current: Identity = Depends(identity)):
        return {"user_id": current.user_id, "role": current.role}

    @app.post("/auth/logout", status_code=204)
    def logout(request: Request, current: Identity = Depends(verified)):
        request.app.state.accounts.logout(current)
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, secure=True, httponly=True, samesite="lax")
        response.delete_cookie(CSRF_COOKIE, secure=True, samesite="lax")
        return response

    @app.post("/documents", status_code=202)
    async def upload(
        request: Request, filename: str, document_id: UUID | None = None, version: str | None = None,
        current: Identity = Depends(verified),
    ):
        if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/pdf":
            raise ApiError(415, "unsupported_media_type", "Send an application/pdf body.")
        state = request.app.state
        data = await _body(request, state.library.config.ingestion.max_upload_mb * 1024**2)
        job_id = await run_in_threadpool(state.jobs.submit, current.user_id, data, filename, document_id, version)
        try:
            await enqueue(state.redis, current.user_id, job_id)
        except RedisError:
            # The worker will enqueue the committed upload when Redis is available.
            logger.warning("Upload saved while the Redis queue was unavailable")
        return {"job_id": job_id}

    @app.get("/jobs")
    def list_jobs(request: Request, current: Identity = Depends(identity)):
        return request.app.state.jobs.status(current.user_id)

    @app.get("/jobs/{job_id}")
    def job_status(request: Request, job_id: UUID, current: Identity = Depends(identity)):
        return request.app.state.jobs.status(current.user_id, job_id)

    @app.get("/documents")
    def list_documents(request: Request, current: Identity = Depends(identity)):
        return request.app.state.library.list_documents(current.user_id)

    @app.get("/documents/{document_id}/versions")
    def list_versions(request: Request, document_id: UUID, current: Identity = Depends(identity)):
        versions = request.app.state.library.list_versions(current.user_id, document_id)
        if not versions:
            raise ApiError(404, "not_found", "The document does not exist.")
        return versions

    @app.get("/documents/{document_id}/versions/{version_id}/pdf")
    def read_pdf(request: Request, document_id: UUID, version_id: UUID, current: Identity = Depends(identity)):
        library = request.app.state.library
        if not any(version.id == version_id for version in library.list_versions(current.user_id, document_id)):
            raise ApiError(404, "not_found", "The document version does not exist.")
        return Response(library.read_pdf(current.user_id, version_id), media_type="application/pdf",
                        headers={"X-Content-Type-Options": "nosniff", "Content-Disposition": "inline"})

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(request: Request, document_id: UUID, current: Identity = Depends(verified)):
        request.app.state.library.delete_document(current.user_id, document_id)
        return Response(status_code=204)

    @app.get("/chats")
    def list_chats(request: Request, current: Identity = Depends(identity)):
        return request.app.state.history.list_chats(current.user_id)

    @app.get("/chats/{chat_id}/messages")
    def chat_messages(request: Request, chat_id: UUID, before: int | None = None, current: Identity = Depends(identity)):
        return request.app.state.history.read(current.user_id, chat_id, before)

    @app.post("/ask")
    async def ask(request: Request, current: Identity = Depends(verified)):
        state, owner = request.app.state, current.user_id
        body = await _json(request, 65536)
        question, versions, requested_chat = body.get("question"), body.get("version_ids"), body.get("chat_id")
        if not isinstance(question, str) or not question.strip() or len(question.strip()) > settings.max_question_chars:
            raise ApiError(422, "invalid_question", f"Questions must contain 1 to {settings.max_question_chars} characters.")
        if "\0" in question or any(0xD800 <= ord(ch) <= 0xDFFF for ch in question):
            raise ApiError(422, "invalid_question", "Use a valid text question.")
        question = question.strip()
        if not isinstance(versions, list) or not 1 <= len(versions) <= 20:
            raise ApiError(422, "invalid_versions", "Select between one and twenty document versions.")
        try:
            version_ids = tuple(dict.fromkeys(UUID(str(value)) for value in versions))
            chat_id = uuid4() if requested_chat is None else UUID(str(requested_chat))
        except ValueError as exc:
            raise ApiError(422, "invalid_request", "The document version or chat identifier is invalid.") from exc

        day, ttl = today()
        await take(state.redis, ("ask-ip", _client_ip(request), 30, 60), ("ask-user", str(owner), 12, 60),
                   ("questions-per-day", f"{owner}:{day}", settings.daily_questions, ttl))
        await run_in_threadpool(state.library.require_versions, owner, version_ids)
        turns = await run_in_threadpool(state.history.recent_turns, owner, chat_id, version_ids) if requested_chat is not None else []

        async def events():
            yield _sse("start", {"chat_id": chat_id})
            work = asyncio.create_task(run_in_threadpool(
                state.library.ask, owner, question, version_ids=version_ids, turns=turns,
            ))
            try:
                while not work.done():
                    done, _ = await asyncio.wait({work}, timeout=10)
                    if not done:
                        yield ": keep-alive\n\n"
                result = await work
                citations = [citation.to_record() for citation in result.citations]
                await run_in_threadpool(
                    state.history.save, owner, chat_id, question, result.answer, citations, new=requested_chat is None,
                )
                yield _sse("answer", {"chat_id": chat_id, "answer": result.answer, "abstained": result.abstained,
                                      "citations": citations, "conflicts": result.metadata.get("conflicts"),
                                      "conflict_policy": result.metadata.get("conflict_policy")})
            except Exception as exc:
                logger.warning("Question stream failed (%s)", type(exc).__name__)
                yield _sse("error", error_body(public_error(exc), request_id.get()))
            finally:
                if not work.done():
                    work.cancel()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    app.include_router(web_router)
    return app
