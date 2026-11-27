"""Authenticated pages and the HTML fragments refreshed by htmx."""

from base64 import b64encode
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..api.auth import CSRF_COOKIE, SESSION_COOKIE, Identity
from ..api.errors import ApiError
from ..errors import PDFReadError
from ..ingestion.highlight import render_highlight

root = Path(__file__).resolve().parent
static_dir = root / "static"
templates = Jinja2Templates(directory=root / "templates")
router = APIRouter(include_in_schema=False)


def _account(request: Request) -> Identity | None:
    try:
        return request.app.state.accounts.authenticate(request.cookies.get(SESSION_COOKIE))
    except ApiError as exc:
        if exc.status != 401:
            raise
        return None


def _page(request: Request, name: str, account: Identity | None = None, **context):
    return templates.TemplateResponse(
        request=request, name=name, context={
            "account": account, "csrf": request.cookies.get(CSRF_COOKIE, "") if account else "", **context,
        },
    )


def _sign_in(request: Request):
    if request.headers.get("HX-Request") == "true":
        return HTMLResponse(headers={"HX-Redirect": "/login"})
    return RedirectResponse("/login", status_code=303)


@router.get("/")
def index(request: Request):
    return RedirectResponse("/library" if _account(request) else "/login", status_code=303)


@router.get("/login")
@router.get("/register")
def auth_page(request: Request):
    if _account(request):
        return RedirectResponse("/library", status_code=303)
    return _page(request, "auth.html", register=request.url.path == "/register")


@router.get("/library")
def library_page(request: Request):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    state = request.app.state
    return _page(request, "library.html", account, active="library",
                 documents=state.library.list_documents(account.user_id), jobs=state.jobs.status(account.user_id),
                 upload_limit=state.library.config.ingestion.max_upload_mb)


@router.get("/fragments/documents")
def document_fragment(request: Request):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    return _page(request, "documents.html", account,
                 documents=request.app.state.library.list_documents(account.user_id))


@router.get("/fragments/uploads")
def job_fragment(request: Request):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    return _page(request, "uploads.html", account, jobs=request.app.state.jobs.status(account.user_id))


@router.get("/fragments/upload-targets")
def upload_targets(request: Request):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    return _page(request, "upload_targets.html", account,
                 documents=request.app.state.library.list_documents(account.user_id))


@router.get("/chat")
def chat_page(request: Request, chat_id: UUID | None = None, version: UUID | None = None):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    state = request.app.state
    documents = state.library.list_documents(account.user_id)
    if version is not None:
        state.library.require_versions(account.user_id, [version])
    messages = state.history.read(account.user_id, chat_id) if chat_id else []
    return _page(
        request, "chat.html", account, active="chat", documents=documents, selected=version,
        chats=state.history.list_chats(account.user_id), chat_id=chat_id, messages=messages,
        max_question_chars=state.max_question_chars,
    )


@router.get("/citations/{version_id}/{chunk_id}/pages/{number}", name="cited_page")
def cited_page(request: Request, version_id: UUID, chunk_id: str, number: int):
    account = _account(request)
    if account is None:
        return _sign_in(request)
    source = request.app.state.library.citation_source(account.user_id, version_id, chunk_id, number)
    preview, highlighted = "", False
    try:
        png, highlighted = render_highlight(source.pdf, number, source.text, ocr=source.ocr)
        preview = b64encode(png).decode("ascii")
    except PDFReadError:
        logging.getLogger(__name__).warning("Could not render cited PDF page %s", number)
    return _page(
        request, "citation.html", account, source=source, version_id=version_id,
        chunk_id=chunk_id, number=number, preview=preview, highlighted=highlighted,
    )
