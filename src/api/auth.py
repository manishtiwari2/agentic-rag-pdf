"""Password verification and opaque, revocable database sessions."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from hmac import compare_digest
from secrets import token_urlsafe
from threading import BoundedSemaphore
from uuid import UUID, uuid4

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from ..errors import StorageError
from ..storage.database import tenant_transaction
from ..storage.schema import audit_log, sessions, users
from .errors import ApiError

SESSION_COOKIE = "__Host-rag_session"
CSRF_COOKIE = "__Host-rag_csrf"
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def email_address(value: str) -> str:
    value = value.strip().lower()
    if len(value) > 254 or not value.isascii() or "\0" in value or not _EMAIL.fullmatch(value):
        raise ApiError(422, "invalid_email", "Enter a valid email address.")
    return value


@dataclass(frozen=True)
class Identity:
    user_id: UUID
    session_id: UUID
    role: str
    csrf_hash: str

    def check_csrf(self, supplied: str | None) -> None:
        if not supplied or len(supplied) > 128 or not compare_digest(self.csrf_hash, digest(supplied)):
            raise ApiError(403, "csrf_failed", "A valid CSRF token is required.")


class Accounts:
    def __init__(self, engine, session_seconds: int) -> None:
        self.engine = engine
        self.session_seconds = session_seconds
        self.hasher = PasswordHasher()
        self._slots = BoundedSemaphore(4)
        self.dummy = self.hasher.hash(token_urlsafe(32))

    def register(self, email: str, password: str) -> None:
        if not 12 <= len(password) <= 256:
            raise ApiError(422, "invalid_password", "Use a password between 12 and 256 characters.")
        with self._slots:
            password_hash = self.hasher.hash(password)
        try:
            with self.engine.begin() as conn:
                new_user = conn.scalar(insert(users).values(
                    id=uuid4(), email=email, password_hash=password_hash, role="user",
                ).on_conflict_do_nothing(index_elements=[users.c.email]).returning(users.c.id))
                if new_user is None:
                    raise ApiError(409, "email_unavailable", "This email address is unavailable.")
        except SQLAlchemyError as exc:
            raise StorageError("The account store is unavailable.") from exc

    def login(self, email: str, password: str) -> tuple[str, str, str]:
        try:
            with self.engine.connect() as conn:
                account = conn.execute(select(users).where(users.c.email == email)).mappings().one_or_none()
        except SQLAlchemyError as exc:
            raise StorageError("The account store is unavailable.") from exc
        try:
            with self._slots:
                valid = len(password) <= 256 and self.hasher.verify(
                    account["password_hash"] if account else self.dummy, password,
                )
        except VerificationError:
            valid = False
        if not valid or account is None:
            raise ApiError(401, "invalid_credentials", "Invalid email or password.")
        owner, session = account["id"], uuid4()
        token, csrf = token_urlsafe(32), token_urlsafe(32)
        with tenant_transaction(self.engine, owner) as conn:
            conn.execute(sessions.delete().where(sessions.c.user_id == owner, sessions.c.expires_at <= func.now()))
            conn.execute(sessions.insert().values(
                user_id=owner, id=session, token_hash=digest(token), csrf_hash=digest(csrf),
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.session_seconds),
            ))
            conn.execute(audit_log.insert().values(user_id=owner, action="login", resource_id=session))
        return f"{owner}.{session}.{token}", csrf, account["role"]

    def authenticate(self, cookie: str | None) -> Identity:
        error = ApiError(401, "authentication_required", "Log in to continue.")
        if cookie is None or len(cookie) > 160:
            raise error
        try:
            owner, session, token = cookie.split(".")
            owner_id, session_id = UUID(owner), UUID(session)
        except ValueError:
            raise error from None
        s, u = sessions.c, users.c
        with tenant_transaction(self.engine, owner_id) as conn:
            row = conn.execute(select(s.token_hash, s.csrf_hash, u.role).select_from(
                sessions.join(users, s.user_id == u.id)
            ).where(s.user_id == owner_id, s.id == session_id, s.expires_at > func.now())).one_or_none()
        if row is None or not compare_digest(row.token_hash, digest(token)):
            raise error
        return Identity(owner_id, session_id, row.role, row.csrf_hash)

    def logout(self, identity: Identity) -> None:
        with tenant_transaction(self.engine, identity.user_id) as conn:
            conn.execute(sessions.delete().where(
                sessions.c.user_id == identity.user_id, sessions.c.id == identity.session_id,
            ))
            conn.execute(audit_log.insert().values(
                user_id=identity.user_id, action="logout", resource_id=identity.session_id,
            ))
