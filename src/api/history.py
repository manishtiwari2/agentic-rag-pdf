"""Store complete question and answer pairs under their owner."""

from uuid import UUID

from sqlalchemy import select

from ..storage.database import tenant_transaction
from ..storage.schema import audit_log, chats, messages
from .errors import ApiError


class History:
    def __init__(self, engine) -> None:
        self.engine = engine

    @staticmethod
    def _chat(conn, owner: UUID, chat_id: UUID, *, lock: bool = False) -> None:
        query = select(chats.c.id).where(chats.c.user_id == owner, chats.c.id == chat_id)
        if conn.scalar(query.with_for_update() if lock else query) is None:
            raise ApiError(404, "not_found", "The chat does not exist.")

    def verify(self, owner: UUID, chat_id: UUID) -> None:
        with tenant_transaction(self.engine, owner) as conn:
            self._chat(conn, owner, chat_id)

    def list_chats(self, owner: UUID) -> list[dict]:
        c = chats.c
        with tenant_transaction(self.engine, owner) as conn:
            return [dict(row) for row in conn.execute(select(c.id, c.title, c.created_at).where(
                c.user_id == owner,
            ).order_by(c.created_at.desc()).limit(100)).mappings()]

    def read(self, owner: UUID, chat_id: UUID, before: int | None = None) -> list[dict]:
        m = messages.c
        with tenant_transaction(self.engine, owner) as conn:
            self._chat(conn, owner, chat_id)
            query = select(m.id, m.role, m.content, m.citations, m.created_at).where(
                m.user_id == owner, m.chat_id == chat_id,
            )
            if before is not None:
                query = query.where(m.id < before)
            return [dict(row) for row in reversed(conn.execute(query.order_by(m.id.desc()).limit(100)).mappings().all())]

    def save(
        self, owner: UUID, chat_id: UUID, question: str, answer: str, citations: list[dict], *, new: bool,
    ) -> None:
        with tenant_transaction(self.engine, owner) as conn:
            if new:
                conn.execute(chats.insert().values(user_id=owner, id=chat_id, title=question[:100]))
            else:
                self._chat(conn, owner, chat_id, lock=True)
            conn.execute(messages.insert(), [
                {"user_id": owner, "chat_id": chat_id, "role": "user", "content": question, "citations": []},
                {"user_id": owner, "chat_id": chat_id, "role": "assistant", "content": answer, "citations": citations},
            ])
            conn.execute(audit_log.insert().values(user_id=owner, action="ask", resource_id=chat_id))
