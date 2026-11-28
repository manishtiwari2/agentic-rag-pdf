"""Store complete question and answer pairs under their owner."""

from uuid import UUID

from sqlalchemy import select

from ..chat.session import DEFAULT_MAX_TURNS, Turn
from ..storage.database import tenant_transaction
from ..storage.library import DocumentLibrary
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

    def recent_turns(self, owner: UUID, chat_id: UUID, version_ids: tuple[UUID, ...]) -> list[Turn]:
        selected = {str(version) for version in version_ids}
        messages = self.read(owner, chat_id)[-2 * DEFAULT_MAX_TURNS:]
        turns = []
        for question, answer in zip(messages[::2], messages[1::2]):
            sources = {citation["document_id"] for citation in answer["citations"]}
            if question["role"] == "user" and answer["role"] == "assistant" and sources and sources <= selected:
                turns.append(Turn(question["content"], question["content"], answer["content"]))
        return turns

    def save(
        self, owner: UUID, chat_id: UUID, question: str, answer: str, citations: list[dict], *,
        version_ids: tuple[UUID, ...], new: bool,
    ) -> None:
        with tenant_transaction(self.engine, owner) as conn:
            # Keep document deletion from racing the final access check and history write.
            DocumentLibrary.version_revision(conn, owner, version_ids, lock=True)
            if new:
                conn.execute(chats.insert().values(user_id=owner, id=chat_id, title=question[:100]))
            else:
                self._chat(conn, owner, chat_id, lock=True)
            conn.execute(messages.insert(), [
                {"user_id": owner, "chat_id": chat_id, "role": "user", "content": question, "citations": []},
                {"user_id": owner, "chat_id": chat_id, "role": "assistant", "content": answer, "citations": citations},
            ])
            conn.execute(audit_log.insert().values(user_id=owner, action="ask", resource_id=chat_id))
