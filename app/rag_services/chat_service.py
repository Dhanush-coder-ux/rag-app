from __future__ import annotations
import logging
from datetime import datetime
from typing import Sequence
from sqlalchemy import delete, select
from sqlalchemy.orm import noload
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chat_session import ChatMessage, ChatSession
from app.rag_services.gemini import generate_chat_title
from app.utils.chat_service import ChatServices as _ChatServices
from app.core.config import settings

logger = logging.getLogger(__name__)

_compress_helper = _ChatServices()


class _Chat:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db


class ChatServices(_Chat):

    async def _session_exists(self, session_id: int) -> bool:
        result = await self.db.execute(
            select(ChatSession.id).where(ChatSession.id == session_id)
        )
        return result.scalar_one_or_none() is not None

    async def _prune_old_messages(self, session_id: int) -> None:
        keep_result = await self.db.execute(
            select(ChatMessage.id)
            .where(ChatMessage.chat_session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(settings.MAX_MESSAGES_PER_SESSION)
        )
        keep_ids: list[int] = [row[0] for row in keep_result.fetchall()]
        if not keep_ids:
            return
        await self.db.execute(
            delete(ChatMessage).where(
                ChatMessage.chat_session_id == session_id,
                ChatMessage.id.not_in(keep_ids),
            )
        )

    async def save_message(
        self,
        session_id: int | None,
        role: str,
        content: str,
        *,
        prune: bool = False,
    ) -> ChatMessage | None:
        if not session_id:
            return None

        try:
            if not await self._session_exists(session_id):
                logger.warning(
                    "save_message skipped — session_id=%s not found in DB",
                    session_id,
                )
                return None

            msg = ChatMessage(
                chat_session_id=session_id,
                role=role,
                content=_compress_helper._compress(content),
                token_count=_compress_helper._estimate_tokens(content),
                
            )
            self.db.add(msg)

            if prune:
                await self._prune_old_messages(session_id)

            await self.db.commit()
            await self.db.refresh(msg)
            logger.debug("Saved %s message id=%s for session %s", role, msg.id, session_id)
            return msg

        except Exception:
            await self.db.rollback()
            logger.exception(
                "Failed to save %s message for session %s — storage skipped",
                role, session_id,
            )
            return None

    async def save_user_message(
        self, session_id: int | None, content: str
    ) -> ChatMessage | None:
        return await self.save_message(session_id, "user", content, prune=False)

    async def save_assistant_message(
        self, session_id: int | None, content: str
    ) -> ChatMessage | None:
        return await self.save_message(session_id, "assistant", content, prune=True)

    async def get_messages(
        self, session_id: int, limit: int = settings.MAX_MESSAGES_PER_SESSION
    ) -> list[dict]:
        """
        Returns decompressed messages with all fields needed by ChatMessageResponse.
        content is always a decompressed str — LargeBinary bytes never leave this method.
        """
        try:
            result = await self.db.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
            rows: Sequence[ChatMessage] = result.scalars().all()
            return [
                {
                    "id":          r.id,
                    "role":        r.role,
                    "content":     _compress_helper._decompress(r.content),
                    "token_count": r.token_count,
                    "created_at":  r.created_at,
                }
                for r in rows
            ]
        except Exception:
            logger.exception("Failed to retrieve messages for session %s", session_id)
            return []

    async def generate_title(self, message: str) -> str:
        try:
            title = await generate_chat_title(message)
            return title.strip().replace("\n", "")[:60]
        except Exception:
            logger.warning("Title generation failed — using message prefix")
            return message[:30]

    async def update_title_if_needed(self, session_id: int, message: str) -> None:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat = result.scalar_one_or_none()
        if not chat or chat.title != "New Chat":
            return

        title = await self.generate_title(message)
        chat.title = title
        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Failed to update title for session %s", session_id)

    async def create_chat_session(self) -> ChatSession:
        try:
            check = await self.db.execute(select(ChatSession).where(ChatSession.title == "New Chat"))
            existing = check.scalar_one_or_none()
            if existing:
                return existing
            session = ChatSession(title="New Chat")
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            return session
        except Exception:
            await self.db.rollback()
            raise

    async def get_chat_sessions(self) -> list[ChatSession]:
 
        try:
            result = await self.db.execute(
                select(ChatSession)
                .options(noload(ChatSession.messages))
                .order_by(ChatSession.updated_at.desc())
            )
            return list(result.scalars().all())
        except Exception:
            await self.db.rollback()
            raise

    async def get_chat_session(self, session_id: int) -> ChatSession | None:

        try:
            result = await self.db.execute(
                select(ChatSession)
                .options(noload(ChatSession.messages))
                .where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()
        except Exception:
            await self.db.rollback()
            raise

    async def delete_chat_session(self, session_id: int) -> bool:
        try:
            result = await self.db.execute(
                select(ChatSession)
                .options(noload(ChatSession.messages))
                .where(ChatSession.id == session_id)
            )
            chat = result.scalar_one_or_none()
            if not chat:
                return False
            await self.db.delete(chat)
            await self.db.commit()
            return True
        except Exception:
            await self.db.rollback()
            raise