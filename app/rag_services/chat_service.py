from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.chat_session import ChatSession
from app.rag_services.gemini import generate_chat_title


class _Chat:
    def __init__(self, db: AsyncSession):
        self.db = db


class ChatServices(_Chat):

    async def generate_title(self, message: str) -> str:
        try:
            title = await generate_chat_title(message)
            return title.strip().replace("\n", "")[:60]   
        except Exception:
            return message[:30]

    async def update_title_if_needed(self, session_id: int, message: str) -> None:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat = result.scalar_one_or_none()

        if not chat:
            return

        if chat.title == "New Chat":
            title = await self.generate_title(message)
            chat.title = title
            try:
                await self.db.commit()
                await self.db.refresh(chat)
            except Exception as exc:
                await self.db.rollback()
                raise exc


    async def create_chat_session(self) -> ChatSession:
        try:
            new_chat = ChatSession(title="New Chat")
            self.db.add(new_chat)
            await self.db.commit()
            await self.db.refresh(new_chat)
            return new_chat
        except Exception as exc:
            await self.db.rollback()
            raise exc

    async def get_chat_sessions(self) -> list[ChatSession]:
        try:
            result = await self.db.execute(
                select(ChatSession).order_by(ChatSession.updated_at.desc())
            )
            return list(result.scalars().all())
        except Exception as exc:
            await self.db.rollback()
            raise exc

    async def get_chat_session(self, session_id: int) -> ChatSession | None:
        try:
            result = await self.db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            await self.db.rollback()
            raise exc

    async def delete_chat_session(self, session_id: int) -> bool:
        try:
            result = await self.db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            chat = result.scalar_one_or_none()
            if not chat:
                return False
            await self.db.delete(chat)
            await self.db.commit()
            return True
        except Exception as exc:
            await self.db.rollback()
            raise exc