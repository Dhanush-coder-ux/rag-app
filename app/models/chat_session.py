from __future__ import annotations
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    title:      Mapped[str]      = mapped_column(String, default="New Chat")
    summary:    Mapped[str|None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="noload",  
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id:              Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    chat_session_id: Mapped[int]      = mapped_column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role:        Mapped[str]      = mapped_column(String(16))
    content:     Mapped[bytes]    = mapped_column(LargeBinary)   
    token_count: Mapped[int|None] = mapped_column(Integer, nullable=True, default=None)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")