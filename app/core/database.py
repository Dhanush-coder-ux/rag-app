from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    # -------------------------------------------------------------------------
    # WHY AUTOCOMMIT FOR CREATE EXTENSION?
    # -------------------------------------------------------------------------
    # gunicorn spawns multiple workers that all call init_db() simultaneously.
    # `CREATE EXTENSION IF NOT EXISTS` inside a regular transaction is NOT
    # truly idempotent under concurrent load:
    #   - Two workers open separate transactions, both see the extension as
    #     absent (not yet committed by either), and both attempt to insert it.
    #   - One succeeds; the other hits a UniqueViolationError on
    #     pg_extension_name_index.
    #
    # AUTOCOMMIT bypasses the transaction layer entirely. PostgreSQL acquires
    # an exclusive lock on the extension catalog row, so only one CREATE
    # EXTENSION runs; every concurrent call sees the committed row and the
    # IF NOT EXISTS guard correctly skips creation.
    # -------------------------------------------------------------------------
    async with engine.connect() as conn:
        # Set autocommit BEFORE executing any statement on this connection
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except IntegrityError:
            # Another worker just created it between our check and execute.
            # Completely safe to ignore — extension is there.
            pass

    # Create all ORM-mapped tables inside a proper transaction.
    # `engine.begin()` commits on success and rolls back on any exception.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
