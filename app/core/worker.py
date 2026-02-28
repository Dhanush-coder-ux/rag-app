# worker.py
import asyncio
from celery import Celery
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.services.document_service import ingest_document

celery_app = Celery(
    "worker",
    broker=str(settings.REDIS_URL),
    backend=str(settings.REDIS_URL),
)




@celery_app.task(name="ingest_document_task")
def ingest_document_task(filename: str, file_bytes: bytes, content_type: str):
    
    async def run():
        async with AsyncSessionLocal() as db:
            return await ingest_document(
                db=db,
                filename=filename,
                file_bytes=file_bytes,
                content_type=content_type
            )

    return asyncio.run(run())