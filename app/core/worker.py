import asyncio
import os
from celery import Celery
from app.core.database import AsyncSessionLocal, engine
from app.core.config import settings
import logging
from app.rag_services.document_service import DocumentService

logger = logging.getLogger(__name__)

# Create a single global event loop for the Celery worker process
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


celery_app = Celery(
    "worker",
    broker=str(settings.REDIS_URL),
    backend=str(settings.REDIS_URL),
)


@celery_app.task(name="ingest_document_task")
def ingest_document_task(filename: str, file_bytes: bytes, content_type: str):
    
    async def run():
        try:
            async with AsyncSessionLocal() as db:
                doc = await DocumentService(db).ingest_document(
                    filename=filename,
                    file_bytes=file_bytes,
                    content_type=content_type,
                    model="nvidia"
                )
                return {"status": "success", "doc_id": doc.id}
        except Exception as e:
            logger.error(f"Failed to ingest document {filename}: {str(e)}")
            return {"status": "failed", "error": str(e)}

    return _loop.run_until_complete(run())


@celery_app.task(name="ingest_youtube_task")
def ingest_youtube_task(youtube_url: str, model: str = "nvidia"):
    async def run():
        try:
            async with AsyncSessionLocal() as db:
                doc = await DocumentService(db).ingest_youtube(
                    youtube_url=youtube_url,
                    model=model
                )
                return {"status": "success", "doc_id": doc.id}
        except Exception as e:
            logger.error(f"Failed to ingest YouTube video {youtube_url}: {str(e)}")
            return {"status": "failed", "error": str(e)}

    return _loop.run_until_complete(run())