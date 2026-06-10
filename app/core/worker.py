# worker.py
import asyncio
import os
from celery import Celery
from app.core.database import AsyncSessionLocal, engine
from app.core.config import settings
import logging
from app.rag_services.document_service import DocumentService

logger = logging.getLogger(__name__)


celery_app = Celery(
    "worker",
    broker=str(settings.REDIS_URL),
    backend=str(settings.REDIS_URL),
)

def _save_file_to_disk(doc_id: int, filename: str, content_type: str, file_bytes: bytes):
    """Persist the uploaded file so it can be served for PDF preview."""
    uploads_dir = settings.UPLOADS_DIR
    os.makedirs(uploads_dir, exist_ok=True)
    ext = ".pdf" if content_type == "application/pdf" else ".txt"
    path = os.path.join(uploads_dir, f"{doc_id}{ext}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    logger.info("Saved uploaded file to %s", path)


@celery_app.task(name="ingest_document_task")
def ingest_document_task(filename: str, file_bytes: bytes, content_type: str):
    
    async def run():
        try:
            async with AsyncSessionLocal() as db:
                doc = await DocumentService(db).ingest_document(
                    filename=filename,
                    file_bytes=file_bytes,
                    content_type=content_type
                )
                # Save raw file for PDF preview
                _save_file_to_disk(doc.id, filename, content_type, file_bytes)
                return {"status": "success", "doc_id": doc.id}
        except Exception as e:
            logger.error(f"Failed to ingest document {filename}: {str(e)}")
            return {"status": "failed", "error": str(e)}
        finally:
            # Clean up connection pool so next task's new event loop doesn't get a closed connection
            await engine.dispose()

    return asyncio.run(run())


@celery_app.task(name="ingest_youtube_task")
def ingest_youtube_task(youtube_url: str, model: str = "gemini"):
    """
    Celery task to ingest YouTube video transcript asynchronously.
    
    Args:
        youtube_url: URL of the YouTube video
        model: Embedding model to use ("gemini" or "llama3")
    """
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
        finally:
            await engine.dispose()

    return asyncio.run(run())