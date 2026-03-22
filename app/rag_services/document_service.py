from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document, Chunk
from app.rag_services.gemini import get_embedding, get_query_embedding
from app.rag_services.chunker import chunk_text, extract_text_from_pdf
from app.core.config import settings


class __DocumentIngestion:
    def __init__(self, db: AsyncSession):
        self.db = db
class DocumentService(__DocumentIngestion):
    async def ingest_document(
            self,
        filename: str,
        file_bytes: bytes,
        content_type: str,
    ) -> Document:
    
        doc = Document(filename=filename, status="processing")
        self.db.add(doc)
        await self.db.flush() 

        try:
            if content_type == "application/pdf":
                raw_text = extract_text_from_pdf(file_bytes)
            else:
                raw_text = file_bytes.decode("utf-8", errors="replace")

            chunks_text = chunk_text(raw_text)

            for idx, chunk_content in enumerate(chunks_text):
                embedding = await get_embedding(chunk_content)
                chunk = Chunk(
                    document_id=doc.id,
                    content=chunk_content,
                    chunk_index=idx,
                    embedding=embedding,
                )
                self.db.add(chunk)

            doc.status = "ready"
            await self.db.commit()
            await self.db.refresh(doc)

        except Exception as exc:
            await self.db.rollback()
            doc.status = "failed"
            await self.db.commit()
            raise exc

        return doc


    async def similarity_search(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[Chunk]:
    
        top_k = top_k or settings.TOP_K_RESULTS
        query_embedding = await get_query_embedding(question)
        
        stmt = (
            select(Chunk)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


    async def list_documents(self) -> list[Document]:
        result = await self.db.execute(select(Document).order_by(Document.created_at.desc()))
        return list(result.scalars().all())


    async def delete_document(self, document_id: int) -> bool:
        result = await self.db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return False
        await self.db.delete(doc)
        await self.db.commit()
        return True
