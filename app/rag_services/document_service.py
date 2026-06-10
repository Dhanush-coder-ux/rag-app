import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from app.models.document import Document, Chunk
from app.llms.llama3.embeddings import Llama3Embeddings
from app.llms.gemini.embeddings import GeminiEmbeddings
from app.rag_services.chunker import chunk_text, extract_text_from_pdf
from app.utils.youtube_extractor import extract_transcript, format_transcript_for_ingestion
from app.core.config import settings

class __DocumentIngestion:
    def __init__(self, db: AsyncSession):
        self.db = db
        # Renamed slightly for cleaner code below
        self.llama_embeds = Llama3Embeddings()
        self.gemini_embeds = GeminiEmbeddings()

class DocumentService(__DocumentIngestion):
    
    async def ingest_document(
        self,
        filename: str,
        file_bytes: bytes,
        content_type: str,
        model: str = "gemini", # 👈 Added model selection parameter
    ) -> Document:
    
        doc = Document(filename=filename, status="processing")
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc) 

        try:
            if content_type == "application/pdf":
                raw_text = extract_text_from_pdf(file_bytes)
            else:
                raw_text = file_bytes.decode("utf-8", errors="replace")

            chunks_text = chunk_text(raw_text)

            sem = asyncio.Semaphore(20) # Process 20 chunks concurrently

            async def process_chunk(idx, chunk_content):
                async with sem:
                    if model == "llama3":
                        embedding = await self.llama_embeds.get_embedding(chunk_content)
                    else:
                        embedding = await self.gemini_embeds.get_embedding(chunk_content)
                    
                    return Chunk(
                        document_id=doc.id,
                        content=chunk_content,
                        chunk_index=idx,
                        embedding=embedding,
                    )

            tasks = [process_chunk(idx, content) for idx, content in enumerate(chunks_text)]
            chunks = await asyncio.gather(*tasks)
            
            for chunk in chunks:
                self.db.add(chunk)

            doc.status = "ready"
            await self.db.commit()
            await self.db.refresh(doc)

        except Exception as exc:
            await self.db.rollback()
            # Update the committed document to failed
            doc.status = "failed"
            self.db.add(doc)
            await self.db.commit()
            raise exc

        return doc


    async def ingest_youtube(
        self,
        youtube_url: str,
        model: str = "gemini",
    ) -> Document:
        """
        Ingest YouTube video transcript and create embeddings.
        
        Args:
            youtube_url: URL of the YouTube video
            model: Embedding model to use ("gemini" or "llama3")
        
        Returns:
            Document object with ingested transcript
        """
        # Create document record first to track status
        doc = Document(
            filename=youtube_url,  # Placeholder until we get metadata
            status="processing",
            source_type="youtube",
            source_url=youtube_url
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        
        try:
            # Extract transcript
            transcript_text, metadata = await extract_transcript(youtube_url)
            
            # Update with actual title
            doc.filename = metadata.get("title", youtube_url)
            
            # Format for ingestion
            raw_text = format_transcript_for_ingestion(transcript_text, metadata)
            
            # Chunk and embed
            chunks_text = chunk_text(raw_text)
            
            sem = asyncio.Semaphore(20)

            async def process_chunk(idx, chunk_content):
                async with sem:
                    if model == "llama3":
                        embedding = await self.llama_embeds.get_embedding(chunk_content)
                    else:
                        embedding = await self.gemini_embeds.get_embedding(chunk_content)
                    
                    return Chunk(
                        document_id=doc.id,
                        content=chunk_content,
                        chunk_index=idx,
                        embedding=embedding,
                    )

            tasks = [process_chunk(idx, content) for idx, content in enumerate(chunks_text)]
            chunks = await asyncio.gather(*tasks)
            
            for chunk in chunks:
                self.db.add(chunk)
            
            doc.status = "ready"
            await self.db.commit()
            await self.db.refresh(doc)
            
            return doc
            
        except Exception as exc:
            await self.db.rollback()
            # Mark the document as failed
            doc.status = "failed"
            self.db.add(doc)
            await self.db.commit()
            raise exc


    async def similarity_search(
        self,
        question: str,
        top_k: int | None = None,
        model: str = "gemini", # 👈 Ensure the search query uses the same model!
        document_ids: list[int] | None = None,
    ) -> list[Chunk]:
    
        top_k = top_k or settings.TOP_K_RESULTS
        
        # 👈 Route the query embedding
        if model == "llama3":
            query_embedding = await self.llama_embeds.get_query_embedding(question)
        else:
            query_embedding = await self.gemini_embeds.get_query_embedding(question)
        
        stmt = select(Chunk).options(joinedload(Chunk.document))
        
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
            
        stmt = (
            stmt.order_by(Chunk.embedding.cosine_distance(query_embedding))
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