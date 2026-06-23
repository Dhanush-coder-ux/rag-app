import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from app.models.document import Document, Chunk
from app.llms.nvidia.embeddings import NvidiaEmbeddings
from app.llms.gemini.embeddings import GeminiEmbeddings
from app.rag_services.chunker import chunk_text, extract_text_from_pdf
from app.utils.youtube_extractor import extract_transcript, format_transcript_for_ingestion
from app.core.config import settings

class __DocumentIngestion:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.nvidia_embeds = NvidiaEmbeddings()
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
                    if model == "nvidia":
                        embedding = await self.nvidia_embeds.get_embedding(chunk_content)
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
            model: Embedding model to use ("gemini" or "nvidia")
        
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
                    if model == "nvidia":
                        embedding = await self.nvidia_embeds.get_embedding(chunk_content)
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
        model: str = "nvidia",  # must match EMBEDDING_PROVIDER
        document_ids: list[int] | None = None,
    ) -> list[Chunk]:
    
        from sqlalchemy import func
        top_k = top_k or settings.TOP_K_RESULTS
        
        # 1. Vector Search Query — use nvidia for any non-gemini model value
        if model == "gemini":
            query_embedding = await self.gemini_embeds.get_query_embedding(question)
        else:  # nvidia, auto, groq, etc. all use the configured NVIDIA embedding
            query_embedding = await self.nvidia_embeds.get_query_embedding(question)
        
        vec_stmt = select(Chunk).options(joinedload(Chunk.document))
        if document_ids:
            vec_stmt = vec_stmt.where(Chunk.document_id.in_(document_ids))
            
        vec_stmt = (
            vec_stmt.order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k * 2)
        )
        
        # 2. Full-Text Search (FTS) Query
        fts_stmt = select(Chunk).options(joinedload(Chunk.document))
        if document_ids:
            fts_stmt = fts_stmt.where(Chunk.document_id.in_(document_ids))
            
        ts_query = func.plainto_tsquery('english', question)
        ts_vector = func.to_tsvector('english', Chunk.content)
        
        fts_stmt = fts_stmt.where(ts_vector.op('@@')(ts_query)).order_by(
            func.ts_rank(ts_vector, ts_query).desc()
        ).limit(top_k * 2)

        # 3. Execute sequentially to avoid AsyncSession concurrent errors
        vec_res = await self.db.execute(vec_stmt)
        fts_res = await self.db.execute(fts_stmt)
        
        vec_chunks = list(vec_res.scalars().all())
        fts_chunks = list(fts_res.scalars().all())

        # 4. Reciprocal Rank Fusion (RRF)
        k = 60
        chunk_scores: dict[int, float] = {}
        chunk_map: dict[int, Chunk] = {}

        for rank, chunk in enumerate(vec_chunks):
            if chunk.id not in chunk_scores:
                chunk_scores[chunk.id] = 0.0
                chunk_map[chunk.id] = chunk
            chunk_scores[chunk.id] += 1.0 / (k + rank)

        for rank, chunk in enumerate(fts_chunks):
            if chunk.id not in chunk_scores:
                chunk_scores[chunk.id] = 0.0
                chunk_map[chunk.id] = chunk
            chunk_scores[chunk.id] += 1.0 / (k + rank)

        # 5. Sort by RRF score and return top_k
        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda c: chunk_scores[c.id],
            reverse=True
        )
        
        return sorted_chunks[:top_k]


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