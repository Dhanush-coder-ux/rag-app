import asyncio
from sqlalchemy import text
from app.core.database import engine

async def apply_index():
    async with engine.begin() as conn:
        print("Creating HNSW index on chunks.embedding...")
        # We use vector_cosine_ops because the similarity search uses cosine distance
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS hnsw_index 
            ON chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        print("Index created successfully!")

if __name__ == "__main__":
    asyncio.run(apply_index())
