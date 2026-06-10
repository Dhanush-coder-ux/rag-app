import httpx
from app.core.config import settings

class Llama3Embeddings:
    async def get_embedding(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def get_query_embedding(self, text: str) -> list[float]:
        return await self.get_embedding(text)