# app/llms/nvidia/embeddings.py
from openai import AsyncOpenAI
from app.core.config import settings


class NvidiaEmbeddings:
    """Embeddings via NVIDIA NIM API using BAAI/bge-m3 (1024-dim)."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=settings.NVIDIA_API_KEY,
            )
        return self._client

    async def get_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            input=[text],
            model="baai/bge-m3",
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"},
        )
        return response.data[0].embedding

    async def get_query_embedding(self, text: str) -> list[float]:
        """Use 'query' input_type for search queries (recommended by BGE-M3)."""
        response = await self.client.embeddings.create(
            input=[text],
            model="baai/bge-m3",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return response.data[0].embedding
