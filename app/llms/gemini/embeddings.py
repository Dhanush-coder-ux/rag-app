from google import genai
from google.genai import types
from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

class GeminiEmbeddings:
    async def get_embedding(self, text: str) -> list[float]:
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
            output_dimensionality=768 
        )
    )

        return response.embeddings[0].values


    async def get_query_embedding(self, text: str) -> list[float]:
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
        contents=text,

        config=types.EmbedContentConfig(
            output_dimensionality=768 
        )
    )

        return response.embeddings[0].values