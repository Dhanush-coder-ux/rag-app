import httpx
import json
from app.core.config import settings

class Llama3Generation:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Set a long timeout (60s) to prevent ReadTimeout for LLM generation
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=60.0))
        return self._client

    async def rewrite_query(self, question: str) -> str:
        prompt = f"""
    Rewrite the user query into a clear question.

    User query:
    {question}

    Rewritten query:
    """
        response = await self.client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()

    async def generate_chat_title(self, message: str) -> str:
        prompt = f"""
    Generate a short 3-5 word title for this chat.

    Message:
    {message}

    Only return the title.
    """

        response = await self.client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()
    
    async def select_route(self, prompt: str) -> str:
        response = await self.client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    
    async def generate_text(self, prompt: str) -> str:
        response = await self.client.post(
           f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    async def generate_answer_stream(self, prompt: str):
        async with self.client.stream(
            "POST",
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]