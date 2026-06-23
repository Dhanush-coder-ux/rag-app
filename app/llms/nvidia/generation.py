from openai import AsyncOpenAI
from app.core.config import settings


class NvidiaGeneration:
    """LLM generation using NVIDIA NIM API (OpenAI-compatible endpoint)."""

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

    @property
    def model(self) -> str:
        return settings.NVIDIA_MODEL

    # ── Chat title ─────────────────────────────────────────────────────────────

    async def generate_chat_title(self, message: str) -> str:
        prompt = (
            "Generate a short 3-5 word title for this conversation. "
            "Return ONLY the title, no punctuation.\n\n"
            f"Message: {message}\n\nTitle:"
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip().replace("\n", "")

    # ── Route selection ────────────────────────────────────────────────────────

    async def select_route(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        return response.choices[0].message.content.strip()

    # ── General text generation ────────────────────────────────────────────────

    async def generate_text(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    # ── RAG streaming answer ───────────────────────────────────────────────────

    async def generate_answer_stream(self, question: str, context_chunks: list[str]):
        context = "\n\n---\n\n".join(context_chunks)
        system_prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the question. "
            'If the answer cannot be found in the context, say "I don\'t have enough information to answer that."'
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
