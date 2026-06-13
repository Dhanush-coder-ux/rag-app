from groq import AsyncGroq
from app.core.config import settings

# Groq free tier: 12,000 TPM. Reserve ~2,500 for system prompt + question + response.
_GROQ_MAX_CONTEXT_TOKENS = 9_000
_CHARS_PER_TOKEN = 4  # rough estimate


def _truncate_context(chunks: list[str], max_tokens: int = _GROQ_MAX_CONTEXT_TOKENS) -> list[str]:
    """Trim context chunks so the total stays within Groq's token budget."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    result, total = [], 0
    for chunk in chunks:
        if total + len(chunk) > max_chars:
            # Fit the remaining budget as a partial chunk
            remaining = max_chars - total
            if remaining > 200:  # only add if meaningful content remains
                result.append(chunk[:remaining] + "…")
            break
        result.append(chunk)
        total += len(chunk)
    return result or chunks[:1]  # always send at least 1 chunk


class GroqGeneration:
    """LLM generation using Groq's ultra-fast inference API."""

    def __init__(self):
        self._client: AsyncGroq | None = None

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._client

    @property
    def model(self) -> str:
        return settings.GROQ_MODEL


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
        # ✂️ Trim context to stay within Groq's free-tier token limit
        safe_chunks = _truncate_context(context_chunks)

        context = "\n\n---\n\n".join(safe_chunks)
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
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

