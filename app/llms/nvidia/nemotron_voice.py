"""
NemotronVoiceGeneration — NVIDIA Nemotron-70B via NIM chat completions.

Model: nvidia/llama-3.1-nemotron-70b-instruct
  - Available via standard /v1/chat/completions (OpenAI-compatible)
  - Uses NEMOTRON_API_KEY (separate from main NVIDIA_API_KEY)
  - Falls back to NVIDIA_API_KEY if NEMOTRON_API_KEY is not set

NOTE:  nvidia/nemotron-voicechat is a DIFFERENT model — it uses a
full-duplex WebSocket audio protocol and is NOT compatible with the
chat-completions endpoint.  We use the 70B instruct variant here.
"""

import logging
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class NemotronVoiceGeneration:
    """
    LLM generation via NVIDIA Nemotron 70B Instruct (NIM, OpenAI-compatible).

    Optimised for voice interactions:
    - Concise, naturally speakable responses
    - Max 512 tokens to keep TTS latency low
    - Temperature 0.65 for natural, varied speech
    """

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            # Prefer the dedicated Nemotron key; fall back to the general key
            api_key = (settings.NEMOTRON_API_KEY or "").strip() or settings.NVIDIA_API_KEY
            if not api_key:
                raise ValueError(
                    "No NVIDIA API key found. "
                    "Set NEMOTRON_API_KEY or NVIDIA_API_KEY in .env"
                )
            self._client = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
        return self._client

    @property
    def model(self) -> str:
        # nvidia/llama-3.1-nemotron-70b-instruct (set in .env)
        return settings.NEMOTRON_VOICE_MODEL

    # ── Streaming RAG answer (voice-optimised) ────────────────────────────────

    async def generate_answer_stream(self, question: str, context_chunks: list[str]):
        """
        Stream an answer from Nemotron 70B, with context from RAG retrieval.
        Designed for TTS: short sentences, no markdown/bullet points/code.
        """
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no document context)"
        system_prompt = (
            "You are a helpful, conversational voice assistant. "
            "Use the context provided to answer the user's question. "
            "Respond in natural spoken language — no markdown, no bullet points, no code blocks. "
            "Keep your answer concise (2–4 sentences) unless a longer explanation is truly needed. "
            'If the answer is not in the context, say "I don\'t have that information in the documents."'
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.65,
                top_p=0.9,
                max_tokens=512,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as exc:
            logger.error(
                "NemotronVoiceGeneration.generate_answer_stream error (model=%s): %r",
                self.model, exc,
            )
            raise

    # ── Non-streaming helpers ──────────────────────────────────────────────────

    async def generate_text(self, prompt: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("NemotronVoiceGeneration.generate_text error: %r", exc)
            raise

    async def generate_chat_title(self, message: str) -> str:
        prompt = (
            "Generate a short 3-5 word title for this conversation. "
            "Return ONLY the title, no punctuation.\n\n"
            f"Message: {message}\n\nTitle:"
        )
        return await self.generate_text(prompt)
