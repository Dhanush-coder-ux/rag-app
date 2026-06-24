import re
import logging
import asyncio
from app.llms.gemini.generation import GeminiGeneration
from app.llms.groq.generation   import GroqGeneration
from app.llms.nvidia.generation  import NvidiaGeneration

logger = logging.getLogger(__name__)

gemini = GeminiGeneration()
groq_  = GroqGeneration()
nvidia = NvidiaGeneration()


# ── Error message builder ─────────────────────────────────────────────────────

def _build_both_failed_msg(gemini_err: Exception, nvidia_err: Exception) -> str:
    """Build a specific, actionable error message based on what actually failed."""
    gemini_str = str(gemini_err)
    nvidia_str  = str(nvidia_err)

    retry_seconds = None
    retry_match = re.search(r'retryDelay["\s:\']+(\d+)', gemini_str)
    if not retry_match:
        retry_match = re.search(r'retry in ([\d.]+)s', gemini_str)
    if retry_match:
        retry_seconds = int(float(retry_match.group(1)))

    is_gemini_quota = "429" in gemini_str or "RESOURCE_EXHAUSTED" in gemini_str
    is_nvidia_error = "401" in nvidia_str or "403" in nvidia_str or "connection" in nvidia_str.lower()

    if is_gemini_quota and is_nvidia_error:
        retry_hint = f" (retry in ~{retry_seconds}s)" if retry_seconds else ""
        wait_line  = f"- Wait {retry_seconds}s and retry\n" if retry_seconds else ""
        return (
            f"⚠️ **Gemini quota exhausted{retry_hint}** — you've hit the free-tier daily limit "
            f"(20 requests/day). NVIDIA NIM fallback also failed.\n\n"
            f"**Options:**\n"
            f"{wait_line}"
            f"- Check your NVIDIA API key is valid\n"
            f"- Upgrade your Gemini API plan"
        )
    elif is_gemini_quota:
        retry_hint = f" Please retry in ~{retry_seconds}s." if retry_seconds else ""
        return f"⚠️ **Gemini quota exhausted** — you've hit the free-tier daily limit.{retry_hint}"
    elif is_nvidia_error:
        return (
            "⚠️ **NVIDIA NIM API error.** Check your `NVIDIA_API_KEY` in `.env` "
            "and ensure the model `z-ai/glm-5.1` is accessible."
        )
    else:
        return (
            f"⚠️ Both AI models failed.\n\n"
            f"- Gemini: `{repr(gemini_err)[:120]}`\n"
            f"- NVIDIA: `{repr(nvidia_err)[:120]}`"
        )


class LLMRouter:

    @staticmethod
    def _build_rag_prompt(question: str, context_chunks: list[str]) -> str:
        """Format the RAG prompt for Llama 3 (Groq uses chat messages directly)."""
        context = "\n\n---\n\n".join(context_chunks)
        return (
            "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
            'If the answer cannot be found in the context, say "I don\'t have enough information to answer that."\n\n'
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    # ── Streaming answer ──────────────────────────────────────────────────────

    @staticmethod
    async def generate_answer_stream(question: str, context_chunks: list[str], model: str = "auto"):
        """Yields (model_name, chunk) tuples so callers know which model actually ran."""

        # ── Explicit: Gemini ──────────────────────────────────────────────────
        if model == "gemini":
            try:
                gen = gemini.generate_answer_stream(question, context_chunks)
                try:
                    first_chunk = await asyncio.wait_for(anext(gen), timeout=15.0)
                except StopAsyncIteration:
                    return
                yield ("gemini-2.5-flash", first_chunk)
                async for chunk in gen:
                    yield ("gemini-2.5-flash", chunk)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, asyncio.TimeoutError):
                    # Rate-limited or timed out → try Groq first, then NVIDIA
                    logger.warning("Gemini failed (%s: %s), falling back to Groq", type(e).__name__, str(e))
                    try:
                        async for chunk in groq_.generate_answer_stream(question, context_chunks):
                            yield ("groq", chunk)
                    except Exception as groq_err:
                        logger.warning("Gemini→Groq fallback failed (%s), trying NVIDIA", groq_err)
                        try:
                            async for chunk in nvidia.generate_answer_stream(question, context_chunks):
                                yield ("nvidia", chunk)
                        except Exception as nvidia_err:
                            yield ("error", _build_both_failed_msg(e, nvidia_err))
                else:
                    raise e

        # ── Explicit: Groq ────────────────────────────────────────────────────
        elif model == "groq":
            try:
                async for chunk in groq_.generate_answer_stream(question, context_chunks):
                    yield ("groq", chunk)
            except Exception as e:
                logger.error("generate_answer_stream Groq failed: %r", e)
                yield ("error", f"⚠️ Groq API error: {repr(e)[:200]}")

        # ── Explicit: NVIDIA ──────────────────────────────────────────────────
        elif model in ("nvidia", "nemotron"):
            try:
                async for chunk in nvidia.generate_answer_stream(question, context_chunks):
                    yield ("nvidia" if model == "nvidia" else "nemotron", chunk)
            except Exception as e:
                logger.error("generate_answer_stream NVIDIA/Nemotron failed: %r", e)
                yield ("error", f"⚠️ NVIDIA NIM error: {repr(e)[:200]}")

        # ── Auto: Gemini → Groq → NVIDIA ──────────────────────────────────────
        else:
            gemini_err = None
            # 1️⃣ Try Gemini
            try:
                gen = gemini.generate_answer_stream(question, context_chunks)
                try:
                    first_chunk = await asyncio.wait_for(anext(gen), timeout=15.0)
                except StopAsyncIteration:
                    return
                yield ("gemini-2.5-flash", first_chunk)
                async for chunk in gen:
                    yield ("gemini-2.5-flash", chunk)
                return
            except Exception as e:
                gemini_err = e
                logger.warning("auto: Gemini failed (%s), trying Groq", e)

            # 2️⃣ Try Groq
            try:
                async for chunk in groq_.generate_answer_stream(question, context_chunks):
                    yield ("groq", chunk)
                return
            except Exception as groq_err:
                logger.warning("auto: Groq failed (%s), trying NVIDIA", groq_err)

            # 3️⃣ Try NVIDIA
            try:
                async for chunk in nvidia.generate_answer_stream(question, context_chunks):
                    yield ("nvidia", chunk)
            except Exception as nvidia_err:
                logger.error(
                    "auto: all models failed. Gemini=%s | Groq=%s | NVIDIA=%s",
                    gemini_err, groq_err, nvidia_err,
                )
                yield ("error", _build_both_failed_msg(gemini_err, nvidia_err))



    # ── Chat title generation ─────────────────────────────────────────────────

    @staticmethod
    async def generate_chat_title(message: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await asyncio.wait_for(gemini.generate_chat_title(message), timeout=15.0)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, asyncio.TimeoutError):
                    try: return await groq_.generate_chat_title(message)
                    except Exception:
                        try: return await nvidia.generate_chat_title(message)
                        except Exception: return message[:40]
                raise e

        elif model == "groq":
            try: return await groq_.generate_chat_title(message)
            except Exception: return message[:40]

        elif model == "nvidia":
            try: return await nvidia.generate_chat_title(message)
            except Exception: return message[:40]

        else:  # auto
            try: return await asyncio.wait_for(gemini.generate_chat_title(message), timeout=15.0)
            except Exception: pass
            
            try: return await groq_.generate_chat_title(message)
            except Exception: pass
            
            try: return await nvidia.generate_chat_title(message)
            except Exception: pass
            
            return message[:40]

    # ── Route selection ───────────────────────────────────────────────────────

    @staticmethod
    async def select_route(prompt: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await asyncio.wait_for(gemini.select_route(prompt), timeout=15.0)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, asyncio.TimeoutError):
                    try: return await groq_.select_route(prompt)
                    except Exception:
                        try: return await nvidia.select_route(prompt)
                        except Exception: return "document"
                raise e

        elif model == "groq":
            try: return await groq_.select_route(prompt)
            except Exception: return "document"

        elif model == "nvidia":
            try: return await nvidia.select_route(prompt)
            except Exception: return "document"

        else:  # auto
            try: return await asyncio.wait_for(gemini.select_route(prompt), timeout=15.0)
            except Exception: pass
            
            try: return await groq_.select_route(prompt)
            except Exception: pass
            
            try: return await nvidia.select_route(prompt)
            except Exception: pass
            
            return "document"

    # ── General text generation ───────────────────────────────────────────────

    @staticmethod
    async def generate_text(prompt: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await asyncio.wait_for(gemini.generate_text(prompt), timeout=15.0)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, asyncio.TimeoutError):
                    try: return await groq_.generate_text(prompt)
                    except Exception:
                        try: return await nvidia.generate_text(prompt)
                        except Exception: return ""
                raise e

        elif model == "groq":
            try: return await groq_.generate_text(prompt)
            except Exception: return ""

        elif model == "nvidia":
            try: return await nvidia.generate_text(prompt)
            except Exception: return ""

        else:  # auto
            try: return await asyncio.wait_for(gemini.generate_text(prompt), timeout=15.0)
            except Exception: pass
            
            try: return await groq_.generate_text(prompt)
            except Exception: pass
            
            try: return await nvidia.generate_text(prompt)
            except Exception: pass
            
            return ""