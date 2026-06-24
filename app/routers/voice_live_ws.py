"""
Live Voice WebSocket — Continuous conversation mode.

Pipeline per turn:
  Groq Whisper-large-v3 (STT)
  → RAG retrieve_context() (BGE-M3 embeddings)
  → Groq llama-3.3-70b-versatile (LLM  — fast, no NVIDIA model access issues)
  → edge-tts AriaNeural (TTS)

WebM FIX:
  MediaRecorder emits the EBML/Segment/Tracks header ONLY in the very first
  binary chunk.  We save that first chunk as `webm_init` and prepend it to
  every subsequent utterance buffer so Groq Whisper always gets a valid file.

  The frontend uses a SINGLE continuous MediaRecorder for the whole session
  (never stops/restarts it).  The backend uses a per-utterance audio_buffer
  that is flushed on each utterance_end signal.
"""

import json
import logging
import re
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from groq import AsyncGroq
import edge_tts

from app.core.config import settings
from app.core.database import get_db
from app.RagPipeline.service import LangGraphService
from app.rag_services.chat_service import ChatServices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["voice-live"])

# Minimum bytes before we send to Whisper.
# A WebM EBML header alone is ~2KB; we need some audio data on top.
MIN_AUDIO_BYTES = 2_000

from app.llms.llm_routers import gemini, groq_, nvidia

async def voice_llm_stream(question: str, context_chunks: list[str], model_pref: str = "auto"):
    """
    Stream a concise, spoken-language response from the selected LLM.
    Yields text tokens.
    """
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no document context)"
    system_prompt = (
        "You are a helpful, conversational voice assistant. "
        "Use the provided context to answer the user's question concisely. "
        "Respond in natural spoken language — no markdown, bullet points, or code. "
        "Keep your answer to 2-4 sentences unless more detail is truly needed. "
        "If the answer is not in the context say: "
        "\"I don't have that information in the documents.\""
    )
    
    # We will try the preferred model first. If it fails (e.g. rate limit), we fallback to another.
    models_to_try = []
    if model_pref == "nvidia":
        models_to_try = ["nvidia", "nemotron", "groq", "gemini"]
    elif model_pref == "groq":
        models_to_try = ["groq", "nemotron", "nvidia", "gemini"]
    elif model_pref == "gemini":
        models_to_try = ["gemini", "nemotron", "nvidia", "groq"]
    else:
        # Auto: Prefer Nemotron (100k daily limits), then NVIDIA, then Groq, then Gemini
        models_to_try = ["nemotron", "nvidia", "groq", "gemini"]

    last_err = None
    for m in models_to_try:
        try:
            if m == "nemotron":
                from openai import AsyncOpenAI
                # Nemotron 70B is robust and effectively unlimited on NIM
                client = AsyncOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=settings.NEMOTRON_API_KEY,
                    max_retries=0, # Fail fast, do not retry
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ]
                stream = await client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-70b-instruct",
                    messages=messages,
                    temperature=0.65,
                    max_tokens=512,
                    stream=True,
                    timeout=7.0
                )
                async for chunk in stream:
                    if chunk.choices:
                        token = chunk.choices[0].delta.content or ""
                        if token: yield token
                return # Success!

            elif m == "nvidia":
                # Override the global nvidia client retries for live voice to ensure fast fallback
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=settings.NVIDIA_API_KEY,
                    max_retries=0, # Fail fast
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ]
                stream = await client.chat.completions.create(
                    model=nvidia.model,
                    messages=messages,
                    temperature=0.65,
                    max_tokens=512,
                    stream=True,
                    timeout=7.0
                )
                async for chunk in stream:
                    if chunk.choices:
                        token = chunk.choices[0].delta.content or ""
                        if token: yield token
                return # Success!

            elif m == "groq":
                client = AsyncGroq(api_key=settings.GROQ_API_KEY, timeout=7.0, max_retries=0)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
                ]
                stream = await client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    temperature=0.65,
                    max_tokens=512,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices:
                        token = chunk.choices[0].delta.content or ""
                        if token: yield token
                return # Success!
                
            elif m == "gemini":
                # Gemini uses its own SDK
                from app.llms.gemini.generation import client as gemini_client
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                
                # generate_content_stream returns an async generator immediately, do NOT await it
                stream = gemini_client.aio.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                # Wait for at least one chunk to catch errors like 429
                async for chunk in stream:
                    if chunk.text: yield chunk.text
                return # Success!

        except Exception as e:
            logger.warning(f"Voice LLM ({m}) failed: {e}. Trying next...")
            last_err = e
            continue

    # If all failed, yield the error so it gets spoken/displayed
    yield f"I'm sorry, my language models are currently unavailable. Error: {last_err}"


# ── Groq STT ──────────────────────────────────────────────────────────────────

async def transcribe_audio(audio_bytes: bytes) -> str:
    """Groq Whisper-large-v3 STT. Returns empty string if nothing recognised."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    response = await client.audio.transcriptions.create(
        file=("utterance.webm", audio_bytes, "audio/webm"),
        model="whisper-large-v3",
        response_format="json",
    )
    return (response.text or "").strip()


# ── TTS ───────────────────────────────────────────────────────────────────────

async def tts_sentence(text: str, websocket: WebSocket) -> None:
    """Generate TTS for one sentence and stream MP3 bytes back."""
    text = text.strip()
    if not text:
        return
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    try:
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if audio_data:
            await websocket.send_bytes(bytes(audio_data))
    except Exception as exc:
        logger.error("TTS error for '%s...': %s", text[:30], exc)


# ── LLM + TTS combined ────────────────────────────────────────────────────────

async def generate_and_speak(
    question: str,
    context_chunks: list[str],
    websocket: WebSocket,
    model_pref: str,
) -> str:
    """
    Stream Groq LLM response token-by-token → send each token to frontend
    → fire TTS at sentence boundaries for minimal latency.
    Returns the full response text.
    """
    full_response = ""
    sentence_buffer = ""

    async for token in voice_llm_stream(question, context_chunks, model_pref):
        full_response    += token
        sentence_buffer  += token

        # Live text streaming to frontend
        await websocket.send_text(json.dumps({"type": "response_chunk", "text": token}))

        # TTS at sentence boundaries
        if any(p in sentence_buffer for p in (".", "!", "?", "\n")):
            parts = re.split(r"(?<=[.!?])\s+|\n+", sentence_buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    await tts_sentence(sentence, websocket)
                sentence_buffer = parts[-1]

    # Flush remaining partial sentence
    if sentence_buffer.strip():
        await tts_sentence(sentence_buffer, websocket)

    return full_response


# ── WebSocket handler ─────────────────────────────────────────────────────────

@router.websocket("/live")
async def live_voice_websocket(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
):
    """
    Persistent WebSocket for continuous multi-turn voice conversation.

    Client → Server:
      1. JSON config: { session_id, mode, model, document_ids, history }
      2. Binary: raw audio/webm chunks (continuous from a SINGLE MediaRecorder)
      3. JSON: { "type": "utterance_end" }  — VAD silence detected
      4. JSON: { "type": "abort" }           — session ended

    Server → Client:
      { "type": "ready" }
      { "type": "session_id", "session_id": N }
      { "type": "status",      "message": "..." }
      { "type": "transcript",  "text": "..." }
      { "type": "response_chunk", "text": "..." }
      Binary: TTS audio (MP3)
      { "type": "turn_end" }   — AI done, resume listening
      { "type": "error",       "message": "..." }
    """
    await websocket.accept()
    svc          = LangGraphService(db=db)
    chat_service = ChatServices(db)

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        cfg = json.loads(raw)
    except asyncio.TimeoutError:
        await websocket.close(code=1008)
        return
    except Exception as exc:
        logger.error("Config error: %s", exc)
        await websocket.close(code=1008)
        return

    session_id:   int | None    = cfg.get("session_id")
    mode:         str           = cfg.get("mode", "documents")
    model:        str           = cfg.get("model", "auto")
    document_ids: list[int]     = cfg.get("document_ids") or []
    history:      list[dict]    = cfg.get("history") or []

    if not session_id:
        try:
            session_id = (await chat_service.create_chat_session()).id
        except Exception as exc:
            logger.error("Session create error: %s", exc)

    await websocket.send_text(json.dumps({"type": "session_id", "session_id": session_id}))
    await websocket.send_text(json.dumps({"type": "ready"}))

    # ── Conversation loop ─────────────────────────────────────────────────────
    #
    # webm_init: first binary chunk from the browser's MediaRecorder.
    # Contains EBML header + Segment header + Tracks — the "file header" of
    # the WebM stream.  We prepend it to every turn 2+ buffer so Whisper can
    # decode each utterance as an independent valid WebM file.
    #
    # The frontend uses ONE continuous MediaRecorder (never stopped/restarted).
    # This guarantees webm_init is correct and stable for the whole session.
    #
    webm_init:    bytes | None  = None
    audio_buffer: bytearray     = bytearray()
    turn_count:   int           = 0

    try:
        while True:
            msg = await websocket.receive()

            # ── Binary: audio chunk ───────────────────────────────────────────
            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                if webm_init is None:
                    # Save the init segment (ONLY the very first chunk ever)
                    webm_init = chunk
                    logger.debug("WebM init saved (%d bytes)", len(webm_init))
                audio_buffer.extend(chunk)

            # ── Text: control ─────────────────────────────────────────────────
            elif "text" in msg:
                try:
                    ctrl = json.loads(msg["text"])
                except Exception:
                    continue

                if ctrl.get("type") == "abort":
                    break

                if ctrl.get("type") != "utterance_end":
                    continue

                # ── Process utterance ─────────────────────────────────────────
                raw_buf       = bytes(audio_buffer)
                audio_buffer  = bytearray()   # Reset for next turn

                if len(raw_buf) < MIN_AUDIO_BYTES:
                    logger.debug("Utterance too short (%d B), skipping", len(raw_buf))
                    await websocket.send_text(json.dumps({
                        "type": "status", "message": "Listening..."
                    }))
                    continue

                # Build a valid standalone WebM file
                if turn_count == 0:
                    # First turn: raw_buf already starts with the init segment
                    utterance_audio = raw_buf
                else:
                    # Turns 2+: prepend the saved init segment
                    utterance_audio = (webm_init or b"") + raw_buf

                turn_count += 1

                # ── STT ───────────────────────────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type": "status", "message": "Transcribing..."
                }))
                try:
                    question = await transcribe_audio(utterance_audio)
                except Exception as exc:
                    logger.error("STT error turn %d: %s", turn_count, exc)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Transcription failed: {exc}"
                    }))
                    await websocket.send_text(json.dumps({"type": "turn_end"}))
                    continue

                if not question:
                    await websocket.send_text(json.dumps({
                        "type": "status", "message": "Didn't catch that, try again..."
                    }))
                    await websocket.send_text(json.dumps({"type": "turn_end"}))
                    continue

                await websocket.send_text(json.dumps({"type": "transcript", "text": question}))
                if session_id:
                    try:
                        await chat_service.save_user_message(session_id=session_id, content=question)
                    except Exception:
                        pass

                # ── RAG retrieval ─────────────────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type": "status", "message": "Searching knowledge base..."
                }))
                context_chunks: list[str] = []
                try:
                    context_chunks = await svc.retrieve_context(
                        question=question,
                        mode=mode,
                        document_ids=document_ids or None,
                    )
                    logger.info("Retrieved %d chunks (turn %d)", len(context_chunks), turn_count)
                except Exception as exc:
                    logger.warning("RAG retrieval skipped: %s", exc)

                # ── LLM + TTS ─────────────────────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type": "status", "message": "Generating response..."
                }))
                try:
                    full_response = await generate_and_speak(question, context_chunks, websocket, model)
                except Exception as exc:
                    logger.error("LLM/TTS error turn %d: %s", turn_count, exc)
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": f"Response failed: {exc}"
                    }))
                    await websocket.send_text(json.dumps({"type": "turn_end"}))
                    continue

                # ── Persist ───────────────────────────────────────────────────
                if full_response and session_id:
                    try:
                        await chat_service.save_assistant_message(session_id=session_id, content=full_response)
                        await chat_service.update_title_if_needed(session_id=session_id, message=question)
                        history = history + [
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": full_response},
                        ]
                    except Exception:
                        pass

                # ── Turn done → frontend resumes listening ────────────────────
                await websocket.send_text(json.dumps({"type": "turn_end"}))
                logger.info("Turn %d done. q=%r resp_len=%d", turn_count, question[:60], len(full_response))

    except WebSocketDisconnect:
        logger.info("Client disconnected after %d turns", turn_count)
    except RuntimeError as exc:
        if "Cannot call" in str(exc) or "Unexpected ASGI" in str(exc):
            logger.info("Client disconnected abruptly (RuntimeError) after %d turns", turn_count)
        else:
            logger.exception("Live voice error: %s", exc)
    except Exception as exc:
        logger.exception("Live voice error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
