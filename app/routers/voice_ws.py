import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession
import edge_tts
from app.core.config import settings
from app.core.database import get_db
from app.RagPipeline.service import LangGraphService
from app.rag_services.chat_service import ChatServices


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["voice"])

async def transcribe_audio(audio_bytes: bytes) -> str:
    """Uses Groq Whisper to transcribe audio bytes."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    # Provide a filename with a supported extension so the API knows the format
    file = ("audio.webm", audio_bytes, "audio/webm")
    response = await client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3",
        response_format="json"
    )
    return response.text

async def generate_tts_and_send(text: str, websocket: WebSocket):
    """Generates TTS using edge-tts and streams MP3 bytes to the WebSocket."""
    if not text.strip(): 
        return
    
    # en-US-AriaNeural is a high quality female voice, en-US-ChristopherNeural for male.
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    try:
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if audio_data:
            await websocket.send_bytes(bytes(audio_data))
    except Exception as e:
        logger.error(f"TTS generation error for text '{text}': {e}")


@router.websocket("/voice")
async def voice_websocket(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    svc = LangGraphService(db=db)
    chat_service = ChatServices(db)
    
    # 1. Receive initial configuration
    try:
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        session_id = config.get("session_id")
        mode = config.get("mode", "hybrid")
        model = config.get("model", "auto")
        document_ids = config.get("document_ids", [])
        history = config.get("history", [])
    except Exception as e:
        logger.error(f"Error receiving config: {e}")
        await websocket.close()
        return

    # 2. Receive audio chunks
    audio_buffer = bytearray()
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
            elif "text" in message:
                if message["text"] == "EOF":
                    break
    except WebSocketDisconnect:
        logger.info("Client disconnected during audio recording")
        return
        
    if not audio_buffer:
        await websocket.send_text(json.dumps({"type": "error", "message": "No audio received"}))
        await websocket.close()
        return

    # 3. Transcribe audio
    try:
        await websocket.send_text(json.dumps({"type": "status", "message": "Transcribing..."}))
        question = await transcribe_audio(bytes(audio_buffer))
        await websocket.send_text(json.dumps({"type": "transcription", "text": question}))
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await websocket.send_text(json.dumps({"type": "error", "message": f"Transcription failed: {str(e)}"}))
        await websocket.close()
        return

    if not question.strip():
        await websocket.send_text(json.dumps({"type": "error", "message": "Could not understand audio. Please try again."}))
        await websocket.close()
        return

    # Save user message to session
    if not session_id:
        new_session = await chat_service.create_chat_session()
        session_id = new_session.id
    await chat_service.save_user_message(session_id=session_id, content=question)

    await websocket.send_text(json.dumps({"type": "session_id", "session_id": session_id}))
    
    # 4. Stream RAG response and generate TTS chunks
    sentence_buffer = ""
    full_response = ""
    
    try:
        async for chunk_str in svc.stream(
            question=question,
            history=history,
            mode=mode,
            model=model,
            document_ids=document_ids
        ):
            # Send the SSE-like chunk back to the client so the UI can update live
            await websocket.send_text(json.dumps({"type": "rag_chunk", "chunk": chunk_str}))
            
            # Extract text to build sentences for TTS
            if "data:" in chunk_str and not chunk_str.startswith("event:"):
                payload = chunk_str.split("data:", 1)[1].strip()
                if payload and payload != "[DONE]":
                    try:
                        parsed = json.loads(payload)
                        text = parsed.get("answer", "") if isinstance(parsed, dict) else (parsed if isinstance(parsed, str) else "")
                    except:
                        text = ""
                    
                    if text:
                        full_response += text
                        sentence_buffer += text
                        
                        # Sentence boundary detection to stream audio ASAP
                        # We look for punctuation that ends a sentence
                        if any(punc in sentence_buffer for punc in ['.', '!', '?', '\n']):
                            import re
                            # Split by sentence endings (. ! ?) followed by whitespace, or newlines
                            parts = re.split(r'(?<=[.!?])\s+|\n+', sentence_buffer)
                            
                            # All but the last part are guaranteed to be complete sentences
                            if len(parts) > 1:
                                for i in range(len(parts) - 1):
                                    sentence = parts[i].strip()
                                    if len(sentence) > 1:
                                        await generate_tts_and_send(sentence, websocket)
                                sentence_buffer = parts[-1]
                            
        # Process any remaining text in the buffer
        if sentence_buffer.strip():
            await generate_tts_and_send(sentence_buffer.strip(), websocket)

        # Save assistant message
        if full_response and session_id:
            await chat_service.save_assistant_message(session_id=session_id, content=full_response)
            await chat_service.update_title_if_needed(session_id=session_id, message=question)

        # Notify frontend that all text and audio chunks have been sent
        await websocket.send_text(json.dumps({"type": "end_of_audio"}))
        
    except WebSocketDisconnect:
        logger.info("Client disconnected during RAG generation")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Generation failed: {str(e)}"}))
        except:
            pass
    
    finally:
        try:
            await websocket.close()
        except:
            pass
