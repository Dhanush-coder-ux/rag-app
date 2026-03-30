from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.RagPipeline.graph import build_rag_graph
from app.RagPipeline.state import AgentState, RagMode
from app.rag_services.gemini import generate_answer_stream
from app.schemas.rag import SourceItem

logger = logging.getLogger(__name__)



def _enrich_sources(raw_sources: List[str], context: List[str]) -> List[SourceItem]:

    items: List[SourceItem] = []

    for url in raw_sources:
        snippet = _find_snippet_for_url(url, context)
        items.append(SourceItem(
            url=url,
            title=_url_to_title(url),
            snippet=snippet,
            source_type="web",
        ))

    for chunk in context:
        if chunk.startswith("[Source: YOUR-DOCUMENT]"):
            text = chunk.replace("[Source: YOUR-DOCUMENT]", "").strip()
            items.append(SourceItem(
                url=None,
                title="Your Document",
                snippet=text[:200] + "..." if len(text) > 200 else text,
                source_type="document",
            ))

    seen = set()
    unique = []
    for item in items:
        key = item.url or item.title
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _find_snippet_for_url(url: str, context: List[str]) -> str | None:
    for chunk in context:
        if url in chunk:
            text = re.sub(rf"\[Source: {re.escape(url)}\]\s*", "", chunk).strip()
            return text[:200] + "..." if len(text) > 200 else text
    return None


def _url_to_title(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path   = parsed.path.rstrip("/").split("/")[-1]
        name   = path.replace("-", " ").replace("_", " ").title() if path else parsed.netloc
        return name or parsed.netloc
    except Exception:
        return url




class LangGraphService:
    def __init__(self, db: AsyncSession) -> None:
        self._db   = db
        self.graph = build_rag_graph(db)
    def _make_initial_state(self, question: str, mode: RagMode = "hybrid") -> AgentState:
        return {
            "question":   question,
            "mode":       mode,             
            "answer":     "",
            "steps":      [],
            "tool":       "none",
            "sources":    [],
            "error":      None,
            "trace_id":   str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence": None,
        }



    async def run(
        self,
        question: str,
        history:  list[dict] | None = None,
        mode:     RagMode           = "hybrid",
    ) -> dict:
        initial           = self._make_initial_state(question, mode=mode)
        initial["history"] = history or []

        logger.info(
            "LangGraphService.run trace_id=%s mode=%s question=%r",
            initial["trace_id"], mode, question,
        )

        result  = await self.graph.ainvoke(initial)
        result["trace_id"] = initial["trace_id"]
        result["created_at"] = initial["created_at"]
        result["sources"] = _enrich_sources(
            result.get("sources", []),
            result.get("context", []),
        )
        result["tool_used"] = _resolve_tool_used(mode, result.get("tool_used"))

        result["history"] = list(initial["history"]) + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": result.get("answer", "")},
        ]

        return result



    async def stream(
        self,
        question: str,
        history:  list[dict] | None = None,
        mode:     RagMode           = "hybrid",
    ) -> AsyncGenerator[str, None]:

        initial            = self._make_initial_state(question, mode=mode)
        initial["history"] = history or []

        logger.info(
            "LangGraphService.stream trace_id=%s mode=%s question=%r",
            initial["trace_id"], mode, question,
        )

        yield f"event: trace\ndata: {initial['trace_id']}\n\n"
        yield f"event: mode\ndata: {mode}\n\n"

        final_state = initial
        last_step   = None

        try:
            async for snapshot in self.graph.astream(initial, stream_mode="values"):
                final_state = snapshot

                steps = snapshot.get("steps") or []
                step  = steps[-1] if steps else None

                if step and step != last_step:
                    yield f"event: step\ndata: {json.dumps(step)}\n\n"
                    last_step = step

                if "reranker" in (snapshot.get("steps") or []):
                    break

            context = final_state.get("context", [])

            if not context:
                yield "data: No relevant information found.\n\n"
                yield "data: [DONE]\n\n"
                return

            rich_sources = _enrich_sources(
                final_state.get("sources", []),
                context,
            )
            if rich_sources:
                yield (
                    f"event: sources\n"
                    f"data: {json.dumps([s.model_dump() for s in rich_sources])}\n\n"
                )

            tool_used = _resolve_tool_used(mode, final_state.get("tool_used"))
            yield f"event: tool_used\ndata: {tool_used}\n\n"

            generate_label = {
                "documents": "🧠 Generating answer...",
                "web":       "🧠 Generating answer...",
                "hybrid":    "✍️ Generating answer...",
            }.get(mode, "🧠 Generating answer...")
            yield f"event: step\ndata: {json.dumps(generate_label)}\n\n"

            query = final_state.get("rewritten_question", final_state["question"])
            async for chunk in generate_answer_stream(query, context):
                if chunk:
                    yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as exc:
            logger.exception("stream error trace_id=%s: %s", initial["trace_id"], exc)
            yield f"data: {json.dumps(f'Error: {str(exc)}')}\n\n"

        yield "data: [DONE]\n\n"



def _resolve_tool_used(
    mode: RagMode,
    state_tool_used: str | None,
) -> str:
    """
    Derive the schema-compatible tool_used value.
    Mode always wins; state value is a fallback for 'auto' / edge cases.
    """
    _map = {
        "documents": "document",
        "web":       "web",
        "hybrid":    "hybrid",
    }
    return _map.get(mode, state_tool_used or "none")