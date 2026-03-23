import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.RagPipeline.graph import build_rag_graph
from app.RagPipeline.state import AgentState
from app.rag_services.gemini import generate_answer_stream


class LangGraphService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self.graph = build_rag_graph(db)

    def _make_initial_state(self, question: str) -> AgentState:
        return {
            "question": question,
            "context": [],
            "answer": "",
            "steps": [],
            "tool": "none",
            "error": None,
            "trace_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def run(self, question: str):
        initial = self._make_initial_state(question)
        initial["history"] = []

        result = await self.graph.ainvoke(initial)

        # 🔥 preserve metadata (VERY IMPORTANT)
        result["trace_id"] = initial["trace_id"]
        result["created_at"] = initial["created_at"]

        return result

    async def stream(self, question: str) -> AsyncGenerator[str, None]:

        initial = self._make_initial_state(question)
        initial["history"] = []

        final_state = initial
        last_step = None

        # 🔥 send trace id
        yield f"event: trace\ndata: {initial['trace_id']}\n\n"

        try:
            async for state_snapshot in self.graph.astream(
                initial,
                stream_mode="values",
            ):
                final_state = state_snapshot

                steps = state_snapshot.get("steps") or []
                step = steps[-1] if steps else None

                if step and step != last_step:
                    yield f"event: step\ndata: {step}\n\n"
                    last_step = step

                if "reranker" in steps:
                    break

            context = final_state.get("context", [])

            if not context:
                yield "data: No relevant info found\n\n"
                yield "data: [DONE]\n\n"
                return

            sources = final_state.get("sources", [])
            if sources:
                yield f"event: sources\ndata: {sources}\n\n"

            query = final_state.get("rewritten_question", final_state["question"])

            async for chunk in generate_answer_stream(query, context):
                if chunk:
                    import json
                    yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

        yield "data: [DONE]\n\n"