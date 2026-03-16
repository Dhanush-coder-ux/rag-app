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

    async def run(self, question: str) -> AgentState:
        """Non-streaming: run the full graph and return the final state."""
        graph = build_rag_graph(self._db)
        return await graph.ainvoke(self._make_initial_state(question))

    async def stream(self, question: str) -> AsyncGenerator[str, None]:
        """
        Streaming strategy:
          1. Run the graph WITHOUT the generator node to get ranked context.
          2. Stream the answer from generate_answer_stream token by token.
          3. Yield each token as an SSE event.
        """
        graph = build_rag_graph(self._db)
        initial = self._make_initial_state(question)

        # Step 1 — run graph up to (but not including) the generator node
        # The graph ends at END after generator, so we use astream and
        # capture the state right after the reranker node.
        final_state: AgentState = initial

        async for state_snapshot in graph.astream(
            initial,
            stream_mode="values",   # yields full state after each node
        ):
            final_state = state_snapshot
            # Stop consuming once we have context (reranker has run)
            # generator_node will be skipped — we stream it ourselves below
            if "reranker" in (state_snapshot.get("steps") or []):
                break

        context = final_state.get("context", [])

        # Step 2 — stream the answer directly
        if not context:
            yield "data: I could not find relevant information to answer your question.\n\n"
            yield "data: [DONE]\n\n"
            return

        async for chunk in generate_answer_stream(
            final_state["question"], context
        ):
            if chunk:
                # Escape any newlines inside a chunk so SSE stays valid
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"

        yield "data: [DONE]\n\n"
