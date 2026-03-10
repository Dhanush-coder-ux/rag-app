from typing import TypedDict, List
from app.rag_services.document_service import DocumentService
from app.rag_services.gemini import generate_answer
from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    steps: List[str]

async def retrieve_node(state: AgentState, db):

    service = DocumentService(db=db)
    chunks = await service.similarity_search(state["question"])
    context_text = [chunk.content for chunk in chunks]
    return {
        "context": context_text,
        "steps": state.get("steps", []) + ["retrieved_docs"]
    }

async def generate_node(state: AgentState):

    response = await generate_answer(
        state["question"],
        state["context"]
    )
    return {
        "answer": response,
        "steps": state.get("steps", []) + ["generated_answer"]
    }



def create_rag_graph(db: AsyncSession):
    workflow = StateGraph(AgentState)

    async def retrieve_wrapper(state: AgentState):
        return await retrieve_node(state, db)

    workflow.add_node("retrieve", retrieve_wrapper)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()