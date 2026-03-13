from typing import TypedDict, List
from app.rag_services.document_service import DocumentService
from app.rag_services.gemini import generate_answer
from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession
from tavily import TavilyClient
import asyncio
from app.core.config import settings

tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    steps: List[str] 
    has_context: bool




async def tavily_search_node(state: AgentState):

    response = await asyncio.to_thread(
        tavily.search,
        query=state["question"]
    )

    results = [r["content"] for r in response["results"]]

    return {
        "context": results,
        "steps": state.get("steps", []) + ["tavily_search"]
    }

async def retrieve_node(state: AgentState, db):

    service = DocumentService(db=db)
    chunks = await service.similarity_search(state["question"])
    context_text = [chunk.content for chunk in chunks]
    return {
        "context": context_text,
        "has_context":len(context_text)>0,
        "steps": state.get("steps", []) + ["retrieved_docs"]
    }
def check_context(state: AgentState):

    context = state.get("context", [])

    if not context:
        return "tavily_search"

    question = state["question"].lower()

    if any(word in context[0].lower() for word in question.split()):
        return "generate"

    return "tavily_search"

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
    workflow.add_node("tavily_search", tavily_search_node)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        check_context,
        {
            "generate": "generate",
            "tavily_search": "tavily_search",
        },
    )

    workflow.add_edge("tavily_search", "generate")

    workflow.add_edge("generate", END)

    return workflow.compile()