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
    tool: str


class LangGraphServices:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def retrieve_tool(self, question: str):
        service = DocumentService(db=self.db)
        chunks = await service.similarity_search(question)
        return [chunk.content for chunk in chunks]

 
    async def tavily_tool(self, question: str):
        response = await asyncio.to_thread(
            tavily.search,
            query=question
        )

        return [r["content"] for r in response["results"]]


    async def tool_node(self, state: AgentState):
        tool = state.get("tool")
        question = state["question"]

        if tool == "retrieve_docs":
            context = await self.retrieve_tool(question)
        elif tool == "tavily_search":
            context = await self.tavily_tool(question)
        else:
            context = []

        return {
            "context": context,
            "steps": state.get("steps", []) + [tool]
        }

    async def agent_node(self, state: AgentState):

        prompt = f"""
            You are an AI assistant.
            User Question:
            {state["question"]}
            Available tools:
            1. retrieve_docs
            2. tavily_search
            Rules:
            - If the question is about internal documents -> retrieve_docs
            - If the question requires internet knowledge -> tavily_search
            Respond ONLY with the tool name.
            """
        decision = (await generate_answer(prompt, [])).strip()
        return {
            "tool": decision,
            "steps": state.get("steps", []) + ["agent_decision"]
        }

 
    async def generate_node(self, state: AgentState):
        answer = await generate_answer(
            state["question"],
            state["context"]
        )

        return {
            "answer": answer,
            "steps": state.get("steps", []) + ["generated_answer"]
        }

    def create_rag_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tool", self.tool_node)
        workflow.add_node("generate", self.generate_node)

        workflow.set_entry_point("agent")

        workflow.add_edge("agent", "tool")
        workflow.add_edge("tool", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()