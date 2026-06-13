from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

ToolName = Literal["retrieve_node", "web_search", "both", "none"]
RagMode  = Literal["documents", "web", "hybrid"]

LLMModel = Literal["auto", "gemini", "llama3"]


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):

    question: str
    mode: RagMode

  
    model: LLMModel       
    used_model: str          
    document_ids: Optional[List[int]]
 
    context: List[str]
    answer: str
    steps: List[str]
    tool: ToolName
    sources: List[str]

    history: List[HistoryMessage]

    error: Optional[str]
    trace_id: str
    created_at: str
    confidence: Optional[float]