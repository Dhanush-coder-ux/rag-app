from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is LangGraph?")
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID for chat memory"
    )


class RagResponse(BaseModel):
    answer: str = Field(..., description="Generated AI answer")
    steps: List[str] = Field(
        default=[],
        description="Pipeline steps executed by the agent"
    )
    tool_used: Optional[str] = Field(
        default=None,
        description="Tool selected by the agent"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Unique request trace id for debugging"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if something failed"
    )