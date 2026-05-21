from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int | None = None
