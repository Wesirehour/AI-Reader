from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCitation(BaseModel):
    document_id: int
    document_name: str
    chunk_index: int
    page: Optional[int] = None
    score: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    model: Optional[str] = None
    provider: Optional[str] = None
    use_retrieval: bool = False
    document_id: Optional[int] = None
    top_k: int = 5
    history: List[Union[ChatMessage, Dict[str, str]]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    model: str
    provider: str
    citations: List[ChatCitation] = Field(default_factory=list)


class ModelInfo(BaseModel):
    name: str


class ModelListResponse(BaseModel):
    provider: str
    base_url: str
    default_model: str
    models: List[ModelInfo]
