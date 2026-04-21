from typing import List, Optional

from pydantic import BaseModel, Field


class ProcessResult(BaseModel):
    document_id: int
    process_status: str
    chunk_count: int
    output_dir: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 5
    document_id: Optional[int] = None


class SearchHit(BaseModel):
    score: float
    document_id: int
    document_name: str
    chunk_id: int
    chunk_index: int
    page: Optional[int] = None
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]


class ChunkItem(BaseModel):
    chunk_index: int
    page: Optional[int] = None
    text: str


class ChunkListResponse(BaseModel):
    document_id: int
    chunks: List[ChunkItem]
