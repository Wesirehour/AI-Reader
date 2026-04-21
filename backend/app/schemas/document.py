from datetime import datetime

from pydantic import BaseModel
from typing import Optional


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_url: str
    file_hash: str
    markdown_path: str
    markdown_url: str
    process_status: str
    process_error: str
    chunk_count: int
    processed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True
