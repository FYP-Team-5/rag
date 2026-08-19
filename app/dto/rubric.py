from typing import Any, Literal

from pydantic import BaseModel

from app.model.rubric import Rubric


class RubricList(BaseModel):
    total: int
    items: list[Rubric]

class RubricProcessingStatus(BaseModel):
    id: str
    document_id: str
    processed: bool
    processing_status: Literal["processing", "completed", "failed"]
    processing_error: str | None = None
    chunk_count: int

class RubricChunk(BaseModel):
    content: str
    metadata: dict[str, Any]

class RubricChunksResponse(BaseModel):
    rubric_id: str
    chunks: list[RubricChunk]

class ArchiveResponse(BaseModel):
    id: str
    archived: bool
