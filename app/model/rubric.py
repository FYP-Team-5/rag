from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Rubric(BaseModel):
    id: str
    title: str
    version: str
    course_id: str | None = None
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    chunk_count: int
    processed: bool = False
    processing_status: Literal["processing", "completed", "failed"] = "processing"
    processing_error: str | None = None
    uploaded_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredRubric(Rubric):
    document_id: str
    s3_bucket: str
    s3_object_key: str
    chunk_ids: list[str]

    def public(self) -> Rubric:
        return Rubric.model_validate(self.model_dump())


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


class DeleteResponse(BaseModel):
    id: str
    deleted: bool
