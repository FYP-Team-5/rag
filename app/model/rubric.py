from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Rubric(BaseModel):
    id: str
    title: str
    version: str
    course_id: str | None = None
    exam_id: str | None = None
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    chunk_count: int
    processed: bool = False
    processing_status: Literal["processing", "completed", "failed"] = "processing"
    processing_error: str | None = None
    archived: bool = False
    uploaded_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredRubric(Rubric):
    document_id: str
    s3_bucket: str
    s3_object_key: str
    chunk_ids: list[str]

    def public(self) -> Rubric:
        return Rubric.model_validate(self.model_dump())

