from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embeddings: str
    collection: str


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
    uploaded_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredRubric(Rubric):
    document_id: str
    storage_path: str
    chunk_ids: list[str]

    def public(self) -> Rubric:
        return Rubric.model_validate(self.model_dump())


class RubricList(BaseModel):
    total: int
    items: list[Rubric]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    rubric_id: str | None = Field(default=None, max_length=128)
    course_id: str | None = Field(default=None, max_length=128)
    k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class RubricChunk(BaseModel):
    content: str
    metadata: dict[str, Any]


class RubricChunksResponse(BaseModel):
    rubric_id: str
    chunks: list[RubricChunk]


class DeleteResponse(BaseModel):
    id: str
    deleted: bool
