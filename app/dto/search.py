from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    course_id: UUID | None = None
    k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
