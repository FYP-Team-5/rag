from __future__ import annotations

from typing import Literal

from pydantic import UUID4, BaseModel


CourseMaterialStatus = Literal[
    "awaiting_upload",
    "processing",
    "completed",
    "failed",
]


class CourseMaterial(BaseModel):
    id: UUID4
    course_id: UUID4
    filename: str
    status: CourseMaterialStatus
    s3_bucket: str
    s3_object_key: str
