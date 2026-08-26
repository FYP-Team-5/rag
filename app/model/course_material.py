from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel


CourseMaterialStatus = Literal[
    "awaiting_upload",
    "processing",
    "completed",
    "failed",
]


class CourseMaterial(BaseModel):
    id: UUID4
    course_id: UUID
    filename: str
    status: CourseMaterialStatus
    s3_bucket: str
    s3_object_key: str
