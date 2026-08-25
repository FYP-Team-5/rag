from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, Field, field_validator

from app.model import CourseMaterial, CourseMaterialStatus


class PresignedUrlRequest(BaseModel):
    course_id: UUID4
    filename: str = Field(min_length=1, max_length=512)

    @field_validator("filename")
    @classmethod
    def filename_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("filename must not be blank")
        return value


class PresignedUrlResponse(BaseModel):
    course_material_id: UUID
    presigned_url: str
    object_key: str
    expires_in: int


class UploadStatusRequest(BaseModel):
    status: Literal["uploaded", "failed", "upload_failed"]


class UploadStatusResponse(BaseModel):
    course_material_id: UUID
    status: Literal["processing", "completed", "deleted"]


class CourseMaterialProcessingStatus(BaseModel):
    id: UUID
    status: CourseMaterialStatus


class CourseMaterialList(BaseModel):
    total: int
    items: list[CourseMaterial]
