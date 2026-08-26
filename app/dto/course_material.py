from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.model import CourseMaterial, CourseMaterialStatus


class PresignedUrlRequest(BaseModel):
    course_id: UUID
    filename: str = Field(min_length=1, max_length=512)
    file_extension: str = Field(min_length=1, max_length=16, examples=[".md"])

    @field_validator("filename")
    @classmethod
    def filename_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("filename must not be blank")
        return value

    @field_validator("file_extension")
    @classmethod
    def normalize_file_extension(cls, value: str) -> str:
        extension = value.strip().lower().removeprefix(".")
        if not extension or not extension.isalnum():
            raise ValueError("file_extension must contain only letters and numbers")
        return f".{extension}"


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
