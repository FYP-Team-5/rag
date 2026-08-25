from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.controller.dependencies import get_course_material_service, require_api_key
from app.db import S3StorageError
from app.dto import (
    CourseMaterialList,
    CourseMaterialProcessingStatus,
    PresignedUrlRequest,
    PresignedUrlResponse,
    UploadStatusRequest,
    UploadStatusResponse,
)
from app.model import CourseMaterial
from app.service import (
    CourseMaterialConflictError,
    CourseMaterialNotFoundError,
    CourseMaterialService,
    CourseMaterialTooLargeError,
    InvalidCourseMaterialError,
)

course_material_router = APIRouter(
    prefix="/course-material",
    tags=["course-material"],
    dependencies=[Depends(require_api_key)],
)


@course_material_router.post(
    "/create_presigned",
    response_model=PresignedUrlResponse,
)
async def create_presigned_url(
    body: PresignedUrlRequest,
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
) -> PresignedUrlResponse:
    try:
        return await service.create_presigned_url(body)
    except InvalidCourseMaterialError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CourseMaterialConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except S3StorageError as exc:
        raise HTTPException(
            status_code=502, detail="Object storage request failed."
        ) from exc


@course_material_router.post(
    "/{material_id}/upload-status",
    response_model=UploadStatusResponse,
)
async def report_upload_status(
    material_id: UUID,
    body: UploadStatusRequest,
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
) -> UploadStatusResponse:
    try:
        return await service.report_upload_status(material_id, body.status)
    except CourseMaterialNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Course material not found."
        ) from exc
    except CourseMaterialConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CourseMaterialTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidCourseMaterialError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except S3StorageError as exc:
        raise HTTPException(
            status_code=502, detail="Object storage request failed."
        ) from exc


@course_material_router.get("", response_model=CourseMaterialList)
async def list_course_materials(
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    course_id: UUID | None = None,
) -> CourseMaterialList:
    total, items = await service.list(offset=offset, limit=limit, course_id=course_id)
    return CourseMaterialList(total=total, items=items)


@course_material_router.get("/{material_id}", response_model=CourseMaterial)
async def get_course_material(
    material_id: UUID,
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
) -> CourseMaterial:
    try:
        return await service.get(material_id)
    except CourseMaterialNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Course material not found."
        ) from exc


@course_material_router.get(
    "/{material_id}/status",
    response_model=CourseMaterialProcessingStatus,
)
async def get_processing_status(
    material_id: UUID,
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
) -> CourseMaterialProcessingStatus:
    try:
        return await service.processing_status(material_id)
    except CourseMaterialNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Course material not found."
        ) from exc
