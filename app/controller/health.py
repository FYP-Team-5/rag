from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.controller.dependencies import get_course_material_service
from app.dto import HealthResponse
from app.service import CourseMaterialService

health_router = APIRouter(tags=["health"])


@health_router.get("/health", response_model=HealthResponse)
async def health(
    service: Annotated[CourseMaterialService, Depends(get_course_material_service)],
) -> HealthResponse:
    components = await service.health()
    if not all(components.values()):
        raise HTTPException(
            status_code=503,
            detail={
                name: "ok" if available else "unavailable"
                for name, available in components.items()
            },
        )
    return HealthResponse(
        status="ok",
        qdrant="ok",
        postgres="ok",
        s3="ok",
        collection=service.settings.qdrant_collection,
    )
