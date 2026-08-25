"""Shared dependencies used by the HTTP route modules."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from app.service import CourseMaterialService, SearchService


def get_course_material_service(request: Request) -> CourseMaterialService:
    return request.app.state.course_material_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    configured = request.app.state.settings.api_key
    if configured and (
        x_api_key is None or not secrets.compare_digest(x_api_key, configured)
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
