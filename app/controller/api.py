import json
import secrets
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader

from app.model import (
    DeleteResponse,
    HealthResponse,
    Rubric,
    RubricChunksResponse,
    RubricList,
    RubricProcessingStatus,
    SearchRequest,
    SearchResponse,
)
from app.service import (
    EmbeddingsServiceError,
    EmptyDocumentError,
    InvalidUploadError,
    RubricConflictError,
    RubricNotFoundError,
    RubricProcessingIncompleteError,
    RubricService,
    S3StorageError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)


def get_service(request: Request) -> RubricService:
    return request.app.state.rubric_service


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    configured = request.app.state.settings.api_key
    if configured and (x_api_key is None or not secrets.compare_digest(x_api_key, configured)):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


health_router = APIRouter(tags=["health"])
router = APIRouter(dependencies=[Depends(require_api_key)])


@health_router.get("/health", response_model=HealthResponse)
async def health(service: Annotated[RubricService, Depends(get_service)]) -> HealthResponse:
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


@router.post("/rubrics", response_model=Rubric, status_code=200, tags=["rubrics"])
async def upload_rubric(
    service: Annotated[RubricService, Depends(get_service)],
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, or Markdown rubric")],
    rubric_id: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form(max_length=300)] = None,
    version: Annotated[str, Form(max_length=64)] = "1",
    course_id: Annotated[str | None, Form(max_length=128)] = None,
    metadata: Annotated[str, Form(description="Optional JSON object")] = "{}",
) -> Rubric:
    try:
        custom_metadata: Any = json.loads(metadata)
        if not isinstance(custom_metadata, dict):
            raise TypeError
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="metadata must be a JSON object.") from exc

    try:
        return await service.ingest(
            file,
            rubric_id=rubric_id,
            title=title,
            version=version,
            course_id=course_id,
            custom_metadata=custom_metadata,
        )
    except RubricConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (InvalidUploadError, UnsupportedDocumentError, EmptyDocumentError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingsServiceError as exc:
        raise HTTPException(status_code=502, detail="Embeddings service request failed.") from exc
    except S3StorageError as exc:
        raise HTTPException(status_code=502, detail="Object storage request failed.") from exc
    finally:
        await file.close()


@router.get("/rubrics", response_model=RubricList, tags=["rubrics"])
async def list_rubrics(
    service: Annotated[RubricService, Depends(get_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RubricList:
    total, items = await service.list(offset=offset, limit=limit)
    return RubricList(total=total, items=items)


@router.get("/rubrics/{rubric_id}", response_model=Rubric, tags=["rubrics"])
async def get_rubric(
    rubric_id: str,
    service: Annotated[RubricService, Depends(get_service)],
) -> Rubric:
    try:
        return await service.get(rubric_id)
    except RubricNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Rubric not found.") from exc


@router.get(
    "/rubrics/{rubric_id}/status",
    response_model=RubricProcessingStatus,
    tags=["rubrics"],
)
async def get_rubric_processing_status(
    rubric_id: str,
    service: Annotated[RubricService, Depends(get_service)],
) -> RubricProcessingStatus:
    try:
        return await service.processing_status(rubric_id)
    except RubricNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Rubric not found.") from exc


@router.get(
    "/rubrics/{rubric_id}/download",
    response_class=RedirectResponse,
    tags=["rubrics"],
)
async def download_rubric(
    rubric_id: str,
    service: Annotated[RubricService, Depends(get_service)],
) -> RedirectResponse:
    try:
        download_url = await service.create_download_url(rubric_id)
    except RubricNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Rubric not found.") from exc
    except S3StorageError as exc:
        raise HTTPException(status_code=502, detail="Object storage request failed.") from exc
    return RedirectResponse(download_url, status_code=307)


@router.get(
    "/rubrics/{rubric_id}/chunks",
    response_model=RubricChunksResponse,
    tags=["rubrics"],
)
async def get_rubric_chunks(
    rubric_id: str,
    service: Annotated[RubricService, Depends(get_service)],
) -> RubricChunksResponse:
    try:
        return await service.get_chunks(rubric_id)
    except RubricNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Rubric not found.") from exc
    except RubricProcessingIncompleteError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/rubrics/{rubric_id}", response_model=DeleteResponse, tags=["rubrics"])
async def delete_rubric(
    rubric_id: str,
    service: Annotated[RubricService, Depends(get_service)],
) -> DeleteResponse:
    try:
        await service.delete(rubric_id)
    except RubricNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Rubric not found.") from exc
    except S3StorageError as exc:
        raise HTTPException(status_code=502, detail="Object storage request failed.") from exc
    return DeleteResponse(id=rubric_id, deleted=True)


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    body: SearchRequest,
    service: Annotated[RubricService, Depends(get_service)],
) -> SearchResponse:
    try:
        return await service.search(body)
    except EmbeddingsServiceError as exc:
        raise HTTPException(status_code=502, detail="Embeddings service request failed.") from exc
