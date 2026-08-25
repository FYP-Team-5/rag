from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.controller.dependencies import get_search_service, require_api_key
from app.dto import SearchRequest, SearchResponse
from app.service import EmbeddingsServiceError, SearchService

search_router = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(require_api_key)],
)


@search_router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    try:
        return await service.search(body)
    except EmbeddingsServiceError as exc:
        raise HTTPException(
            status_code=502, detail="Embeddings service request failed."
        ) from exc
