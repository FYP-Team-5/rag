from app.dto import SearchRequest, SearchResponse


class SearchService:
    """Semantic course-material search use cases."""

    def __init__(self, core) -> None:
        self.core = core

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self.core.search(request)
