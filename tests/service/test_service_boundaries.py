import asyncio

import pytest
from pydantic import ValidationError

from app.dto import SearchRequest
from app.service import SearchService


class CoreSpy:
    def __init__(self):
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        return "result"


def test_search_service_delegates_the_validated_request() -> None:
    core = CoreSpy()
    service = SearchService(core)
    request = SearchRequest(query="photosynthesis", k=50, score_threshold=-1)
    assert asyncio.run(service.search(request)) == "result"
    assert core.requests == [request]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "x", "k": 0},
        {"query": "x", "k": 51},
        {"query": "x", "score_threshold": -1.01},
        {"query": "x", "score_threshold": 1.01},
        {"query": "x" * 4001},
        {"query": "x", "course_id": "c" * 37},
    ],
)
def test_search_request_rejects_boundary_mutations(payload) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_search_request_accepts_inclusive_boundaries() -> None:
    assert SearchRequest(query="x", k=1, score_threshold=-1).k == 1
    assert SearchRequest(query="x", k=50, score_threshold=1).score_threshold == 1
