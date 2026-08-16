import pytest
from pydantic import ValidationError

from app.model import SearchRequest, SearchResponse, SearchResult


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "valid", "k": 0},
        {"query": "valid", "k": 51},
        {"query": "valid", "score_threshold": 1.1},
    ],
)
def test_search_request_rejects_invalid_values(payload) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_search_response_serializes_results() -> None:
    response = SearchResponse(
        query="accuracy",
        results=[
            SearchResult(
                content="Use accurate evidence.",
                score=0.91,
                metadata={"rubric_id": "rubric-1"},
            )
        ],
    )

    assert response.model_dump() == {
        "query": "accuracy",
        "results": [
            {
                "content": "Use accurate evidence.",
                "score": 0.91,
                "metadata": {"rubric_id": "rubric-1"},
            }
        ],
    }
