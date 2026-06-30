import json

import httpx
import pytest

from app.embeddings import EmbeddingsServiceError, RemoteEmbeddings


def test_remote_embeddings_uses_openai_contract_and_response_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert body == {"model": "test-model", "input": ["first", "second"]}
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer token"},
    )
    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        client=client,
    )

    assert embeddings.embed_documents(["first", "second"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_remote_embeddings_rejects_invalid_vector_count() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": []})
        )
    )
    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        client=client,
    )

    with pytest.raises(EmbeddingsServiceError):
        embeddings.embed_query("query")
