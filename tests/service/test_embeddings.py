import json

import httpx
import pytest

from app.service import EmbeddingsServiceError, RemoteEmbeddings


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


def test_remote_embeddings_rejects_wrong_configured_dimension() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1, 2]}]},
            )
        )
    )
    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        dimension=3,
        client=client,
    )

    with pytest.raises(EmbeddingsServiceError, match="2-dimensional vectors"):
        embeddings.embed_query("query")


def test_remote_embeddings_wraps_connection_errors() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("service unavailable", request=request)

    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(unavailable)),
    )

    with pytest.raises(EmbeddingsServiceError, match="service unavailable"):
        embeddings.embed_query("query")


def test_remote_embeddings_batches_document_requests() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [len(text), index]}
                    for index, text in enumerate(inputs)
                ]
            },
        )

    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        batch_size=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vectors = embeddings.embed_documents(["a", "bb", "ccc"])

    assert requests == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]


@pytest.mark.parametrize(
    "response_body",
    [
        {},
        {"data": "not-a-list"},
        {"data": [{"index": 0}]},
        {"data": [{"index": 0, "embedding": []}]},
        {"data": [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [1]}]},
    ],
)
def test_remote_embeddings_rejects_malformed_responses(response_body) -> None:
    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=response_body)
            )
        ),
    )

    with pytest.raises(EmbeddingsServiceError):
        embeddings.embed_query("query")


def test_empty_document_batch_does_not_make_request() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No request should be made for an empty batch.")

    embeddings = RemoteEmbeddings(
        url="http://embeddings:8000/v1/embeddings",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(unexpected_request)),
    )

    assert embeddings.embed_documents([]) == []
