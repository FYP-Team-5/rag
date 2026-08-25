from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import models

import app.db.qdrant_repository as qdrant_module
from app.db import QdrantRepository


class NoNetworkEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError(
            "embeddings must not be called during Qdrant initialization"
        )

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError(
            "embeddings must not be called during Qdrant initialization"
        )


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.5, 1.0]


class FakeQdrantClient:
    def __init__(self, *, collection_exists: bool = False, vector_config=None) -> None:
        self._collection_exists = collection_exists
        self._vector_config = vector_config
        self.created_collection = None
        self.payload_indexes: list[str] = []
        self.closed = False
        self.records = []
        self.delete_call = None
        self.upsert_call = None
        self.query_call = None

    def collection_exists(self, collection: str) -> bool:
        return self._collection_exists

    def create_collection(self, **kwargs) -> None:
        self.created_collection = kwargs

    def get_collection(self, collection: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=self._vector_config),
            )
        )

    def create_payload_index(self, *, field_name: str, **kwargs) -> None:
        self.payload_indexes.append(field_name)

    def retrieve(self, **kwargs):
        return self.records

    def delete(self, **kwargs) -> None:
        self.delete_call = kwargs

    def upsert(self, **kwargs) -> None:
        self.upsert_call = kwargs

    def query_points(self, **kwargs):
        self.query_call = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "page_content": "match",
                        "metadata": {"chunk_index": 0},
                    },
                    score=0.9,
                )
            ]
        )

    def close(self) -> None:
        self.closed = True


def make_repository(embeddings: Embeddings | None = None) -> QdrantRepository:
    return QdrantRepository(
        url="http://qdrant:6333",
        api_key=None,
        collection="course-materials",
        embeddings=embeddings or NoNetworkEmbeddings(),
    )


def test_initialize_creates_collection_without_calling_embeddings(monkeypatch) -> None:
    client = FakeQdrantClient()
    embeddings = NoNetworkEmbeddings()

    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **kwargs: client)

    repository = make_repository(embeddings)
    repository.initialize(384)

    assert client.created_collection["collection_name"] == "course-materials"
    assert client.created_collection["vectors_config"].size == 384
    assert (
        client.created_collection["vectors_config"].distance == models.Distance.COSINE
    )
    assert client.payload_indexes == [
        "metadata.document_id",
        "metadata.course_material_id",
        "metadata.course_id",
    ]


@pytest.mark.parametrize(
    ("vector_config", "message"),
    [
        (
            models.VectorParams(size=768, distance=models.Distance.COSINE),
            "dimension 768",
        ),
        (
            models.VectorParams(size=384, distance=models.Distance.DOT),
            "COSINE is required",
        ),
        (
            {"named": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            "unnamed",
        ),
    ],
)
def test_initialize_rejects_incompatible_collection(
    monkeypatch,
    vector_config,
    message: str,
) -> None:
    client = FakeQdrantClient(collection_exists=True, vector_config=vector_config)
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **kwargs: client)
    repository = make_repository()

    with pytest.raises(RuntimeError, match=message):
        repository.initialize(384)


def test_search_builds_qdrant_filters() -> None:
    embeddings = RecordingEmbeddings()
    repository = make_repository(embeddings)
    client = FakeQdrantClient()
    repository._client = client

    results = repository.search(
        "accuracy",
        k=3,
        course_id="HIST-101",
        score_threshold=0.5,
    )

    assert results[0][0].page_content == "match"
    assert embeddings.query_calls == ["accuracy"]
    kwargs = client.query_call
    assert kwargs["query"] == [0.5, 1.0]
    assert kwargs["limit"] == 3
    assert kwargs["score_threshold"] == 0.5
    dumped_filter = kwargs["query_filter"].model_dump()
    assert [condition["key"] for condition in dumped_filter["must"]] == [
        "metadata.course_id",
    ]


def test_add_documents_calls_model_then_upserts_vectors() -> None:
    embeddings = RecordingEmbeddings()
    repository = make_repository(embeddings)
    client = FakeQdrantClient()
    repository._client = client
    documents = [
        Document(page_content="first", metadata={"chunk_index": 0}),
        Document(page_content="second", metadata={"chunk_index": 1}),
    ]

    repository.add_documents(documents, ["chunk-1", "chunk-2"])

    assert embeddings.document_calls == [["first", "second"]]
    assert client.upsert_call["collection_name"] == "course-materials"
    assert client.upsert_call["wait"] is True
    points = client.upsert_call["points"]
    assert [point.id for point in points] == ["chunk-1", "chunk-2"]
    assert [point.vector for point in points] == [[0.0, 1.0], [1.0, 1.0]]
    assert points[0].payload == {
        "page_content": "first",
        "metadata": {"chunk_index": 0},
    }


def test_retrieve_converts_qdrant_payloads_to_documents() -> None:
    repository = make_repository()
    client = FakeQdrantClient()
    client.records = [
        SimpleNamespace(
            payload={
                "page_content": "Criterion text",
                "metadata": {"chunk_index": 2},
            }
        )
    ]
    repository._client = client

    documents = repository.retrieve(["chunk-1"])

    assert documents == [
        Document(page_content="Criterion text", metadata={"chunk_index": 2})
    ]


def test_delete_by_document_uses_document_payload_filter() -> None:
    repository = make_repository()
    client = FakeQdrantClient()
    repository._client = client

    repository.delete_by_document("document-1")

    assert client.delete_call["collection_name"] == "course-materials"
    assert client.delete_call["wait"] is True
    selector = client.delete_call["points_selector"].model_dump()
    assert selector["filter"]["must"][0]["key"] == "metadata.document_id"
    assert selector["filter"]["must"][0]["match"]["value"] == "document-1"
