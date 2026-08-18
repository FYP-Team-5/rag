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


class FakeQdrantClient:
    def __init__(self, *, collection_exists: bool = False, vector_config=None) -> None:
        self._collection_exists = collection_exists
        self._vector_config = vector_config
        self.created_collection = None
        self.payload_indexes: list[str] = []
        self.closed = False
        self.records = []
        self.delete_call = None

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

    def close(self) -> None:
        self.closed = True


class FakeVectorStore:
    def __init__(self) -> None:
        self.search_call = None
        self.added = None
        self.deleted = None

    def similarity_search_with_score(self, query: str, **kwargs):
        self.search_call = (query, kwargs)
        return [(Document(page_content="match", metadata={"chunk_index": 0}), 0.9)]

    def add_documents(self, documents, *, ids) -> None:
        self.added = (documents, ids)

    def delete(self, *, ids) -> None:
        self.deleted = ids


def make_repository(embeddings: Embeddings | None = None) -> QdrantRepository:
    return QdrantRepository(
        url="http://qdrant:6333",
        api_key=None,
        collection="rubrics",
        embeddings=embeddings or NoNetworkEmbeddings(),
    )


def test_initialize_creates_collection_without_calling_embeddings(monkeypatch) -> None:
    client = FakeQdrantClient()
    embeddings = NoNetworkEmbeddings()

    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **kwargs: client)

    repository = make_repository(embeddings)
    repository.initialize(384)

    assert client.created_collection["collection_name"] == "rubrics"
    assert client.created_collection["vectors_config"].size == 384
    assert (
        client.created_collection["vectors_config"].distance == models.Distance.COSINE
    )
    assert client.payload_indexes == [
        "metadata.document_id",
        "metadata.rubric_id",
        "metadata.course_id",
        "metadata.exam_id",
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
    repository = make_repository()
    vector_store = FakeVectorStore()
    repository._vector_store = vector_store

    results = repository.search(
        "accuracy",
        k=3,
        rubric_id="rubric-1",
        course_id="HIST-101",
        exam_id="history-midterm",
        score_threshold=0.5,
    )

    assert results[0][0].page_content == "match"
    query, kwargs = vector_store.search_call
    assert query == "accuracy"
    assert kwargs["k"] == 3
    assert kwargs["score_threshold"] == 0.5
    dumped_filter = kwargs["filter"].model_dump()
    assert [condition["key"] for condition in dumped_filter["must"]] == [
        "metadata.rubric_id",
        "metadata.course_id",
        "metadata.exam_id",
    ]


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

    assert client.delete_call["collection_name"] == "rubrics"
    assert client.delete_call["wait"] is True
    selector = client.delete_call["points_selector"].model_dump()
    assert selector["filter"]["must"][0]["key"] == "metadata.document_id"
    assert selector["filter"]["must"][0]["match"]["value"] == "document-1"
