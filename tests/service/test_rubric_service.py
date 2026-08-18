import asyncio
import threading
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from starlette.datastructures import Headers

import app.db.qdrant_repository as qdrant_module
from app.config import Settings
from app.db import PostgresRubricRepository
from app.model import SearchRequest
from app.service import (
    EmbeddingsServiceError,
    RemoteEmbeddings,
    RubricProcessingIncompleteError,
    RubricService,
)


class FakeVectors:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.deleted_ids: list[str] = []
        self.deleted_document_ids: list[str] = []
        self.search_kwargs = None

    def health(self) -> bool:
        return True

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.documents.update(dict(zip(ids, documents, strict=True)))

    def delete(self, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)
        for chunk_id in ids:
            self.documents.pop(chunk_id, None)

    def delete_by_document(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)
        self.documents = {
            chunk_id: document
            for chunk_id, document in self.documents.items()
            if document.metadata.get("document_id") != document_id
        }

    def retrieve(self, ids: list[str]) -> list[Document]:
        return [self.documents[chunk_id] for chunk_id in reversed(ids)]

    def search(self, query: str, **kwargs):
        self.search_kwargs = (query, kwargs)
        document = next(iter(self.documents.values()))
        return [(document, 0.87)]


class FailingVectors(FakeVectors):
    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        if documents:
            self.documents[ids[0]] = documents[0]
        raise EmbeddingsServiceError("Embeddings service unavailable.")


class BlockingVectors(FakeVectors):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release background processing")
        super().add_documents(documents, ids)


class FakeDocumentStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.initialized = False
        self.closed = False

    def initialize(self) -> None:
        self.initialized = True

    def close(self) -> None:
        self.closed = True

    def health(self) -> bool:
        return True

    def upload(
        self,
        path: Path,
        object_key: str,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        self.objects[object_key] = path.read_bytes()

    def delete(self, object_key: str) -> None:
        self.objects.pop(object_key, None)

    def create_download_url(self, object_key: str, filename: str) -> str:
        return f"http://localhost:8333/{object_key}?signed=true"


def make_metadata_store() -> PostgresRubricRepository:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = PostgresRubricRepository(engine=engine)
    repository.initialize()
    return repository


def make_service(tmp_path: Path, vectors: FakeVectors | None = None) -> RubricService:
    settings = Settings(
        processing_dir=tmp_path / "processing",
        chunk_size=100,
        chunk_overlap=10,
    )
    settings.processing_dir.mkdir(parents=True)
    service = RubricService(
        settings,
        metadata_store=make_metadata_store(),
        document_store=FakeDocumentStore(),  # type: ignore[arg-type]
    )
    service.vectors = vectors or FakeVectors()  # type: ignore[assignment]
    return service


def make_upload(content: bytes = b"# Accuracy\nUse relevant evidence.") -> UploadFile:
    return UploadFile(
        BytesIO(content),
        filename="rubric.md",
        headers=Headers({"content-type": "text/markdown"}),
    )


async def ingest_and_wait(
    service: RubricService,
    *,
    content: bytes = b"# Accuracy\nUse relevant evidence.",
    rubric_id: str = "history-v1",
):
    accepted = await service.ingest(
        make_upload(content),
        rubric_id=rubric_id,
        title="History rubric",
        version="1",
        course_id="HIST-101",
        exam_id="history-midterm",
        custom_metadata={"teacher": "Ada"},
    )
    status = await service.wait_for_processing(accepted.id)
    return accepted, status, await service.get(accepted.id)


def test_startup_does_not_call_embeddings_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    initialized_dimensions: list[int] = []

    class FakeQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def collection_exists(self, collection: str) -> bool:
            return False

        def create_collection(self, *, vectors_config, **kwargs) -> None:
            initialized_dimensions.append(vectors_config.size)

        def create_payload_index(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    def fail_if_called(embeddings: RemoteEmbeddings, text: str) -> list[float]:
        raise AssertionError(
            "The embeddings service must not be called during startup."
        )

    monkeypatch.setattr(qdrant_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(RemoteEmbeddings, "embed_query", fail_if_called)
    monkeypatch.setattr(RemoteEmbeddings, "embed_documents", fail_if_called)

    document_store = FakeDocumentStore()
    service = RubricService(
        Settings(embeddings_dimension=768, processing_dir=tmp_path / "processing"),
        metadata_store=make_metadata_store(),
        document_store=document_store,  # type: ignore[arg-type]
    )

    asyncio.run(service.initialize())
    asyncio.run(service.close())

    assert initialized_dimensions == [768]


def test_ingest_retrieve_search_and_delete_workflow(tmp_path: Path) -> None:
    vectors = FakeVectors()
    service = make_service(tmp_path, vectors)
    content = b"# Accuracy\n" + (b"Use relevant evidence and reasoning. " * 20)

    _accepted, status, rubric = asyncio.run(ingest_and_wait(service, content=content))

    stored = asyncio.run(service.get_stored(rubric.id))
    document_store = service.document_store
    assert isinstance(document_store, FakeDocumentStore)
    assert document_store.objects[stored.s3_object_key] == content
    assert status.processed is True
    assert status.processing_status == "completed"
    assert rubric.chunk_count > 1
    assert stored.chunk_ids == list(vectors.documents)
    assert all(
        document.metadata["rubric_id"] == "history-v1"
        and document.metadata["exam_id"] == "history-midterm"
        for document in vectors.documents.values()
    )

    chunks = asyncio.run(service.get_chunks(rubric.id))
    assert [chunk.metadata["chunk_index"] for chunk in chunks.chunks] == list(
        range(rubric.chunk_count)
    )

    response = asyncio.run(
        service.search(
            SearchRequest(
                query="accuracy",
                rubric_id=rubric.id,
                course_id="HIST-101",
                exam_id="history-midterm",
                k=2,
            )
        )
    )
    assert response.results[0].score == 0.87
    assert vectors.search_kwargs == (
        "accuracy",
        {
            "k": 2,
            "rubric_id": "history-v1",
            "course_id": "HIST-101",
            "exam_id": "history-midterm",
            "score_threshold": None,
        },
    )

    asyncio.run(service.archive(rubric.id))

    assert vectors.deleted_document_ids == []
    assert stored.s3_object_key in document_store.objects
    assert asyncio.run(service.get(rubric.id)).archived is True
    total, visible = asyncio.run(service.list(offset=0, limit=10))
    assert total == 0
    assert visible == []


def test_download_url_uses_s3_key_from_postgres(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    _, _, rubric = asyncio.run(ingest_and_wait(service))

    url = asyncio.run(service.create_download_url(rubric.id))

    assert url.startswith("http://localhost:8333/rubrics/history-v1/")
    assert url.endswith("?signed=true")


def test_processing_failure_is_recorded_and_partial_vectors_are_cleaned(
    tmp_path: Path,
) -> None:
    vectors = FailingVectors()
    service = make_service(tmp_path, vectors)

    accepted, status, rubric = asyncio.run(ingest_and_wait(service))

    assert accepted.processed is False
    assert status.processed is False
    assert status.processing_status == "failed"
    assert "EmbeddingsServiceError" in (status.processing_error or "")
    assert rubric.processing_status == "failed"
    assert list(service.settings.processing_dir.iterdir()) == []
    assert len(service.metadata_store.list()) == 1
    document_store = service.document_store
    assert isinstance(document_store, FakeDocumentStore)
    assert len(document_store.objects) == 1
    assert vectors.documents == {}
    assert len(vectors.deleted_document_ids) == 1

    with pytest.raises(RubricProcessingIncompleteError, match="failed"):
        asyncio.run(service.get_chunks(accepted.id))


def test_upload_returns_while_embeddings_continue_in_background(tmp_path: Path) -> None:
    vectors = BlockingVectors()
    service = make_service(tmp_path, vectors)

    async def scenario():
        accepted = await service.ingest(
            make_upload(),
            rubric_id="history-v1",
            title=None,
            version="1",
            course_id="HIST-101",
            exam_id="history-midterm",
            custom_metadata={},
        )
        assert await asyncio.to_thread(vectors.started.wait, 1)
        status_during_processing = await service.processing_status(accepted.id)
        vectors.release.set()
        completed = await service.wait_for_processing(accepted.id)
        return accepted, status_during_processing, completed

    accepted, during, completed = asyncio.run(scenario())

    assert accepted.processed is False
    assert during.processing_status == "processing"
    assert completed.processed is True
    assert completed.processing_status == "completed"


@pytest.mark.parametrize("rubric_id", ["../escape", "contains spaces"])
def test_ingest_rejects_invalid_custom_ids(tmp_path: Path, rubric_id: str) -> None:
    service = make_service(tmp_path)

    with pytest.raises(ValueError, match="rubric_id"):
        asyncio.run(
            service.ingest(
                make_upload(),
                rubric_id=rubric_id,
                title=None,
                version="1",
                course_id="HIST-101",
                exam_id="history-midterm",
                custom_metadata={},
            )
        )


def test_ingest_generates_id_when_custom_id_is_empty(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    async def scenario():
        result = await service.ingest(
            make_upload(),
            rubric_id="",
            title=None,
            version="1",
            course_id="HIST-101",
            exam_id="history-midterm",
            custom_metadata={},
        )
        await service.wait_for_processing(result.id)
        return result

    result = asyncio.run(scenario())

    assert result.id
