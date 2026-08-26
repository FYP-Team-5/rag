import asyncio
import threading
from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db import PostgresCourseMaterialRepository, S3StorageError
from app.dto import PresignedUrlRequest, SearchRequest
from app.service import (
    CourseMaterialService,
    EmbeddingsServiceError,
)

COURSE_ID = UUID("d87cecc2-e224-4e91-aeef-8c4773452674")


class FakeVectors:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.deleted_document_ids: list[str] = []
        self.search_kwargs = None

    def health(self) -> bool:
        return True

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.documents.update(dict(zip(ids, documents, strict=True)))

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


class FailOnceVectors(FakeVectors):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        if not self.failed_once:
            self.failed_once = True
            if documents:
                self.documents[ids[0]] = documents[0]
            raise EmbeddingsServiceError("Embeddings service unavailable.")
        super().add_documents(documents, ids)


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

    def create_upload_url(self, object_key: str) -> str:
        return f"http://localhost:8333/{object_key}?signed=true"

    def object_size(self, object_key: str) -> int:
        try:
            return len(self.objects[object_key])
        except KeyError as exc:
            raise S3StorageError("missing object") from exc

    def download(self, object_key: str, destination: Path) -> None:
        try:
            destination.write_bytes(self.objects[object_key])
        except KeyError as exc:
            raise S3StorageError("missing object") from exc


def make_metadata_store() -> PostgresCourseMaterialRepository:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = PostgresCourseMaterialRepository(engine=engine)
    repository.initialize()
    return repository


def make_service(
    tmp_path: Path, vectors: FakeVectors | None = None
) -> CourseMaterialService:
    settings = Settings(
        processing_dir=tmp_path / "processing",
        chunk_size=100,
        chunk_overlap=10,
    )
    settings.processing_dir.mkdir(parents=True)
    service = CourseMaterialService(
        settings,
        metadata_store=make_metadata_store(),
        document_store=FakeDocumentStore(),  # type: ignore[arg-type]
    )
    service.vectors = vectors or FakeVectors()  # type: ignore[assignment]
    return service


async def create_upload_and_wait(
    service: CourseMaterialService,
    *,
    content: bytes = b"# Accuracy\nUse relevant evidence.",
):
    created = await service.create_presigned_url(
        PresignedUrlRequest(
            course_id=COURSE_ID,
            filename="lecture.md",
        )
    )
    document_store = service.document_store
    assert isinstance(document_store, FakeDocumentStore)
    document_store.objects[created.object_key] = content
    accepted = await service.report_upload_status(
        created.course_material_id, "uploaded"
    )
    status = await service.wait_for_processing(created.course_material_id)
    material = await service.get(created.course_material_id)
    return created, accepted, status, material


def test_presign_persists_filename_course_and_object_key(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    created = asyncio.run(
        service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="slides.pdf")
        )
    )
    stored = service.metadata_store.get(created.course_material_id)

    assert stored.course_id == COURSE_ID
    assert stored.filename == "slides.pdf"
    assert stored.s3_object_key == created.object_key
    assert stored.status == "awaiting_upload"


def test_presign_does_not_validate_or_receive_file_content(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    created = asyncio.run(
        service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="unvalidated.csv")
        )
    )

    assert created.object_key.endswith(".csv")
    assert service.metadata_store.get(created.course_material_id).status == (
        "awaiting_upload"
    )


def test_uploaded_status_processes_and_searches_course_material(
    tmp_path: Path,
) -> None:
    vectors = FakeVectors()
    service = make_service(tmp_path, vectors)
    content = b"# Accuracy\n" + (b"Use relevant evidence and reasoning. " * 20)

    created, accepted, status, material = asyncio.run(
        create_upload_and_wait(service, content=content)
    )

    assert accepted.status == "processing"
    assert status.status == "completed"
    assert material.status == "completed"
    assert len(vectors.documents) > 1
    assert all(
        document.metadata["course_material_id"] == str(created.course_material_id)
        and document.metadata["course_id"] == str(COURSE_ID)
        for document in vectors.documents.values()
    )

    response = asyncio.run(
        service.search(
            SearchRequest(
                query="accuracy",
                course_id=str(COURSE_ID),
                k=2,
            )
        )
    )
    assert response.results[0].score == 0.87
    assert vectors.search_kwargs == (
        "accuracy",
        {
            "k": 2,
            "course_id": COURSE_ID,
            "score_threshold": None,
        },
    )


def test_failed_upload_status_deletes_postgres_entry(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    created = asyncio.run(
        service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="lecture.md")
        )
    )

    response = asyncio.run(
        service.report_upload_status(created.course_material_id, "failed")
    )

    assert response.status == "deleted"
    assert service.metadata_store.list() == []


def test_uploaded_status_requires_object_to_exist(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    created = asyncio.run(
        service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="lecture.md")
        )
    )

    with pytest.raises(S3StorageError, match="missing object"):
        asyncio.run(
            service.report_upload_status(created.course_material_id, "uploaded")
        )

    assert service.metadata_store.get(created.course_material_id).status == (
        "awaiting_upload"
    )


def test_processing_failure_is_recorded_and_vectors_are_cleaned(
    tmp_path: Path,
) -> None:
    vectors = FailingVectors()
    service = make_service(tmp_path, vectors)

    created, _accepted, status, material = asyncio.run(create_upload_and_wait(service))

    assert status.status == "failed"
    assert material.status == "failed"
    assert list(service.settings.processing_dir.iterdir()) == []
    assert vectors.documents == {}
    assert vectors.deleted_document_ids == [str(created.course_material_id)]


def test_failed_processing_can_be_retried_after_existing_vectors_are_deleted(
    tmp_path: Path,
) -> None:
    vectors = FailOnceVectors()
    service = make_service(tmp_path, vectors)

    async def scenario():
        created, _accepted, failed, _material = await create_upload_and_wait(service)
        vectors.documents["stale-chunk"] = Document(
            page_content="stale",
            metadata={"document_id": str(created.course_material_id)},
        )

        accepted = await service.retry_processing(created.course_material_id)
        completed = await service.wait_for_processing(created.course_material_id)
        return created, failed, accepted, completed

    created, failed, accepted, completed = asyncio.run(scenario())

    assert failed.status == "failed"
    assert accepted.status == "processing"
    assert completed.status == "completed"
    assert "stale-chunk" not in vectors.documents
    assert vectors.deleted_document_ids == [
        str(created.course_material_id),
        str(created.course_material_id),
    ]
    assert vectors.documents


def test_processing_retry_requires_failed_status(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    created = asyncio.run(
        service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="lecture.md")
        )
    )

    with pytest.raises(ValueError, match="failed processing"):
        asyncio.run(service.retry_processing(created.course_material_id))


def test_upload_callback_returns_while_embeddings_run_in_background(
    tmp_path: Path,
) -> None:
    vectors = BlockingVectors()
    service = make_service(tmp_path, vectors)

    async def scenario():
        created = await service.create_presigned_url(
            PresignedUrlRequest(course_id=COURSE_ID, filename="lecture.md")
        )
        document_store = service.document_store
        assert isinstance(document_store, FakeDocumentStore)
        document_store.objects[created.object_key] = b"# Accuracy\nEvidence"
        accepted = await service.report_upload_status(
            created.course_material_id, "uploaded"
        )
        assert await asyncio.to_thread(vectors.started.wait, 1)
        during = await service.processing_status(created.course_material_id)
        vectors.release.set()
        completed = await service.wait_for_processing(created.course_material_id)
        return accepted, during, completed

    accepted, during, completed = asyncio.run(scenario())

    assert accepted.status == "processing"
    assert during.status == "processing"
    assert completed.status == "completed"
