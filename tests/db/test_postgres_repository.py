from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine

from app.db import (
    PostgresRubricRepository,
    RubricRecordConflictError,
    RubricRecordNotFoundError,
)
from app.model import StoredRubric


def make_repository() -> PostgresRubricRepository:
    return PostgresRubricRepository(
        engine=create_engine("sqlite+pysqlite:///:memory:")
    )


def make_stored_rubric(
    rubric_id: str = "rubric-1",
    *,
    uploaded_at: datetime | None = None,
) -> StoredRubric:
    return StoredRubric(
        id=rubric_id,
        document_id=f"document-{rubric_id}",
        title="History rubric",
        version="1",
        course_id="HIST-101",
        filename="rubric.md",
        content_type="text/markdown",
        size_bytes=42,
        sha256="0" * 64,
        chunk_count=2,
        processed=True,
        processing_status="completed",
        uploaded_at=uploaded_at or datetime.now(UTC),
        metadata={"term": "fall"},
        s3_bucket="rubric-documents",
        s3_object_key=f"rubrics/{rubric_id}/document.md",
        chunk_ids=["chunk-1", "chunk-2"],
    )


def test_postgres_repository_round_trip_and_delete() -> None:
    repository = make_repository()
    repository.initialize()
    stored = make_stored_rubric()

    repository.save(stored)

    assert repository.health()
    assert repository.exists(stored.id)
    retrieved = repository.get(stored.id)
    assert retrieved.model_dump(exclude={"uploaded_at"}) == stored.model_dump(
        exclude={"uploaded_at"}
    )

    repository.delete(stored.id)

    assert not repository.exists(stored.id)
    with pytest.raises(RubricRecordNotFoundError):
        repository.get(stored.id)


def test_postgres_repository_rejects_duplicate_ids() -> None:
    repository = make_repository()
    repository.initialize()
    repository.save(make_stored_rubric())

    with pytest.raises(RubricRecordConflictError):
        repository.save(make_stored_rubric())


def test_postgres_repository_lists_newest_first() -> None:
    repository = make_repository()
    repository.initialize()
    now = datetime.now(UTC)
    repository.save(make_stored_rubric("older", uploaded_at=now - timedelta(days=1)))
    repository.save(make_stored_rubric("newer", uploaded_at=now))

    records = repository.list()

    assert [record.id for record in records] == ["newer", "older"]


def test_postgres_repository_updates_processing_state() -> None:
    repository = make_repository()
    repository.initialize()
    stored = make_stored_rubric()
    stored.processed = False
    stored.processing_status = "processing"
    stored.chunk_count = 0
    stored.chunk_ids = []
    repository.save(stored)

    repository.mark_processing_completed(stored.id, ["chunk-a", "chunk-b"])
    completed = repository.get(stored.id)

    assert completed.processed is True
    assert completed.processing_status == "completed"
    assert completed.chunk_count == 2
    assert completed.chunk_ids == ["chunk-a", "chunk-b"]

    repository.mark_processing_failed(stored.id, "embedding request failed")
    failed = repository.get(stored.id)

    assert failed.processed is False
    assert failed.processing_status == "failed"
    assert failed.processing_error == "embedding request failed"
    assert failed.chunk_count == 0
    assert failed.chunk_ids == []
