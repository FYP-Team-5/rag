from datetime import UTC, datetime

from app.model import Rubric, StoredRubric


def test_stored_rubric_public_removes_storage_fields() -> None:
    stored = StoredRubric(
        id="rubric-1",
        document_id="document-1",
        title="Rubric",
        version="1",
        filename="rubric.md",
        content_type="text/markdown",
        size_bytes=20,
        sha256="0" * 64,
        chunk_count=1,
        processed=True,
        processing_status="completed",
        uploaded_at=datetime.now(UTC),
        s3_bucket="rubric-documents",
        s3_object_key="rubrics/rubric-1/document-1.md",
        chunk_ids=["chunk-1"],
    )

    public = stored.public()

    assert isinstance(public, Rubric)
    assert "s3_bucket" not in public.model_dump()
    assert "s3_object_key" not in public.model_dump()
    assert "chunk_ids" not in public.model_dump()
    assert "document_id" not in public.model_dump()
    assert public.processed is True
    assert public.processing_status == "completed"


def test_rubric_metadata_defaults_are_independent() -> None:
    common = {
        "id": "rubric-1",
        "title": "Rubric",
        "version": "1",
        "filename": "rubric.md",
        "content_type": "text/markdown",
        "size_bytes": 20,
        "sha256": "0" * 64,
        "chunk_count": 1,
        "uploaded_at": datetime.now(UTC),
    }
    first = Rubric(**common)
    second = Rubric(**{**common, "id": "rubric-2"})

    first.metadata["owner"] = "Ada"

    assert second.metadata == {}
