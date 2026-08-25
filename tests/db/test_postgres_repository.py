from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, inspect, text

from app.db import (
    CourseMaterialRecordConflictError,
    CourseMaterialRecordNotFoundError,
    PostgresCourseMaterialRepository,
)
from app.model import CourseMaterial

COURSE_ID = UUID("d87cecc2-e224-4e91-aeef-8c4773452674")


def make_repository() -> PostgresCourseMaterialRepository:
    return PostgresCourseMaterialRepository(
        engine=create_engine("sqlite+pysqlite:///:memory:")
    )


def make_material(material_id: UUID | None = None) -> CourseMaterial:
    material_id = material_id or uuid4()
    return CourseMaterial(
        id=material_id,
        course_id=COURSE_ID,
        filename="lecture.md",
        status="awaiting_upload",
        s3_bucket="course-materials",
        s3_object_key=f"{COURSE_ID}/{material_id}.md",
    )


def test_postgres_repository_round_trip_and_delete() -> None:
    repository = make_repository()
    repository.initialize()
    material = make_material()

    repository.save(material)

    assert repository.health()
    assert repository.get(material.id) == material

    repository.delete(material.id)

    with pytest.raises(CourseMaterialRecordNotFoundError):
        repository.get(material.id)


def test_postgres_repository_rejects_duplicate_ids() -> None:
    repository = make_repository()
    repository.initialize()
    material = make_material()
    repository.save(material)

    with pytest.raises(CourseMaterialRecordConflictError):
        repository.save(material)


def test_postgres_repository_filters_by_course() -> None:
    repository = make_repository()
    repository.initialize()
    first = make_material()
    second = make_material()
    second.course_id = uuid4()
    repository.save(first)
    repository.save(second)

    records = repository.list(course_id=COURSE_ID)

    assert [record.id for record in records] == [first.id]


def test_repository_updates_processing_state() -> None:
    repository = make_repository()
    repository.initialize()
    material = make_material()
    repository.save(material)

    assert repository.mark_processing(material.id)
    assert not repository.mark_processing(material.id)
    assert repository.get(material.id).status == "processing"

    repository.mark_processing_completed(material.id)
    assert repository.get(material.id).status == "completed"

    repository.mark_processing_failed(material.id)
    assert repository.get(material.id).status == "failed"


def test_initialize_drops_no_longer_used_course_material_columns() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE course_materials (
                    id VARCHAR(36) PRIMARY KEY,
                    course_id VARCHAR(36) NOT NULL,
                    filename VARCHAR(512) NOT NULL,
                    content_type VARCHAR(255) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    processing_error TEXT,
                    chunk_count INTEGER NOT NULL,
                    created_at DATETIME NOT NULL,
                    uploaded_at DATETIME,
                    s3_bucket VARCHAR(255) NOT NULL,
                    s3_object_key TEXT NOT NULL UNIQUE,
                    chunk_ids JSON NOT NULL
                )
                """
            )
        )

    PostgresCourseMaterialRepository(engine=engine).initialize()

    assert {
        column["name"] for column in inspect(engine).get_columns("course_materials")
    } == {
        "id",
        "course_id",
        "filename",
        "status",
        "s3_bucket",
        "s3_object_key",
    }
