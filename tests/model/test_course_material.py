from uuid import uuid4

from app.model import CourseMaterial


def test_course_material_contains_database_and_storage_fields() -> None:
    material = CourseMaterial(
        id=uuid4(),
        course_id=uuid4(),
        filename="lecture.md",
        status="awaiting_upload",
        s3_bucket="course-materials",
        s3_object_key="course/material.md",
    )

    assert material.model_dump().keys() == {
        "id",
        "course_id",
        "filename",
        "status",
        "s3_bucket",
        "s3_object_key",
    }
