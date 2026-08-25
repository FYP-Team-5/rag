"""Database-backed domain models."""

from app.model.course_material import (
    CourseMaterial,
    CourseMaterialStatus,
)

__all__ = ["CourseMaterial", "CourseMaterialStatus"]
