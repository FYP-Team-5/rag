from __future__ import annotations

from uuid import UUID

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.model import CourseMaterial

metadata = MetaData()

course_materials = Table(
    "course_materials",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("course_id", String(36), nullable=False, index=True),
    Column("filename", String(512), nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("s3_bucket", String(255), nullable=False),
    Column("s3_object_key", Text, nullable=False, unique=True),
)


class CourseMaterialRecordNotFoundError(KeyError):
    pass


class CourseMaterialRecordConflictError(ValueError):
    pass


class PostgresCourseMaterialRepository:
    """Persists course-material upload and processing state."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        if engine is None and database_url is None:
            raise ValueError("database_url is required when engine is not provided.")
        self.engine = engine or create_engine(database_url, pool_pre_ping=True)

    def initialize(self) -> None:
        # The legacy rubrics table is intentionally left untouched. Removing it would
        # be a destructive migration and it is no longer read by this service.
        metadata.create_all(self.engine)
        self._drop_unused_columns()

    def _drop_unused_columns(self) -> None:
        existing = {
            column["name"]
            for column in inspect(self.engine).get_columns("course_materials")
        }
        unused = (
            "content_type",
            "processing_error",
            "chunk_count",
            "chunk_ids",
            "created_at",
            "uploaded_at",
        )
        with self.engine.begin() as connection:
            connection.execute(
                text("DROP INDEX IF EXISTS ix_course_materials_created_at")
            )
            for column in unused:
                if column in existing:
                    connection.execute(
                        text(f"ALTER TABLE course_materials DROP COLUMN {column}")
                    )

    def close(self) -> None:
        self.engine.dispose()

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(func.count()).select_from(course_materials))
            return True
        except SQLAlchemyError:
            return False

    def save(self, stored: CourseMaterial) -> None:
        values = stored.model_dump()
        values["id"] = str(values["id"])
        values["course_id"] = str(values["course_id"])
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(course_materials).values(**values))
        except IntegrityError as exc:
            raise CourseMaterialRecordConflictError(str(stored.id)) from exc

    def get(self, material_id: UUID | str) -> CourseMaterial:
        statement = select(course_materials).where(
            course_materials.c.id == str(material_id)
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        if row is None:
            raise CourseMaterialRecordNotFoundError(str(material_id))
        return self._to_model(row)

    def list(self, *, course_id: UUID | str | None = None) -> list[CourseMaterial]:
        statement = select(course_materials)
        if course_id is not None:
            statement = statement.where(course_materials.c.course_id == str(course_id))
        statement = statement.order_by(course_materials.c.id)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._to_model(row) for row in rows]

    def delete(self, material_id: UUID | str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(course_materials).where(
                    course_materials.c.id == str(material_id)
                )
            )
        if result.rowcount == 0:
            raise CourseMaterialRecordNotFoundError(str(material_id))

    def mark_processing(
        self,
        material_id: UUID | str,
    ) -> bool:
        statement = (
            update(course_materials)
            .where(
                course_materials.c.id == str(material_id),
                course_materials.c.status == "awaiting_upload",
            )
            .values(status="processing")
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        return result.rowcount == 1

    def mark_processing_completed(self, material_id: UUID | str) -> None:
        statement = (
            update(course_materials)
            .where(course_materials.c.id == str(material_id))
            .values(status="completed")
        )
        self._execute_required(statement, material_id)

    def mark_processing_failed(self, material_id: UUID | str) -> None:
        statement = (
            update(course_materials)
            .where(course_materials.c.id == str(material_id))
            .values(status="failed")
        )
        self._execute_required(statement, material_id)

    def _execute_required(self, statement, material_id: UUID | str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        if result.rowcount == 0:
            raise CourseMaterialRecordNotFoundError(str(material_id))

    @staticmethod
    def _to_model(row: RowMapping) -> CourseMaterial:
        return CourseMaterial.model_validate(dict(row))
