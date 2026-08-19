from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Integer,
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

from app.model import StoredRubric

metadata = MetaData()

rubrics = Table(
    "rubrics",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("document_id", String(36), nullable=False, unique=True),
    Column("title", String(300), nullable=False),
    Column("version", String(64), nullable=False),
    Column("course_id", String(128), nullable=True, index=True),
    Column("exam_id", String(128), nullable=True, index=True),
    Column("filename", String(512), nullable=False),
    Column("content_type", String(255), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=False, index=True),
    Column("chunk_count", Integer, nullable=False),
    Column("processed", Boolean, nullable=False),
    Column("processing_status", String(32), nullable=False),
    Column("processing_error", Text, nullable=True),
    Column("archived", Boolean, nullable=False, default=False),
    Column("uploaded_at", DateTime(timezone=True), nullable=False, index=True),
    Column("custom_metadata", JSON, nullable=False),
    Column("s3_bucket", String(255), nullable=False),
    Column("s3_object_key", Text, nullable=False, unique=True),
    Column("chunk_ids", JSON, nullable=False),
)


class RubricRecordNotFoundError(KeyError):
    pass


class RubricRecordConflictError(ValueError):
    pass


class PostgresRubricRepository:
    """Persists document metadata in PostgreSQL through SQLAlchemy Core."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        if engine is None and database_url is None:
            raise ValueError("database_url is required when engine is not provided.")
        self.engine = engine or create_engine(
            database_url,
            pool_pre_ping=True,
        )

    def initialize(self) -> None:
        metadata.create_all(self.engine)
        self._migrate_processing_columns()

    def _migrate_processing_columns(self) -> None:
        """Upgrade metadata tables created before asynchronous processing existed."""
        existing = {
            column["name"] for column in inspect(self.engine).get_columns("rubrics")
        }
        statements: list[str] = []
        boolean_true = "TRUE" if self.engine.dialect.name == "postgresql" else "1"
        if "processed" not in existing:
            statements.append(
                f"ALTER TABLE rubrics ADD COLUMN processed BOOLEAN NOT NULL "
                f"DEFAULT {boolean_true}"
            )
        if "processing_status" not in existing:
            statements.append(
                "ALTER TABLE rubrics ADD COLUMN processing_status VARCHAR(32) "
                "NOT NULL DEFAULT 'completed'"
            )
        if "processing_error" not in existing:
            statements.append("ALTER TABLE rubrics ADD COLUMN processing_error TEXT")
        if "exam_id" not in existing:
            statements.append("ALTER TABLE rubrics ADD COLUMN exam_id VARCHAR(128)")
        if "archived" not in existing:
            statements.append(
                "ALTER TABLE rubrics ADD COLUMN archived BOOLEAN NOT NULL DEFAULT FALSE"
            )
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rubrics_processing_status "
                    "ON rubrics (processing_status)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rubrics_exam_id ON rubrics (exam_id)"
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_rubrics_exam_version "
                    "ON rubrics (exam_id, version) WHERE exam_id IS NOT NULL"
                )
            )

    def close(self) -> None:
        self.engine.dispose()

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(func.count()).select_from(rubrics))
            return True
        except SQLAlchemyError:
            return False

    def exists(self, rubric_id: str) -> bool:
        statement = select(rubrics.c.id).where(rubrics.c.id == rubric_id).limit(1)
        with self.engine.connect() as connection:
            return connection.execute(statement).first() is not None

    def save(self, stored: StoredRubric) -> None:
        values = stored.model_dump()
        values["custom_metadata"] = values.pop("metadata")
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(rubrics).values(**values))
        except IntegrityError as exc:
            raise RubricRecordConflictError(stored.id) from exc

    def get(self, rubric_id: str) -> StoredRubric:
        statement = select(rubrics).where(rubrics.c.id == rubric_id)
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        if row is None:
            raise RubricRecordNotFoundError(rubric_id)
        return self._to_model(row)

    def list(
        self,
        *,
        course_id: str | None = None,
        exam_id: str | None = None,
        include_archived: bool = False,
    ) -> list[StoredRubric]:
        statement = select(rubrics)
        if course_id is not None:
            statement = statement.where(rubrics.c.course_id == course_id)
        if exam_id is not None:
            statement = statement.where(rubrics.c.exam_id == exam_id)
        if not include_archived:
            statement = statement.where(rubrics.c.archived.is_(False))
        statement = statement.order_by(rubrics.c.uploaded_at.desc())
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._to_model(row) for row in rows]

    def delete(self, rubric_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(rubrics).where(rubrics.c.id == rubric_id)
            )
        if result.rowcount == 0:
            raise RubricRecordNotFoundError(rubric_id)

    def archive(self, rubric_id: str) -> None:
        statement = (
            update(rubrics).where(rubrics.c.id == rubric_id).values(archived=True)
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        if result.rowcount == 0:
            raise RubricRecordNotFoundError(rubric_id)

    def mark_processing_completed(self, rubric_id: str, chunk_ids: list[str]) -> None:
        statement = (
            update(rubrics)
            .where(rubrics.c.id == rubric_id)
            .values(
                processed=True,
                processing_status="completed",
                processing_error=None,
                chunk_count=len(chunk_ids),
                chunk_ids=chunk_ids,
            )
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        if result.rowcount == 0:
            raise RubricRecordNotFoundError(rubric_id)

    def mark_processing_failed(self, rubric_id: str, error: str) -> None:
        statement = (
            update(rubrics)
            .where(rubrics.c.id == rubric_id)
            .values(
                processed=False,
                processing_status="failed",
                processing_error=error[:2000],
                chunk_count=0,
                chunk_ids=[],
            )
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        if result.rowcount == 0:
            raise RubricRecordNotFoundError(rubric_id)

    @staticmethod
    def _to_model(row: RowMapping) -> StoredRubric:
        values = dict(row)
        values["metadata"] = values.pop("custom_metadata")
        return StoredRubric.model_validate(values)
