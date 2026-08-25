from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document

from app.config import Settings
from app.db import (
    CourseMaterialRecordConflictError,
    CourseMaterialRecordNotFoundError,
    PostgresCourseMaterialRepository,
    QdrantRepository,
    S3DocumentRepository,
)
from app.dto import (
    CourseMaterialProcessingStatus,
    PresignedUrlRequest,
    PresignedUrlResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UploadStatusResponse,
)
from app.model import CourseMaterial
from app.service.document_processor import SUPPORTED_EXTENSIONS, DocumentProcessor
from app.service.embeddings import RemoteEmbeddings

logger = logging.getLogger(__name__)


class CourseMaterialNotFoundError(KeyError):
    pass


class CourseMaterialConflictError(ValueError):
    pass


class InvalidCourseMaterialError(ValueError):
    pass


class CourseMaterialTooLargeError(ValueError):
    pass


class CourseMaterialService:
    def __init__(
        self,
        settings: Settings,
        *,
        metadata_store: PostgresCourseMaterialRepository | None = None,
        document_store: S3DocumentRepository | None = None,
    ) -> None:
        self.settings = settings
        self.processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
        self.metadata_store = metadata_store or PostgresCourseMaterialRepository(
            settings.database_url
        )
        self.document_store = document_store or S3DocumentRepository(
            endpoint_url=settings.s3_endpoint_url,
            public_endpoint_url=settings.s3_public_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
            bucket=settings.s3_course_materials_bucket,
            presigned_url_expiry_seconds=settings.s3_presigned_url_expiry_seconds,
            cors_allowed_origins=settings.allowed_origins,
        )
        self.embeddings: RemoteEmbeddings | None = None
        self.vectors: QdrantRepository | None = None
        self._write_lock = asyncio.Lock()
        self._processing_tasks: dict[str, asyncio.Task[None]] = {}

    async def initialize(self) -> None:
        self.settings.processing_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self.metadata_store.initialize)
        await asyncio.to_thread(self.document_store.initialize)
        await asyncio.to_thread(self._initialize_vectors)
        await self._fail_interrupted_processing()

    def _initialize_vectors(self) -> None:
        embeddings = RemoteEmbeddings(
            url=self.settings.embeddings_url,
            model=self.settings.embeddings_model,
            dimension=self.settings.embeddings_dimension,
            api_key=self.settings.embeddings_api_key,
            timeout=self.settings.embeddings_timeout_seconds,
            batch_size=self.settings.embeddings_batch_size,
            max_retries=self.settings.embeddings_max_retries,
        )
        vectors = QdrantRepository(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            collection=self.settings.qdrant_collection,
            embeddings=embeddings,
        )
        try:
            vectors.initialize(self.settings.embeddings_dimension)
        except Exception:
            vectors.close()
            embeddings.close()
            raise
        self.embeddings = embeddings
        self.vectors = vectors

    async def close(self) -> None:
        if self._processing_tasks:
            await asyncio.gather(
                *list(self._processing_tasks.values()), return_exceptions=True
            )
        if self.vectors is not None:
            await asyncio.to_thread(self.vectors.close)
        if self.embeddings is not None:
            await asyncio.to_thread(self.embeddings.close)
        await asyncio.to_thread(self.document_store.close)
        await asyncio.to_thread(self.metadata_store.close)

    async def health(self) -> dict[str, bool]:
        qdrant_healthy = self.vectors is not None and await asyncio.to_thread(
            self.vectors.health
        )
        postgres_healthy, s3_healthy = await asyncio.gather(
            asyncio.to_thread(self.metadata_store.health),
            asyncio.to_thread(self.document_store.health),
        )
        return {
            "qdrant": qdrant_healthy,
            "postgres": postgres_healthy,
            "s3": s3_healthy,
        }

    async def create_presigned_url(
        self, request: PresignedUrlRequest
    ) -> PresignedUrlResponse:
        filename = self._safe_filename(request.filename)
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise InvalidCourseMaterialError(
                "A .pdf, .docx, .txt, or .md course-material file is required."
            )
        material_id = uuid.uuid4()
        object_key = f"{request.course_id}/{material_id}{extension}"
        stored = CourseMaterial(
            id=material_id,
            course_id=request.course_id,
            filename=filename,
            status="awaiting_upload",
            s3_bucket=self.settings.s3_course_materials_bucket,
            s3_object_key=object_key,
        )

        try:
            await asyncio.to_thread(self.metadata_store.save, stored)
        except CourseMaterialRecordConflictError as exc:
            raise CourseMaterialConflictError(
                f"Course material '{material_id}' already exists."
            ) from exc

        try:
            presigned_url = await asyncio.to_thread(
                self.document_store.create_upload_url,
                object_key,
            )
        except Exception:
            try:
                await asyncio.to_thread(self.metadata_store.delete, material_id)
            except Exception:
                logger.exception(
                    "Unable to clean metadata after presigning failed for %s",
                    material_id,
                )
            raise

        return PresignedUrlResponse(
            course_material_id=material_id,
            presigned_url=presigned_url,
            object_key=object_key,
            expires_in=self.settings.s3_presigned_url_expiry_seconds,
        )

    async def report_upload_status(
        self,
        material_id: UUID,
        status: str,
    ) -> UploadStatusResponse:
        async with self._write_lock:
            stored = await self.get_stored(material_id)
            if status in {"failed", "upload_failed"}:
                if stored.status != "awaiting_upload":
                    raise CourseMaterialConflictError(
                        "Only an upload awaiting completion can be reported as failed."
                    )
                await asyncio.to_thread(self.metadata_store.delete, material_id)
                return UploadStatusResponse(
                    course_material_id=material_id, status="deleted"
                )

            if stored.status in {"processing", "completed"}:
                return UploadStatusResponse(
                    course_material_id=material_id, status=stored.status
                )
            if stored.status == "failed":
                raise CourseMaterialConflictError(
                    "The course material has already failed processing."
                )

            processing_path = (
                self.settings.processing_dir
                / f"{material_id}{Path(stored.filename).suffix.lower()}"
            )
            await self._download_uploaded_file(stored, processing_path)
            transitioned = await asyncio.to_thread(
                self.metadata_store.mark_processing,
                material_id,
            )
            if not transitioned:
                processing_path.unlink(missing_ok=True)
                return UploadStatusResponse(
                    course_material_id=material_id, status="processing"
                )

            task = asyncio.create_task(
                self._process_document(stored, processing_path),
                name=f"process-course-material-{material_id}",
            )
            key = str(material_id)
            self._processing_tasks[key] = task
            task.add_done_callback(
                lambda completed, item_id=key: self._forget_processing_task(
                    item_id, completed
                )
            )
            return UploadStatusResponse(
                course_material_id=material_id, status="processing"
            )

    async def _download_uploaded_file(
        self,
        stored: CourseMaterial,
        destination: Path,
    ) -> None:
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        size = await asyncio.to_thread(
            self.document_store.object_size, stored.s3_object_key
        )
        if size <= 0:
            raise InvalidCourseMaterialError("The uploaded file is empty.")
        if size > max_bytes:
            raise CourseMaterialTooLargeError(
                f"File exceeds the {self.settings.max_upload_size_mb} MB upload limit."
            )
        try:
            await asyncio.to_thread(
                self.document_store.download,
                stored.s3_object_key,
                destination,
            )
            downloaded_size = destination.stat().st_size
            if downloaded_size != size:
                raise InvalidCourseMaterialError(
                    "The uploaded file changed while it was being downloaded."
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    async def _process_document(
        self,
        stored: CourseMaterial,
        processing_path: Path,
    ) -> None:
        chunk_ids: list[str] = []
        try:
            chunks = await asyncio.to_thread(self.processor.process, processing_path)
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            enriched: list[Document] = []
            for index, chunk in enumerate(chunks):
                loader_metadata = {
                    key: value
                    for key, value in chunk.metadata.items()
                    if key != "source" and value is not None
                }
                enriched.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata={
                            **loader_metadata,
                            "course_material_id": str(stored.id),
                            "document_id": str(stored.id),
                            "course_id": str(stored.course_id),
                            "filename": stored.filename,
                            "chunk_index": index,
                        },
                    )
                )
            await asyncio.to_thread(
                self._require_vectors().add_documents,
                enriched,
                chunk_ids,
            )
            await asyncio.to_thread(
                self.metadata_store.mark_processing_completed,
                stored.id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize loader/dependency errors
            await self._clean_document_vectors(str(stored.id))
            try:
                await asyncio.to_thread(
                    self.metadata_store.mark_processing_failed,
                    stored.id,
                )
            except Exception:
                logger.exception(
                    "Unable to record processing failure for course material %s",
                    stored.id,
                )
            logger.error(
                "Course-material processing failed for %s: %s",
                stored.id,
                self._processing_error_message(exc),
            )
        finally:
            processing_path.unlink(missing_ok=True)

    async def _clean_document_vectors(self, document_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._require_vectors().delete_by_document, document_id
            )
        except Exception:
            logger.exception("Unable to clean Qdrant vectors for %s", document_id)

    async def _fail_interrupted_processing(self) -> None:
        records = await asyncio.to_thread(self.metadata_store.list)
        for stored in records:
            if stored.status != "processing":
                continue
            await self._clean_document_vectors(str(stored.id))
            await asyncio.to_thread(
                self.metadata_store.mark_processing_failed,
                stored.id,
            )

    def _forget_processing_task(
        self,
        material_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._processing_tasks.get(material_id) is task:
            self._processing_tasks.pop(material_id, None)

    async def get(self, material_id: UUID) -> CourseMaterial:
        return await self.get_stored(material_id)

    async def get_stored(self, material_id: UUID) -> CourseMaterial:
        try:
            return await asyncio.to_thread(self.metadata_store.get, material_id)
        except CourseMaterialRecordNotFoundError as exc:
            raise CourseMaterialNotFoundError(str(material_id)) from exc

    async def list(
        self,
        *,
        offset: int,
        limit: int,
        course_id: UUID | None = None,
    ) -> tuple[int, list[CourseMaterial]]:
        records = await asyncio.to_thread(self.metadata_store.list, course_id=course_id)
        return len(records), records[offset : offset + limit]

    async def processing_status(
        self, material_id: UUID
    ) -> CourseMaterialProcessingStatus:
        stored = await self.get_stored(material_id)
        return CourseMaterialProcessingStatus(
            id=stored.id,
            status=stored.status,
        )

    async def wait_for_processing(
        self, material_id: UUID
    ) -> CourseMaterialProcessingStatus:
        task = self._processing_tasks.get(str(material_id))
        if task is not None:
            await task
        return await self.processing_status(material_id)

    async def search(self, request: SearchRequest) -> SearchResponse:
        results = await asyncio.to_thread(
            self._require_vectors().search,
            request.query,
            k=request.k,
            course_id=str(request.course_id) if request.course_id else None,
            score_threshold=request.score_threshold,
        )
        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    content=document.page_content,
                    score=score,
                    metadata={
                        key: value
                        for key, value in document.metadata.items()
                        if not key.startswith("_")
                    },
                )
                for document, score in results
            ],
        )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        normalized = filename.strip().replace("\\", "/")
        basename = normalized.rsplit("/", 1)[-1]
        if basename in {"", ".", ".."}:
            raise InvalidCourseMaterialError("A valid filename is required.")
        return basename

    @staticmethod
    def _processing_error_message(exc: Exception) -> str:
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    def _require_vectors(self) -> QdrantRepository:
        if self.vectors is None:
            raise RuntimeError("Course-material service has not been initialized.")
        return self.vectors
