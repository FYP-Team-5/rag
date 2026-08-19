from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from langchain_core.documents import Document

from app.config import Settings
from app.db import (
    PostgresRubricRepository,
    QdrantRepository,
    RubricRecordConflictError,
    RubricRecordNotFoundError,
    S3DocumentRepository,
)
from app.dto import (
    RubricChunk,
    RubricChunksResponse,
    RubricProcessingStatus,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.model import Rubric, StoredRubric
from app.service.document_processor import SUPPORTED_EXTENSIONS, DocumentProcessor
from app.service.embeddings import RemoteEmbeddings

RUBRIC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
logger = logging.getLogger(__name__)


class RubricNotFoundError(KeyError):
    pass


class RubricConflictError(ValueError):
    pass


class InvalidUploadError(ValueError):
    pass


class UploadTooLargeError(ValueError):
    pass


class RubricProcessingIncompleteError(RuntimeError):
    pass


class RubricService:
    def __init__(
        self,
        settings: Settings,
        *,
        metadata_store: PostgresRubricRepository | None = None,
        document_store: S3DocumentRepository | None = None,
    ) -> None:
        self.settings = settings
        self.processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
        self.metadata_store = metadata_store or PostgresRubricRepository(
            settings.database_url
        )
        self.document_store = document_store or S3DocumentRepository(
            endpoint_url=settings.s3_endpoint_url,
            public_endpoint_url=settings.s3_public_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
            bucket=settings.s3_bucket,
            presigned_url_expiry_seconds=settings.s3_presigned_url_expiry_seconds,
        )
        self.embeddings: RemoteEmbeddings | None = None
        self.vectors: QdrantRepository | None = None
        self._write_lock = asyncio.Lock()
        self._processing_tasks: dict[str, asyncio.Task[None]] = {}

    async def initialize(self) -> None:
        self.settings.processing_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self.metadata_store.initialize)
        await asyncio.to_thread(self.document_store.initialize)
        await asyncio.to_thread(self._initialize_sync)
        await self._fail_interrupted_processing()

    def _initialize_sync(self) -> None:
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
                *list(self._processing_tasks.values()),
                return_exceptions=True,
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

    async def ingest(
        self,
        upload: UploadFile,
        *,
        rubric_id: str | None,
        title: str | None,
        version: str,
        course_id: str,
        exam_id: str,
        custom_metadata: dict[str, Any],
    ) -> Rubric:
        filename = Path(upload.filename or "").name
        extension = Path(filename).suffix.lower()
        if not filename or extension not in SUPPORTED_EXTENSIONS:
            raise InvalidUploadError(
                "A .pdf, .docx, .txt, or .md rubric file is required."
            )

        rubric_id = rubric_id or str(uuid.uuid4())
        if not RUBRIC_ID_PATTERN.fullmatch(rubric_id):
            raise InvalidUploadError(
                "rubric_id must be 1-128 characters and contain only letters, numbers, "
                "periods, underscores, or hyphens."
            )

        for field_name, value in (("course_id", course_id), ("exam_id", exam_id)):
            if not RUBRIC_ID_PATTERN.fullmatch(value):
                raise InvalidUploadError(
                    f"{field_name} must be 1-128 characters and contain only letters, "
                    "numbers, periods, underscores, or hyphens."
                )

        async with self._write_lock:
            if await asyncio.to_thread(self.metadata_store.exists, rubric_id):
                raise RubricConflictError(f"Rubric '{rubric_id}' already exists.")

            document_id = str(uuid.uuid4())
            processing_path = self.settings.processing_dir / f"{document_id}{extension}"
            object_key = f"rubrics/{rubric_id}/{document_id}{extension}"
            processing_task: asyncio.Task[None] | None = None
            metadata_written = False
            try:
                size_bytes, digest = await self._save_upload(upload, processing_path)
                uploaded_at = datetime.now(UTC)
                stored = StoredRubric(
                    id=rubric_id,
                    document_id=document_id,
                    title=title or Path(filename).stem,
                    version=version,
                    course_id=course_id,
                    exam_id=exam_id,
                    filename=filename,
                    content_type=upload.content_type or "application/octet-stream",
                    size_bytes=size_bytes,
                    sha256=digest,
                    chunk_count=0,
                    processed=False,
                    processing_status="processing",
                    processing_error=None,
                    uploaded_at=uploaded_at,
                    metadata=custom_metadata,
                    s3_bucket=self.settings.s3_bucket,
                    s3_object_key=object_key,
                    chunk_ids=[],
                )
                try:
                    await asyncio.to_thread(self.metadata_store.save, stored)
                except RubricRecordConflictError as exc:
                    raise RubricConflictError(
                        f"Rubric '{rubric_id}' already exists."
                    ) from exc
                metadata_written = True

                upload_result: asyncio.Future[bool] = (
                    asyncio.get_running_loop().create_future()
                )
                processing_task = asyncio.create_task(
                    self._process_document(stored, processing_path, upload_result),
                    name=f"process-rubric-{rubric_id}",
                )
                self._processing_tasks[rubric_id] = processing_task
                processing_task.add_done_callback(
                    lambda task, item_id=rubric_id: self._forget_processing_task(
                        item_id, task
                    )
                )

                try:
                    await asyncio.to_thread(
                        self.document_store.upload,
                        processing_path,
                        object_key,
                        content_type=upload.content_type or "application/octet-stream",
                        metadata={
                            "rubric-id": rubric_id,
                            "document-id": document_id,
                            "course-id": course_id,
                            "exam-id": exam_id,
                            "sha256": digest,
                        },
                    )
                except Exception:
                    upload_result.set_result(False)
                    await processing_task
                    await asyncio.to_thread(self.metadata_store.delete, rubric_id)
                    metadata_written = False
                    raise

                upload_result.set_result(True)
                return (await self.get_stored(rubric_id)).public()
            except Exception:
                if metadata_written and processing_task is None:
                    try:
                        await asyncio.to_thread(self.metadata_store.delete, rubric_id)
                    except Exception:
                        logger.exception(
                            "Unable to clean metadata after failed ingest for rubric %s",
                            rubric_id,
                        )
                raise
            finally:
                if processing_task is None:
                    processing_path.unlink(missing_ok=True)

    async def _process_document(
        self,
        stored: StoredRubric,
        processing_path: Path,
        upload_result: asyncio.Future[bool],
    ) -> None:
        processing_error: Exception | None = None
        chunk_ids: list[str] = []
        enriched: list[Document] = []
        try:
            try:
                chunks = await asyncio.to_thread(
                    self.processor.process, processing_path
                )
                chunk_ids = [str(uuid.uuid4()) for _ in chunks]
                common_metadata: dict[str, Any] = {
                    "rubric_id": stored.id,
                    "document_id": stored.document_id,
                    "title": stored.title,
                    "version": stored.version,
                    "filename": stored.filename,
                    "content_type": stored.content_type,
                    "sha256": stored.sha256,
                    "uploaded_at": stored.uploaded_at.isoformat(),
                    "custom": stored.metadata,
                }
                if stored.course_id is not None:
                    common_metadata["course_id"] = stored.course_id
                if stored.exam_id is not None:
                    common_metadata["exam_id"] = stored.exam_id

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
                                **common_metadata,
                                "chunk_index": index,
                                "chunk_id": chunk_ids[index],
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - document loaders vary by format
                processing_error = exc

            if not await upload_result:
                return

            if processing_error is None:
                try:
                    await asyncio.to_thread(
                        self._require_vectors().add_documents,
                        enriched,
                        chunk_ids,
                    )
                except Exception as exc:  # noqa: BLE001 - normalize dependency failures
                    processing_error = exc

            if processing_error is not None:
                await self._clean_document_vectors(stored.document_id)
                await asyncio.to_thread(
                    self.metadata_store.mark_processing_failed,
                    stored.id,
                    self._processing_error_message(processing_error),
                )
                logger.error(
                    "Document processing failed for rubric %s: %s",
                    stored.id,
                    self._processing_error_message(processing_error),
                )
                return

            await asyncio.to_thread(
                self.metadata_store.mark_processing_completed,
                stored.id,
                chunk_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Background processing task crashed for rubric %s", stored.id
            )
            upload_succeeded = (
                upload_result.done()
                and not upload_result.cancelled()
                and upload_result.result()
            )
            if upload_succeeded:
                await self._clean_document_vectors(stored.document_id)
                try:
                    await asyncio.to_thread(
                        self.metadata_store.mark_processing_failed,
                        stored.id,
                        self._processing_error_message(exc),
                    )
                except Exception:
                    logger.exception(
                        "Unable to record processing failure for rubric %s",
                        stored.id,
                    )
        finally:
            processing_path.unlink(missing_ok=True)

    async def _clean_document_vectors(self, document_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._require_vectors().delete_by_document,
                document_id,
            )
        except Exception:
            logger.exception(
                "Unable to clean Qdrant vectors for document %s", document_id
            )

    async def _fail_interrupted_processing(self) -> None:
        records = await asyncio.to_thread(self.metadata_store.list)
        for stored in records:
            if stored.processing_status != "processing":
                continue
            await self._clean_document_vectors(stored.document_id)
            await asyncio.to_thread(
                self.metadata_store.mark_processing_failed,
                stored.id,
                "Processing was interrupted by a service restart.",
            )

    def _forget_processing_task(
        self,
        rubric_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._processing_tasks.get(rubric_id) is task:
            self._processing_tasks.pop(rubric_id, None)

    @staticmethod
    def _processing_error_message(exc: Exception) -> str:
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    async def _save_upload(
        self, upload: UploadFile, destination: Path
    ) -> tuple[int, str]:
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        size = 0
        digest = hashlib.sha256()
        with destination.open("xb") as output:
            while data := await upload.read(1024 * 1024):
                size += len(data)
                if size > max_bytes:
                    raise UploadTooLargeError(
                        f"File exceeds the {self.settings.max_upload_size_mb} MB upload limit."
                    )
                digest.update(data)
                output.write(data)
        if size == 0:
            raise InvalidUploadError("The uploaded file is empty.")
        return size, digest.hexdigest()

    async def list(
        self,
        *,
        offset: int,
        limit: int,
        course_id: str | None = None,
        exam_id: str | None = None,
        include_archived: bool = False,
    ) -> tuple[int, list[Rubric]]:
        records = await asyncio.to_thread(
            self.metadata_store.list,
            course_id=course_id,
            exam_id=exam_id,
            include_archived=include_archived,
        )
        return len(records), [
            item.public() for item in records[offset : offset + limit]
        ]

    async def get(self, rubric_id: str) -> Rubric:
        return (await self.get_stored(rubric_id)).public()

    async def get_stored(self, rubric_id: str) -> StoredRubric:
        try:
            return await asyncio.to_thread(self.metadata_store.get, rubric_id)
        except RubricRecordNotFoundError as exc:
            raise RubricNotFoundError(rubric_id) from exc

    async def create_download_url(self, rubric_id: str) -> str:
        stored = await self.get_stored(rubric_id)
        return await asyncio.to_thread(
            self.document_store.create_download_url,
            stored.s3_object_key,
            stored.filename,
        )

    async def processing_status(self, rubric_id: str) -> RubricProcessingStatus:
        stored = await self.get_stored(rubric_id)
        return RubricProcessingStatus(
            id=stored.id,
            document_id=stored.document_id,
            processed=stored.processed,
            processing_status=stored.processing_status,
            processing_error=stored.processing_error,
            chunk_count=stored.chunk_count,
        )

    async def wait_for_processing(self, rubric_id: str) -> RubricProcessingStatus:
        task = self._processing_tasks.get(rubric_id)
        if task is not None:
            await task
        return await self.processing_status(rubric_id)

    async def archive(self, rubric_id: str) -> None:
        async with self._write_lock:
            task = self._processing_tasks.get(rubric_id)
            if task is not None:
                await task
            await self.get_stored(rubric_id)
            await asyncio.to_thread(self.metadata_store.archive, rubric_id)

    async def get_chunks(self, rubric_id: str) -> RubricChunksResponse:
        stored = await self.get_stored(rubric_id)
        if not stored.processed:
            raise RubricProcessingIncompleteError(
                f"Rubric '{rubric_id}' processing is {stored.processing_status}."
            )
        documents = await asyncio.to_thread(
            self._require_vectors().retrieve,
            stored.chunk_ids,
        )
        chunks = [
            RubricChunk(content=document.page_content, metadata=document.metadata)
            for document in documents
        ]
        chunks.sort(key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))
        return RubricChunksResponse(rubric_id=rubric_id, chunks=chunks)

    async def search(self, request: SearchRequest) -> SearchResponse:
        results = await asyncio.to_thread(
            self._require_vectors().search,
            request.query,
            k=request.k,
            rubric_id=request.rubric_id,
            course_id=request.course_id,
            exam_id=request.exam_id,
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

    def _require_vectors(self) -> QdrantRepository:
        if self.vectors is None:
            raise RuntimeError("Rubric service has not been initialized.")
        return self.vectors
