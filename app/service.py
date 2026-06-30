from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from app.embeddings import RemoteEmbeddings
from app.models import (
    Rubric,
    RubricChunk,
    RubricChunksResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StoredRubric,
)


RUBRIC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RubricNotFoundError(KeyError):
    pass


class RubricConflictError(ValueError):
    pass


class InvalidUploadError(ValueError):
    pass


class UploadTooLargeError(ValueError):
    pass


class RubricService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processor = DocumentProcessor(settings.chunk_size, settings.chunk_overlap)
        self.client: QdrantClient | None = None
        self.embeddings: RemoteEmbeddings | None = None
        self.vector_store: QdrantVectorStore | None = None
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        for directory in (
            self.settings.uploads_dir,
            self.settings.manifests_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        embeddings = RemoteEmbeddings(
            url=self.settings.embeddings_url,
            model=self.settings.embeddings_model,
            api_key=self.settings.embeddings_api_key,
            timeout=self.settings.embeddings_timeout_seconds,
            batch_size=self.settings.embeddings_batch_size,
            max_retries=self.settings.embeddings_max_retries,
        )
        try:
            dimension = len(embeddings.embed_query("embedding dimension probe"))
        except Exception:
            embeddings.close()
            raise
        client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30,
        )

        collection = self.settings.qdrant_collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
        else:
            info = client.get_collection(collection)
            vectors = info.config.params.vectors
            stored_dimension = getattr(vectors, "size", None)
            if stored_dimension is not None and stored_dimension != dimension:
                raise RuntimeError(
                    f"Embedding dimension {dimension} does not match existing Qdrant "
                    f"collection dimension {stored_dimension}. Use a new collection name or "
                    "restore the original embedding model."
                )

        for field in ("metadata.rubric_id", "metadata.course_id"):
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                # Qdrant reports an error when the index already exists on some versions.
                if "already exists" not in str(exc).lower():
                    raise

        self.client = client
        self.embeddings = embeddings
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )

    async def close(self) -> None:
        if self.client is not None:
            await asyncio.to_thread(self.client.close)
        if self.embeddings is not None:
            await asyncio.to_thread(self.embeddings.close)

    async def health(self) -> dict[str, bool]:
        qdrant_healthy = False
        embeddings_healthy = False
        try:
            if self.client is not None:
                await asyncio.to_thread(
                    self.client.get_collection, self.settings.qdrant_collection
                )
                qdrant_healthy = True
        except Exception:
            pass
        try:
            headers = {}
            if self.settings.embeddings_api_key:
                headers["Authorization"] = f"Bearer {self.settings.embeddings_api_key}"
            async with httpx.AsyncClient(
                timeout=self.settings.embeddings_timeout_seconds,
                headers=headers,
            ) as client:
                response = await client.get(self.settings.embeddings_health_url)
                response.raise_for_status()
                embeddings_healthy = True
        except httpx.HTTPError:
            pass
        return {"qdrant": qdrant_healthy, "embeddings": embeddings_healthy}

    async def ingest(
        self,
        upload: UploadFile,
        *,
        rubric_id: str | None,
        title: str | None,
        version: str,
        course_id: str | None,
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

        if course_id is not None and len(course_id) > 128:
            raise InvalidUploadError("course_id must be at most 128 characters.")

        async with self._write_lock:
            if self._manifest_path(rubric_id).exists():
                raise RubricConflictError(f"Rubric '{rubric_id}' already exists.")

            document_id = str(uuid.uuid4())
            storage_path = self.settings.uploads_dir / f"{document_id}{extension}"
            chunk_ids: list[str] = []
            vectors_written = False
            try:
                size_bytes, digest = await self._save_upload(upload, storage_path)
                chunks = await asyncio.to_thread(self.processor.process, storage_path)
                uploaded_at = datetime.now(UTC)
                chunk_ids = [str(uuid.uuid4()) for _ in chunks]
                common_metadata: dict[str, Any] = {
                    "rubric_id": rubric_id,
                    "document_id": document_id,
                    "title": title or Path(filename).stem,
                    "version": version,
                    "filename": filename,
                    "content_type": upload.content_type or "application/octet-stream",
                    "sha256": digest,
                    "uploaded_at": uploaded_at.isoformat(),
                    "custom": custom_metadata,
                }
                if course_id is not None:
                    common_metadata["course_id"] = course_id

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
                                **common_metadata,
                                "chunk_index": index,
                                "chunk_id": chunk_ids[index],
                            },
                        )
                    )

                vector_store = self._require_vector_store()
                await asyncio.to_thread(vector_store.add_documents, enriched, ids=chunk_ids)
                vectors_written = True

                stored = StoredRubric(
                    id=rubric_id,
                    document_id=document_id,
                    title=title or Path(filename).stem,
                    version=version,
                    course_id=course_id,
                    filename=filename,
                    content_type=upload.content_type or "application/octet-stream",
                    size_bytes=size_bytes,
                    sha256=digest,
                    chunk_count=len(enriched),
                    uploaded_at=uploaded_at,
                    metadata=custom_metadata,
                    storage_path=str(storage_path),
                    chunk_ids=chunk_ids,
                )
                await asyncio.to_thread(self._write_manifest, stored)
                return stored.public()
            except Exception:
                if vectors_written and chunk_ids:
                    try:
                        await asyncio.to_thread(
                            self._require_vector_store().delete, ids=chunk_ids
                        )
                    except Exception:
                        pass
                storage_path.unlink(missing_ok=True)
                raise

    async def _save_upload(self, upload: UploadFile, destination: Path) -> tuple[int, str]:
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

    async def list(self, *, offset: int, limit: int) -> tuple[int, list[Rubric]]:
        manifests = await asyncio.to_thread(self._read_all_manifests)
        manifests.sort(key=lambda item: item.uploaded_at, reverse=True)
        return len(manifests), [item.public() for item in manifests[offset : offset + limit]]

    async def get(self, rubric_id: str) -> Rubric:
        return (await asyncio.to_thread(self._read_manifest, rubric_id)).public()

    async def get_stored(self, rubric_id: str) -> StoredRubric:
        return await asyncio.to_thread(self._read_manifest, rubric_id)

    async def delete(self, rubric_id: str) -> None:
        async with self._write_lock:
            stored = await asyncio.to_thread(self._read_manifest, rubric_id)
            await asyncio.to_thread(
                self._require_vector_store().delete, ids=stored.chunk_ids
            )
            Path(stored.storage_path).unlink(missing_ok=True)
            self._manifest_path(rubric_id).unlink(missing_ok=True)

    async def get_chunks(self, rubric_id: str) -> RubricChunksResponse:
        stored = await asyncio.to_thread(self._read_manifest, rubric_id)
        if self.client is None:
            raise RuntimeError("Rubric service has not been initialized.")
        records = await asyncio.to_thread(
            self.client.retrieve,
            collection_name=self.settings.qdrant_collection,
            ids=stored.chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        chunks: list[RubricChunk] = []
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", {})
            chunks.append(
                RubricChunk(
                    content=str(payload.get("page_content", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        chunks.sort(key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))
        return RubricChunksResponse(rubric_id=rubric_id, chunks=chunks)

    async def search(self, request: SearchRequest) -> SearchResponse:
        conditions: list[models.FieldCondition] = []
        if request.rubric_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.rubric_id",
                    match=models.MatchValue(value=request.rubric_id),
                )
            )
        if request.course_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.course_id",
                    match=models.MatchValue(value=request.course_id),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        kwargs: dict[str, Any] = {"k": request.k, "filter": query_filter}
        if request.score_threshold is not None:
            kwargs["score_threshold"] = request.score_threshold

        results = await asyncio.to_thread(
            self._require_vector_store().similarity_search_with_score,
            request.query,
            **kwargs,
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

    def _manifest_path(self, rubric_id: str) -> Path:
        # IDs are validated before writes; this guard also protects read/delete paths.
        if not RUBRIC_ID_PATTERN.fullmatch(rubric_id):
            raise RubricNotFoundError(rubric_id)
        return self.settings.manifests_dir / f"{rubric_id}.json"

    def _write_manifest(self, stored: StoredRubric) -> None:
        path = self._manifest_path(stored.id)
        temporary = path.with_suffix(f".json.{secrets.token_hex(6)}.tmp")
        temporary.write_text(stored.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _read_manifest(self, rubric_id: str) -> StoredRubric:
        path = self._manifest_path(rubric_id)
        try:
            return StoredRubric.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RubricNotFoundError(rubric_id) from exc

    def _read_all_manifests(self) -> list[StoredRubric]:
        manifests: list[StoredRubric] = []
        for path in self.settings.manifests_dir.glob("*.json"):
            try:
                manifests.append(
                    StoredRubric.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
        return manifests

    def _require_vector_store(self) -> QdrantVectorStore:
        if self.vector_store is None:
            raise RuntimeError("Rubric service has not been initialized.")
        return self.vector_store
