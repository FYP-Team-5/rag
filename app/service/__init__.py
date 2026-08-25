"""Document processing, embedding, and ingestion services."""

from app.db import S3StorageError
from app.service.document_processor import (
    SUPPORTED_EXTENSIONS,
    DocumentProcessor,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from app.service.embeddings import EmbeddingsServiceError, RemoteEmbeddings
from app.service.course_material_service import (
    CourseMaterialConflictError,
    CourseMaterialNotFoundError,
    CourseMaterialService,
    CourseMaterialTooLargeError,
    InvalidCourseMaterialError,
)
from app.service.search_service import SearchService

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentProcessor",
    "CourseMaterialConflictError",
    "CourseMaterialNotFoundError",
    "CourseMaterialService",
    "CourseMaterialTooLargeError",
    "EmbeddingsServiceError",
    "EmptyDocumentError",
    "InvalidCourseMaterialError",
    "RemoteEmbeddings",
    "S3StorageError",
    "SearchService",
    "UnsupportedDocumentError",
]
