"""Document processing, embedding, and ingestion services."""

from app.db import S3StorageError
from app.service.document_processor import (
    SUPPORTED_EXTENSIONS,
    DocumentProcessor,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from app.service.embeddings import EmbeddingsServiceError, RemoteEmbeddings
from app.service.rubric_service import (
    InvalidUploadError,
    RubricConflictError,
    RubricNotFoundError,
    RubricProcessingIncompleteError,
    RubricService,
    UploadTooLargeError,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentProcessor",
    "EmbeddingsServiceError",
    "EmptyDocumentError",
    "InvalidUploadError",
    "RemoteEmbeddings",
    "RubricConflictError",
    "RubricNotFoundError",
    "RubricProcessingIncompleteError",
    "RubricService",
    "S3StorageError",
    "UnsupportedDocumentError",
    "UploadTooLargeError",
]
