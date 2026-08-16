from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


class QdrantRepository:
    """Abstracts collection management and vector operations in Qdrant."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection: str,
        embeddings: Embeddings,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.embeddings = embeddings
        self._client: QdrantClient | None = None
        self._vector_store: QdrantVectorStore | None = None

    def initialize(self, embedding_dimension: int) -> None:
        client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        self._client = client

        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
        else:
            info = client.get_collection(self.collection)
            vector_config = info.config.params.vectors
            if isinstance(vector_config, dict):
                vector_config = vector_config.get("")
            if vector_config is None:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' does not contain the "
                    "expected unnamed dense vector."
                )
            if vector_config.size != embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension {embedding_dimension} does not match existing "
                    f"Qdrant collection dimension {vector_config.size}. Use a new collection "
                    "name or restore the original embedding model."
                )
            if vector_config.distance != models.Distance.COSINE:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' uses "
                    f"{vector_config.distance.name} distance; COSINE is required."
                )

        for field in (
            "metadata.document_id",
            "metadata.rubric_id",
            "metadata.course_id",
        ):
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                # Some Qdrant versions return an error for an existing payload index.
                if "already exists" not in str(exc).lower():
                    raise

        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection,
            embedding=self.embeddings,
            # Dimension and distance are validated above without calling embeddings.
            validate_collection_config=False,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def health(self) -> bool:
        try:
            self._require_client().get_collection(self.collection)
            return True
        except Exception:
            return False

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self._require_vector_store().add_documents(documents, ids=ids)

    def delete(self, ids: list[str]) -> None:
        self._require_vector_store().delete(ids=ids)

    def delete_by_document(self, document_id: str) -> None:
        self._require_client().delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def retrieve(self, ids: Sequence[str]) -> list[Document]:
        records = self._require_client().retrieve(
            collection_name=self.collection,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
        )
        documents: list[Document] = []
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", {})
            documents.append(
                Document(
                    page_content=str(payload.get("page_content", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return documents

    def search(
        self,
        query: str,
        *,
        k: int,
        rubric_id: str | None = None,
        course_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        conditions: list[models.FieldCondition] = []
        if rubric_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.rubric_id",
                    match=models.MatchValue(value=rubric_id),
                )
            )
        if course_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.course_id",
                    match=models.MatchValue(value=course_id),
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None
        kwargs: dict[str, Any] = {"k": k, "filter": query_filter}
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        return self._require_vector_store().similarity_search_with_score(query, **kwargs)

    def _require_client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("Qdrant repository has not been initialized.")
        return self._client

    def _require_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            raise RuntimeError("Qdrant repository has not been initialized.")
        return self._vector_store
