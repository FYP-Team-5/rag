from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingsServiceError(RuntimeError):
    """Raised when the embedding model container returns an invalid response."""


class RemoteEmbeddings(Embeddings):
    """HTTP client for an OpenAI-compatible embedding model container."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        dimension: int | None = None,
        api_key: str | None = None,
        timeout: float = 30,
        batch_size: int = 64,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self._log_url = str(httpx.URL(url).copy_with(query=None, fragment=None))
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self._owns_client = client is None
        if client is not None:
            self._client = client
            return

        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            headers=headers,
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=max_retries),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed(texts[start : start + self.batch_size]))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        started_at = time.perf_counter()
        status_code: int | None = None
        logger.info(
            "Embeddings request started model=%s endpoint=%s input_count=%d",
            self.model,
            self._log_url,
            len(texts),
        )
        try:
            response = self._client.post(
                self.url,
                json={"model": self.model, "input": list(texts)},
            )
            status_code = response.status_code
            response.raise_for_status()
            body: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception(
                "Embeddings request failed model=%s endpoint=%s input_count=%d "
                "status_code=%s duration_ms=%.2f",
                self.model,
                self._log_url,
                len(texts),
                status_code,
                (time.perf_counter() - started_at) * 1000,
            )
            raise EmbeddingsServiceError(
                f"Embeddings request to {self.url} failed: {exc}"
            ) from exc

        try:
            vectors = self._parse_response(body, len(texts))
        except EmbeddingsServiceError:
            logger.exception(
                "Invalid embeddings response model=%s endpoint=%s input_count=%d "
                "status_code=%s duration_ms=%.2f",
                self.model,
                self._log_url,
                len(texts),
                status_code,
                (time.perf_counter() - started_at) * 1000,
            )
            raise

        logger.info(
            "Embeddings request completed model=%s endpoint=%s input_count=%d "
            "status_code=%d vector_count=%d dimension=%d duration_ms=%.2f",
            self.model,
            self._log_url,
            len(texts),
            status_code,
            len(vectors),
            len(vectors[0]),
            (time.perf_counter() - started_at) * 1000,
        )
        return vectors

    def _parse_response(
        self,
        body: Any,
        expected_count: int,
    ) -> list[list[float]]:
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise EmbeddingsServiceError(
                "Embeddings response must contain a 'data' array."
            )

        indexed: list[tuple[int, list[float]]] = []
        for position, item in enumerate(body["data"]):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingsServiceError(
                    "Each embeddings response item must contain an 'embedding' array."
                )
            try:
                vector = [float(value) for value in item["embedding"]]
                index = int(item.get("index", position))
            except (TypeError, ValueError) as exc:
                raise EmbeddingsServiceError(
                    "Embedding values and indexes must be numeric."
                ) from exc
            if not vector:
                raise EmbeddingsServiceError("The embeddings service returned an empty vector.")
            indexed.append((index, vector))

        indexed.sort(key=lambda pair: pair[0])
        if len(indexed) != expected_count or [index for index, _ in indexed] != list(
            range(expected_count)
        ):
            raise EmbeddingsServiceError(
                "The embeddings service returned the wrong number of vectors or indexes."
            )

        dimensions = {len(vector) for _, vector in indexed}
        if len(dimensions) != 1:
            raise EmbeddingsServiceError(
                "The embeddings service returned vectors with inconsistent dimensions."
            )
        returned_dimension = dimensions.pop()
        if self.dimension is not None and returned_dimension != self.dimension:
            raise EmbeddingsServiceError(
                f"The embeddings service returned {returned_dimension}-dimensional vectors; "
                f"{self.dimension} were configured."
            )
        return [vector for _, vector in indexed]
