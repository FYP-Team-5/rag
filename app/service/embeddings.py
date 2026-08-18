from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings


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
        try:
            response = self._client.post(
                self.url,
                json={"model": self.model, "input": list(texts)},
            )
            response.raise_for_status()
            body: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingsServiceError(
                f"Embeddings request to {self.url} failed: {exc}"
            ) from exc

        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise EmbeddingsServiceError("Embeddings response must contain a 'data' array.")

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
        if len(indexed) != len(texts) or [index for index, _ in indexed] != list(
            range(len(texts))
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
