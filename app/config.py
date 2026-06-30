from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Rubric RAG Service"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    api_key: str | None = None
    cors_origins: str = "*"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rubric_chunks"

    embeddings_url: str = "http://embeddings:8000/v1/embeddings"
    embeddings_health_url: str = "http://embeddings:8000/health"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    embeddings_api_key: str | None = None
    embeddings_timeout_seconds: float = Field(default=30, gt=0, le=300)
    embeddings_batch_size: int = Field(default=64, ge=1, le=512)
    embeddings_max_retries: int = Field(default=2, ge=0, le=10)
    uploads_dir: Path = Path("/data/uploads")
    manifests_dir: Path = Path("/data/manifests")

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)

    @property
    def allowed_origins(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
