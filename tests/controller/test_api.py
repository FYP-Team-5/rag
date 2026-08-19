from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.dto import RubricProcessingStatus, SearchResponse
from app.main import create_app
from app.model import Rubric
from app.service import EmbeddingsServiceError, S3StorageError


class FakeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingested: list[Rubric] = []

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def health(self) -> dict[str, bool]:
        return {"qdrant": True, "postgres": True, "s3": True}

    async def ingest(self, upload, **kwargs) -> Rubric:
        content = await upload.read()
        rubric = Rubric(
            id=kwargs["rubric_id"] or "generated-id",
            title=kwargs["title"] or "Rubric",
            version=kwargs["version"],
            course_id=kwargs["course_id"],
            exam_id=kwargs["exam_id"],
            filename=upload.filename,
            content_type=upload.content_type,
            size_bytes=len(content),
            sha256="0" * 64,
            chunk_count=0,
            processed=False,
            processing_status="processing",
            uploaded_at=datetime.now(UTC),
            metadata=kwargs["custom_metadata"],
        )
        self.ingested.append(rubric)
        return rubric

    async def list(self, *, offset: int, limit: int, **kwargs):
        return len(self.ingested), self.ingested[offset : offset + limit]

    async def search(self, request) -> SearchResponse:
        return SearchResponse(query=request.query, results=[])

    async def create_download_url(self, rubric_id: str) -> str:
        return f"http://localhost:8333/rubric-documents/{rubric_id}?signed=true"

    async def processing_status(self, rubric_id: str) -> RubricProcessingStatus:
        return RubricProcessingStatus(
            id=rubric_id,
            document_id=f"document-{rubric_id}",
            processed=False,
            processing_status="processing",
            chunk_count=0,
        )

    async def archive(self, rubric_id: str) -> None:
        for rubric in self.ingested:
            if rubric.id == rubric_id:
                rubric.archived = True
                return


class UnhealthyService(FakeService):
    async def health(self) -> dict[str, bool]:
        return {"qdrant": False, "postgres": True, "s3": True}


class EmbeddingsUnavailableService(FakeService):
    async def search(self, request) -> SearchResponse:
        raise EmbeddingsServiceError("Embeddings service is unavailable.")


class ObjectStorageUnavailableService(FakeService):
    async def ingest(self, upload, **kwargs) -> Rubric:
        raise S3StorageError("Object storage is unavailable.")

    async def create_download_url(self, rubric_id: str) -> str:
        raise S3StorageError("Object storage is unavailable.")


def make_client(
    tmp_path: Path,
    api_key: str | None = None,
    service_type: type[FakeService] = FakeService,
) -> TestClient:
    settings = Settings(
        api_key=api_key,
        processing_dir=tmp_path / "processing",
    )
    return TestClient(create_app(settings=settings, service=service_type(settings)))


def test_upload_and_list_rubrics(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/rubrics",
            files={
                "file": (
                    "history.md",
                    b"# Criteria\nAccurate evidence",
                    "text/markdown",
                )
            },
            data={
                "rubric_id": "history-essay-v1",
                "title": "History short answer",
                "course_id": "HIST-101",
                "exam_id": "history-midterm",
                "metadata": '{"teacher":"Ada"}',
            },
        )
        listing = client.get("/api/v1/rubrics")

    assert response.status_code == 200
    assert response.json()["id"] == "history-essay-v1"
    assert response.json()["metadata"] == {"teacher": "Ada"}
    assert response.json()["processed"] is False
    assert response.json()["processing_status"] == "processing"
    assert listing.json()["total"] == 1


def test_rejects_bad_metadata(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/rubrics",
            files={"file": ("rubric.txt", b"criteria", "text/plain")},
            data={
                "course_id": "HIST-101",
                "exam_id": "history-midterm",
                "metadata": "[]",
            },
        )

    assert response.status_code == 422


def test_api_key_is_optional_but_enforced_when_configured(tmp_path: Path) -> None:
    with make_client(tmp_path, api_key="secret") as client:
        unauthorized = client.get("/api/v1/rubrics")
        authorized = client.get("/api/v1/rubrics", headers={"X-API-Key": "secret"})
        health = client.get("/health")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert health.status_code == 200


def test_swagger_ui_and_api_key_security_scheme(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        root = client.get("/", follow_redirects=False)
        swagger = client.get("/docs")
        openapi = client.get("/openapi.json")
        schema = openapi.json()

    assert root.status_code in (302, 307)
    assert root.headers["location"] == "/docs"
    assert swagger.status_code == 200
    assert "SwaggerUIBundle" in swagger.text
    assert "url: '/openapi.json'" in swagger.text
    assert openapi.status_code == 200
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["in"] == "header"
    assert [tag["name"] for tag in schema["tags"]] == ["health", "rubrics", "search"]
    assert schema["paths"]["/api/v1/rubrics"]["post"]["tags"] == ["rubrics"]
    assert schema["paths"]["/api/v1/search"]["post"]["tags"] == ["search"]


def test_health_reports_required_database_failure(tmp_path: Path) -> None:
    with make_client(tmp_path, service_type=UnhealthyService) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {"qdrant": "unavailable", "postgres": "ok", "s3": "ok"}
    }


def test_upload_is_accepted_without_embeddings_but_search_returns_bad_gateway(
    tmp_path: Path,
) -> None:
    with make_client(tmp_path, service_type=EmbeddingsUnavailableService) as client:
        upload = client.post(
            "/api/v1/rubrics",
            files={"file": ("rubric.md", b"# Criteria", "text/markdown")},
            data={"course_id": "HIST-101", "exam_id": "history-midterm"},
        )
        search = client.post("/api/v1/search", json={"query": "accuracy"})

    assert upload.status_code == 200
    assert upload.json()["processed"] is False
    assert search.status_code == 502
    assert search.json() == {"detail": "Embeddings service request failed."}


def test_processing_status_endpoint(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/api/v1/rubrics/history-v1/status")

    assert response.status_code == 200
    assert response.json() == {
        "id": "history-v1",
        "document_id": "document-history-v1",
        "processed": False,
        "processing_status": "processing",
        "processing_error": None,
        "chunk_count": 0,
    }


def test_delete_endpoint_soft_archives_rubric(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.delete("/api/v1/rubrics/history-v1")

    assert response.status_code == 200
    assert response.json() == {"id": "history-v1", "archived": True}


def test_search_request_is_validated_before_service_call(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post("/api/v1/search", json={"query": "", "k": 0})

    assert response.status_code == 422


def test_download_redirects_to_presigned_s3_url(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get(
            "/api/v1/rubrics/rubric-1/download",
            follow_redirects=False,
        )

    assert response.status_code == 307
    assert response.headers["location"] == (
        "http://localhost:8333/rubric-documents/rubric-1?signed=true"
    )


def test_object_storage_failures_return_bad_gateway(tmp_path: Path) -> None:
    with make_client(tmp_path, service_type=ObjectStorageUnavailableService) as client:
        upload = client.post(
            "/api/v1/rubrics",
            files={"file": ("rubric.md", b"# Criteria", "text/markdown")},
            data={"course_id": "HIST-101", "exam_id": "history-midterm"},
        )
        download = client.get(
            "/api/v1/rubrics/rubric-1/download",
            follow_redirects=False,
        )

    assert upload.status_code == 502
    assert upload.json() == {"detail": "Object storage request failed."}
    assert download.status_code == 502
    assert download.json() == {"detail": "Object storage request failed."}
