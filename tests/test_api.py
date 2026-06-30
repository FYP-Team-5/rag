from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Rubric, SearchResponse


class FakeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingested: list[Rubric] = []

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def health(self) -> bool:
        return {"qdrant": True, "embeddings": True}

    async def ingest(self, upload, **kwargs) -> Rubric:
        content = await upload.read()
        rubric = Rubric(
            id=kwargs["rubric_id"] or "generated-id",
            title=kwargs["title"] or "Rubric",
            version=kwargs["version"],
            course_id=kwargs["course_id"],
            filename=upload.filename,
            content_type=upload.content_type,
            size_bytes=len(content),
            sha256="0" * 64,
            chunk_count=2,
            uploaded_at=datetime.now(UTC),
            metadata=kwargs["custom_metadata"],
        )
        self.ingested.append(rubric)
        return rubric

    async def list(self, *, offset: int, limit: int):
        return len(self.ingested), self.ingested[offset : offset + limit]

    async def search(self, request) -> SearchResponse:
        return SearchResponse(query=request.query, results=[])


def make_client(tmp_path: Path, api_key: str | None = None) -> TestClient:
    settings = Settings(
        api_key=api_key,
        uploads_dir=tmp_path / "uploads",
        manifests_dir=tmp_path / "manifests",
    )
    return TestClient(create_app(settings=settings, service=FakeService(settings)))


def test_upload_and_list_rubrics(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/rubrics",
            files={"file": ("history.md", b"# Criteria\nAccurate evidence", "text/markdown")},
            data={
                "rubric_id": "history-essay-v1",
                "title": "History short answer",
                "course_id": "HIST-101",
                "metadata": '{"teacher":"Ada"}',
            },
        )
        listing = client.get("/api/v1/rubrics")

    assert response.status_code == 201
    assert response.json()["id"] == "history-essay-v1"
    assert response.json()["metadata"] == {"teacher": "Ada"}
    assert listing.json()["total"] == 1


def test_rejects_bad_metadata(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/rubrics",
            files={"file": ("rubric.txt", b"criteria", "text/plain")},
            data={"metadata": "[]"},
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
        schema = client.get("/openapi.json").json()

    assert root.status_code in (302, 307)
    assert root.headers["location"] == "/docs"
    assert swagger.status_code == 200
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["in"] == "header"
