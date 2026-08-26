import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import S3StorageError
from app.dto import (
    CourseMaterialProcessingStatus,
    PresignedUrlResponse,
    SearchResponse,
    UploadStatusResponse,
)
from app.main import create_app
from app.model import CourseMaterial
from app.service import EmbeddingsServiceError

COURSE_ID = UUID("d87cecc2-e224-4e91-aeef-8c4773452674")
MATERIAL_ID = UUID("10a7de1e-55f5-43b7-8208-c152ef77a5b5")


class FakeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.materials: dict[UUID, CourseMaterial] = {}

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def health(self) -> dict[str, bool]:
        return {"qdrant": True, "postgres": True, "s3": True}

    async def create_presigned_url(self, request) -> PresignedUrlResponse:
        material = CourseMaterial(
            id=MATERIAL_ID,
            course_id=request.course_id,
            filename=request.filename,
            status="awaiting_upload",
            s3_bucket="course-materials",
            s3_object_key=f"{request.course_id}/{MATERIAL_ID}{request.file_extension}",
        )
        self.materials[MATERIAL_ID] = material
        return PresignedUrlResponse(
            course_material_id=MATERIAL_ID,
            presigned_url="http://localhost:8333/course-materials?signed=true",
            object_key=f"{request.course_id}/{MATERIAL_ID}{request.file_extension}",
            expires_in=900,
        )

    async def report_upload_status(
        self, material_id: UUID, status: str
    ) -> UploadStatusResponse:
        if status == "failed":
            self.materials.pop(material_id, None)
            return UploadStatusResponse(
                course_material_id=material_id, status="deleted"
            )
        self.materials[material_id].status = "processing"
        return UploadStatusResponse(course_material_id=material_id, status="processing")

    async def retry_processing(self, material_id: UUID) -> UploadStatusResponse:
        self.materials[material_id].status = "processing"
        return UploadStatusResponse(course_material_id=material_id, status="processing")

    async def list(self, *, offset: int, limit: int, **kwargs):
        items = list(self.materials.values())
        return len(items), items[offset : offset + limit]

    async def get(self, material_id: UUID) -> CourseMaterial:
        return self.materials[material_id]

    async def processing_status(
        self, material_id: UUID
    ) -> CourseMaterialProcessingStatus:
        material = self.materials.get(material_id)
        return CourseMaterialProcessingStatus(
            id=material_id,
            status=material.status if material else "processing",
        )

    async def search(self, request) -> SearchResponse:
        return SearchResponse(query=request.query, results=[])


class UnhealthyService(FakeService):
    async def health(self) -> dict[str, bool]:
        return {"qdrant": False, "postgres": True, "s3": True}


class EmbeddingsUnavailableService(FakeService):
    async def search(self, request) -> SearchResponse:
        raise EmbeddingsServiceError("Embeddings service is unavailable.")


class ObjectStorageUnavailableService(FakeService):
    async def create_presigned_url(self, request) -> PresignedUrlResponse:
        raise S3StorageError("Object storage is unavailable.")


def make_client(
    tmp_path: Path,
    api_key: str | None = None,
    service_type: type[FakeService] = FakeService,
) -> TestClient:
    settings = Settings(api_key=api_key, processing_dir=tmp_path / "processing")
    return TestClient(create_app(settings=settings, service=service_type(settings)))


def test_create_presigned_and_report_uploaded(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/v1/course-material/create_presigned",
            json={
                "course_id": str(COURSE_ID),
                "filename": "Lecture notes",
                "file_extension": ".pdf",
            },
        )
        status = client.post(
            f"/api/v1/course-material/{MATERIAL_ID}/upload-status",
            json={"status": "uploaded"},
        )
        listing = client.get("/api/v1/course-material")

    assert created.status_code == 200
    assert created.json()["course_material_id"] == str(MATERIAL_ID)
    assert "content_type" not in created.json()
    assert created.json()["expires_in"] == 900
    assert status.json() == {
        "course_material_id": str(MATERIAL_ID),
        "status": "processing",
    }
    assert listing.json()["total"] == 1


def test_failed_upload_deletes_material(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        client.post(
            "/api/v1/course-material/create_presigned",
            json={
                "course_id": str(COURSE_ID),
                "filename": "Lecture notes",
                "file_extension": ".pdf",
            },
        )
        response = client.post(
            f"/api/v1/course-material/{MATERIAL_ID}/upload-status",
            json={"status": "failed"},
        )
        listing = client.get("/api/v1/course-material")

    assert response.json()["status"] == "deleted"
    assert listing.json()["total"] == 0


def test_retry_processing(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        client.post(
            "/api/v1/course-material/create_presigned",
            json={
                "course_id": str(COURSE_ID),
                "filename": "Lecture notes",
                "file_extension": ".pdf",
            },
        )
        client.app.state.course_material_service.materials[MATERIAL_ID].status = (
            "failed"
        )
        response = client.post(
            f"/api/v1/course-material/{MATERIAL_ID}/retry-processing"
        )

    assert response.status_code == 200
    assert response.json() == {
        "course_material_id": str(MATERIAL_ID),
        "status": "processing",
    }


def test_old_rubric_upload_route_is_removed(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post("/api/v1/rubrics")

    assert response.status_code == 404


def test_api_key_is_optional_but_enforced_when_configured(tmp_path: Path) -> None:
    with make_client(tmp_path, api_key="secret") as client:
        unauthorized = client.get("/api/v1/course-material")
        authorized = client.get(
            "/api/v1/course-material", headers={"X-API-Key": "secret"}
        )
        health = client.get("/health")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert health.status_code == 200


def test_swagger_routes_are_split_by_controller(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        schema = client.get("/openapi.json").json()

    assert [tag["name"] for tag in schema["tags"]] == [
        "health",
        "course-material",
        "search",
    ]
    assert schema["paths"]["/health"]["get"]["tags"] == ["health"]
    assert schema["paths"]["/api/v1/course-material/create_presigned"]["post"][
        "tags"
    ] == ["course-material"]
    assert schema["paths"]["/api/v1/search"]["post"]["tags"] == ["search"]
    assert (
        schema["components"]["schemas"]["CourseMaterial"]["properties"]["course_id"][
            "format"
        ]
        == "uuid"
    )
    assert (
        schema["components"]["schemas"]["PresignedUrlRequest"]["properties"][
            "course_id"
        ]["format"]
        == "uuid"
    )
    search_course_id = schema["components"]["schemas"]["SearchRequest"]["properties"][
        "course_id"
    ]
    assert search_course_id["anyOf"][0]["format"] == "uuid"
    list_course_id = next(
        parameter
        for parameter in schema["paths"]["/api/v1/course-material"]["get"]["parameters"]
        if parameter["name"] == "course_id"
    )
    assert list_course_id["schema"]["anyOf"][0]["format"] == "uuid"


def test_health_reports_required_database_failure(tmp_path: Path) -> None:
    with make_client(tmp_path, service_type=UnhealthyService) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {"qdrant": "unavailable", "postgres": "ok", "s3": "ok"}
    }


def test_http_errors_and_validation_errors_are_logged(
    tmp_path: Path,
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.main"):
        with make_client(tmp_path, service_type=UnhealthyService) as client:
            unavailable = client.get("/health")
            invalid = client.get("/api/v1/course-material/not-a-uuid")

    assert unavailable.status_code == 503
    assert invalid.status_code == 422
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "HTTP error method=GET path=/health status_code=503" in message
        for message in messages
    )
    assert any(
        "Request validation failed method=GET "
        "path=/api/v1/course-material/not-a-uuid" in message
        for message in messages
    )


def test_search_returns_bad_gateway_when_embeddings_are_unavailable(
    tmp_path: Path,
) -> None:
    with make_client(tmp_path, service_type=EmbeddingsUnavailableService) as client:
        response = client.post("/api/v1/search", json={"query": "accuracy"})

    assert response.status_code == 502


def test_presign_returns_bad_gateway_when_s3_is_unavailable(tmp_path: Path) -> None:
    with make_client(tmp_path, service_type=ObjectStorageUnavailableService) as client:
        response = client.post(
            "/api/v1/course-material/create_presigned",
            json={
                "course_id": str(uuid4()),
                "filename": "Lecture notes",
                "file_extension": ".pdf",
            },
        )

    assert response.status_code == 502
