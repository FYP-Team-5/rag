from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from app.db import S3DocumentRepository, S3StorageError


class FakeS3Client:
    def __init__(self, buckets: list[str] | None = None) -> None:
        self.buckets = buckets or []
        self.created_bucket = None
        self.uploaded = None
        self.deleted = None
        self.presigned = None
        self.closed = False
        self.fail_list = False
        self.fail_presign = False
        self.object = b"course material"
        self.cors = None

    def list_buckets(self):
        if self.fail_list:
            raise ClientError(
                {"Error": {"Code": "500", "Message": "unavailable"}},
                "ListBuckets",
            )
        return {"Buckets": [{"Name": name} for name in self.buckets]}

    def create_bucket(self, **kwargs) -> None:
        self.created_bucket = kwargs

    def put_bucket_cors(self, **kwargs) -> None:
        self.cors = kwargs

    def head_bucket(self, **kwargs) -> None:
        if self.fail_list:
            raise ClientError(
                {"Error": {"Code": "500", "Message": "unavailable"}},
                "HeadBucket",
            )

    def head_object(self, **kwargs):
        if self.fail_list:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}},
                "HeadObject",
            )
        return {"ContentLength": len(self.object)}

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        Path(destination).write_bytes(self.object)

    def upload_file(self, *args, **kwargs) -> None:
        self.uploaded = (args, kwargs)

    def delete_object(self, **kwargs) -> None:
        self.deleted = kwargs

    def generate_presigned_url(self, *args, **kwargs) -> str:
        if self.fail_presign:
            raise ClientError(
                {"Error": {"Code": "500", "Message": "unavailable"}},
                "GeneratePresignedUrl",
            )
        self.presigned = (args, kwargs)
        return "http://localhost:8333/rubric-documents/document.pdf?signed=true"

    def close(self) -> None:
        self.closed = True


def make_repository(
    client: FakeS3Client,
    public_client: FakeS3Client | None = None,
) -> S3DocumentRepository:
    return S3DocumentRepository(
        endpoint_url="http://seaweedfs:8333",
        public_endpoint_url="http://localhost:8333",
        access_key="key",
        secret_key="secret",
        region="us-east-1",
        bucket="rubric-documents",
        presigned_url_expiry_seconds=900,
        client=client,
        public_client=public_client or client,
    )


def test_s3_repository_creates_missing_bucket() -> None:
    client = FakeS3Client()
    repository = make_repository(client)

    repository.initialize()

    assert client.created_bucket == {"Bucket": "rubric-documents"}


def test_s3_repository_configures_browser_upload_cors() -> None:
    client = FakeS3Client(["course-materials"])
    repository = S3DocumentRepository(
        endpoint_url="http://seaweedfs:8333",
        public_endpoint_url="http://localhost:8333",
        access_key="key",
        secret_key="secret",
        region="us-east-1",
        bucket="course-materials",
        presigned_url_expiry_seconds=900,
        cors_allowed_origins=["http://localhost:3000"],
        client=client,
        public_client=client,
    )

    repository.initialize()

    rule = client.cors["CORSConfiguration"]["CORSRules"][0]
    assert client.cors["Bucket"] == "course-materials"
    assert rule["AllowedOrigins"] == ["http://localhost:3000"]
    assert "PUT" in rule["AllowedMethods"]


def test_s3_repository_upload_delete_and_presign(tmp_path: Path) -> None:
    client = FakeS3Client(["rubric-documents"])
    public_client = FakeS3Client(["rubric-documents"])
    repository = make_repository(client, public_client)
    source = tmp_path / "rubric.pdf"
    source.write_bytes(b"pdf")

    repository.initialize()
    repository.upload(
        source,
        "rubrics/rubric-1/document.pdf",
        content_type="application/pdf",
        metadata={"rubric-id": "rubric-1"},
    )
    url = repository.create_download_url(
        "rubrics/rubric-1/document.pdf",
        'rubric".pdf',
    )
    repository.delete("rubrics/rubric-1/document.pdf")

    assert client.uploaded[0] == (
        str(source),
        "rubric-documents",
        "rubrics/rubric-1/document.pdf",
    )
    assert client.uploaded[1]["ExtraArgs"] == {
        "ContentType": "application/pdf",
        "Metadata": {"rubric-id": "rubric-1"},
    }
    assert public_client.presigned[1]["ExpiresIn"] == 900
    assert (
        'filename="rubric.pdf"'
        in public_client.presigned[1]["Params"]["ResponseContentDisposition"]
    )
    assert url.endswith("?signed=true")
    assert client.deleted == {
        "Bucket": "rubric-documents",
        "Key": "rubrics/rubric-1/document.pdf",
    }


def test_s3_repository_creates_presigned_upload_url() -> None:
    client = FakeS3Client(["rubric-documents"])
    public_client = FakeS3Client(["rubric-documents"])
    repository = make_repository(client, public_client)

    url = repository.create_upload_url(
        "course-materials/material-1/lecture.pdf",
    )

    assert public_client.presigned == (
        ("put_object",),
        {
            "Params": {
                "Bucket": "rubric-documents",
                "Key": "course-materials/material-1/lecture.pdf",
            },
            "ExpiresIn": 900,
        },
    )
    assert url.endswith("?signed=true")


def test_s3_repository_inspects_and_downloads_uploaded_object(tmp_path: Path) -> None:
    client = FakeS3Client(["rubric-documents"])
    repository = make_repository(client)
    destination = tmp_path / "material.md"

    size = repository.object_size("course/material.md")
    repository.download("course/material.md", destination)

    assert size == len(client.object)
    assert destination.read_bytes() == client.object


def test_s3_repository_wraps_presigned_upload_url_errors() -> None:
    client = FakeS3Client(["rubric-documents"])
    public_client = FakeS3Client(["rubric-documents"])
    public_client.fail_presign = True
    repository = make_repository(client, public_client)

    with pytest.raises(S3StorageError, match="upload URL"):
        repository.create_upload_url(
            "course-materials/material-1/lecture.pdf",
        )


def test_s3_repository_reports_health_and_wraps_initialization_errors() -> None:
    client = FakeS3Client()
    client.fail_list = True
    repository = make_repository(client)

    assert not repository.health()
    with pytest.raises(S3StorageError, match="initialize"):
        repository.initialize()
