from pathlib import Path
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class S3StorageError(RuntimeError):
    pass


class S3DocumentRepository:
    """Stores original documents in an S3-compatible object store."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        public_endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
        bucket: str,
        presigned_url_expiry_seconds: int,
        cors_allowed_origins: list[str] | None = None,
        client: BaseClient | None = None,
        public_client: BaseClient | None = None,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.presigned_url_expiry_seconds = presigned_url_expiry_seconds
        self.cors_allowed_origins = cors_allowed_origins or []
        client_options: dict[str, Any] = {
            "service_name": "s3",
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        }
        self._client = client or boto3.client(
            endpoint_url=endpoint_url,
            **client_options,
        )
        self._public_client = public_client or boto3.client(
            endpoint_url=public_endpoint_url,
            **client_options,
        )

    def initialize(self) -> None:
        try:
            buckets = self._client.list_buckets().get("Buckets", [])
            if not any(bucket.get("Name") == self.bucket for bucket in buckets):
                options: dict[str, Any] = {"Bucket": self.bucket}
                if self.region != "us-east-1":
                    options["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.region
                    }
                self._client.create_bucket(**options)
            if self.cors_allowed_origins:
                self._client.put_bucket_cors(
                    Bucket=self.bucket,
                    CORSConfiguration={
                        "CORSRules": [
                            {
                                "AllowedHeaders": ["*"],
                                "AllowedMethods": ["PUT", "GET", "HEAD"],
                                "AllowedOrigins": self.cors_allowed_origins,
                                "ExposeHeaders": ["ETag"],
                                "MaxAgeSeconds": 3600,
                            }
                        ]
                    },
                )
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError("Unable to initialize the document bucket.") from exc

    def close(self) -> None:
        self._client.close()
        if self._public_client is not self._client:
            self._public_client.close()

    def health(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return True
        except (BotoCoreError, ClientError):
            return False

    def upload(
        self,
        path: Path,
        object_key: str,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        extra_args: dict[str, Any] = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        try:
            self._client.upload_file(
                str(path),
                self.bucket,
                object_key,
                ExtraArgs=extra_args,
            )
        except (BotoCoreError, ClientError, OSError) as exc:
            raise S3StorageError(
                "Unable to upload the document to object storage."
            ) from exc

    def delete(self, object_key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=object_key)
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError(
                "Unable to delete the document from object storage."
            ) from exc

    def object_size(self, object_key: str) -> int:
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=object_key)
            return int(response["ContentLength"])
        except (BotoCoreError, ClientError, KeyError, TypeError, ValueError) as exc:
            raise S3StorageError("Unable to inspect the uploaded document.") from exc

    def download(self, object_key: str, destination: Path) -> None:
        try:
            self._client.download_file(self.bucket, object_key, str(destination))
        except (BotoCoreError, ClientError, OSError) as exc:
            raise S3StorageError("Unable to download the uploaded document.") from exc

    def create_download_url(self, object_key: str, filename: str) -> str:
        safe_filename = filename.replace('"', "")
        try:
            return self._public_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": object_key,
                    "ResponseContentDisposition": (
                        f'attachment; filename="{safe_filename}"'
                    ),
                },
                ExpiresIn=self.presigned_url_expiry_seconds,
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError("Unable to create a document download URL.") from exc

    def create_upload_url(self, object_key: str) -> str:
        try:
            return self._public_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=self.presigned_url_expiry_seconds,
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError("Unable to create a document upload URL.") from exc

    def uri(self, object_key: str) -> str:
        return f"s3://{self.bucket}/{object_key}"
