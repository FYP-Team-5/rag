from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    postgres: str
    s3: str
    collection: str
