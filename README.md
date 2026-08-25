# Course Material RAG service

A FastAPI service for course materials uploaded directly from the frontend to
S3-compatible storage. It stores upload and processing state in PostgreSQL, chunks
uploaded PDF, DOCX, TXT, and Markdown files, requests embeddings from an external
model server, and stores vectors in Qdrant.

## Upload lifecycle

```text
Frontend                  RAG API                 PostgreSQL / S3 / Qdrant
   |                         |                              |
   |-- request PUT URL ----->|-- save awaiting upload ---->| PostgreSQL
   |<-- id + signed URL -----|                              |
   |-- PUT file directly --------------------------------->| S3
   |-- status=uploaded ----->|-- verify + download -------->| S3
   |<-- status=processing ---|-- chunk + embed ----------->| Qdrant
   |                         |-- status=completed --------->| PostgreSQL
```

If the frontend reports `failed`, the service deletes the corresponding PostgreSQL
record. It only starts processing after an `uploaded` callback and after verifying
that the object exists and is within the configured size limit.

The old rubric upload API is no longer exposed. The legacy `rubrics` table is left
untouched to avoid destructive data loss, but this application no longer reads it.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check PostgreSQL, S3, and Qdrant readiness |
| `POST` | `/api/v1/course-material/create_presigned` | Persist metadata and create a signed S3 PUT URL |
| `POST` | `/api/v1/course-material/{id}/upload-status` | Report `uploaded`, `failed`, or `upload_failed` |
| `GET` | `/api/v1/course-material` | List course materials |
| `GET` | `/api/v1/course-material/{id}` | Read course-material metadata |
| `GET` | `/api/v1/course-material/{id}/status` | Poll processing state |
| `POST` | `/api/v1/search` | Search embedded course materials, optionally by `course_id` |

`/health`, `/course-material`, and `/search` are implemented in separate controller
modules. If `API_KEY` is configured, send it as `X-API-Key` for `/api/v1` routes.

## Example

Request an upload URL:

```bash
curl -X POST http://localhost:8000/api/v1/course-material/create_presigned \
  -H 'Content-Type: application/json' \
  -d '{
    "course_id":"d87cecc2-e224-4e91-aeef-8c4773452674",
    "filename":"week-01.pdf"
  }'
```

The response includes `course_material_id`, `presigned_url`, `object_key`,
and `expires_in`. Upload the file directly to the returned URL:

```bash
curl -X PUT "$PRESIGNED_URL" \
  --upload-file ./week-01.pdf
```

Report the result:

```bash
curl -X POST \
  http://localhost:8000/api/v1/course-material/$COURSE_MATERIAL_ID/upload-status \
  -H 'Content-Type: application/json' \
  -d '{"status":"uploaded"}'
```

Use `{"status":"failed"}` (or `upload_failed`) when the direct upload fails. Poll
`/api/v1/course-material/{id}/status` until the state is `completed` or `failed`.

## Run

Docker Compose includes PostgreSQL, Qdrant, and SeaweedFS. An OpenAI-compatible
embedding server must be reachable at `EMBEDDINGS_URL`.

```bash
cp .env.example .env
docker compose up --build -d
```

Swagger UI is available at <http://localhost:8000/docs>.

Important settings:

- `DATABASE_URL`
- `S3_ENDPOINT_URL` and `S3_PUBLIC_ENDPOINT_URL`
- `S3_COURSE_MATERIALS_BUCKET`
- `S3_PRESIGNED_URL_EXPIRY_SECONDS`
- `QDRANT_URL` and `QDRANT_COLLECTION`
- `EMBEDDINGS_URL`, `EMBEDDINGS_MODEL`, and `EMBEDDINGS_DIMENSION`
- `MAX_UPLOAD_SIZE_MB`

Run checks with:

```bash
make lint
make test
```
