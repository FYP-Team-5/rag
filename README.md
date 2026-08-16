# Rubric RAG service

A self-hosted FastAPI and LangChain microservice for ingesting grading rubrics and
retrieving the most relevant rubric passages. It deliberately does not call an LLM:
the separate grading service sends a question to `/api/v1/search`, then puts the
returned passages into its grading prompt.

## What is included

- FastAPI asynchronous upload, processing-status, catalog, download, deletion, and
  semantic-search endpoints
- LangChain loaders and recursive text splitting for PDF, DOCX, TXT, and Markdown
- Remote embeddings through an OpenAI-compatible `embeddings` microservice
- Qdrant vector storage with rubric/course payload indexes
- SeaweedFS S3-compatible storage for original documents and presigned downloads
- PostgreSQL metadata storage for document, object, and chunk identifiers
- Persistent Docker volumes for Qdrant, SeaweedFS, and PostgreSQL
- Optional API-key authentication, upload limits, checksums, health checks, and tests

```text
Rubric admin  --upload-->  FastAPI ---------> SeaweedFS S3 (original file)
                              |  |
                              |  +----------> PostgreSQL (status + metadata)
                              |
                              +--background-> chunking -> embeddings -> Qdrant

Frontend/status service --------poll--------> GET /rubrics/{id}/status
```

## Run it

Requirements: Docker with Docker Compose v2. Qdrant, PostgreSQL, and SeaweedFS are
included in the Compose project. The embeddings service may be started separately
when ingestion or semantic search is needed.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Startup validates Qdrant, creates the PostgreSQL metadata table, ensures the S3
bucket exists, and does not call the embeddings service. Once healthy:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc UI: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
- API health: <http://localhost:8000/health>
- Qdrant dashboard: <http://localhost:6333/dashboard>
- SeaweedFS S3 endpoint: <http://localhost:8333>

If `API_KEY` is set in `.env`, add `-H "X-API-Key: $API_KEY"` to all `/api/v1`
requests. In Swagger, click **Authorize**, enter the key, and then use **Try it out**
on any endpoint. The health endpoint remains unauthenticated for orchestration.

## Embeddings microservice contract

This repository does not implement the embeddings service. By default, the API calls
`http://embeddings:8000/v1/embeddings`, where `embeddings` is the Docker DNS name of
that separate microservice. It must expose an OpenAI-compatible endpoint.

Request:

```http
POST /v1/embeddings
Content-Type: application/json
Authorization: Bearer <EMBEDDINGS_API_KEY>  # only when configured

{
  "model": "BAAI/bge-small-en-v1.5",
  "input": ["first rubric chunk", "second rubric chunk"]
}
```

Required response shape:

```json
{
  "data": [
    {"index": 0, "embedding": [0.012, -0.034, 0.056]},
    {"index": 1, "embedding": [0.078, -0.090, 0.123]}
  ],
  "model": "BAAI/bge-small-en-v1.5"
}
```

The service must return one non-empty, consistently sized numeric vector per input.
Response items may be out of order when their zero-based `index` values are present.
If `EMBEDDINGS_API_KEY` is configured, the RAG service sends it to the embeddings
endpoint as a Bearer token.

The client batches document chunks according to `EMBEDDINGS_BATCH_SIZE`. A single
query string is sent for semantic search. `EMBEDDINGS_DIMENSION` creates a new Qdrant
collection or validates an existing collection during startup. It must match the
configured embedding model. Returned vectors are checked against this value at
runtime.

The embeddings service is a runtime dependency, not a startup dependency. An upload
can therefore return `200` after its original file reaches SeaweedFS even when the
embeddings service is unavailable. In that case its asynchronous status changes to
`failed`; semantic search itself still returns `502` when embeddings are unavailable.

## Document storage

Uploads are streamed to a temporary file. Text extraction and chunking begin while
that file is uploaded to the configured SeaweedFS S3 bucket. PostgreSQL initially
stores the document with `processed=false` and `processing_status="processing"`.
Once the S3 upload succeeds, the API returns `200` without waiting for embedding or
Qdrant writes. The temporary file is removed when background processing finishes.

Every Qdrant chunk contains the immutable PostgreSQL `document_id`. When processing
completes, PostgreSQL is updated to `processed=true`, `processing_status="completed"`,
and receives the final chunk IDs. On failure, all Qdrant points matching that
`document_id` are deleted and PostgreSQL records `processing_status="failed"` plus
an error message. The original SeaweedFS document is retained so it can still be
downloaded or inspected. Processing interrupted by a service restart is also marked
failed and its document-owned vectors are cleaned up.

`GET /api/v1/rubrics/{id}/download` looks up the object metadata in PostgreSQL and
redirects the caller to a short-lived presigned S3 URL. Set
`S3_PUBLIC_ENDPOINT_URL` to the SeaweedFS address that browsers can reach; the
default is `http://localhost:8333` for local development.

If the embeddings container is managed in the same Compose project, add it as a
service named `embeddings`, for example in `compose.override.yaml`:

```yaml
services:
  embeddings:
    image: your-registry/your-embeddings-service:tag
    expose:
      - "8000"
```

Compose then resolves the default `embeddings` hostname automatically. If it runs
elsewhere, set both URLs in `.env`, for example:

```dotenv
EMBEDDINGS_URL=https://embeddings.internal/v1/embeddings
EMBEDDINGS_MODEL=your-model-name
EMBEDDINGS_DIMENSION=768
EMBEDDINGS_API_KEY=replace-me
```

## Upload a rubric

Uploads use multipart form data because the request contains a file. `metadata` is
an optional JSON object encoded as a form string.

```bash
curl -X POST http://localhost:8000/api/v1/rubrics \
  -F 'file=@./examples/history-rubric.pdf' \
  -F 'rubric_id=history-short-answer-v1' \
  -F 'title=History short-answer rubric' \
  -F 'version=1.0' \
  -F 'course_id=HIST-101' \
  -F 'metadata={"term":"fall","teacher":"Ada"}'
```

Only the file is required. If omitted, `rubric_id` is generated, `title` comes from
the filename, and `version` is `1`.

A successful upload response means the original document is durably stored, not
that chunking and embeddings have completed. The response and normal metadata
endpoints include:

```json
{
  "id": "history-short-answer-v1",
  "processed": false,
  "processing_status": "processing",
  "processing_error": null,
  "chunk_count": 0
}
```

Poll the status endpoint until `processed` becomes `true` or
`processing_status` becomes `failed`:

```bash
curl http://localhost:8000/api/v1/rubrics/history-short-answer-v1/status
```

## Retrieve context for grading

The grading service should normally pass `rubric_id`; this prevents passages from
another assignment from entering the grading prompt.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How should factual accuracy and supporting evidence be scored?",
    "rubric_id": "history-short-answer-v1",
    "k": 5
  }'
```

Example response:

```json
{
  "query": "How should factual accuracy and supporting evidence be scored?",
  "results": [
    {
      "content": "4 points: The response is factually accurate...",
      "score": 0.83,
      "metadata": {
        "rubric_id": "history-short-answer-v1",
        "course_id": "HIST-101",
        "page": 0,
        "chunk_index": 2
      }
    }
  ]
}
```

`score` is cosine similarity (higher is closer). An optional `score_threshold` can
exclude weak matches. Search can also be filtered by `course_id`.

For a short rubric, a grader may need every criterion rather than semantically
selected passages. `GET /api/v1/rubrics/{id}/chunks` returns all chunks in document
order, ready to place in a prompt. Use semantic search for large rubric/reference
sets and the chunks endpoint when complete rubric coverage is required.

## Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness check |
| `GET` | `/api/v1/rubrics` | Paginated rubric catalog |
| `GET` | `/api/v1/rubrics/{id}` | Rubric metadata |
| `GET` | `/api/v1/rubrics/{id}/status` | Asynchronous processing status |
| `GET` | `/api/v1/rubrics/{id}/chunks` | All processed chunks in document order |
| `GET` | `/api/v1/rubrics/{id}/download` | Redirect to a presigned S3 download |
| `DELETE` | `/api/v1/rubrics/{id}` | S3 object, PostgreSQL metadata, and vectors |
| `POST` | `/api/v1/search` | Relevant chunks with scores and metadata |

## Configuration

All settings are environment variables; see [.env.example](.env.example). Important
ones are the PostgreSQL connection, internal/public S3 endpoints, S3 bucket and
credentials, embeddings URL/model/dimension/key, Qdrant collection, chunk settings,
upload limit, CORS origins, and API key. Changing the embedding model after data has
been indexed changes the meaning of stored vectors even when their dimensions match.
Use a new `QDRANT_COLLECTION` and re-upload rubrics whenever changing models.

The Compose setup is suitable for local development and a single-host deployment.
Before internet-facing production use, put the API and S3 endpoint behind TLS, change
all default credentials, set `API_KEY`, avoid publishing database administration
ports, back up all three data volumes, and use highly available storage deployments
if downtime or data loss is unacceptable.

## Develop and test locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ruff==0.16.3
make ci
```

The [GitHub Actions workflow](.github/workflows/ci.yml) runs on every push and pull
request. Its lint and test jobs run independently on Python 3.12, matching the
Docker image. Dependencies are installed directly in the workflow from
`requirements.txt`; there is no separate CI requirements file. The lint job installs
the pinned Ruff version and runs `python -m ruff check app tests`. The test job runs
`python -m pytest -q`.

Stop services without deleting data using `docker compose down`. To intentionally
delete all stored documents, metadata, and vectors, run `docker compose down -v`.
