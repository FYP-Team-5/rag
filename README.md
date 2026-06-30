# Rubric RAG service

A self-hosted FastAPI and LangChain microservice for ingesting grading rubrics and
retrieving the most relevant rubric passages. It deliberately does not call an LLM:
the separate grading service sends a question to `/api/v1/search`, then puts the
returned passages into its grading prompt.

## What is included

- FastAPI upload, catalog, download, deletion, and semantic-search endpoints
- LangChain loaders and recursive text splitting for PDF, DOCX, TXT, and Markdown
- Remote embeddings through an OpenAI-compatible `embeddings` microservice
- Qdrant vector storage with rubric/course payload indexes
- Persistent Docker volumes for Qdrant, original files, and manifests
- Optional API-key authentication, upload limits, checksums, health checks, and tests

```text
Rubric admin  --upload-->  FastAPI/LangChain  ---> original-file volume
                                     |  |
Grading LLM service --search-------->+  +-----> embeddings service
                                     |
                                     +--------> Qdrant vector database
```

## Run it

Requirements: Docker with Docker Compose v2 and a reachable embeddings service that
implements the contract below.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

The API probes the embeddings service during startup and will not become ready if it
cannot obtain an embedding. Once healthy:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc UI: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
- API health: <http://localhost:8000/health>
- Qdrant dashboard: <http://localhost:6333/dashboard>

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
It must also expose `GET /health` (configured with `EMBEDDINGS_HEALTH_URL`) and return
any 2xx response when ready. If `EMBEDDINGS_API_KEY` is configured, the RAG service
sends the same Bearer token to both endpoints.

The client batches document chunks according to `EMBEDDINGS_BATCH_SIZE`. A single
query string is sent for semantic search. At startup, it sends the input
`["embedding dimension probe"]`; the returned vector length creates a new Qdrant
collection or validates an existing collection.

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
EMBEDDINGS_HEALTH_URL=https://embeddings.internal/health
EMBEDDINGS_MODEL=your-model-name
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
| `GET` | `/api/v1/rubrics/{id}/chunks` | All processed chunks in document order |
| `GET` | `/api/v1/rubrics/{id}/download` | Original rubric file |
| `DELETE` | `/api/v1/rubrics/{id}` | Original, manifest, and all vectors |
| `POST` | `/api/v1/search` | Relevant chunks with scores and metadata |

## Configuration

All settings are environment variables; see [.env.example](.env.example). Important
ones are the embeddings URL/model/key, collection name, chunk size/overlap, upload
limit, CORS origins, and API key. Changing the embedding model after data has been
indexed changes the meaning of stored vectors even when their dimensions match. Use
a new `QDRANT_COLLECTION` and re-upload rubrics whenever changing models.

The Compose setup is suitable for local development and a single-host deployment.
Before internet-facing production use, put the API behind TLS, set `API_KEY`, do not
publish Qdrant's port, back up both volumes, and use a highly available Qdrant
deployment if downtime or data loss is unacceptable.

## Develop and test locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Stop services without deleting data using `docker compose down`. To intentionally
delete all stored rubrics and vectors, run `docker compose down -v`.
