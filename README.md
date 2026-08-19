# Rubric RAG service

A self-hosted FastAPI service that stores versioned grading rubrics for courses and
exams/quizzes. It streams original documents to SeaweedFS, stores lifecycle and
ownership metadata in PostgreSQL, chunks documents in the background, directly asks
an external model container for embeddings, and writes the vectors to Qdrant.

The grading service reads rubric metadata directly from PostgreSQL and retrieves
exact question-mapped chunks from Qdrant. Semantic search remains available for
instructor discovery and other RAG consumers.

```text
Instructor --upload rubric + course/exam/version--> RAG API
                       |               |               |
                       |               |               +-> PostgreSQL metadata
                       |               +------------------> SeaweedFS original
                       +--chunk text--> embedding model
                                              |
                       +--vectors + metadata<-+---> Qdrant points

Instructor --map chunk indexes--> Grading API
Frontend --student attempt/answers--> Grading API --exact IDs--> Qdrant
```

## Responsibilities

- Accept PDF, DOCX, TXT, and Markdown rubric uploads.
- Require `course_id` and `exam_id` on every new rubric.
- Allow multiple rubric versions per exam, with one record per `(exam_id, version)`.
- Store the original document, checksum, processing status, and ordered chunk IDs.
- Call the embedding model container directly for document batches and search
  queries, validate its vectors, and write those vectors to Qdrant.
- Add rubric, course, exam, version, document, and chunk-index metadata to Qdrant.
- List and filter rubric versions by course or exam.
- Return chunks in document order for question mapping.
- Archive obsolete versions without deleting evidence needed by attempts.

The RAG service does not own courses, exams, questions, students, attempts, answers,
or grades. Those are managed by the grading service.

### Database-backed models

| Model | Attributes | Purpose |
|---|---|---|
| `Rubric` | `id`, `title`, `version`, `course_id`, `exam_id`, `filename`, `content_type`, `size_bytes`, `sha256`, `chunk_count`, `processed`, `processing_status`, `processing_error`, `archived`, `uploaded_at`, `metadata` | Public domain representation of persisted rubric metadata. |
| `StoredRubric` | All `Rubric` fields plus `document_id`, `s3_bucket`, `s3_object_key`, `chunk_ids` | Full `rubrics` table model, including internal object-storage and Qdrant references. |
| Qdrant chunk record | point `id`, embedding vector, `page_content`, metadata (`document_id`, `rubric_id`, optional `course_id`/`exam_id`, `version`, `chunk_index`, custom metadata) | Stores an embedded document chunk for exact retrieval and semantic search. |

### DTOs

DTO definitions live in `app/dto/`; they describe HTTP and service-boundary payloads rather than database entities.

| DTO | Attributes | Purpose |
|---|---|---|
| `SearchRequest` | `query`, `rubric_id`, `course_id`, `exam_id`, `k`, `score_threshold` | Validates semantic-search input and optional filters. |
| `SearchResult` | `content`, `score`, `metadata` | Represents one ranked Qdrant result. |
| `SearchResponse` | `query`, `results` | Search endpoint response. |
| `RubricList` | `total`, `items` | Paginated/list response containing public rubrics. |
| `RubricProcessingStatus` | `id`, `document_id`, `processed`, `processing_status`, `processing_error`, `chunk_count` | Reports asynchronous ingestion state. |
| `RubricChunk` | `content`, `metadata` | Public representation of a retrieved chunk. |
| `RubricChunksResponse` | `rubric_id`, `chunks` | Ordered chunk-list response for a rubric. |
| `ArchiveResponse` | `id`, `archived` | Confirms that a rubric was archived. |
| `HealthResponse` | `status`, `qdrant`, `postgres`, `s3`, `collection` | Health endpoint response. |

## Embedding execution boundary

Embedding orchestration runs in this RAG service. The model remains isolated in
another Docker container and only performs inference:

```text
RAG API: extract -> chunk -> batch text -> HTTP POST /v1/embeddings
                                             |
Embedding model container: text ----------> vectors
                                             |
RAG API: validate count/dimensions <---------+
         attach metadata -> direct Qdrant upsert
```

For semantic search, RAG sends query text to the same model endpoint and passes the
returned query vector directly to Qdrant. The model container does not need
PostgreSQL, S3, or Qdrant credentials. There is no intermediate embeddings service
inside this repository.

## Run with Docker

Requirements are Docker Compose v2 and an OpenAI-compatible embedding model server.
PostgreSQL, Qdrant, and SeaweedFS are included.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

The base Compose file does not start the embedding model. Before uploading a rubric,
either declare that container as `embeddings` in `compose.override.yaml`, or attach
an already-running model container after the RAG network exists:

```bash
docker network connect --alias embeddings rubric-rag_default <model-container-name>
```

With the default configuration, the model must listen on port `8000` inside that
network and expose `/v1/embeddings`. Set `EMBEDDINGS_URL` when its container name,
port, or route differs.

Startup creates or upgrades the rubric table, initializes the Qdrant collection and
payload indexes, and creates the S3 bucket. It does not call embeddings until an
upload is processed or semantic search is used.

- API and Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health: <http://localhost:8000/health>
- Qdrant dashboard: <http://localhost:6333/dashboard>
- SeaweedFS S3 endpoint: <http://localhost:8333>

If `API_KEY` is configured, include `X-API-Key` on every `/api/v1` call. `/health`
remains unauthenticated.

## Complete user flows

### 1. Operator checks readiness

```bash
curl http://localhost:8000/health
```

Readiness covers PostgreSQL, Qdrant, and SeaweedFS. The embedding model is a runtime
dependency and intentionally is not part of startup readiness. The RAG API can be
healthy while an upload later fails processing because the model is unavailable;
check each upload's status endpoint.

### 2. Instructor creates the course and exam catalog

Create the course, exam/quiz, questions, and attempt limit through grading before
upload. The exam's rubric ID must match the RAG upload:

```bash
curl -X POST http://localhost:8001/api/v1/courses \
  -H 'Content-Type: application/json' \
  -d '{"id":"HIST-101","title":"World History"}'

curl -X POST http://localhost:8001/api/v1/courses/HIST-101/exams \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"history-midterm",
    "title":"History midterm",
    "type":"exam",
    "max_attempts":1,
    "rubric_id":"history-midterm-rubric-v1",
    "questions":[
      {"id":"history-midterm-q1","prompt":"Explain the primary cause.","max_score":10}
    ]
  }'
```

### 3. Instructor uploads a rubric

Uploads use multipart form data. `file`, `course_id`, and `exam_id` are required.
`rubric_id` is optional at the RAG layer but should always be supplied for an exam.
`title`, `version` (default `1`), and JSON-object `metadata` are optional.

```bash
curl -X POST http://localhost:8000/api/v1/rubrics \
  -F 'file=@./examples/history-rubric.pdf' \
  -F 'rubric_id=history-midterm-rubric-v1' \
  -F 'title=History midterm rubric' \
  -F 'version=1' \
  -F 'course_id=HIST-101' \
  -F 'exam_id=history-midterm' \
  -F 'metadata={"term":"fall","teacher":"Ada"}'
```

A successful `200` means the original is stored and background processing started;
it does not mean embeddings are complete. The response starts with
`processed=false`, `processing_status="processing"`, and `chunk_count=0`.

Invalid files, IDs, metadata, or empty documents return `422`. Duplicate rubric IDs
or duplicate `(exam_id, version)` pairs return `409`.

### 4. Instructor waits for processing

```bash
curl http://localhost:8000/api/v1/rubrics/history-midterm-rubric-v1/status
```

Poll until `processed` is `true` and `processing_status` is `completed`. A failed
result includes `processing_error`. On failure, document-owned vectors are removed
from Qdrant while the original file remains available for inspection. Work
interrupted by restart is also marked failed.

### 5. Instructor lists and filters rubric versions

```bash
curl 'http://localhost:8000/api/v1/rubrics?course_id=HIST-101'
curl 'http://localhost:8000/api/v1/rubrics?exam_id=history-midterm'
curl 'http://localhost:8000/api/v1/rubrics?exam_id=history-midterm&include_archived=true'
```

The catalog is newest-first and paginated with `offset` and `limit`. Archived
versions are hidden unless `include_archived=true`.

### 6. Instructor inspects metadata

```bash
curl http://localhost:8000/api/v1/rubrics/history-midterm-rubric-v1
```

This returns lifecycle, ownership, checksum, file, and custom metadata without
exposing internal S3 keys or the ordered point-ID list.

### 7. Instructor inspects chunks and maps questions

After processing, retrieve all chunks in document order:

```bash
curl http://localhost:8000/api/v1/rubrics/history-midterm-rubric-v1/chunks
```

Each result includes `chunk_index`. Map all indexes containing relevant criteria to
each grading question:

```bash
curl -X PUT \
  http://localhost:8001/api/v1/exams/history-midterm/questions/history-midterm-q1/rubric-chunks \
  -H 'Content-Type: application/json' \
  -d '{"chunk_indexes":[0,1]}'
```

Grading validates rubric course/exam ownership and each index. Student attempts
cannot start until every question is mapped.

### 8. User downloads the original rubric

```bash
curl -L -o history-midterm-rubric.pdf \
  http://localhost:8000/api/v1/rubrics/history-midterm-rubric-v1/download
```

The API returns a `307` redirect to a short-lived presigned URL. Configure
`S3_PUBLIC_ENDPOINT_URL` with an address the caller's browser can reach.

### 9. Instructor performs semantic search

Filter by the narrowest known ownership keys to avoid cross-exam context:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"How is supporting evidence scored?",
    "course_id":"HIST-101",
    "exam_id":"history-midterm",
    "rubric_id":"history-midterm-rubric-v1",
    "k":5,
    "score_threshold":0.4
  }'
```

Results contain text, cosine-similarity score, and payload metadata. Grading instead
uses PostgreSQL's point IDs and exact Qdrant retrieval, so mapped criteria are not
omitted by similarity ranking.

### 10. Instructor uploads and activates a new version

Use a new rubric ID and version for the same exam:

```bash
curl -X POST http://localhost:8000/api/v1/rubrics \
  -F 'file=@./history-rubric-v2.pdf' \
  -F 'rubric_id=history-midterm-rubric-v2' \
  -F 'version=2' \
  -F 'course_id=HIST-101' \
  -F 'exam_id=history-midterm'
```

After processing, activate it in grading and review mappings:

```bash
curl -X PUT http://localhost:8001/api/v1/exams/history-midterm/rubric \
  -H 'Content-Type: application/json' \
  -d '{"rubric_id":"history-midterm-rubric-v2"}'
```

The current grading implementation does not snapshot question mappings. Finish all
in-progress attempts for this exam before activation or remapping; see the grading
README limitations.

### 11. Instructor archives an obsolete version

```bash
curl -X DELETE \
  http://localhost:8000/api/v1/rubrics/history-midterm-rubric-v1
```

This is a soft archive despite the `DELETE` method. It sets `archived=true` and hides
the version from normal listings. PostgreSQL metadata, the S3 original, and Qdrant
chunks remain available so previously saved attempts retain their evidence. An
archived version cannot be selected for a new attempt.

## API summary

| Method | Path | User action |
|---|---|---|
| `GET` | `/health` | Check required storage dependencies |
| `POST` | `/api/v1/rubrics` | Upload a course/exam rubric version |
| `GET` | `/api/v1/rubrics` | List/filter rubric versions |
| `GET` | `/api/v1/rubrics/{id}` | Inspect rubric metadata |
| `GET` | `/api/v1/rubrics/{id}/status` | Poll background processing |
| `GET` | `/api/v1/rubrics/{id}/chunks` | Inspect ordered chunks for mapping |
| `GET` | `/api/v1/rubrics/{id}/download` | Download the original |
| `DELETE` | `/api/v1/rubrics/{id}` | Soft-archive a rubric version |
| `POST` | `/api/v1/search` | Semantically search rubric chunks |

## Embedding model container contract

`EMBEDDINGS_URL` must expose OpenAI-compatible `POST /v1/embeddings`. The response
must contain one consistently sized numeric vector per input under `data`. When
`EMBEDDINGS_API_KEY` is set, it is sent as a Bearer token.

RAG sends document chunks in batches and sends a one-item batch for a search query:

```json
{
  "model": "BAAI/bge-small-en-v1.5",
  "input": ["first rubric chunk", "second rubric chunk"]
}
```

The model container must respond with one vector for each input. RAG sorts by
`index`, verifies that no index or vector is missing, converts values to floats, and
checks every vector against `EMBEDDINGS_DIMENSION` before writing anything to
Qdrant:

```json
{
  "data": [
    {"index": 0, "embedding": [0.012, -0.034, 0.056]},
    {"index": 1, "embedding": [0.078, -0.090, 0.123]}
  ]
}
```

`EMBEDDINGS_DIMENSION` creates or validates the Qdrant collection and must match the
model. Changing embedding models changes vector meaning even when dimensions match;
use a new collection and re-upload rubrics.

The model container must be addressable from the RAG API container. If it is already
running, attach it to the RAG network and give it the default DNS alias:

```bash
docker network connect --alias embeddings rubric-rag_default <model-container-name>
```

The default `EMBEDDINGS_URL=http://embeddings:8000/v1/embeddings` will then resolve
directly to that container. Alternatively, declare the model in
`compose.override.yaml` so Compose attaches it automatically:

```yaml
services:
  embeddings:
    image: your-registry/your-embedding-model-server:tag
    expose:
      - "8000"
```

Or point `.env` elsewhere:

```dotenv
EMBEDDINGS_URL=https://embeddings.internal/v1/embeddings
EMBEDDINGS_MODEL=your-model-name
EMBEDDINGS_DIMENSION=768
EMBEDDINGS_API_KEY=replace-me
```

## Storage and operational behavior

Uploads stream to a temporary file while the original is sent to S3-compatible
storage. PostgreSQL first records `processing`; extraction and chunking happen in
RAG, inference happens in the model container, and RAG directly upserts each finished
vector plus `page_content` and metadata into Qdrant. PostgreSQL is then completed
with the ordered point IDs. Points are validated by immutable `document_id` as well
as rubric ownership.

Stop without deleting data using `docker compose down`. Use
`docker compose down -v` only when intentionally deleting all local documents,
metadata, and vectors.

For production, change default credentials, require authentication, use TLS and
restricted CORS, stop publishing storage ports, and back up all three data volumes.

## Develop, test, and release

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ruff==0.16.3
make ci
```

The pull-request [CI workflow](.github/workflows/ci.yml) installs dependencies in
the YAML, then runs Ruff and pytest. Configure both as required branch-protection
checks to prevent a failing PR from being merged. Post-merge actions cannot undo an
already completed merge.

The [post-merge workflow](.github/workflows/post-merge.yml) reruns lint and tests on
every push to `main`. It creates the next tag, starting at `v0.1`, only after both
pass. The GHCR Docker build/publish job is included but completely commented out.
