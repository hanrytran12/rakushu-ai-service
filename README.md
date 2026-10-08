# Rakushu AI Service

AI service for Rakushu's video ingestion, speech recognition, Japanese NLP, knowledge lookup, translation, and Bunsetsu analysis pipeline.

## Architecture

```text
Client
  |
  v
FastAPI
  |
  +-- /api/v1/pipeline
  |      |
  |      +-- URL / upload input
  |      +-- SSE streaming
  |      v
  |   pipeline_streamer
  |      |
  |      +-- media / YouTube
  |      +-- ASR
  |      +-- NLP / knowledge
  |      +-- LLM translation
  |      +-- Bunsetsu
  |      +-- output store
  |
  +-- /api/v1/nlp
  |      +-- completed-video tokens
  |      +-- completed-video Bunsetsu
  |
  +-- /api/v1/dictionary
         +-- dictionary lookup

Services and domain logic live under src/services.
Shared models live under src/models.
Pipeline orchestration lives under src/pipeline.
HTTP routing lives under src/api.
```

## Project Structure

```text
src/
├── api/
│   ├── api_server.py          # FastAPI application entry point
│   ├── api_models.py          # HTTP request/response models
│   ├── pipeline_router.py     # Video pipeline endpoints
│   ├── nlp_router.py          # NLP output endpoints
│   └── dictionary_router.py   # Dictionary/internal sync endpoints
├── pipeline/
│   ├── pipeline_streamer.py   # End-to-end streaming pipeline
│   ├── pipeline_runner.py     # Pipeline execution
│   └── pipeline_formatter.py  # Pipeline output formatting
├── services/
│   ├── asr_service.py
│   ├── nlp_service.py
│   ├── llm_service.py
│   ├── knowledge_service.py
│   ├── oov_service.py
│   ├── bunsetsu_service.py
│   ├── bunsetsu_graph_service.py
│   ├── hierarchical_matcher.py
│   ├── ginza_reference_service.py
│   ├── media_inspector.py
│   ├── youtube_service.py
│   └── pipeline_output_store.py
├── models/                    # Domain/data schemas
├── utils/                     # Shared utilities
└── demo-page.html             # SSE demo UI

docs/
├── system-architecture.md
└── code-standards.md

tests/
└── test-*.py
```

## Running the API

From the repository root:

```bash
python -m uvicorn src.api.api_server:app --port 8000 --reload
```

The service is available at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Demo UI: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`

## Main API

### 1. Process a video URL

`POST /api/v1/pipeline/stream-url`

Request:

```json
{
  "url": "https://www.youtube.com/watch?v=..."
}
```

The endpoint returns an SSE stream containing progressive pipeline events.

Compatibility aliases are also available:

- `POST /api/v1/pipeline/process-url`
- `POST /api/v1/pipeline/process-url/stream`

### 2. Process an uploaded video

`POST /api/v1/pipeline/stream-upload`

Send a multipart form upload using the `file` field. The `video` field is also accepted for compatibility.

Compatibility aliases:

- `POST /api/v1/pipeline/upload-video`
- `POST /api/v1/pipeline/upload-stream`
- `POST /api/v1/pipeline/upload-video/stream`
- `POST /api/v1/pipeline/stream-video`

Supported media extensions include:

`.mp4`, `.mov`, `.m4a`, `.mp3`, `.wav`, `.mkv`, `.webm`.

### 3. Read completed NLP tokens

`GET /api/v1/nlp/videos/{video_id}/tokens`

Returns the GiNZA JSON and flattened token data stored by the completed pipeline.

### 4. Read completed Bunsetsu analysis

`GET /api/v1/nlp/videos/{video_id}/bunsetsu`

Builds Bunsetsu, dependency relations, roots, and sentence-level results from the stored NLP output.

### 5. Dictionary lookup

`GET /api/v1/dictionary/lookup?term=...`

Performs an offline dictionary lookup through the knowledge service.

### 6. Internal dictionary/OOV synchronization

These endpoints are intended for backend-to-AI-service synchronization:

- `POST /api/v1/internal/dictionary/sync`
- `POST /api/v1/internal/oov/sync-status`

## Pipeline Flow

The current end-to-end flow is intentionally kept as the primary API surface:

```text
Video URL / Upload
      ↓
Media preparation
      ↓
ASR
      ↓
Sentence segmentation / NLP
      ↓
Knowledge & candidate processing
      ↓
LLM translation
      ↓
Bunsetsu / sentence enrichment
      ↓
Pipeline output store
      ↓
SSE completed events
```

The service also contains recovery/validation logic for translation chunks so missing sentence results can be retried without restarting the entire translation job.

## Testing

Run the API regression tests:

```bash
python -m unittest tests/test-pipeline-api.py
python -m unittest tests/test-nlp-output-api.py
```

Run the full test suite:

```bash
python -m unittest discover -s tests -p "test-*.py"
```

Check Python compilation:

```bash
python -m compileall -q src tests
```

## Development Notes

- Keep HTTP concerns in `src/api`.
- Keep pipeline orchestration in `src/pipeline`.
- Keep business/domain logic in `src/services`.
- Keep schemas/data models in `src/models`.
- Avoid adding separate stage APIs unless there is a concrete consumer/use case.
- Preserve the existing end-to-end pipeline as the main production flow.
- Do not commit secrets or local generated data.

See `docs/system-architecture.md` and `docs/code-standards.md` for project-specific guidance.
