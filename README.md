# Rakushu Engine

## Japanese Language Processing Engine

Rakushu Engine is the language-processing engine behind the Rakushu Japanese-learning platform. It transforms Japanese video and audio into structured linguistic and learning data through media ingestion, ASR, Japanese NLP, knowledge matching, contextual LLM processing, OOV learning, and Bunsetsu analysis.

> **Positioning:** Rakushu Engine is a processing engine, not a standalone AI product. The main Rakushu API owns platform business workflows; Rakushu Engine owns computational language-processing workloads.

---

## 1. Project Overview

### 1.1 Project Name

| Item | Description |
| --- | --- |
| **Short name** | **Rakushu Engine** |
| **Full name** | **Rakushu — Japanese Language Processing Engine** |
| **Primary purpose** | Japanese video/audio language processing |
| **Backend** | Python |
| **API framework** | FastAPI |
| **Processing style** | Synchronous pipeline orchestration with progressive SSE events |
| **LLM runtime** | Local Ollama-compatible runtime |
| **Knowledge storage** | Local/offline linguistic knowledge sources |

### 1.2 Short Description

Rakushu Engine provides the computational language-processing layer for Rakushu.

It receives Japanese video or audio from a YouTube URL or uploaded media file, converts speech into text, segments the text into sentences, analyzes Japanese linguistic structure, resolves known dictionary knowledge, uses an LLM for contextual translation and candidate selection, discovers OOV vocabulary, and builds Bunsetsu-based learning structures.

The output is exposed through FastAPI endpoints and progressive Server-Sent Events (SSE) so downstream Rakushu services can consume both intermediate processing results and the final structured output.

---

## 2. Core Solutions

The current engine is organized around three major processing capabilities.

### Core Feature 01 — Media Ingestion & Japanese ASR

Rakushu Engine accepts Japanese video/audio as the starting point of the processing pipeline.

Key capabilities include:

- YouTube URL ingestion
- Uploaded video processing
- YouTube video ID extraction and propagation
- Media inspection and validation
- Audio/video processing through FFmpeg-compatible tooling
- Japanese speech recognition with Whisper / faster-whisper
- Subtitle segment generation
- Source metadata propagation through the pipeline
- Progressive processing status through SSE

The result of this stage is a timestamped Japanese transcription that becomes the input to downstream linguistic processing.

### Core Feature 02 — Japanese Linguistic Intelligence & Knowledge Matching

The engine converts raw Japanese transcription into structured linguistic knowledge.

Key capabilities include:

- Sentence segmentation
- Japanese tokenization
- POS and lemma analysis
- Reading and dependency metadata
- Bunsetsu-oriented analysis
- Grammar matching
- Phrase matching
- Compound-word matching
- Word / dictionary lookup
- Hierarchical knowledge matching
- Server-owned dictionary candidate generation
- OOV detection

Knowledge matching follows a server-controlled model:

```text
Japanese Token
      |
      +----> Grammar
      |
      +----> Phrase
      |
      +----> Compound Word
      |
      +----> Word / Dictionary
                  |
                  v
          Candidate Set
```

The engine does not allow the LLM to invent arbitrary dictionary meanings. Candidate meanings originate from engine-owned linguistic knowledge and are validated before being propagated downstream.

### Core Feature 03 — Contextual LLM Translation & OOV Learning

The engine uses a local LLM as a contextual processing component rather than as the source of truth for dictionary knowledge.

Key capabilities include:

- Japanese → Vietnamese contextual translation
- Local sentence context construction
- Chunked translation for long content
- Candidate selection using occurrence-level `token_key`
- Server-side validation of LLM selections
- Structured OOV learning
- Missing-sentence detection
- Targeted retry of missing sentences
- Adaptive chunk splitting
- Partial-result recovery

The LLM processing model is:

```text
Server Knowledge
      |
      v
Candidate Set
      |
      v
LLM Candidate Selection
      |
      v
Engine Resolver / Validation
      |
      v
Validated Meaning
```

This keeps the LLM responsible for contextual interpretation while keeping linguistic knowledge ownership inside the engine.

---

## 3. Applied Methods & Technologies

Rakushu Engine combines conventional language processing with LLM-assisted processing.

| Method / Technology | Application |
| --- | --- |
| **NLP** | Japanese sentence segmentation, tokenization, POS, lemma, reading and dependency analysis |
| **ASR** | Japanese speech-to-text transcription from video/audio |
| **LLM** | Contextual translation, candidate selection and structured OOV learning |
| **Knowledge Matching** | Resolving grammar, phrase, compound-word and dictionary knowledge |
| **OOV Detection** | Identifying language items not sufficiently covered by known knowledge |
| **Occurrence-level Selection** | Binding LLM candidate selection to the exact token occurrence |
| **Server-side Validation** | Rejecting invalid or hallucinated LLM selections |
| **Chunked Processing** | Bounded LLM requests for long videos |
| **Adaptive Recovery** | Retrying missing sentences and recursively splitting failed chunks |
| **Bunsetsu Analysis** | Building Japanese phrase structures for learning interfaces |
| **SSE** | Streaming intermediate pipeline results progressively |

### Relationship with SRS

SRS (Spaced Repetition System) is a learning-domain capability owned by the broader Rakushu platform rather than by the processing engine itself.

Rakushu Engine contributes the linguistic structures, translations, meanings, OOV candidates, and Bunsetsu data that can be consumed by SRS-oriented learning workflows.

---

# 4. Implementation

## 4.1 Processing Architecture

Rakushu Engine separates HTTP transport, pipeline orchestration, processing services, shared models, and infrastructure helpers.

```text
src/
├── api/       # FastAPI transport and routers
├── pipeline/  # End-to-end pipeline orchestration and streaming
├── services/  # ASR, NLP, LLM, Knowledge, OOV, Bunsetsu and integrations
├── models/    # Shared language and pipeline data contracts
└── utils/     # Shared infrastructure helpers
```

Dependency direction:

```text
API
 |
 v
Pipeline
 |
 v
Services
 |
 v
Models
```

Models are kept independent from API transport and pipeline orchestration.

### Layer responsibilities

#### API Layer — `src/api/`

Responsible for:

- FastAPI application configuration
- HTTP endpoints
- Request/response contracts
- File upload handling
- Streaming responses
- Dictionary endpoints
- NLP output endpoints
- Health checks

The API layer exposes processing capabilities but does not own the processing algorithms.

#### Pipeline Layer — `src/pipeline/`

Responsible for:

- End-to-end stage orchestration
- YouTube and upload input handling
- Stage sequencing
- Heartbeat-aware long-running operations
- Chunk planning
- Sentence processing
- Progressive SSE events
- Partial-result handling
- Final pipeline status

The pipeline is the execution coordinator of the engine.

#### Service Layer — `src/services/`

Responsible for individual processing capabilities:

- ASR
- NLP
- Knowledge matching
- LLM enrichment
- OOV handling
- Bunsetsu construction
- YouTube/media processing
- Pipeline output persistence
- Backend synchronization

#### Model Layer — `src/models/`

Contains shared processing contracts such as:

- Subtitle segments
- Sentence subtitles
- Tokens
- Dictionary entries
- OOV candidates
- Bunsetsu phrases
- Grammar/phrase/compound entries
- Matched knowledge units
- Pipeline result structures

#### Utility Layer — `src/utils/`

Contains reusable infrastructure helpers such as:

- SSE formatting
- Logging
- Shared pipeline helpers
- Common runtime utilities

---

## 4.2 End-to-End Pipeline

The main processing flow is:

```text
Video / Audio
     |
     v
[Media Ingestion]
     |
     v
[Whisper ASR]
     |
     v
[Sentence Segmentation]
     |
     v
[Japanese NLP]
     |
     v
[Knowledge Matching]
     |
     v
[Candidate Generation]
     |
     v
[LLM Chunk Translation]
     |
     v
[Candidate Validation]
     |
     +------> [OOV Learning]
     |
     v
[Bunsetsu Analysis]
     |
     v
[Structured Learning Output]
```

### Stage 0 — Media Source

For YouTube input, the engine extracts the canonical video ID and retrieves media metadata.

The video ID is propagated into the resulting pipeline data and SSE payloads so downstream systems can correlate processing results with the original source.

For uploaded media, the engine assigns an upload-scoped video ID.

### Stage 1 — ASR

Whisper / faster-whisper converts Japanese speech into timestamped text.

The ASR stage produces a subtitle/media segment containing:

- Video ID
- Source metadata
- Start time
- End time
- Japanese transcription

### Stage 2 — Japanese NLP

The engine segments transcription into sentences and analyzes each sentence.

NLP data can include:

- Surface form
- Lemma
- POS
- Reading
- Dependency information
- Token identity
- Bunsetsu-related information

GiNZA is used for Japanese linguistic analysis, with Janome available as a fallback tokenizer where applicable.

### Stage 3 — Knowledge Matching

Each sentence is analyzed against engine-owned linguistic knowledge.

The matching hierarchy includes:

```text
Token / Token Sequence
       |
       +--> Grammar
       +--> Phrase
       +--> Compound Word
       +--> Word / Dictionary
       |
       v
Matched Knowledge + Candidate Meanings
```

Candidate selection is occurrence-aware. A token occurrence receives its own `token_key`, allowing the engine to distinguish repeated surface forms appearing in different positions.

### Stage 4 — LLM Translation

Long content is divided into bounded translation chunks.

The current design uses both:

- Maximum character limits
- Maximum sentence-count limits

Conceptually:

```text
Translation Job
      |
      v
Chunk Planner
      |
      +---- Chunk 1
      +---- Chunk 2
      +---- Chunk 3
             |
             v
       LLM Batch Request
             |
             v
          Validate
             |
       +-----+------+
       |            |
    Complete      Missing
       |            |
       v            v
    Continue    Retry Missing
                    |
                    v
               Adaptive Split
                    |
                    v
                  Retry
```

This prevents one incomplete LLM response from invalidating an entire long-video translation job.

### Stage 5 — Candidate Validation

LLM output is treated as untrusted structured data.

For token selections, the engine verifies:

- `token_key` exists
- `candidate_id` belongs to the requested token occurrence
- Candidate meaning exists in the server-provided candidate set
- Token surface does not conflict with the actual occurrence
- Sentence number matches when supplied

Invalid selections are rejected rather than silently accepted.

### Stage 6 — OOV Learning

Unknown vocabulary can be converted into structured OOV candidates.

OOV data can contain:

- Term
- Part of speech
- Suggested meaning
- Confidence score
- Source sentence/context

Candidates can then be synchronized with the broader Rakushu backend for downstream linguistic knowledge enrichment.

### Stage 7 — Bunsetsu Analysis

The engine groups Japanese tokens into Bunsetsu-oriented learning structures.

A Bunsetsu phrase is represented around:

- Jiritsugo
- Fuzokugo
- Contextual meaning
- Source sentence information

These structures are intended for downstream subtitle and learning interfaces.

---

# 5. API

Rakushu Engine exposes a lightweight FastAPI interface.

## 5.1 Pipeline API

Public pipeline endpoints:

```text
POST /api/v1/pipeline/stream-url
POST /api/v1/pipeline/stream-upload
```

The engine also retains compatibility aliases for older pipeline routes.

### YouTube URL processing

```text
Client
  |
  v
POST /api/v1/pipeline/stream-url
  |
  v
YouTube Download
  |
  v
ASR
  |
  v
NLP + Knowledge
  |
  v
LLM Translation
  |
  v
Bunsetsu
  |
  v
SSE Events
```

### Upload processing

```text
Client
  |
  v
POST /api/v1/pipeline/stream-upload
  |
  v
Temporary Media
  |
  v
ASR
  |
  v
NLP + Knowledge
  |
  v
LLM Translation
  |
  v
Bunsetsu
  |
  v
SSE Events
```

## 5.2 NLP API

```text
GET /api/v1/nlp/videos/{video_id}/tokens
GET /api/v1/nlp/videos/{video_id}/bunsetsu
```

These endpoints expose stored structured NLP output for a processed video.

## 5.3 Dictionary API

```text
GET /api/v1/dictionary/lookup
```

The dictionary endpoint provides lookup access to engine-managed linguistic knowledge.

## 5.4 Internal Synchronization API

```text
POST /api/v1/internal/dictionary/sync
POST /api/v1/internal/oov/sync-status
```

These endpoints support synchronization between the processing engine and the broader Rakushu platform.

## 5.5 Health API

```text
GET /api/v1/pipeline/health
```

Provides a lightweight service health check.

---

# 6. Progressive SSE Processing

The pipeline is designed to expose intermediate results instead of waiting for the entire video to finish.

Conceptually:

```text
Client
  |
  v
FastAPI StreamingResponse
  |
  v
Pipeline Streamer
  |
  +--> upload_ready / youtube_ready
  |
  +--> progress
  |
  +--> asr_completed
  |
  +--> sentences_identified
  |
  +--> sentence_processed
  |
  +--> chunk_completed
  |
  +--> pipeline_completed
```

### Chunk-level progress

Each completed translation chunk can report:

- Chunk number
- Start/end sentence
- Batch expected count
- Batch returned count
- Batch processing time
- Fallback failures
- Sentence translations

This allows downstream clients to display completed learning content progressively.

---

# 7. Domain & Data Models

Core language and pipeline concepts include:

- `SubtitleSegment`
- `SentenceSubtitle`
- `TokenModel`
- `DictionaryEntry`
- `OovCandidate`
- `BunsetsuPhrase`
- `PipelineResult`
- `GrammarEntry`
- `PhraseEntry`
- `CompoundWordEntry`
- `MatchedKnowledgeUnit`

These models provide explicit contracts between processing stages rather than passing raw infrastructure objects between services.

### Linguistic knowledge model

```text
Linguistic Knowledge
       |
       +--> Grammar
       +--> Phrase
       +--> Compound Word
       +--> Dictionary Entry
       |
       +--> OOV Candidate
                |
                v
          Backend / Curator
```

The engine therefore acts as the computational producer of structured linguistic data, while the broader Rakushu platform can own persistence, curation, learner state, and business workflows.

---

# 8. Technology Stack

| Technology | Role |
| --- | --- |
| **Python** | Core implementation language |
| **FastAPI** | HTTP API and streaming transport |
| **Whisper / faster-whisper** | Japanese speech recognition |
| **GiNZA / spaCy** | Japanese NLP and linguistic analysis |
| **Janome** | Fallback Japanese tokenizer |
| **Ollama-compatible local LLM** | Contextual translation and structured language enrichment |
| **Qwen** | Default local LLM model configuration |
| **FFmpeg** | Media/audio processing |
| **SQLite / local knowledge sources** | Offline/local linguistic knowledge |
| **SSE** | Progressive pipeline updates |

---

# 9. Deployment Role

Rakushu Engine is designed to run as an independently deployable processing component alongside the main Rakushu API.

```text
                Rakushu Platform
                       |
              +--------+--------+
              |                 |
              v                 v
        Rakushu API       Rakushu Engine
        Business Layer    Processing Layer
              |                 |
              |          +------+------+
              |          |             |
              |          v             v
              |       Local LLM   Knowledge Store
              |          |
              +----------+
                    |
                    v
             Learning Data
```

### Responsibility boundary

**Rakushu API owns:**

- Platform business workflows
- Learner/application state
- Subscription and entitlement workflows
- Curator workflows
- Platform-level persistence and integration

**Rakushu Engine owns:**

- Media processing
- ASR
- Japanese NLP
- Linguistic knowledge matching
- LLM translation and candidate selection
- OOV discovery
- Bunsetsu construction
- Pipeline execution and recovery
- Processing output synchronization

This separation keeps computational language processing independent from the main application domain.

---

# 10. Engineering Principles

- **YAGNI** — implement only capabilities required by the processing engine.
- **KISS** — keep pipeline stages explicit and understandable.
- **DRY** — reuse stable processing and validation logic.
- **Validation first** — never trust structured LLM output without server-side validation.
- **Knowledge ownership** — dictionary meanings originate from engine-owned knowledge sources.
- **Occurrence-level identity** — repeated tokens are distinguished by their actual occurrence.
- **Partial recovery** — recover successful work instead of restarting an entire translation job.
- **Bounded LLM calls** — constrain long-content requests with chunk size and sentence-count limits.
- **Progressive processing** — expose useful intermediate results through SSE.
- **Clear responsibility** — API, pipeline, services, and models have distinct roles.
- **Explicit contracts** — processing stages communicate through structured models.

---

# 11. Repository Structure

```text
rakushu-engine/
├── src/
│   ├── api/
│   ├── models/
│   ├── pipeline/
│   ├── services/
│   └── utils/
├── tests/
├── docs/
├── plans/
├── scripts/
├── samples/
├── data/
└── README.md
```

The repository also contains development tooling and workspace-specific engineering resources that are intentionally separated from the runtime processing code.

---

# 12. Documentation

Architecture and engineering documentation is maintained separately from the README.

Key documents include:

- `docs/system-architecture.md`
- `docs/code-standards.md`
- `docs/development-roadmap.md`
- `docs/project-changelog.md`

Implementation plans are maintained under `plans/` following the project's development workflow.

---

# 13. Contributors

| Contributor | Responsibility |
| --- | --- |
| **Trần Hoàng Hòa** | Backend / System Development |
| **Trần Hoàng Huy** | Engine / NLP / LLM / System Development |

---

## Architecture at a Glance

```text
                         RAKUSHU
                            |
              +-------------+-------------+
              |                           |
              v                           v
        Rakushu API                Rakushu Engine
        Business Layer             Processing Layer
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
                   ASR              NLP/Knowledge      LLM
                    |                 |                 |
                    +-----------------+-----------------+
                                      |
                                      v
                               OOV / Bunsetsu
                                      |
                                      v
                            Structured Learning Data
                                      |
                                      v
                             Rakushu Learning Platform
```

<p align="center">
  <b>Rakushu Engine</b><br>
  Japanese Language Processing Engine
</p>
