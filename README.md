# Rakushu Engine

## Japanese Language Processing Engine

Rakushu Engine is the language-processing engine behind the Rakushu Japanese-learning platform. It transforms Japanese video and audio into structured learning data through ASR, Japanese NLP, knowledge matching, contextual LLM processing, OOV learning, and Bunsetsu analysis.

> **Positioning:** Rakushu Engine is a processing engine, not a standalone AI product. It provides the language-processing capabilities used by the Rakushu platform.

---

## 1. Overview

Rakushu Engine owns the language-processing workflow of Rakushu.

### Core pipeline

```text
Video / Audio
     |
     v
ASR
     |
     v
Japanese NLP
     |
     v
Knowledge Matching
     |
     v
LLM Translation & Candidate Selection
     |
     v
OOV Learning
     |
     v
Bunsetsu Analysis
     |
     v
Structured Learning Output
```

### Core capabilities

- Japanese speech recognition with Whisper
- Japanese sentence segmentation and morphological analysis
- Grammar, phrase, compound-word, and dictionary matching
- Knowledge-grounded LLM translation and candidate selection
- Server-side validation of LLM selections
- OOV discovery and learning
- Bunsetsu analysis
- Chunked translation for long content
- Adaptive retry and partial-result recovery
- Progressive SSE pipeline events
- Dedicated NLP, dictionary, and Bunsetsu APIs

---

## 2. Architecture

Rakushu Engine separates HTTP transport, pipeline orchestration, processing services, and domain models.

```text
src/
├── api/       # FastAPI transport and routers
├── pipeline/  # Pipeline orchestration and streaming
├── services/  # ASR, NLP, LLM, Knowledge, OOV, Bunsetsu, integrations
├── models/    # Shared language and pipeline models
└── utils/     # Shared infrastructure helpers
```

Dependency direction:

```text
API -> Pipeline -> Services -> Models
```

Models do not depend on API or pipeline orchestration.

### Layer responsibilities

**API Layer**

FastAPI routers, request/response contracts, SSE responses, dictionary endpoints, and NLP output endpoints.

**Pipeline Layer**

End-to-end orchestration, stage sequencing, streaming events, chunk processing, and result formatting.

**Service Layer**

ASR, NLP, LLM, Knowledge, OOV, Bunsetsu, media, and external integration logic.

**Model Layer**

Shared domain and data contracts used between processing stages.

**Utility Layer**

Reusable infrastructure helpers such as SSE and logging.

---

## 3. Processing Design

### 3.1 ASR

Whisper converts Japanese speech into text. FFmpeg is used for media/audio processing, while validation and safety checks are applied around transcription.

### 3.2 Japanese NLP

GiNZA provides sentence segmentation, tokenization, POS, lemma, reading, dependency metadata, and Bunsetsu-related information.

Janome is available as a fallback tokenizer.

### 3.3 Knowledge Matching

Known linguistic knowledge is resolved from server-owned sources.

```text
Token
  |
  +--> Grammar
  +--> Phrase
  +--> Compound Word
  +--> Word / Dictionary
          |
          v
     Candidate Set
          |
          v
     LLM Selection
          |
          v
 Server-side Validation
          |
          v
    Validated Meaning
```

The engine remains the source of truth for dictionary meanings. The LLM selects among server-provided candidates rather than creating arbitrary dictionary meanings.

### 3.4 LLM Enrichment

LLM processing is used for:

- Vietnamese translation
- Contextual interpretation
- Dictionary candidate selection
- Structured OOV learning

LLM output is treated as untrusted structured data and validated before it is propagated downstream.

### 3.5 Chunked Translation

Long content is processed using bounded chunks.

```text
Chunk Planner
     |
     v
LLM Batch
     |
     v
Validation
     |
     +---- complete ----> Continue
     |
     +---- missing ------> Retry Missing
                              |
                              v
                         Adaptive Split
                              |
                              v
                            Retry
```

This prevents one incomplete LLM response from invalidating an entire translation job.

### 3.6 OOV Learning

Unknown vocabulary can be converted into structured OOV candidates for downstream review and enrichment.

### 3.7 Bunsetsu

The engine groups Japanese tokens into Bunsetsu-based learning units and exposes the resulting structure to downstream consumers.

---

## 4. API

Rakushu Engine exposes a lightweight FastAPI interface.

### Pipeline

```text
/api/v1/pipeline
    ├── /stream-url
    └── /stream-upload
```

### NLP

```text
/api/v1/nlp
    ├── /videos/{video_id}/tokens
    └── /videos/{video_id}/bunsetsu
```

### Dictionary

```text
/api/v1/dictionary
    └── /lookup
```

### Internal synchronization

```text
/api/v1/internal
    ├── /dictionary/sync
    └── /oov/sync-status
```

### SSE streaming

Pipeline progress can be streamed progressively to clients.

```text
Client
  |
  v
FastAPI
  |
  v
Pipeline Streamer
  |
  +--> sentence_processed
  +--> chunk_completed
  +--> pipeline_completed
```

---

## 5. Technology Stack

| Technology | Role |
|---|---|
| **FastAPI** | HTTP and SSE transport |
| **Whisper / faster-whisper** | Japanese speech recognition |
| **GiNZA / spaCy** | Japanese NLP |
| **Janome** | NLP fallback tokenizer |
| **Ollama / Qwen** | Local LLM processing |
| **SQLite** | Offline linguistic knowledge |
| **FFmpeg** | Media and audio processing |
| **SSE** | Progressive pipeline updates |

---

## 6. Domain Models

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

These models allow processing stages to communicate through explicit contracts instead of raw infrastructure objects.

---

## 7. End-to-End Flow

```text
[YouTube URL / Upload]
          |
          v
      [FastAPI]
          |
          v
 [Pipeline Streamer]
          |
          v
    [Whisper ASR]
          |
          v
    [GiNZA NLP]
          |
          v
 [Knowledge Matching]
          |
          v
   [LLM Enrichment]
          |
          v
 [Validation / Recovery]
          |
          +------> [OOV Candidates]
          |
          v
 [Bunsetsu Analysis]
          |
          v
[Structured Learning Output]
```

---

## 8. Deployment Role

Rakushu Engine is designed to run as an independently deployable processing component alongside the main Rakushu backend.

```text
Client
  |
  v
Rakushu API
  |
  v
Rakushu Engine
  |
  +--> LLM Runtime
  +--> SQLite Knowledge Store
  +--> Pipeline Output
  +--> Backend Synchronization
```

The main Rakushu backend owns application and business workflows. Rakushu Engine owns computational language-processing workloads.

---

## 9. Engineering Principles

- **YAGNI** — implement only what the engine actually needs.
- **KISS** — keep processing flows explicit and understandable.
- **DRY** — share stable processing and validation logic.
- **Validation first** — never trust structured LLM output without validation.
- **Partial recovery** — recover successful work instead of restarting an entire job.
- **Clear ownership** — API, pipeline, services, and models have distinct responsibilities.
- **Domain contracts** — processing stages communicate through explicit data models.

---

## 10. Repository Structure

```text
rakushu-engine/
├── src/
├── tests/
├── docs/
├── plans/
├── scripts/
├── samples/
├── data/
└── README.md
```

---

## 11. Contributors

| Contributor | Responsibility |
|---|---|
| **Hòa** | Backend / System Development |
| **Huy** | Engine / NLP / LLM / System Development |

---

<p align="center">
  <b>Rakushu Engine</b><br>
  Japanese Language Processing Engine
</p>
