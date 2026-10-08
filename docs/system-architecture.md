# Rakushu Engine Architecture

## Pipeline Data Flow
```
[Video / Audio File]
       │
       ▼
[ASR Service (Whisper)] ──► SUBTITLE_SEGMENT (Text, Duration)
       │
       ▼
[NLP Service]           ──► TOKEN (Surface, Lemma, POS, Reading, Romaji)
       │
       ▼
[Knowledge Service]     ──► MATCHED_KNOWLEDGE & DICTIONARY_SENSE_CANDIDATES & OOV_CANDIDATE
       │
       ▼
[LLM Service]           ──► Contextual Translation & Candidate Selection & Structured OOV Learning
       │
       ▼
[Engine Resolver]   ──► Validated Dictionary Meaning
       │
       ▼
[Bunsetsu Service]      ──► PHRASE (1 Jiritsugo + N Fuzokugo + Meaning)
```

## Entity Relationship Mapping (ERD)
- `SUBTITLE_SEGMENT`: 1:M with `TOKEN` and `PHRASE`.
- `TOKEN`: references `DICTIONARY_ENTRY` or triggers `OOV_CANDIDATE`.
- `OOV_CANDIDATE`: routed to `CURATOR_REVIEW` for human-in-the-loop vocabulary enrichment.
- `PHRASE`: interactive Bunsetsu units rendered directly on the Learner video subtitle canvas.

## Source Code Structure

```
src/
├── api/       # FastAPI transport and routers
├── models/    # Domain/API-independent data contracts
├── services/  # Domain and external-integration services
├── pipeline/  # Pipeline orchestration and streaming
└── utils/     # Shared infrastructure helpers
```

Dependency direction: API -> Pipeline -> Services -> Models. Pipeline/API may consume models directly. Models must not depend on API or service orchestration.
