# Rakushu AI Service Architecture

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
[Knowledge Service]     ──► MATCHED_KNOWLEDGE & OOV_CANDIDATE
       │
       ▼
[LLM Service]           ──► Contextual Translation & Structured OOV Learning
       │
       ▼
[Bunsetsu Service]      ──► PHRASE (1 Jiritsugo + N Fuzokugo + Meaning)
```

## Entity Relationship Mapping (ERD)
- `SUBTITLE_SEGMENT`: 1:M with `TOKEN` and `PHRASE`.
- `TOKEN`: references `DICTIONARY_ENTRY` or triggers `OOV_CANDIDATE`.
- `OOV_CANDIDATE`: routed to `CURATOR_REVIEW` for human-in-the-loop vocabulary enrichment.
- `PHRASE`: interactive Bunsetsu units rendered directly on the Learner video subtitle canvas.
