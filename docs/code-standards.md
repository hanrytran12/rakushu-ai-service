# Rakushu Engine Code Standards

1. **File Naming**: Kebab-case naming (e.g. `asr-service.py`, `nlp-service.py`, `bunsetsu-service.py`).
2. **File Length**: Maximum 200 lines per file for optimal context and maintainability.
3. **Core Principles**: YAGNI, KISS, DRY.
4. **Data Contract**: Follow Logical ERD schemas (`SUBTITLE_SEGMENT`, `TOKEN`, `DICTIONARY_ENTRY`, `OOV_CANDIDATE`, `PHRASE`).
5. **Quality Gate**: Comprehensive unit tests covering NLP tokenization, dictionary sense selection, OOV detection, and Bunsetsu grouping logic.
6. **Meaning Source of Truth**: For dictionary-known tokens, LLM output may only select a dictionary candidate; Rakushu Engine resolves the final meaning. LLM-generated meanings remain limited to OOV learning.

## Source Structure

Use responsibility-driven packages under `src/`:
- `api/`: FastAPI transport, request/response schemas, routers.
- `models/`: domain and persistence data contracts.
- `services/`: ASR, NLP, knowledge, LLM, OOV, media, and external integrations.
- `pipeline/`: orchestration, streaming, and pipeline result formatting.
- `utils/`: reusable infrastructure helpers such as SSE and logging.

New code should import from the owning package rather than the root `src` compatibility exports.
