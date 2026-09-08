# Rakushu AI Service Code Standards

1. **File Naming**: Kebab-case naming (e.g. `asr-service.py`, `nlp-service.py`, `bunsetsu-service.py`).
2. **File Length**: Maximum 200 lines per file for optimal context and maintainability.
3. **Core Principles**: YAGNI, KISS, DRY.
4. **Data Contract**: Follow Logical ERD schemas (`SUBTITLE_SEGMENT`, `TOKEN`, `DICTIONARY_ENTRY`, `OOV_CANDIDATE`, `PHRASE`).
5. **Quality Gate**: Comprehensive unit tests covering all NLP tokenization, OOV detection, and Bunsetsu grouping logic.
