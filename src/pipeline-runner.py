"""Rakushu AI Service: End-to-End ASR -> NLP -> Bunsetsu & OOV Pipeline Runner."""
import sys
import os
import json
import logging
from datetime import datetime

# Adjust module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import (
    SubtitleSegment,
    PipelineResult,
    AsrService,
    NlpService,
    KnowledgeService,
    LlmEnrichmentService,
    BunsetsuService,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RakushuPipeline")


def run_pipeline(media_path: str = "samples/japanese_podcast_10s.mp4") -> PipelineResult:
    print("\n" + "=" * 80)
    print(" [RAKUSHU AI CORE] ASR -> NLP -> BUNSETSU & OOV PIPELINE EXECUTION")
    print("=" * 80)

    # 1. ASR Transcription via Whisper
    logger.info(">>> STEP 1: Processing video via Whisper ASR...")
    asr_svc = AsrService()
    segment = asr_svc.transcribe(media_path)
    logger.info(f"   [ASR Result] Duration: {segment.start_time:.1f}s -> {segment.end_time:.1f}s")
    logger.info(f"   [Transcription]: \"{segment.text}\"")

    # 2. NLP Morphological Analysis
    logger.info("\n>>> STEP 2: Feeding transcription to NLP morphological engine...")
    nlp_svc = NlpService()
    tokens = nlp_svc.tokenize(segment)
    logger.info(f"   Extracted {len(tokens)} tokens:")
    print(f"   {'Surface':<14} | {'POS':<14} | {'Lemma':<12} | {'Reading':<14} | {'Romaji'}")
    print("   " + "-" * 70)
    for tok in tokens:
        print(f"   {tok.surface:<14} | {tok.pos:<14} | {tok.lemma:<12} | {tok.reading:<14} | {tok.romaji}")

    # 3. Knowledge Base Matching & OOV Detection
    logger.info("\n>>> STEP 3: Cross-referencing tokens against Knowledge Base (JMDict sidecar)...")
    knowledge_svc = KnowledgeService()
    matched_knowledge, oov_tokens = knowledge_svc.match_tokens(tokens)

    logger.info(f"   [Knowledge Hit] Found {len(matched_knowledge)} known terms in dictionary:")
    for k in matched_knowledge:
        print(f"     [OK] {k.term} ({k.reading}, {k.pos}) [{k.jlpt_level}]: {k.meaning}")

    logger.info(f"   [OOV Flagged] Detected {len(oov_tokens)} Out-Of-Vocabulary candidate tokens:")
    for oov in oov_tokens:
        print(f"     [!] [OOV Candidate] Surface: \"{oov.surface}\" | POS: {oov.pos} | Reading: {oov.reading}")

    # 4. LLM Contextual Translation & OOV Learning
    logger.info("\n>>> STEP 4: Calling LLM for Contextual Translation & OOV Schema Enrichment...")
    llm_svc = LlmEnrichmentService()
    translation_vi, enriched_oovs = llm_svc.enrich_and_learn_oov(segment.text, matched_knowledge, oov_tokens)
    logger.info(f"   [Full Translation (VI)]: \"{translation_vi}\"")
    logger.info(f"   [Learned OOV Structured Output]: {len(enriched_oovs)} candidates formatted to DICTIONARY_ENTRY schema:")
    for cand in enriched_oovs:
        print(f"     [*] Term: {cand.term:<12} | POS: {cand.suggested_pos:<8} | Meaning: {cand.suggested_meaning}")

    # 5. Bunsetsu Grouping (Post-processing)
    logger.info("\n>>> STEP 5: Bunsetsu Grouping Engine (Jiritsugo + Fuzokugo clusters)...")
    bunsetsu_svc = BunsetsuService()
    bunsetsu_phrases = bunsetsu_svc.group_bunsetsu(segment, tokens)
    logger.info(f"   Generated {len(bunsetsu_phrases)} interactive Bunsetsu phrase units for Learner UI:")
    print(f"   {'Order':<5} | {'Timestamp':<14} | {'Bunsetsu Text':<22} | {'Learner Chunk Meaning'}")
    print("   " + "-" * 75)
    for bp in bunsetsu_phrases:
        time_span = f"{bp.start_time:.1f}s - {bp.end_time:.1f}s"
        print(f"   #{bp.phrase_order:<4} | {time_span:<14} | {bp.text:<22} | {bp.translation}")

    # Final Outcome packaging
    result = PipelineResult(
        segment=segment,
        tokens=tokens,
        matched_knowledge=matched_knowledge,
        oov_candidates=enriched_oovs,
        bunsetsu_phrases=bunsetsu_phrases
    )

    print("\n" + "=" * 80)
    print(" [OUTCOME] PIPELINE OUTCOME FOR UI & CURATOR DASHBOARD:")
    print("=" * 80)
    print(f"- Transcription: \"{result.segment.text}\"")
    print(f"- Vietnamese Meaning: \"{translation_vi}\"")
    print(f"- Bunsetsu Overlay Clusters ({len(result.bunsetsu_phrases)} chunks):")
    for bp in result.bunsetsu_phrases:
        print(f"   [{bp.text}] -> {bp.translation}")
    print(f"- System OOV Candidates for Curator Review ({len(result.oov_candidates)} items):")
    for o in result.oov_candidates:
        print(f"   - [Candidate ID: {o.oov_candidate_id[:8]}] Term: {o.term} | POS: {o.suggested_pos} | Meaning: {o.suggested_meaning}")

    return result


if __name__ == "__main__":
    media_file = sys.argv[1] if len(sys.argv) > 1 else "samples/japanese_podcast_10s.mp4"
    if not os.path.exists(media_file):
        import importlib
        create_media = importlib.import_module("samples.create-sample-media")
        create_media.create_sample_media(media_file)

    run_pipeline(media_file)
