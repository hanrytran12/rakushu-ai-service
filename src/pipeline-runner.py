"""Rakushu AI Service: End-to-End ASR -> NLP -> Bunsetsu & OOV Pipeline Runner."""
import sys
import os
import time
import logging
from typing import List

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
    SentenceSubtitle,
    PipelineResult,
    AsrService,
    NlpService,
    KnowledgeService,
    LlmEnrichmentService,
    BunsetsuService,
    YouTubeService,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RakushuPipeline")


def run_pipeline(media_path: str = "samples/japanese_podcast_10s.mp4") -> PipelineResult:
    total_start_time = time.time()
    print("\n" + "=" * 80)
    print(" [RAKUSHU AI CORE] ASR -> GINZA NLP -> SENTENCE TRANSLATION PIPELINE")
    print("=" * 80)

    # 0. Detect YouTube URL and Download MP4 directly
    youtube_svc = YouTubeService()
    video_meta = {}
    media_file_path = media_path
    if youtube_svc.is_youtube_url(media_path):
        logger.info(f">>> DETECTED YOUTUBE URL: {media_path}")
        logger.info(">>> Downloading YouTube video (.mp4 format)...")
        yt_t0 = time.time()
        media_file_path, video_meta = youtube_svc.download_video(media_path)
        logger.info(f"   [YouTube Video Download Completed] In: {time.time() - yt_t0:.2f}s")
        logger.info(f"   [Saved MP4 Path]: {media_file_path}")
        logger.info(f"   [Video Title]: \"{video_meta.get('title', '')}\"")

    # 1. ASR Transcription via Whisper
    t0 = time.time()
    logger.info(">>> STEP 1: Processing video via Whisper ASR...")
    asr_svc = AsrService()
    full_segment = asr_svc.transcribe(media_file_path)
    if video_meta:
        full_segment.video_id = video_meta.get("video_id", full_segment.video_id)
        full_segment.video_title = video_meta.get("title", "")
        full_segment.source_url = media_path
        full_segment.video_path = media_file_path
    elif os.path.exists(media_path):
        full_segment.video_path = media_path
    logger.info(f"   [ASR Result] Duration: {full_segment.start_time:.1f}s -> {full_segment.end_time:.1f}s")
    logger.info(f"   [Transcription]: \"{full_segment.text}\"")
    logger.info(f"   [STEP 1 COMPLETED] Time: {time.time() - t0:.2f}s")

    # Initialize Services
    nlp_svc, knowledge_svc = NlpService(), KnowledgeService()
    llm_svc, bunsetsu_svc = LlmEnrichmentService(), BunsetsuService()

    if not full_segment.text.strip():
        logger.warning("   [ASR Notice] No speech detected in media. Returning empty result.")
        return PipelineResult(segment=full_segment, full_translation="", tokens=[], matched_knowledge=[], oov_candidates=[], bunsetsu_phrases=[], sentences=[])

    # 2. Stage 1: Full Transcription Translation for Global Context
    t1 = time.time()
    logger.info("\n>>> STEP 2: Translating full transcription for overarching global context...")
    full_translation = llm_svc.translate_full_transcription(full_segment.text)
    full_segment.translation = full_translation
    logger.info(f"   [Global Translation (VI)]: \"{full_translation}\"")
    logger.info(f"   [STEP 2 COMPLETED] Time: {time.time() - t1:.2f}s")

    # 3. Split Transcription into Sentences via GiNZA
    t2 = time.time()
    logger.info("\n>>> STEP 3: Splitting transcription into sentences via GiNZA...")
    sentence_texts = nlp_svc.split_sentences(full_segment.text)
    logger.info(f"   Identified {len(sentence_texts)} discrete sentences. Time: {time.time() - t2:.2f}s")

    total_chars = max(len(full_segment.text), 1)
    total_dur = full_segment.end_time - full_segment.start_time
    curr_char = 0
    sentence_subtitles, all_tokens, all_matched, all_bunsetsu = [], [], [], []

    # 4. Stage 2: Sentence-by-Sentence Contextual Translation & Token Meaning Extraction
    t3 = time.time()
    for idx, sent_text in enumerate(sentence_texts, start=1):
        s_t0 = time.time()
        sent_len = len(sent_text)
        s_start = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)
        curr_char += sent_len
        s_end = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)

        sent_seg = SubtitleSegment(
            segment_id=f"{full_segment.segment_id}-s{idx}",
            video_id=full_segment.video_id,
            video_title=full_segment.video_title,
            source_url=full_segment.source_url,
            video_path=full_segment.video_path,
            start_time=s_start,
            end_time=s_end, text=sent_text, sequence_number=idx,
        )
        logger.info(f"\n--- Processing Sentence #{idx} [{s_start:.1f}s - {s_end:.1f}s]: \"{sent_text}\" ---")

        # Step A: NLP Morphological Analysis
        tokens = nlp_svc.tokenize(sent_seg)
        # Step B: Knowledge Matching & OOV Detection
        matched_k, oov_toks = knowledge_svc.match_tokens(tokens)
        all_matched.extend(matched_k)

        # Step C: Sentence Translation guided by Global Context & Token Meanings
        s_trans_vi, token_meanings, enriched_oovs = llm_svc.translate_sentence_with_context(
            sentence_text=sent_text, full_text=full_segment.text,
            full_translation=full_translation, tokens=tokens,
            matched_knowledge=matched_k, oov_tokens=oov_toks
        )
        sent_seg.translation = s_trans_vi
        logger.info(f"   [Sentence Translation (VI)]: \"{s_trans_vi}\"")

        # Step D: Assign Contextual Meanings to Tokens
        for tok in tokens:
            if tok.surface in token_meanings:
                tok.context_meaning = token_meanings[tok.surface]
        all_tokens.extend(tokens)

        # Step E: Bunsetsu Grouping with contextual token meanings
        phrases = bunsetsu_svc.group_bunsetsu(sent_seg, tokens, token_meanings)
        all_bunsetsu.extend(phrases)

        sentence_subtitles.append(SentenceSubtitle(
            segment=sent_seg, translation=s_trans_vi, tokens=tokens,
            matched_knowledge=matched_k, oov_candidates=enriched_oovs, bunsetsu_phrases=phrases,
        ))
        logger.info(f"   [Sentence #{idx} Done] Elapsed: {time.time() - s_t0:.2f}s")
    logger.info(f"   [STEP 4 COMPLETED] All sentences translated in: {time.time() - t3:.2f}s")

    # 5. Assemble Consolidated Pipeline Result
    result = PipelineResult(
        segment=full_segment, full_translation=full_translation,
        tokens=all_tokens, matched_knowledge=all_matched,
        oov_candidates=[o for s in sentence_subtitles for o in s.oov_candidates],
        bunsetsu_phrases=all_bunsetsu, sentences=sentence_subtitles,
    )

    # 6. Display Summary Report
    print("\n" + "=" * 80)
    print(" [OUTCOME] TWO-STAGE TRANSLATION & TOKEN MEANING BREAKDOWN:")
    print("=" * 80)
    print(f"Global Transcript: \"{result.segment.text}\"")
    print(f"Global Translation: \"{result.full_translation}\"")
    for idx, s in enumerate(result.sentences, start=1):
        print(f"\n[Sentence #{idx}] ({s.segment.start_time:.1f}s -> {s.segment.end_time:.1f}s)")
        print(f"  JP: {s.segment.text}\n  VI: {s.translation}")
        print("  Token-level contextual meanings:")
        for t in s.tokens:
            if t.context_meaning:
                print(f"    - {t.surface} ({t.pos}): {t.context_meaning}")
        print("  Bunsetsu interactive chunks:")
        for bp in s.bunsetsu_phrases:
            print(f"    * [{bp.text}] ({bp.start_time:.1f}s-{bp.end_time:.1f}s) -> {bp.translation}")

    if result.segment.source_url:
        print(f"\nSource URL: {result.segment.source_url}")
        print(f"Video Title: {result.segment.video_title}")
    if result.segment.video_path:
        print(f"Video Path: {result.segment.video_path}")

    total_elapsed = time.time() - total_start_time
    logger.info(f"\n>>> [PIPELINE COMPLETED] Total processing time for video: {total_elapsed:.2f}s")
    return result


if __name__ == "__main__":
    media_file = sys.argv[1] if len(sys.argv) > 1 else "samples/japanese_podcast_10s.mp4"
    yt_svc = YouTubeService()
    if not yt_svc.is_youtube_url(media_file) and not os.path.exists(media_file):
        import importlib
        create_media = importlib.import_module("samples.create-sample-media")
        create_media.create_sample_media(media_file)

    run_pipeline(media_file)

