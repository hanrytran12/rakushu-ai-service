"""Streaming Pipeline Runner: Yields real-time SSE events as sentences are processed."""
import os
import sys
import time
import logging
import importlib
from typing import Generator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SubtitleSegment, DictionaryEntry, AsrService,
    NlpService, KnowledgeService, LlmEnrichmentService, BunsetsuService, YouTubeService,
    InvalidMediaError, InvalidLanguageError, ProhibitedContentError, MediaSourceType
)

_stream_utils = importlib.import_module(".stream-utils", package="src")
format_sse = _stream_utils.format_sse
execute_with_heartbeat = _stream_utils.execute_with_heartbeat
build_sentence_payload = _stream_utils.build_sentence_payload
log_sentence_breakdown = _stream_utils.log_sentence_breakdown

logger = logging.getLogger("PipelineStreamer")


def stream_pipeline(media_path: str) -> Generator[str, None, None]:
    """Processes media and yields SSE events progressively at each stage and sentence."""
    t_start = time.time()
    t_yt, t_asr, t_trans, t_seg, t_sentences = 0.0, 0.0, 0.0, 0.0, 0.0
    logger.info(f"\n{'='*75}\n[STREAM PIPELINE] Starting for media: '{media_path}'\n{'='*75}")
    yield format_sse("pipeline_started", {"media_path": media_path, "timestamp": time.time()})

    try:
        # Step 0: Source Detection & Download
        youtube_svc = YouTubeService()
        video_meta, media_file_path = {}, media_path
        is_yt = youtube_svc.is_youtube_url(media_path)
        source_type = MediaSourceType.YOUTUBE_URL if is_yt else MediaSourceType.LOCAL_UPLOAD

        if is_yt:
            t0_yt = time.time()
            logger.info(f">>> [Step 0: YouTube] Detected URL: '{media_path}'. Downloading...")
            yield format_sse("progress", {"step": "youtube_download", "message": f"Downloading YouTube: {media_path}"})
            try:
                media_file_path, video_meta = youtube_svc.download_video(media_path)
                t_yt = round(time.time() - t0_yt, 2)
                logger.info(f"    [Step 0 Done] Title: \"{video_meta.get('title', '')}\" | Saved: {media_file_path} | Time: {t_yt}s")
                yield format_sse("youtube_ready", {"title": video_meta.get("title", ""), "video_id": video_meta.get("video_id", "")})
            except Exception as e:
                logger.error(f"    [Step 0 Failed] YouTube download error: {e}")
                yield format_sse("error", {"error_type": "DownloadError", "message": str(e)})
                return

        # Step 1: ASR Transcription with Heartbeat Keep-Alive
        t0_asr = time.time()
        logger.info(">>> [Step 1: ASR] Transcribing speech via Whisper ASR...")
        yield format_sse("progress", {"step": "asr", "message": "Transcribing speech via Whisper ASR..."})
        asr_svc = AsrService()
        title = video_meta.get("title", "")
        try:
            full_segment = yield from execute_with_heartbeat(
                asr_svc.transcribe, media_file_path, title=title, source_type=source_type
            )
        except (InvalidLanguageError, ProhibitedContentError, InvalidMediaError) as err:
            logger.error(f"    [Step 1 Validation Error]: {err}")
            yield format_sse("error", {"error_type": "ValidationError", "message": str(err)})
            return
        except Exception as exc:
            logger.error(f"    [Step 1 ASR Error]: {exc}")
            yield format_sse("error", {"error_type": "AsrError", "message": str(exc)})
            return

        if video_meta:
            full_segment.video_id = video_meta.get("video_id", full_segment.video_id)
            full_segment.video_title = video_meta.get("title", "")
            full_segment.source_url = media_path

        duration = round(max(full_segment.end_time - full_segment.start_time, 0.0), 2)
        t_asr = round(time.time() - t0_asr, 2)
        logger.info(f"    [Step 1 Done] Transcribed {duration}s in {t_asr}s | Text: \"{full_segment.text}\"")
        yield format_sse("asr_completed", {"text": full_segment.text, "duration": duration, "video_title": full_segment.video_title})

        if not full_segment.text.strip():
            logger.info("    [Step 1] No speech detected in audio.")
            yield format_sse("pipeline_completed", {"message": "No speech detected", "total_sentences": 0})
            return

        # Step 2: Global Translation
        t0_trans = time.time()
        logger.info(">>> [Step 2: Global Translation] Translating full transcription via LLM...")
        yield format_sse("progress", {"step": "global_translation", "message": "Generating global translation..."})
        llm_svc = LlmEnrichmentService()
        try:
            full_translation = yield from execute_with_heartbeat(
                llm_svc.translate_full_transcription, full_segment.text
            )
        except Exception as exc:
            logger.warning(f"    [Step 2 Warning] Global translation error: {exc}")
            full_translation = ""
        full_segment.translation = full_translation
        t_trans = round(time.time() - t0_trans, 2)
        logger.info(f"    [Step 2 Done] Global VI completed in {t_trans}s | VI: \"{full_translation}\"")
        yield format_sse("global_translation_completed", {"translation": full_translation})

        # Step 3: Sentence Segmentation
        t0_seg = time.time()
        nlp_svc, knowledge_svc = NlpService(), KnowledgeService()
        bunsetsu_svc = BunsetsuService()
        try:
            sentence_texts = nlp_svc.split_sentences(full_segment.text)
        except Exception as exc:
            logger.error(f"Sentence splitting error: {exc}")
            sentence_texts = [full_segment.text] if full_segment.text else []
        total_sentences = len(sentence_texts)
        t_seg = round(time.time() - t0_seg, 2)
        logger.info(f">>> [Step 3: Segmentation] Identified {total_sentences} sentences in {t_seg}s.")
        yield format_sse("sentences_identified", {"total_sentences": total_sentences})

        total_chars = max(len(full_segment.text), 1)
        total_dur = full_segment.end_time - full_segment.start_time
        curr_char = 0
        total_grammar, total_phrases, total_compounds = 0, 0, 0
        all_oovs = []
        t0_sentences = time.time()

        # Step 4: Stream Sentences Progressively
        for idx, sent_text in enumerate(sentence_texts, start=1):
            t0_sent = time.time()
            sent_len = len(sent_text)
            s_start = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)
            curr_char += sent_len
            s_end = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)

            sent_seg = SubtitleSegment(
                segment_id=f"{full_segment.segment_id}-s{idx}", video_id=full_segment.video_id,
                start_time=s_start, end_time=s_end, text=sent_text, sequence_number=idx,
            )

            try:
                tokens = nlp_svc.tokenize(sent_seg)
                hier_units, oov_toks = knowledge_svc.match_hierarchical(tokens, sent_text)
                matched_k = [DictionaryEntry(term=u.surface, reading=u.reading, pos=u.unit_type, meaning=u.meaning) for u in hier_units]

                s_trans_vi, token_meanings, enriched_oovs = yield from execute_with_heartbeat(
                    llm_svc.translate_sentence_with_context,
                    sentence_text=sent_text, full_text=full_segment.text,
                    full_translation=full_translation, tokens=tokens,
                    matched_knowledge=matched_k, oov_tokens=oov_toks
                )
                for tok in tokens:
                    if tok.surface in token_meanings:
                        tok.context_meaning = token_meanings[tok.surface]

                phrases = bunsetsu_svc.group_bunsetsu(sent_seg, tokens, token_meanings)
                for u in hier_units:
                    if u.unit_type == "GRAMMAR": total_grammar += 1
                    elif u.unit_type == "PHRASE": total_phrases += 1
                    elif u.unit_type == "COMPOUND_WORD": total_compounds += 1
                all_oovs.extend(enriched_oovs)

                t_sent = round(time.time() - t0_sent, 2)
                log_sentence_breakdown(logger, idx, total_sentences, s_start, s_end, t_sent, sent_text, s_trans_vi, tokens, phrases, enriched_oovs)
                payload = build_sentence_payload(sent_seg, idx, total_sentences, s_start, s_end, sent_text, s_trans_vi, phrases, tokens, hier_units, enriched_oovs)
            except Exception as sent_err:
                logger.error(f"Error enriching sentence {idx}: {sent_err}", exc_info=True)
                payload = build_sentence_payload(sent_seg, idx, total_sentences, s_start, s_end, sent_text, "", [], [], [], [])

            yield format_sse("sentence_processed", payload)

        # Step 5: Final Summary
        t_sentences = round(time.time() - t0_sentences, 2)
        total_time = round(time.time() - t_start, 2)
        avg_sent = round(t_sentences / max(total_sentences, 1), 2)
        logger.info(
            f"\n{'='*75}\n[PIPELINE COMPLETED] Summary Timing Breakdown:\n"
            f" - Step 0 (YouTube Download):  {t_yt}s\n"
            f" - Step 1 (Whisper ASR):       {t_asr}s (Audio Duration: {duration}s)\n"
            f" - Step 2 (Global Translation): {t_trans}s\n"
            f" - Step 3 (Segmentation):      {t_seg}s ({total_sentences} sentences)\n"
            f" - Step 4 (Sentence Enrich):   {t_sentences}s (avg: {avg_sent}s/câu)\n"
            f" >>> TOTAL TIME ELAPSED:       {total_time}s\n{'='*75}\n"
        )
        summary_payload = {
            "video_id": full_segment.video_id, "video_title": full_segment.video_title,
            "duration": duration, "total_processing_time": total_time,
            "summary": {
                "total_sentences": total_sentences, "total_grammar_matched": total_grammar,
                "total_phrases_matched": total_phrases, "total_compound_words_matched": total_compounds,
                "total_oov_discovered": len(all_oovs)
            }
        }
        yield format_sse("pipeline_completed", summary_payload)
    except Exception as fatal_exc:
        logger.error(f"Fatal streaming pipeline exception: {fatal_exc}", exc_info=True)
        yield format_sse("error", {"error_type": "PipelineFatalError", "message": str(fatal_exc)})
