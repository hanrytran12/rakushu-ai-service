"""Formatter transforming PipelineResult domain models into API Response schemas."""
import importlib
from typing import Dict, Any

_api_models = importlib.import_module(".api-models", package="src")
PipelineApiResponse = _api_models.PipelineApiResponse
PipelineData = _api_models.PipelineData
PipelineSummary = _api_models.PipelineSummary
SentenceItem = _api_models.SentenceItem
TokenItem = _api_models.TokenItem
BunsetsuItem = _api_models.BunsetsuItem
KnowledgeUnitItem = _api_models.KnowledgeUnitItem
OovItem = _api_models.OovItem


def format_pipeline_result(result) -> PipelineApiResponse:
    """Formats PipelineResult into clean structured PipelineApiResponse."""
    seg = result.segment
    duration = round(max(seg.end_time - seg.start_time, 0.0), 2)
    
    # Calculate statistics
    ku_list = result.knowledge_units or []
    summary = PipelineSummary(
        total_sentences=len(result.sentences),
        total_tokens=len(result.tokens),
        total_grammar_matched=sum(1 for u in ku_list if getattr(u, "unit_type", "") == "GRAMMAR"),
        total_phrases_matched=sum(1 for u in ku_list if getattr(u, "unit_type", "") == "PHRASE"),
        total_compound_words_matched=sum(1 for u in ku_list if getattr(u, "unit_type", "") == "COMPOUND_WORD"),
        total_oov_discovered=len(result.oov_candidates)
    )

    sentences = []
    for s in result.sentences:
        tokens = [
            TokenItem(
                surface=t.surface, reading=t.reading, lemma=t.lemma,
                pos=t.pos, xpos=getattr(t, "xpos", ""), deprel=getattr(t, "deprel", ""),
                context_meaning=t.context_meaning
            ) for t in s.tokens
        ]
        bunsetsu = [
            BunsetsuItem(
                phrase_id=getattr(b, "phrase_id", ""),
                text=b.text,
                translation=b.translation,
                start_time=b.start_time,
                end_time=b.end_time,
                token_surfaces=getattr(b, "token_surfaces", [])
            ) for b in s.bunsetsu_phrases
        ]
        kus = []
        for u in (s.knowledge_units or []):
            d = u.to_dict() if hasattr(u, "to_dict") else u
            kus.append(KnowledgeUnitItem(**d))
        oovs = [
            OovItem(
                term=o.term, suggested_pos=o.suggested_pos,
                suggested_meaning=o.suggested_meaning,
                confidence_score=o.confidence_score
            ) for o in s.oov_candidates
        ]
        sentences.append(SentenceItem(
            sentence_id=s.sentence_id,
            sequence_number=s.segment.sequence_number,
            start_time=s.segment.start_time,
            end_time=s.segment.end_time,
            text=s.segment.text,
            translation=s.translation,
            tokens=tokens,
            bunsetsu_phrases=bunsetsu,
            knowledge_units=kus,
            oov_candidates=oovs
        ))

    data = PipelineData(
        video_id=seg.video_id or "local_video",
        video_title=seg.video_title or "Uploaded Media",
        source_url=seg.source_url or "",
        duration=duration,
        global_transcript=seg.text,
        global_translation=result.full_translation,
        sentences=sentences,
        summary=summary
    )
    return PipelineApiResponse(success=True, data=data)
