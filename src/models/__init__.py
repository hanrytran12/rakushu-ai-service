"""Domain data contracts for Rakushu AI Service."""
import importlib

_schema = importlib.import_module(".schema_models", package=__name__)
SubtitleSegment = _schema.SubtitleSegment
TokenModel = _schema.TokenModel
DictionaryEntry = _schema.DictionaryEntry
DefinitionTag = _schema.DefinitionTag
InflectionRule = _schema.InflectionRule
OovCandidate = _schema.OovCandidate
BunsetsuPhrase = _schema.BunsetsuPhrase
SentenceSubtitle = _schema.SentenceSubtitle
PipelineResult = _schema.PipelineResult
MediaInspectionResult = _schema.MediaInspectionResult
VideoType = _schema.VideoType
MediaSourceType = _schema.MediaSourceType

_ext = importlib.import_module(".extended_models", package=__name__)
GrammarEntry = _ext.GrammarEntry
PhraseEntry = _ext.PhraseEntry
CompoundWordEntry = _ext.CompoundWordEntry
MatchedKnowledgeUnit = _ext.MatchedKnowledgeUnit

_curator = importlib.import_module(".curator_models", package=__name__)
CuratorReview = _curator.CuratorReview
OovStatus = _curator.OovStatus
CuratorDecision = _curator.CuratorDecision

__all__ = [
    "SubtitleSegment", "TokenModel", "DictionaryEntry", "DefinitionTag",
    "InflectionRule", "OovCandidate", "BunsetsuPhrase", "SentenceSubtitle",
    "PipelineResult", "MediaInspectionResult", "VideoType", "MediaSourceType",
    "GrammarEntry", "PhraseEntry", "CompoundWordEntry", "MatchedKnowledgeUnit",
    "CuratorReview", "OovStatus", "CuratorDecision",
]