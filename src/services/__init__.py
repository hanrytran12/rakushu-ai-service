"""Domain and external-integration services."""
import importlib

_MODULES = {
    "AsrService": (".asr_service", "AsrService"),
    "NlpService": (".nlp_service", "NlpService"),
    "KnowledgeService": (".knowledge_service", "KnowledgeService"),
    "HierarchicalKnowledgeMatcher": (".hierarchical_matcher", "HierarchicalKnowledgeMatcher"),
    "LlmEnrichmentService": (".llm_service", "LlmEnrichmentService"),
    "LlmOovValidator": (".llm_oov_validator", "LlmOovValidator"),
    "BunsetsuService": (".bunsetsu_service", "BunsetsuService"),
    "YouTubeService": (".youtube_service", "YouTubeService"),
    "MediaInspector": (".media_inspector", "MediaInspector"),
    "InvalidMediaError": (".media_inspector", "InvalidMediaError"),
    "InvalidLanguageError": (".media_inspector", "InvalidLanguageError"),
    "ProhibitedContentError": (".media_inspector", "ProhibitedContentError"),
    "GinzaReferenceService": (".ginza_reference_service", "GinzaReferenceService"),
    "OovService": (".oov_service", "OovService"),
}

def __getattr__(name):
    target = _MODULES.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value

__all__ = list(_MODULES)
