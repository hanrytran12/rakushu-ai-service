"""Domain and external-integration services."""
import importlib

_MODULES = {
    "AsrService": (".asr-service", "AsrService"),
    "NlpService": (".nlp-service", "NlpService"),
    "KnowledgeService": (".knowledge-service", "KnowledgeService"),
    "HierarchicalKnowledgeMatcher": (".hierarchical-matcher", "HierarchicalKnowledgeMatcher"),
    "LlmEnrichmentService": (".llm-service", "LlmEnrichmentService"),
    "LlmOovValidator": (".llm-oov-validator", "LlmOovValidator"),
    "BunsetsuService": (".bunsetsu-service", "BunsetsuService"),
    "YouTubeService": (".youtube-service", "YouTubeService"),
    "MediaInspector": (".media-inspector", "MediaInspector"),
    "InvalidMediaError": (".media-inspector", "InvalidMediaError"),
    "InvalidLanguageError": (".media-inspector", "InvalidLanguageError"),
    "ProhibitedContentError": (".media-inspector", "ProhibitedContentError"),
    "GinzaReferenceService": (".ginza-reference-service", "GinzaReferenceService"),
    "OovService": (".oov-service", "OovService"),
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
