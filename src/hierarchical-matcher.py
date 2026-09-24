"""Hierarchical Knowledge Matcher: Grammar -> Phrase -> CompoundWord -> Word."""
from typing import List, Tuple, Set
from src import (
    TokenModel, MatchedKnowledgeUnit, GrammarEntry, PhraseEntry,
    CompoundWordEntry
)


class HierarchicalKnowledgeMatcher:
    """Matches sentence tokens sequentially through Grammar, Phrase, CompoundWord, and Word layers."""

    def __init__(self, knowledge_service):
        self.ks = knowledge_service

    def match(self, tokens: List[TokenModel], sentence_text: str = "") -> Tuple[List[MatchedKnowledgeUnit], List[TokenModel]]:
        if not tokens:
            return [], []

        n = len(tokens)
        consumed: Set[int] = set()
        matched_units: List[MatchedKnowledgeUnit] = []

        # Multi-token matching: Longest match first (length 8 down to 2)
        # Priority order for same length: Grammar -> Phrase -> CompoundWord
        for length in range(min(n, 8), 1, -1):
            for i in range(n - length + 1):
                span_indices = list(range(i, i + length))
                if any(idx in consumed for idx in span_indices):
                    continue
                span_surface = "".join(tokens[k].surface for k in span_indices)
                
                # Check Grammar (Priority 1)
                g_entry = self.ks.lookup_grammar(span_surface)
                if g_entry:
                    matched_units.append(MatchedKnowledgeUnit(
                        unit_type="GRAMMAR",
                        surface=span_surface,
                        reading=g_entry.reading or span_surface,
                        meaning=g_entry.meaning,
                        start_token_idx=i,
                        end_token_idx=i + length - 1,
                        grammar_info=g_entry
                    ))
                    for k in span_indices:
                        consumed.add(k)
                    continue

                # Check Phrase (Priority 2)
                p_entry = self.ks.lookup_phrase(span_surface)
                if p_entry:
                    matched_units.append(MatchedKnowledgeUnit(
                        unit_type="PHRASE",
                        surface=span_surface,
                        reading=p_entry.reading or span_surface,
                        meaning=p_entry.meaning,
                        start_token_idx=i,
                        end_token_idx=i + length - 1,
                        phrase_info=p_entry
                    ))
                    for k in span_indices:
                        consumed.add(k)
                    continue

                # Check Compound Word (Priority 3)
                c_entry = self.ks.lookup_compound_word(span_surface)
                if c_entry:
                    matched_units.append(MatchedKnowledgeUnit(
                        unit_type="COMPOUND_WORD",
                        surface=span_surface,
                        reading=c_entry.reading or span_surface,
                        meaning=c_entry.meaning,
                        start_token_idx=i,
                        end_token_idx=i + length - 1,
                        compound_info=c_entry
                    ))
                    for k in span_indices:
                        consumed.add(k)
                    continue

        # Also check single tokens that might be compound words
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            c_entry = self.ks.lookup_compound_word(tok.surface) or self.ks.lookup_compound_word(tok.lemma)
            if c_entry:
                matched_units.append(MatchedKnowledgeUnit(
                    unit_type="COMPOUND_WORD",
                    surface=tok.surface,
                    reading=c_entry.reading or tok.reading or tok.surface,
                    meaning=c_entry.meaning,
                    start_token_idx=i,
                    end_token_idx=i,
                    compound_info=c_entry
                ))
                consumed.add(i)

        # 4. WORD MATCHING (Priority 4)
        oov_tokens: List[TokenModel] = []
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            if tok.pos in ["PUNCTUATION", "PARTICLE", "AUX_VERB"]:
                continue
            w_entry = self.ks.lookup(tok.surface) or self.ks.lookup(tok.lemma)
            if w_entry:
                matched_units.append(MatchedKnowledgeUnit(
                    unit_type="WORD",
                    surface=tok.surface,
                    reading=w_entry.reading or tok.reading or tok.surface,
                    meaning=w_entry.meaning,
                    start_token_idx=i,
                    end_token_idx=i,
                    word_info=w_entry.to_dict()
                ))
                consumed.add(i)
            else:
                oov_tokens.append(tok)

        matched_units.sort(key=lambda u: u.start_token_idx)
        return matched_units, oov_tokens
