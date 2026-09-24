"""Knowledge Service: Offline SQLite dictionary database (JMDict + JLPT) and OOV detector."""
import os
import sqlite3
import json
from typing import List, Tuple, Dict, Optional, Set
from src import (
    TokenModel, DictionaryEntry, OovCandidate, DefinitionTag, InflectionRule,
    GrammarEntry, PhraseEntry, CompoundWordEntry, MatchedKnowledgeUnit
)


class KnowledgeService:
    """Manages dictionary lookup against local SQLite DB and identifies OOV words."""

    STAGING_DB = os.path.join(os.path.dirname(__file__), "..", "data", "jmdict_1000_staging.db")
    DICT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "dictionary.db")
    DEFAULT_DB_PATH = STAGING_DB if os.path.exists(STAGING_DB) else DICT_DB

    CORE_FALLBACK: Dict[str, DictionaryEntry] = {
        "皆さん": DictionaryEntry(term="皆さん", reading="みなさん", pos="NOUN", meaning="mọi người"),
        "こんにちは": DictionaryEntry(term="こんにちは", reading="こんにちは", pos="INTERJECTION", meaning="xin chào"),
        "今日": DictionaryEntry(term="今日", reading="きょう", pos="NOUN", meaning="hôm nay"),
        "日本": DictionaryEntry(term="日本", reading="にほん", pos="NOUN", meaning="Nhật Bản"),
        "面白い": DictionaryEntry(term="面白い", reading="おもしろい", pos="ADJECTIVE", meaning="thú vị"),
        "話す": DictionaryEntry(term="話す", reading="はなす", pos="VERB", meaning="nói, trò chuyện"),
        "話します": DictionaryEntry(term="話します", reading="はなします", pos="VERB", meaning="nói chuyện"),
        "について": DictionaryEntry(term="について", reading="について", pos="PARTICLE", meaning="về"),
    }

    def __init__(self, db_path: Optional[str] = None, exclude_terms: Optional[Set[str]] = None):
        self.db_path = db_path or os.path.abspath(self.DEFAULT_DB_PATH)
        env_terms = os.environ.get("RAKUSHU_EXCLUDE_TERMS", "ポッドキャスト,若者言葉")
        self.exclude_terms = set(exclude_terms) if exclude_terms is not None else {t.strip() for t in env_terms.split(",") if t.strip()}
        self.in_memory_cache: Dict[str, DictionaryEntry] = {}

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        return sqlite3.connect(self.db_path) if os.path.exists(self.db_path) else None

    def lookup(self, word: str) -> Optional[DictionaryEntry]:
        """Looks up a word in cache, SQLite DB, or fallback."""
        if not word or word in self.exclude_terms:
            return None
        if word in self.in_memory_cache:
            return self.in_memory_cache[word]

        conn = self._get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='staging_entries'")
                if cur.fetchone():
                    cur.execute(
                        "SELECT term, reading, pos, meaning_vi, gloss_en FROM staging_entries "
                        "WHERE term = ? OR reading = ? LIMIT 1",
                        (word, word)
                    )
                    r = cur.fetchone()
                    if r:
                        entry = DictionaryEntry(term=r[0], reading=r[1], pos=r[2], meaning=r[3] or r[4] or "")
                        self.in_memory_cache[word] = entry
                        return entry
                else:
                    cur.execute(
                        "SELECT term, reading, pos, definition_tags, rules, score, meaning, sequence, term_tags "
                        "FROM dictionary WHERE term = ? OR reading = ? LIMIT 1",
                        (word, word),
                    )
                    row = cur.fetchone()
                    if row:
                        entry = DictionaryEntry(
                            term=row[0], reading=row[1], pos=row[2],
                            definition_tags=row[3] or "", rules=row[4] or "",
                            score=int(row[5] or 1), meaning=row[6] or "",
                            sequence=int(row[7] or 0), term_tags=row[8] or ""
                        )
                        self.in_memory_cache[word] = entry
                        return entry
            finally:
                conn.close()

        return self.CORE_FALLBACK.get(word)

    def get_tag_info(self, tag_name: str) -> Optional[DefinitionTag]:
        """Queries linguistic explanation for a definition tag (e.g. 'v1', 'adj-i', 'Buddh')."""
        conn = self._get_connection()
        if not conn or not tag_name: return None
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, category, description_en, description_vi FROM definition_tags WHERE name = ?", (tag_name,))
            r = cur.fetchone()
            return DefinitionTag(name=r[0], category=r[1], description_en=r[2], description_vi=r[3]) if r else None
        finally:
            conn.close()

    def get_rule_info(self, rule_code: str) -> Optional[InflectionRule]:
        """Queries explanation for verb/adjective inflection rules."""
        conn = self._get_connection()
        if not conn or not rule_code: return None
        try:
            cur = conn.cursor()
            cur.execute("SELECT rule_code, name_vi, description_vi, examples FROM inflection_rules WHERE rule_code = ?", (rule_code,))
            r = cur.fetchone()
            return InflectionRule(rule_code=r[0], name_vi=r[1], description_vi=r[2], examples=r[3]) if r else None
        finally:
            conn.close()

    def lookup_grammar(self, pattern: str) -> Optional[GrammarEntry]:
        """Looks up a grammar pattern (e.g. 'わけにはいかない', '～に対して')."""
        if not pattern: return None
        conn = self._get_connection()
        if not conn: return None
        try:
            cur = conn.cursor()
            clean = pattern.lstrip("～~- ")
            cur.execute(
                "SELECT id, pattern, reading, jlpt_level, formation, meaning, nuance, examples, source "
                "FROM grammar WHERE pattern = ? OR pattern = ? OR reading = ? LIMIT 1",
                (pattern, clean, pattern)
            )
            r = cur.fetchone()
            return GrammarEntry(id=r[0], pattern=r[1], reading=r[2], jlpt_level=r[3],
                                formation=r[4], meaning=r[5], nuance=r[6], examples=r[7], source=r[8]) if r else None
        finally:
            conn.close()

    def lookup_phrase(self, term: str) -> Optional[PhraseEntry]:
        """Looks up an idiomatic phrase, yojijukugo, or expression."""
        if not term: return None
        conn = self._get_connection()
        if not conn: return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, term, reading, phrase_type, meaning, tags, source "
                "FROM phrases WHERE term = ? OR reading = ? LIMIT 1",
                (term, term)
            )
            r = cur.fetchone()
            return PhraseEntry(id=r[0], term=r[1], reading=r[2], phrase_type=r[3],
                               meaning=r[4], tags=r[5], source=r[6]) if r else None
        finally:
            conn.close()

    def lookup_compound_word(self, term: str) -> Optional[CompoundWordEntry]:
        """Looks up a compound word (verb or noun) with decomposed components."""
        if not term: return None
        conn = self._get_connection()
        if not conn: return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, term, reading, compound_type, components, components_reading, transitivity, meaning, structure "
                "FROM compound_words WHERE term = ? OR reading = ? LIMIT 1",
                (term, term)
            )
            r = cur.fetchone()
            if not r: return None
            comps = json.loads(r[4]) if r[4] else []
            comp_reads = json.loads(r[5]) if r[5] else []
            return CompoundWordEntry(id=r[0], term=r[1], reading=r[2], compound_type=r[3],
                                     components=comps, components_reading=comp_reads,
                                     transitivity=r[6], meaning=r[7], structure=r[8])
        finally:
            conn.close()

    def match_tokens(self, tokens: List[TokenModel]) -> Tuple[List[DictionaryEntry], List[TokenModel]]:
        matched, oov_tokens, seen_matched = [], [], set()
        for token in tokens:
            if token.pos in ["PUNCTUATION", "PARTICLE", "AUX_VERB"]: continue
            entry = self.lookup(token.surface) or self.lookup(token.lemma)
            if entry:
                if entry.term not in seen_matched:
                    matched.append(entry); seen_matched.add(entry.term)
            else:
                oov_tokens.append(token)
        return matched, oov_tokens

    def match_hierarchical(self, tokens: List[TokenModel], sentence_text: str = "") -> Tuple[List[MatchedKnowledgeUnit], List[TokenModel]]:
        """Matches tokens through priority hierarchy: Grammar -> Phrase -> CompoundWord -> Word."""
        import importlib
        matcher_cls = importlib.import_module(".hierarchical-matcher", package="src").HierarchicalKnowledgeMatcher
        return matcher_cls(self).match(tokens, sentence_text)

    def register_enriched_oov(self, entry: DictionaryEntry):
        self.in_memory_cache[entry.term] = entry
        if entry.term in self.exclude_terms: self.exclude_terms.remove(entry.term)
        conn = self._get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO dictionary (term, reading, pos, definition_tags, rules, score, meaning, sequence, term_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (entry.term, entry.reading, entry.pos, entry.definition_tags, entry.rules, entry.score, entry.meaning, entry.sequence, entry.term_tags))
                conn.commit()
            finally:
                conn.close()
