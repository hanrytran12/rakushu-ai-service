"""Knowledge Service: Offline SQLite dictionary database (JMDict + JLPT) and OOV detector."""
import os
import sqlite3
import json
import re
from typing import List, Tuple, Dict, Optional, Set
from src.models import (
    TokenModel, DictionaryEntry, DefinitionTag, InflectionRule,
    GrammarEntry, PhraseEntry, CompoundWordEntry, MatchedKnowledgeUnit
)


class KnowledgeService:
    """Manages dictionary lookup against local SQLite DB and identifies OOV words."""

    STAGING_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jmdict_1000_staging.db")
    DICT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dictionary.db")
    DEFAULT_DB_PATH = STAGING_DB if os.path.exists(STAGING_DB) else DICT_DB

    _FB = [
        ("皆さん", "みなさん", "NOUN", "mọi người"), ("こんにちは", "こんにちは", "INTERJECTION", "xin chào"),
        ("今日", "きょう", "NOUN", "hôm nay"), ("日本", "にほん", "NOUN", "Nhật Bản"),
        ("面白い", "おもしろい", "ADJECTIVE", "thú vị"), ("話す", "はなす", "VERB", "nói, trò chuyện"),
        ("話します", "はなします", "VERB", "nói chuyện"), ("について", "について", "PARTICLE", "về")
    ]
    CORE_FALLBACK: Dict[str, DictionaryEntry] = {t[0]: DictionaryEntry(term=t[0], reading=t[1], pos=t[2], meaning=t[3]) for t in _FB}

    def __init__(self, db_path: Optional[str] = None, exclude_terms: Optional[Set[str]] = None):
        self.db_path = db_path or os.path.abspath(self.DEFAULT_DB_PATH)
        env_terms = os.environ.get("RAKUSHU_EXCLUDE_TERMS", "ポッドキャスト,若者言葉")
        self.exclude_terms = set(exclude_terms) if exclude_terms is not None else {t.strip() for t in env_terms.split(",") if t.strip()}
        self.in_memory_cache: Dict[str, DictionaryEntry] = {}

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        return sqlite3.connect(self.db_path) if os.path.exists(self.db_path) else None

    def _query_one(self, sql: str, params: Tuple = ()) -> Optional[Tuple]:
        conn = self._get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()

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

    @staticmethod
    def _expand_meanings(meaning: str) -> List[str]:
        """Splits dictionary multi-sense text into one selectable meaning per candidate."""
        if not meaning:
            return []

        # Dictionary imports use several conventions: numbered senses, newlines,
        # and semicolon-separated glosses. Normalize all of them to one sense/row.
        normalized = re.sub(r"\r\n?", "\n", meaning).strip()
        parts = re.split(
            r"\n+|;\s*|(?:(?<=\s)|^)(?=\d+[.)]\s*)",
            normalized,
        )

        senses: List[str] = []
        seen = set()
        for part in parts:
            sense = re.sub(r"^\s*\d+[.)]\s*", "", part).strip()
            if not sense:
                continue
            key = re.sub(r"\s+", " ", sense).casefold()
            if key in seen:
                continue
            seen.add(key)
            senses.append(sense)
        return senses

    def lookup_candidates(self, word: str, limit: int = 5) -> List[DictionaryEntry]:
        """Returns bounded dictionary senses for contextual meaning selection."""
        if not word or word in self.exclude_terms or limit <= 0:
            return []

        candidates: List[DictionaryEntry] = []
        seen = set()
        conn = self._get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='staging_entries'")
                if cur.fetchone():
                    cur.execute(
                        "SELECT term, reading, pos, meaning_vi, gloss_en "
                        "FROM staging_entries WHERE term = ? OR reading = ? "
                        "ORDER BY rowid LIMIT ?",
                        (word, word, limit * 3),
                    )
                    for term, reading, pos, meaning_vi, gloss_en in cur.fetchall():
                        for sense in self._expand_meanings((meaning_vi or gloss_en or "").strip()):
                            key = (term, reading, pos, sense)
                            if sense and key not in seen:
                                seen.add(key)
                                candidates.append(
                                    DictionaryEntry(
                                        term=term, reading=reading, pos=pos, meaning=sense
                                    )
                                )
                            if len(candidates) >= limit:
                                break
                        if len(candidates) >= limit:
                            break
                else:
                    cur.execute(
                        "SELECT term, reading, pos, definition_tags, rules, score, "
                        "meaning, sequence, term_tags FROM dictionary "
                        "WHERE term = ? OR reading = ? "
                        "ORDER BY score DESC, sequence ASC LIMIT ?",
                        (word, word, limit * 3),
                    )
                    for row in cur.fetchall():
                        for sense in self._expand_meanings((row[6] or "").strip()):
                            key = (row[0], row[1], row[2], sense)
                            if sense and key not in seen:
                                seen.add(key)
                                candidates.append(
                                    DictionaryEntry(
                                        term=row[0], reading=row[1], pos=row[2],
                                        definition_tags=row[3] or "", rules=row[4] or "",
                                        score=int(row[5] or 1), meaning=sense,
                                        sequence=int(row[7] or 0), term_tags=row[8] or "",
                                    )
                                )
                            if len(candidates) >= limit:
                                break
                        if len(candidates) >= limit:
                            break
            finally:
                conn.close()

        if not candidates:
            fallback = self.CORE_FALLBACK.get(word)
            if fallback:
                candidates.append(fallback)
        return candidates

    def build_token_candidates(
        self, tokens: List[TokenModel], limit: int = 5
    ) -> Dict[str, List[Dict[str, str]]]:
        """Builds occurrence-level dictionary candidates keyed by local token key."""
        result: Dict[str, List[Dict[str, str]]] = {}
        for index, token in enumerate(tokens):
            if token.pos in ("PUNCTUATION", "PARTICLE", "AUX_VERB"):
                continue

            token_key = f"t{index + 1}"
            entries = self.lookup_candidates(token.surface, limit=limit)
            if not entries and token.lemma != token.surface:
                entries = self.lookup_candidates(token.lemma, limit=limit)
            if not entries:
                continue

            result[token_key] = [
                {
                    "candidate_id": f"{token_key}-c{candidate_index}",
                    "token_key": token_key,
                    "token_id": token.token_id,
                    "token": token.surface,
                    "lemma": token.lemma,
                    "term": entry.term,
                    "reading": entry.reading,
                    "pos": entry.pos,
                    "meaning": entry.meaning,
                }
                for candidate_index, entry in enumerate(entries, start=1)
            ]
        return result

    def get_tag_info(self, tag_name: str) -> Optional[DefinitionTag]:
        if not tag_name: return None
        r = self._query_one("SELECT name, category, description_en, description_vi FROM definition_tags WHERE name = ?", (tag_name,))
        return DefinitionTag(name=r[0], category=r[1], description_en=r[2], description_vi=r[3]) if r else None

    def get_rule_info(self, rule_code: str) -> Optional[InflectionRule]:
        if not rule_code: return None
        r = self._query_one("SELECT rule_code, name_vi, description_vi, examples FROM inflection_rules WHERE rule_code = ?", (rule_code,))
        return InflectionRule(rule_code=r[0], name_vi=r[1], description_vi=r[2], examples=r[3]) if r else None

    def lookup_grammar(self, pattern: str) -> Optional[GrammarEntry]:
        """Looks up a grammar pattern (e.g. 'わけにはいかない', '～に対して')."""
        if not pattern: return None
        clean = pattern.lstrip("～~- ")
        r = self._query_one("SELECT id, pattern, reading, jlpt_level, formation, meaning, nuance, examples, source FROM grammar WHERE pattern = ? OR pattern = ? OR reading = ? LIMIT 1", (pattern, clean, pattern))
        return GrammarEntry(id=r[0], pattern=r[1], reading=r[2], jlpt_level=r[3], formation=r[4], meaning=r[5], nuance=r[6], examples=r[7], source=r[8]) if r else None

    def lookup_phrase(self, term: str) -> Optional[PhraseEntry]:
        """Looks up an idiomatic phrase, yojijukugo, or expression."""
        if not term: return None
        r = self._query_one("SELECT id, term, reading, phrase_type, meaning, tags, source FROM phrases WHERE term = ? OR reading = ? LIMIT 1", (term, term))
        return PhraseEntry(id=r[0], term=r[1], reading=r[2], phrase_type=r[3], meaning=r[4], tags=r[5], source=r[6]) if r else None

    def lookup_compound_word(self, term: str) -> Optional[CompoundWordEntry]:
        """Looks up a compound word (verb or noun) with decomposed components."""
        if not term: return None
        r = self._query_one("SELECT id, term, reading, compound_type, components, components_reading, transitivity, meaning, structure FROM compound_words WHERE term = ? OR reading = ? LIMIT 1", (term, term))
        if not r: return None
        comps = json.loads(r[4]) if r[4] else []
        comp_reads = json.loads(r[5]) if r[5] else []
        return CompoundWordEntry(id=r[0], term=r[1], reading=r[2], compound_type=r[3], components=comps, components_reading=comp_reads, transitivity=r[6], meaning=r[7], structure=r[8])

    def match_tokens(self, tokens: List[TokenModel]) -> Tuple[List[DictionaryEntry], List[TokenModel]]:
        matched, oov_tokens, seen_matched = [], [], set()
        for token in tokens:
            if token.pos in ["PUNCTUATION", "PARTICLE", "AUX_VERB"]:
                continue
            entry = self.lookup(token.surface) or self.lookup(token.lemma)
            if entry:
                if entry.term not in seen_matched:
                    matched.append(entry)
                    seen_matched.add(entry.term)
            else:
                oov_tokens.append(token)
        return matched, oov_tokens

    def match_hierarchical(self, tokens: List[TokenModel], sentence_text: str = "") -> Tuple[List[MatchedKnowledgeUnit], List[TokenModel]]:
        """Matches tokens through priority hierarchy: Grammar -> Phrase -> CompoundWord -> Word."""
        import importlib
        matcher_cls = importlib.import_module(".hierarchical_matcher", package="src.services").HierarchicalKnowledgeMatcher
        return matcher_cls(self).match(tokens, sentence_text)

    def register_enriched_oov(self, entry: DictionaryEntry, new_status: str = "ADAPTED", original_term: Optional[str] = None):
        self.in_memory_cache[entry.term] = entry
        if entry.term in self.exclude_terms:
            self.exclude_terms.remove(entry.term)
        conn = self._get_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='staging_entries'")
            if cur.fetchone():
                cur.execute("INSERT OR REPLACE INTO staging_entries (term, reading, pos, meaning_vi, gloss_en) VALUES (?, ?, ?, ?, ?)",
                            (entry.term, entry.reading, entry.pos, entry.meaning, entry.meaning))
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dictionary'")
            if cur.fetchone():
                cur.execute("INSERT OR REPLACE INTO dictionary (term, reading, pos, definition_tags, rules, score, meaning, sequence, term_tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (entry.term, entry.reading, entry.pos, entry.definition_tags, entry.rules, entry.score, entry.meaning, entry.sequence, entry.term_tags))
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='oov_candidates'")
            if cur.fetchone():
                target = original_term or entry.term
                cur.execute("UPDATE oov_candidates SET status = ?, updated_at = datetime('now') WHERE (term = ? OR term = ?) AND status = 'PENDING_CURATOR_REVIEW'",
                            (new_status, target, entry.term))
            conn.commit()
        finally:
            conn.close()

    def is_rejected_oov(self, term: str) -> bool:
        if not term: return False
        return self._query_one("SELECT 1 FROM oov_candidates WHERE term = ? AND status = 'REJECTED' LIMIT 1", (term,)) is not None

    def update_oov_status(self, term: str, new_status: str, candidate_id: Optional[str] = None) -> bool:
        conn = self._get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='oov_candidates'")
            if not cur.fetchone():
                return False
            if candidate_id:
                cur.execute("UPDATE oov_candidates SET status = ?, updated_at = datetime('now') WHERE oov_candidate_id = ? OR term = ?", (new_status, candidate_id, term))
            else:
                cur.execute("UPDATE oov_candidates SET status = ?, updated_at = datetime('now') WHERE term = ?", (new_status, term))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
