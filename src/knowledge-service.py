"""Knowledge Service: Offline SQLite dictionary database (JMDict + JLPT) and OOV detector."""
import os
import sqlite3
from typing import List, Tuple, Dict, Optional, Set
from src import TokenModel, DictionaryEntry, OovCandidate


class KnowledgeService:
    """Manages dictionary lookup against local SQLite DB and identifies OOV words."""

    DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dictionary.db")

    CORE_FALLBACK: Dict[str, DictionaryEntry] = {
        "皆さん": DictionaryEntry(term="皆さん", reading="みなさん", pos="NOUN", meaning="mọi người, tất cả các bạn (everyone)", jlpt_level="N5", commonality=1),
        "こんにちは": DictionaryEntry(term="こんにちは", reading="こんにちは", pos="INTERJECTION", meaning="xin chào (hello/good afternoon)", jlpt_level="N5", commonality=1),
        "今日": DictionaryEntry(term="今日", reading="きょう", pos="NOUN", meaning="hôm nay (today)", jlpt_level="N5", commonality=1),
        "日本": DictionaryEntry(term="日本", reading="にほん", pos="NOUN", meaning="Nhật Bản (Japan)", jlpt_level="N5", commonality=1),
        "面白い": DictionaryEntry(term="面白い", reading="おもしろい", pos="ADJECTIVE", meaning="thú vị, hấp dẫn, vui tính (interesting/fun)", jlpt_level="N5", commonality=1),
        "話す": DictionaryEntry(term="話す", reading="はなす", pos="VERB", meaning="nói, trò chuyện (to speak/talk)", jlpt_level="N5", commonality=1),
        "話します": DictionaryEntry(term="話します", reading="はなします", pos="VERB", meaning="nói, trò chuyện (thể lịch sự ます)", jlpt_level="N5", commonality=1),
        "について": DictionaryEntry(term="について", reading="について", pos="PARTICLE", meaning="về vấn đề gì đó (about / regarding)", jlpt_level="N4", commonality=1),
    }

    def __init__(self, db_path: Optional[str] = None, exclude_terms: Optional[Set[str]] = None):
        self.db_path = db_path or os.path.abspath(self.DEFAULT_DB_PATH)
        if exclude_terms is not None:
            self.exclude_terms = set(exclude_terms)
        else:
            env_terms = os.environ.get("RAKUSHU_EXCLUDE_TERMS", "ポッドキャスト,若者言葉")
            self.exclude_terms = {t.strip() for t in env_terms.split(",") if t.strip()}
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
                cur.execute(
                    "SELECT term, reading, pos, meaning, jlpt_level, commonality FROM dictionary WHERE term = ? OR reading = ? LIMIT 1",
                    (word, word),
                )
                row = cur.fetchone()
                if row:
                    entry = DictionaryEntry(
                        term=row[0], reading=row[1], pos=row[2], meaning=row[3],
                        jlpt_level=row[4] or "", commonality=int(row[5] or 1)
                    )
                    self.in_memory_cache[word] = entry
                    return entry
            finally:
                conn.close()

        return self.CORE_FALLBACK.get(word)

    def match_tokens(self, tokens: List[TokenModel]) -> Tuple[List[DictionaryEntry], List[TokenModel]]:
        """Cross-references tokens against knowledge base; categorizes into known vs OOV."""
        matched: List[DictionaryEntry] = []
        oov_tokens: List[TokenModel] = []
        seen_matched = set()

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

    def register_enriched_oov(self, entry: DictionaryEntry):
        """Adds validated OOV term back into system knowledge base and persists to SQLite."""
        self.in_memory_cache[entry.term] = entry
        if entry.term in self.exclude_terms:
            self.exclude_terms.remove(entry.term)
        conn = self._get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO dictionary (term, reading, pos, meaning, jlpt_level, commonality)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (entry.term, entry.reading, entry.pos, entry.meaning, entry.jlpt_level, entry.commonality))
                conn.commit()
            finally:
                conn.close()
