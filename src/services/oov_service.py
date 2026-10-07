"""OOV Service: SQLite candidate persistence, Curator Review workflow, and Knowledge Merger."""
import os
import sqlite3
import datetime
import logging
import importlib
from typing import List, Optional, Tuple, Dict, Any
from src.models import (
    OovCandidate, CuratorReview, OovStatus, CuratorDecision,
    DictionaryEntry,
)

_knowledge_mod = importlib.import_module(".knowledge_service", package="src.services")
KnowledgeService = _knowledge_mod.KnowledgeService

_init_oov_tables = importlib.import_module(".oov_schema", package="src.models").init_oov_tables
logger = logging.getLogger("OovService")


class OovService:
    """Manages OOV lifecycle, persistent storage, curator reviews, and dictionary synchronization."""

    def __init__(self, db_path: Optional[str] = None, knowledge_service: Optional[KnowledgeService] = None):
        self.db_path = db_path or os.path.abspath(KnowledgeService.DEFAULT_DB_PATH)
        self.knowledge_service = knowledge_service or KnowledgeService(db_path=self.db_path)
        with self._get_connection() as conn:
            _init_oov_tables(conn)

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def persist_candidate(self, cand: OovCandidate) -> OovCandidate:
        """Saves a validated OOV candidate to database."""
        now = datetime.datetime.utcnow().isoformat()
        if not cand.detected_at:
            cand.detected_at = now
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT oov_candidate_id, status FROM oov_candidates WHERE term = ? LIMIT 1", (cand.term,))
            row = cur.fetchone()
            if row:
                cand.oov_candidate_id = row[0]
                return cand
            cur.execute("""
                INSERT INTO oov_candidates (
                    oov_candidate_id, token_id, term, tentative_reading, tentative_pos,
                    suggested_meaning, context_snippet, confidence_score, status, detected_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cand.oov_candidate_id, cand.token_id, cand.term, cand.tentative_reading,
                cand.suggested_pos, cand.suggested_meaning, cand.context_snippet,
                cand.confidence_score, cand.status, cand.detected_at, now
            ))
            conn.commit()
        return cand

    def get_candidate(self, candidate_id: str) -> Optional[OovCandidate]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT oov_candidate_id, token_id, term, tentative_reading, tentative_pos,
                       suggested_meaning, context_snippet, confidence_score, status, detected_at
                FROM oov_candidates WHERE oov_candidate_id = ?
            """, (candidate_id,))
            r = cur.fetchone()
            if not r:
                return None
            return OovCandidate(
                oov_candidate_id=r[0], token_id=r[1], term=r[2], tentative_reading=r[3] or "",
                suggested_pos=r[4] or "", suggested_meaning=r[5] or "", context_snippet=r[6] or "",
                confidence_score=float(r[7] or 0.0), status=r[8], detected_at=r[9] or ""
            )

    def list_candidates(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> Tuple[int, List[OovCandidate]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            where_sql = "WHERE status = ?" if status else ""
            params = (status,) if status else ()
            cur.execute(f"SELECT COUNT(*) FROM oov_candidates {where_sql}", params)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT oov_candidate_id, token_id, term, tentative_reading, tentative_pos,
                       suggested_meaning, context_snippet, confidence_score, status, detected_at
                FROM oov_candidates {where_sql}
                ORDER BY detected_at DESC LIMIT ? OFFSET ?
            """, params + (limit, offset))
            rows = cur.fetchall()
            items = [
                OovCandidate(
                    oov_candidate_id=r[0], token_id=r[1], term=r[2], tentative_reading=r[3] or "",
                    suggested_pos=r[4] or "", suggested_meaning=r[5] or "", context_snippet=r[6] or "",
                    confidence_score=float(r[7] or 0.0), status=r[8], detected_at=r[9] or ""
                ) for r in rows
            ]
            return total, items

    def process_review(
        self,
        candidate_id: str,
        curator_id: str,
        decision: str,
        edited_term: Optional[str] = None,
        edited_reading: Optional[str] = None,
        edited_pos: Optional[str] = None,
        edited_meaning: Optional[str] = None,
        comment: str = ""
    ) -> Tuple[bool, str, Optional[CuratorReview]]:
        """Processes curator decision: ADAPT -> merges to system knowledge; REJECT -> discards."""
        cand = self.get_candidate(candidate_id)
        if not cand:
            return False, f"Candidate '{candidate_id}' not found", None

        now = datetime.datetime.utcnow().isoformat()
        review = CuratorReview(
            oov_candidate_id=candidate_id, curator_id=curator_id, decision=decision,
            edited_term=edited_term or cand.term,
            edited_reading=edited_reading or cand.tentative_reading or cand.term,
            edited_pos=edited_pos or cand.suggested_pos or "NOUN",
            edited_meaning=edited_meaning or cand.suggested_meaning,
            comment=comment, reviewed_at=now
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            new_status = OovStatus.ADAPTED if decision == CuratorDecision.ADAPT else OovStatus.REJECTED
            cur.execute("UPDATE oov_candidates SET status = ?, updated_at = ? WHERE oov_candidate_id = ?", (new_status, now, candidate_id))
            conn.commit()

        if decision == CuratorDecision.ADAPT:
            self._merge_into_system_knowledge(
                term=review.edited_term, reading=review.edited_reading,
                pos=review.edited_pos, meaning=review.edited_meaning
            )
            return True, f"OOV candidate '{review.edited_term}' adapted and merged into system knowledge.", review

        return True, f"OOV candidate '{cand.term}' rejected by curator.", review

    def process_manual_add(
        self,
        candidate_id: str,
        curator_id: str,
        term: str,
        reading: str,
        pos: str,
        meaning: str,
        comment: str = ""
    ) -> Tuple[bool, str, Optional[CuratorReview]]:
        """Curator rejected AI identification, but manually input vocabulary info to merge."""
        cand = self.get_candidate(candidate_id)
        if not cand:
            return False, f"Candidate '{candidate_id}' not found", None

        now = datetime.datetime.utcnow().isoformat()
        review = CuratorReview(
            oov_candidate_id=candidate_id, curator_id=curator_id, decision=CuratorDecision.MANUAL_ADD,
            edited_term=term, edited_reading=reading, edited_pos=pos,
            edited_meaning=meaning, comment=comment, reviewed_at=now
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE oov_candidates SET status = ?, updated_at = ? WHERE oov_candidate_id = ?", (OovStatus.RESOLVED_MANUAL, now, candidate_id))
            conn.commit()

        self._merge_into_system_knowledge(term=term, reading=reading, pos=pos, meaning=meaning)
        return True, f"Token '{term}' manually registered and merged into system knowledge.", review

    def _merge_into_system_knowledge(self, term: str, reading: str, pos: str, meaning: str):
        """Merges reviewed term directly into the primary dictionary table and synchronizes cache."""
        entry = DictionaryEntry(
            term=term, reading=reading, pos=pos,
            meaning=meaning, definition_tags="curator-verified",
            rules="", score=100, sequence=1, term_tags="OOV_ENRICHED"
        )
        self.knowledge_service.register_enriched_oov(entry)
        logger.info(f"[System Knowledge Merged] Successfully added '{term}' ({reading}) to dictionary.")