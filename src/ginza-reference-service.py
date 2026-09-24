"""GiNZA Reference Service: Linguistic explanations for UniDic XPOS, Deprel, and Bunsetsu."""
import os
import sqlite3
from typing import Dict, Optional


class GinzaReferenceService:
    """Provides linguistic lookups against local GiNZA reference SQLite database."""

    DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ginza_reference.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.abspath(self.DEFAULT_DB_PATH)
        self._cache_xpos: Dict[str, Dict[str, str]] = {}
        self._cache_deprel: Dict[str, Dict[str, str]] = {}
        self._cache_bunsetsu: Dict[str, Dict[str, str]] = {}

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        return sqlite3.connect(self.db_path) if os.path.exists(self.db_path) else None

    def explain_xpos(self, code: str) -> Optional[Dict[str, str]]:
        """Queries linguistic explanation for UniDic/Sudachi XPOS tag (e.g. '助詞-格助詞')."""
        if not code:
            return None
        if code in self._cache_xpos:
            return self._cache_xpos[code]

        conn = self._get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute("SELECT code, name, vietnamese, description FROM xpos WHERE code = ? LIMIT 1", (code,))
            r = cur.fetchone()
            if r:
                res = {"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]}
                self._cache_xpos[code] = res
                return res
        finally:
            conn.close()
        return None

    def explain_deprel(self, code: str) -> Optional[Dict[str, str]]:
        """Queries explanation for Universal Dependency relation (e.g. 'compound', 'fixed')."""
        if not code:
            return None
        if code in self._cache_deprel:
            return self._cache_deprel[code]

        conn = self._get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute("SELECT code, name, vietnamese, description FROM deprel WHERE code = ? LIMIT 1", (code,))
            r = cur.fetchone()
            if r:
                res = {"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]}
                self._cache_deprel[code] = res
                return res
        finally:
            conn.close()
        return None

    def explain_bunsetsu_pos(self, code: str) -> Optional[Dict[str, str]]:
        """Queries explanation for Bunsetsu position type."""
        if not code:
            return None
        if code in self._cache_bunsetsu:
            return self._cache_bunsetsu[code]

        conn = self._get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT code, name, vietnamese, description FROM bunsetsu_position_type WHERE code = ? LIMIT 1",
                (code,)
            )
            r = cur.fetchone()
            if r:
                res = {"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]}
                self._cache_bunsetsu[code] = res
                return res
        finally:
            conn.close()
        return None
