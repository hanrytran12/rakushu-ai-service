"""Database schema initialization for OOV candidates and curator reviews."""
import sqlite3


def init_oov_tables(conn: sqlite3.Connection):
    """Creates tables and indexes for OOV candidates and curator reviews if not present."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS oov_candidates (
            oov_candidate_id TEXT PRIMARY KEY,
            token_id TEXT,
            term TEXT NOT NULL,
            tentative_reading TEXT,
            tentative_pos TEXT,
            suggested_meaning TEXT,
            context_snippet TEXT,
            confidence_score REAL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'PENDING_CURATOR_REVIEW',
            detected_at TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oov_term ON oov_candidates(term)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oov_status ON oov_candidates(status)")
    conn.commit()
