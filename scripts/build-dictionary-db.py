"""Database Builder: Downloads and compiles JLPT and Yomitan Vietnamese dictionary into SQLite."""
import os
import sys
import csv
import json
import sqlite3
import zipfile
import urllib.request
from typing import Dict, Tuple, Any

JLPT_URL = "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/out/all.csv"
JMDICT_VI_URL = "https://raw.githubusercontent.com/dreamofi/yomichan-Vietnamese-dictionary/master/vietnameseDict/jmdict_vietnamese.zip"


def download_cached(url: str, dest_path: str) -> str:
    """Downloads a file if not already cached locally."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[Cache Hit] Using existing file: {dest_path}")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[Downloading] {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out_f.write(chunk)
    print(f"[Done] Downloaded {os.path.getsize(dest_path)} bytes to {dest_path}")
    return dest_path


def map_pos(tag_str: str) -> str:
    """Maps Yomichan/JMdict POS abbreviations to GiNZA NLP POS tags."""
    if not tag_str:
        return "NOUN"
    t = tag_str.lower()
    if any(k in t for k in ["v1", "v5", "vk", "vs", "vt", "vi", "aux-v"]):
        return "VERB"
    if any(k in t for k in ["adj-i", "adj-na", "adj-no", "adj"]):
        return "ADJECTIVE"
    if "adv" in t:
        return "ADVERB"
    if "prt" in t:
        return "PARTICLE"
    if "int" in t:
        return "INTERJECTION"
    if "conj" in t:
        return "CONJUNCTION"
    if "num" in t:
        return "NUMERAL"
    return "NOUN"


def parse_jlpt(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Parses JLPT vocabulary list into mapping table."""
    jlpt_map: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expr = (row.get("expression") or "").strip()
            reading = (row.get("reading") or "").strip()
            meaning = (row.get("meaning") or "").strip()
            tags = (row.get("tags") or "").strip()
            level = ""
            for tag in tags.split():
                m = re.search(r"JLPT_N?([1-5])", tag)
                if m:
                    level = f"N{m.group(1)}"
                    break
            if expr:
                jlpt_map[f"{expr}::{reading}"] = {"level": level, "meaning_en": meaning}
                if expr not in jlpt_map:
                    jlpt_map[expr] = {"level": level, "meaning_en": meaning}
    return jlpt_map


def build_database(db_path: str = "data/dictionary.db", cache_dir: str = "data/cache"):
    """Downloads resources and compiles the unified SQLite dictionary."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    jlpt_file = download_cached(JLPT_URL, os.path.join(cache_dir, "jlpt_all.csv"))
    jmdict_zip = download_cached(JMDICT_VI_URL, os.path.join(cache_dir, "jmdict_vietnamese.zip"))

    print("[Parsing] Loading JLPT vocabulary mapping...")
    jlpt_map = parse_jlpt(jlpt_file)
    print(f"Loaded {len(jlpt_map)} JLPT index references.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table & indexes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            term TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '',
            pos TEXT NOT NULL DEFAULT 'NOUN',
            meaning TEXT NOT NULL,
            jlpt_level TEXT DEFAULT '',
            commonality INTEGER DEFAULT 1,
            PRIMARY KEY (term, reading)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_term ON dictionary(term);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_reading ON dictionary(reading);")

    print("[Extracting] Processing Yomitan Vietnamese term banks from zip...")
    total_inserted = 0
    batch = []

    with zipfile.ZipFile(jmdict_zip, "r") as z:
        term_files = [f for f in z.namelist() if f.startswith("term_bank_") and f.endswith(".json")]
        for t_file in term_files:
            with z.open(t_file) as f_in:
                entries = json.load(f_in)
                for item in entries:
                    term = item[0].strip()
                    reading = item[1].strip() if len(item) > 1 else ""
                    pos_tags = item[2] if len(item) > 2 else ""
                    score = int(item[4]) if len(item) > 4 and isinstance(item[4], int) else 1
                    raw_gloss = item[5] if len(item) > 5 else []

                    # Extract meanings
                    meanings = []
                    for g in raw_gloss:
                        if isinstance(g, str):
                            meanings.append(g.strip())
                        elif isinstance(g, dict) and "content" in g:
                            meanings.append(str(g["content"]).strip())
                    meaning_str = "; ".join(meanings) if meanings else term

                    # Determine JLPT Level
                    jlpt_info = jlpt_map.get(f"{term}::{reading}") or jlpt_map.get(term) or {}
                    jlpt_level = jlpt_info.get("level", "")
                    pos = map_pos(pos_tags)

                    batch.append((term, reading, pos, meaning_str, jlpt_level, score))
                    if len(batch) >= 2000:
                        cur.executemany("""
                            INSERT OR REPLACE INTO dictionary
                            (term, reading, pos, meaning, jlpt_level, commonality)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, batch)
                        total_inserted += len(batch)
                        batch = []

    if batch:
        cur.executemany("""
            INSERT OR REPLACE INTO dictionary
            (term, reading, pos, meaning, jlpt_level, commonality)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)
        total_inserted += len(batch)

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM dictionary;")
    count = cur.fetchone()[0]
    conn.close()

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n[Success] Database compiled: {db_path}")
    print(f"Total entries: {count:,} | Database size: {db_size_mb:.2f} MB")


if __name__ == "__main__":
    build_database()
