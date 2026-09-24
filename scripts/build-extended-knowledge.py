"""Extended Knowledge Builder: Ingests Grammar, CompoundWords, and Phrases into SQLite."""
import os, sys, json, sqlite3, urllib.request, re, csv
from typing import List, Tuple, Dict, Any
from sudachipy import dictionary as suda_dict, tokenizer as suda_tok

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "jmdict_1000_staging.db")
GITHUB_RAW = "https://raw.githubusercontent.com"
GRAMMAR_BASE = f"{GITHUB_RAW}/aiko-tanaka/Grammar-Dictionaries/main/nihongo_no_sensei"
COMPOUND_VERBS_URL = f"{GITHUB_RAW}/moltinginstar/japanese-compound-verbs/main/data/deck.csv"

def download_cached(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while chunk := resp.read(1024 * 128):
            out_f.write(chunk)
    return dest_path

def parse_nihongo_sensei_entry(item: list) -> Tuple[str, str, str, str, str, str, str]:
    term, reading, raw_def = item[0].strip(), item[1].strip(), item[5] if len(item) > 5 else []
    text_content = ""
    for block in raw_def:
        if isinstance(block, dict) and "content" in block:
            for c in block["content"]:
                text_content += c if isinstance(c, str) else (c.get("content", "") if isinstance(c, dict) else "")
        elif isinstance(block, str): text_content += block
    lvl_m = re.search(r"【(Ｎ[１-５]|N[1-5])文法】", text_content)
    lvl_map = {"Ｎ１": "N1", "Ｎ２": "N2", "Ｎ３": "N3", "Ｎ４": "N4", "Ｎ５": "N5"}
    jlpt = lvl_map.get(lvl_m.group(1), lvl_m.group(1)) if lvl_m else ""
    def get_section(name: str) -> str:
        m = re.search(rf"{name}\n(.*?)(?=\n(?:接続|意味|解説|例文|備考|---END---)|\Z)", text_content, re.S)
        return m.group(1).strip() if m else ""
    formation = get_section("接続")
    meaning = get_section("意味") or term
    nuance = get_section("解説")
    examples = get_section("例文")
    return (term, reading, jlpt, formation, meaning, nuance, examples)

def build_grammar(cur: sqlite3.Cursor):
    cur.execute("DROP TABLE IF EXISTS grammar;")
    cur.execute("""CREATE TABLE grammar (
        id INTEGER PRIMARY KEY AUTOINCREMENT, pattern TEXT NOT NULL, reading TEXT DEFAULT '',
        jlpt_level TEXT DEFAULT '', formation TEXT DEFAULT '', meaning TEXT NOT NULL,
        nuance TEXT DEFAULT '', examples TEXT DEFAULT '', source TEXT DEFAULT 'nihongo_no_sensei');""")
    cur.execute("CREATE INDEX idx_grammar_pattern ON grammar(pattern);")
    cur.execute("CREATE INDEX idx_grammar_level ON grammar(jlpt_level);")
    rows, seen = [], set()
    for i in range(1, 6):
        fpath = download_cached(f"{GRAMMAR_BASE}/term_bank_{i}.json", os.path.join(CACHE_DIR, f"grammar_sensei_{i}.json"))
        with open(fpath, "r", encoding="utf-8") as f:
            for item in json.load(f):
                parsed = parse_nihongo_sensei_entry(item)
                key = (parsed[0], parsed[2])
                if key not in seen:
                    seen.add(key); rows.append(parsed)
    cur.executemany("INSERT INTO grammar (pattern, reading, jlpt_level, formation, meaning, nuance, examples) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    print(f"  [Grammar] Inserted {len(rows)} grammar entries.")

def build_phrases(cur: sqlite3.Cursor):
    cur.execute("DROP TABLE IF EXISTS phrases;")
    cur.execute("""CREATE TABLE phrases (
        id INTEGER PRIMARY KEY AUTOINCREMENT, term TEXT NOT NULL, reading TEXT DEFAULT '',
        phrase_type TEXT NOT NULL DEFAULT 'EXPRESSION', meaning TEXT NOT NULL,
        tags TEXT DEFAULT '', source TEXT DEFAULT 'JMdict');""")
    cur.execute("CREATE INDEX idx_phrases_term ON phrases(term);")
    cur.execute("CREATE INDEX idx_phrases_reading ON phrases(reading);")
    cur.execute("CREATE INDEX idx_phrases_type ON phrases(phrase_type);")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='staging_entries';")
    has_staging = cur.fetchone() is not None
    rows, seen = [], set()
    if has_staging:
        cur.execute("SELECT term, reading, pos, meaning_vi, gloss_en FROM staging_entries WHERE pos LIKE '%expression%' OR pos LIKE '%idiom%' OR pos LIKE '%yoji%' OR pos LIKE '%proverb%';")
        for term, reading, pos_tag, m_vi, g_en in cur.fetchall():
            key = (term, reading)
            if key in seen: continue
            seen.add(key)
            p_type = "YOJIJUKUGO" if "yoji" in pos_tag else ("PROVERB" if "proverb" in pos_tag else ("IDIOM" if "idiom" in pos_tag else "EXPRESSION"))
            meaning = m_vi if m_vi else g_en
            rows.append((term, reading, p_type, meaning, pos_tag))
    else:
        cur.execute("SELECT term, reading, definition_tags, meaning FROM dictionary WHERE definition_tags LIKE '%exp%' OR definition_tags LIKE '%id%' OR definition_tags LIKE '%yoji%' OR definition_tags LIKE '%proverb%';")
        for term, reading, tags, meaning in cur.fetchall():
            key = (term, reading)
            if key in seen: continue
            seen.add(key)
            p_type = "YOJIJUKUGO" if "yoji" in tags else ("PROVERB" if "proverb" in tags else ("IDIOM" if "id" in tags else "EXPRESSION"))
            rows.append((term, reading, p_type, meaning, tags))
    cur.executemany("INSERT INTO phrases (term, reading, phrase_type, meaning, tags) VALUES (?, ?, ?, ?, ?)", rows)
    print(f"  [Phrases] Inserted {len(rows)} phrase/idiom entries.")

def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ").strip()

def build_compound_words(cur: sqlite3.Cursor):
    cur.execute("DROP TABLE IF EXISTS compound_words;")
    cur.execute("""CREATE TABLE compound_words (
        id INTEGER PRIMARY KEY AUTOINCREMENT, term TEXT NOT NULL, reading TEXT DEFAULT '',
        compound_type TEXT NOT NULL, components TEXT NOT NULL, components_reading TEXT DEFAULT '[]',
        transitivity TEXT DEFAULT '', meaning TEXT DEFAULT '', structure TEXT DEFAULT '');""")
    cur.execute("CREATE INDEX idx_compound_term ON compound_words(term);")
    cur.execute("CREATE INDEX idx_compound_type ON compound_words(compound_type);")
    verb_csv = download_cached(COMPOUND_VERBS_URL, os.path.join(CACHE_DIR, "compound_verbs_deck.csv"))
    rows, seen = [], set()
    with open(verb_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v, hr, v1, v2 = r.get("verb", "").strip(), r.get("verb_hiragana", "").strip(), r.get("verb_1", "").strip(), r.get("verb_2", "").strip()
            if not v or (v, hr) in seen: continue
            seen.add((v, hr))
            m = clean_html(r.get("meaning_en", "")) or r.get("meaning", "")
            rows.append((v, hr, "VERB", json.dumps([v1, v2], ensure_ascii=False), json.dumps([r.get("verb_1_hiragana",""), r.get("verb_2_hiragana","")], ensure_ascii=False), r.get("transitivity", ""), m, "V1 + V2"))
    tok = suda_dict.Dictionary().create()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='staging_entries';")
    has_staging = cur.fetchone() is not None
    if has_staging:
        cur.execute("SELECT term, reading, meaning_vi, gloss_en FROM staging_entries WHERE pos LIKE '%noun%' AND length(term) >= 4 LIMIT 15000;")
        for term, reading, m_vi, g_en in cur.fetchall():
            if (term, reading) in seen or not re.match(r"^[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+$", term): continue
            sub_tokens = tok.tokenize(term, suda_tok.Tokenizer.SplitMode.A)
            if len(sub_tokens) > 1:
                comps = [m.surface() for m in sub_tokens]
                comp_reads = [m.reading_form() for m in sub_tokens]
                seen.add((term, reading))
                m = m_vi or g_en or ""
                rows.append((term, reading, "NOUN", json.dumps(comps, ensure_ascii=False), json.dumps(comp_reads, ensure_ascii=False), "", m[:120].strip(), " + ".join([f"N{i+1}" for i in range(len(comps))])))
    else:
        cur.execute("SELECT term, reading, meaning FROM dictionary WHERE pos = 'NOUN' AND length(term) >= 4 AND score >= 1 LIMIT 12000;")
        for term, reading, meaning in cur.fetchall():
            if (term, reading) in seen or not re.match(r"^[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+$", term): continue
            sub_tokens = tok.tokenize(term, suda_tok.Tokenizer.SplitMode.A)
            if len(sub_tokens) > 1:
                comps = [m.surface() for m in sub_tokens]
                comp_reads = [m.reading_form() for m in sub_tokens]
                seen.add((term, reading))
                rows.append((term, reading, "NOUN", json.dumps(comps, ensure_ascii=False), json.dumps(comp_reads, ensure_ascii=False), "", meaning[:120].strip(), " + ".join([f"N{i+1}" for i in range(len(comps))])))
    cur.executemany("INSERT INTO compound_words (term, reading, compound_type, components, components_reading, transitivity, meaning, structure) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    print(f"  [CompoundWords] Inserted {len(rows)} compound words.")

def build_extended_knowledge(db_path: str = DEFAULT_DB):
    print(f"Target Database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    print("Building Extended Knowledge Tables (Grammar, Phrases, CompoundWords)...")
    build_grammar(cur)
    build_phrases(cur)
    build_compound_words(cur)
    conn.commit()
    conn.close()
    print(f"[Done] Extended knowledge tables populated successfully into {db_path}.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    build_extended_knowledge(target)
