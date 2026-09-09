"""Database Builder: Compiles 100% Yomitan Vietnamese dictionary into SQLite."""
import os, sys, json, sqlite3, zipfile, urllib.request, re
from typing import Dict, List, Tuple

JMDICT_VI_URL = "https://raw.githubusercontent.com/dreamofi/yomichan-Vietnamese-dictionary/master/vietnameseDict/jmdict_vietnamese.zip"

TAG_VI_MAP = {
    "n": "Danh từ thông dụng", "v1": "Động từ nhóm 2 (Ichidan)", "v5": "Động từ nhóm 1 (Godan)",
    "v5r": "Động từ nhóm 1 đuôi ru", "v5k": "Động từ nhóm 1 đuôi ku", "v5s": "Động từ nhóm 1 đuôi su",
    "v5t": "Động từ nhóm 1 đuôi tsu", "v5m": "Động từ nhóm 1 đuôi mu", "v5b": "Động từ nhóm 1 đuôi bu",
    "v5g": "Động từ nhóm 1 đuôi gu", "v5u": "Động từ nhóm 1 đuôi u", "vs": "Động từ nhóm 3 / Danh từ Suru",
    "vk": "Động từ Kuru (Đến)", "adj-i": "Tính từ đuôi -i", "adj-na": "Tính từ đuôi -na",
    "adj-no": "Danh từ làm định ngữ (kèm no)", "adv": "Phó từ", "adv-to": "Phó từ kèm trợ từ to",
    "prt": "Trợ từ", "int": "Thán từ", "conj": "Liên từ", "num": "Số từ", "sl": "Tiếng lóng / Slang",
    "id": "Thành ngữ / Quán ngữ", "Buddh": "Thuật ngữ Phật giáo", "MA": "Thuật ngữ Võ thuật",
    "food": "Ẩm thực / Thức ăn", "comp": "Thuật ngữ Tin học", "ling": "Thuật ngữ Ngôn ngữ học",
    "anat": "Thuật ngữ Giải phẫu", "geom": "Thuật ngữ Hình học", "ecol": "Thuật ngữ Sinh thái học",
    "chem": "Thuật ngữ Hóa học", "math": "Thuật ngữ Toán học", "physics": "Thuật ngữ Vật lý",
    "mil": "Thuật ngữ Quân sự", "law": "Thuật ngữ Pháp luật", "econ": "Thuật ngữ Kinh tế",
    "P": "Từ vựng phổ biến", "ichi": "Top từ thông dụng (Ichimango)", "news": "Hay dùng trên báo chí",
    "spec": "Từ vựng đặc biệt", "gai": "Từ mượn tiếng nước ngoài", "abbr": "Từ viết tắt",
    "arch": "Từ cổ / Nghĩa cổ", "fam": "Cách nói thân mật / gia đình", "pol": "Thể lịch sự",
    "hon": "Kính ngữ (Sonkeigo)", "hum": "Khiêm nhường ngữ (Kenjougo)", "derog": "Từ khiếm nhã / xúc phạm"
}

INFLECTION_RULES_DATA = [
    ("v1", "Động từ nhóm 2 (一段動詞 - Ichidan)", "Bỏ đuôi る kết hợp trực tiếp với trợ động từ.", "食べる -> 食べた, 食べて, 食べない, 食べられる"),
    ("v5", "Động từ nhóm 1 (五段動詞 - Godan)", "Biến đổi hàng âm u sang a/i/u/e/o và biến âm âm vị Te/Ta.", "行く -> 行った, 話す -> 話して, 飲む -> 飲んだ"),
    ("vs", "Động từ nhóm 3 (サ変動詞 - Suru)", "Gồm động từ する và các danh từ ghép mang ý nghĩa hành động.", "する -> して, した, しない, 勉強する -> 勉強した"),
    ("vk", "Động từ nhóm 3 (カ変動詞 - Kuru)", "Động từ bất quy tắc 来る (kuru).", "来る -> 来て, 来た, 来ない (kô-nai)"),
    ("adj-i", "Tính từ đuôi -i (形容詞)", "Bỏ い đổi thành かった (quá khứ), くない (phủ định), く (phó từ).", "高い -> 高かった, 高くない, 高く"),
    ("v5 v1", "Động từ lưỡng tính (Godan / Ichidan)", "Có thể chia theo cả 2 cách Godan hoặc Ichidan tùy vùng.", "交ぜる / 交じる"),
    ("v5 vs", "Động từ lưỡng tính (Godan / Suru)", "Động từ có thể biến đổi theo kiểu Godan hoặc đi với Suru.", "愛する / 愛す")
]

def download_cached(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while chunk := resp.read(1024 * 256):
            out_f.write(chunk)
    return dest_path

def map_pos(tag_str: str) -> str:
    t = (tag_str or "").lower()
    if any(k in t for k in ["v1", "v5", "vk", "vs", "vt", "vi", "aux-v"]): return "VERB"
    if any(k in t for k in ["adj-i", "adj-na", "adj-no", "adj"]): return "ADJECTIVE"
    if "adv" in t: return "ADVERB"
    if "prt" in t: return "PARTICLE"
    if "int" in t: return "INTERJECTION"
    if "conj" in t: return "CONJUNCTION"
    if "num" in t: return "NUMERAL"
    return "NOUN"

def clean_meaning(raw_gloss: list, term: str) -> str:
    meanings = []
    for g in raw_gloss:
        if isinstance(g, str): meanings.append(g.strip())
        elif isinstance(g, dict) and "content" in g: meanings.append(str(g["content"]).strip())
    raw = "; ".join(meanings) if meanings else term
    lines = raw.split("\n")
    vi_cands = []
    for line in lines:
        m = re.search(r"-\s*\{[^}]+\},\s*([^,\n;]+)", line)
        if m: vi_cands.append(m.group(1).strip()); continue
        words = [w.strip() for w in line.split(";") if any(c in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ" for c in w.lower())]
        if words:
            for w in words:
                sub = re.sub(r"^[^\s]+\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s]+$", "", w).strip()
                vi_cands.append(sub if sub else w)
    return ", ".join(dict.fromkeys(vi_cands[:2])) if vi_cands else raw.split(";")[0].strip()[:60]

def build_database(db_path: str = "data/dictionary.db", cache_dir: str = "data/cache"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    jmdict_zip = download_cached(JMDICT_VI_URL, os.path.join(cache_dir, "jmdict_vietnamese.zip"))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS dictionary;")
    cur.execute("DROP TABLE IF EXISTS definition_tags;")
    cur.execute("DROP TABLE IF EXISTS inflection_rules;")

    cur.execute("""CREATE TABLE dictionary (
        term TEXT NOT NULL, reading TEXT NOT NULL DEFAULT '', pos TEXT NOT NULL DEFAULT 'NOUN',
        definition_tags TEXT DEFAULT '', rules TEXT DEFAULT '', meaning TEXT NOT NULL,
        commonality INTEGER DEFAULT 1, sequence INTEGER DEFAULT 0, jlpt_level TEXT DEFAULT '',
        PRIMARY KEY (term, reading));""")
    cur.execute("CREATE INDEX idx_dict_term ON dictionary(term);")
    cur.execute("CREATE INDEX idx_dict_reading ON dictionary(reading);")

    cur.execute("""CREATE TABLE definition_tags (
        name TEXT PRIMARY KEY, category TEXT DEFAULT '', description_en TEXT NOT NULL, description_vi TEXT NOT NULL);""")
    cur.execute("""CREATE TABLE inflection_rules (
        rule_code TEXT PRIMARY KEY, name_vi TEXT NOT NULL, description_vi TEXT NOT NULL, examples TEXT NOT NULL);""")

    cur.executemany("INSERT INTO inflection_rules VALUES (?, ?, ?, ?)", INFLECTION_RULES_DATA)

    with zipfile.ZipFile(jmdict_zip, "r") as z:
        tags_raw = json.load(z.open("tag_bank_1.json"))
        tag_rows = []
        for t in tags_raw:
            name, cat, desc_en = t[0], t[1], t[3]
            desc_vi = TAG_VI_MAP.get(name, desc_en)
            tag_rows.append((name, cat, desc_en, desc_vi))
        cur.executemany("INSERT OR REPLACE INTO definition_tags VALUES (?, ?, ?, ?)", tag_rows)

        term_files = [f for f in z.namelist() if f.startswith("term_bank_") and f.endswith(".json")]
        batch = []
        for t_file in term_files:
            for item in json.load(z.open(t_file)):
                term = item[0].strip()
                reading = item[1].strip() if len(item) > 1 else ""
                def_tags = item[2].strip() if len(item) > 2 else ""
                rule = item[3].strip() if len(item) > 3 else ""
                score = int(item[4]) if len(item) > 4 and isinstance(item[4], int) else 1
                meaning = clean_meaning(item[5] if len(item) > 5 else [], term)
                seq = int(item[6]) if len(item) > 6 and isinstance(item[6], int) else 0
                pos = map_pos(def_tags)
                batch.append((term, reading, pos, def_tags, rule, meaning, score, seq, ""))
                if len(batch) >= 4000:
                    cur.executemany("INSERT OR REPLACE INTO dictionary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch = []
        if batch:
            cur.executemany("INSERT OR REPLACE INTO dictionary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)

    conn.commit()
    conn.close()
    print(f"[Done] Compiled {db_path} ({os.path.getsize(db_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_database()
