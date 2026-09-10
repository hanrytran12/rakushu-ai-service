"""Database Builder: Compiles 100% Yomitan Vietnamese dictionary into SQLite."""
import os, sys, json, sqlite3, zipfile, urllib.request, re
from typing import Dict, List, Tuple

JMDICT_VI_URL = "https://raw.githubusercontent.com/dreamofi/yomichan-Vietnamese-dictionary/master/vietnameseDict/jmdict_vietnamese.zip"

EXPLICIT_TAG_MAP = {
    "news": ("frequent", "Xuất hiện nhiều trên báo chí"), "ichi": ("frequent", "Top từ thông dụng (Ichimango)"),
    "spec": ("frequent", "Từ thông dụng đặc biệt"), "gai": ("frequent", "Từ mượn tiếng nước ngoài"),
    "P": ("popular", "Từ vựng đại chúng phổ biến"), "ksb": ("dialect", "Phương ngữ Kansai (Kansai-ben)"),
    "osb": ("dialect", "Phương ngữ Osaka (Osaka-ben)"), "ktb": ("dialect", "Phương ngữ Kanto (Kanto-ben)"),
    "kyb": ("dialect", "Phương ngữ Kyoto (Kyoto-ben)"), "kyu": ("dialect", "Phương ngữ Kyushu (Kyushu-ben)"),
    "thb": ("dialect", "Phương ngữ Tohoku (Tohoku-ben)"), "tsug": ("dialect", "Phương ngữ Tsugaru (Tsugaru-ben)"),
    "tsb": ("dialect", "Phương ngữ Tosa (Tosa-ben)"), "nab": ("dialect", "Phương ngữ Nagano (Nagano-ben)"),
    "rkb": ("dialect", "Phương ngữ Ryukyu / Okinawa"), "hob": ("dialect", "Phương ngữ Hokkaido (Hokkaido-ben)"),
    "Buddh": ("field", "Thuật ngữ Phật giáo"), "MA": ("field", "Thuật ngữ Võ thuật"),
    "comp": ("field", "Thuật ngữ Tin học / Máy tính"), "food": ("field", "Thuật ngữ Ẩm thực / Món ăn"),
    "med": ("field", "Thuật ngữ Y học / Y tế"), "anat": ("field", "Thuật ngữ Giải phẫu học"),
    "geom": ("field", "Thuật ngữ Hình học"), "chem": ("field", "Thuật ngữ Hóa học"),
    "physics": ("field", "Thuật ngữ Vật lý học"), "math": ("field", "Thuật ngữ Toán học"),
    "law": ("field", "Thuật ngữ Pháp luật"), "econ": ("field", "Thuật ngữ Kinh tế học"),
    "finc": ("field", "Thuật ngữ Tài chính"), "bus": ("field", "Thuật ngữ Kinh doanh"),
    "engr": ("field", "Thuật ngữ Kỹ thuật / Công trình"), "biol": ("field", "Thuật ngữ Sinh học"),
    "bot": ("field", "Thuật ngữ Thực vật học"), "zool": ("field", "Thuật ngữ Động vật học"),
    "geol": ("field", "Thuật ngữ Địa chất học"), "astron": ("field", "Thuật ngữ Thiên văn học"),
    "archit": ("field", "Thuật ngữ Kiến trúc"), "ling": ("field", "Thuật ngữ Ngôn ngữ học"),
    "mil": ("field", "Thuật ngữ Quân sự"), "Shinto": ("field", "Thuật ngữ Thần đạo (Shinto)"),
    "sumo": ("field", "Thuật ngữ Đấu vật Sumo"), "shogi": ("field", "Thuật ngữ Cờ tướng Shogi"),
    "mahj": ("field", "Thuật ngữ Mạt chược"), "sports": ("field", "Thuật ngữ Thể thao"),
    "baseb": ("field", "Thuật ngữ Bóng chày"), "music": ("field", "Thuật ngữ Âm nhạc"),
    "male": ("misc", "Cách nói / từ ngữ của nam giới"), "fem": ("misc", "Cách nói / từ ngữ của nữ giới"),
    "sl": ("misc", "Tiếng lóng (Slang)"), "m-sl": ("misc", "Tiếng lóng trong Manga"),
    "male-sl": ("misc", "Tiếng lóng của nam giới"), "col": ("misc", "Khẩu ngữ / Văn nói hàng ngày"),
    "fam": ("misc", "Cách nói thân mật, gia đình"), "pol": ("misc", "Thể lịch sự (Teineigo)"),
    "hon": ("misc", "Kính ngữ tôn kính (Sonkeigo)"), "hum": ("misc", "Khiêm nhường ngữ (Kenjougo)"),
    "derog": ("misc", "Từ khiếm nhã / xúc phạm / miệt thị"), "vulg": ("misc", "Từ thô tục / tục tĩu"),
    "X": ("misc", "Từ nhạy cảm / 18+"), "sens": ("misc", "Từ ngữ nhạy cảm"),
    "joc": ("misc", "Cách nói hài hước / trêu đùa"), "proverb": ("misc", "Tục ngữ / Châm ngôn"),
    "yoji": ("misc", "Thành ngữ 4 chữ Hán (Yojijukugo)"), "id": ("misc", "Quán ngữ / Thành ngữ"),
    "poet": ("misc", "Từ ngữ thi ca / Văn chương"), "chn": ("misc", "Ngôn ngữ trẻ em"),
    "abbr": ("misc", "Từ viết tắt"), "arch": ("misc", "Từ cổ / Lối dùng cổ"),
    "obs": ("misc", "Từ lỗi thời"), "obsc": ("misc", "Từ tối nghĩa, ít dùng"),
    "rare": ("misc", "Từ hiếm gặp"), "uk": ("misc", "Thường chỉ viết bằng Kana"),
    "uK": ("misc", "Thường chỉ viết bằng Kanji"), "ek": ("misc", "Chỉ viết bằng Kana"),
    "eK": ("misc", "Chỉ viết bằng Kanji"), "ateji": ("misc", "Mượn chữ Hán đọc theo âm (Ateji)"),
    "gikun": ("misc", "Cách đọc nghĩa chữ Hán (Gikun)"), "ik": ("misc", "Kana dùng bất quy tắc"),
    "ok": ("misc", "Kana kiểu cổ"), "iK": ("misc", "Kanji dùng bất quy tắc"),
    "oK": ("misc", "Kanji kiểu cổ"), "io": ("misc", "Okurigana bất quy tắc"),
    "oik": ("misc", "Cách viết Kana cổ / bất quy tắc"), "on-mim": ("misc", "Từ tượng thanh / tượng hình"),
    "adj-kari": ("partOfSpeech", "Tính từ đuôi kari (cổ)"), "adj-ku": ("partOfSpeech", "Tính từ đuôi ku (cổ)"),
    "adj-shiku": ("partOfSpeech", "Tính từ đuôi shiku (cổ)"), "adj-nari": ("partOfSpeech", "Tính từ đuôi nari (cổ)"),
    "adj-pn": ("partOfSpeech", "Từ hạn định đứng trước danh từ (Rentaishi)"),
    "vs": ("partOfSpeech", "Động từ nhóm 3 (Suru)"), "vs-c": ("partOfSpeech", "Động từ Su (tiền thân của Suru)"),
    "vr": ("partOfSpeech", "Động từ bất quy tắc đuôi ru"), "vn": ("partOfSpeech", "Động từ bất quy tắc đuôi nu"),
    "v-unspec": ("partOfSpeech", "Động từ chưa phân nhóm"), "exp": ("partOfSpeech", "Cụm từ biểu đạt"),
    "num": ("partOfSpeech", "Số từ"), "cop-da": ("partOfSpeech", "Hệ từ khẳng định (da)"),
    "unc": ("partOfSpeech", "Từ loại chưa phân loại cụ thể")
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

def format_raw_meaning(raw_gloss: list, term: str) -> str:
    if not raw_gloss: return term
    return "\n".join(str(g).strip() for g in raw_gloss if str(g).strip())

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
        definition_tags TEXT DEFAULT '', rules TEXT DEFAULT '', score INTEGER DEFAULT 1,
        meaning TEXT NOT NULL, sequence INTEGER DEFAULT 0, term_tags TEXT DEFAULT '',
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
            name, raw_cat, desc_en = t[0], t[1], t[3]
            if name in EXPLICIT_TAG_MAP:
                cat, desc_vi = EXPLICIT_TAG_MAP[name]
            else:
                cat = raw_cat or "partOfSpeech"
                d = desc_en.lower()
                tag_patterns = {
                    "ichidan": "Động từ nhóm 2 (Ichidan)", "godan": "Động từ nhóm 1 (Godan)",
                    "suru": "Động từ nhóm 3 (Suru)", "kuru": "Động từ Kuru (Đến)",
                    "transitive": "Tha động từ (có tân ngữ)", "intransitive": "Tự động từ (không cần tân ngữ)",
                    "irregular": "Động từ bất quy tắc", "common noun": "Danh từ thông thường",
                    "proper noun": "Danh từ riêng", "temporal": "Danh từ chỉ thời gian",
                    "counter": "Từ chỉ số đếm (lượng từ)", "pronoun": "Đại từ",
                    "adjective": "Tính từ", "adverb": "Phó từ", "particle": "Trợ từ",
                    "conjunction": "Liên từ", "interjection": "Thán từ", "prefix": "Tiền tố",
                    "suffix": "Hậu tố", "expression": "Cụm từ biểu đạt"
                }
                desc_vi = next((v for k, v in tag_patterns.items() if k in d), desc_en)
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
                meaning = format_raw_meaning(item[5] if len(item) > 5 else [], term)
                seq = int(item[6]) if len(item) > 6 and isinstance(item[6], int) else 0
                term_tags = item[7].strip() if len(item) > 7 else ""
                pos = map_pos(def_tags)
                batch.append((term, reading, pos, def_tags, rule, score, meaning, seq, term_tags))
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
