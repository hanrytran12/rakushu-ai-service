import sqlite3
import json
import os

DB_PATH = r"d:\Clone\rakushu-ai-service\data\ginza_reference.db"
JSON_PATH = r"d:\Clone\rakushu-ai-service\data\ginza_reference.json"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 1. XPOS DATA (Full 52 UniDic/Sudachi Tags extracted directly from dictionary)
xpos_data = [
    ("代名詞", "Pronoun", "Đại từ", "Đại từ chỉ người, vật, địa điểm (ví dụ: 私, 彼, ここ)"),
    ("副詞", "Adverb", "Phó từ", "Phó từ bổ nghĩa hành động/trạng thái (ví dụ: ゆっくり, すぐ, もっと)"),
    ("助動詞", "Auxiliary Verb", "Trợ động từ", "Trợ động từ biểu thị thể, thời, ý chí (ví dụ: だ, です, ます, ない, た, れる)"),
    ("助詞-係助詞", "Particle - Binding", "Trợ từ liên kết/nhấn mạnh", "Trợ từ nhấn mạnh/nêu chủ đề (ví dụ: は, も, こそ, さえ)"),
    ("助詞-副助詞", "Particle - Adverbial", "Phó trợ từ", "Trợ từ biểu thị mức độ, phạm vi (ví dụ: だけ, ばかり, ほど, くらい)"),
    ("助詞-接続助詞", "Particle - Conjunctional", "Trợ từ nối tiếp", "Nối tiếp mệnh đề chỉ điều kiện, nguyên nhân, chuyển tiếp (ví dụ: て, たら, ので, から)"),
    ("助詞-格助詞", "Particle - Case", "Trợ từ cách (Cách trợ từ)", "Chỉ quan hệ ngữ pháp giữa danh từ và vị ngữ (ví dụ: が, を, に, で, へ, と, から, より)"),
    ("助詞-準体助詞", "Particle - Phrasal", "Trợ từ danh từ hoá", "Trợ từ thay thế hoặc danh từ hoá ngữ (ví dụ: の - trong 赤いのが好き)"),
    ("助詞-終助詞", "Particle - Sentence-final", "Trợ từ cuối câu (Thán trợ từ)", "Biểu thị ngữ điệu, cảm xúc người nói ở cuối câu (ví dụ: ね, よ, な, わ, か)"),
    ("動詞-一般", "Verb - General", "Động từ thông thường", "Động từ độc lập biểu thị hành động, trạng thái (ví dụ: 走る, 読む, 聞く)"),
    ("動詞-非自立可能", "Verb - Bound/Auxiliary capable", "Động từ có thể làm trợ động từ", "Động từ phụ/hỗ trợ ngữ pháp khi đi sau て (ví dụ: いる, みる, くる, いく, くれる)"),
    ("名詞-助動詞語幹", "Noun - Auxiliary Stem", "Gốc danh từ làm trợ động từ", "Gốc từ danh từ có chức năng tương tự trợ động từ"),
    ("名詞-固有名詞-一般", "Proper Noun - General", "Danh từ riêng thông thường", "Tên tổ chức, thương hiệu, tác phẩm (ví dụ: Google, トヨタ)"),
    ("名詞-固有名詞-人名-一般", "Proper Noun - Person General", "Tên người chung", "Tên nhân vật lịch sử, tên đầy đủ"),
    ("名詞-固有名詞-人名-名", "Proper Noun - Given Name", "Tên riêng (Tên gọi)", "Phần tên trong họ tên người (ví dụ: 太郎, 花子, Huy)"),
    ("名詞-固有名詞-人名-姓", "Proper Noun - Surname", "Họ của người", "Phần họ trong họ tên người (ví dụ: 田中, 佐藤, Nguyễn)"),
    ("名詞-固有名詞-地名-一般", "Proper Noun - Place General", "Địa danh chung", "Tên thành phố, quận, địa điểm (ví dụ: 銀座, 新宿, 京都)"),
    ("名詞-固有名詞-地名-国", "Proper Noun - Country", "Tên quốc gia", "Tên đất nước (ví dụ: 日本, ベトナム, アメリカ)"),
    ("名詞-数詞", "Noun - Numeral", "Số từ", "Chữ số, số lượng (ví dụ: 一, 二, 1, 2)"),
    ("名詞-普通名詞-サ変可能", "Noun - Suru Verb capable", "Danh từ có thể ghép する", "Danh từ hành động (ví dụ: 勉強, 説明, 案内, 連絡)"),
    ("名詞-普通名詞-サ変形状詞可能", "Noun - Suru & Na-Adj capable", "Danh từ vừa làm する vừa làm tính từ な", "Từ linh hoạt (ví dụ: 安心, 心配)"),
    ("名詞-普通名詞-一般", "Noun - Common General", "Danh từ chung", "Danh từ thông dụng chỉ sự vật, hiện tượng (ví dụ: 友達, 言葉, 本)"),
    ("名詞-普通名詞-副詞可能", "Noun - Adverbial capable", "Danh từ kiêm phó từ", "Danh từ có thể dùng độc lập như phó từ chỉ thời gian (ví dụ: 昨日, 今日, 今)"),
    ("名詞-普通名詞-助数詞可能", "Noun - Counter capable", "Danh từ có thể làm lượng từ", "Danh từ dùng kèm số để đếm (ví dụ: 箇, 冊)"),
    ("名詞-普通名詞-形状詞可能", "Noun - Na-Adj capable", "Danh từ có thể làm tính từ な", "Danh từ có thể đi với な (ví dụ: 自由, 平和)"),
    ("形容詞-一般", "Adjective - General", "Tính từ đuôi い", "Tính từ i thuần túy (ví dụ: 美しい, 高い, 寒い)"),
    ("形容詞-非自立可能", "Adjective - Bound capable", "Tính từ đuôi い phụ thuộc", "Tính từ đuôi i làm bổ ngữ phụ thuộc (ví dụ: ない, よい, ほしい)"),
    ("形状詞-タリ", "Adjectival Noun - Tari", "Tính từ hình thái たり", "Tính từ cổ/văn viết kết thúc bằng たり (ví dụ: 堂々, 悠々)"),
    ("形状詞-一般", "Adjectival Noun - General (Na-Adj)", "Tính từ đuôi な (Hình trạng từ)", "Tính từ na thông dụng (ví dụ: 丁寧, 静か, 便利, きれい)"),
    ("形状詞-助動詞語幹", "Adjectival Noun - Aux Stem", "Gốc hình trạng từ đi kèm trợ động từ", "Gốc từ tính từ ghép trợ từ"),
    ("感動詞-フィラー", "Interjection - Filler", "Từ đệm ngập ngừng", "Âm thanh đệm khi suy nghĩ trong hội thoại (ví dụ: ええと, あのー, うーん)"),
    ("感動詞-一般", "Interjection - General", "Thán từ chung", "Tiếng chào, kêu gọi, cảm thán (ví dụ: はい, いいえ, ありがとう, こんにちは)"),
    ("接尾辞-動詞的", "Suffix - Verbal", "Hậu tố tạo động từ", "Hậu tố gắn sau tạo động từ (ví dụ: ぶる, づける)"),
    ("接尾辞-名詞的-サ変可能", "Suffix - Noun Suru", "Hậu tố tạo danh từ ghép する", "Hậu tố biến thành danh từ sahen (ví dụ: 化, 視)"),
    ("接尾辞-名詞的-一般", "Suffix - Noun General", "Hậu tố tạo danh từ chung", "Hậu tố chỉ người, vật, tình trạng (ví dụ: たち, 達, 性)"),
    ("接尾辞-名詞的-副詞可能", "Suffix - Noun Adverbial", "Hậu tố tạo phó từ", "Hậu tố tạo từ chỉ thời gian, tần suất (ví dụ: ごと, 毎)"),
    ("接尾辞-名詞的-助数詞", "Suffix - Counter", "Hậu tố lượng từ (Đơn vị đếm)", "Đơn vị đếm đứng sau số từ (ví dụ: 個, 本, 人, 匹, 歳)"),
    ("接尾辞-形容詞的", "Suffix - Adjectival", "Hậu tố tạo tính từ đuôi い", "Hậu tố biến từ thành tính từ đuôi i (ví dụ: ぽい, らしい)"),
    ("接尾辞-形状詞的", "Suffix - Na-Adj", "Hậu tố tạo tính từ đuôi な", "Hậu tố biến từ thành tính từ đuôi na (ví dụ: 的, 風)"),
    ("接続詞", "Conjunction", "Liên từ", "Từ nối câu độc lập (ví dụ: しかし, だから, また, そして)"),
    ("接頭辞", "Prefix", "Tiếp đầu ngữ (Tiền tố)", "Đứng trước danh từ/động từ để lịch sự hoặc bổ nghĩa (ví dụ: お, ご, 御, 第, 超)"),
    ("空白", "Whitespace", "Khoảng trắng", "Khoảng trống giữa các từ hoặc thụt lề"),
    ("補助記号-一般", "Auxiliary Symbol - General", "Ký hiệu phụ trợ chung", "Các ký tự biểu tượng, dấu chấm lửng (ví dụ: …, ・)"),
    ("補助記号-句点", "Auxiliary Symbol - Period", "Dấu chấm hết câu", "Dấu chấm tròn tiếng Nhật (。)"),
    ("補助記号-括弧閉", "Auxiliary Symbol - Close Bracket", "Dấu đóng ngoặc", "Ngoặc đóng (ví dụ: 」, ）, 】)"),
    ("補助記号-括弧開", "Auxiliary Symbol - Open Bracket", "Dấu mở ngoặc", "Ngoặc mở (ví dụ: 「, （, 【)"),
    ("補助記号-読点", "Auxiliary Symbol - Comma", "Dấu phẩy ngắt nhịp", "Dấu phẩy tiếng Nhật (、)"),
    ("補助記号-ＡＡ-一般", "Auxiliary Symbol - ASCII Art", "Ký hiệu ASCII Art", "Hình vẽ tạo từ ký tự text"),
    ("補助記号-ＡＡ-顔文字", "Auxiliary Symbol - Kaomoji", "Ký hiệu mặt cười Kaomoji", "Biểu tượng cảm xúc Nhật (ví dụ: (^^), (T_T))"),
    ("記号-一般", "Symbol - General", "Ký hiệu chung", "Ký hiệu văn bản thông dụng"),
    ("記号-文字", "Symbol - Character", "Ký hiệu chữ cái đặc biệt", "Ký hiệu như chữ La Tinh, chữ cái Hy Lạp"),
    ("連体詞", "Adnominal (Determiner)", "Liên thể từ", "Từ chỉ định không biến đổi đứng trước danh từ (ví dụ: この, その, あの, ある, いわゆる)")
]

# 2. DEPREL DATA (Universal Dependency Relations in GiNZA)
deprel_data = [
    ("root", "Root", "Vị ngữ trung tâm (Gốc câu)", "Động từ/danh từ vị ngữ đóng vai trò trung tâm của cả câu"),
    ("nsubj", "Nominal Subject", "Chủ ngữ danh từ", "Chủ ngữ của câu hoặc của mệnh đề phụ (đi với が hoặc は)"),
    ("obj", "Object", "Tân ngữ trực tiếp", "Tân ngữ nhận tác động trực tiếp của động từ (đi với を)"),
    ("iobj", "Indirect Object", "Tân ngữ gián tiếp", "Đối tượng tiếp nhận hành động (thường đi với に)"),
    ("obl", "Oblique Nominal", "Trạng ngữ danh từ", "Thành phần danh ngữ chỉ nơi chốn, thời gian, công cụ, đối tác (đi với で, に, から, について)"),
    ("advmod", "Adverbial Modifier", "Bổ ngữ phó từ", "Phó từ bổ nghĩa cho động từ, tính từ hoặc câu (ví dụ: 昨日, とても)"),
    ("advcl", "Adverbial Clause", "Mệnh đề trạng ngữ phụ", "Mệnh đề phụ chỉ nguyên nhân, điều kiện, tiền đề, so sánh (ví dụ: たら, て, より)"),
    ("acl", "Adnominal Clause", "Mệnh đề định ngữ", "Mệnh đề bổ nghĩa cho danh từ đứng sau (ví dụ: 勉強をしている [友達], 分からない [言葉])"),
    ("amod", "Adjectival Modifier", "Bổ ngữ tính từ", "Tính từ bổ nghĩa trực tiếp cho danh từ (ví dụ: 美しい 花)"),
    ("nmod", "Nominal Modifier", "Bổ ngữ danh từ", "Danh từ bổ nghĩa cho danh từ khác qua trợ từ の (ví dụ: 試験の [勉強])"),
    ("nummod", "Numeric Modifier", "Bổ ngữ số lượng", "Số lượng bổ nghĩa cho danh từ (ví dụ: 3冊の 本)"),
    ("compound", "Compound", "Từ ghép phức hợp", "Thành phần tạo nên danh từ ghép phức hợp (ví dụ: 日本語 + 能力 + 試験, 接頭辞 ご + 一緒)"),
    ("case", "Case Particle", "Trợ từ cách", "Trợ từ gắn với danh từ để thể hiện vai trò cách cú pháp (ví dụ: の, を, に, で, より)"),
    ("mark", "Marker", "Trợ từ kết nối mệnh đề", "Trợ từ hoặc liên từ đánh dấu kết thúc mệnh đề phụ (ví dụ: て, から, ので)"),
    ("aux", "Auxiliary", "Trợ từ/Trợ động từ bổ trợ", "Trợ động từ gắn kèm vị ngữ chỉ thì, thể, phủ định, kính ngữ (ví dụ: た, ます, ない, だ)"),
    ("cop", "Copula", "Hệ từ liên kết", "Từ nối định danh quan hệ A là B (ví dụ: だ, である)"),
    ("fixed", "Fixed Multiword", "Cụm từ ngữ cố định", "Cụm từ nhiều thành phần kết hợp thành một khối ngữ pháp (ví dụ: に+つい+て, て+みる, て+いる, て+くれる)"),
    ("flat", "Flat Name/Multiword", "Cụm tên ngang hàng", "Các từ ghép tên người, danh hiệu không có quan hệ phân cấp (ví dụ: 田中 太郎)"),
    ("conj", "Conjunct", "Thành phần đẳng lập", "Thành phần được liên kết đẳng lập với từ phía trước (ví dụ: リンゴ と バナナ)"),
    ("cc", "Coordinating Conjunction", "Liên từ kết hợp", "Liên từ liên kết thành phần đẳng lập (ví dụ: そして, また)"),
    ("csubj", "Clausal Subject", "Mệnh đề làm chủ ngữ", "Mệnh đề đóng vai trò chủ ngữ của câu"),
    ("ccomp", "Clausal Complement", "Mệnh đề bổ ngữ nội dung", "Mệnh đề trích dẫn hoặc làm bổ ngữ cho động từ nhận thức (ví dụ: と 思う)"),
    ("det", "Determiner", "Từ chỉ định", "Từ chỉ định đứng trước danh từ (ví dụ: この, その)"),
    ("discourse", "Discourse Element", "Thành phần đệm/giao tiếp", "Thán từ hoặc thành phần tương tác đối thoại"),
    ("dislocated", "Dislocated Element", "Thành phần chủ đề tách rời", "Thành phần chuyển dịch chủ đề ở đầu câu đi với は"),
    ("clf", "Classifier", "Lượng từ / Đơn vị đếm", "Từ đếm đi liền số lượng"),
    ("appos", "Appositional Modifier", "Đồng vị ngữ", "Danh từ giải thích làm rõ cho danh từ đứng trước"),
    ("vocative", "Vocative", "Hô ngữ", "Từ dùng để gọi đáp"),
    ("punct", "Punctuation", "Dấu câu", "Dấu chấm, phẩy, ngoặc đơn")
]

# 3. BUNSETSU POSITION TYPE DATA (Full 6 Position Types in GiNZA Bunsetsu Recognizer)
bunsetsu_pos_type_data = [
    ("ROOT", "Root Head of Clause/Sentence", "Gốc vị ngữ của văn tiết trung tâm",
     "Token là gốc ngữ pháp của văn tiết trung tâm của câu (t.i == t.head.i), thường là vị ngữ chính."),
    ("SEM_HEAD", "Semantic Head of Bunsetsu", "Trọng tâm ngữ nghĩa của văn tiết",
     "Token mang nội dung ngữ nghĩa chính (Content Word) của văn tiết, ví dụ danh từ chính hoặc động từ chính trong cụm."),
    ("SYN_HEAD", "Syntactic Head of Bunsetsu", "Trọng tâm cú pháp của văn tiết",
     "Token chức năng (Function Word: AUX, ADP, SCONJ, CCONJ, PART) đứng cuối cùng của chuỗi từ chức năng, liên kết văn tiết với phần còn lại."),
    ("FUNC", "Secondary Function Word", "Từ chức năng phụ trợ",
     "Token chức năng đứng trước SYN_HEAD trong chuỗi từ chức năng của văn tiết (ví dụ khi có nhiều trợ động từ nối tiếp)."),
    ("CONT", "Continuation / Internal Token", "Thành phần nội bộ tiếp nối",
     "Token nằm trong khối nội bộ của văn tiết (tiền tố, từ cấu thành từ ghép, hoặc dấu câu kết nối)."),
    ("NO_HEAD", "Non-Head Punctuation", "Dấu câu không có đầu ngữ nghĩa",
     "Token dấu câu đứng riêng hoặc không tạo thành trọng tâm ngữ nghĩa (thường có dep_ == 'punct').")
]

# If DB file exists, delete it first to ensure a fresh rebuild without old tables
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE xpos (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vietnamese TEXT NOT NULL,
    description TEXT
)
""")

cur.execute("""
CREATE TABLE deprel (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vietnamese TEXT NOT NULL,
    description TEXT
)
""")

cur.execute("""
CREATE TABLE bunsetsu_position_type (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vietnamese TEXT NOT NULL,
    description TEXT
)
""")

cur.executemany("INSERT INTO xpos VALUES (?, ?, ?, ?)", xpos_data)
cur.executemany("INSERT INTO deprel VALUES (?, ?, ?, ?)", deprel_data)
cur.executemany("INSERT INTO bunsetsu_position_type VALUES (?, ?, ?, ?)", bunsetsu_pos_type_data)

conn.commit()
conn.close()

# Export JSON (without upos)
full_db_json = {
    "xpos": [{"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]} for r in xpos_data],
    "deprel": [{"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]} for r in deprel_data],
    "bunsetsu_position_type": [{"code": r[0], "name": r[1], "vietnamese": r[2], "description": r[3]} for r in bunsetsu_pos_type_data]
}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(full_db_json, f, ensure_ascii=False, indent=2)

print("Database recreated successfully at:", DB_PATH)
print("JSON reference recreated successfully at:", JSON_PATH)
print(f"Tables in DB: XPOS={len(xpos_data)}, DepRel={len(deprel_data)}, BunsetsuPosType={len(bunsetsu_pos_type_data)}")
