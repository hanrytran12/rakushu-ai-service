"""Knowledge Service: Mock dictionary database (JMDict sidecar) and OOV detector."""
from typing import List, Tuple, Dict, Optional
from src import TokenModel, DictionaryEntry, OovCandidate


class KnowledgeService:
    """Manages dictionary lookup against JMDict/Tatoeba sidecar DB and identifies OOV words."""

    def __init__(self):
        # Simulated JMDict / System Knowledge Base
        # Intentionally omitting "若者言葉" and "ポッドキャスト" to test OOV detection
        self.mock_dictionary: Dict[str, DictionaryEntry] = {
            "皆さん": DictionaryEntry(
                term="皆さん", reading="みなさん", pos="NOUN",
                meaning="mọi người, tất cả các bạn (everyone)", jlpt_level="N5", commonality=1
            ),
            "こんにちは": DictionaryEntry(
                term="こんにちは", reading="こんにちは", pos="INTERJECTION",
                meaning="xin chào (hello/good afternoon)", jlpt_level="N5", commonality=1
            ),
            "今日": DictionaryEntry(
                term="今日", reading="きょう", pos="NOUN",
                meaning="hôm nay (today)", jlpt_level="N5", commonality=1
            ),
            "日本": DictionaryEntry(
                term="日本", reading="にほん", pos="NOUN",
                meaning="Nhật Bản (Japan)", jlpt_level="N5", commonality=1
            ),
            "面白い": DictionaryEntry(
                term="面白い", reading="おもしろい", pos="ADJECTIVE",
                meaning="thú vị, hấp dẫn, vui tính (interesting/fun)", jlpt_level="N5", commonality=1
            ),
            "話す": DictionaryEntry(
                term="話す", reading="はなす", pos="VERB",
                meaning="nói, trò chuyện (to speak/talk)", jlpt_level="N5", commonality=1
            ),
            "話します": DictionaryEntry(
                term="話します", reading="はなします", pos="VERB",
                meaning="nói, trò chuyện (thể lịch sự ます)", jlpt_level="N5", commonality=1
            ),
            "について": DictionaryEntry(
                term="について", reading="について", pos="PARTICLE",
                meaning="về vấn đề gì đó (about / regarding)", jlpt_level="N4", commonality=1
            ),
        }

    def match_tokens(self, tokens: List[TokenModel]) -> Tuple[List[DictionaryEntry], List[TokenModel]]:
        """Cross-references tokens against knowledge base; categorizes into known vs OOV."""
        matched: List[DictionaryEntry] = []
        oov_tokens: List[TokenModel] = []
        seen_matched = set()

        for token in tokens:
            # Punctuation, particles, and auxiliary verbs are functional grammar, not vocab OOV
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

    def lookup(self, word: str) -> Optional[DictionaryEntry]:
        return self.mock_dictionary.get(word)

    def register_enriched_oov(self, entry: DictionaryEntry):
        """Adds validated OOV term back into system knowledge base."""
        self.mock_dictionary[entry.term] = entry
