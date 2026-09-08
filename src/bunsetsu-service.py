"""Bunsetsu Grouping Service: Japanese phrase boundary segmentation and alignment."""
from typing import List, Dict
from src import TokenModel, BunsetsuPhrase, SubtitleSegment


class BunsetsuService:
    """Groups tokens into Bunsetsu (文節: 1 Jiritsugo content word + N Fuzokugo particles)."""

    JIRITSUGO_POS = {"NOUN", "VERB", "ADJECTIVE", "INTERJECTION", "ADVERB", "PRONOUN"}
    FUZOKUGO_POS = {"PARTICLE", "AUX_VERB", "PUNCTUATION"}

    PHRASE_TRANSLATIONS: Dict[str, str] = {
        "皆さん、": "Mọi người,",
        "こんにちは。": "xin chào.",
        "今日の": "của hôm nay,",
        "ポッドキャストでは、": "trong tập podcast thì,",
        "日本の": "của Nhật Bản,",
        "面白い": "thú vị,",
        "若者言葉について": "về tiếng lóng giới trẻ,",
        "話します。": "sẽ nói chuyện / trò chuyện.",
    }

    def group_bunsetsu(self, segment: SubtitleSegment, tokens: List[TokenModel]) -> List[BunsetsuPhrase]:
        """Partitions sequential tokens into Bunsetsu phrases with timestamps and meanings."""
        phrases: List[BunsetsuPhrase] = []
        current_cluster: List[TokenModel] = []

        def flush_cluster(cluster: List[TokenModel], order: int):
            if not cluster:
                return None
            combined_text = "".join(t.surface for t in cluster)
            meaning = self.PHRASE_TRANSLATIONS.get(
                combined_text,
                self._generate_cluster_meaning(cluster)
            )
            # Distribute time proportionally across segment duration
            total_chars = max(len(segment.text), 1)
            start_ratio = cluster[0].start_position / total_chars
            end_ratio = cluster[-1].end_position / total_chars
            duration = segment.end_time - segment.start_time

            return BunsetsuPhrase(
                segment_id=segment.segment_id,
                text=combined_text,
                translation=meaning,
                start_time=round(segment.start_time + start_ratio * duration, 2),
                end_time=round(segment.start_time + end_ratio * duration, 2),
                phrase_order=order,
                token_surfaces=[t.surface for t in cluster]
            )

        order = 1
        for token in tokens:
            # Punctuation attaches to the preceding word
            if token.pos == "PUNCTUATION":
                current_cluster.append(token)
                phrase = flush_cluster(current_cluster, order)
                if phrase:
                    phrases.append(phrase)
                    order += 1
                current_cluster = []
                continue

            # Start of a new Jiritsugo (content word) triggers boundary flush
            # Note: Consecutive NOUNs (e.g. 若者 + 言葉) and NOUN + する (e.g. 勉強 + する) stay together
            if token.pos in self.JIRITSUGO_POS:
                is_compound_noun = (
                    token.pos == "NOUN" and
                    current_cluster and
                    all(t.pos == "NOUN" for t in current_cluster)
                )
                is_suru_verb = (
                    token.lemma == "する" and
                    current_cluster and
                    current_cluster[-1].pos == "NOUN"
                )
                if current_cluster and not (is_compound_noun or is_suru_verb):
                    phrase = flush_cluster(current_cluster, order)
                    if phrase:
                        phrases.append(phrase)
                        order += 1
                    current_cluster = []

            current_cluster.append(token)

        # Flush any remaining tokens
        if current_cluster:
            phrase = flush_cluster(current_cluster, order)
            if phrase:
                phrases.append(phrase)

        return phrases

    def _generate_cluster_meaning(self, cluster: List[TokenModel]) -> str:
        meanings = [t.lemma for t in cluster if t.pos != "PUNCTUATION"]
        return " + ".join(meanings)
