"""Build Bunsetsu records and dependency relations from stored GiNZA tokens."""
from typing import Any, Dict, List, Optional


FUNC_DEPS = {"case", "mark", "aux", "cop", "fixed"}


class BunsetsuGraphService:
    """Pure transformation from GiNZA token JSON to Bunsetsu dependency graph."""

    def build(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not tokens:
            return {"bunsetsu": [], "relations": [], "roots": []}

        groups = self._group_tokens(tokens)
        token_to_bunsetsu = {
            token["id"]: index
            for index, group in enumerate(groups, start=1)
            for token in group
        }
        by_id: Dict[int, Dict[str, Any]] = {}

        for index, group in enumerate(groups, start=1):
            group_ids = {token["id"] for token in group}
            external = [
                token for token in group
                if token.get("head_absolute", 0) == 0
                or token.get("head_absolute") not in group_ids
            ]
            external.sort(key=lambda token: (self._role(token) != "stem", token["id"]))
            head = external[0] if external else group[0]
            stem = [token for token in group if self._role(token) == "stem"] or [head]
            func = [
                token for token in group
                if self._role(token) == "func" and token not in stem
            ]
            punct = [token for token in group if self._role(token) == "punct"]

            head_target = head.get("head_absolute", 0)
            is_root = str(head.get("dep", "")).upper() == "ROOT"
            dep_to = 0 if is_root else token_to_bunsetsu.get(head_target, 0)
            record = {
                "id": index,
                "text": "".join(token.get("orth", "") for token in group),
                "token_ids": [token["id"] for token in group],
                "tokens": group,
                "head": {
                    "token_id": head["id"],
                    "orth": head.get("orth", ""),
                    "lemma": head.get("lemma", ""),
                    "pos": head.get("pos", ""),
                    "dep": head.get("dep", ""),
                },
                "stem": {
                    "text": "".join(token.get("orth", "") for token in stem),
                    "lemma": "".join(token.get("orth", "") for token in stem[:-1])
                    + stem[-1].get("lemma", ""),
                    "token_ids": [token["id"] for token in stem],
                },
                "func": {
                    "text": "".join(token.get("orth", "") for token in func),
                    "lemmas": [token.get("lemma", "") for token in func],
                    "token_ids": [token["id"] for token in func],
                },
                "punct": [token["id"] for token in punct],
                "particle": self._particle_of(head, func),
                "dep": {
                    "to": dep_to,
                    "rel": head.get("dep", ""),
                    "to_token_id": head_target or None,
                },
                "clause_head": head.get("clause_head"),
                "children": [],
            }
            by_id[index] = record

        for record in by_id.values():
            dep_to = record["dep"]["to"]
            if dep_to and dep_to in by_id:
                by_id[dep_to]["children"].append(record["id"])

        for record in by_id.values():
            record["children"].sort()
            record["case_frame"] = [
                {
                    "particle": by_id[child]["particle"],
                    "bunsetsu": child,
                    "lemma": by_id[child]["stem"]["lemma"],
                    "rel": by_id[child]["dep"]["rel"],
                }
                for child in record["children"]
                if by_id[child]["particle"]
            ]

        relations = []
        for record in by_id.values():
            dep_to = record["dep"]["to"]
            if dep_to:
                relations.append({
                    "from_bunsetsu": record["id"],
                    "to_bunsetsu": dep_to,
                    "relation": record["dep"]["rel"],
                    "head_token_id": record["head"]["token_id"],
                    "head_token": record["head"]["orth"],
                    "particle": record["particle"],
                })

        return {
            "bunsetsu": list(by_id.values()),
            "relations": relations,
            "roots": [record["id"] for record in by_id.values() if record["dep"]["to"] == 0],
        }

    @staticmethod
    def _group_tokens(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        for token in tokens:
            if token.get("bunsetu_bi_label") == "B" or not groups:
                groups.append([])
            groups[-1].append(token)
        return groups

    @staticmethod
    def _role(token: Dict[str, Any]) -> str:
        if token.get("pos") == "PUNCT":
            return "punct"
        if token.get("bunsetu_position_type") == "SYN_HEAD" or token.get("dep") in FUNC_DEPS:
            return "func"
        return "stem"

    @staticmethod
    def _particle_of(head: Dict[str, Any], func: List[Dict[str, Any]]) -> Optional[str]:
        if head.get("pos") in {"VERB", "AUX"}:
            return None
        adp = [token.get("lemma") for token in func if token.get("pos") in {"ADP", "SCONJ"}]
        if adp:
            return adp[-1]
        for token in func:
            if token.get("pos") == "AUX" and token.get("lemma") == "だ" and token.get("orth") == "に":
                return "に"
        return None
