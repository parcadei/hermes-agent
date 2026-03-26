"""REBEL knowledge graph extraction and traversal.

Provides RebelKGMixin with methods for:
- Lazy REBEL model loading (Babelscape/rebel-large, 350M params)
- REBEL decoder output parsing into (head, relation, tail) triples
- Batch triple extraction from text
- Forward + reverse KG triple insertion
- Query-time graph hop for decompose sub-queries
"""

from __future__ import annotations

import logging


_cr_log = logging.getLogger("conflict_res")


class RebelKGMixin:
    """Mixin providing REBEL-based KG extraction and graph hop methods.

    Requires the host class to provide:
    - self._rebel_kg: dict[str, dict[str, list[str]]] or None
    - self._rebel_kg_reverse: dict[str, dict[str, list[str]]] or None
    - self._rebel_model: model instance, None, or False sentinel
    - self._rebel_tokenizer: tokenizer instance or None
    - self._rebel_device: str or None
    """

    def _ensure_rebel(self):
        """Lazy-load REBEL model (Babelscape/rebel-large, 350M params)."""
        if getattr(self, "_rebel_model", None) is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._rebel_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rebel_tokenizer = AutoTokenizer.from_pretrained(
                "Babelscape/rebel-large"
            )
            self._rebel_model = AutoModelForSeq2SeqLM.from_pretrained(
                "Babelscape/rebel-large",
                device_map={"": self._rebel_device},
            )
            self._rebel_model.eval()
            logging.getLogger(__name__).info(
                "REBEL loaded (rebel-large) on %s", self._rebel_device
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "REBEL load failed: %s — KG extraction disabled", e
            )
            self._rebel_model = False  # sentinel: tried and failed

    @staticmethod
    def _parse_rebel_output(decoded: str) -> list[dict]:
        """Parse REBEL decoder output into (head, relation, tail) dicts.

        REBEL encodes: <triplet> subject <subj> relation <obj> object
        """
        triplets = []
        for part in decoded.split("<triplet>"):
            part = part.strip()
            if not part or part in ("<s>", "</s>"):
                continue
            part = part.replace("</s>", "").replace("<pad>", "").strip()
            if "<subj>" in part and "<obj>" in part:
                subj_split = part.split("<subj>")
                head = subj_split[0].strip()
                rest = subj_split[1] if len(subj_split) > 1 else ""
                obj_split = rest.split("<obj>")
                relation = obj_split[0].strip()
                tail = obj_split[1].strip() if len(obj_split) > 1 else ""
                if head and relation and tail:
                    triplets.append(
                        {"head": head, "relation": relation, "tail": tail}
                    )
        return triplets

    def _rebel_extract_batch(self, texts: list[str]) -> list[list[dict]]:
        """Extract triples from a batch of texts using REBEL.

        Returns a list of triple-lists, one per input text.
        """
        self._ensure_rebel()
        if not self._rebel_model:
            return [[] for _ in texts]

        import torch

        all_triples: list[list[dict]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._rebel_tokenizer(
                batch,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self._rebel_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._rebel_model.generate(
                    **inputs, max_length=128, num_beams=1
                )
            decoded = self._rebel_tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )
            for dec in decoded:
                all_triples.append(self._parse_rebel_output(dec))

        return all_triples

    def _rebel_kg_add(self, head: str, relation: str, tail: str) -> None:
        """Add a triple to the REBEL KG (forward + reverse index)."""
        h = head.lower().strip()
        r = relation.lower().strip()
        t_key = tail.lower().strip()
        if not h or not r or not t_key:
            return

        # Forward: subject -> relation -> [objects]
        if h not in self._rebel_kg:
            self._rebel_kg[h] = {}
        if r not in self._rebel_kg[h]:
            self._rebel_kg[h][r] = []
        if tail.strip() not in self._rebel_kg[h][r]:
            self._rebel_kg[h][r].append(tail.strip())

        # Reverse: object -> relation -> [subjects]
        if t_key not in self._rebel_kg_reverse:
            self._rebel_kg_reverse[t_key] = {}
        if r not in self._rebel_kg_reverse[t_key]:
            self._rebel_kg_reverse[t_key][r] = []
        if head.strip() not in self._rebel_kg_reverse[t_key][r]:
            self._rebel_kg_reverse[t_key][r].append(head.strip())

    # Relation keyword matching for KG query-time hop
    _KG_STOP_WORDS = frozenset({
        "what", "is", "the", "of", "a", "an", "did", "does", "was", "were",
        "where", "who", "which", "how", "do", "has", "had", "that", "this",
        "in", "on", "at", "to", "for", "by", "with", "from", "are", "be",
    })
    _KG_STEM_MAP = {
        "born": "birth", "die": "death", "died": "death",
        "married": "spouse", "marry": "spouse",
        "play": "sport", "plays": "sport", "played": "sport",
        "follow": "religion", "follows": "religion", "followed": "religion",
        "work": "field", "works": "field", "worked": "field",
        "educated": "educated", "studied": "educated",
        "created": "origin", "compose": "genre", "composed": "genre",
        "perform": "genre", "performed": "genre",
        "hold": "position", "held": "position",
        "capital": "capital",
    }

    def _rebel_kg_hop(self, sub_query: str) -> str | None:
        """Try to answer a sub-query via REBEL KG graph traversal.

        1. Find KG entities that appear in the sub-query (longest first).
        2. For each, find relations whose keywords overlap the sub-query.
        3. Return the object/subject if a clear match is found.
        4. Return None to fall through to retrieval+LLM.
        """
        if not self._rebel_kg:
            return None

        sq_lower = sub_query.lower()
        candidates = self._kg_hop_find_candidates(sq_lower)
        if not candidates:
            return None

        # Prefer longest entity match (most specific)
        candidates.sort(key=lambda c: len(c[0]), reverse=True)

        q_tokens = self._kg_hop_tokenize_query(sq_lower)

        return self._kg_hop_match_relation(sq_lower, candidates, q_tokens)

    def _kg_hop_find_candidates(
        self, sq_lower: str,
    ) -> list[tuple[str, dict, bool]]:
        """Find KG entities that appear in the sub-query text.

        Returns (entity_key, relations_dict, is_reverse) tuples.
        """
        candidates: list[tuple[str, dict, bool]] = []
        for entity_key, rels in self._rebel_kg.items():
            if entity_key in sq_lower:
                candidates.append((entity_key, rels, False))
        for entity_key, rels in self._rebel_kg_reverse.items():
            if entity_key in sq_lower:
                candidates.append((entity_key, rels, True))
        return candidates

    def _kg_hop_tokenize_query(self, sq_lower: str) -> set[str]:
        """Tokenize query, remove stop words, and add stemmed variants."""
        q_tokens: set[str] = set()
        for w in sq_lower.split():
            w = w.strip(".,?!;:'\"()[]")
            if w and w not in self._KG_STOP_WORDS:
                q_tokens.add(w)
        # Add stemmed variants
        stemmed = set(q_tokens)
        for t in q_tokens:
            if t in self._KG_STEM_MAP:
                stemmed.add(self._KG_STEM_MAP[t])
        return stemmed

    def _kg_hop_match_relation(
        self,
        sq_lower: str,
        candidates: list[tuple[str, dict, bool]],
        stemmed: set[str],
    ) -> str | None:
        """Match candidate entities against relations using token overlap."""
        for entity_key, relations, is_reverse in candidates:
            # Remove entity from query tokens to avoid self-matching
            entity_tokens = set(entity_key.split())
            q_clean = stemmed - entity_tokens

            best_rel = None
            best_overlap = 0
            for rel, values in relations.items():
                rel_tokens = set(rel.replace("_", " ").split()) - self._KG_STOP_WORDS
                if not rel_tokens:
                    continue
                overlap = len(rel_tokens & q_clean)
                # Also check if full relation is substring of query
                if overlap == 0 and rel in sq_lower:
                    overlap = len(rel_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rel = rel

            if best_rel is not None and best_overlap > 0:
                values = relations[best_rel]
                # Return last added (most recent = temporal supersession)
                answer = values[-1]
                _cr_log.info(
                    "KG hop: entity=%s rel=%s -> %s (reverse=%s)",
                    entity_key, best_rel, answer, is_reverse,
                )
                return answer

        return None
