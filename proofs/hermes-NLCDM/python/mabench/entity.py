"""Entity extraction and retrieval operations (GLiNER-based).

Provides EntityMixin with methods for:
- Lazy GLiNER model loading
- Batch and single-text entity extraction
- Query entity extraction (with regex fallback)
- Entity-based text search via inverted index
- Entity-first retrieval (narrow before cosine ranking)
- Entity index rebuild from existing memory store
- Entity graph expansion for multi-hop bridging
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np


class EntityMixin:
    """Mixin providing entity extraction, indexing, and retrieval methods.

    Requires the host class to provide:
    - self._gliner_model: GLiNER model instance or None/False sentinel
    - self._gliner_labels: list of entity type labels
    - self._gliner_entity_index: dict[str, set[int]] inverted index
    - self.coupled_engine: engine with .memory_store attribute
    - self.retrieve_num: int, max results to return
    - self._embed_query(text): method returning query embedding vector
    """

    def _ensure_gliner(self):
        """Lazy-load GLiNER entity extractor."""
        if getattr(self, "_gliner_model", None) is not None:
            return
        try:
            from gliner import GLiNER
            self._gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
            logging.getLogger(__name__).info("GLiNER loaded (gliner_multi-v2.1)")
        except ImportError:
            logging.getLogger(__name__).warning(
                "GLiNER not installed — entity extraction disabled. "
                "Install with: pip install gliner"
            )
            self._gliner_model = False  # sentinel: tried and failed

    def _extract_entities_batch(
        self, texts: list[str]
    ) -> tuple[list[str | None], list[list[dict]]]:
        """Extract entities from texts using GLiNER.

        Returns:
            primary_entities: one primary entity per text (person-preferred),
                or None if no entity found.
            all_entities: list of entity dicts per text, each dict has
                keys 'text', 'label', 'score'.
        """
        self._ensure_gliner()
        if not getattr(self, "_gliner_model", None):
            return [None] * len(texts), [[] for _ in texts]

        labels = getattr(self, "_gliner_labels",
                         ["person", "organization", "location", "country", "city"])
        results = self._gliner_model.batch_predict_entities(
            texts, labels, threshold=0.4,
        )
        primary: list[str | None] = []
        for ents in results:
            if not ents:
                primary.append(None)
                continue
            persons = [e for e in ents if e["label"] == "person"]
            if persons:
                primary.append(persons[0]["text"])
            else:
                best = max(ents, key=lambda e: e["score"])
                primary.append(best["text"])
        return primary, results

    def _extract_primary_entity(self, texts: list[str]) -> list[str | None]:
        """Extract the primary subject entity from each text using GLiNER."""
        primary, _ = self._extract_entities_batch(texts)
        return primary

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from a query string using GLiNER.

        Returns a list of entity text strings found in the query.
        Falls back to regex proper-noun extraction if GLiNER is unavailable.
        """
        self._ensure_gliner()
        if getattr(self, "_gliner_model", None) and self._gliner_model is not False:
            labels = self._gliner_labels
            results = self._gliner_model.predict_entities(
                query, labels, threshold=0.3,  # lower threshold for short queries
            )
            return [e["text"] for e in results if len(e["text"].strip()) > 1]

        # Fallback: regex proper nouns (less precise but works without GLiNER)
        entities: list[str] = []
        _stop = {"what", "where", "which", "when", "who", "how", "that", "this",
                 "with", "from", "about", "based", "their", "does", "have", "been",
                 "associated", "knowledge", "pool", "provided", "the", "question"}
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query):
            word = m.group(1)
            if word.lower() not in _stop:
                entities.append(word)
        return entities

    def _entity_text_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Find facts by entity name match using GLiNER entity index.

        Extracts entities from the query using GLiNER, looks them up in
        the inverted entity index built during ingestion, and returns
        matching facts ranked by number of matching entities.

        Falls back to regex proper-noun extraction if GLiNER is unavailable.
        """
        # Extract query entities via GLiNER
        query_entities = self._extract_query_entities(query)

        if not query_entities:
            return []

        # Look up each query entity in the inverted index
        store = self.coupled_engine.memory_store
        fact_scores: dict[int, int] = {}  # fact_idx -> match count
        for ent_text in query_entities:
            key = ent_text.lower().strip()
            matching_indices = self._gliner_entity_index.get(key, set())
            for idx in matching_indices:
                if idx < len(store):
                    fact_scores[idx] = fact_scores.get(idx, 0) + 1

        if not fact_scores:
            return []

        # Sort by match count desc, take top_k
        ranked = sorted(fact_scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, match_count in ranked:
            entry = store[idx]
            results.append({
                "text": entry.text,
                "score": float(match_count),
                "recency": getattr(entry, "recency", 0.0),
                "source": "entity_index",
            })
        return results

    def _entity_first_retrieve(self, query: str) -> list[dict] | None:
        """Entity-first retrieval: narrow to an entity's facts before ranking.

        Extracts entities from the query, looks them up in the GLiNER
        entity index, and returns ONLY that entity's facts ranked by
        cosine similarity to the query.

        Returns None if no entity is found (caller should fall back to
        BM25/cosine). Returns a list of result dicts if entity matched.

        This eliminates template confusion: instead of ranking 17K facts
        where 500 share the same template (mean pairwise cos=0.976),
        we rank only the 3-5 facts belonging to the target entity.
        """
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            return None

        store = self.coupled_engine.memory_store
        # Union of fact indices across all extracted entities
        matched_indices: set[int] = set()
        for ent_text in query_entities:
            key = ent_text.lower().strip()
            idx_set = self._gliner_entity_index.get(key, set())
            matched_indices.update(idx_set)

        if not matched_indices:
            return None

        # Cosine-rank within the entity's fact subset
        q_emb = self._embed_query(query)
        scored: list[tuple[float, int]] = []
        for idx in matched_indices:
            if idx >= len(store):
                continue
            entry = store[idx]
            emb = getattr(entry, "embedding", None)
            if emb is not None:
                sim = float(np.dot(q_emb, emb) / (
                    np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9
                ))
            else:
                sim = 0.0
            scored.append((sim, idx))

        # Sort by cosine similarity descending
        scored.sort(key=lambda x: -x[0])

        results = []
        for sim, idx in scored[: self.retrieve_num]:
            entry = store[idx]
            results.append({
                "text": entry.text,
                "score": sim,
                "recency": getattr(entry, "recency", 0.0),
                "source": "entity_first",
            })
        return results

    def rebuild_entity_index(self, batch_size: int = 64) -> int:
        """Rebuild GLiNER entity index from existing memory_store.

        Use after loading a cache that was built without the entity index.
        Runs GLiNER batch extraction on all facts (~24 min for 17K facts on GPU).
        Returns the number of unique entities indexed.
        """
        store = self.coupled_engine.memory_store
        n = len(store)
        if n == 0:
            return 0

        self._ensure_gliner()
        if not getattr(self, "_gliner_model", None):
            print("GLiNER not available — cannot rebuild entity index")
            return 0

        self._gliner_entity_index = defaultdict(set)
        labels = self._gliner_labels

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            texts = [store[i].text for i in range(start, end)]
            results = self._gliner_model.batch_predict_entities(
                texts, labels, threshold=0.4,
            )
            for i, ents in enumerate(results):
                fact_idx = start + i
                for ent in ents:
                    key = ent["text"].lower().strip()
                    if len(key) > 1:
                        self._gliner_entity_index[key].add(fact_idx)
            if (end % 1000) < batch_size:
                print(f"  Entity index: {end}/{n} facts processed, "
                      f"{len(self._gliner_entity_index)} unique entities")

        return len(self._gliner_entity_index)

    def _entity_graph_expand(
        self, seed_facts: list[str], top_k: int = 5
    ) -> list[dict]:
        """Expand seed facts via entity co-mention graph.

        For each seed fact, find its entities in the index, then find OTHER
        facts that share those entities. This enables multi-hop bridging:
        "X plays basketball" + "basketball created in USA" -> connects X to USA.

        Returns facts NOT already in seed_facts, ranked by number of
        entity connections to seed facts.
        """
        if not self._gliner_entity_index:
            return []

        store = self.coupled_engine.memory_store
        seed_set = set(seed_facts)

        # Find all entities mentioned in seed facts
        seed_entities: set[str] = set()
        seed_indices: set[int] = set()
        for fact_text in seed_facts:
            for ent_key, fact_indices in self._gliner_entity_index.items():
                for idx in fact_indices:
                    if idx < len(store) and store[idx].text == fact_text:
                        seed_entities.add(ent_key)
                        seed_indices.add(idx)
                        break

        if not seed_entities:
            return []

        # Find facts that share entities with seed facts (but aren't seeds)
        neighbor_scores: dict[int, int] = {}
        for ent_key in seed_entities:
            for idx in self._gliner_entity_index.get(ent_key, set()):
                if idx not in seed_indices and idx < len(store):
                    neighbor_scores[idx] = neighbor_scores.get(idx, 0) + 1

        if not neighbor_scores:
            return []

        ranked = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            entry = store[idx]
            if entry.text not in seed_set:
                results.append({
                    "text": entry.text,
                    "score": float(score),
                    "recency": getattr(entry, "recency", 0.0),
                    "source": "entity_graph",
                })
        return results
