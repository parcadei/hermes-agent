"""Three-layer memory architecture: stated, inferred, derived.

Layer 1 STATED:   grounded facts from user's own words (permanent).
Layer 2 INFERRED: semantic bridges — synonym/type links (disposable cache).
Layer 3 DERIVED:  deterministic gists computed from stated layer.

Design principles:
- The graph stores the user's words, NOT the model's interpretation.
- Grounding check: fact.object must substring-match fact.source_text.
- Inferred layer is navigation only — NEVER appears in model prompt.
- Model sees: source text + stated facts + gists.
- No LLM calls for infrastructure. Only deterministic operations.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Optional

from mabench.entity_graph import EntityGraph
from mabench.ontology import FacetEntity


# ── Default synonym dictionary (no LLM, hand-built) ─────────────

DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "airline": ["plane", "flight", "flying", "aviation"],
    "hotel": ["accommodation", "lodging", "stay", "resort"],
    "restaurant": ["dining", "food", "eating", "cafe"],
    "car": ["vehicle", "driving", "automobile"],
    "gym": ["workout", "exercise", "fitness", "training"],
    "doctor": ["medical", "health", "physician", "clinic"],
    "university": ["college", "school", "education", "campus"],
    "programming": ["coding", "software", "development", "code"],
    "music": ["song", "instrument", "band", "concert"],
    "movie": ["film", "cinema", "theater"],
    "book": ["reading", "novel", "literature"],
    "pet": ["animal", "companion"],
    "travel": ["trip", "vacation", "journey", "tour"],
}


class _EntityBM25:
    """Lightweight BM25 over entity documents (name + facet values per entity)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []       # entity canonical name per doc
        self._turn_sets: list[set[int]] = []  # mention_turns per doc
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._n: int = 0
        self._dirty: bool = True  # needs rebuild

    def add_entity(self, canonical: str, doc_text: str, turns: set[int]) -> None:
        """Add or update an entity document."""
        self._docs.append(canonical)
        self._turn_sets.append(turns)
        tokens = doc_text.lower().split()
        self._doc_lens.append(len(tokens))
        tf: dict[str, int] = defaultdict(int)
        seen: set[str] = set()
        for t in tokens:
            tf[t] += 1
            if t not in seen:
                self._df[t] += 1
                seen.add(t)
        self._tf.append(dict(tf))
        self._n += 1
        self._dirty = True

    def _finalize(self) -> None:
        if not self._dirty:
            return
        total = sum(self._doc_lens) if self._doc_lens else 0
        self._avgdl = total / max(self._n, 1)
        self._dirty = False

    def query(self, q: str, top_k: int = 20) -> list[tuple[str, set[int], float]]:
        """Search entities by BM25. Returns (canonical, turn_set, score) tuples."""
        import math
        self._finalize()
        if self._n == 0:
            return []
        tokens = q.lower().split()
        scores = [0.0] * self._n
        for t in tokens:
            if t not in self._df:
                continue
            idf = math.log((self._n - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1)
            for i in range(self._n):
                tf_val = self._tf[i].get(t, 0)
                if tf_val == 0:
                    continue
                dl = self._doc_lens[i]
                num = tf_val * (self.k1 + 1)
                den = tf_val + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * num / den
        # Return top-k with score > 0
        indexed = [(i, scores[i]) for i in range(self._n) if scores[i] > 0]
        indexed.sort(key=lambda x: -x[1])
        return [
            (self._docs[i], self._turn_sets[i], s)
            for i, s in indexed[:top_k]
        ]


class LayeredMemory:
    """Three-layer memory: stated facts, inferred bridges, derived gists."""

    def __init__(self, embed_fn: Optional[Callable[..., Any]] = None) -> None:
        self.stated = EntityGraph(embed_fn=embed_fn)
        self.inferred = EntityGraph()
        self.derived = EntityGraph()
        self._entity_bm25 = _EntityBM25()

    # ── Ingestion ────────────────────────────────────────────────

    def ingest_entity(self, entity: FacetEntity) -> list[str]:
        """Ingest a faceted entity into the stated layer.

        Uses build_graph_from_entities logic but on the existing stated graph.
        Returns list of grounding failure descriptions (empty = fully grounded).
        """
        norm = self.stated._normalize(entity.name)
        if norm and norm not in self.stated.entities:
            from mabench.entity_graph import EntityNode
            node = EntityNode(canonical=norm)
            node.aliases.add(norm)
            self.stated.entities[norm] = node
            self.stated.alias_index[norm] = norm

        for facet_name, facet_value in entity.facets.items():
            self.stated.ingest_triple(
                entity.name, facet_name, facet_value,
                turn_idx=entity.turn_idx,
                source_text=entity.source_text,
            )

        # Feed entity BM25 index: document = name + facet values
        doc_parts = [entity.name]
        for v in entity.facets.values():
            if v and v.strip():
                doc_parts.append(v.strip())
        node = self.stated.entities.get(norm)
        turns = node.mention_turns if node else {entity.turn_idx} if entity.turn_idx >= 0 else set()
        self._entity_bm25.add_entity(norm, " ".join(doc_parts), turns)

        return self._check_grounding_for_entity(entity)

    def _check_grounding_for_entity(self, entity: FacetEntity) -> list[str]:
        """Check grounding for a single entity's facets against its source_text."""
        failures = []
        if not entity.source_text:
            return failures
        src = entity.source_text.lower()
        for facet_name, facet_value in entity.facets.items():
            val = facet_value.lower().strip()
            if val in src:
                continue
            if all(w in src for w in val.split() if len(w) >= 3):
                continue
            failures.append(
                f"{entity.name}.{facet_name}='{facet_value}' not in source"
            )
        return failures

    # ── Grounding verification ───────────────────────────────────

    def grounding_rate(self) -> tuple[int, int, list[str]]:
        """Compute grounding rate over the stated layer.

        Returns (grounded_count, total_count, ungrounded_details).
        A fact is grounded if its object value appears as a substring in
        source_text (case-insensitive), with word-level fallback for reordering.
        Facts without source_text are skipped.
        """
        grounded = 0
        total = 0
        ungrounded: list[str] = []
        for canon, node in self.stated.entities.items():
            for pred, facts in node.facts.items():
                for f in facts:
                    if not f.source_text:
                        continue
                    total += 1
                    src = f.source_text.lower()
                    val = f.object.lower().strip()
                    if val in src:
                        grounded += 1
                    elif all(w in src for w in val.split() if len(w) >= 3):
                        grounded += 1
                    else:
                        ungrounded.append(
                            f"{canon}.{pred}='{f.object}' not in turn {f.turn_idx}"
                        )
        return grounded, total, ungrounded

    # ── Layer traversal ──────────────────────────────────────────

    def traverse_layers(self, query_term: str) -> list[str]:
        """Resolve a query through layers: stated -> inferred expansion -> stated.

        Returns list of canonical entity names from the STATED layer.
        Uses resolve_exact() to avoid false substring merges (e.g. "la" in "plane").
        """
        # Step 1: direct lookup in stated layer (exact only)
        direct = self.stated.resolve_exact(query_term)
        if direct is not None:
            return [direct]

        # Step 2: expand through inferred layer
        inferred_canon = self.inferred.resolve_exact(query_term)
        if inferred_canon is not None:
            node = self.inferred.entities.get(inferred_canon)
            if node is not None:
                matches = []
                for related_canon in node.related:
                    stated_canon = self.stated.resolve_exact(related_canon)
                    if stated_canon is not None:
                        matches.append(stated_canon)
                if matches:
                    return list(dict.fromkeys(matches))

        # Step 3: search inferred facts for query_term as an object value
        for canon, node in self.inferred.entities.items():
            for pred, facts in node.facts.items():
                for f in facts:
                    if query_term.lower().strip() == f.object.lower().strip():
                        # The subject entity might exist in stated
                        stated_canon = self.stated.resolve_exact(canon)
                        if stated_canon is not None:
                            return [stated_canon]
                        # Or subject might link to a stated entity
                        inferred_node = self.inferred.entities.get(
                            self.inferred.resolve_exact(canon) or "")
                        if inferred_node:
                            for r in inferred_node.related:
                                sc = self.stated.resolve_exact(r)
                                if sc is not None:
                                    return [sc]

        return []

    # ── Retrieval interface (RRF-compatible) ─────────────────────

    def retrieve_for_question(self, question_text: str, max_turns: int = 20) -> list[int]:
        """BM25 search over entity documents, return ranked evidence turn indices.

        Uses the entity BM25 index (name + facet values per entity) built
        during ingestion. Falls back to traverse_layers for synonym resolution
        when BM25 has no hits. Ablation (2026-04-13): BM25 63.4% vs exact 42.3%.
        """
        if not question_text or not question_text.strip():
            return []

        q = question_text.strip()
        # Strip dynamic prompt prefix
        if "Now Answer the Question:" in q:
            q = q.split("Now Answer the Question:", 1)[1]
            if "\n" in q:
                q = q.split("\n")[0]
            q = q.strip()

        # Lazy-build BM25 from stated entities if not populated via ingest_entity
        if self._entity_bm25._n == 0 and self.stated.entities:
            self._rebuild_bm25_from_stated()

        # Primary: BM25 search over entity documents
        hits = self._entity_bm25.query(q, top_k=max_turns * 2)
        if hits:
            turn_scores: dict[int, float] = defaultdict(float)
            for _canon, turns, score in hits:
                for t in turns:
                    turn_scores[t] += score
            ranked = sorted(turn_scores, key=lambda t: -turn_scores[t])
            return ranked[:max_turns]

        # Fallback: traverse_layers (synonym resolution via inferred layer)
        import re
        from mabench.entity_graph import _STOP_WORDS

        words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", q)
        candidates = [w for w in words if w.lower() not in _STOP_WORDS and len(w) >= 2]
        word_list = q.split()
        for n in (2, 3):
            for i in range(len(word_list) - n + 1):
                candidates.append(" ".join(word_list[i:i+n]))

        resolved_entities: set[str] = set()
        for cand in candidates:
            results = self.traverse_layers(cand)
            resolved_entities.update(results)

        if not resolved_entities:
            return []

        turn_counts: dict[int, int] = defaultdict(int)
        for canon in resolved_entities:
            node = self.stated.entities.get(canon)
            if node is not None:
                for t in node.mention_turns:
                    turn_counts[t] += 1

        ranked = sorted(turn_counts, key=lambda t: (-turn_counts[t], t))
        return ranked[:max_turns]

    def _rebuild_bm25_from_stated(self) -> None:
        """Build BM25 index from stated entities (for triple-only ingestion paths)."""
        self._entity_bm25 = _EntityBM25()
        for canon, node in self.stated.entities.items():
            doc_parts = [canon]
            for pred, facts in node.facts.items():
                for f in facts:
                    if not f.superseded and f.object.strip():
                        doc_parts.append(f.object.strip())
            self._entity_bm25.add_entity(canon, " ".join(doc_parts), node.mention_turns)

    # ── Inferred layer management ────────────────────────────────

    def build_inferred_layer(
        self,
        synonym_dict: Optional[dict[str, list[str]]] = None,
        entity_domains: Optional[dict[str, str]] = None,
    ) -> None:
        """Build Layer 2 (inferred) from stated entities + synonym dict.

        Creates:
        - Type links: entity -> what -> domain (e.g., Delta -> what -> airline)
        - Synonym links: domain -> synonym -> word (e.g., airline -> synonym -> plane)

        Args:
            synonym_dict: domain -> list of synonyms. Defaults to DEFAULT_SYNONYMS.
            entity_domains: entity_canonical -> domain. Explicit mapping of
                entities to their domain categories. When provided, these take
                priority over heuristic matching.
        """
        syns = synonym_dict if synonym_dict is not None else DEFAULT_SYNONYMS
        ent_doms = entity_domains or {}

        # Build reverse index: synonym_word -> domain
        reverse: dict[str, str] = {}
        for domain, words in syns.items():
            reverse[domain.lower()] = domain.lower()
            for w in words:
                reverse[w.lower()] = domain.lower()

        # For each stated entity, determine its domain
        for canon, node in self.stated.entities.items():
            canon_lower = canon.lower()

            # Priority 1: explicit entity_domains mapping
            matched_domain = ent_doms.get(canon_lower)
            if matched_domain is not None:
                matched_domain = matched_domain.lower()
            else:
                # Priority 2: heuristic match against synonym dict
                matched_domain = reverse.get(canon_lower)
                if matched_domain is None:
                    for word in canon_lower.split():
                        matched_domain = reverse.get(word)
                        if matched_domain:
                            break

            if matched_domain:
                # Create type link in inferred: entity -> what -> domain
                self.inferred.ingest_triple(
                    canon, "what", matched_domain, turn_idx=-1
                )
                # Create synonym links: domain -> synonym -> word
                if matched_domain in syns:
                    for syn_word in syns[matched_domain]:
                        self.inferred.ingest_triple(
                            matched_domain, "synonym", syn_word, turn_idx=-1
                        )

    def wipe_inferred(self) -> None:
        """Clear the inferred layer completely. Stated layer is untouched."""
        self.inferred = EntityGraph()

    # ── Gist extraction (from stated layer ONLY) ─────────────────

    def extract_gists(
        self,
        min_convergent_entities: int = 2,
        min_progression_steps: int = 2,
        min_shared_turns: int = 1,
    ) -> dict[str, list[dict]]:
        """Extract all gist types from the stated layer.

        Returns dict with keys: convergent, progression, cooccurrence.
        """
        return {
            "convergent": self._extract_convergent(min_convergent_entities),
            "progression": self._extract_progression(min_progression_steps),
            "cooccurrence": self._extract_cooccurrence(min_shared_turns),
        }

    def _extract_convergent(self, min_entities: int = 2) -> list[dict]:
        """Find facet values shared by multiple entities in stated layer."""
        value_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        for canon, node in self.stated.entities.items():
            for pred, facts in node.facts.items():
                for f in facts:
                    if f.superseded:
                        continue
                    key = (pred, f.object.lower().strip())
                    value_index[key].add(canon)

        gists = []
        for (facet, value), entities in value_index.items():
            if len(entities) >= min_entities:
                gists.append({
                    "type": "convergent",
                    "facet": facet,
                    "value": value,
                    "entities": sorted(entities),
                    "count": len(entities),
                })
        return sorted(gists, key=lambda g: -g["count"])

    def _extract_progression(self, min_steps: int = 2) -> list[dict]:
        """Find entities with multi-step facet progressions via deltas()."""
        gists = []
        for canon in self.stated.entities:
            d = self.stated.deltas(canon)
            if d is None:
                continue
            for pred, changes in d.items():
                if len(changes) >= min_steps:
                    gists.append({
                        "type": "progression",
                        "entity": canon,
                        "facet": pred,
                        "steps": len(changes),
                        "progression": [f.object for f in changes],
                        "span": (changes[0].turn_idx, changes[-1].turn_idx),
                    })
        return sorted(gists, key=lambda g: -g["steps"])

    def _extract_cooccurrence(self, min_shared_turns: int = 2) -> list[dict]:
        """Find entity pairs that co-occur in multiple turns."""
        entities = list(self.stated.entities.keys())
        gists = []
        for i, a in enumerate(entities):
            a_turns = self.stated.entities[a].mention_turns
            for b in entities[i + 1:]:
                b_turns = self.stated.entities[b].mention_turns
                shared = a_turns & b_turns
                if len(shared) >= min_shared_turns:
                    gists.append({
                        "type": "cooccurrence",
                        "entity_a": a,
                        "entity_b": b,
                        "shared_turns": sorted(shared),
                        "count": len(shared),
                    })
        return sorted(gists, key=lambda g: -g["count"])

    # ── Prompt formatter ─────────────────────────────────────────

    def format_gists_for_prompt(self, gists: Optional[dict] = None) -> str:
        """Format gists as human-readable text for LLM context.

        If gists is None, extracts fresh from stated layer.
        Returns empty string if no gists exist.
        """
        if gists is None:
            gists = self.extract_gists()

        lines: list[str] = []

        for g in gists.get("convergent", []):
            entities_str = " and ".join(g["entities"])
            lines.append(f"{entities_str} both {g['facet']} = {g['value']}")

        for g in gists.get("progression", []):
            steps_str = " -> ".join(g["progression"])
            lines.append(f"{g['entity']} {g['facet']}: {steps_str}")

        for g in gists.get("cooccurrence", []):
            lines.append(
                f"{g['entity_a']} and {g['entity_b']} mentioned together "
                f"in {g['count']} turn{'s' if g['count'] != 1 else ''}"
            )

        return "\n".join(lines)
