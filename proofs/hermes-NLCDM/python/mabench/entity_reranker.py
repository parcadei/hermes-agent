"""Entity-position residual reranking module (arXiv:2603.19664).

Standalone module that hooks into the Hermes pipeline via callbacks:
  - At ingestion: FactStore calls annotate_entry() after creating each MemoryEntry
  - At retrieval: Retriever calls rerank() after post-retrieval processing

Architecture: multiplicative gating on isolated entity residuals.

  entity_cos > 0.95   → gate = 1.0  (correct entity, no penalty)
  entity_cos 0.80-0.95 → gate = 0.5  (same category, 2× penalty)
  entity_cos < 0.80   → gate = 0.2  (cross-domain, 5× penalty)
  final_score = cosine_score × gate

Why isolated (not in-context): in-context entity residuals suffer from
template context bleed — "The capital of [France]" ≈ "The capital of
[Germany]" at entity position (cos=0.992, gap=0.006). Isolated entity
residuals (entity tokenized alone) achieve 0.134 gap.

Usage::

    reranker = EntityReranker(scorer, nlp=spacy_model)

    # At ingestion (called by FactStore):
    reranker.annotate_entry(memory_entry, segmented_fact)

    # At retrieval (called by Retriever):
    results = reranker.rerank(query_text, results, engine)
"""

from __future__ import annotations

import numpy as np


class EntityReranker:
    """Standalone entity-position residual reranker.

    Constructed once by HermesAgent and passed to FactStore + Retriever.
    Revert = don't construct it; baseline code stays untouched.
    """

    # Gate thresholds (calibrated from discrimination test v3)
    GATE_HIGH = 0.95       # above → entity match
    GATE_LOW = 0.80        # below → wrong entity
    GATE_MATCH = 1.0       # gate value for matching entity
    GATE_AMBIGUOUS = 0.5   # gate value for same-category different entity
    GATE_MISMATCH = 0.2    # gate value for wrong entity

    def __init__(self, scorer, nlp=None) -> None:
        self._scorer = scorer
        self._nlp = nlp
        # entity_text -> isolated residual (many facts share the same entity)
        self._entity_cache: dict[str, np.ndarray] = {}
        # fact_text -> entity residual (for O(1) lookup at rerank time)
        self._text_to_residual: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Ingestion: annotate MemoryEntries with entity residuals
    # ------------------------------------------------------------------

    def annotate_entry(self, entry, fact) -> None:
        """Compute and store isolated entity residual on a MemoryEntry.

        Called by FactStore._store_single_fact after each entry is created.
        """
        entity = self._extract_entity_from_fact(fact)
        if entity is None:
            return
        residual = self._get_entity_residual(entity)
        entry.entity_residual = residual
        self._text_to_residual[entry.text] = residual

    # ------------------------------------------------------------------
    # Retrieval: multiplicative gating rerank
    # ------------------------------------------------------------------

    def rerank(
        self, query_text: str, results: list[dict], engine,
    ) -> list[dict]:
        """Rerank candidates using multiplicative entity gate.

        Called by Retriever._retrieve_v2 after post-retrieval processing.
        Facts without stored entity_residual keep their original score.
        """
        query_entity = self._extract_query_entity(query_text)
        if query_entity is None:
            return results

        q_residual = self._get_entity_residual(query_entity)

        reranked = []
        for r in results:
            text = r.get("text", "")
            fact_residual = self._lookup_residual(text, engine)

            if fact_residual is not None:
                entity_cos = float(np.dot(q_residual, fact_residual))
                gate = self._gate(entity_cos)
            else:
                entity_cos = None
                gate = 1.0

            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * gate
            adjusted["entity_gate"] = gate
            adjusted["entity_cos"] = entity_cos
            reranked.append(adjusted)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entity_from_fact(self, fact) -> str | None:
        """Extract primary entity from a SegmentedFact.

        Tries coref entities first, then triple subjects, then spaCy NER.
        """
        if getattr(fact, "entities", ()):
            return fact.entities[0]
        if getattr(fact, "triples", ()):
            return fact.triples[0][0]
        return self._spacy_extract(getattr(fact, "text", ""))

    def _extract_query_entity(self, query_text: str) -> str | None:
        """Extract entity from a query string using spaCy NER."""
        return self._spacy_extract(query_text)

    def _spacy_extract(self, text: str) -> str | None:
        """Run spaCy NER and return the first entity."""
        if self._nlp is None or not text:
            return None
        doc = self._nlp(text)
        if not doc.ents:
            return None
        persons = [e for e in doc.ents if e.label_ == "PERSON"]
        if persons:
            return persons[0].text
        return doc.ents[0].text

    def _get_entity_residual(self, entity_text: str) -> np.ndarray:
        """Get isolated entity residual, cached by entity string."""
        if entity_text in self._entity_cache:
            return self._entity_cache[entity_text]
        residual = self._scorer.embed_entity_isolated(entity_text)
        self._entity_cache[entity_text] = residual
        return residual

    def _lookup_residual(self, text: str, engine) -> np.ndarray | None:
        """Look up entity residual for a fact text.

        Fast path: _text_to_residual dict (populated during ingestion).
        Slow path: scan engine.memory_store (for cache-loaded states).
        """
        residual = self._text_to_residual.get(text)
        if residual is not None:
            return residual
        for entry in engine.memory_store:
            if entry.text == text and entry.entity_residual is not None:
                self._text_to_residual[text] = entry.entity_residual
                return entry.entity_residual
        return None

    @classmethod
    def _gate(cls, entity_cos: float) -> float:
        """Multiplicative gate: penalizes wrong-entity facts."""
        if entity_cos > cls.GATE_HIGH:
            return cls.GATE_MATCH
        elif entity_cos > cls.GATE_LOW:
            return cls.GATE_AMBIGUOUS
        else:
            return cls.GATE_MISMATCH
