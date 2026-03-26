"""Ingestion store pipeline for HermesMemoryAgent.

Extracts _store and its decomposed helpers from HermesMemoryAgent into a
mixin. The main entry point _store orchestrates: segmentation, embedding,
triple extraction, supersession, engine storage, index updates, and
post-chunk bookkeeping (FAISS rebuild, synonym dict, entity graph,
belief propagation, dream cycle).

Each helper method has a single responsibility and stays under 50 lines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mabench.ingestion import IngestionResult, SegmentedFact


@dataclass
class _PrepareResult:
    """Intermediate state from _store_prepare.

    Bundles the segmentation result with pre-computed embeddings,
    REBEL triples, and placeholder entity arrays so the per-fact
    loop can consume them without re-computing.
    """

    result: IngestionResult
    fact_texts: list[str]
    embeddings: np.ndarray
    rebel_triples_batch: list[list[dict]] | None
    primary_entities: list[str | None]


class StoreMixin:
    """Mixin providing the _store ingestion pipeline and its helpers.

    Requires the host class to provide:
    - self._ingestion: IngestionPipeline instance
    - self._scorer: EmbeddingRelevanceScorer with embed_batch()
    - self.coupled_engine: CoupledEngine with memory_store, store(),
      replace_entry(), rebuild_faiss_if_dirty(), flush_session(), dream()
    - self.orchestrator: MemoryOrchestrator (V1 store)
    - self.cooc_boost_retrieval: bool
    - self.supersession: bool
    - self._store_count: int
    - self._triple_index: defaultdict(list)
    - self._text_to_store_idx: dict[str, int]
    - self._fact_entities: dict[str, tuple]
    - self._superseded_texts: set
    - self._supersession_graph: dict
    - self._supersedes: defaultdict(set)
    - self._supersession_log: list
    - self._gliner_entity_index: defaultdict(set)
    - self._belief_index: BeliefIndex or None
    - self._bm25_index: BM25Index or None
    - self._triadic: TriadicMemory or None
    - self._chain_cache: ChainCache or None
    - self.dream_interval: int
    - self._entity_accumulator: _MutableEntityGraph
    - self._rebel_extract_batch(): method from RebelKGMixin
    - self._rebel_kg_add(): method from RebelKGMixin
    """

    def _store(self, text: str) -> str:
        """Store text through ingestion pipeline + V1 + V2 layers.

        Orchestrates: segment -> embed -> extract triples ->
        per-fact (supersession check, store, index updates) ->
        post-chunk (FAISS rebuild, synonyms, entity graph, belief, dream).
        """
        prep = self._store_prepare(text)
        if not prep.fact_texts:
            return "Memorized"

        for fi, (fact, emb, entity) in enumerate(
            zip(prep.result.facts, prep.embeddings, prep.primary_entities)
        ):
            self._store_single_fact(fi, fact, emb, entity, prep)

        self._store_post_chunk(prep.result)
        return "Memorized"

    def _store_prepare(self, text: str) -> _PrepareResult:
        """Segment text, embed facts, extract REBEL triples.

        Returns a _PrepareResult bundling all pre-computed data
        for the per-fact loop.
        """
        result = self._ingestion.segment(text)
        fact_texts = [f.text for f in result.facts]

        rebel_triples_batch: list[list[dict]] | None = None
        run_rebel = getattr(self, "_rebel_kg", None) is not None

        # Embed first, then REBEL sequentially (shared GPU).
        embeddings = self._scorer.embed_batch(fact_texts)
        if run_rebel:
            rebel_triples_batch = self._rebel_extract_batch(fact_texts)

        # GLiNER entity extraction skipped — consumers disabled.
        primary_entities: list[str | None] = [None] * len(fact_texts)
        all_entities: list[list] = [[] for _ in fact_texts]

        # Build entity index with insertion offset.
        base_idx = len(self.coupled_engine.memory_store)
        for i, ents in enumerate(all_entities):
            fact_idx = base_idx + i
            for ent in ents:
                key = ent["text"].lower().strip()
                if len(key) > 1:
                    self._gliner_entity_index[key].add(fact_idx)

        return _PrepareResult(
            result=result,
            fact_texts=fact_texts,
            embeddings=embeddings,
            rebel_triples_batch=rebel_triples_batch,
            primary_entities=primary_entities,
        )

    def _store_single_fact(
        self,
        fi: int,
        fact: SegmentedFact,
        emb: np.ndarray,
        entity: str | None,
        prep: _PrepareResult,
    ) -> None:
        """Process one fact: supersession check, store, index, count."""
        superseded_idx, superseded_text = self._store_check_supersession(fact)

        if superseded_idx is not None:
            ms_idx = self._store_replace_entry(
                fact, emb, superseded_idx, superseded_text
            )
        else:
            ms_idx = self._store_new_entry(fact, emb, entity)

        self._store_update_indices(
            fi, fact, ms_idx, prep.rebel_triples_batch
        )
        self._store_update_triple_index(fact, superseded_text)
        self._store_detect_belief_conflicts(fact)
        self._store_count += 1

    def _store_check_supersession(
        self, fact: SegmentedFact
    ) -> tuple[int | None, str | None]:
        """Check if fact supersedes an existing entry.

        Returns (superseded_idx, superseded_text) or (None, None).
        Uses triple-based (S,P) matching with entity substring gate.
        """
        if not self.supersession or not fact.triples:
            return None, None

        new_entities = (
            set(e.lower() for e in fact.entities) if fact.entities else set()
        )
        self._fact_entities[fact.text] = fact.entities

        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            for existing_obj, existing_text in self._triple_index[key]:
                if existing_obj == new_obj:
                    continue
                # Entity gate: substring matching.
                old_ents = [
                    e.lower()
                    for e in self._fact_entities.get(existing_text, ())
                ]
                existing_lower = existing_text.lower()
                fact_lower = fact.text.lower()
                shared = any(
                    e in existing_lower for e in new_entities if len(e) > 2
                ) or any(
                    e in fact_lower for e in old_ents if len(e) > 2
                )
                if not shared:
                    continue
                old_idx = self._text_to_store_idx.get(existing_text)
                if old_idx is not None:
                    return old_idx, existing_text
        return None, None

    def _store_replace_entry(
        self,
        fact: SegmentedFact,
        emb: np.ndarray,
        superseded_idx: int,
        superseded_text: str,
    ) -> int:
        """Replace a superseded entry in-place.

        Logs the supersession, replaces in coupled engine, updates
        reverse lookup, supersession graph, chain cache, and belief.
        Returns the store index.
        """
        old_entry = self.coupled_engine.memory_store[superseded_idx]
        self._supersession_log.append({
            "old_text": old_entry.text,
            "new_text": fact.text,
            "old_recency": old_entry.recency,
            "new_recency": float(self._store_count),
            "timestamp": time.time(),
        })
        self.coupled_engine.replace_entry(
            superseded_idx, fact.text, emb, recency=float(self._store_count)
        )
        ms_idx = superseded_idx
        # Update reverse lookup.
        del self._text_to_store_idx[superseded_text]
        self._text_to_store_idx[fact.text] = ms_idx
        # Track superseded text for query-time safety net.
        self._superseded_texts.add(superseded_text)
        # Supersession graph for query-time CR.
        self._supersession_graph[superseded_text] = fact.text
        self._supersedes[fact.text].add(superseded_text)
        # Chain cache invalidation.
        if getattr(self, "_chain_cache", None) is not None:
            self._chain_cache.invalidate_fact(superseded_text)
        # Bayesian belief update.
        if self._belief_index is not None:
            self._belief_index.on_conflict(superseded_text, fact.text)
        return ms_idx

    def _store_new_entry(
        self,
        fact: SegmentedFact,
        emb: np.ndarray,
        entity: str | None,
    ) -> int:
        """Store a new fact (no supersession).

        Stores via V1 orchestrator (when cooc_boost inactive) and
        V2 coupled engine. Returns the store index.
        """
        if not self.cooc_boost_retrieval:
            self.orchestrator.store(content=fact.text, embedding=emb)
        ms_idx = self.coupled_engine.store(
            text=fact.text,
            embedding=emb,
            recency=float(self._store_count),
            entity=entity,
        )
        self._text_to_store_idx[fact.text] = ms_idx
        return ms_idx

    def _store_update_indices(
        self,
        fi: int,
        fact: SegmentedFact,
        ms_idx: int,
        rebel_triples_batch: list[list[dict]] | None,
    ) -> None:
        """Update BM25, triadic, REBEL KG, and belief indices."""
        # BM25.
        if getattr(self, "_bm25_index", None) is not None:
            self._bm25_index.add(ms_idx, fact.text)
        # Triadic.
        if getattr(self, "_triadic", None) is not None:
            self._triadic.store_fact(fact.text, list(fact.triples))
        # REBEL KG.
        if rebel_triples_batch is not None and fi < len(rebel_triples_batch):
            for triple in rebel_triples_batch[fi]:
                self._rebel_kg_add(
                    triple["head"], triple["relation"], triple["tail"]
                )
        # Belief index.
        if self._belief_index is not None:
            self._belief_index.register_fact(fact.text)

    def _store_update_triple_index(
        self,
        fact: SegmentedFact,
        superseded_text: str | None,
    ) -> None:
        """Update the (subject, predicate) -> [(object, text)] index.

        Removes superseded entries and appends the new fact's triples.
        """
        if not fact.triples:
            return
        if fact.text not in self._fact_entities:
            self._fact_entities[fact.text] = fact.entities
        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            if superseded_text is not None:
                self._triple_index[key] = [
                    (o, t)
                    for o, t in self._triple_index[key]
                    if t != superseded_text
                ]
            self._triple_index[key].append((new_obj, fact.text))

    def _store_detect_belief_conflicts(self, fact: SegmentedFact) -> None:
        """Detect conflicts for belief index when supersession is off.

        When supersession is disabled but belief is active, scan the
        triple index for (S,P) matches with different objects and
        fire belief.on_conflict for each pair.
        """
        if self.supersession or not self._belief_index or not fact.triples:
            return

        new_entities = (
            set(e.lower() for e in fact.entities) if fact.entities else set()
        )
        if fact.text not in self._fact_entities:
            self._fact_entities[fact.text] = fact.entities

        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            for existing_obj, existing_text in self._triple_index[key]:
                if existing_obj == new_obj:
                    continue
                old_ents = [
                    e.lower()
                    for e in self._fact_entities.get(existing_text, ())
                ]
                existing_lower = existing_text.lower()
                fact_lower = fact.text.lower()
                shared = any(
                    e in existing_lower for e in new_entities if len(e) > 2
                ) or any(
                    e in fact_lower for e in old_ents if len(e) > 2
                )
                if shared:
                    self._belief_index.on_conflict(existing_text, fact.text)

    def _store_post_chunk(self, result: IngestionResult) -> None:
        """Post-chunk operations after all facts are stored.

        Handles FAISS rebuild, triadic synonym dict, entity graph
        accumulation, belief propagation, session flush, supersession
        diagnostics, and dream cycle.
        """
        # Deferred FAISS rebuild: one per chunk.
        self.coupled_engine.rebuild_faiss_if_dirty()

        # Synonym dict for triadic.
        if getattr(self, "_triadic", None) is not None:
            if getattr(result, "coref_clusters", None):
                self._triadic.add_coref_clusters(result.coref_clusters)
            if getattr(result, "verb_mappings", None):
                self._triadic.add_verb_mappings(result.verb_mappings)

        # Entity graph accumulation.
        if hasattr(self, "_entity_accumulator"):
            offset = self._store_count - len(result.facts)
            self._entity_accumulator.merge(result.entity_graph, offset)

        # Belief propagation.
        if self._belief_index is not None and hasattr(self, "_entity_accumulator"):
            self._belief_index.propagate(
                self._entity_accumulator,
                self.coupled_engine.memory_store,
            )

        # Flush Hebbian session buffer.
        if (
            hasattr(self.coupled_engine, "_session_buffer")
            and len(getattr(self.coupled_engine, "_session_buffer", [])) > 0
        ):
            self.coupled_engine.flush_session()

        # Supersession diagnostics: log every 50 stores.
        if (
            self.supersession
            and self._store_count % 50 == 0
            and self._store_count > 0
        ):
            logging.getLogger("supersession").info(
                "STATS after %d stores: %d unique (S,P) keys, "
                "%d superseded texts, %d graph edges, %d total triples",
                self._store_count,
                len(self._triple_index),
                len(self._superseded_texts),
                len(self._supersession_graph),
                sum(len(v) for v in self._triple_index.values()),
            )

        # Dream cycle.
        if self.dream_interval > 0 and self._store_count % self.dream_interval == 0:
            self.coupled_engine.dream()
