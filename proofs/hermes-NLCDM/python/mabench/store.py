"""Ingestion store pipeline — standalone FactStore.

_resolve_* properties support both standalone (explicit __init__) and
mixin mode (attributes set by HermesMemoryAgent.__init__).
StoreMixin is a backward-compat alias for FactStore.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
    entity_residuals: list[np.ndarray | None] | None = None


class FactStore:
    """Standalone ingestion store pipeline.

    Orchestrates: segment -> embed -> extract triples -> per-fact store ->
    post-chunk ops (FAISS rebuild, synonym dict, entity graph, belief, dream).
    When used as a mixin (no __init__ call), _resolve_* properties fall
    back to reading the old self.* attributes set by the host class.
    """

    def __init__(
        self, engine, scorer, ingestion, orchestrator, *,
        supersession_state=None, entity_index=None, rebel=None,
        bm25_index=None, triadic=None, chain_cache=None,
        belief_index=None, entity_accumulator=None,
        supersession: bool = False, cooc_boost: bool = False,
        dream_interval: int = 0, entity_rerank: bool = False,
    ) -> None:
        self._engine = engine
        self._scorer = scorer
        self._ingestion = ingestion
        self._orchestrator = orchestrator
        self._supersession_state = supersession_state
        self._entity_idx = entity_index
        self._rebel = rebel
        self._bm25_index = bm25_index
        self._triadic = triadic
        self._chain_cache = chain_cache
        self._belief_index = belief_index
        self._entity_accumulator = entity_accumulator
        self._supersession = supersession
        self._cooc_boost = cooc_boost
        self._dream_interval = dream_interval
        self._entity_rerank = entity_rerank
        self._store_count = 0

    # -- Attribute resolution: standalone vs mixin fallback ---------------

    def _ss(self, attr: str):
        """Read a SupersessionState field, falling back to host class attr."""
        ss = getattr(self, "_supersession_state", None)
        if ss is not None:
            return getattr(ss, attr)
        return getattr(self, f"_{attr}")  # e.g. self._triple_index

    @property
    def _resolve_engine(self):
        if hasattr(self, "_engine"):
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_orchestrator(self):
        if hasattr(self, "_orchestrator"):
            return self._orchestrator
        return self.orchestrator  # type: ignore[attr-defined]

    @property
    def _resolve_supersession(self) -> bool:
        if hasattr(self, "_supersession"):
            return self._supersession
        return getattr(self, "supersession", False)

    @property
    def _resolve_cooc_boost(self) -> bool:
        if hasattr(self, "_cooc_boost"):
            return self._cooc_boost
        return getattr(self, "cooc_boost_retrieval", False)

    @property
    def _resolve_dream_interval(self) -> int:
        if hasattr(self, "_dream_interval"):
            return self._dream_interval
        return getattr(self, "dream_interval", 0)

    @property
    def _resolve_triple_index(self):
        return self._ss("triple_index")

    @property
    def _resolve_text_to_store_idx(self):
        return self._ss("text_to_store_idx")

    @property
    def _resolve_fact_entities(self):
        return self._ss("fact_entities")

    @property
    def _resolve_superseded_texts(self):
        return self._ss("superseded_texts")

    @property
    def _resolve_supersession_graph(self):
        return self._ss("supersession_graph")

    @property
    def _resolve_supersedes(self):
        return self._ss("supersedes")

    @property
    def _resolve_supersession_log(self):
        return self._ss("supersession_log")

    @property
    def _resolve_gliner_entity_index(self):
        ei = getattr(self, "_entity_idx", None)
        if ei is not None:
            return ei.index
        return self._gliner_entity_index  # type: ignore[attr-defined]

    @property
    def _resolve_rebel_kg(self):
        if hasattr(self, "_rebel"):
            return self._rebel._kg if self._rebel is not None else None
        return getattr(self, "_rebel_kg", None)

    def _resolve_rebel_extract_batch(self, texts):
        if hasattr(self, "_rebel") and self._rebel is not None:
            return self._rebel.extract_batch(texts)
        return self._rebel_extract_batch(texts)  # type: ignore[attr-defined]

    def _resolve_rebel_kg_add(self, h, r, t):
        if hasattr(self, "_rebel") and self._rebel is not None:
            return self._rebel.add_triple(h, r, t)
        return self._rebel_kg_add(h, r, t)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Entity extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_primary_entity(fact) -> str | None:
        """Extract primary entity from a SegmentedFact.

        Tries spaCy NER entities first (from ingestion pipeline),
        then falls back to the first triple subject.
        """
        if fact.entities:
            return fact.entities[0]
        if fact.triples:
            return fact.triples[0][0]
        return None

    # ------------------------------------------------------------------
    # Public API: _store and decomposed helpers
    # ------------------------------------------------------------------

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
        """Segment text, embed facts, extract REBEL triples."""
        result = self._ingestion.segment(text)
        fact_texts = [f.text for f in result.facts]

        rebel_triples_batch: list[list[dict]] | None = None
        run_rebel = self._resolve_rebel_kg is not None

        # Embed first, then REBEL sequentially (shared GPU).
        embeddings = self._scorer.embed_batch(fact_texts)
        if run_rebel:
            rebel_triples_batch = self._resolve_rebel_extract_batch(fact_texts)

        # GLiNER entity extraction skipped -- consumers disabled.
        primary_entities: list[str | None] = [None] * len(fact_texts)
        all_entities: list[list] = [[] for _ in fact_texts]

        # Entity-position residuals for entity reranking.
        entity_residuals: list[np.ndarray | None] | None = None
        if self._entity_rerank:
            # Extract primary entity from spaCy NER (fact.entities) or triples
            batch_entities = []
            batch_texts = []
            batch_indices = []
            for i, fact in enumerate(result.facts):
                ent = self._extract_primary_entity(fact)
                if ent is not None:
                    batch_entities.append(ent)
                    batch_texts.append(fact.text)
                    batch_indices.append(i)
            entity_residuals = [None] * len(fact_texts)
            if batch_texts:
                residuals = self._scorer.embed_entity_position_batch(
                    batch_texts, batch_entities,
                )
                for j, idx in enumerate(batch_indices):
                    entity_residuals[idx] = residuals[j]

        # Build entity index with insertion offset.
        base_idx = len(self._resolve_engine.memory_store)
        for i, ents in enumerate(all_entities):
            fact_idx = base_idx + i
            for ent in ents:
                key = ent["text"].lower().strip()
                if len(key) > 1:
                    self._resolve_gliner_entity_index[key].add(fact_idx)

        return _PrepareResult(
            result=result,
            fact_texts=fact_texts,
            embeddings=embeddings,
            rebel_triples_batch=rebel_triples_batch,
            primary_entities=primary_entities,
            entity_residuals=entity_residuals,
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

        # Entity-position residual for this fact (None if not enabled)
        ent_res = (
            prep.entity_residuals[fi]
            if prep.entity_residuals is not None
            else None
        )

        if superseded_idx is not None:
            ms_idx = self._store_replace_entry(
                fact, emb, superseded_idx, superseded_text,
                entity_residual=ent_res,
            )
        else:
            ms_idx = self._store_new_entry(fact, emb, entity,
                                           entity_residual=ent_res)

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
        if not self._resolve_supersession or not fact.triples:
            return None, None

        new_entities = (
            set(e.lower() for e in fact.entities) if fact.entities else set()
        )
        self._resolve_fact_entities[fact.text] = fact.entities

        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            for existing_obj, existing_text in self._resolve_triple_index[key]:
                if existing_obj == new_obj:
                    continue
                # Entity gate: substring matching.
                old_ents = [
                    e.lower()
                    for e in self._resolve_fact_entities.get(existing_text, ())
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
                old_idx = self._resolve_text_to_store_idx.get(existing_text)
                if old_idx is not None:
                    return old_idx, existing_text
        return None, None

    def _store_replace_entry(
        self,
        fact: SegmentedFact,
        emb: np.ndarray,
        superseded_idx: int,
        superseded_text: str,
        entity_residual: np.ndarray | None = None,
    ) -> int:
        """Replace a superseded entry in-place.

        Logs the supersession, replaces in coupled engine, updates
        reverse lookup, supersession graph, chain cache, and belief.
        Returns the store index.
        """
        engine = self._resolve_engine
        old_entry = engine.memory_store[superseded_idx]
        self._resolve_supersession_log.append({
            "old_text": old_entry.text,
            "new_text": fact.text,
            "old_recency": old_entry.recency,
            "new_recency": float(self._store_count),
            "timestamp": time.time(),
        })
        engine.replace_entry(
            superseded_idx, fact.text, emb, recency=float(self._store_count)
        )
        if entity_residual is not None:
            engine.memory_store[superseded_idx].entity_residual = entity_residual
        ms_idx = superseded_idx
        # Update reverse lookup.
        del self._resolve_text_to_store_idx[superseded_text]
        self._resolve_text_to_store_idx[fact.text] = ms_idx
        # Track superseded text for query-time safety net.
        self._resolve_superseded_texts.add(superseded_text)
        # Supersession graph for query-time CR.
        self._resolve_supersession_graph[superseded_text] = fact.text
        self._resolve_supersedes[fact.text].add(superseded_text)
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
        entity_residual: np.ndarray | None = None,
    ) -> int:
        """Store a new fact (no supersession).

        Stores via V1 orchestrator (when cooc_boost inactive) and
        V2 coupled engine. Returns the store index.
        """
        if not self._resolve_cooc_boost:
            self._resolve_orchestrator.store(content=fact.text, embedding=emb)
        ms_idx = self._resolve_engine.store(
            text=fact.text,
            embedding=emb,
            recency=float(self._store_count),
            entity=entity,
        )
        if entity_residual is not None:
            self._resolve_engine.memory_store[ms_idx].entity_residual = entity_residual
        self._resolve_text_to_store_idx[fact.text] = ms_idx
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
                self._resolve_rebel_kg_add(
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
        if fact.text not in self._resolve_fact_entities:
            self._resolve_fact_entities[fact.text] = fact.entities
        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            if superseded_text is not None:
                self._resolve_triple_index[key] = [
                    (o, t)
                    for o, t in self._resolve_triple_index[key]
                    if t != superseded_text
                ]
            self._resolve_triple_index[key].append((new_obj, fact.text))

    def _store_detect_belief_conflicts(self, fact: SegmentedFact) -> None:
        """Detect conflicts for belief index when supersession is off.

        When supersession is disabled but belief is active, scan the
        triple index for (S,P) matches with different objects and
        fire belief.on_conflict for each pair.
        """
        if self._resolve_supersession or not self._belief_index or not fact.triples:
            return

        new_entities = (
            set(e.lower() for e in fact.entities) if fact.entities else set()
        )
        if fact.text not in self._resolve_fact_entities:
            self._resolve_fact_entities[fact.text] = fact.entities

        for subj, pred, obj in fact.triples:
            key = (subj.lower().strip(), pred.lower().strip())
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            for existing_obj, existing_text in self._resolve_triple_index[key]:
                if existing_obj == new_obj:
                    continue
                old_ents = [
                    e.lower()
                    for e in self._resolve_fact_entities.get(existing_text, ())
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
        engine = self._resolve_engine

        # Deferred FAISS rebuild: one per chunk.
        engine.rebuild_faiss_if_dirty()

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
                engine.memory_store,
            )

        # Flush Hebbian session buffer.
        if (
            hasattr(engine, "_session_buffer")
            and len(getattr(engine, "_session_buffer", [])) > 0
        ):
            engine.flush_session()

        # Supersession diagnostics: log every 50 stores.
        if (
            self._resolve_supersession
            and self._store_count % 50 == 0
            and self._store_count > 0
        ):
            logging.getLogger("supersession").info(
                "STATS after %d stores: %d unique (S,P) keys, "
                "%d superseded texts, %d graph edges, %d total triples",
                self._store_count,
                len(self._resolve_triple_index),
                len(self._resolve_superseded_texts),
                len(self._resolve_supersession_graph),
                sum(len(v) for v in self._resolve_triple_index.values()),
            )

        # Dream cycle.
        if self._resolve_dream_interval > 0 and self._store_count % self._resolve_dream_interval == 0:
            engine.dream()


# Backward-compat alias: code importing StoreMixin still works
StoreMixin = FactStore
