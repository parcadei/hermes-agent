"""Ingestion store pipeline — standalone FactStore.

_resolve_* properties support both standalone (explicit __init__) and
mixin mode (attributes set by HermesMemoryAgent.__init__).
StoreMixin is a backward-compat alias for FactStore.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
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
    gliner2_triples_batch: list[list[tuple[str, str, str]]] | None = None
    cooc_edges_batch: list[list[tuple[str, str, str]]] | None = None
    relik_triples_batch: list[list[tuple[str, str, str]]] | None = None


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
        dream_interval: int = 0, entity_reranker=None,
        knowledge_graph=None, gliner2_extractor=None,
        cooc_extractor=None,
        relik_extractor=None,
        cooc_kg=None,
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
        self._entity_reranker = entity_reranker
        self._knowledge_graph = knowledge_graph
        self._gliner2_extractor = gliner2_extractor
        self._cooc_extractor = cooc_extractor
        self._relik_extractor = relik_extractor
        self._cooc_kg = cooc_kg
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
    def _resolve_subject_index(self):
        return self._ss("subject_index")

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
    # Public API: _store and decomposed helpers
    # ------------------------------------------------------------------

    def finalize_cooc_kg(self) -> None:
        """Call after ingestion batch completes to compute NPMI weights.

        Safe to call when no CoocKG is configured (no-op).
        """
        if self._cooc_kg is not None:
            self._cooc_kg.compute_npmi()

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
        """Segment text, embed facts, extract triples (REBEL + co-occurrence + GLiNER2)."""
        result = self._ingestion.segment(text)
        fact_texts = [f.text for f in result.facts]

        rebel_triples_batch: list[list[dict]] | None = None
        run_rebel = self._resolve_rebel_kg is not None

        # Embed first, then REBEL sequentially (shared GPU).
        embeddings = self._scorer.embed_batch(fact_texts)
        if run_rebel:
            rebel_triples_batch = self._resolve_rebel_extract_batch(fact_texts)

        # Entity extraction: co-occurrence extractor doubles as entity source.
        primary_entities: list[str | None] = [None] * len(fact_texts)
        cooc_edges_batch = None
        if self._cooc_extractor is not None:
            if self._cooc_kg is not None:
                # CoocKG path: sentence-window extraction feeds both the KG
                # and the entity index in one pass.
                spacy_docs = list(self._ingestion._nlp.pipe(fact_texts))
                entities_batch = self._cooc_extractor.extract_and_feed_cooc_kg(
                    fact_texts, self._cooc_kg, spacy_docs=spacy_docs,
                )
                # Still need co-occurrence edges for KnowledgeGraph + TriadicMemory
                cooc_edges_batch = self._cooc_extractor.extract_batch(fact_texts)
            else:
                # No CoocKG: use original path
                cooc_edges_batch = self._cooc_extractor.extract_batch(fact_texts)
                entities_batch = self._cooc_extractor.extract_entities_batch(fact_texts)
            # Populate entity index from extracted entities.
            base_idx = len(self._resolve_engine.memory_store)
            for i, ents in enumerate(entities_batch):
                fact_idx = base_idx + i
                for ent in ents:
                    key = ent["text"].lower().strip()
                    if len(key) > 1:
                        self._resolve_gliner_entity_index[key].add(fact_idx)
                # Primary entity: prefer person, fallback to highest score.
                if ents:
                    person_ents = [e for e in ents if e["label"] == "person"]
                    best = person_ents[0] if person_ents else max(ents, key=lambda e: e["score"])
                    primary_entities[i] = best["text"].strip()
        else:
            # No co-occurrence extractor: entity index stays empty.
            base_idx = len(self._resolve_engine.memory_store)

        # GLiNER2 triple extraction (dense triples for supersession).
        gliner2_triples_batch = None
        if self._gliner2_extractor is not None:
            gliner2_triples_batch = self._gliner2_extractor.extract_batch(
                fact_texts
            )

        # ReLiK v2 triple extraction (canonical Wikidata triples for supersession).
        relik_triples_batch = None
        if self._relik_extractor is not None:
            relik_triples_batch = self._relik_extractor.extract_batch(
                fact_texts
            )

        return _PrepareResult(
            result=result,
            fact_texts=fact_texts,
            embeddings=embeddings,
            rebel_triples_batch=rebel_triples_batch,
            primary_entities=primary_entities,
            gliner2_triples_batch=gliner2_triples_batch,
            cooc_edges_batch=cooc_edges_batch,
            relik_triples_batch=relik_triples_batch,
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
        # Merge GLiNER2 triples with spaCy triples (union, deduped).
        if (prep.gliner2_triples_batch is not None
                and fi < len(prep.gliner2_triples_batch)):
            g2_triples = prep.gliner2_triples_batch[fi]
            if g2_triples:
                combined = tuple(set(fact.triples) | set(g2_triples))
                fact = replace(fact, triples=combined)

        # Merge ReLiK v2 triples (canonical Wikidata predicates for supersession).
        if (prep.relik_triples_batch is not None
                and fi < len(prep.relik_triples_batch)):
            relik_triples = prep.relik_triples_batch[fi]
            if relik_triples:
                combined = tuple(set(fact.triples) | set(relik_triples))
                fact = replace(fact, triples=combined)

        superseded_idx, superseded_text = self._store_check_supersession(fact)

        if superseded_idx is not None:
            ms_idx = self._store_replace_entry(
                fact, emb, superseded_idx, superseded_text,
            )
        else:
            ms_idx = self._store_new_entry(fact, emb, entity)

        # Entity reranker annotation (isolated entity residual)
        if self._entity_reranker is not None:
            entry = self._resolve_engine.memory_store[ms_idx]
            self._entity_reranker.annotate_entry(entry, fact)

        self._store_update_indices(
            fi, fact, ms_idx, prep.rebel_triples_batch,
            prep.cooc_edges_batch,
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
        self._resolve_text_to_store_idx[fact.text] = ms_idx
        return ms_idx

    def _store_update_indices(
        self,
        fi: int,
        fact: SegmentedFact,
        ms_idx: int,
        rebel_triples_batch: list[list[dict]] | None,
        cooc_edges_batch: list[list[tuple[str, str, str]]] | None = None,
    ) -> None:
        """Update BM25, triadic, REBEL KG, knowledge graph, and belief indices.

        Co-occurrence edges (from fast-path bi-encoder NER) are routed to
        KnowledgeGraph and TriadicMemory. These consumers don't need
        predicates — they need entity adjacency.  spaCy/GLiNER2 triples
        continue to feed supersession (handled in _store_update_triple_index).
        """
        # BM25.
        if getattr(self, "_bm25_index", None) is not None:
            self._bm25_index.add(ms_idx, fact.text)

        # Determine which triples feed triadic and KG.
        # Priority: co-occurrence edges (fast path) > spaCy/GLiNER2 triples (fallback).
        cooc_edges = None
        if cooc_edges_batch is not None and fi < len(cooc_edges_batch):
            cooc_edges = cooc_edges_batch[fi]

        # Triadic: co-occurrence edges if available, else spaCy triples.
        if getattr(self, "_triadic", None) is not None:
            if cooc_edges:
                self._triadic.store_fact(fact.text, cooc_edges)
            else:
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

        # Knowledge graph: co-occurrence edges if available, else spaCy/GLiNER2 triples.
        if self._knowledge_graph is not None:
            if cooc_edges:
                self._knowledge_graph.add_fact(
                    fact.text, ms_idx, cooc_edges,
                    recency=float(self._store_count),
                )
            elif fact.triples:
                self._knowledge_graph.add_fact(
                    fact.text, ms_idx, list(fact.triples),
                    recency=float(self._store_count),
                )

    @classmethod
    def _normalize_pred(cls, pred: str) -> str:
        """Normalize a predicate (identity — all variants stored as-is).

        Predicate disambiguation is handled at QUERY time by cosine
        similarity fallback in QueryEngine._predicate_cosine_fallback,
        not at ingestion time via hardcoded synonym maps.
        """
        return pred

    def _store_update_triple_index(
        self,
        fact: SegmentedFact,
        superseded_text: str | None,
    ) -> None:
        """Update the (subject, predicate) -> [(object, text)] index.

        Removes superseded entries and appends the new fact's triples.
        Also updates the knowledge graph when supersession occurs.

        Bidirectional subject_index — indexes both (S,P)->O and
        (O,P)->S so triples are findable regardless of whether the
        LLM put the entity as subject or object.

        Predicate variants (e.g. "type of music" vs "plays") are stored
        as-is; disambiguation is handled at query time by cosine
        similarity fallback in QueryEngine._predicate_cosine_fallback.
        """
        if not fact.triples:
            return
        if fact.text not in self._resolve_fact_entities:
            self._resolve_fact_entities[fact.text] = fact.entities
        for subj, pred, obj in fact.triples:
            raw_pred = pred.lower().strip()
            norm_pred = self._normalize_pred(raw_pred)
            key = (subj.lower().strip(), norm_pred)
            new_obj = obj.lower().strip()
            if not key[0] or not key[1]:
                continue
            # Knowledge graph: supersede old triple if present.
            if superseded_text is not None and self._knowledge_graph is not None:
                for old_obj, old_text in self._resolve_triple_index[key]:
                    if old_text == superseded_text:
                        ms_idx = self._resolve_text_to_store_idx.get(fact.text)
                        if ms_idx is not None:
                            self._knowledge_graph.supersede(
                                old_triple=(subj, pred, old_obj),
                                new_triple=(subj, pred, obj),
                                new_fact_index=ms_idx,
                                new_recency=float(self._store_count),
                            )
                        break
            if superseded_text is not None:
                self._resolve_triple_index[key] = [
                    (o, t)
                    for o, t in self._resolve_triple_index[key]
                    if t != superseded_text
                ]
            self._resolve_triple_index[key].append((new_obj, fact.text))

            # Update subject_index (subject -> {predicate: (newest_obj, text)})
            # Newest-wins: overwrite previous entry for same (subj, pred).
            subj_key = key[0]  # already lowered/stripped
            pred_key = norm_pred
            si = self._resolve_subject_index
            if subj_key not in si:
                si[subj_key] = {}
            si[subj_key][pred_key] = (new_obj, fact.text)

            # Bidirectional index: also store reverse (object -> pred -> subject).
            # LLMs non-deterministically invert subject/object direction for
            # sentences like "The X of Y is Z" — sometimes Z becomes subject.
            # Storing the reverse ensures the triple is findable either way.
            if new_obj and len(new_obj) >= 3:
                if new_obj not in si:
                    si[new_obj] = {}
                si[new_obj][pred_key] = (subj_key, fact.text)

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

        # Flush triadic tensor writes (buffered during per-fact loop).
        if getattr(self, "_triadic", None) is not None:
            self._triadic.flush()

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
