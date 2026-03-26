"""Core retrieval pipeline: embedding, candidate scoring, post-retrieval processing.

Converted from RetrievalMixin to standalone Retriever class.
All self.* attribute access replaced with explicit config + instance fields.

Migration notes for callers:
  - RetrievalMixin is a backward-compat alias for Retriever.
  - Mixin callers (hermes_agent.py) still work via MRO -- the alias
    keeps the import path stable.  HermesMemoryAgent.__init__ sets
    self.coupled_engine / self._scorer / self._belief_index / etc. directly;
    the _resolve_* properties fall back to those when _cfg is absent.
  - Standalone usage:
      from mabench.retrieval import Retriever
      from mabench.configs import RetrievalConfig
      retriever = Retriever(engine, scorer, config=RetrievalConfig(cooc_boost=True))
      results = retriever.retrieve("What sport does Abbiati play?")
"""

from __future__ import annotations

import numpy as np

from mabench.configs import BeliefConfig, CRConfig, DedupConfig, RetrievalConfig


class Retriever:
    """Standalone retrieval pipeline: embed, retrieve candidates, post-process.

    Can be used independently or composed into HermesMemoryAgent.

    Usage::

        retriever = Retriever(engine, scorer, config=RetrievalConfig(cooc_boost=True))
        results = retriever.retrieve("What sport does Abbiati play?")
    """

    def __init__(
        self,
        engine=None,
        scorer=None,
        belief_index=None,
        config: RetrievalConfig | None = None,
        cr_config: CRConfig | None = None,
        dedup_config: DedupConfig | None = None,
        belief_config: BeliefConfig | None = None,
        nlp=None,
        entity_reranker=None,
    ) -> None:
        # _engine: CoupledEngine with .memory_store, .query(), .query_cooc_boost(), etc.
        self._engine = engine
        # _scorer.embed(text) -> np.ndarray of shape (dim,)
        self._scorer = scorer
        # _belief_index: BeliefIndex with .is_excluded(text)/.score(text), or None
        self._belief_index = belief_index
        # _nlp: spaCy Language model for entity extraction
        self._nlp = nlp
        # _entity_reranker: EntityReranker instance, or None
        self._entity_reranker = entity_reranker

        # _cfg: RetrievalConfig(retrieve_num=10, cooc_boost=False, ...)
        self._cfg = config or RetrievalConfig()
        # _cr_cfg: CRConfig(enabled=False, overfetch=3)
        self._cr_cfg = cr_config or CRConfig()
        # _dedup_cfg: DedupConfig(enabled=False, threshold=0.95)
        self._dedup_cfg = dedup_config or DedupConfig()
        # _belief_cfg: BeliefConfig(enabled=False, overfetch=3)
        self._belief_cfg = belief_config or BeliefConfig()

    # ------------------------------------------------------------------
    # Attribute resolution: new config or old self.* fallback
    # ------------------------------------------------------------------
    # Pattern: hasattr(self, '_cfg') distinguishes standalone (has _cfg)
    # from mixin mode (no _cfg, uses raw self.X set by host class).

    @property
    def _resolve_engine(self):
        """CoupledEngine -- from __init__ or host class self.coupled_engine."""
        if hasattr(self, "_engine") and self._engine is not None:
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_scorer(self):
        """EmbeddingRelevanceScorer -- from __init__ or host class self._scorer."""
        if hasattr(self, "_cfg") and hasattr(self, "_scorer") and self._scorer is not None:
            return self._scorer
        # In mixin mode, _scorer is set directly by host __init__
        return self._scorer  # type: ignore[has-type]

    @property
    def _resolve_belief_index(self):
        """BeliefIndex or None -- from __init__ or host class self._belief_index."""
        if hasattr(self, "_cfg"):
            return self._belief_index
        return self._belief_index  # type: ignore[has-type]

    @property
    def _resolve_retrieve_num(self) -> int:
        """Base number of results to retrieve."""
        if hasattr(self, "_cfg"):
            return self._cfg.retrieve_num
        return self.retrieve_num  # type: ignore[attr-defined]

    @property
    def _resolve_conflict_resolution(self) -> bool:
        """Whether conflict resolution is active."""
        if hasattr(self, "_cr_cfg"):
            return self._cr_cfg.enabled
        return getattr(self, "conflict_resolution", False)

    @property
    def _resolve_cosine_dedup(self) -> bool:
        """Whether cosine deduplication is active."""
        if hasattr(self, "_dedup_cfg"):
            return self._dedup_cfg.enabled
        return getattr(self, "cosine_dedup", False)

    @property
    def _resolve_cr_overfetch(self) -> int:
        """CR overfetch multiplier."""
        if hasattr(self, "_cr_cfg"):
            return self._cr_cfg.overfetch
        return getattr(self, "_cr_overfetch", 3)

    @property
    def _resolve_belief_overfetch(self) -> int:
        """Belief overfetch multiplier."""
        if hasattr(self, "_belief_cfg"):
            return self._belief_cfg.overfetch
        return getattr(self, "_belief_overfetch", 3)

    @property
    def _resolve_transfer(self) -> bool:
        """Whether transfer retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.transfer
        return getattr(self, "transfer_retrieval", False)

    @property
    def _resolve_coretrieval(self) -> bool:
        """Whether coretrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval
        return getattr(self, "coretrieval_retrieval", False)

    @property
    def _resolve_cooc_boost(self) -> bool:
        """Whether co-occurrence boosted cosine is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_boost
        return getattr(self, "cooc_boost_retrieval", False)

    @property
    def _resolve_ppr(self) -> bool:
        """Whether personalized PageRank is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr
        return getattr(self, "ppr_retrieval", False)

    @property
    def _resolve_hybrid(self) -> bool:
        """Whether hybrid retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.hybrid
        return getattr(self, "hybrid_retrieval", False)

    @property
    def _resolve_associative(self) -> bool:
        """Whether associative retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.associative
        return getattr(self, "associative_retrieval", False)

    @property
    def _resolve_sparse(self) -> bool:
        """Whether sparse retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.sparse
        return getattr(self, "sparse_retrieval", False)

    @property
    def _resolve_transfer_k(self) -> int | None:
        """Transfer k parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.transfer_k
        return getattr(self, "transfer_k", None)

    @property
    def _resolve_coretrieval_bonus(self) -> float:
        """Coretrieval bonus parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval_bonus
        return getattr(self, "coretrieval_bonus", 0.3)

    @property
    def _resolve_coretrieval_min_count(self) -> float:
        """Coretrieval minimum count parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval_min_count
        return getattr(self, "coretrieval_min_count", 2.0)

    @property
    def _resolve_cooc_weight(self) -> float:
        """Co-occurrence weight parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_weight
        return getattr(self, "cooc_weight", 0.3)

    @property
    def _resolve_cooc_gate_threshold(self) -> float:
        """Co-occurrence gate threshold."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_gate_threshold
        return getattr(self, "cooc_gate_threshold", 0.0)

    @property
    def _resolve_ppr_weight(self) -> float:
        """Personalized PageRank weight."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr_weight
        return getattr(self, "ppr_weight", 0.5)

    @property
    def _resolve_ppr_damping(self) -> float:
        """Personalized PageRank damping factor."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr_damping
        return getattr(self, "ppr_damping", 0.85)

    @property
    def _resolve_bm25_weight(self) -> float:
        """BM25 hybrid weight (0 = no BM25)."""
        if hasattr(self, "_cfg"):
            return self._cfg.bm25_weight
        return getattr(self, "bm25_weight", 0.0)

    @property
    def _resolve_dream_weight(self) -> float:
        """Dream consolidation weight."""
        if hasattr(self, "_cfg"):
            return self._cfg.dream_weight
        return getattr(self, "dream_weight", 0.0)

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _embed_query(self, query_text: str):
        """Embed a query string.

        Returns:
            numpy array of shape (dim,) -- the query embedding.
        """
        return self._resolve_scorer.embed(query_text)

    def _compute_effective_k(self, top_k: int) -> int:
        """Compute the effective retrieval count after overfetch multipliers.

        Over-fetches when CR, cosine dedup, or belief is active to give
        filtering more candidates.  Conflicting facts have cosine ~0.98-0.99,
        so top-10 drops correct versions.

        Args:
            top_k: Base number of results before overfetch.

        Returns:
            Effective k after applying all overfetch multipliers.
        """
        effective_k = top_k
        if self._resolve_conflict_resolution and self._resolve_cr_overfetch > 1:
            effective_k = top_k * self._resolve_cr_overfetch
        if self._resolve_cosine_dedup:
            effective_k = max(effective_k, top_k * self._resolve_cr_overfetch)
        if self._resolve_belief_index is not None:
            effective_k = max(effective_k, top_k * self._resolve_belief_overfetch)
        return effective_k

    def _retrieve_candidates(
        self,
        q_emb,
        top_k: int,
        query_text: str | None = None,
    ) -> list[dict]:
        """Run the appropriate retrieval backend based on config flags.

        Computes effective_k via ``_compute_effective_k`` (CR overfetch,
        cosine dedup, belief overfetch).  The *top_k* parameter is the BASE
        retrieve_num before overfetch multipliers.

        Args:
            q_emb: Query embedding (numpy array).
            top_k: Base number of results to retrieve (before overfetch).
            query_text: Original query string, needed by the cooc+bm25
                backend.  May be ``None`` for backends that don't use it.

        Returns:
            List of result dicts with at minimum {"text": str, "score": float}.
        """
        effective_k = self._compute_effective_k(top_k)
        engine = self._resolve_engine

        if self._resolve_transfer:
            results = engine.query_transfer(
                embedding=q_emb,
                top_k=effective_k,
                transfer_k=self._resolve_transfer_k,
            )
        elif self._resolve_coretrieval:
            results = engine.query_coretrieval(
                embedding=q_emb,
                top_k=effective_k,
                coretrieval_bonus=self._resolve_coretrieval_bonus,
                min_coretrieval_count=self._resolve_coretrieval_min_count,
            )
        elif self._resolve_cooc_boost and self._resolve_ppr:
            results = engine.query_cooc_ppr(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
                ppr_weight=self._resolve_ppr_weight,
                damping=self._resolve_ppr_damping,
            )
        elif self._resolve_cooc_boost and self._resolve_bm25_weight > 0:
            results = engine.query_cooc_bm25(
                embedding=q_emb,
                query_text=query_text,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                bm25_weight=self._resolve_bm25_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
            )
        elif self._resolve_cooc_boost:
            results = engine.query_cooc_boost(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                dream_weight=self._resolve_dream_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
            )
        elif self._resolve_ppr:
            results = engine.query_ppr(
                embedding=q_emb,
                top_k=effective_k,
                ppr_weight=self._resolve_ppr_weight,
                damping=self._resolve_ppr_damping,
            )
        elif self._resolve_hybrid:
            results = engine.query_hybrid(
                embedding=q_emb,
                top_k=effective_k,
            )
        elif self._resolve_associative or self._resolve_sparse:
            results = engine.query_associative(
                embedding=q_emb,
                top_k=effective_k,
                sparse=self._resolve_sparse,
            )
        else:
            results = engine.query(
                embedding=q_emb,
                top_k=effective_k,
            )

        return results

    def _apply_post_retrieval(self, results: list[dict]) -> list[dict]:
        """Apply post-retrieval reranking and filtering.

        Currently: belief reranking when active, truncation to retrieve_num.

        Args:
            results: Raw retrieval results.

        Returns:
            Reranked and truncated results.
        """
        if self._resolve_belief_index is not None:
            results = self._apply_belief_rerank(results)[
                : self._resolve_retrieve_num
            ]
        return results

    def _retrieve_v2(self, query_text: str) -> list[dict]:
        """Run V2 embedding retrieval for a single query string.

        Composed from independent stages:
          1. _embed_query      -- vectorize the query text
          2. _retrieve_candidates -- run the backend retrieval switch
          3. _apply_post_retrieval -- belief rerank + truncation
          4. entity reranker (optional) -- via external EntityReranker callback
        """
        q_emb = self._embed_query(query_text)
        results = self._retrieve_candidates(
            q_emb, self._resolve_retrieve_num, query_text=query_text,
        )
        results = self._apply_post_retrieval(results)
        if self._entity_reranker is not None:
            results = self._entity_reranker.rerank(
                query_text, results, self._resolve_engine,
            )
        return results

    def _apply_belief_rerank(self, results: list[dict]) -> list[dict]:
        """Rerank retrieval results using Bayesian belief scores.

        1. Exclude facts below hard_floor (P(current) too low).
        2. Multiply each result's score by P(current).
        3. Re-sort by adjusted score descending.
        """
        belief = self._resolve_belief_index
        reranked = []
        for r in results:
            text = r.get("text", "")
            if belief.is_excluded(text):
                continue
            p = belief.score(text)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * p
            reranked.append(adjusted)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Public API aliases (for standalone callers who prefer clean names)
    # ------------------------------------------------------------------
    embed_query = _embed_query
    retrieve = _retrieve_v2
    retrieve_candidates = _retrieve_candidates
    compute_effective_k = _compute_effective_k
    apply_post_retrieval = _apply_post_retrieval
    apply_belief_rerank = _apply_belief_rerank


# Backward-compat alias -- hermes_agent.py imports RetrievalMixin
RetrievalMixin = Retriever
