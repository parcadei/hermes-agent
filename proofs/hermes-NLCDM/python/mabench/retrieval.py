"""Core retrieval pipeline: embedding, candidate scoring, post-retrieval processing.

Provides RetrievalMixin with methods for:
- Query embedding via scorer
- Backend dispatch (cosine, cooc, PPR, BM25, hybrid, associative, transfer, coretrieval)
- Overfetch computation for conflict resolution / belief reranking
- Belief-based reranking and truncation
- Top-level retrieve_v2 entry point composing embed -> candidates -> post-retrieval
"""

from __future__ import annotations


class RetrievalMixin:
    """Mixin providing retrieval, re-ranking, and belief-based filtering.

    Requires the host class to provide:
    - self._scorer: EmbeddingRelevanceScorer with .embed(text) method
    - self.coupled_engine: engine with query_* methods and .memory_store
    - self._belief_index: belief index with .is_excluded/.score, or None
    - self._belief_overfetch: int, overfetch multiplier for belief
    - self._cr_overfetch: int, overfetch multiplier for CR
    - self.retrieve_num: int, base number of results to retrieve
    - self.conflict_resolution: bool, whether CR is active
    - self.cosine_dedup: bool, whether cosine dedup is active
    - self.transfer_retrieval: bool
    - self.coretrieval_retrieval: bool
    - self.cooc_boost_retrieval: bool
    - self.ppr_retrieval: bool
    - self.hybrid_retrieval: bool
    - self.associative_retrieval: bool
    - self.sparse_retrieval: bool
    - self.transfer_k: int
    - self.coretrieval_bonus: float
    - self.coretrieval_min_count: int
    - self.cooc_weight: float
    - self.cooc_gate_threshold: float
    - self.ppr_weight: float
    - self.ppr_damping: float
    - self.bm25_weight: float
    - self.dream_weight: float
    """

    def _embed_query(self, query_text: str):
        """Embed a query string.

        Returns:
            numpy array of shape (dim,) -- the query embedding.
        """
        return self._scorer.embed(query_text)

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
        if self.conflict_resolution and self._cr_overfetch > 1:
            effective_k = top_k * self._cr_overfetch
        if self.cosine_dedup:
            effective_k = max(effective_k, top_k * self._cr_overfetch)
        if self._belief_index is not None:
            effective_k = max(effective_k, top_k * self._belief_overfetch)
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

        if self.transfer_retrieval:
            results = self.coupled_engine.query_transfer(
                embedding=q_emb,
                top_k=effective_k,
                transfer_k=self.transfer_k,
            )
        elif self.coretrieval_retrieval:
            results = self.coupled_engine.query_coretrieval(
                embedding=q_emb,
                top_k=effective_k,
                coretrieval_bonus=self.coretrieval_bonus,
                min_coretrieval_count=self.coretrieval_min_count,
            )
        elif self.cooc_boost_retrieval and self.ppr_retrieval:
            results = self.coupled_engine.query_cooc_ppr(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                gate_threshold=self.cooc_gate_threshold,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.cooc_boost_retrieval and self.bm25_weight > 0:
            results = self.coupled_engine.query_cooc_bm25(
                embedding=q_emb,
                query_text=query_text,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                bm25_weight=self.bm25_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.cooc_boost_retrieval:
            results = self.coupled_engine.query_cooc_boost(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                dream_weight=self.dream_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.ppr_retrieval:
            results = self.coupled_engine.query_ppr(
                embedding=q_emb,
                top_k=effective_k,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.hybrid_retrieval:
            results = self.coupled_engine.query_hybrid(
                embedding=q_emb,
                top_k=effective_k,
            )
        elif self.associative_retrieval or self.sparse_retrieval:
            results = self.coupled_engine.query_associative(
                embedding=q_emb,
                top_k=effective_k,
                sparse=self.sparse_retrieval,
            )
        else:
            results = self.coupled_engine.query(
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
        if self._belief_index is not None:
            results = self._apply_belief_rerank(results)[: self.retrieve_num]
        return results

    def _retrieve_v2(self, query_text: str) -> list[dict]:
        """Run V2 embedding retrieval for a single query string.

        Composed from three independent stages:
          1. _embed_query      -- vectorize the query text
          2. _retrieve_candidates -- run the backend retrieval switch
          3. _apply_post_retrieval -- belief rerank + truncation
        """
        q_emb = self._embed_query(query_text)
        results = self._retrieve_candidates(
            q_emb, self.retrieve_num, query_text=query_text,
        )
        return self._apply_post_retrieval(results)

    def _apply_belief_rerank(self, results: list[dict]) -> list[dict]:
        """Rerank retrieval results using Bayesian belief scores.

        1. Exclude facts below hard_floor (P(current) too low).
        2. Multiply each result's score by P(current).
        3. Re-sort by adjusted score descending.
        """
        reranked = []
        for r in results:
            text = r.get("text", "")
            if self._belief_index.is_excluded(text):
                continue
            p = self._belief_index.score(text)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * p
            reranked.append(adjusted)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked
