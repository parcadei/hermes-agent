"""HyperedgeIntersection: entity-constrained reranking for decompose hops.

Uses the GLiNER entity index to penalize retrieval scores for facts that don't
mention the intermediate answer entity.  Standalone class -- no mixin, no
framework inheritance.  Only dependency: numpy.

Spec: thoughts/shared/plans/knowledge-graph-retrieval/spec.md  (Module 1)
"""

from __future__ import annotations

import numpy as np


class HyperedgeIntersection:
    """Penalize retrieval scores for facts that don't mention a given entity.

    Parameters
    ----------
    entity_index:
        Mapping from lowercase entity name to set of fact-store indices.
        Held by *reference* -- mutations by the caller (e.g. during ongoing
        ingestion) are immediately visible.
    penalty:
        Multiplicative penalty applied to scores of unmatched facts.
        0.0 = hard filter, 1.0 = no effect.  Default 0.1.
    min_candidates:
        Minimum number of entity-matched facts required to activate
        the penalty.  If fewer facts match, scores are returned unchanged.
    """

    def __init__(
        self,
        entity_index: dict[str, set[int]],
        *,
        penalty: float = 0.1,
        min_candidates: int = 3,
    ) -> None:
        self._entity_index = entity_index  # reference, not copy
        self._penalty = penalty
        self._min_candidates = min_candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        intermediate_entity: str,
        scores: np.ndarray,
        store_size: int,
    ) -> np.ndarray:
        """Penalize scores for facts that don't mention *intermediate_entity*.

        Exact-match lookup only.  If matched facts < *min_candidates*, scores
        are returned unchanged (copy).

        Parameters
        ----------
        intermediate_entity:
            Raw entity string from LLM output.  Normalized to lowercase/strip.
        scores:
            1-D retrieval scores, shape ``(N,)`` where ``N >= store_size``.
        store_size:
            Number of real facts.  Only indices ``0 .. store_size-1`` are
            eligible for penalty; anything beyond is left untouched.

        Returns
        -------
        np.ndarray -- adjusted scores (always a fresh copy).
        """
        normalized = intermediate_entity.lower().strip()
        matched = self._entity_index.get(normalized, set())
        return self._apply_penalty(scores, matched, store_size)

    def rerank_with_fuzzy_match(
        self,
        intermediate_entity: str,
        scores: np.ndarray,
        store_size: int,
    ) -> tuple[np.ndarray, set[int]]:
        """Like :meth:`rerank` but with escalating fuzzy match.

        Escalation order:
        1. Exact match
        2. Substring (entity in key **or** key in entity)

        Each level is tried only if the previous one yielded fewer than
        *min_candidates* matched indices.

        Returns
        -------
        tuple[np.ndarray, set[int]]
            (adjusted_scores, matched_indices)
        """
        matched = self._fuzzy_lookup(intermediate_entity)
        adjusted = self._apply_penalty(scores, matched, store_size)
        return adjusted, matched

    def entity_coverage(self, entity: str) -> int:
        """Return count of facts matching *entity* using fuzzy logic."""
        return len(self._fuzzy_lookup(entity))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_lookup(self, entity: str) -> set[int]:
        """Escalating entity lookup: exact -> substring."""
        normalized = entity.lower().strip()

        # Level 1: exact match
        matched = set(self._entity_index.get(normalized, set()))
        if len(matched) >= self._min_candidates:
            return matched

        # Level 2: substring (entity in key OR key in entity)
        for key, indices in self._entity_index.items():
            if normalized in key or key in normalized:
                matched |= indices

        return matched

    def _apply_penalty(
        self,
        scores: np.ndarray,
        matched: set[int],
        store_size: int,
    ) -> np.ndarray:
        """Return a copy of *scores* with unmatched indices penalized.

        If ``len(matched) < min_candidates`` the copy is returned unchanged.
        Only indices ``0 .. store_size-1`` are considered; anything beyond
        *store_size* passes through untouched.
        """
        result = scores.copy()

        if len(matched) < self._min_candidates:
            return result

        if store_size <= 0 or len(result) == 0:
            return result

        # Build a boolean mask: True for indices that should be penalized.
        # Only consider 0..store_size-1; anything beyond is untouched.
        effective = min(store_size, len(result))
        for i in range(effective):
            if i not in matched:
                result[i] *= self._penalty

        return result
