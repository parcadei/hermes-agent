"""Chain cache with supersession invalidation.

Stores completed decompose chains keyed by query embedding similarity,
with supersession-driven invalidation via a reverse index from evidence
facts to cached chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CachedChain:
    """A cached decompose chain result."""

    query_embedding: np.ndarray
    query_text: str
    sub_queries: list[str]
    intermediate_answers: dict[str, str]
    evidence_facts: list[str]
    final_answer: str
    chain_id: int


class ChainCache:
    """Query chain cache with supersession invalidation.

    Stores completed decompose chains and provides cosine-similarity
    lookup. When a fact is superseded during ingestion, all chains
    that used that fact as evidence are evicted via a reverse index.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize empty cache."""
        self._threshold = similarity_threshold
        # chain_id -> CachedChain
        self._chains: dict[int, CachedChain] = {}
        # fact_text -> set of chain_ids that use this fact
        self._fact_to_chains: dict[str, set[int]] = {}

    def lookup(self, query_embedding: np.ndarray) -> Optional[CachedChain]:
        """Find a cached chain matching this query. Returns None on miss.

        Returns the chain with highest cosine similarity above threshold.
        """
        if not self._chains:
            return None

        best_chain: Optional[CachedChain] = None
        best_sim = -1.0

        for chain in self._chains.values():
            sim = float(np.dot(query_embedding, chain.query_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chain.query_embedding)
            ))
            if sim >= self._threshold and sim > best_sim:
                best_sim = sim
                best_chain = chain

        return best_chain

    def store(self, chain: CachedChain) -> None:
        """Cache a completed chain. Builds reverse index from evidence_facts."""
        self._chains[chain.chain_id] = chain
        for fact in chain.evidence_facts:
            if fact not in self._fact_to_chains:
                self._fact_to_chains[fact] = set()
            self._fact_to_chains[fact].add(chain.chain_id)

    def invalidate_fact(self, fact_text: str) -> int:
        """Invalidate all chains using this fact. Returns count evicted.

        Called when supersession fires during ingestion.
        """
        chain_ids = self._fact_to_chains.get(fact_text, set())
        if not chain_ids:
            return 0

        # Copy to avoid mutation during iteration
        to_evict = set(chain_ids)
        evicted = 0

        for chain_id in to_evict:
            if chain_id in self._chains:
                chain = self._chains[chain_id]
                # Remove this chain from all its fact entries in the reverse index
                for fact in chain.evidence_facts:
                    if fact in self._fact_to_chains:
                        self._fact_to_chains[fact].discard(chain_id)
                        if not self._fact_to_chains[fact]:
                            del self._fact_to_chains[fact]
                del self._chains[chain_id]
                evicted += 1

        return evicted

    def __len__(self) -> int:
        """Number of cached chains."""
        return len(self._chains)
