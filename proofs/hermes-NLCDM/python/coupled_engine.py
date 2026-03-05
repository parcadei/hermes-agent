"""Coupled Engine — runtime wrapping NLCDM proof system and dream operations.

Exposes store/query/dream interface for a Hopfield-based memory lifecycle.
All operations are pure numpy/scipy. No GPU, no external inference.

Primary state: {X, β}
  X: pattern matrix — each row is a stored embedding vector
  β: inverse temperature controlling precision/associativity tradeoff

Derived state:
  W: coupling matrix = X^T @ X (zero diagonal), lazy-computed from X

Dream cycle operates on X via proof-aligned operations (dream_cycle_xb).
W is never independently evolved — it's always derivable from the current X.

Privacy invariant: dream() NEVER accesses text content.
"""

from __future__ import annotations

import json
import math
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from dream_metrics import (
    capacity_utilization,
    count_spurious_attractors,
    measure_attractor_depth,
)
from dream_ops import (
    DreamParams,
    DreamReport,
    dream_cycle_xb,
    hopfield_update,
    hopfield_update_biased,
    spreading_activation,
    spreading_activation_biased,
)
from nlcdm_core import cosine_sim, sparsemax


# --- V1 Bridge Functions (Lean-proven in hermes-memory) ---
# Monotonicity chain proven in HermesNLCDM.MonotonicityChain:
#   time↑ → strength↓ → importance↓ → N↓ → capacity_margin↑
# See: proofs/hermes-memory/HermesMemory/StrengthDecay.lean
# See: proofs/hermes-memory/HermesMemory/MemoryDynamics.lean
# See: proofs/hermes-memory/HermesMemory/NoveltyBonus.lean


def _strength_decay(beta: float, S0: float, t: float) -> float:
    """S(t) = S₀ · e^(-β·t). Antitone in t (proven: strengthDecay_antitone)."""
    return S0 * np.exp(-beta * t)


def _importance_update(imp: float, delta: float, signal: float) -> float:
    """imp' = clamp(imp + δ·signal, 0, 1). Monotone in signal."""
    return max(0.0, min(imp + delta * signal, 1.0))


def _novelty_bonus(N0: float, gamma: float, t: float) -> float:
    """novelty(t) = N₀ · e^(-γ·t). Antitone in t (proven: noveltyBonus_antitone)."""
    return N0 * np.exp(-gamma * t)


@dataclass
class MemoryEntry:
    """A single memory in the coupled engine."""

    text: str
    embedding: np.ndarray
    importance: float = 0.5
    creation_time: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    tagged: bool = False
    recency: float = 0.0  # Serial number or store order; higher = newer
    layer: str = "user_knowledge"  # "user_knowledge", "agent_meta", "procedural"
    fact_type: str = "general"  # freeform, e.g. "finding", "reasoning_chain", "decision"


# Layer-aware importance seed map.
# Each cognitive layer has a default importance floor that determines
# dream survival (tagged threshold >= 0.7). This is a dream-survival
# mechanism, not a retrieval-ranking mechanism (Finding 1 from adversarial review).
_LAYER_IMPORTANCE = {
    "user_knowledge": 0.5,
    "agent_meta": 0.7,
    "procedural": 0.8,
}

# Fact-type refinements within a layer. These adjust the layer seed
# for specific high-value fact types. The fact_type takes precedence
# over the layer seed to allow fine-grained importance control.
_FACT_TYPE_IMPORTANCE_OVERRIDE = {
    "reasoning_chain": 0.75,
}


def _resolve_layer_importance(
    layer: str,
    fact_type: str,
    emotional_S_min: float = 0.3,
) -> float:
    """Resolve the base importance for a (layer, fact_type) pair.

    Lookup rule:
    1. If fact_type has an override, use that (e.g. reasoning_chain -> 0.75).
    2. Else use the layer seed (e.g. procedural -> 0.8).
    3. Unknown layers fall back to 0.5.

    For layer="user_knowledge", respects emotional_S_min if higher than the
    layer default (backward compat for callers setting emotional_S_min > 0.5).
    """
    if fact_type in _FACT_TYPE_IMPORTANCE_OVERRIDE:
        return _FACT_TYPE_IMPORTANCE_OVERRIDE[fact_type]
    base = _LAYER_IMPORTANCE.get(layer, 0.5)
    # For user_knowledge, respect engine's emotional_S_min if higher
    if layer == "user_knowledge":
        base = max(base, emotional_S_min)
    return base


class CoupledEngine:
    """Runtime engine wrapping NLCDM proofs and dream operations.

    W is (dim, dim) in embedding space — the same space that dream_ops
    functions operate on. Each store() adds a Hebbian outer-product
    contribution: W += emb ⊗ emb.
    """

    CAPACITY_RATIO = 0.138

    def __init__(
        self,
        dim: int,
        beta: float = 5.0,
        decay_rate: float = 0.01,
        importance_delta: float = 0.05,
        novelty_N0: float = 0.2,
        novelty_gamma: float = 0.05,
        emotional_tagging: bool = False,
        emotional_S_min: float = 0.3,
        reconsolidation: bool = False,
        reconsolidation_eta: float = 0.01,
        contradiction_aware: bool = False,
        contradiction_threshold: float = 0.85,
        recency_alpha: float = 0.0,
        dream_params: DreamParams | None = None,
        hebbian_epsilon: float = 0.01,
        ppr_blend_weight: float = 0.3,
        ppr_damping: float = 0.5,
    ):
        self.dim = dim
        self.beta = beta
        self.decay_rate = decay_rate
        self.importance_delta = importance_delta
        self.novelty_N0 = novelty_N0
        self.novelty_gamma = novelty_gamma
        self.emotional_tagging = emotional_tagging
        self.emotional_S_min = emotional_S_min
        self.reconsolidation = reconsolidation
        self.reconsolidation_eta = reconsolidation_eta
        self.contradiction_aware = contradiction_aware
        self.contradiction_threshold = contradiction_threshold
        self.recency_alpha = recency_alpha
        self.dream_params = dream_params
        self.memory_store: list[MemoryEntry] = []
        self.ppr_blend_weight = ppr_blend_weight
        self.ppr_damping = ppr_damping
        self._embeddings_cache: np.ndarray | None = None
        self._W_cache: np.ndarray | None = None
        # FAISS ANN index for O(1) nearest-neighbor search
        self._faiss_index = None  # faiss.IndexFlatIP, lazy init
        self.hebbian_epsilon = hebbian_epsilon
        self._session_buffer: list[np.ndarray] = []
        self._session_indices: list[int] = []  # indices of facts stored in current session
        self._co_occurrence: dict[int, dict[int, float]] = {}  # index -> {neighbor: weight}
        self._co_retrieval: dict[int, dict[int, float]] = {}  # co-retrieval graph (right invariance)
        self._co_retrieval_query_count: int = 0
        self._W_temporal = np.zeros((dim, dim), dtype=np.float64)

    @property
    def n_memories(self) -> int:
        return len(self.memory_store)

    @property
    def W(self) -> np.ndarray:
        """Coupling matrix derived from pattern store X.

        W = sum_i outer(x_i, x_i) with zero diagonal.
        Lazy-computed and cached; invalidated when X changes.
        """
        if self._W_cache is None:
            X = self._embeddings_matrix()
            if X.shape[0] == 0:
                self._W_cache = np.zeros((self.dim, self.dim), dtype=np.float64)
            else:
                self._W_cache = X.T @ X
                np.fill_diagonal(self._W_cache, 0.0)
        return self._W_cache

    def _embeddings_matrix(self) -> np.ndarray:
        """Return (N, dim) matrix of all stored embeddings."""
        if self._embeddings_cache is None:
            if self.n_memories == 0:
                self._embeddings_cache = np.zeros((0, self.dim), dtype=np.float64)
            else:
                self._embeddings_cache = np.array(
                    [m.embedding for m in self.memory_store], dtype=np.float64
                )
        return self._embeddings_cache

    def _invalidate_cache(self) -> None:
        self._embeddings_cache = None
        self._W_cache = None

    def _compute_effective_importance(self, m: MemoryEntry, now: float) -> float:
        """Compute time-decayed importance with novelty bonus.

        Bridge from V1 dynamics (Lean-proven monotonicity chain):
          effective = strength_decay(β, base_importance, Δt_access)
                    + novelty_bonus(N₀, γ, Δt_creation)

        The decay term ensures old unaccessed memories lose importance.
        The novelty term gives new memories a survival advantage.
        """
        dt_access = max(now - m.last_access_time, 0.0)
        dt_creation = max(now - m.creation_time, 0.0)
        decayed = _strength_decay(self.decay_rate, m.importance, dt_access)
        novelty = _novelty_bonus(self.novelty_N0, self.novelty_gamma, dt_creation)
        return min(max(decayed + novelty, 0.0), 1.0)

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------

    def _prediction_error(self, emb: np.ndarray) -> float:
        """Compute prediction error: cosine distance to nearest stored pattern.

        Returns a value in [0, 1] where:
          0 = identical to an existing pattern (fully predicted)
          1 = maximally different from all existing patterns (fully novel)

        Proven well-formed in EmotionalTagging.lean: prediction_error_well_formed.

        Uses FAISS index for O(1) lookup when available, falling back to
        matrix multiply for backward compatibility.
        """
        if self.n_memories == 0:
            return 1.0  # First memory is maximally novel

        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return 1.0

        # FAISS path: O(1) nearest-neighbor search
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            emb_normed = (emb / emb_norm).astype(np.float32).reshape(1, -1)
            D, _I = self._faiss_index.search(emb_normed, 1)
            max_sim = float(D[0, 0])
            return max(0.0, min(1.0 - max_sim, 1.0))

        # Fallback: matrix multiply
        X = self._embeddings_matrix()
        sims = X @ emb / (np.linalg.norm(X, axis=1) * emb_norm + 1e-12)
        max_sim = float(np.max(sims))
        # Cosine distance: 1 - max_similarity, clamped to [0, 1]
        return max(0.0, min(1.0 - max_sim, 1.0))

    def _emotional_S0(self, prediction_error: float) -> float:
        """Map prediction error to initial strength S₀.

        S₀ = S_min + (1 - S_min) · d

        Proven in EmotionalTagging.lean:
          - prediction_error_well_formed: S₀ ∈ [S_min, 1.0]
          - prediction_error_monotone: monotone in distance
          - emotional_bridge_safety_monotone: composes with bridge theorem
        """
        return self.emotional_S_min + (1.0 - self.emotional_S_min) * prediction_error

    def store(
        self,
        text: str,
        embedding: np.ndarray,
        importance: float | None = None,
        recency: float = 0.0,
        layer: str = "user_knowledge",
        fact_type: str = "general",
    ) -> int:
        """Add a new memory, optionally replacing contradictions.

        When contradiction_aware=True: if the new embedding has cosine
        similarity > contradiction_threshold with an existing pattern,
        the existing pattern is replaced (text, embedding, timestamps
        updated) and importance is set to max(old, new). This resolves
        version conflicts at store time rather than relying on dream.

        When emotional_tagging is enabled and importance is not explicitly
        provided, the importance is set based on prediction error (cosine
        distance to nearest existing pattern):
          - Novel patterns → high S₀ (stored hot)
          - Redundant patterns → low S₀ (stored cool, near S_min)

        Explicit importance always takes precedence over emotional tagging.
        """
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        # Importance resolution: layer-aware seeding with emotional tagging.
        # When importance is not explicitly provided, resolve from layer/fact_type.
        # Explicit importance always takes precedence over seeding.
        if importance is None:
            layer_base = _resolve_layer_importance(
                layer, fact_type, self.emotional_S_min
            )
            if self.emotional_tagging:
                pred_error = self._prediction_error(emb)
                importance = layer_base + (1.0 - layer_base) * pred_error
            else:
                importance = layer_base

        # L2-normalize for FAISS inner-product = cosine similarity
        emb_norm = np.linalg.norm(emb)
        emb_normed = emb / (emb_norm + 1e-12) if emb_norm > 1e-12 else emb

        # Lazy-init FAISS index
        if faiss is not None and self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(self.dim)

        # Contradiction detection: find the most similar existing pattern
        if self.contradiction_aware and self.n_memories > 0:
            best_idx = -1
            best_sim = -1.0

            if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                # FAISS path: O(1) nearest-neighbor search
                emb_f32 = emb_normed.astype(np.float32).reshape(1, -1)
                D, I = self._faiss_index.search(emb_f32, 1)
                best_sim = float(D[0, 0])
                best_idx = int(I[0, 0])
            elif emb_norm > 1e-12:
                # Fallback: matrix multiply
                X = self._embeddings_matrix()
                sims = X @ emb / (
                    np.linalg.norm(X, axis=1) * emb_norm + 1e-12
                )
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

            if best_idx >= 0 and best_sim > self.contradiction_threshold:
                # Replace: update text, embedding, timestamps;
                # keep the higher importance of old vs new.
                old = self.memory_store[best_idx]
                now = time.time()
                old.text = text
                old.embedding = emb
                old.importance = max(old.importance, importance)
                old.recency = max(old.recency, recency)
                # Preserve the more protective layer (Finding 5 mitigation):
                # if the old memory's layer has a higher seed, keep it.
                old_effective = _resolve_layer_importance(old.layer, old.fact_type, self.emotional_S_min)
                new_effective = _resolve_layer_importance(layer, fact_type, self.emotional_S_min)
                if new_effective >= old_effective:
                    old.layer = layer
                    old.fact_type = fact_type
                # else: keep old.layer (it provides higher protection)
                old.creation_time = now
                old.last_access_time = now
                old.access_count += 1
                old.tagged = old.importance >= 0.7
                self._invalidate_cache()
                # Update FAISS index: overwrite the old row with the new embedding
                if self._faiss_index is not None:
                    # Rebuild the FAISS index to reflect the replacement.
                    # For IndexFlatIP, the simplest correct approach is rebuild.
                    self._faiss_index.reset()
                    all_embs = np.array(
                        [m.embedding / (np.linalg.norm(m.embedding) + 1e-12)
                         for m in self.memory_store],
                        dtype=np.float32,
                    )
                    self._faiss_index.add(all_embs)
                self._session_buffer.append(emb.copy())
                self._session_indices.append(best_idx)
                return best_idx

        now = time.time()
        idx = self.n_memories
        entry = MemoryEntry(
            text=text,
            embedding=emb,
            importance=importance,
            creation_time=now,
            last_access_time=now,
            access_count=0,
            tagged=importance >= 0.7,
            recency=recency,
            layer=layer,
            fact_type=fact_type,
        )
        self.memory_store.append(entry)
        self._invalidate_cache()

        # Add the new embedding to the FAISS index
        if self._faiss_index is not None:
            emb_f32 = emb_normed.astype(np.float32).reshape(1, -1)
            self._faiss_index.add(emb_f32)

        self._session_buffer.append(emb.copy())
        self._session_indices.append(idx)
        return idx

    # ------------------------------------------------------------------
    # temporal Hebbian binding
    # ------------------------------------------------------------------

    def flush_session(self, epsilon: float | None = None) -> int:
        """Flush session buffer, creating Hebbian cross-links for all pairs.

        For N items in buffer, creates N*(N-1)/2 symmetric outer-product pairs
        in _W_temporal. Does NOT invalidate the embeddings/W cache since
        _W_temporal is separate primary state.

        Returns number of pairs bound.
        """
        buf = self._session_buffer
        n = len(buf)
        if n <= 1:
            self._session_buffer = []
            self._session_indices = []
            return 0

        eps = epsilon if epsilon is not None else self.hebbian_epsilon
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                self._W_temporal += eps * (
                    np.outer(buf[i], buf[j]) + np.outer(buf[j], buf[i])
                )
                pairs += 1

        # Record index-space co-occurrence for two-hop retrieval
        # Weight = 1/session_size: small sessions (bridge pairs) get strong links,
        # large sessions (background noise) get diluted links.
        indices = self._session_indices
        session_weight = 1.0 / max(len(indices), 1)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                neighbors_a = self._co_occurrence.setdefault(a, {})
                neighbors_b = self._co_occurrence.setdefault(b, {})
                # Accumulate weights (same pair in multiple sessions gets stronger)
                neighbors_a[b] = neighbors_a.get(b, 0.0) + session_weight
                neighbors_b[a] = neighbors_b.get(a, 0.0) + session_weight
        self._session_indices = []

        self._session_buffer = []
        return pairs

    def reset_temporal(self) -> None:
        """Reset temporal bindings to zero."""
        self._W_temporal = np.zeros((self.dim, self.dim), dtype=np.float64)
        self._session_buffer = []
        self._session_indices = []
        self._co_occurrence: dict[int, dict[int, float]] = {}
        self._co_retrieval: dict[int, dict[int, float]] = {}
        self._co_retrieval_query_count: int = 0

    @property
    def W_full(self) -> np.ndarray:
        """Combined coupling matrix: auto-derived W + temporal Hebbian W."""
        return self.W + self._W_temporal

    # ------------------------------------------------------------------
    # PPR scoring over co-retrieval graph
    # ------------------------------------------------------------------

    def _ppr_scores(
        self,
        seed_indices: np.ndarray,
        seed_scores: np.ndarray,
        alpha: float | None = None,
        n_iter: int = 20,
    ) -> np.ndarray:
        """Personalized PageRank over co-retrieval graph.

        alpha: damping factor (default: self.ppr_damping, 0.5 = HippoRAG default).
        Returns: np.ndarray of PPR scores for all patterns.
        """
        if alpha is None:
            alpha = self.ppr_damping

        N = self.n_memories
        if N == 0:
            return np.zeros(0)

        # Teleport vector: seed patterns weighted by their scores
        teleport = np.zeros(N)
        for idx, score in zip(seed_indices, seed_scores):
            if 0 <= idx < N:
                teleport[int(idx)] = score
        total = teleport.sum()
        if total > 0:
            teleport /= total
        else:
            return np.zeros(N)

        # PPR iteration
        ppr = teleport.copy()
        for _ in range(n_iter):
            new_ppr = (1 - alpha) * teleport
            for i in range(N):
                if i not in self._co_retrieval:
                    continue
                neighbors = self._co_retrieval[i]
                if not neighbors:
                    continue
                total_weight = sum(neighbors.values())
                if total_weight <= 0:
                    continue
                for j, w in neighbors.items():
                    if 0 <= j < N:
                        new_ppr[j] += alpha * ppr[i] * (w / total_weight)
            ppr = new_ppr

        return ppr

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve memories by direct cosine similarity + recency weighting.

        At high pattern counts (>100), spreading activation loses
        discriminative power — the converged state becomes an average
        over all patterns, giving uniform ~0.83 scores.  Direct cosine
        similarity preserves sharp discrimination.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Direct cosine similarity between query and each stored pattern
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cosine_scores = embeddings @ emb / norms

        # Phase 13 coherence switching (HybridSwitching.lean):
        # Multi-signal scoring (recency weighting) is only applied when
        # signal coherence > 0, i.e., when recency REINFORCES the cosine
        # ranking. When coherence ≤ 0 (recency contradicts cosine rank-1),
        # fall back to pure cosine to avoid rank inversion.
        scores = cosine_scores.copy()
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec  # [0, 1]
                multi_scores = cosine_scores * (
                    1.0 + self.recency_alpha * norm_rec
                )
                # Coherence check: does multi-signal preserve cosine rank-1?
                cosine_rank1 = int(np.argmax(cosine_scores))
                multi_rank1 = int(np.argmax(multi_scores))
                if cosine_rank1 == multi_rank1:
                    # Coherence > 0: multi-signal preserves ranking → use it
                    scores = multi_scores
                # else: coherence ≤ 0 → stay with cosine_scores (fallback)

        # PPR co-retrieval expansion (if graph has edges)
        has_co_retrieval = any(
            len(v) > 0 for v in self._co_retrieval.values()
        ) if self._co_retrieval else False

        if has_co_retrieval and N > 1:
            seed_k = min(5, N)
            seed_idx = np.argsort(cosine_scores)[::-1][:seed_k]
            seed_vals = cosine_scores[seed_idx]
            ppr = self._ppr_scores(seed_idx, seed_vals)

            # Blend: (1-w)*direct + w*ppr, normalized
            # Use 99th percentile to avoid single outlier dominating
            ppr_99 = np.percentile(ppr, 99) if N > 1 else ppr.max()
            if ppr_99 > 0:
                ppr_normalized = np.clip(ppr / ppr_99, 0.0, 1.0)
                blended = (
                    (1 - self.ppr_blend_weight) * scores
                    + self.ppr_blend_weight * ppr_normalized
                )
                # Hybrid switching guard (HybridSwitching.lean):
                # If PPR-boosted scores invert cosine rank-1, fall back to
                # pre-PPR scores to preserve the proven rank preservation.
                cosine_rank1 = int(np.argmax(cosine_scores))
                blended_rank1 = int(np.argmax(blended))
                if cosine_rank1 == blended_rank1:
                    scores = blended
                # else: PPR would invert rank-1 → keep pre-PPR scores

        k = min(top_k, N)
        top_indices = np.argsort(scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append(
                {
                    "index": int(i),
                    "score": float(scores[i]),
                    "text": self.memory_store[i].text,
                    "layer": self.memory_store[i].layer,
                    "fact_type": self.memory_store[i].fact_type,
                }
            )

        # Reconsolidation: ξ → ξ + η(q - ξ), then renormalize.
        # Patterns migrate toward queries that access them.
        # Safe: only strengthens through use, nothing gets starved.
        if self.reconsolidation:
            eta = self.reconsolidation_eta
            for i in top_indices:
                m = self.memory_store[i]
                old_emb = m.embedding
                m.embedding = old_emb + eta * (emb - old_emb)
                norm = np.linalg.norm(m.embedding)
                if norm > 1e-12:
                    m.embedding = m.embedding / norm
            self._invalidate_cache()

        # Co-retrieval edge logging: facts retrieved together form a clique.
        # Hebbian edge weight: strength = sqrt(score_i * score_j) * specificity
        # where specificity = 1/sqrt(k) penalizes large result sets.
        # This builds the "right invariance" graph — edges encode genuine
        # associative structure discovered through usage, not chunk boundaries.
        self._co_retrieval_query_count += 1
        k_eff = len(top_indices)
        specificity = 1.0 / math.sqrt(k_eff) if k_eff > 0 else 1.0
        for ii in range(k_eff):
            for jj in range(ii + 1, k_eff):
                a, b = int(top_indices[ii]), int(top_indices[jj])
                strength = math.sqrt(
                    max(float(scores[top_indices[ii]]), 0.0)
                    * max(float(scores[top_indices[jj]]), 0.0)
                )
                edge_delta = strength * specificity
                nbrs_a = self._co_retrieval.setdefault(a, {})
                nbrs_b = self._co_retrieval.setdefault(b, {})
                nbrs_a[b] = nbrs_a.get(b, 0.0) + edge_delta
                nbrs_b[a] = nbrs_b.get(a, 0.0) + edge_delta

        return results

    def query_readonly(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Read-only retrieval: returns results + logs co-retrieval, no side effects.

        Identical to query() except:
        - Does NOT mutate m.access_count, m.importance, m.last_access_time
        - Does NOT trigger reconsolidation (no embedding drift)
        - DOES log co-retrieval edges
        - DOES increment self._co_retrieval_query_count

        Use case: retrieval practice probes that build co-retrieval graph structure
        without corrupting longitudinal eval scoring (which depends on access_count
        and importance).

        Returns:
            List of dicts with keys: 'index', 'score', 'text'
            Same format as query().
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dimension {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Direct cosine similarity between query and each stored pattern
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cosine_scores = embeddings @ emb / norms

        # Phase 13 coherence switching (same logic as query())
        scores = cosine_scores.copy()
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                multi_scores = cosine_scores * (
                    1.0 + self.recency_alpha * norm_rec
                )
                cosine_rank1 = int(np.argmax(cosine_scores))
                multi_rank1 = int(np.argmax(multi_scores))
                if cosine_rank1 == multi_rank1:
                    scores = multi_scores

        k = min(top_k, N)
        top_indices = np.argsort(scores)[::-1][:k]

        # Build results WITHOUT mutating access_count, last_access_time, importance
        results = []
        for i in top_indices:
            results.append(
                {
                    "index": int(i),
                    "score": float(scores[i]),
                    "text": self.memory_store[i].text,
                    "layer": self.memory_store[i].layer,
                    "fact_type": self.memory_store[i].fact_type,
                }
            )

        # NO reconsolidation — no embedding drift in read-only mode

        # Co-retrieval edge logging: Hebbian weights (same formula as query())
        self._co_retrieval_query_count += 1
        k_eff = len(top_indices)
        specificity = 1.0 / math.sqrt(k_eff) if k_eff > 0 else 1.0
        for ii in range(k_eff):
            for jj in range(ii + 1, k_eff):
                a, b = int(top_indices[ii]), int(top_indices[jj])
                strength = math.sqrt(
                    max(float(scores[top_indices[ii]]), 0.0)
                    * max(float(scores[top_indices[jj]]), 0.0)
                )
                edge_delta = strength * specificity
                nbrs_a = self._co_retrieval.setdefault(a, {})
                nbrs_b = self._co_retrieval.setdefault(b, {})
                nbrs_a[b] = nbrs_a.get(b, 0.0) + edge_delta
                nbrs_b[a] = nbrs_b.get(a, 0.0) + edge_delta

        return results

    def query_associative(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        sparse: bool = False,
    ) -> list[dict]:
        """Retrieve via spreading activation: cosine + Hopfield association.

        Two-pass retrieval:
          1. Direct cosine similarity (same as query())
          2. Spreading activation: run query through Hopfield dynamics to find
             associated patterns that cosine alone misses
          3. Merge results, deduplicating by index, taking max score

        When sparse=True, uses sparsemax instead of softmax in the Hopfield
        dynamics. Sparsemax assigns exactly zero weight to irrelevant patterns,
        eliminating the blurring problem that causes dense Hopfield retrieval
        to fail (Hu et al., NeurIPS 2023).

        Useful for cross-domain queries where the answer requires connecting
        two facts with low mutual cosine similarity.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Pass 1: Direct cosine
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cosine_scores = embeddings @ emb / norms

        # Pass 2: Spreading activation — converge query in Hopfield landscape
        if np.any(self._W_temporal != 0):
            converged = spreading_activation_biased(
                beta=beta,
                patterns=embeddings,
                xi=emb / emb_norm,
                W_temporal=self._W_temporal,
                max_steps=20,
                attention_fn=sparsemax if sparse else None,
            )
        else:
            converged = spreading_activation(
                beta=beta,
                patterns=embeddings,
                xi=emb / emb_norm,
                max_steps=20,
                attention_fn=sparsemax if sparse else None,
            )
        conv_norm = np.linalg.norm(converged)
        if conv_norm > 1e-12:
            converged = converged / conv_norm
        spread_norms = np.linalg.norm(embeddings, axis=1) * conv_norm + 1e-12
        spread_scores = embeddings @ converged / spread_norms

        # Merge: take max of cosine and spread scores per index
        combined_scores = np.maximum(cosine_scores, spread_scores)

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                combined_scores = combined_scores * (
                    1.0 + self.recency_alpha * norm_rec
                )

        k = min(top_k, N)
        top_indices = np.argsort(combined_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append(
                {
                    "index": int(i),
                    "score": float(combined_scores[i]),
                    "text": self.memory_store[i].text,
                }
            )

        return results

    def query_hybrid(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Hybrid retrieval: cosine seeds → one Hopfield expansion step → union.

        Stage 1: Cosine retrieves top-K direct matches.
        Stage 2: Average the K seed embeddings into a probe vector.
                 Run ONE spreading activation step (not convergence) via
                 the coupling matrix X^T X. This spreads from seed patterns
                 toward associatively linked patterns that have low cosine
                 to the original query but high coupling to the retrieved set.
        Stage 3: Return the union of cosine top-K + expansion top-K, deduplicated.

        Uses sparsemax on the single step: zero weight on unrelated patterns,
        nonzero weight on the 5-10 patterns most coupled to the seed set.

        This avoids the convergence mush problem (one step only) and lets
        cosine do what it's good at (direct matching) while the coupling
        matrix does what IT's good at (finding associated patterns).
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Stage 1: Direct cosine — find the top-K seed matches
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1)
        cosine_scores = embeddings @ emb / (norms * emb_norm + 1e-12)

        k = min(top_k, N)
        cosine_top = np.argsort(cosine_scores)[::-1][:k]

        # Stage 2: Average seed embeddings into probe, one Hopfield step
        seed_embeddings = embeddings[cosine_top]  # (K, d)
        probe = seed_embeddings.mean(axis=0)      # (d,)
        probe_norm = np.linalg.norm(probe)
        if probe_norm > 1e-12:
            probe = probe / probe_norm

        # One step, not convergence — sparsemax keeps it sharp
        if np.any(self._W_temporal != 0):
            expanded = hopfield_update_biased(
                beta=beta,
                patterns=embeddings,
                xi=probe,
                W_temporal=self._W_temporal,
                attention_fn=sparsemax,
            )
        else:
            expanded = hopfield_update(
                beta=beta,
                patterns=embeddings,
                xi=probe,
                attention_fn=sparsemax,
            )
        exp_norm = np.linalg.norm(expanded)
        if exp_norm > 1e-12:
            expanded = expanded / exp_norm

        # Score all patterns against the expanded vector
        expansion_scores = embeddings @ expanded / (norms + 1e-12)
        expansion_top = np.argsort(expansion_scores)[::-1][:k]

        # Stage 3: Union — deduplicate, take max score per index
        seen = set()
        union_indices = []
        for idx in cosine_top:
            if idx not in seen:
                seen.add(idx)
                union_indices.append(idx)
        for idx in expansion_top:
            if idx not in seen:
                seen.add(idx)
                union_indices.append(idx)

        # Score: max(cosine, expansion) for each index
        combined_scores = np.maximum(cosine_scores, expansion_scores)

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                combined_scores = combined_scores * (
                    1.0 + self.recency_alpha * norm_rec
                )

        # Sort union by combined score, take top_k
        union_indices.sort(key=lambda i: combined_scores[i], reverse=True)
        union_indices = union_indices[:top_k]

        now = time.time()
        results = []
        for i in union_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append(
                {
                    "index": int(i),
                    "score": float(combined_scores[i]),
                    "text": self.memory_store[i].text,
                }
            )

        # Reconsolidation on accessed patterns
        if self.reconsolidation:
            eta = self.reconsolidation_eta
            top_indices = [r["index"] for r in results]
            for i in top_indices:
                m = self.memory_store[i]
                old_emb = m.embedding
                m.embedding = old_emb + eta * (emb - old_emb)
                norm = np.linalg.norm(m.embedding)
                if norm > 1e-12:
                    m.embedding = m.embedding / norm
            self._invalidate_cache()

        return results

    # ------------------------------------------------------------------
    # Phase 10: transfer retrieval via Hopfield update
    # ------------------------------------------------------------------

    def query_transfer(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        transfer_k: int | None = None,
    ) -> list[dict]:
        """Transfer retrieval: cosine + Hopfield update scoring.

        Uses the proven transfer bound (TransferDynamics.lean):
          cosine(T(ξ), target) ≥ σ / (1 + (N-1)·exp(-β·δ))

        where T(ξ) is the Hopfield update of query ξ, σ is the bridge-target
        similarity, and δ is the alignment gap of the bridge over other patterns.

        Steps:
          1. Compute direct cosine scores for all patterns.
          2. Run one Hopfield update: T(ξ) = Σ softmax(β·X^Tξ)_μ · x_μ
             If transfer_k is set, only use the top-transfer_k patterns by
             cosine for the Hopfield store (reduces N in the bound).
          3. Compute transfer scores: cosine(T(ξ), x_μ) for all μ.
          4. Combined score: max(cosine, transfer) per pattern.
          5. Return top-k by combined score.

        Args:
            embedding: query vector (d,)
            beta: inverse temperature (default: self.beta)
            top_k: number of results to return
            transfer_k: if set, prefilter to this many patterns for the
                Hopfield store. Reduces N in the theorem bound, giving
                stronger transfer through bridges. If None, use all patterns.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Step 1: Direct cosine scores
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        query_normed = emb / emb_norm
        norms = np.linalg.norm(embeddings, axis=1)
        cosine_scores = embeddings @ query_normed / (norms + 1e-12)

        # Step 2: Hopfield update — one step
        if transfer_k is not None and transfer_k < N:
            # Prefilter: use top-transfer_k patterns for the Hopfield store
            prefilter_idx = np.argsort(cosine_scores)[::-1][:transfer_k]
            hop_patterns = embeddings[prefilter_idx]
        else:
            hop_patterns = embeddings

        T_xi = hopfield_update(beta, hop_patterns, query_normed)
        T_norm = np.linalg.norm(T_xi)
        if T_norm < 1e-12:
            # Degenerate case: fall back to pure cosine
            combined_scores = cosine_scores
        else:
            T_normed = T_xi / T_norm
            # Step 3: Transfer scores — cosine of T(ξ) with all patterns
            transfer_scores = embeddings @ T_normed / (norms + 1e-12)
            # Step 4: Combined = max(cosine, transfer)
            combined_scores = np.maximum(cosine_scores, transfer_scores)

        # Step 5: Top-k by combined score
        k = min(top_k, N)
        top_indices = np.argsort(combined_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(combined_scores[i]),
                "text": m.text,
            })

        # Co-retrieval edge logging: Hebbian weights
        self._co_retrieval_query_count += 1
        k_eff = len(top_indices)
        specificity = 1.0 / math.sqrt(k_eff) if k_eff > 0 else 1.0
        for ii in range(k_eff):
            for jj in range(ii + 1, k_eff):
                a, b = int(top_indices[ii]), int(top_indices[jj])
                strength = math.sqrt(
                    max(float(combined_scores[top_indices[ii]]), 0.0)
                    * max(float(combined_scores[top_indices[jj]]), 0.0)
                )
                edge_delta = strength * specificity
                nbrs_a = self._co_retrieval.setdefault(a, {})
                nbrs_b = self._co_retrieval.setdefault(b, {})
                nbrs_a[b] = nbrs_a.get(b, 0.0) + edge_delta
                nbrs_b[a] = nbrs_b.get(a, 0.0) + edge_delta

        return results

    # ------------------------------------------------------------------
    # two-hop co-occurrence retrieval
    # ------------------------------------------------------------------

    def query_twohop(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        first_hop_k: int = 10,
        co_occurrence_bonus: float = 0.3,
    ) -> list[dict]:
        """Two-hop retrieval: cosine -> co-occurrence expansion -> score union.

        Stage 1: Cosine retrieves top-first_hop_k direct matches.
        Stage 2: Expand each hit through co-occurrence links to find
                 facts that were stored in the same session.
        Stage 3: Score the union by cosine similarity, return top-k.

        This enables cross-domain retrieval when bridge facts have low
        mutual cosine but were co-stored (same session/chunk).
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Stage 1: Direct cosine similarity
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        scores = embeddings @ emb / norms

        # First hop: top-k by cosine
        first_hop_indices = set(np.argsort(scores)[::-1][:first_hop_k].tolist())

        # Stage 2: Expand through co-occurrence
        expanded_indices = set(first_hop_indices)
        for idx in first_hop_indices:
            if idx in self._co_occurrence:
                expanded_indices.update(self._co_occurrence[idx].keys())

        # Transfer bound filter (TransferDynamics.lean):
        #   score_bound = 1 / (1 + (N-1) * exp(-beta * delta_min))
        # Expanded candidates below this bound are spurious retrievals.
        if len(expanded_indices) > len(first_hop_indices) and N > 1:
            embeddings_mat = self._embeddings_matrix()
            sq_norms = np.sum(embeddings_mat ** 2, axis=1)
            norms_all = np.sqrt(sq_norms)[:, None]
            normed = embeddings_mat / (norms_all + 1e-12)
            # Fast δ_min estimate: min cosine distance among first-hop patterns
            fh_list = sorted(first_hop_indices)
            if len(fh_list) >= 2:
                fh_normed = normed[fh_list]
                fh_cos = fh_normed @ fh_normed.T
                iu = np.triu_indices(len(fh_list), k=1)
                delta_min = float(np.min(1.0 - fh_cos[iu]))
                if delta_min <= 0:
                    delta_min = 0.01
                transfer_bound = 1.0 / (1.0 + (N - 1) * np.exp(-beta * delta_min))
                # Filter expanded-only candidates by their raw cosine score
                filtered_expanded = set(first_hop_indices)
                for idx in expanded_indices:
                    if idx in first_hop_indices:
                        continue
                    if float(scores[idx]) >= transfer_bound:
                        filtered_expanded.add(idx)
                expanded_indices = filtered_expanded

        # Stage 3: Score and rank the expanded set by cosine
        # Facts discovered via co-occurrence (not in first_hop_indices) get a
        # bonus so they survive the top-k cutoff -- they have low cosine to the
        # query by construction (that's the whole point of two-hop retrieval).
        expanded_list = sorted(expanded_indices)
        expanded_scores = []
        for idx in expanded_list:
            score = float(scores[idx])
            if idx not in first_hop_indices:
                score += co_occurrence_bonus
            expanded_scores.append((idx, score))
        expanded_scores.sort(key=lambda x: -x[1])

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [self.memory_store[idx].recency for idx, _ in expanded_scores],
                dtype=np.float64,
            )
            max_rec = recencies.max() if len(recencies) > 0 else 0.0
            if max_rec > 0.0:
                for i, (idx, score) in enumerate(expanded_scores):
                    norm_rec = recencies[i] / max_rec
                    expanded_scores[i] = (idx, score * (1.0 + self.recency_alpha * norm_rec))
                expanded_scores.sort(key=lambda x: -x[1])

        # Take top-k from expanded set
        k = min(top_k, len(expanded_scores))
        now = time.time()
        results = []
        for idx, score in expanded_scores[:k]:
            m = self.memory_store[idx]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(idx),
                "score": float(score),
                "text": m.text,
            })

        return results

    # ------------------------------------------------------------------
    # co-retrieval graph retrieval
    # ------------------------------------------------------------------

    def query_coretrieval(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        first_hop_k: int = 10,
        coretrieval_bonus: float = 0.3,
        min_coretrieval_count: float = 2.0,
    ) -> list[dict]:
        """Two-hop retrieval using co-retrieval graph instead of co-occurrence.

        Same algorithm as query_twohop but expands through co-retrieval edges
        (facts frequently retrieved together) instead of co-occurrence edges
        (facts stored in the same session/chunk).

        Co-retrieval encodes the right invariance: genuine associative structure
        discovered through usage patterns. Co-occurrence encodes the wrong
        invariance: chunk boundaries from the ingestion pipeline.

        min_coretrieval_count filters edges — only expand through pairs
        co-retrieved at least this many times. Prevents noise from
        single coincidental co-retrievals.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        scores = embeddings @ emb / norms

        first_hop_indices = set(np.argsort(scores)[::-1][:first_hop_k].tolist())

        # Expand through co-retrieval edges (filtered by min count)
        expanded_indices = set(first_hop_indices)
        for idx in first_hop_indices:
            neighbors = self._co_retrieval.get(idx, {})
            for neighbor, count in neighbors.items():
                if count >= min_coretrieval_count:
                    expanded_indices.add(neighbor)

        expanded_list = sorted(expanded_indices)
        expanded_scores = []
        for idx in expanded_list:
            score = float(scores[idx])
            if idx not in first_hop_indices:
                score += coretrieval_bonus
            expanded_scores.append((idx, score))
        expanded_scores.sort(key=lambda x: -x[1])

        if self.recency_alpha > 0.0:
            recencies = np.array(
                [self.memory_store[idx].recency for idx, _ in expanded_scores],
                dtype=np.float64,
            )
            max_rec = recencies.max() if len(recencies) > 0 else 0.0
            if max_rec > 0.0:
                for i, (idx, score) in enumerate(expanded_scores):
                    norm_rec = recencies[i] / max_rec
                    expanded_scores[i] = (idx, score * (1.0 + self.recency_alpha * norm_rec))
                expanded_scores.sort(key=lambda x: -x[1])

        k = min(top_k, len(expanded_scores))
        now = time.time()
        results = []
        for idx, score in expanded_scores[:k]:
            m = self.memory_store[idx]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(idx),
                "score": float(score),
                "text": m.text,
            })

        # Log co-retrieval edges from this query too: Hebbian weights
        top_result_indices = [r["index"] for r in results]
        top_result_scores = [r["score"] for r in results]
        self._co_retrieval_query_count += 1
        k_eff = len(top_result_indices)
        specificity = 1.0 / math.sqrt(k_eff) if k_eff > 0 else 1.0
        for ii in range(k_eff):
            for jj in range(ii + 1, k_eff):
                a, b = top_result_indices[ii], top_result_indices[jj]
                strength = math.sqrt(
                    max(top_result_scores[ii], 0.0)
                    * max(top_result_scores[jj], 0.0)
                )
                edge_delta = strength * specificity
                nbrs_a = self._co_retrieval.setdefault(a, {})
                nbrs_b = self._co_retrieval.setdefault(b, {})
                nbrs_a[b] = nbrs_a.get(b, 0.0) + edge_delta
                nbrs_b[a] = nbrs_b.get(a, 0.0) + edge_delta

        return results

    def query_cooc_boost(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        cooc_weight: float = 0.3,
        gate_threshold: float = 0.0,
    ) -> list[dict]:
        """Full-store co-occurrence boost: every memory gets credit from neighbors.

        Unlike query_twohop which only expands from the top-k cosine hits,
        this method scores co-occurrence signal across the ENTIRE store.

        For each memory i:
          final_score[i] = cosine_score[i]
                         + cooc_weight * mean(cosine_score[j] for j in cooc[i])

        This means fact_B at cosine rank 81 still gets a boost if its
        co-occurrence partner fact_A scored 0.60, even though neither
        is in the cosine top-10.

        Confidence gating: if gate_threshold > 0 and the top cosine score
        exceeds the threshold, skip the co-occurrence boost entirely and
        return pure cosine results. This prevents hubness noise from
        overriding confident retrievals at scale.

        Complexity: O(N + sum of co-occurrence degrees) — cheap for sparse graphs.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Cosine scores for ALL memories
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cos_scores = embeddings @ emb / norms

        # Confidence gating: if top cosine score is high enough, skip boost
        if gate_threshold > 0.0 and cos_scores.max() >= gate_threshold:
            final_scores = cos_scores.copy()
        else:
            # Co-occurrence boost: each memory gets weighted mean cosine of neighbors.
            # Edge weights encode session size — bridge pairs (session of 2) get
            # weight 0.5 per partner, while background facts in sessions of 20 get
            # weight 0.05 each. This naturally prioritizes co-occurrence from small,
            # focused sessions.
            final_scores = cos_scores.copy()
            for idx in range(N):
                neighbors = self._co_occurrence.get(idx, {})
                if neighbors:
                    weighted_sum = sum(
                        w * cos_scores[j] for j, w in neighbors.items() if j < N
                    )
                    total_weight = sum(
                        w for j, w in neighbors.items() if j < N
                    )
                    if total_weight > 0:
                        final_scores[idx] += cooc_weight * weighted_sum / total_weight

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                final_scores = final_scores * (1.0 + self.recency_alpha * norm_rec)

        k = min(top_k, N)
        top_indices = np.argsort(final_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(final_scores[i]),
                "text": m.text,
            })

        return results

    def query_cooc_sparse_topk(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        cooc_weight: float = 0.3,
        neighbor_k: int = 5,
    ) -> list[dict]:
        """Sparse top-k neighbor co-occurrence boost.

        Like query_cooc_boost but only considers the top-k strongest
        co-occurrence edges per memory (by edge weight). This prevents
        the hubness problem at scale where averaging many weak neighbors
        washes out discriminative signal.

        Complexity: O(N * neighbor_k) — still cheap.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cos_scores = embeddings @ emb / norms

        final_scores = cos_scores.copy()
        for idx in range(N):
            neighbors = self._co_occurrence.get(idx, {})
            if neighbors:
                # Keep only top-k neighbors by edge weight
                valid = [(j, w) for j, w in neighbors.items() if j < N]
                valid.sort(key=lambda x: -x[1])
                top_neighbors = valid[:neighbor_k]
                if top_neighbors:
                    weighted_sum = sum(w * cos_scores[j] for j, w in top_neighbors)
                    total_weight = sum(w for _, w in top_neighbors)
                    if total_weight > 0:
                        final_scores[idx] += cooc_weight * weighted_sum / total_weight

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                final_scores = final_scores * (1.0 + self.recency_alpha * norm_rec)

        k = min(top_k, N)
        top_indices = np.argsort(final_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(final_scores[i]),
                "text": m.text,
            })

        return results

    def query_cooc_sparsemax(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        cooc_weight: float = 0.3,
    ) -> list[dict]:
        """Sparsemax-attention co-occurrence boost.

        Instead of averaging neighbor cosine scores uniformly, applies
        sparsemax over the neighbor cosine scores to produce sparse
        attention weights. Only the highest-scoring neighbors contribute
        non-zero weight, killing the hubness noise from dense averaging.

        Based on Hu et al. (NeurIPS 2023) sparse modern Hopfield model:
        sparsemax assigns exactly zero weight to irrelevant patterns.

        Complexity: O(N * max_degree * log(max_degree)) due to sparsemax sort.
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cos_scores = embeddings @ emb / norms

        final_scores = cos_scores.copy()
        for idx in range(N):
            neighbors = self._co_occurrence.get(idx, {})
            if neighbors:
                valid_js = [j for j in neighbors if j < N]
                if valid_js:
                    neighbor_cos = np.array([cos_scores[j] for j in valid_js])
                    # Sparsemax: produces sparse attention weights over neighbors
                    attn = sparsemax(neighbor_cos)
                    # Weighted sum with sparse attention (most weights are zero)
                    boost = float(attn @ neighbor_cos)
                    final_scores[idx] += cooc_weight * boost

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                final_scores = final_scores * (1.0 + self.recency_alpha * norm_rec)

        k = min(top_k, N)
        top_indices = np.argsort(final_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(final_scores[i]),
                "text": m.text,
            })

        return results

    def query_ppr(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        damping: float = 0.85,
        ppr_steps: int = 10,
        ppr_weight: float = 0.5,
    ) -> list[dict]:
        """Personalized PageRank on co-occurrence graph seeded by cosine.

        Propagates retrieval signal through the co-occurrence graph with
        geometric decay. Multi-hop paths are naturally handled.

        Algorithm:
          1. Seed vector s = softmax(cosine_scores * temperature)
          2. PPR iteration: r = (1-d)*s + d*A*r
             where A is row-normalized co-occurrence adjacency
          3. Final score = (1 - ppr_weight) * cosine + ppr_weight * r

        Inspired by HippoRAG (NeurIPS 2024).
        """
        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Cosine scores
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cos_scores = embeddings @ emb / norms

        # Build seed vector: softmax of cosine scores
        # Temperature scales discrimination — higher = sharper
        temperature = 5.0
        exp_scores = np.exp(temperature * (cos_scores - cos_scores.max()))
        seed = exp_scores / exp_scores.sum()

        # Build row-normalized adjacency matrix (sparse, as dict)
        # Uses co-occurrence edge weights, then row-normalizes for diffusion.
        adj_weights: dict[int, dict[int, float]] = {}
        for idx in range(N):
            neighbors = self._co_occurrence.get(idx, {})
            valid = {j: w for j, w in neighbors.items() if j < N}
            if valid:
                total = sum(valid.values())
                adj_weights[idx] = {j: w / total for j, w in valid.items()}

        # PPR iteration: r = (1-d)*seed + d*A@r
        r = seed.copy()
        for _ in range(ppr_steps):
            r_new = (1.0 - damping) * seed
            for idx in range(N):
                if idx in adj_weights:
                    for j, w in adj_weights[idx].items():
                        r_new[idx] += damping * w * r[j]
            r = r_new

        # Normalize PPR scores to [0, 1] range
        r_max = r.max()
        if r_max > 1e-12:
            r_normed = r / r_max
        else:
            r_normed = r

        # Final combined score
        final_scores = (1.0 - ppr_weight) * cos_scores + ppr_weight * r_normed

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                final_scores = final_scores * (1.0 + self.recency_alpha * norm_rec)

        k = min(top_k, N)
        top_indices = np.argsort(final_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(final_scores[i]),
                "text": m.text,
            })

        return results

    def query_quantum_walk(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
        walk_time: float = 1.0,
        qw_weight: float = 0.5,
    ) -> list[dict]:
        """Quantum walk on co-occurrence graph for interference-based retrieval.

        Replaces classical diffusion (PageRank, co-occurrence boost) with
        unitary evolution exp(-iAt) on the co-occurrence adjacency matrix.
        Classical diffusion converges to degree-proportional equilibrium
        (hubness). Quantum walk oscillates — paths to hub nodes destructively
        interfere (many incoherent directions), paths to bridge partners
        constructively interfere (direct coherent path).

        This is the Wick rotation: t → it transforms the heat equation
        (diffusion, spreading, equilibrium) into the Schrödinger equation
        (oscillation, interference, discrimination).

        Algorithm:
          1. Compute cosine similarity vector q (query vs all facts)
          2. Build sparse co-occurrence adjacency matrix A
          3. Evolve quantum state: ψ = exp(-i·A·t) @ q
          4. Born rule: probability = |ψ|²
          5. Final score = (1 - qw_weight) * cosine + qw_weight * |ψ|²

        walk_time controls the propagation depth. Too small ≈ identity (just
        cosine). Too large = rapid oscillation. Sweet spot is related to the
        inverse spectral gap of A.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import expm_multiply

        if beta is None:
            beta = self.beta

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {emb.shape[0]} != engine dim {self.dim}"
            )

        N = self.n_memories
        if N == 0:
            return []

        embeddings = self._embeddings_matrix()

        # Cosine scores
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return []
        norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
        cos_scores = embeddings @ emb / norms

        # Build sparse adjacency matrix from co-occurrence dict
        rows, cols, vals = [], [], []
        for idx in range(N):
            neighbors = self._co_occurrence.get(idx, {})
            for j, w in neighbors.items():
                if j < N:
                    rows.append(idx)
                    cols.append(j)
                    vals.append(w)

        if not rows:
            # No co-occurrence edges — fall back to pure cosine
            final_scores = cos_scores.copy()
        else:
            A = csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))),
                shape=(N, N),
            )

            # Quantum walk: ψ = exp(-i·A·t) @ q
            # expm_multiply computes this via Krylov subspace — never forms
            # the full matrix exponential. O(nnz * krylov_dim) per call.
            psi = expm_multiply(-1j * A * walk_time, cos_scores.astype(complex))

            # Born rule: probability amplitudes → scores
            qw_scores = np.abs(psi) ** 2

            # Normalize to same scale as cosine for clean blending
            qw_max = qw_scores.max()
            if qw_max > 1e-12:
                qw_scores = qw_scores * (cos_scores.max() / qw_max)

            final_scores = (1.0 - qw_weight) * cos_scores + qw_weight * qw_scores

        # Recency weighting
        if self.recency_alpha > 0.0:
            recencies = np.array(
                [m.recency for m in self.memory_store], dtype=np.float64
            )
            max_rec = recencies.max()
            if max_rec > 0.0:
                norm_rec = recencies / max_rec
                final_scores = final_scores * (1.0 + self.recency_alpha * norm_rec)

        k = min(top_k, N)
        top_indices = np.argsort(final_scores)[::-1][:k]

        now = time.time()
        results = []
        for i in top_indices:
            m = self.memory_store[i]
            m.access_count += 1
            m.last_access_time = now
            m.importance = _importance_update(
                m.importance, self.importance_delta, 1.0
            )
            results.append({
                "index": int(i),
                "score": float(final_scores[i]),
                "text": m.text,
            })

        return results

    # ------------------------------------------------------------------
    # capacity — Lean-aligned capacity computation (Capacity.lean)
    # ------------------------------------------------------------------

    def _compute_capacity_ratio(
        self, embeddings: np.ndarray, max_sample: int = 500
    ) -> float:
        """Compute capacity utilization ratio N / N_max.

        Uses the Lean-proven formula (Capacity.lean, theorem
        capacity_criterion):

            4·N·β·M² ≤ exp(β·δ)  ⟹  N_max = exp(β·δ) / (4·β·M²)

        where δ is the minimum pairwise separation (cosine distance)
        and M² is the maximum squared pattern norm.

        Returns capacity_ratio = N / N_max.  Values < 1.0 mean the store
        has room; values > 1.0 mean the store is at or above capacity.
        """
        N = embeddings.shape[0]
        if N <= 1:
            return 0.0

        # M² = max squared norm across all patterns
        sq_norms = np.sum(embeddings ** 2, axis=1)
        M_sq = float(np.max(sq_norms))
        if M_sq < 1e-12:
            return 0.0

        # Normalize for cosine distance computation
        norms = np.sqrt(sq_norms)[:, None]
        normed = embeddings / (norms + 1e-12)

        # Compute δ_min = minimum pairwise cosine distance
        if N <= max_sample:
            cos_matrix = normed @ normed.T
            iu = np.triu_indices(N, k=1)
            pairwise_cos = cos_matrix[iu]
        else:
            rng = np.random.default_rng(0)
            n_pairs = max_sample * (max_sample - 1) // 2
            idx_a = rng.integers(0, N, size=n_pairs)
            idx_b = rng.integers(0, N, size=n_pairs)
            mask = idx_a != idx_b
            idx_a, idx_b = idx_a[mask], idx_b[mask]
            pairwise_cos = np.sum(
                normed[idx_a] * normed[idx_b], axis=1
            )

        delta_min = float(np.min(1.0 - pairwise_cos))
        if delta_min <= 0:
            delta_min = 0.01  # floor to avoid exp(0) degeneracy

        # Lean formula: N_max = exp(β·δ) / (4·β·M²)
        beta = self.beta
        n_max = np.exp(beta * delta_min) / (4.0 * beta * M_sq)
        if n_max < 1.0:
            n_max = 1.0

        return float(N) / n_max

    def should_dream(self, low: float = 0.5, high: float = float("inf")) -> bool:
        """Check if dreaming would be useful based on capacity utilization.

        The Lean bound (N_max = exp(βδ)/(4βM²)) is a conservative worst-case
        guarantee dominated by the closest pattern pair. Real stores routinely
        exceed it. Dream consolidation helps whenever utilization is above a
        minimum threshold — there is no practical upper cutoff because
        exceeding the Lean bound doesn't mean patterns have collapsed, only
        that the proof guarantee no longer holds.

        Returns True when utilization >= low (default 0.5).
        """
        N = self.n_memories
        if N <= 1:
            return False
        embeddings = self._embeddings_matrix()
        utilization = self._compute_capacity_ratio(embeddings)
        return bool(low <= utilization <= high)

    # ------------------------------------------------------------------
    # dream — proof-aligned {X, β} dream cycle
    # ------------------------------------------------------------------

    def dream(
        self,
        tagged_indices: list[int] | None = None,
        seed: int | None = None,
        dream_params: DreamParams | None = None,
    ) -> dict:
        """Run proof-aligned dream cycle on {X, beta}. NEVER accesses text.

        Runs the redesigned dream pipeline:
          1. nrem_repulsion_xb (SHY: push apart close patterns)
          2. nrem_prune_xb (remove near-duplicates)
          3. nrem_merge_xb (merge similar groups into centroids)
          4. rem_unlearn_xb (destabilize mixture states)
          5. rem_explore_cross_domain_xb (cross-domain associations)

        Capacity-aware gating (Phase 14, Capacity.lean):
          When N/N_max < 0.8, prune/merge thresholds are tightened to 0.99
          so only exact duplicates are removed.  This prevents premature
          consolidation when the network has ample room.

        Modifies the pattern store X. Memory count may decrease (prune/merge).
        W is derived, not primary.
        """
        N = self.n_memories
        if N == 0:
            return {
                "modified": False, "associations": [], "n_tagged": 0,
                "pruned": 0, "merged": 0, "n_before": 0, "n_after": 0,
                "capacity_ratio": 0.0,
            }

        # Compute effective importances via V1 bridge (decay + novelty)
        now = time.time()
        importances = np.array([
            self._compute_effective_importance(m, now)
            for m in self.memory_store
        ])

        # Auto-tag: uses effective importance, NOT text
        if tagged_indices is None:
            tagged_indices = [
                i for i, m in enumerate(self.memory_store)
                if importances[i] >= 0.7
            ]
        if not tagged_indices:
            tagged_indices = list(range(N))

        embeddings = self._embeddings_matrix()

        # --- Phase 14: Capacity-aware dream gating (Capacity.lean) ---
        capacity_ratio = self._compute_capacity_ratio(embeddings)

        # Use explicit dream_params if provided, else fall back to stored default
        effective_params = dream_params or self.dream_params or DreamParams()

        # When well below capacity, tighten thresholds to only prune
        # exact duplicates (cosine > 0.99).  This prevents premature
        # consolidation that removes patterns the eval still needs.
        CAPACITY_GATE = 0.8
        NEAR_DUP_PRUNE = 0.995   # Only prune exact duplicates
        NEAR_DUP_MERGE = 0.99    # Only merge near-exact duplicates
        if capacity_ratio < CAPACITY_GATE:
            effective_params = dataclasses.replace(
                effective_params,
                prune_threshold=max(effective_params.prune_threshold, NEAR_DUP_PRUNE),
                merge_threshold=max(effective_params.merge_threshold, NEAR_DUP_MERGE),
            )

        report: DreamReport = dream_cycle_xb(
            embeddings, self.beta,
            tagged_indices=tagged_indices,
            importances=importances,
            seed=seed,
            params=effective_params,
        )

        # --- Decay co-retrieval edges (one dream cycle worth of decay) ---
        decay_factor = _strength_decay(self.decay_rate, 1.0, 1.0)
        for idx in list(self._co_retrieval.keys()):
            for nbr in list(self._co_retrieval[idx].keys()):
                self._co_retrieval[idx][nbr] *= decay_factor
                if self._co_retrieval[idx][nbr] < 0.01:
                    del self._co_retrieval[idx][nbr]
            # Clean up empty neighbor dicts
            if not self._co_retrieval[idx]:
                del self._co_retrieval[idx]

        # --- Apply structural changes ---

        pruned_set = set(report.pruned_indices)
        merged_originals: set[int] = set()
        for group in report.merge_map.values():
            merged_originals.update(group)

        # Build new memory store
        new_store: list[MemoryEntry] = []

        # Pass 1: keep non-pruned, non-merged entries with updated embeddings
        output_idx = 0
        for orig_idx in range(N):
            if orig_idx in pruned_set or orig_idx in merged_originals:
                continue
            entry = self.memory_store[orig_idx]
            entry.embedding = report.patterns[output_idx]
            new_store.append(entry)
            output_idx += 1

        # Pass 2: add merged centroid entries
        for out_idx, group in sorted(report.merge_map.items()):
            best_orig = max(group, key=lambda i: self.memory_store[i].importance)
            merged_entry = MemoryEntry(
                text=self.memory_store[best_orig].text,
                embedding=report.patterns[out_idx],
                importance=min(
                    max(importances[g] for g in group) + 0.1, 1.0,
                ),
                creation_time=min(
                    self.memory_store[g].creation_time for g in group
                ),
                last_access_time=max(self.memory_store[g].last_access_time for g in group),
                access_count=sum(
                    self.memory_store[g].access_count for g in group
                ),
                recency=self.memory_store[best_orig].recency,
                tagged=True,
                layer=self.memory_store[best_orig].layer,
                fact_type=self.memory_store[best_orig].fact_type,
            )
            new_store.append(merged_entry)

        self.memory_store = new_store
        self._invalidate_cache()

        # Rebuild FAISS index to match the new memory_store
        if self._faiss_index is not None:
            self._faiss_index.reset()
            if self.memory_store:
                all_embs = np.array(
                    [m.embedding / (np.linalg.norm(m.embedding) + 1e-12)
                     for m in self.memory_store],
                    dtype=np.float32,
                )
                self._faiss_index.add(all_embs)

        # --- Log dream associations into co-retrieval graph ---
        if report.associations:
            # Build X4-index -> new_store-index remap
            # Pass 1 survivors: X4[0..k-1] -> new_store[0..k-1] (identity)
            # Pass 2 centroids: X4[merge_key] -> new_store[k + position]
            n_survivors = sum(
                1 for i in range(N)
                if i not in pruned_set and i not in merged_originals
            )
            x4_to_new: dict[int, int] = {}
            # Survivors are identity-mapped
            for idx in range(n_survivors):
                x4_to_new[idx] = idx
            # Centroids follow survivors
            for pos, merge_key in enumerate(sorted(report.merge_map.keys())):
                x4_to_new[merge_key] = n_survivors + pos

            for ai, aj, sim in report.associations:
                ni = x4_to_new.get(ai)
                nj = x4_to_new.get(aj)
                if ni is not None and nj is not None and ni != nj:
                    nbrs_ni = self._co_retrieval.setdefault(ni, {})
                    nbrs_nj = self._co_retrieval.setdefault(nj, {})
                    # Use similarity as weight (not binary 1.0)
                    nbrs_ni[nj] = max(nbrs_ni.get(nj, 0.0), sim)
                    nbrs_nj[ni] = max(nbrs_nj.get(ni, 0.0), sim)

        return {
            "modified": True,
            "n_tagged": len(tagged_indices),
            "associations": report.associations,
            "pruned": len(report.pruned_indices),
            "merged": len(report.merge_map),
            "n_before": N,
            "n_after": len(new_store),
            "capacity_ratio": capacity_ratio,
        }

    # ------------------------------------------------------------------
    # tag
    # ------------------------------------------------------------------

    def tag(self, index: int, importance: float) -> None:
        if index < 0 or index >= self.n_memories:
            raise IndexError(
                f"Memory index {index} out of range [0, {self.n_memories})"
            )
        self.memory_store[index].importance = importance
        self.memory_store[index].tagged = importance >= 0.7

    # ------------------------------------------------------------------
    # get_metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        N = self.n_memories
        if N == 0:
            return {
                "memory_count": 0,
                "spurious_count": 0,
                "mean_attractor_depth": 0.0,
                "capacity_utilization": 0.0,
            }

        embeddings = self._embeddings_matrix()
        spurious = count_spurious_attractors(self.W, embeddings, self.beta)

        depths = [
            measure_attractor_depth(self.W, embeddings, self.beta, i)
            for i in range(N)
        ]
        mean_depth = float(np.mean(depths))

        cap = capacity_utilization(self.W, embeddings, self.beta)

        return {
            "memory_count": N,
            "spurious_count": spurious,
            "mean_attractor_depth": mean_depth,
            "capacity_utilization": cap,
        }

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize to {path}.npz + {path}.json.

        Only saves X (embeddings) — W is derived from X on load.
        """
        path = Path(path)

        embeddings = self._embeddings_matrix()
        np.savez(
            str(path) + ".npz",
            embeddings=embeddings,
            dim=np.array([self.dim]),
            beta=np.array([self.beta]),
            W_temporal=self._W_temporal,
        )

        metadata = {
            "dim": self.dim,
            "beta": self.beta,
            "decay_rate": self.decay_rate,
            "importance_delta": self.importance_delta,
            "novelty_N0": self.novelty_N0,
            "novelty_gamma": self.novelty_gamma,
            "hebbian_epsilon": self.hebbian_epsilon,
            "co_occurrence": {
                str(k): {str(j): w for j, w in v.items()}
                for k, v in self._co_occurrence.items()
            },
            "co_retrieval": {
                str(k): {str(j): w for j, w in v.items()}
                for k, v in self._co_retrieval.items()
            },
            "co_retrieval_query_count": self._co_retrieval_query_count,
            "memories": [
                {
                    "text": m.text,
                    "importance": m.importance,
                    "creation_time": m.creation_time,
                    "last_access_time": m.last_access_time,
                    "access_count": m.access_count,
                    "tagged": m.tagged,
                    "layer": m.layer,
                    "fact_type": m.fact_type,
                }
                for m in self.memory_store
            ],
        }
        with open(str(path) + ".json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CoupledEngine:
        """Deserialize from {path}.npz + {path}.json.

        W is derived from X on access, not loaded.
        """
        path = Path(path)

        data = np.load(str(path) + ".npz")
        embeddings = data["embeddings"]
        dim = int(data["dim"][0])
        beta = float(data["beta"][0])

        with open(str(path) + ".json") as f:
            metadata = json.load(f)

        engine = cls(
            dim=dim,
            beta=beta,
            decay_rate=metadata.get("decay_rate", 0.01),
            importance_delta=metadata.get("importance_delta", 0.05),
            novelty_N0=metadata.get("novelty_N0", 0.0),
            novelty_gamma=metadata.get("novelty_gamma", 0.1),
        )

        for i, meta in enumerate(metadata["memories"]):
            entry = MemoryEntry(
                text=meta["text"],
                embedding=embeddings[i],
                importance=meta["importance"],
                creation_time=meta["creation_time"],
                last_access_time=meta.get("last_access_time", meta["creation_time"]),
                access_count=meta["access_count"],
                tagged=meta["tagged"],
                layer=meta.get("layer", "user_knowledge"),
                fact_type=meta.get("fact_type", "general"),
            )
            engine.memory_store.append(entry)

        engine._invalidate_cache()
        engine.hebbian_epsilon = metadata.get("hebbian_epsilon", 0.01)
        if "W_temporal" in data:
            engine._W_temporal = data["W_temporal"]
        else:
            engine._W_temporal = np.zeros((dim, dim), dtype=np.float64)

        # Restore co-occurrence links (weighted edges)
        co_occur_raw = metadata.get("co_occurrence", {})
        engine._co_occurrence = {}
        for k, v in co_occur_raw.items():
            if isinstance(v, dict):
                engine._co_occurrence[int(k)] = {int(j): float(w) for j, w in v.items()}
            else:
                # Backward compat: old format was {k: [list of indices]}
                engine._co_occurrence[int(k)] = {int(j): 1.0 for j in v}

        # Restore co-retrieval graph
        coret_raw = metadata.get("co_retrieval", {})
        engine._co_retrieval = {}
        for k, v in coret_raw.items():
            engine._co_retrieval[int(k)] = {int(j): float(w) for j, w in v.items()}
        engine._co_retrieval_query_count = metadata.get("co_retrieval_query_count", 0)

        return engine
