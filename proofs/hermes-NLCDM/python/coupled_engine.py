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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dream_metrics import (
    capacity_utilization,
    count_spurious_attractors,
    measure_attractor_depth,
)
from dream_ops import (
    DreamReport,
    dream_cycle_xb,
    spreading_activation,
)
from nlcdm_core import cosine_sim


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
        self.memory_store: list[MemoryEntry] = []
        self._embeddings_cache: np.ndarray | None = None
        self._W_cache: np.ndarray | None = None

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
        """
        if self.n_memories == 0:
            return 1.0  # First memory is maximally novel
        X = self._embeddings_matrix()
        # Cosine similarities: emb · X^T (both unit-normed or close)
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 1e-12:
            return 1.0
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
        self, text: str, embedding: np.ndarray, importance: float | None = None
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

        # Emotional tagging: compute S₀ from prediction error when no
        # explicit importance was provided
        if importance is None:
            if self.emotional_tagging:
                pred_error = self._prediction_error(emb)
                importance = self._emotional_S0(pred_error)
            else:
                importance = 0.5

        # Contradiction detection: find the most similar existing pattern
        if self.contradiction_aware and self.n_memories > 0:
            X = self._embeddings_matrix()
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 1e-12:
                sims = X @ emb / (
                    np.linalg.norm(X, axis=1) * emb_norm + 1e-12
                )
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                if best_sim > self.contradiction_threshold:
                    # Replace: update text, embedding, timestamps;
                    # keep the higher importance of old vs new.
                    old = self.memory_store[best_idx]
                    now = time.time()
                    old.text = text
                    old.embedding = emb
                    old.importance = max(old.importance, importance)
                    old.creation_time = now
                    old.last_access_time = now
                    old.access_count += 1
                    old.tagged = old.importance >= 0.7
                    self._invalidate_cache()
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
        )
        self.memory_store.append(entry)
        self._invalidate_cache()

        return idx

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: np.ndarray,
        beta: float | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve memories via spreading activation.

        Uses spreading_activation from dream_ops (modern Hopfield dynamics
        in pattern space), then ranks stored patterns by cosine overlap
        with the converged state.
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

        # Spreading activation: converge in pattern space
        x_conv = spreading_activation(beta, embeddings, emb)

        # Score each pattern by cosine similarity with converged state
        scores = np.array([cosine_sim(x_conv, embeddings[i]) for i in range(N)])

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

        return results

    # ------------------------------------------------------------------
    # dream — proof-aligned {X, β} dream cycle
    # ------------------------------------------------------------------

    def dream(
        self,
        tagged_indices: list[int] | None = None,
        seed: int | None = None,
    ) -> dict:
        """Run proof-aligned dream cycle on {X, beta}. NEVER accesses text.

        Runs the redesigned dream pipeline:
          1. nrem_repulsion_xb (SHY: push apart close patterns)
          2. nrem_prune_xb (remove near-duplicates)
          3. nrem_merge_xb (merge similar groups into centroids)
          4. rem_unlearn_xb (destabilize mixture states)
          5. rem_explore_cross_domain_xb (cross-domain associations)

        Modifies the pattern store X. Memory count may decrease (prune/merge).
        W is derived, not primary.
        """
        N = self.n_memories
        if N == 0:
            return {
                "modified": False, "associations": [], "n_tagged": 0,
                "pruned": 0, "merged": 0, "n_before": 0, "n_after": 0,
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

        report: DreamReport = dream_cycle_xb(
            embeddings, self.beta,
            tagged_indices=tagged_indices,
            importances=importances,
            seed=seed,
        )

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
                access_count=sum(
                    self.memory_store[g].access_count for g in group
                ),
                tagged=True,
            )
            new_store.append(merged_entry)

        self.memory_store = new_store
        self._invalidate_cache()

        return {
            "modified": True,
            "n_tagged": len(tagged_indices),
            "associations": report.associations,
            "pruned": len(report.pruned_indices),
            "merged": len(report.merge_map),
            "n_before": N,
            "n_after": len(new_store),
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
        )

        metadata = {
            "dim": self.dim,
            "beta": self.beta,
            "decay_rate": self.decay_rate,
            "importance_delta": self.importance_delta,
            "novelty_N0": self.novelty_N0,
            "novelty_gamma": self.novelty_gamma,
            "memories": [
                {
                    "text": m.text,
                    "importance": m.importance,
                    "creation_time": m.creation_time,
                    "last_access_time": m.last_access_time,
                    "access_count": m.access_count,
                    "tagged": m.tagged,
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
            )
            engine.memory_store.append(entry)

        engine._invalidate_cache()
        return engine
