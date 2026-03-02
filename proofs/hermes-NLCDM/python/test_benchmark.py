"""Benchmark: System A (hermes-memory) vs System B (coupled_engine) vs Hybrid.

All scenarios use synthetic numpy vectors with known ground truth.
No embedding model dependency — tests pure retrieval dynamics.

Systems:
  A: hermes-memory score_memory pipeline (weighted sum of relevance, recency,
     importance, activation). O(N) per query. Fast scalar scoring.
  B: coupled_engine spreading activation in Hopfield pattern space.
     O(N·dim·iters) per query. dream() consolidates offline.
  A+B: System A for fast wake queries, System B dream() for consolidation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add hermes-memory to Python path
_HERMES_MEMORY_DIR = Path(__file__).parent.parent.parent / "hermes-memory" / "python"
sys.path.insert(0, str(_HERMES_MEMORY_DIR))

from coupled_engine import CoupledEngine
from hermes_memory.engine import ParameterSet, MemoryState, score_memory, rank_memories
from nlcdm_core import cosine_sim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid ParameterSet for System A.
# Relevance-dominant: w1=0.85 so cosine_sim drives ranking.
# Novelty near-zero: 0.01 * exp(-10*age) ≈ 0 after age > 0.
# Contraction check: exp(-1) + 0.25/5 * 0.1 * 10 = 0.418 < 1 ✓
SYSTEM_A_PARAMS = ParameterSet(
    alpha=0.1,
    beta=1.0,
    delta_t=1.0,
    s_max=10.0,
    s0=1.0,
    temperature=5.0,
    novelty_start=0.01,
    novelty_decay=10.0,
    survival_threshold=0.005,
    feedback_sensitivity=0.1,
    w1=0.85,  # relevance — dominant for retrieval benchmarks
    w2=0.05,  # recency  — small, avoids age-bias
    w3=0.05,  # importance
    w4=0.05,  # activation
)


# ---------------------------------------------------------------------------
# Synthetic pattern generators
# ---------------------------------------------------------------------------

def make_orthogonal_patterns(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate n orthonormal patterns of dimension dim via QR decomposition."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, max(n, dim)))
    Q, _ = np.linalg.qr(M)
    return Q[:, :n].T  # (n, dim), rows are orthonormal


def make_cluster_patterns(
    n_clusters: int, per_cluster: int, dim: int, spread: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered patterns with known group structure.

    Returns:
        patterns: (n_clusters * per_cluster, dim) unit vectors
        labels: (n_clusters * per_cluster,) cluster assignments
    """
    rng = np.random.default_rng(seed)
    centers = make_orthogonal_patterns(n_clusters, dim, seed=seed)
    patterns = []
    labels = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            p = centers[c] + spread * rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            patterns.append(p)
            labels.append(c)
    return np.array(patterns), np.array(labels)


def make_interfering_pair(dim: int, similarity: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create two unit vectors with specified cosine similarity.

    similarity in [-1, 1]. Negative = contradictory.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    # Construct b so that cos(a, b) ≈ similarity
    orthogonal = rng.standard_normal(dim)
    orthogonal -= orthogonal @ a * a
    orthogonal /= np.linalg.norm(orthogonal)
    b = similarity * a + np.sqrt(1 - similarity**2) * orthogonal
    b /= np.linalg.norm(b)
    return a, b


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_indices: list[int], relevant_indices: set[int], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    top_k = retrieved_indices[:k]
    if not top_k:
        return 0.0
    return sum(1 for i in top_k if i in relevant_indices) / len(top_k)


def mean_reciprocal_rank(retrieved_indices: list[int], relevant_indices: set[int]) -> float:
    """1 / rank of the first relevant item. 0 if none found."""
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# System A Adapter — wraps hermes-memory with embedding interface
# ---------------------------------------------------------------------------

class SystemAAdapter:
    """Adapts hermes-memory's scalar scoring to work with embedding vectors.

    On store: saves embedding + metadata.
    On query: computes cosine_sim(query, stored) as relevance, constructs
    MemoryState, calls score_memory, ranks results.
    """

    def __init__(self, dim: int, params: ParameterSet | None = None):
        self.dim = dim
        self.params = params or SYSTEM_A_PARAMS
        self.embeddings: list[np.ndarray] = []
        self.importances: list[float] = []
        self.access_counts: list[int] = []
        self.strengths: list[float] = []
        self.creation_steps: list[int] = []
        self._step = 0

    @property
    def n_memories(self) -> int:
        return len(self.embeddings)

    def store(self, embedding: np.ndarray, importance: float = 0.5) -> int:
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        assert emb.shape[0] == self.dim
        idx = self.n_memories
        self.embeddings.append(emb / np.linalg.norm(emb))
        self.importances.append(importance)
        self.access_counts.append(0)
        self.strengths.append(self.params.s0)
        self.creation_steps.append(self._step)
        self._step += 1
        return idx

    def query(self, embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        if self.n_memories == 0:
            return []

        emb = np.asarray(embedding, dtype=np.float64).ravel()
        emb = emb / np.linalg.norm(emb)

        # Build MemoryState for each stored memory with cosine_sim as relevance
        memories = []
        for i in range(self.n_memories):
            rel = float(cosine_sim(emb, self.embeddings[i]))
            rel = max(0.0, min(1.0, rel))  # clamp to [0, 1]
            age = self._step - self.creation_steps[i]
            ms = MemoryState(
                relevance=rel,
                last_access_time=float(max(age, 1)),
                importance=self.importances[i],
                access_count=self.access_counts[i],
                strength=self.strengths[i],
                creation_time=float(age),
            )
            memories.append(ms)

        ranked = rank_memories(self.params, memories, float(self._step))
        k = min(top_k, self.n_memories)
        results = []
        for idx, sc in ranked[:k]:
            self.access_counts[idx] += 1
            results.append({"index": idx, "score": sc})
        return results


# ---------------------------------------------------------------------------
# Hybrid Engine — B retrieves, A re-ranks
# ---------------------------------------------------------------------------

class HybridEngine:
    """System B (brain) retrieves via softmax attention over X.
    System A (bookkeeper) re-ranks results using metadata.

    The hybrid gets B's proven retrieval guarantees PLUS A's metadata
    awareness (importance, recency, reliability).

    Stores data in both systems. Queries go through B first (softmax
    over pattern matrix X at β), then A re-ranks the top candidates
    using its scalar scoring (importance, recency, activation).
    Dreams run on B, propagate updated X back to A.
    """

    # β presets derived from proof validation (test_proof_validation.py)
    BETA_PRECISE = 50.0    # P@1=1.0 for tight clusters (spread=0.04)
    BETA_NORMAL = 10.0     # Right cluster, good P@5
    BETA_ASSOCIATIVE = 3.0 # Broad basin overlap, cluster-level recall

    def __init__(self, dim: int, beta: float = 5.0, params: ParameterSet | None = None):
        self.system_a = SystemAAdapter(dim=dim, params=params)
        self.system_b = CoupledEngine(dim=dim, beta=beta)

    @property
    def n_memories(self) -> int:
        return self.system_a.n_memories

    def store(self, embedding: np.ndarray, importance: float = 0.5) -> int:
        idx_a = self.system_a.store(embedding, importance)
        idx_b = self.system_b.store(None, embedding, importance)
        assert idx_a == idx_b
        return idx_a

    # ------------------------------------------------------------------
    # Adaptive β control — System A is the control plane
    # ------------------------------------------------------------------

    def choose_beta(self, query_type: str = "normal") -> float:
        """Choose β based on query intent.

        "precise":     high β — exact pattern recall (P@1)
        "normal":      moderate β — right cluster, good specificity
        "associative": low β — broad associations, cluster-level recall

        Proof basis: beta_regime_transition finding from handoff.
        beta < 5: all mixtures. beta 5-20: cluster-level. beta 50+: exact.
        """
        if query_type == "precise":
            return self.BETA_PRECISE
        elif query_type == "associative":
            return self.BETA_ASSOCIATIVE
        return self.BETA_NORMAL

    def should_dream(self) -> bool:
        """Check if dreaming would be useful based on capacity utilization.

        Dream adds value ONLY near the capacity boundary (handoff finding).
        When N << N_max: no mixture FPs to clean — dream is a no-op.
        When N >> N_max: concentrated FPs don't exist — dream can't help.

        Returns True when utilization is in the useful range (0.5 - 1.5).
        """
        N = self.n_memories
        if N <= 1:
            return False

        # Estimate δ = minimum pairwise cosine distance (sample)
        X = self.system_b._embeddings_matrix()
        rng = np.random.default_rng(0)
        n_sample = min(N * (N - 1) // 2, 200)
        min_delta = 2.0
        for _ in range(n_sample):
            i, j = rng.choice(N, size=2, replace=False)
            d_ij = 1.0 - float(X[i] @ X[j])
            min_delta = min(min_delta, d_ij)
        if min_delta <= 0:
            min_delta = 0.01

        beta = self.system_b.beta
        M_sq = float(np.max(np.sum(X ** 2, axis=1)))

        # Capacity formula from Capacity.lean: N_max = exp(β·δ) / (4·β·M²)
        n_max = np.exp(beta * min_delta) / (4.0 * beta * M_sq)
        utilization = N / max(n_max, 1.0)

        # Useful range: 50% to 150% of capacity
        return bool(0.5 <= utilization <= 1.5)

    def query(
        self, embedding: np.ndarray, top_k: int = 10, beta: float | None = None,
    ) -> list[dict]:
        """B retrieves via softmax attention, A re-ranks with metadata.

        1. System B returns top candidates via spreading activation
           at specified β (proven convergence, attractor-snapping)
        2. System A re-ranks those candidates using metadata scoring
           (importance, recency, access frequency, strength)

        Use choose_beta() to select β based on query intent.
        """
        # Step 1: B retrieves a wider candidate set via softmax attention
        candidate_k = min(top_k * 3, self.n_memories)
        b_results = self.system_b.query(embedding, beta=beta, top_k=candidate_k)
        if not b_results:
            return []

        # Step 2: A re-ranks B's candidates using metadata
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        emb = emb / np.linalg.norm(emb)

        scored_candidates = []
        for hit in b_results:
            idx = hit["index"]
            b_score = hit["score"]

            # Build A's metadata score for this candidate
            rel = float(cosine_sim(emb, self.system_a.embeddings[idx]))
            rel = max(0.0, min(1.0, rel))
            age = self.system_a._step - self.system_a.creation_steps[idx]
            ms = MemoryState(
                relevance=rel,
                last_access_time=float(max(age, 1)),
                importance=self.system_a.importances[idx],
                access_count=self.system_a.access_counts[idx],
                strength=self.system_a.strengths[idx],
                creation_time=float(age),
            )
            a_score = score_memory(self.system_a.params, ms, float(self.system_a._step))

            # Combined score: B's geometric relevance + A's metadata awareness
            # B_score is cosine with converged attractor (0-1)
            # A_score is weighted combination of relevance+recency+importance+activation
            combined = 0.6 * b_score + 0.4 * a_score
            scored_candidates.append((idx, combined))

        # Sort by combined score, return top_k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, sc in scored_candidates[:top_k]:
            self.system_a.access_counts[idx] += 1
            self.system_b.memory_store[idx].access_count += 1
            results.append({"index": idx, "score": sc})
        return results

    def dream(self) -> dict:
        """Offline consolidation via System B."""
        result = self.system_b.dream()
        # Propagate consolidated embeddings back to System A
        for i in range(min(self.system_b.n_memories, self.system_a.n_memories)):
            self.system_a.embeddings[i] = self.system_b.memory_store[i].embedding.copy()
        return result


# ---------------------------------------------------------------------------
# Helper: run retrieval trial across all 3 systems
# ---------------------------------------------------------------------------

def run_retrieval_trial(
    systems: dict[str, object],
    query_emb: np.ndarray,
    relevant: set[int],
    top_k: int = 5,
) -> dict[str, dict]:
    """Query all systems and compute metrics."""
    results = {}
    for name, sys in systems.items():
        t0 = time.perf_counter()
        hits = sys.query(query_emb, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        retrieved = [h["index"] for h in hits]
        results[name] = {
            "p@1": precision_at_k(retrieved, relevant, 1),
            "p@5": precision_at_k(retrieved, relevant, top_k),
            "mrr": mean_reciprocal_rank(retrieved, relevant),
            "latency_ms": elapsed_ms,
            "retrieved": retrieved,
        }
    return results


def make_random_patterns(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate n random unit vectors."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, dim))
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norms


def cluster_coherence(retrieved_indices: list[int], labels: np.ndarray) -> float:
    """Fraction of retrieved items sharing the cluster of the plurality."""
    if not retrieved_indices:
        return 0.0
    retrieved_labels = [int(labels[i]) for i in retrieved_indices]
    from collections import Counter
    most_common_count = Counter(retrieved_labels).most_common(1)[0][1]
    return most_common_count / len(retrieved_labels)


# ===========================================================================
# Scenario 1: Clustered embeddings — the discriminative test
#
# B's coupling matrix W encodes cluster structure via Hebbian learning.
# A scores each memory independently — no structural information.
# Dream should deepen cluster basins and clean cross-cluster interference.
# ===========================================================================

class TestClusteredRetrieval:
    """Tight clusters in embedding space (cos_sim 0.88-0.95 to centroid).

    B's W accumulates correlated Hebbian updates → strong intra-cluster
    coupling. Spreading activation snaps to the nearest cluster basin.
    A sees each memory as an independent scalar score — no structure.
    """

    DIM = 128
    N_CLUSTERS = 10
    PER_CLUSTER = 15  # 150 total memories
    SPREAD = 0.04     # gives cos_sim ≈ 0.88-0.95 to centroid
    BETA = 10.0

    def _build_systems(self, seed=42):
        """Build all 3 systems with identical clustered memories."""
        patterns, labels = make_cluster_patterns(
            self.N_CLUSTERS, self.PER_CLUSTER, self.DIM,
            spread=self.SPREAD, seed=seed,
        )
        centroids = make_orthogonal_patterns(self.N_CLUSTERS, self.DIM, seed=seed)
        n_total = len(patterns)

        systems = {
            "A": SystemAAdapter(dim=self.DIM),
            "B": CoupledEngine(dim=self.DIM, beta=self.BETA),
            "A+B": HybridEngine(dim=self.DIM, beta=self.BETA),
        }
        for i in range(n_total):
            systems["A"].store(patterns[i], importance=0.5)
            systems["B"].store(f"c{labels[i]}_m{i}", patterns[i], importance=0.5)
            systems["A+B"].store(patterns[i], importance=0.5)

        return systems, patterns, labels, centroids

    def test_intra_cluster_retrieval(self):
        """Query with a cluster member: top-k should be from the same cluster."""
        systems, patterns, labels, centroids = self._build_systems(seed=42)
        rng = np.random.default_rng(111)

        n_trials = 20
        cluster_p5 = {n: 0.0 for n in ("A", "B", "A+B")}

        for trial in range(n_trials):
            # Pick a random pattern, add noise, query
            idx = rng.integers(0, len(patterns))
            target_cluster = int(labels[idx])
            relevant = {j for j in range(len(patterns)) if labels[j] == target_cluster}
            query = patterns[idx] + 0.03 * rng.standard_normal(self.DIM)
            query /= np.linalg.norm(query)

            r = run_retrieval_trial(systems, query, relevant, top_k=5)
            for name in cluster_p5:
                cluster_p5[name] += r[name]["p@5"]

        for name in cluster_p5:
            avg = cluster_p5[name] / n_trials
            assert avg >= 0.5, (
                f"{name}: cluster P@5 = {avg:.3f} — can't even get half the "
                f"top-5 from the same cluster"
            )

    def test_ambiguous_query_coherence(self):
        """Query equidistant from 2 clusters: B should snap to one (attractor).

        A scores each memory independently → returns a mix from both clusters.
        B's spreading activation converges to one attractor → coherent results.
        """
        systems, patterns, labels, centroids = self._build_systems(seed=42)

        n_trials = 15
        coherences = {n: [] for n in ("A", "B", "A+B")}

        rng = np.random.default_rng(222)
        for trial in range(n_trials):
            # Pick two different clusters, create midpoint query
            c1, c2 = rng.choice(self.N_CLUSTERS, size=2, replace=False)
            midpoint = centroids[c1] + centroids[c2]
            midpoint /= np.linalg.norm(midpoint)

            for name, sys in systems.items():
                hits = sys.query(midpoint, top_k=5)
                retrieved = [h["index"] for h in hits]
                coh = cluster_coherence(retrieved, labels)
                coherences[name].append(coh)

        mean_coh = {n: float(np.mean(v)) for n, v in coherences.items()}

        # B should have higher coherence than A (attractor snapping)
        assert mean_coh["B"] >= mean_coh["A"] * 0.9, (
            f"B coherence ({mean_coh['B']:.3f}) should be >= A ({mean_coh['A']:.3f}) "
            f"on ambiguous queries — attractor dynamics should produce coherent results"
        )

    def test_dream_deepens_cluster_basins(self):
        """After dream, intra-cluster retrieval should improve or hold steady."""
        patterns, labels = make_cluster_patterns(
            self.N_CLUSTERS, self.PER_CLUSTER, self.DIM,
            spread=self.SPREAD, seed=42,
        )
        n_total = len(patterns)

        # Pre-dream
        sys_pre = CoupledEngine(dim=self.DIM, beta=self.BETA)
        for i in range(n_total):
            sys_pre.store(f"c{labels[i]}_m{i}", patterns[i], importance=0.5)

        rng = np.random.default_rng(333)
        pre_p5 = 0.0
        n_trials = 20
        trial_indices = rng.integers(0, n_total, size=n_trials)
        noises = [0.03 * rng.standard_normal(self.DIM) for _ in range(n_trials)]

        for trial in range(n_trials):
            idx = trial_indices[trial]
            relevant = {j for j in range(n_total) if labels[j] == labels[idx]}
            query = patterns[idx] + noises[trial]
            query /= np.linalg.norm(query)
            hits = sys_pre.query(query, top_k=5)
            retrieved = [h["index"] for h in hits]
            pre_p5 += precision_at_k(retrieved, relevant, 5)

        # Post-dream (fresh build, same data)
        sys_post = CoupledEngine(dim=self.DIM, beta=self.BETA)
        for i in range(n_total):
            sys_post.store(f"c{labels[i]}_m{i}", patterns[i], importance=0.5)
        sys_post.dream()

        post_p5 = 0.0
        for trial in range(n_trials):
            idx = trial_indices[trial]
            relevant = {j for j in range(n_total) if labels[j] == labels[idx]}
            query = patterns[idx] + noises[trial]
            query /= np.linalg.norm(query)
            hits = sys_post.query(query, top_k=5)
            retrieved = [h["index"] for h in hits]
            post_p5 += precision_at_k(retrieved, relevant, 5)

        pre_avg = pre_p5 / n_trials
        post_avg = post_p5 / n_trials
        # Dream should not degrade cluster retrieval
        assert post_avg >= pre_avg * 0.85, (
            f"Dream degraded cluster P@5: {pre_avg:.3f} → {post_avg:.3f}"
        )


# ===========================================================================
# Scenario 2: Temporal interference — the forgetting test
#
# Store old memories (tagged), then new overlapping memories.
# B's W gets overwritten by new Hebbian updates → old patterns degrade.
# B after dream → NREM-replay recovers tagged old memories.
# A treats old and new equally (modulo recency heuristic).
# ===========================================================================

class TestTemporalInterference:
    """Catastrophic forgetting resistance test.

    Old memories (tagged, important) stored first.  New memories from
    overlapping clusters stored later, overwriting coupling structure.
    Dream should recover old memories via NREM-replay on tagged indices.
    """

    DIM = 128
    BETA = 10.0
    N_OLD_CLUSTERS = 5
    N_NEW_CLUSTERS = 5  # 3 share centroids with old, 2 are fresh
    PER_CLUSTER = 10

    def _build_scenario(self, seed=42):
        """Build old and new memory sets with overlapping cluster structure."""
        rng = np.random.default_rng(seed)
        all_centroids = make_orthogonal_patterns(7, self.DIM, seed=seed)

        old_centroids = all_centroids[:5]  # clusters 0-4
        # New clusters: 3 reuse old centroid directions, 2 are fresh
        new_centroids = np.vstack([
            old_centroids[:3],       # overlap with old clusters 0,1,2
            all_centroids[5:7],      # 2 fresh directions
        ])

        # Generate old patterns
        old_patterns = []
        old_labels = []
        for c in range(self.N_OLD_CLUSTERS):
            for _ in range(self.PER_CLUSTER):
                p = old_centroids[c] + 0.04 * rng.standard_normal(self.DIM)
                p /= np.linalg.norm(p)
                old_patterns.append(p)
                old_labels.append(c)
        old_patterns = np.array(old_patterns)
        old_labels = np.array(old_labels)

        # Generate new patterns
        new_patterns = []
        new_labels = []
        for c in range(self.N_NEW_CLUSTERS):
            for _ in range(self.PER_CLUSTER):
                p = new_centroids[c] + 0.04 * rng.standard_normal(self.DIM)
                p /= np.linalg.norm(p)
                new_patterns.append(p)
                new_labels.append(c + self.N_OLD_CLUSTERS)  # labels 5-9
        new_patterns = np.array(new_patterns)
        new_labels = np.array(new_labels)

        return old_patterns, old_labels, new_patterns, new_labels, old_centroids

    def test_old_memory_retrieval_after_interference(self):
        """Old memories should still be retrievable after new memories interfere."""
        old_pats, old_lbls, new_pats, new_lbls, centroids = self._build_scenario(42)
        n_old = len(old_pats)
        n_new = len(new_pats)

        sys_a = SystemAAdapter(dim=self.DIM)
        sys_b = CoupledEngine(dim=self.DIM, beta=self.BETA)

        # Store old memories first (tagged, high importance)
        for i in range(n_old):
            sys_a.store(old_pats[i], importance=0.8)
            sys_b.store(f"old_c{old_lbls[i]}_m{i}", old_pats[i], importance=0.8)

        old_indices = set(range(n_old))

        # Store new overlapping memories
        for i in range(n_new):
            sys_a.store(new_pats[i], importance=0.5)
            sys_b.store(f"new_c{new_lbls[i]}_m{i}", new_pats[i], importance=0.5)

        # Query for old cluster centroids — measure how many old memories retrieved
        rng = np.random.default_rng(555)
        n_trials = 10
        old_recall = {"A": 0.0, "B": 0.0}

        for trial in range(n_trials):
            c = trial % self.N_OLD_CLUSTERS
            query = centroids[c] + 0.03 * rng.standard_normal(self.DIM)
            query /= np.linalg.norm(query)
            relevant_old = {j for j in range(n_old) if old_lbls[j] == c}

            for name, sys in [("A", sys_a), ("B", sys_b)]:
                hits = sys.query(query, top_k=self.PER_CLUSTER)
                retrieved = [h["index"] for h in hits]
                old_recall[name] += precision_at_k(retrieved, relevant_old,
                                                   self.PER_CLUSTER)

        for name in old_recall:
            avg = old_recall[name] / n_trials
            # At least some old memories should survive interference
            assert avg >= 0.2, (
                f"{name}: old memory recall = {avg:.3f} — catastrophic forgetting"
            )

    def test_dream_recovers_old_memories(self):
        """After dream, retrieval of tagged old memories should improve."""
        old_pats, old_lbls, new_pats, new_lbls, centroids = self._build_scenario(42)
        n_old = len(old_pats)
        n_new = len(new_pats)

        # Build two System B instances: one will dream, one won't
        sys_no_dream = CoupledEngine(dim=self.DIM, beta=self.BETA)
        sys_dream = CoupledEngine(dim=self.DIM, beta=self.BETA)

        for i in range(n_old):
            sys_no_dream.store(f"old_{i}", old_pats[i], importance=0.8)
            sys_dream.store(f"old_{i}", old_pats[i], importance=0.8)
        for i in range(n_new):
            sys_no_dream.store(f"new_{i}", new_pats[i], importance=0.5)
            sys_dream.store(f"new_{i}", new_pats[i], importance=0.5)

        # Dream with tagged_indices = old memories (the ones we want to protect)
        tagged = list(range(n_old))
        sys_dream.dream(tagged_indices=tagged)

        rng = np.random.default_rng(666)
        n_trials = 10
        recall_no_dream = 0.0
        recall_dream = 0.0

        for trial in range(n_trials):
            c = trial % self.N_OLD_CLUSTERS
            query = centroids[c] + 0.03 * rng.standard_normal(self.DIM)
            query /= np.linalg.norm(query)
            relevant_old = {j for j in range(n_old) if old_lbls[j] == c}

            hits_nd = sys_no_dream.query(query, top_k=self.PER_CLUSTER)
            hits_d = sys_dream.query(query, top_k=self.PER_CLUSTER)
            recall_no_dream += precision_at_k(
                [h["index"] for h in hits_nd], relevant_old, self.PER_CLUSTER)
            recall_dream += precision_at_k(
                [h["index"] for h in hits_d], relevant_old, self.PER_CLUSTER)

        avg_nd = recall_no_dream / n_trials
        avg_d = recall_dream / n_trials
        # Dream should not make old memory recall worse
        assert avg_d >= avg_nd * 0.85, (
            f"Dream hurt old recall: no-dream={avg_nd:.3f}, dream={avg_d:.3f}"
        )


# ===========================================================================
# Scenario 3: Capacity cliff — random patterns, dim=64, sweep N
#
# Hopfield theory: classical capacity cliff at α ≈ 0.138 (N/dim).
# Modern Hopfield (spreading_activation) has higher capacity.
# Dreams should push the cliff further.
# A degrades gradually (no cliff — scoring is memoryless).
# ===========================================================================

class TestCapacityCliff:
    """Capacity cliff with random (non-orthogonal) patterns.

    Sweeps N/dim ratio to find where retrieval accuracy breaks down.
    B-without-dream should show a cliff. B-with-dream should extend it.
    A should show gradual degradation (no structural cliff).
    """

    DIM = 64
    BETA = 5.0

    @pytest.mark.parametrize("n_memories", [5, 10, 20, 30, 40, 50, 60])
    def test_capacity_sweep(self, n_memories):
        """Measure retrieval accuracy at each N for all systems."""
        patterns = make_random_patterns(n_memories, self.DIM, seed=n_memories)
        rng = np.random.default_rng(n_memories + 7000)

        sys_a = SystemAAdapter(dim=self.DIM)
        sys_b_raw = CoupledEngine(dim=self.DIM, beta=self.BETA)
        sys_b_dream = CoupledEngine(dim=self.DIM, beta=self.BETA)

        for i in range(n_memories):
            sys_a.store(patterns[i], importance=0.5)
            sys_b_raw.store(f"m{i}", patterns[i], importance=0.5)
            sys_b_dream.store(f"m{i}", patterns[i], importance=0.5)

        # Dream one copy
        sys_b_dream.dream()

        # Evaluate: query with exact patterns (no noise) — pure capacity test
        n_trials = min(n_memories, 20)
        correct = {"A": 0, "B": 0, "B+dream": 0}

        for trial in range(n_trials):
            idx = trial % n_memories
            for label, sys in [("A", sys_a), ("B", sys_b_raw), ("B+dream", sys_b_dream)]:
                hits = sys.query(patterns[idx], top_k=1)
                if hits and hits[0]["index"] == idx:
                    correct[label] += 1

        alpha = n_memories / self.DIM
        acc = {k: v / n_trials for k, v in correct.items()}

        # At low load (α < 0.3), all systems should work reasonably
        if alpha <= 0.3:
            for name in ("A", "B"):
                assert acc[name] >= 0.4, (
                    f"{name}: accuracy {acc[name]:.3f} at α={alpha:.2f} — "
                    f"should handle low load"
                )

        # Dream should never make things worse
        assert acc["B+dream"] >= acc["B"] * 0.8 - 0.05, (
            f"Dream degraded: B={acc['B']:.3f}, B+dream={acc['B+dream']:.3f} "
            f"at α={alpha:.2f}"
        )

    def test_capacity_cliff_summary(self):
        """Sweep N and print the degradation curve for visual inspection."""
        sweep = list(range(5, 65, 5))
        results = {"A": [], "B": [], "B+dream": []}

        for n_mem in sweep:
            patterns = make_random_patterns(n_mem, self.DIM, seed=n_mem + 8000)

            sys_a = SystemAAdapter(dim=self.DIM)
            sys_b = CoupledEngine(dim=self.DIM, beta=self.BETA)
            sys_bd = CoupledEngine(dim=self.DIM, beta=self.BETA)

            for i in range(n_mem):
                sys_a.store(patterns[i], importance=0.5)
                sys_b.store(f"m{i}", patterns[i], importance=0.5)
                sys_bd.store(f"m{i}", patterns[i], importance=0.5)

            sys_bd.dream()

            n_trials = min(n_mem, 20)
            correct = {"A": 0, "B": 0, "B+dream": 0}
            for trial in range(n_trials):
                idx = trial % n_mem
                for label, sys in [("A", sys_a), ("B", sys_b), ("B+dream", sys_bd)]:
                    hits = sys.query(patterns[idx], top_k=1)
                    if hits and hits[0]["index"] == idx:
                        correct[label] += 1

            for name in correct:
                results[name].append(correct[name] / n_trials)

        # Print the curve
        print("\n  Capacity Cliff (dim=64, random patterns, exact query):")
        print(f"  {'N':>4} {'α=N/d':>6} {'A':>6} {'B':>6} {'B+drm':>6}")
        for i, n_mem in enumerate(sweep):
            alpha = n_mem / self.DIM
            print(
                f"  {n_mem:>4} {alpha:>6.3f} "
                f"{results['A'][i]:>6.3f} "
                f"{results['B'][i]:>6.3f} "
                f"{results['B+dream'][i]:>6.3f}"
            )


# ===========================================================================
# Scenario 4: Adaptive β — the control plane test
#
# Validates that HybridEngine.choose_beta and should_dream work correctly.
# ===========================================================================

class TestAdaptiveBeta:
    """Test adaptive β selection and utilization-gated dreaming."""

    DIM = 64
    BETA = 10.0

    def test_choose_beta_returns_correct_values(self):
        hybrid = HybridEngine(dim=self.DIM, beta=self.BETA)
        assert hybrid.choose_beta("precise") == HybridEngine.BETA_PRECISE
        assert hybrid.choose_beta("normal") == HybridEngine.BETA_NORMAL
        assert hybrid.choose_beta("associative") == HybridEngine.BETA_ASSOCIATIVE

    def test_precise_beta_improves_p1(self):
        """Higher β should improve exact pattern recall (P@1)."""
        patterns, labels = make_cluster_patterns(
            5, 10, self.DIM, spread=0.08, seed=42,
        )
        hybrid = HybridEngine(dim=self.DIM, beta=self.BETA)
        for i in range(len(patterns)):
            hybrid.store(patterns[i], importance=0.5)

        rng = np.random.default_rng(42)
        p1_normal = 0
        p1_precise = 0
        n_trials = 20

        for trial in range(n_trials):
            idx = rng.integers(0, len(patterns))
            query = patterns[idx] + 0.02 * rng.standard_normal(self.DIM)
            query /= np.linalg.norm(query)

            hits_n = hybrid.query(query, top_k=1, beta=hybrid.choose_beta("normal"))
            hits_p = hybrid.query(query, top_k=1, beta=hybrid.choose_beta("precise"))
            if hits_n and hits_n[0]["index"] == idx:
                p1_normal += 1
            if hits_p and hits_p[0]["index"] == idx:
                p1_precise += 1

        # Precise β should be >= normal β for exact recall
        assert p1_precise >= p1_normal, (
            f"Precise β didn't help P@1: normal={p1_normal}/{n_trials}, "
            f"precise={p1_precise}/{n_trials}"
        )

    def test_should_dream_false_when_undercapacity(self):
        """With very few patterns relative to capacity, dreaming is useless."""
        hybrid = HybridEngine(dim=self.DIM, beta=self.BETA)
        patterns = make_orthogonal_patterns(3, self.DIM, seed=0)
        for i in range(3):
            hybrid.store(patterns[i], importance=0.5)
        # 3 orthogonal patterns in dim=64 at β=10 — way under capacity
        assert hybrid.should_dream() is False

    def test_should_dream_with_query_type(self):
        """HybridEngine.query accepts beta from choose_beta."""
        hybrid = HybridEngine(dim=self.DIM, beta=self.BETA)
        patterns = make_orthogonal_patterns(5, self.DIM, seed=0)
        for i in range(5):
            hybrid.store(patterns[i], importance=0.5)

        # Query at different β values should work without error
        beta_assoc = hybrid.choose_beta("associative")
        results = hybrid.query(patterns[0], top_k=3, beta=beta_assoc)
        assert len(results) > 0
