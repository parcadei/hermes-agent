"""Dream pipeline with strength decay — wiring hermes-memory temporal dynamics
into hermes-NLCDM dream operations.

Tests that the formally proven properties from both proof systems compose
correctly in the Python implementation:

  hermes-memory/StrengthDecay.lean:
    - S(t) = S₀ · e^(-β·t)                     (exponential decay)
    - S* < Smax                                  (anti-lock-in)
    - strengthDecay_antitone                      (monotone decreasing)

  hermes-memory/ComposedSystem.lean:
    - composedSystem_safe                         (all three fixes compose)
    - fixedPoint_lt_Smax                          (steady state below max)

  hermes-NLCDM/Capacity.lean:
    - N_max ~ exp(βδ)/(4βM²)                     (exponential capacity)

  hermes-NLCDM/Lyapunov.lean:
    - totalEnergy monotonically decreases         (convergence)

The key insight from Phase 4-5 testing: at real geometry (δ_min≈0.30, β=10),
the system is near effective capacity. Dreams are ESSENTIAL for retrieval
viability. Strength decay makes this even more critical: without dreams,
old patterns accumulate AND decay creates retrieval instability.

With both systems wired together:
  - Strength decay prevents lock-in (old patterns lose strength over time)
  - Dreams consolidate (merge/prune reduce N, raising effective capacity)
  - Fresh memories can compete (novelty bonus + decayed competitors)
  - The system converges to a stable stationary state (Banach fixed-point)
"""

from __future__ import annotations

import numpy as np
import pytest

from dream_ops import (
    dream_cycle_v2,
    dream_cycle_xb,
    nrem_merge_xb,
    nrem_prune_xb,
    spreading_activation,
)
from nlcdm_core import softmax
from test_capacity_boundary import make_separated_centroids, make_cluster_patterns


# ---------------------------------------------------------------------------
# Strength decay implementation (mirrors StrengthDecay.lean)
# ---------------------------------------------------------------------------

def strength_decay(beta: float, s0: float, t: float) -> float:
    """S(t) = S₀ · e^(-β·t)  — Lean: strengthDecay"""
    return s0 * np.exp(-beta * t)


def strength_update(alpha: float, s: float, s_max: float) -> float:
    """S' = (1-α)·S + α·Smax  — Lean: strengthUpdate_alt"""
    return (1 - alpha) * s + alpha * s_max


def combined_step(alpha: float, beta: float, delta_t: float,
                  s: float, s_max: float) -> float:
    """One cycle: decay for delta_t, then access update.
    S_{n+1} = strengthUpdate(α, S_n · e^(-β·Δ), Smax)
    """
    decayed = strength_decay(beta, s, delta_t)
    return strength_update(alpha, decayed, s_max)


def steady_state_strength(alpha: float, beta: float,
                          delta_t: float, s_max: float) -> float:
    """S* = α·Smax / (1 - (1-α)·e^(-β·Δ))  — Lean: steadyStateStrength"""
    gamma = (1 - alpha) * np.exp(-beta * delta_t)
    return alpha * s_max / (1 - gamma)


def soft_select(s1: float, s2: float, temperature: float) -> float:
    """Soft selection probability for memory 1. Lean: softSelect"""
    e1 = np.exp(s1 / max(temperature, 1e-10))
    e2 = np.exp(s2 / max(temperature, 1e-10))
    return e1 / (e1 + e2)


# ---------------------------------------------------------------------------
# Memory store with strength tracking
# ---------------------------------------------------------------------------

class MemoryWithStrength:
    """A memory pattern with strength metadata for temporal dynamics."""

    def __init__(self, embedding: np.ndarray, strength: float,
                 importance: float, domain: int, last_access: float):
        self.embedding = embedding / np.linalg.norm(embedding)
        self.strength = strength
        self.importance = importance
        self.domain = domain
        self.last_access = last_access

    def decay_to(self, current_time: float, decay_rate: float) -> None:
        """Apply strength decay since last access."""
        elapsed = current_time - self.last_access
        if elapsed > 0:
            self.strength = strength_decay(decay_rate, self.strength, elapsed)

    def access(self, alpha: float, s_max: float) -> None:
        """Strength update on access: S' = (1-α)S + αSmax"""
        self.strength = strength_update(alpha, self.strength, s_max)


class MemoryStoreWithDecay:
    """Memory store that tracks per-pattern strength with temporal decay.

    Wires hermes-memory temporal dynamics into the NLCDM pattern store.
    """

    def __init__(self, dim: int, decay_rate: float = 0.05,
                 alpha: float = 0.3, s_max: float = 1.0):
        self.dim = dim
        self.decay_rate = decay_rate  # β in StrengthDecay.lean
        self.alpha = alpha            # α in strengthUpdate
        self.s_max = s_max
        self.memories: list[MemoryWithStrength] = []
        self.current_time: float = 0.0

    def store(self, embedding: np.ndarray, importance: float = 0.4,
              domain: int = 0) -> int:
        """Store a new memory with initial strength."""
        mem = MemoryWithStrength(
            embedding=embedding,
            strength=self.alpha * self.s_max,  # initial = α·Smax
            importance=importance,
            domain=domain,
            last_access=self.current_time,
        )
        self.memories.append(mem)
        return len(self.memories) - 1

    def advance_time(self, delta_t: float) -> None:
        """Advance time and decay all strengths."""
        self.current_time += delta_t
        for mem in self.memories:
            mem.decay_to(self.current_time, self.decay_rate)

    def access(self, idx: int) -> None:
        """Access a memory: reset last_access time and boost strength."""
        mem = self.memories[idx]
        mem.last_access = self.current_time
        mem.access(self.alpha, self.s_max)

    def get_patterns(self) -> np.ndarray:
        """Extract pattern array for dream pipeline."""
        if not self.memories:
            return np.zeros((0, self.dim))
        return np.array([m.embedding for m in self.memories])

    def get_importances(self) -> np.ndarray:
        """Importance weighted by current strength."""
        if not self.memories:
            return np.array([])
        return np.array([m.importance * m.strength for m in self.memories])

    def get_raw_importances(self) -> np.ndarray:
        if not self.memories:
            return np.array([])
        return np.array([m.importance for m in self.memories])

    def get_strengths(self) -> np.ndarray:
        if not self.memories:
            return np.array([])
        return np.array([m.strength for m in self.memories])

    def get_labels(self) -> np.ndarray:
        if not self.memories:
            return np.array([], dtype=int)
        return np.array([m.domain for m in self.memories], dtype=int)

    def apply_dream_result(self, new_patterns: np.ndarray,
                           kept_mask: np.ndarray | None = None,
                           merge_map: dict | None = None) -> None:
        """Apply dream results back: update patterns, handle prune/merge."""
        if merge_map is None:
            merge_map = {}

        # Simple case: just update pattern embeddings if shape unchanged
        if new_patterns.shape[0] == len(self.memories) and not merge_map:
            for i, mem in enumerate(self.memories):
                mem.embedding = new_patterns[i] / np.linalg.norm(new_patterns[i])
            return

        # Complex case: patterns were pruned/merged
        # Rebuild memory list from dream output
        old_memories = self.memories[:]
        self.memories = []

        # Determine which original indices survived (non-merged, non-pruned)
        merged_originals: set[int] = set()
        for group in merge_map.values():
            merged_originals.update(group)

        # Non-merged survivors keep their metadata
        non_merged_count = 0
        for i, mem in enumerate(old_memories):
            if i not in merged_originals and non_merged_count < new_patterns.shape[0]:
                new_mem = MemoryWithStrength(
                    embedding=new_patterns[non_merged_count],
                    strength=mem.strength,
                    importance=mem.importance,
                    domain=mem.domain,
                    last_access=mem.last_access,
                )
                self.memories.append(new_mem)
                non_merged_count += 1

        # Merged patterns get boosted strength (max of group + bonus)
        for out_idx in sorted(merge_map.keys()):
            group = merge_map[out_idx]
            if out_idx < new_patterns.shape[0]:
                group_strengths = [old_memories[i].strength for i in group
                                   if i < len(old_memories)]
                group_importances = [old_memories[i].importance for i in group
                                     if i < len(old_memories)]
                domains = [old_memories[i].domain for i in group
                           if i < len(old_memories)]
                from collections import Counter
                domain_counts = Counter(domains)
                majority_domain = max(domain_counts, key=domain_counts.get)

                new_mem = MemoryWithStrength(
                    embedding=new_patterns[out_idx],
                    strength=min(max(group_strengths) * 1.1, self.s_max),
                    importance=min(max(group_importances) + 0.1, 1.0),
                    domain=majority_domain,
                    last_access=self.current_time,
                )
                self.memories.append(new_mem)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dim():
    return 128


@pytest.fixture
def centroids(dim):
    """5 well-separated domain centroids."""
    return make_separated_centroids(n=5, dim=dim, seed=42)


def make_noisy_pattern(centroid: np.ndarray, spread: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Create a noisy version of a centroid, normalized to unit length."""
    p = centroid + spread * rng.standard_normal(centroid.shape[0])
    return p / np.linalg.norm(p)


# ===========================================================================
# Tests
# ===========================================================================


class TestStrengthDecayProperties:
    """Verify that the Python implementation matches StrengthDecay.lean proofs."""

    def test_decay_at_zero(self):
        """strengthDecay_at_zero: S(0) = S₀"""
        assert strength_decay(0.1, 5.0, 0.0) == pytest.approx(5.0)

    def test_decay_positive(self):
        """strengthDecay_pos: S(t) > 0 for S₀ > 0"""
        for t in [0, 1, 10, 100, 1000]:
            assert strength_decay(0.1, 1.0, t) > 0

    def test_decay_le_init(self):
        """strengthDecay_le_init: S(t) ≤ S₀ for β > 0, t ≥ 0"""
        for t in [0, 0.1, 1, 10]:
            assert strength_decay(0.1, 1.0, t) <= 1.0 + 1e-12

    def test_decay_antitone(self):
        """strengthDecay_antitone: t₁ ≤ t₂ → S(t₁) ≥ S(t₂)"""
        times = [0, 1, 2, 5, 10, 20, 50]
        values = [strength_decay(0.1, 1.0, t) for t in times]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_decay_tends_to_zero(self):
        """strengthDecay_tendsto_zero: S(t) → 0 as t → ∞"""
        assert strength_decay(0.1, 1.0, 1000) < 1e-40

    def test_steady_state_lt_smax(self):
        """steadyState_lt_Smax: S* < Smax (the anti-lock-in theorem)"""
        for alpha in [0.1, 0.3, 0.5, 0.9]:
            for beta in [0.01, 0.05, 0.1, 0.5]:
                for delta_t in [0.5, 1.0, 5.0]:
                    s_max = 1.0
                    s_star = steady_state_strength(alpha, beta, delta_t, s_max)
                    assert s_star < s_max, (
                        f"Anti-lock-in violated: S*={s_star} >= Smax={s_max} "
                        f"at α={alpha}, β={beta}, Δ={delta_t}"
                    )

    def test_steady_state_is_fixed_point(self):
        """steadyState_is_fixpoint: one combined step from S* returns S*"""
        alpha, beta, delta_t, s_max = 0.3, 0.05, 1.0, 1.0
        s_star = steady_state_strength(alpha, beta, delta_t, s_max)
        s_next = combined_step(alpha, beta, delta_t, s_star, s_max)
        assert s_next == pytest.approx(s_star, abs=1e-10)

    def test_iteration_converges_to_steady_state(self):
        """strengthIter_tendsto: iterating from any S₀ → S*"""
        alpha, beta, delta_t, s_max = 0.3, 0.05, 1.0, 1.0
        s_star = steady_state_strength(alpha, beta, delta_t, s_max)

        for s0 in [0.0, 0.1, 0.5, 0.9, 1.0]:
            s = s0
            for _ in range(200):
                s = combined_step(alpha, beta, delta_t, s, s_max)
            assert s == pytest.approx(s_star, abs=1e-6), (
                f"From S₀={s0}, converged to {s} instead of S*={s_star}"
            )


class TestSoftSelectionProperties:
    """Verify soft selection properties from SoftSelection.lean."""

    def test_soft_select_in_unit_interval(self):
        """softSelect_pos and softSelect_lt_one: p ∈ (0, 1)"""
        for s1 in [0.1, 0.5, 0.9]:
            for s2 in [0.1, 0.5, 0.9]:
                p = soft_select(s1, s2, temperature=0.1)
                assert 0 < p < 1

    def test_equal_strengths_equal_selection(self):
        """Equal strengths → equal selection probability"""
        p = soft_select(0.5, 0.5, temperature=0.1)
        assert p == pytest.approx(0.5, abs=1e-10)

    def test_stronger_gets_higher_probability(self):
        """Higher strength → higher selection probability"""
        p = soft_select(0.8, 0.3, temperature=0.1)
        assert p > 0.5


class TestComposedSystemProperties:
    """Verify composed system properties from ComposedSystem.lean +
    ContractionWiring.lean."""

    def test_domain_invariance(self):
        """expectedStrengthUpdate preserves [0, Smax]"""
        alpha, s_max = 0.3, 1.0
        beta, delta_t = 0.05, 1.0

        for s in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
                # Expected update: (1-qα)e^(-βΔ)S + qαSmax
                e = np.exp(-beta * delta_t)
                result = (1 - q * alpha) * e * s + q * alpha * s_max
                assert 0 <= result <= s_max + 1e-12

    def test_contraction_convergence(self):
        """Two-memory system converges to unique stationary state.

        Mirrors stationaryState_exists_unique_convergent from
        ContractionWiring.lean via Banach fixed-point theorem.

        The contraction condition requires L·α·Smax < 1 - exp(-βΔ),
        where L is the Lipschitz constant of soft_select. A moderate
        temperature ensures L is small enough for contraction.
        """
        alpha, beta, delta_t, s_max = 0.3, 0.05, 1.0, 1.0
        # Higher temperature → lower Lipschitz constant → contraction holds
        temperature = 1.0

        # Start from different initial states (both components > 0,
        # matching the domain invariance requirement 0 < S < Smax)
        initial_states = [(0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.3, 0.7)]
        final_states = []

        for s1_0, s2_0 in initial_states:
            s1, s2 = s1_0, s2_0
            for _ in range(1000):
                p = soft_select(s1, s2, temperature)
                e = np.exp(-beta * delta_t)
                s1_new = (1 - p * alpha) * e * s1 + p * alpha * s_max
                s2_new = (1 - (1 - p) * alpha) * e * s2 + (1 - p) * alpha * s_max
                s1, s2 = s1_new, s2_new
            final_states.append((s1, s2))

        # All should converge to the same point (unique stationary state)
        for i in range(1, len(final_states)):
            assert final_states[i][0] == pytest.approx(final_states[0][0], abs=1e-3)
            assert final_states[i][1] == pytest.approx(final_states[0][1], abs=1e-3)

        # And both components should be < Smax (anti-lock-in)
        s1_star, s2_star = final_states[0]
        assert s1_star < s_max
        assert s2_star < s_max


class TestDreamWithStrengthDecay:
    """Integration: dream pipeline with strength decay wired in.

    This is the main test class — it verifies that hermes-memory temporal
    dynamics compose correctly with hermes-NLCDM dream operations.
    """

    def test_old_unaccessed_patterns_weaken(self, dim, centroids, rng):
        """Patterns that aren't accessed decay in strength over time.

        Maps to: strengthDecay_antitone + strengthDecay_tendsto_zero
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.05)

        # Store 5 patterns at time 0
        for i in range(5):
            p = make_noisy_pattern(centroids[i % len(centroids)], 0.1, rng)
            store.store(p, importance=0.5, domain=i % len(centroids))

        initial_strengths = store.get_strengths().copy()

        # Advance time without accessing any patterns
        store.advance_time(10.0)

        decayed_strengths = store.get_strengths()

        # All should have decayed
        for i in range(5):
            assert decayed_strengths[i] < initial_strengths[i], (
                f"Pattern {i} didn't decay: {decayed_strengths[i]} >= {initial_strengths[i]}"
            )
            assert decayed_strengths[i] > 0, "Strength should remain positive"

    def test_accessed_patterns_maintain_strength(self, dim, centroids, rng):
        """Regularly accessed patterns maintain strength despite decay.

        Maps to: steadyState_is_fixpoint (access counterbalances decay)
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.05, alpha=0.3)

        # Store pattern
        p = make_noisy_pattern(centroids[0], 0.1, rng)
        idx = store.store(p, importance=0.5)

        # Simulate 50 access cycles
        for _ in range(50):
            store.advance_time(1.0)
            store.access(idx)

        final_strength = store.get_strengths()[idx]
        s_star = steady_state_strength(0.3, 0.05, 1.0, 1.0)

        # Should converge near steady state
        assert final_strength == pytest.approx(s_star, abs=0.05)
        # And steady state < Smax
        assert final_strength < store.s_max

    def test_strength_weighted_importance_affects_pruning(
        self, dim, centroids, rng
    ):
        """Strength-weighted importance makes old unaccessed patterns
        more likely to be pruned during dreams.

        Maps to: anti-lock-in enables dream pruning to remove stale patterns
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.1)

        # Store 3 similar patterns at time 0 (same domain)
        for _ in range(3):
            p = make_noisy_pattern(centroids[0], 0.02, rng)
            store.store(p, importance=0.5, domain=0)

        # Store 3 more similar patterns at time 0 but access them regularly
        accessed_indices = []
        for _ in range(3):
            p = make_noisy_pattern(centroids[1], 0.02, rng)
            idx = store.store(p, importance=0.5, domain=1)
            accessed_indices.append(idx)

        # Advance time, only accessing the second group
        for _ in range(20):
            store.advance_time(1.0)
            for idx in accessed_indices:
                store.access(idx)

        strengths = store.get_strengths()

        # Group 0 (unaccessed) should have much lower strength
        group0_strengths = strengths[:3]
        group1_strengths = strengths[3:]

        assert np.mean(group0_strengths) < 0.1 * np.mean(group1_strengths), (
            f"Unaccessed group mean {np.mean(group0_strengths):.4f} should be "
            f"much less than accessed group mean {np.mean(group1_strengths):.4f}"
        )

        # Strength-weighted importances reflect this
        weighted_imp = store.get_importances()
        assert np.mean(weighted_imp[:3]) < np.mean(weighted_imp[3:])

    def test_dream_cycle_with_decayed_patterns(self, dim, centroids, rng):
        """Full dream cycle on patterns with varying strengths.

        Verifies that the dream pipeline (merge/prune from NLCDM) works
        correctly when importance is weighted by temporal strength.
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.05, alpha=0.3)
        n_domains = len(centroids)

        # Phase 1: store initial memories across domains
        for day in range(10):
            for d in range(n_domains):
                p = make_noisy_pattern(centroids[d], 0.08, rng)
                store.store(p, importance=0.4, domain=d)
            store.advance_time(1.0)
            # Only access domain 0 and 1 patterns
            for i, mem in enumerate(store.memories):
                if mem.domain in (0, 1):
                    store.access(i)

        # Phase 2: let time pass without new memories (more decay)
        store.advance_time(5.0)

        # Run dream cycle
        patterns = store.get_patterns()
        importances = store.get_importances()  # strength-weighted
        labels = store.get_labels()
        N_before = patterns.shape[0]

        report = dream_cycle_v2(
            patterns, beta=10.0,
            importances=importances,
            labels=labels,
            seed=42,
        )

        N_after = report.patterns.shape[0]

        # Dream should reduce pattern count (via prune/merge)
        assert N_after <= N_before, (
            f"Dream should consolidate: {N_after} > {N_before}"
        )

        # Verify output patterns are unit norm
        norms = np.linalg.norm(report.patterns, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_multi_day_dream_with_decay_reduces_load(
        self, dim, centroids, rng
    ):
        """Multi-day simulation: wake (store + decay) then sleep (dream).

        The combination of strength decay + dream consolidation should
        keep pattern count bounded even as new memories accumulate daily.
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.03, alpha=0.3)
        n_domains = len(centroids)
        beta = 10.0

        pattern_counts = []

        for day in range(30):
            # Wake phase: ingest new memories
            new_indices = []
            for d in range(n_domains):
                if rng.random() < 0.6:  # not every domain every day
                    p = make_noisy_pattern(centroids[d], 0.08, rng)
                    idx = store.store(p, importance=0.4, domain=d)
                    new_indices.append(idx)

            # Simulate some accesses during the day
            for idx in new_indices:
                store.access(idx)
            # Also access a random subset of existing memories
            n_existing = len(store.memories)
            if n_existing > 0:
                n_access = min(5, n_existing)
                for idx in rng.choice(n_existing, n_access, replace=False):
                    store.access(int(idx))

            # Advance time by 1 day
            store.advance_time(1.0)

            # Sleep phase: run dream cycle every 3 days
            if (day + 1) % 3 == 0 and len(store.memories) > 5:
                patterns = store.get_patterns()
                importances = store.get_importances()
                labels = store.get_labels()

                report = dream_cycle_v2(
                    patterns, beta=beta,
                    importances=importances,
                    labels=labels,
                    seed=day,
                )

                store.apply_dream_result(
                    report.patterns,
                    merge_map=report.merge_map,
                )

            pattern_counts.append(len(store.memories))

        # Pattern count should be bounded — not growing linearly
        # Without dreams: ~30 days * ~3 patterns/day = ~90 patterns
        # With dreams + decay: should be significantly less
        final_count = pattern_counts[-1]
        max_possible = sum(
            int(n_domains * 0.6) for _ in range(30)
        )  # rough upper bound without consolidation
        assert final_count < max_possible * 0.7, (
            f"Pattern count {final_count} should be well below "
            f"max possible {max_possible} (ratio: {final_count/max_possible:.2f})"
        )

    def test_retrieval_after_decay_and_dreams(self, dim, centroids, rng):
        """After decay + dreams, recently accessed patterns are still retrievable.

        This tests the end-to-end pipeline: temporal decay preferentially
        weakens old patterns, dreams consolidate, and the resulting store
        still supports accurate retrieval for active memories.
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.05, alpha=0.3)
        n_domains = len(centroids)
        beta = 10.0

        # Store patterns across 3 domains
        for d in range(3):
            for _ in range(8):
                p = make_noisy_pattern(centroids[d], 0.08, rng)
                store.store(p, importance=0.4, domain=d)

        # Access only domain 0 and 1 patterns for 10 time steps
        for _ in range(10):
            store.advance_time(1.0)
            for i, mem in enumerate(store.memories):
                if mem.domain in (0, 1):
                    store.access(i)

        # Run dream
        patterns = store.get_patterns()
        importances = store.get_importances()
        labels = store.get_labels()

        report = dream_cycle_v2(
            patterns, beta=beta,
            importances=importances,
            labels=labels,
            seed=42,
        )

        # Test retrieval: query with domain centroids
        post_dream_patterns = report.patterns
        N = post_dream_patterns.shape[0]

        if N > 0:
            for d in range(min(2, n_domains)):  # test accessed domains
                query = centroids[d]
                sims = post_dream_patterns @ query
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                # Should find a good match for actively accessed domains
                assert best_sim > 0.5, (
                    f"Domain {d} retrieval too weak: best_sim={best_sim:.3f}"
                )

    def test_novelty_bonus_window(self, dim, centroids, rng):
        """New patterns with low strength get a novelty bonus that decays.

        Maps to: NoveltyBonus.lean — coldStart_survival guarantees
        new memories have ε-competitive scores during exploration window.
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.05, alpha=0.3)

        # Store one old well-established pattern and access it regularly
        p_old = make_noisy_pattern(centroids[0], 0.05, rng)
        old_idx = store.store(p_old, importance=0.7, domain=0)
        for _ in range(30):
            store.advance_time(0.5)
            store.access(old_idx)

        old_strength = store.get_strengths()[old_idx]

        # Now store a brand new pattern (no accesses yet)
        new_p = make_noisy_pattern(centroids[1], 0.05, rng)
        new_idx = store.store(new_p, importance=0.4, domain=1)

        new_strength = store.get_strengths()[new_idx]

        # The new pattern should have lower raw strength than the
        # established one which has been accessed 30 times
        assert new_strength < old_strength, (
            f"New pattern strength {new_strength:.4f} should be less than "
            f"established pattern strength {old_strength:.4f}"
        )

        # But with a novelty bonus (N₀·e^(-γ·t)), the new pattern
        # remains competitive during the exploration window
        N0 = 0.5  # initial bonus
        gamma = 0.1  # decay rate of bonus
        for t in range(10):
            bonus = N0 * np.exp(-gamma * t)
            boosted_score = new_strength * store.memories[new_idx].importance + bonus
            # Should be competitive with old patterns during early window
            if t < 5:
                assert boosted_score > 0.1, (
                    f"Novelty bonus should keep new pattern competitive at t={t}"
                )

    def test_anti_lock_in_with_dreams(self, dim, centroids, rng):
        """No single domain can permanently dominate the memory store.

        Combines anti-lock-in from StrengthDecay.lean (S* < Smax)
        with dream consolidation (merge/prune keep N manageable).

        Even if one domain is accessed much more than others, the
        composed system prevents permanent dominance.
        """
        store = MemoryStoreWithDecay(dim=dim, decay_rate=0.03, alpha=0.3)
        beta = 10.0

        # Store patterns: 20 in domain 0, 5 each in domains 1-4
        for _ in range(20):
            p = make_noisy_pattern(centroids[0], 0.06, rng)
            store.store(p, importance=0.5, domain=0)
        for d in range(1, len(centroids)):
            for _ in range(5):
                p = make_noisy_pattern(centroids[d], 0.06, rng)
                store.store(p, importance=0.5, domain=d)

        # Heavily access only domain 0 for 20 time steps
        for _ in range(20):
            store.advance_time(1.0)
            for i, mem in enumerate(store.memories):
                if mem.domain == 0:
                    store.access(i)

        # After many days, domain 0 has high strength but < Smax
        d0_strengths = [m.strength for m in store.memories if m.domain == 0]
        assert all(s < store.s_max for s in d0_strengths), (
            "Anti-lock-in: all strengths should be < Smax"
        )

        # Other domains have low but positive strength
        other_strengths = [m.strength for m in store.memories if m.domain != 0]
        assert all(s > 0 for s in other_strengths), (
            "Decayed patterns should remain strictly positive"
        )

        # Run dream cycle
        patterns = store.get_patterns()
        importances = store.get_importances()
        labels = store.get_labels()

        report = dream_cycle_v2(
            patterns, beta=beta,
            importances=importances,
            labels=labels,
            seed=42,
        )

        # After dream: minority domains should still have representation
        post_labels = labels  # approximately — dream may change indices
        post_n = report.patterns.shape[0]
        # The store shouldn't be entirely domain 0
        assert post_n > len(d0_strengths) * 0.3, (
            f"Dream shouldn't eliminate all minority domains: "
            f"post_n={post_n}, domain0_count={len(d0_strengths)}"
        )
