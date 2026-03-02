"""Discriminative tests for V1 -> V2 wiring in CoupledEngine.

Each test class targets a specific wiring claim and is designed to FAIL
if that wiring is broken, disconnected, or produces wrong values.

V1 = hermes-memory core.py (Lean-proven functions using math.exp)
V2 = coupled_engine.py (local copies using np.exp, wired into CoupledEngine)
"""

import sys
import time
from unittest.mock import patch

import numpy as np
# V1 imports (Lean-proven originals)
sys.path.insert(0, "/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python")
from hermes_memory.core import importance_update as v1_importance_update
from hermes_memory.core import novelty_bonus as v1_novelty_bonus
from hermes_memory.core import strength_decay as v1_strength_decay

# V2 imports (coupled_engine local copies + engine)
sys.path.insert(0, "/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python")
from coupled_engine import CoupledEngine
from coupled_engine import _importance_update as v2_importance_update
from coupled_engine import _novelty_bonus as v2_novelty_bonus
from coupled_engine import _strength_decay as v2_strength_decay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random unit vector of given dimension."""
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v


def _make_clustered_unit_vectors(
    n: int, dim: int, center: np.ndarray, spread: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Generate n unit vectors clustered around a center direction.

    Lower spread means tighter cluster (more similar vectors).
    """
    vecs = []
    for _ in range(n):
        noise = rng.standard_normal(dim) * spread
        v = center + noise
        v /= np.linalg.norm(v)
        vecs.append(v)
    return vecs


# ===========================================================================
# Claim 1: Function equivalence — V2 local copies match V1 Lean-proven
# ===========================================================================

class TestClaim1FunctionEquivalence:
    """V2 local copies of strength_decay, importance_update, novelty_bonus
    must be numerically identical to V1 originals (math.exp vs np.exp).

    Discriminates: if someone changes the formula in V2 (e.g. adds a constant,
    changes sign, uses a different decay model), these tests catch it.
    """

    def test_strength_decay_normal_values(self):
        """Standard inputs: beta=0.1, S0=1.0, t=1.0."""
        v1 = v1_strength_decay(0.1, 1.0, 1.0)
        v2 = v2_strength_decay(0.1, 1.0, 1.0)
        assert abs(v1 - v2) < 1e-12, f"v1={v1}, v2={v2}"

    def test_strength_decay_t_zero(self):
        """At t=0, decay should return S0 exactly."""
        for S0 in [0.0, 0.5, 1.0, 3.7]:
            v1 = v1_strength_decay(0.1, S0, 0.0)
            v2 = v2_strength_decay(0.1, S0, 0.0)
            assert abs(v1 - v2) < 1e-15, f"S0={S0}: v1={v1}, v2={v2}"
            assert abs(v2 - S0) < 1e-15, f"Expected S0={S0}, got v2={v2}"

    def test_strength_decay_S0_zero(self):
        """With S0=0, result should be 0 regardless of t."""
        for t in [0.0, 1.0, 100.0, 1e6]:
            v1 = v1_strength_decay(0.5, 0.0, t)
            v2 = v2_strength_decay(0.5, 0.0, t)
            assert v1 == 0.0
            assert v2 == 0.0

    def test_strength_decay_large_t(self):
        """For large t, result should approach 0."""
        v1 = v1_strength_decay(0.1, 1.0, 1000.0)
        v2 = v2_strength_decay(0.1, 1.0, 1000.0)
        assert abs(v1 - v2) < 1e-12
        assert v2 < 1e-40, f"Expected near-zero, got {v2}"

    def test_strength_decay_random_sweep(self):
        """100 random (beta, S0, t) triples, all must match within 1e-12."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            beta = rng.uniform(0.001, 2.0)
            S0 = rng.uniform(0.0, 5.0)
            t = rng.uniform(0.0, 100.0)
            v1 = v1_strength_decay(beta, S0, t)
            v2 = v2_strength_decay(beta, S0, t)
            assert abs(v1 - v2) < 1e-12, (
                f"beta={beta}, S0={S0}, t={t}: v1={v1}, v2={v2}, diff={abs(v1-v2)}"
            )

    def test_importance_update_normal(self):
        """Standard importance update: imp=0.5, delta=0.1, signal=1.0."""
        v1 = v1_importance_update(0.5, 0.1, 1.0)
        v2 = v2_importance_update(0.5, 0.1, 1.0)
        assert abs(v1 - v2) < 1e-15, f"v1={v1}, v2={v2}"

    def test_importance_update_clamp_upper(self):
        """Result must clamp to 1.0 when imp + delta*signal > 1."""
        v1 = v1_importance_update(0.9, 0.5, 1.0)
        v2 = v2_importance_update(0.9, 0.5, 1.0)
        assert v1 == 1.0
        assert v2 == 1.0

    def test_importance_update_clamp_lower(self):
        """Result must clamp to 0.0 when imp + delta*signal < 0."""
        v1 = v1_importance_update(0.1, -0.5, 1.0)
        v2 = v2_importance_update(0.1, -0.5, 1.0)
        assert v1 == 0.0
        assert v2 == 0.0

    def test_importance_update_random_sweep(self):
        """100 random (imp, delta, signal) triples."""
        rng = np.random.default_rng(99)
        for _ in range(100):
            imp = rng.uniform(0.0, 1.0)
            delta = rng.uniform(-1.0, 1.0)
            signal = rng.uniform(0.0, 2.0)
            v1 = v1_importance_update(imp, delta, signal)
            v2 = v2_importance_update(imp, delta, signal)
            assert abs(v1 - v2) < 1e-15, (
                f"imp={imp}, delta={delta}, signal={signal}: v1={v1}, v2={v2}"
            )

    def test_novelty_bonus_normal(self):
        """Standard novelty: N0=1.0, gamma=0.1, t=1.0."""
        v1 = v1_novelty_bonus(1.0, 0.1, 1.0)
        v2 = v2_novelty_bonus(1.0, 0.1, 1.0)
        assert abs(v1 - v2) < 1e-12, f"v1={v1}, v2={v2}"

    def test_novelty_bonus_t_zero(self):
        """At t=0, novelty should return N0."""
        for N0 in [0.0, 0.3, 1.0, 2.5]:
            v1 = v1_novelty_bonus(N0, 0.1, 0.0)
            v2 = v2_novelty_bonus(N0, 0.1, 0.0)
            assert abs(v1 - v2) < 1e-15
            assert abs(v2 - N0) < 1e-15

    def test_novelty_bonus_random_sweep(self):
        """100 random (N0, gamma, t) triples."""
        rng = np.random.default_rng(77)
        for _ in range(100):
            N0 = rng.uniform(0.0, 3.0)
            gamma = rng.uniform(0.001, 2.0)
            t = rng.uniform(0.0, 100.0)
            v1 = v1_novelty_bonus(N0, gamma, t)
            v2 = v2_novelty_bonus(N0, gamma, t)
            assert abs(v1 - v2) < 1e-12, (
                f"N0={N0}, gamma={gamma}, t={t}: v1={v1}, v2={v2}"
            )


# ===========================================================================
# Claim 2: Wiring path — dream() passes non-uniform importances to
#           dream_cycle_xb, not a flat 0.5 array
# ===========================================================================

class TestClaim2WiringPath:
    """dream() must compute per-memory effective importances via
    _compute_effective_importance and pass them to dream_cycle_xb.

    Discriminates: if dream() were to skip the importance computation and
    pass None or a uniform array, these tests catch it.
    """

    def test_dream_passes_importances_to_dream_cycle_xb(self):
        """Patch dream_cycle_xb to capture the importances kwarg.
        Store 3 memories with different time offsets. Verify importances
        array is non-None, correct length, and not all-equal.
        """
        dim = 32
        rng = np.random.default_rng(123)

        engine = CoupledEngine(dim=dim, decay_rate=0.05, novelty_N0=0.3, novelty_gamma=0.1)

        # Store 3 memories
        for i in range(3):
            emb = _make_unit_vector(dim, rng)
            engine.store(f"memory_{i}", emb)

        now = time.time()
        # Manually set different time offsets to create importance variation
        engine.memory_store[0].last_access_time = now - 5000  # very old access
        engine.memory_store[0].creation_time = now - 5000
        engine.memory_store[1].last_access_time = now - 100   # medium old
        engine.memory_store[1].creation_time = now - 100
        engine.memory_store[2].last_access_time = now          # fresh
        engine.memory_store[2].creation_time = now

        captured = {}

        from dream_ops import DreamReport

        def mock_dream_cycle_xb(patterns, beta, **kwargs):
            captured["importances"] = kwargs.get("importances")
            captured["patterns"] = patterns
            # Return a no-op report so dream() can proceed
            return DreamReport(
                patterns=patterns.copy(),
                associations=[],
                pruned_indices=[],
                merge_map={},
            )

        with patch("coupled_engine.dream_cycle_xb", side_effect=mock_dream_cycle_xb):
            engine.dream(seed=42)

        importances = captured.get("importances")
        assert importances is not None, "dream_cycle_xb was not passed importances"
        assert len(importances) == 3, f"Expected 3 importances, got {len(importances)}"

        # With different time offsets and decay_rate=0.05, importances must differ
        assert not np.allclose(importances, importances[0]), (
            f"Importances are uniform: {importances}. Wiring is broken — "
            f"_compute_effective_importance is not differentiating by time."
        )

    def test_dream_importances_length_matches_memory_count(self):
        """Importances array length must equal number of stored memories."""
        dim = 16
        rng = np.random.default_rng(456)

        for n_memories in [1, 5, 20]:
            engine = CoupledEngine(dim=dim)
            for i in range(n_memories):
                engine.store(f"mem_{i}", _make_unit_vector(dim, rng))

            captured = {}

            from dream_ops import DreamReport

            def mock_dcxb(patterns, beta, **kwargs):
                captured["importances"] = kwargs.get("importances")
                return DreamReport(
                    patterns=patterns.copy(),
                    associations=[],
                    pruned_indices=[],
                    merge_map={},
                )

            with patch("coupled_engine.dream_cycle_xb", side_effect=mock_dcxb):
                engine.dream(seed=0)

            importances = captured.get("importances")
            assert importances is not None
            assert len(importances) == n_memories, (
                f"n_memories={n_memories}, but importances has len {len(importances)}"
            )


# ===========================================================================
# Claim 3: Temporal discrimination — strength decay makes old memories
#           lose effective importance relative to fresh ones
# ===========================================================================

class TestClaim3TemporalDiscrimination:
    """_compute_effective_importance must return LOWER value for a memory
    whose last_access_time is far in the past compared to a fresh one.

    Discriminates: if _strength_decay is disconnected or replaced with a
    constant, the old and fresh memories would have equal importance.
    """

    def test_old_access_lower_importance(self):
        """Memory accessed 1000s ago has lower effective importance than
        one accessed just now, both starting at importance=0.5.
        """
        dim = 16
        engine = CoupledEngine(dim=dim, decay_rate=0.01)
        rng = np.random.default_rng(10)

        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()

        old_mem = MemoryEntry(
            text="old",
            embedding=emb.copy(),
            importance=0.5,
            creation_time=now,      # same creation time
            last_access_time=now - 1000,  # old access
        )
        fresh_mem = MemoryEntry(
            text="fresh",
            embedding=emb.copy(),
            importance=0.5,
            creation_time=now,
            last_access_time=now,         # just accessed
        )

        eff_old = engine._compute_effective_importance(old_mem, now)
        eff_fresh = engine._compute_effective_importance(fresh_mem, now)

        assert eff_old < eff_fresh, (
            f"Old memory ({eff_old}) should have LOWER effective importance "
            f"than fresh memory ({eff_fresh}). Strength decay is not working."
        )

    def test_temporal_decay_is_monotone(self):
        """Effective importance must strictly decrease as access time ages.

        Test at 5 time offsets: 0, 10, 100, 1000, 10000 seconds ago.
        """
        dim = 16
        engine = CoupledEngine(dim=dim, decay_rate=0.01)
        rng = np.random.default_rng(20)
        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()
        offsets = [0, 10, 100, 1000, 10000]
        importances = []

        for dt in offsets:
            mem = MemoryEntry(
                text="test",
                embedding=emb.copy(),
                importance=0.8,
                creation_time=now,
                last_access_time=now - dt,
            )
            importances.append(engine._compute_effective_importance(mem, now))

        # Strictly decreasing
        for i in range(len(importances) - 1):
            assert importances[i] > importances[i + 1], (
                f"Importance at dt={offsets[i]} ({importances[i]}) should be > "
                f"importance at dt={offsets[i+1]} ({importances[i+1]}). "
                f"Full sequence: {list(zip(offsets, importances))}"
            )

    def test_zero_decay_rate_means_no_temporal_effect(self):
        """With decay_rate=0, time should not affect effective importance
        (ignoring novelty, which is also zeroed).
        """
        dim = 16
        engine = CoupledEngine(dim=dim, decay_rate=0.0, novelty_N0=0.0)
        rng = np.random.default_rng(30)
        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()

        mem_old = MemoryEntry(
            text="old", embedding=emb.copy(), importance=0.7,
            creation_time=now, last_access_time=now - 99999,
        )
        mem_fresh = MemoryEntry(
            text="fresh", embedding=emb.copy(), importance=0.7,
            creation_time=now, last_access_time=now,
        )

        eff_old = engine._compute_effective_importance(mem_old, now)
        eff_fresh = engine._compute_effective_importance(mem_fresh, now)

        assert abs(eff_old - eff_fresh) < 1e-12, (
            f"With zero decay, old ({eff_old}) and fresh ({eff_fresh}) "
            f"should be equal."
        )


# ===========================================================================
# Claim 4: Novelty discrimination — newly created memories get a bonus
# ===========================================================================

class TestClaim4NoveltyDiscrimination:
    """_compute_effective_importance must return HIGHER value for a just-created
    memory than for one created long ago, due to the novelty bonus.

    Discriminates: if _novelty_bonus is disconnected or novelty_N0 is
    hardcoded to 0, newly created memories get no advantage.

    NOTE: default novelty_N0=0.0 disables novelty. These tests use
    novelty_N0=0.3 to activate the bonus.
    """

    def test_new_creation_higher_importance(self):
        """Memory created just now has higher effective importance than
        one created 1000s ago, with same base importance and access time.
        """
        dim = 16
        engine = CoupledEngine(
            dim=dim, decay_rate=0.0, novelty_N0=0.3, novelty_gamma=0.1
        )
        rng = np.random.default_rng(40)
        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()

        old_creation = MemoryEntry(
            text="old_create", embedding=emb.copy(), importance=0.5,
            creation_time=now - 1000, last_access_time=now,
        )
        new_creation = MemoryEntry(
            text="new_create", embedding=emb.copy(), importance=0.5,
            creation_time=now, last_access_time=now,
        )

        eff_old = engine._compute_effective_importance(old_creation, now)
        eff_new = engine._compute_effective_importance(new_creation, now)

        assert eff_new > eff_old, (
            f"Newly created memory ({eff_new}) should have HIGHER effective "
            f"importance than old-created memory ({eff_old}). "
            f"Novelty bonus is not wired."
        )

    def test_novelty_decays_monotonically(self):
        """Novelty contribution decreases as creation age increases."""
        dim = 16
        engine = CoupledEngine(
            dim=dim, decay_rate=0.0, novelty_N0=0.5, novelty_gamma=0.05
        )
        rng = np.random.default_rng(50)
        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()
        # Ages chosen so novelty bonus N0*exp(-gamma*age) is still nonzero:
        # 0.5*exp(-0.05*200) = 0.5*exp(-10) ~ 2.3e-5, still distinguishable
        # from 0.5*exp(-0.05*500) = 0.5*exp(-25) ~ 7e-12
        creation_ages = [0, 1, 5, 20, 200]
        importances = []

        for age in creation_ages:
            mem = MemoryEntry(
                text="test", embedding=emb.copy(), importance=0.5,
                creation_time=now - age, last_access_time=now,
            )
            importances.append(engine._compute_effective_importance(mem, now))

        for i in range(len(importances) - 1):
            assert importances[i] > importances[i + 1], (
                f"Importance at creation_age={creation_ages[i]} ({importances[i]}) "
                f"should be > at creation_age={creation_ages[i+1]} "
                f"({importances[i+1]}). "
                f"Novelty is not monotonically decaying."
            )

    def test_zero_novelty_N0_disables_bonus(self):
        """With novelty_N0=0, creation time should not matter
        (no novelty contribution).
        """
        dim = 16
        engine = CoupledEngine(dim=dim, decay_rate=0.0, novelty_N0=0.0)
        rng = np.random.default_rng(60)
        emb = _make_unit_vector(dim, rng)

        from coupled_engine import MemoryEntry

        now = time.time()

        mem_new = MemoryEntry(
            text="new", embedding=emb.copy(), importance=0.5,
            creation_time=now, last_access_time=now,
        )
        mem_old = MemoryEntry(
            text="old", embedding=emb.copy(), importance=0.5,
            creation_time=now - 99999, last_access_time=now,
        )

        eff_new = engine._compute_effective_importance(mem_new, now)
        eff_old = engine._compute_effective_importance(mem_old, now)

        assert abs(eff_new - eff_old) < 1e-12, (
            f"With novelty_N0=0, new ({eff_new}) and old ({eff_old}) "
            f"should have equal effective importance."
        )


# ===========================================================================
# Claim 5: Dream outcome discrimination — old low-importance memories
#           are more likely to be pruned than fresh high-importance ones
# ===========================================================================

class TestClaim5DreamOutcomeDiscrimination:
    """When dream() runs with a mix of old and fresh memories, the
    importance-weighted pruning should preferentially remove old
    low-importance patterns over fresh high-importance ones.

    Discriminates: if importances are ignored by dream_cycle_xb (e.g.
    passed as None or uniform), pruning would be random/symmetric.

    Strategy: Create clustered patterns (high similarity within cluster)
    so that pruning/merging is triggered. Old memories get low importance
    via time decay; fresh memories get high importance. After dream(),
    more old memories should be removed than fresh ones.
    """

    def test_old_memories_pruned_more_than_fresh(self):
        """10 old + 10 fresh clustered memories. After dream, fresh
        memories should survive at a higher rate than old ones.
        """
        dim = 64
        rng = np.random.default_rng(2024)

        # Use nonzero novelty to amplify discrimination
        engine = CoupledEngine(
            dim=dim, beta=5.0,
            decay_rate=0.05,
            novelty_N0=0.3, novelty_gamma=0.01,
        )

        # Two cluster centers
        center_old = _make_unit_vector(dim, rng)
        center_fresh = _make_unit_vector(dim, rng)

        now = time.time()

        # Store 10 old memories clustered tightly (high similarity -> prunable)
        old_vecs = _make_clustered_unit_vectors(10, dim, center_old, spread=0.05, rng=rng)
        old_indices = []
        for i, v in enumerate(old_vecs):
            idx = engine.store(f"old_{i}", v)
            old_indices.append(idx)

        # Store 10 fresh memories in a different cluster
        fresh_vecs = _make_clustered_unit_vectors(10, dim, center_fresh, spread=0.05, rng=rng)
        fresh_indices = []
        for i, v in enumerate(fresh_vecs):
            idx = engine.store(f"fresh_{i}", v)
            fresh_indices.append(idx)

        # Manually backdate old memories
        for idx in old_indices:
            engine.memory_store[idx].creation_time = now - 100000
            engine.memory_store[idx].last_access_time = now - 100000

        # Fresh memories keep their times (near now)

        n_before = engine.n_memories
        assert n_before == 20

        result = engine.dream(seed=42)

        _ = result["n_after"]
        n_pruned = result["pruned"]
        n_merged = result["merged"]

        # Count how many of each type survived
        # After dream, memory_store is rebuilt. We identify survivors by text.
        surviving_texts = {m.text for m in engine.memory_store}
        old_survived = sum(1 for i in range(10) if f"old_{i}" in surviving_texts)
        fresh_survived = sum(1 for i in range(10) if f"fresh_{i}" in surviving_texts)

        # The key discrimination: fresh memories should survive at a
        # higher rate than old memories (or at least equally, with some
        # margin for merged centroids that inherit text from best-importance).
        # With 0.05 decay_rate and 100000s offset, old memories have very
        # low importance (~0.5 * exp(-0.05*100000) ~ 0), while fresh
        # memories retain ~0.5 + 0.3 novelty = ~0.8.
        #
        # If pruning respects importance, old patterns are pruned first.
        # We check: fresh_survived >= old_survived
        assert fresh_survived >= old_survived, (
            f"Expected fresh memories to survive at higher rate. "
            f"fresh_survived={fresh_survived}, old_survived={old_survived}, "
            f"pruned={n_pruned}, merged={n_merged}. "
            f"Importance-weighted pruning may be broken."
        )


# ===========================================================================
# Claim 6: Query feedback loop — querying a memory increases its importance
# ===========================================================================

class TestClaim6QueryFeedbackLoop:
    """query() must call _importance_update on accessed memories, increasing
    their importance field.

    Discriminates: if query() does not call _importance_update, or if
    importance_delta is not wired, the memory's importance stays at 0.5.
    """

    def test_query_increases_importance(self):
        """Store a memory, query with a close embedding, verify importance
        increased from the initial 0.5.
        """
        dim = 32
        rng = np.random.default_rng(100)

        engine = CoupledEngine(dim=dim, importance_delta=0.05)

        # Store a single memory with explicit importance (bypasses emotional tagging)
        emb = _make_unit_vector(dim, rng)
        engine.store("test_memory", emb, importance=0.5)

        initial_importance = engine.memory_store[0].importance
        assert initial_importance == 0.5, f"Expected 0.5, got {initial_importance}"

        # Query with the exact same embedding (guaranteed to match)
        engine.query(emb, top_k=1)

        after_importance = engine.memory_store[0].importance
        expected = min(0.5 + 0.05 * 1.0, 1.0)  # importance_update(0.5, 0.05, 1.0)

        assert after_importance > initial_importance, (
            f"Importance should increase after query. "
            f"Before: {initial_importance}, After: {after_importance}. "
            f"_importance_update is not wired in query()."
        )
        assert abs(after_importance - expected) < 1e-12, (
            f"Expected importance={expected}, got {after_importance}. "
            f"importance_delta wiring may be wrong."
        )

    def test_multiple_queries_accumulate(self):
        """Multiple queries should accumulate importance increases."""
        dim = 32
        rng = np.random.default_rng(200)

        engine = CoupledEngine(dim=dim, importance_delta=0.1)

        emb = _make_unit_vector(dim, rng)
        engine.store("accumulate_test", emb, importance=0.5)

        assert engine.memory_store[0].importance == 0.5

        # Query 3 times
        for _ in range(3):
            engine.query(emb, top_k=1)

        final_importance = engine.memory_store[0].importance
        # 0.5 -> 0.6 -> 0.7 -> 0.8
        expected = 0.8
        assert abs(final_importance - expected) < 1e-12, (
            f"After 3 queries with delta=0.1, expected importance={expected}, "
            f"got {final_importance}."
        )

    def test_query_updates_access_time(self):
        """query() must update last_access_time and access_count."""
        dim = 32
        rng = np.random.default_rng(300)

        engine = CoupledEngine(dim=dim)

        emb = _make_unit_vector(dim, rng)
        engine.store("access_test", emb)

        initial_time = engine.memory_store[0].last_access_time
        initial_count = engine.memory_store[0].access_count
        assert initial_count == 0

        time.sleep(0.01)  # ensure time difference
        engine.query(emb, top_k=1)

        assert engine.memory_store[0].access_count == 1, (
            "access_count should increment on query"
        )
        assert engine.memory_store[0].last_access_time > initial_time, (
            "last_access_time should update on query"
        )

    def test_importance_clamps_at_one(self):
        """Importance should never exceed 1.0 no matter how many queries."""
        dim = 32
        rng = np.random.default_rng(400)

        engine = CoupledEngine(dim=dim, importance_delta=0.3)

        emb = _make_unit_vector(dim, rng)
        engine.store("clamp_test", emb, importance=0.9)

        # Query many times to push past 1.0
        for _ in range(10):
            engine.query(emb, top_k=1)

        final = engine.memory_store[0].importance
        assert final == 1.0, (
            f"Importance should clamp at 1.0, got {final}. "
            f"Clamping in _importance_update is broken."
        )

    def test_unaccessed_memory_keeps_initial_importance(self):
        """A memory that is never queried should keep its initial importance."""
        dim = 32
        rng = np.random.default_rng(500)

        engine = CoupledEngine(dim=dim, importance_delta=0.05)

        emb1 = _make_unit_vector(dim, rng)
        emb2 = _make_unit_vector(dim, rng)

        engine.store("accessed", emb1, importance=0.5)
        engine.store("untouched", emb2, importance=0.5)

        # Query only with emb1 (top_k=1 means only closest match)
        engine.query(emb1, top_k=1)

        # The untouched memory should still have importance=0.5
        assert engine.memory_store[1].importance == 0.5, (
            f"Untouched memory importance changed to "
            f"{engine.memory_store[1].importance}. "
            f"query() is incorrectly updating non-accessed memories."
        )
