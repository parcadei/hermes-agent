# Behavioral Specification: Temporal Hebbian Binding for CoupledEngine

Created: 2026-03-04
Author: architect-agent

## Overview

Add a **separate** weight matrix `W_temporal` to `CoupledEngine` that encodes
temporal co-occurrence between facts stored in the same session. When multiple
facts arrive together (same `send_message(memorizing=True)` call in MABench),
Hebbian binding creates symmetric cross-links:
`W_temporal += epsilon * (outer(a_i, a_j) + outer(a_j, a_i))` for all pairs
i != j in the session. This enables spreading activation to find cross-domain
associations that pure cosine similarity (X^T X) cannot.

## Problem Statement

Currently W = X^T X (sum of outer(x_i, x_i) self-loops) with zeroed diagonal.
This is algebraically equivalent to cosine similarity between stored patterns.
When a query asks about two facts that were co-presented but have low mutual
cosine similarity, spreading activation converges to the same result as cosine
retrieval -- it cannot discover the temporal association because W encodes only
self-similarity, not co-occurrence.

## Critical Design Constraint: Modern Hopfield Dynamics

**VERIFIED:** The functions `hopfield_update()` and `spreading_activation()` in
`dream_ops.py` take `patterns` (the X matrix), NOT W. They compute:

```
xi_new = sum_mu attn(beta, X^T xi)_mu * x_mu
```

This means the temporal signal **cannot** be injected by simply adding
W_temporal to the cached W property and expecting `spreading_activation()` to
pick it up. The spreading activation dynamics operate on pattern similarities
`X^T xi`, not on an explicit coupling matrix.

**Resolution:** We introduce a new `hopfield_update_biased()` function that
adds a W_temporal bias to the pattern similarities before applying attention:

```python
def hopfield_update_biased(
    beta: float,
    patterns: np.ndarray,       # (N, d)
    xi: np.ndarray,             # (d,)
    W_temporal: np.ndarray,     # (d, d) additive bias in embedding space
    attention_fn=None,
) -> np.ndarray:
    """Hopfield update with temporal bias injection.

    similarities = X^T xi + X^T (W_temporal @ xi) * temporal_scale
    weights = attn(beta, similarities)
    xi_new = weights @ X
    """
    if attention_fn is None:
        attention_fn = softmax
    base_similarities = patterns @ xi                     # (N,)
    temporal_probe = W_temporal @ xi                       # (d,)
    temporal_similarities = patterns @ temporal_probe      # (N,)
    t_norm = np.linalg.norm(temporal_similarities)
    if t_norm > 1e-12:
        temporal_similarities = temporal_similarities / t_norm
    combined = base_similarities + temporal_similarities   # (N,)
    weights = attention_fn(beta, combined)                 # (N,)
    return weights @ patterns                              # (d,)
```

The temporal probe `W_temporal @ xi` maps the query into "what was co-stored
with things like xi", then `X^T (W_temporal @ xi)` measures which stored
patterns are similar to that co-occurrence image. This additive bias tilts
attention toward temporally linked patterns without changing the base Hopfield
dynamics.

A corresponding `spreading_activation_biased()` iterates this update to
convergence.


## New State in CoupledEngine

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `_session_buffer` | `list[np.ndarray]` | `[]` | Embeddings accumulated during the current session |
| `_W_temporal` | `np.ndarray` | `zeros((dim, dim))` | Temporal co-occurrence coupling matrix |
| `hebbian_epsilon` | `float` | `0.01` | Default Hebbian binding strength |

### Constructor Changes

```python
def __init__(
    self,
    dim: int,
    ...,
    hebbian_epsilon: float = 0.01,   # NEW parameter
):
    ...
    self.hebbian_epsilon = hebbian_epsilon
    self._session_buffer: list[np.ndarray] = []
    self._W_temporal = np.zeros((dim, dim), dtype=np.float64)
```


## New Methods

### `flush_session(epsilon: float | None = None) -> int`

Flush the session buffer, creating Hebbian cross-links for all pairs.

**Signature:**
```python
def flush_session(self, epsilon: float | None = None) -> int:
```

**Parameters:**
- `epsilon`: Hebbian learning rate. If `None`, uses `self.hebbian_epsilon`.

**Returns:**
- Number of pairs bound (int). For N items in buffer, returns N*(N-1)//2.

**Behavior:**
1. If `len(self._session_buffer) <= 1`: clear buffer, return 0 (no pairs to bind).
2. For all pairs `(a_i, a_j)` where `i < j` in `_session_buffer`:
   - `self._W_temporal += eps * (np.outer(a_i, a_j) + np.outer(a_j, a_i))`
3. Clear `self._session_buffer`.
4. Return the number of pairs processed.

**Does NOT** call `_invalidate_cache()` -- W_temporal is separate from the
auto-derived W cache (which is X^T X from embeddings).


### `reset_temporal() -> None`

Reset temporal bindings to zero. Primarily for testing.

**Signature:**
```python
def reset_temporal(self) -> None:
```

**Behavior:**
1. `self._W_temporal = np.zeros((self.dim, self.dim), dtype=np.float64)`
2. `self._session_buffer.clear()`


### `W_full` property

Combined coupling matrix for diagnostics and metrics.

**Signature:**
```python
@property
def W_full(self) -> np.ndarray:
```

**Returns:** `self.W + self._W_temporal`

**Note:** This is provided for diagnostics/metrics only. The query methods use
`_W_temporal` directly via the biased Hopfield update, not via this property.


## Modified Methods

### `store()` -- append to session buffer

After the existing store logic (line 345 in current code), append the stored
embedding to the session buffer:

```python
# At end of store(), after FAISS index update:
self._session_buffer.append(emb.copy())
return idx
```

**Contradiction replacement case:** When a contradiction is detected and an
existing entry is replaced (line 298-323), the NEW embedding (not the old one)
is appended to the session buffer. The old embedding is no longer in the store
and should not participate in Hebbian binding.

**Invariant:** `len(self._session_buffer)` increases by exactly 1 per `store()`
call, regardless of whether a contradiction replacement occurred.


### `query_associative()` -- use biased spreading activation

Replace the spreading activation call (line 479-485) with the biased variant:

```python
# Current:
converged = spreading_activation(
    beta=beta, patterns=embeddings, xi=emb / emb_norm,
    max_steps=20, attention_fn=sparsemax if sparse else None,
)

# New:
if np.any(self._W_temporal != 0):
    converged = spreading_activation_biased(
        beta=beta, patterns=embeddings, xi=emb / emb_norm,
        W_temporal=self._W_temporal,
        max_steps=20, attention_fn=sparsemax if sparse else None,
    )
else:
    converged = spreading_activation(
        beta=beta, patterns=embeddings, xi=emb / emb_norm,
        max_steps=20, attention_fn=sparsemax if sparse else None,
    )
```

The guard `np.any(self._W_temporal != 0)` avoids the overhead of the biased
path when no temporal bindings exist (backward-compatible fast path).


### `query_hybrid()` -- use biased Hopfield update

Replace the single Hopfield step (line 585-589) with the biased variant:

```python
# Current:
expanded = hopfield_update(
    beta=beta, patterns=embeddings, xi=probe, attention_fn=sparsemax,
)

# New:
if np.any(self._W_temporal != 0):
    expanded = hopfield_update_biased(
        beta=beta, patterns=embeddings, xi=probe,
        W_temporal=self._W_temporal, attention_fn=sparsemax,
    )
else:
    expanded = hopfield_update(
        beta=beta, patterns=embeddings, xi=probe, attention_fn=sparsemax,
    )
```


### `save()` -- persist W_temporal

Add `W_temporal` to the `.npz` file:

```python
np.savez(
    str(path) + ".npz",
    embeddings=embeddings,
    dim=np.array([self.dim]),
    beta=np.array([self.beta]),
    W_temporal=self._W_temporal,         # NEW
)
```

Add `hebbian_epsilon` to the JSON metadata:

```python
metadata = {
    ...
    "hebbian_epsilon": self.hebbian_epsilon,
}
```


### `load()` -- restore W_temporal

After constructing the engine, restore W_temporal:

```python
engine.hebbian_epsilon = metadata.get("hebbian_epsilon", 0.01)

if "W_temporal" in data:
    engine._W_temporal = data["W_temporal"]
else:
    engine._W_temporal = np.zeros((dim, dim), dtype=np.float64)
```

**Backward compatibility:** Old save files that lack `W_temporal` get a zero
matrix, preserving existing behavior.


### `dream()` -- W_temporal survives dreaming

The dream cycle operates on the patterns matrix X (via `dream_cycle_xb`), NOT
on W or W_temporal. When dream prunes or merges patterns, the memory_store
changes but W_temporal remains in embedding space and does not need index-based
remapping.

**No changes required to dream().** W_temporal encodes relationships in
embedding-space coordinates `(d, d)`, not pattern-index coordinates `(N, N)`.
The Hebbian links `outer(a_i, a_j)` embed the directional relationship between
two embedding vectors, which remains valid regardless of which patterns survive
dreaming.


## New Functions in `dream_ops.py`

### `hopfield_update_biased()`

```python
def hopfield_update_biased(
    beta: float,
    patterns: np.ndarray,
    xi: np.ndarray,
    W_temporal: np.ndarray,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Single Hopfield update with additive temporal bias.

    similarities = X xi + normalize(X (W_temporal xi))
    weights = attn(beta, similarities)
    xi_new = weights @ X

    The temporal probe W_temporal @ xi maps the query into the space of
    "things co-stored with patterns similar to xi". Projecting this back
    through X^T yields per-pattern temporal affinity scores that tilt
    attention toward temporally associated patterns.

    Args:
        beta: inverse temperature
        patterns: (N, d) stored patterns
        xi: (d,) current state vector
        W_temporal: (d, d) temporal co-occurrence matrix
        attention_fn: attention function (default: softmax)

    Returns:
        (d,) updated state vector
    """
```


### `spreading_activation_biased()`

```python
def spreading_activation_biased(
    beta: float,
    patterns: np.ndarray,
    xi: np.ndarray,
    W_temporal: np.ndarray,
    max_steps: int = 50,
    tol: float = 1e-6,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Iterate hopfield_update_biased until convergence.

    Same convergence criterion as spreading_activation: ||x_{t+1} - x_t|| < tol.

    Args:
        beta: inverse temperature
        patterns: (N, d) stored patterns
        xi: (d,) initial state
        W_temporal: (d, d) temporal co-occurrence matrix
        max_steps: maximum iterations
        tol: convergence tolerance
        attention_fn: attention function (default: softmax)

    Returns:
        (d,) converged state vector
    """
```


## MABench Integration: `HermesMemoryAgent._store()`

### Current behavior (VERIFIED at line 304-337)

```python
def _store(self, text: str) -> str:
    facts = self._parse_facts(text)
    embeddings = self._scorer.embed_batch(fact_texts)
    for (sn, fact_text), emb in zip(facts, embeddings):
        self.orchestrator.store(content=fact_text, embedding=emb)
        self.coupled_engine.store(text=original_text, embedding=emb, recency=...)
        self._store_count += 1
    # Optional dream cycle
    if self.dream_interval > 0 and self._store_count % self.dream_interval == 0:
        self.coupled_engine.dream()
    return "Memorized"
```

### Required change

Add `flush_session()` call after the per-fact loop, before the dream cycle:

```python
def _store(self, text: str) -> str:
    facts = self._parse_facts(text)
    embeddings = self._scorer.embed_batch(fact_texts)
    for (sn, fact_text), emb in zip(facts, embeddings):
        self.orchestrator.store(content=fact_text, embedding=emb)
        self.coupled_engine.store(text=original_text, embedding=emb, recency=...)
        self._store_count += 1

    # Flush temporal session: bind all facts in this chunk via Hebbian links
    self.coupled_engine.flush_session()

    # Optional dream cycle
    if self.dream_interval > 0 and self._store_count % self.dream_interval == 0:
        self.coupled_engine.dream()
    return "Memorized"
```

**Granularity:** One flush per `send_message(memorizing=True)` call, which
corresponds to one chunk in FactConsolidation. Each chunk typically contains
10-20 facts that were presented together as a coherent unit.


### `reset()` changes

The `reset()` method creates a new `CoupledEngine` instance, so W_temporal and
the session buffer are naturally zeroed. No changes required IF the constructor
initializes the new fields. The `hebbian_epsilon` parameter should be forwarded:

```python
self.coupled_engine = CoupledEngine(
    dim=self.dim,
    ...,
    hebbian_epsilon=self.coupled_engine.hebbian_epsilon,  # preserve setting
)
```


## Behavioral Contracts (Testable Assertions)

### Contract 1: Session buffering accumulates without side effects

```python
def test_session_buffer_accumulates():
    engine = CoupledEngine(dim=8)
    a = random_unit_vector(8, seed=0)
    b = random_unit_vector(8, seed=1)

    engine.store("fact_a", a)
    engine.store("fact_b", b)

    # Session buffer has 2 embeddings
    assert len(engine._session_buffer) == 2
    # W_temporal is still zero -- no flush yet
    assert np.allclose(engine._W_temporal, 0.0)
    # W (auto-derived) is unaffected by session buffer
    W_before = engine.W.copy()
    # W should just be outer(a,a) + outer(b,b) with zeroed diagonal
    expected = np.outer(a, a) + np.outer(b, b)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(engine.W, expected, atol=1e-12)
```

### Contract 2: Flush creates symmetric cross-links

```python
def test_flush_creates_symmetric_cross_links():
    engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
    a = random_unit_vector(8, seed=0)
    b = random_unit_vector(8, seed=1)

    engine.store("fact_a", a)
    engine.store("fact_b", b)
    n_pairs = engine.flush_session()

    assert n_pairs == 1  # C(2,2) = 1 pair
    expected = 0.1 * (np.outer(a, b) + np.outer(b, a))
    np.testing.assert_allclose(engine._W_temporal, expected, atol=1e-12)
    # Symmetry
    np.testing.assert_allclose(engine._W_temporal, engine._W_temporal.T, atol=1e-12)
```

### Contract 3: Flush clears buffer

```python
def test_flush_clears_buffer():
    engine = CoupledEngine(dim=8)
    engine.store("a", random_unit_vector(8, seed=0))
    engine.store("b", random_unit_vector(8, seed=1))
    assert len(engine._session_buffer) == 2

    engine.flush_session()
    assert len(engine._session_buffer) == 0
```

### Contract 4: W_temporal is additive across flushes

```python
def test_multiple_flushes_accumulate():
    engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)

    # Session 1: facts a, b
    a = random_unit_vector(8, seed=0)
    b = random_unit_vector(8, seed=1)
    engine.store("a", a)
    engine.store("b", b)
    engine.flush_session()
    W_after_1 = engine._W_temporal.copy()

    # Session 2: facts c, d
    c = random_unit_vector(8, seed=2)
    d = random_unit_vector(8, seed=3)
    engine.store("c", c)
    engine.store("d", d)
    engine.flush_session()
    W_after_2 = engine._W_temporal.copy()

    # W_temporal grew
    assert np.linalg.norm(W_after_2) > np.linalg.norm(W_after_1)
    # First session's contribution is still present
    expected_1 = 0.1 * (np.outer(a, b) + np.outer(b, a))
    expected_2 = expected_1 + 0.1 * (np.outer(c, d) + np.outer(d, c))
    np.testing.assert_allclose(engine._W_temporal, expected_2, atol=1e-12)
```

### Contract 5: Epsilon controls binding strength linearly

```python
def test_epsilon_scales_linearly():
    dim = 8
    a = random_unit_vector(dim, seed=0)
    b = random_unit_vector(dim, seed=1)

    engine_1x = CoupledEngine(dim=dim, hebbian_epsilon=0.01)
    engine_1x.store("a", a)
    engine_1x.store("b", b)
    engine_1x.flush_session()

    engine_5x = CoupledEngine(dim=dim, hebbian_epsilon=0.05)
    engine_5x.store("a", a)
    engine_5x.store("b", b)
    engine_5x.flush_session()

    ratio = np.linalg.norm(engine_5x._W_temporal) / np.linalg.norm(engine_1x._W_temporal)
    np.testing.assert_allclose(ratio, 5.0, atol=1e-10)
```

### Contract 6: Spreading activation uses temporal bindings

```python
def test_query_associative_finds_temporal_link():
    """Two facts with low cosine similarity but co-stored should be linked."""
    dim = 32
    engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

    # Create two nearly orthogonal vectors (low cosine similarity)
    rng = np.random.default_rng(42)
    patterns = make_orthogonal_patterns(4, dim, seed=42)
    a, b, c, d = patterns[0], patterns[1], patterns[2], patterns[3]

    # Store a and b in the same session (co-occurring)
    engine.store("capital of France is Paris", a)
    engine.store("France population is 67 million", b)
    engine.flush_session()

    # Store c and d in a different session
    engine.store("Japan GDP is high", c)
    engine.store("Brazil has rainforests", d)
    engine.flush_session()

    # Query with a: without temporal, b has ~0 cosine to a (orthogonal)
    # With temporal binding, b should be boosted in associative results
    results_assoc = engine.query_associative(a, top_k=4, sparse=True)
    result_indices = [r["index"] for r in results_assoc]

    # b (index 1) should appear in top results due to temporal link
    assert 1 in result_indices[:2], (
        f"Temporal co-occurrence should surface b; got indices {result_indices}"
    )
```

### Contract 7: Cosine query is NOT affected by W_temporal

```python
def test_cosine_query_unaffected():
    """query() uses direct cosine, not spreading activation -- W_temporal irrelevant."""
    dim = 16
    a = random_unit_vector(dim, seed=0)
    b = random_unit_vector(dim, seed=1)

    # Engine without temporal
    engine_base = CoupledEngine(dim=dim)
    engine_base.store("a", a)
    engine_base.store("b", b)

    # Engine with temporal bindings
    engine_temporal = CoupledEngine(dim=dim, hebbian_epsilon=0.1)
    engine_temporal.store("a", a)
    engine_temporal.store("b", b)
    engine_temporal.flush_session()

    query = random_unit_vector(dim, seed=99)
    results_base = engine_base.query(query, top_k=2)
    results_temporal = engine_temporal.query(query, top_k=2)

    # Same scores (query() does NOT use W or W_temporal)
    for rb, rt in zip(results_base, results_temporal):
        np.testing.assert_allclose(rb["score"], rt["score"], atol=1e-12)
```

### Contract 8: Dream cycle does not corrupt W_temporal

```python
def test_dream_preserves_W_temporal():
    dim = 16
    engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

    # Store and flush 3 sessions
    for session in range(3):
        for i in range(4):
            emb = random_unit_vector(dim, seed=session * 100 + i)
            engine.store(f"s{session}_f{i}", emb)
        engine.flush_session()

    W_temporal_before = engine._W_temporal.copy()
    engine.dream()
    W_temporal_after = engine._W_temporal

    # W_temporal is in embedding space (d,d), not index space -- dream doesn't touch it
    np.testing.assert_array_equal(W_temporal_before, W_temporal_after)
```

### Contract 9: Save/load roundtrip preserves W_temporal

```python
def test_save_load_roundtrip(tmp_path):
    dim = 16
    engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

    a = random_unit_vector(dim, seed=0)
    b = random_unit_vector(dim, seed=1)
    engine.store("a", a)
    engine.store("b", b)
    engine.flush_session()

    save_path = tmp_path / "engine_test"
    engine.save(save_path)
    loaded = CoupledEngine.load(save_path)

    np.testing.assert_allclose(loaded._W_temporal, engine._W_temporal, atol=1e-12)
    assert loaded.hebbian_epsilon == engine.hebbian_epsilon
    assert len(loaded._session_buffer) == 0  # buffer is transient, not saved
```

### Contract 10: Edge cases

```python
def test_flush_empty_buffer_is_noop():
    engine = CoupledEngine(dim=8)
    n = engine.flush_session()
    assert n == 0
    assert np.allclose(engine._W_temporal, 0.0)

def test_flush_single_fact_is_noop():
    engine = CoupledEngine(dim=8)
    engine.store("solo", random_unit_vector(8, seed=0))
    n = engine.flush_session()
    assert n == 0
    assert np.allclose(engine._W_temporal, 0.0)
    assert len(engine._session_buffer) == 0

def test_flush_three_facts_creates_three_pairs():
    engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
    a = random_unit_vector(8, seed=0)
    b = random_unit_vector(8, seed=1)
    c = random_unit_vector(8, seed=2)
    engine.store("a", a)
    engine.store("b", b)
    engine.store("c", c)
    n = engine.flush_session()
    assert n == 3  # C(3,2) = 3 pairs: (a,b), (a,c), (b,c)

    expected = 0.1 * (
        np.outer(a, b) + np.outer(b, a) +
        np.outer(a, c) + np.outer(c, a) +
        np.outer(b, c) + np.outer(c, b)
    )
    np.testing.assert_allclose(engine._W_temporal, expected, atol=1e-12)

def test_flush_n_facts_creates_n_choose_2_pairs():
    """Verify combinatorial count for larger sessions."""
    engine = CoupledEngine(dim=8, hebbian_epsilon=0.01)
    N = 10
    for i in range(N):
        engine.store(f"fact_{i}", random_unit_vector(8, seed=i))
    n = engine.flush_session()
    assert n == N * (N - 1) // 2  # 45 pairs
```

### Contract 11: Contradiction replacement uses new embedding in session buffer

```python
def test_contradiction_replacement_buffers_new_embedding():
    """When store() replaces a contradicting entry, the NEW embedding enters the buffer."""
    dim = 16
    engine = CoupledEngine(
        dim=dim, hebbian_epsilon=0.1,
        contradiction_aware=True, contradiction_threshold=0.8,
    )
    # Store initial fact
    a = random_unit_vector(dim, seed=0)
    engine.store("Paris is in France", a)
    engine.flush_session()  # clear session 1

    # Store a very similar embedding (above contradiction threshold)
    # that should trigger replacement
    a_updated = a + 0.05 * random_unit_vector(dim, seed=99)
    a_updated /= np.linalg.norm(a_updated)

    b = random_unit_vector(dim, seed=1)
    engine.store("Paris is the capital of France", a_updated)  # replaces index 0
    engine.store("Berlin is in Germany", b)
    engine.flush_session()

    # Session buffer should have contained a_updated (not a) and b
    # So W_temporal should encode outer(a_updated, b) + outer(b, a_updated)
    expected_temporal = 0.1 * (np.outer(a_updated, b) + np.outer(b, a_updated))
    # Plus the session-1 contribution (which was just a single fact -> no pairs)
    np.testing.assert_allclose(engine._W_temporal, expected_temporal, atol=1e-10)
```

### Contract 12: reset_temporal clears everything

```python
def test_reset_temporal():
    engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
    engine.store("a", random_unit_vector(8, seed=0))
    engine.store("b", random_unit_vector(8, seed=1))
    engine.flush_session()

    assert not np.allclose(engine._W_temporal, 0.0)  # has bindings

    engine.reset_temporal()
    assert np.allclose(engine._W_temporal, 0.0)
    assert len(engine._session_buffer) == 0
```


## What Must NOT Change

| Component | Reason |
|-----------|--------|
| `query()` (cosine retrieval) | Does not use W or spreading activation. Pure cosine + recency. |
| `dream()` / `dream_cycle_xb()` | Operates on patterns X, not W. W_temporal is in (d,d) embedding space and is unrelated to the index-based operations in dream. |
| `W` property | Still returns `X^T X` with zeroed diagonal. This is the auto-derived Hopfield coupling. |
| `_invalidate_cache()` | Still clears only `_embeddings_cache` and `_W_cache`. W_temporal is not a cache -- it is primary state. |
| `hopfield_update()` (existing) | Unchanged. New `hopfield_update_biased()` is added alongside. |
| `spreading_activation()` (existing) | Unchanged. New `spreading_activation_biased()` is added alongside. |
| `_embeddings_matrix()` | Unchanged. Session buffer is separate from the embeddings cache. |


## Invariants

1. **Symmetry:** `W_temporal == W_temporal.T` at all times. Every Hebbian update
   adds `outer(a,b) + outer(b,a)`, which is symmetric.

2. **Monotone accumulation:** `||W_temporal||_F` is monotonically non-decreasing
   across flushes (no subtraction ever occurs, epsilon >= 0).

3. **Dimension consistency:** `W_temporal.shape == (dim, dim)` always.

4. **Buffer transience:** `_session_buffer` is transient state. It is NOT saved
   to disk and is empty after `flush_session()`, `reset_temporal()`, or
   construction.

5. **Independence from W cache:** `_invalidate_cache()` does NOT affect
   `_W_temporal`. Conversely, `flush_session()` does NOT call
   `_invalidate_cache()`.

6. **Backward compatibility:** An engine with `hebbian_epsilon=0` or that never
   calls `flush_session()` behaves identically to the pre-temporal engine.
   Loading old save files that lack `W_temporal` produces a zero matrix.


## Design Decisions and Rationale

### Q: Default epsilon?
**A:** 0.01. This is small enough that temporal bindings provide a gentle bias
without overwhelming the base cosine/Hopfield signal. The CMA-ES optimizer can
tune this parameter alongside existing hyperparameters.

### Q: Constructor parameter or flush_session parameter?
**A:** Both. `hebbian_epsilon` is a constructor parameter (the default), with
`flush_session(epsilon=...)` allowing per-call override. This lets the MABench
adapter use a fixed default while tests can vary epsilon per call.

### Q: Does _invalidate_cache() clear W_temporal?
**A:** No. W_temporal is primary state, not a derived cache. It accumulates
across the entire engine lifetime. `_invalidate_cache()` only clears the
embeddings cache and the auto-derived W cache.

### Q: Should there be a reset for W_temporal?
**A:** Yes, `reset_temporal()` for testing and for the MABench `reset()` call
between contexts. It zeroes both W_temporal and the session buffer.

### Q: How does contradiction replacement interact with session buffer?
**A:** The replaced entry's OLD embedding is already in the memory_store (but
now overwritten). The session buffer receives the NEW embedding that was just
stored. This is correct because the new embedding is the one that should form
temporal associations with other facts in the session.

### Q: Why not modify the existing `spreading_activation` to accept W_temporal?
**A:** Backward compatibility. `spreading_activation` is used by other code
paths and tests that don't have W_temporal. Adding a new function
`spreading_activation_biased` keeps the existing interface stable while
providing the temporal capability to query methods that need it.

### Q: Why normalize the temporal_similarities in hopfield_update_biased?
**A:** The temporal probe `W_temporal @ xi` can have arbitrary magnitude
depending on accumulated epsilon and number of flushes. Normalizing the
projected similarities ensures the temporal signal acts as a directional bias
(which patterns to attend to) rather than a magnitude override that could
dominate the base cosine signal.

### Q: What about large sessions (N=100+ facts)?
**A:** N*(N-1)/2 outer products is O(N^2 * d^2). For d=1024 and N=20 (typical
MABench chunk), this is 190 * 1M = 190M flops, negligible. For N=100, it's
4950 pairs, still under 5B flops (~10ms on CPU). No optimization needed for
expected workloads.


## Implementation Phases

### Phase 1: Foundation (dream_ops.py)
**Files to create/modify:**
- `dream_ops.py` -- add `hopfield_update_biased()` and `spreading_activation_biased()`

**Acceptance:**
- [ ] `hopfield_update_biased` with zero W_temporal equals `hopfield_update`
- [ ] `spreading_activation_biased` with zero W_temporal equals `spreading_activation`
- [ ] Unit tests for both functions pass

**Estimated effort:** Small

### Phase 2: CoupledEngine core (coupled_engine.py)
**Files to modify:**
- `coupled_engine.py` -- add `_session_buffer`, `_W_temporal`, `hebbian_epsilon`,
  `flush_session()`, `reset_temporal()`, `W_full` property, modify `store()`,
  `save()`, `load()`

**Dependencies:** None (Phase 1 is independent)

**Acceptance:**
- [ ] Contracts 1-5, 10-12 pass
- [ ] Contract 9 (save/load roundtrip) passes
- [ ] Existing test suite (`test_coupled_engine.py`) still passes

**Estimated effort:** Medium

### Phase 3: Query integration (coupled_engine.py + dream_ops.py)
**Files to modify:**
- `coupled_engine.py` -- modify `query_associative()`, `query_hybrid()` to use
  biased variants when W_temporal is non-zero

**Dependencies:** Phase 1, Phase 2

**Acceptance:**
- [ ] Contract 6 (temporal link discovery) passes
- [ ] Contract 7 (cosine query unchanged) passes
- [ ] Contract 8 (dream compatibility) passes
- [ ] Existing test suite still passes

**Estimated effort:** Small

### Phase 4: MABench integration (hermes_agent.py)
**Files to modify:**
- `mabench/hermes_agent.py` -- add `flush_session()` call in `_store()`, forward
  `hebbian_epsilon` in `reset()`

**Dependencies:** Phase 2

**Acceptance:**
- [ ] `_store()` calls `flush_session()` after each chunk
- [ ] `reset()` preserves `hebbian_epsilon` setting
- [ ] MABench eval produces results (no regression on SubEM)

**Estimated effort:** Small

### Phase 5: Testing
**Files to create:**
- `tests/test_temporal_hebbian.py` -- all 12 contracts above

**Coverage target:** All contracts pass. No regressions in existing
`test_coupled_engine.py` or `test_dream_ops.py`.

**Estimated effort:** Medium


## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Temporal bias overwhelms cosine signal | High: retrieval quality degrades | Normalization in `hopfield_update_biased`. Default epsilon=0.01 is conservative. CMA-ES can tune. |
| W_temporal grows unboundedly over many sessions | Medium: numerical instability | Monitor `||W_temporal||_F` in `get_metrics()`. Add optional spectral normalization if needed (future work). |
| Backward compatibility breakage in save/load | Medium: old checkpoints fail | Guard with `if "W_temporal" in data` in `load()`. Old files get zero matrix. |
| Performance regression from biased Hopfield | Low: extra matmul per query | Guard with `np.any(W_temporal != 0)` to skip biased path when unnecessary. Cost is one (d,d) @ (d,) matmul = 1M flops for d=1024. |
| Session buffer leaked across unrelated chunks | High: false temporal links | `flush_session()` called exactly once per `_store()` call. Buffer is always cleared. |


## Open Questions

- [ ] Should `get_metrics()` report W_temporal statistics (norm, sparsity, spectral radius)?
- [ ] Should temporal bindings decay over dream cycles (W_temporal *= decay_factor)?
- [ ] Should there be a max-norm constraint on W_temporal to prevent unbounded growth?
- [ ] Should the CMA-ES optimizer include `hebbian_epsilon` as a tunable parameter?
- [ ] Should `query_hybrid` use a different temporal injection strategy than `query_associative` (e.g., additive probe bias instead of biased Hopfield step)?


## File Change Summary

| File | Changes |
|------|---------|
| `coupled_engine.py` | Add `_session_buffer`, `_W_temporal`, `hebbian_epsilon` to `__init__`. Add `flush_session()`, `reset_temporal()`, `W_full` property. Modify `store()`, `query_associative()`, `query_hybrid()`, `save()`, `load()`. |
| `dream_ops.py` | Add `hopfield_update_biased()`, `spreading_activation_biased()`. |
| `mabench/hermes_agent.py` | Add `flush_session()` call in `_store()`. Forward `hebbian_epsilon` in `reset()`. |
| `tests/test_temporal_hebbian.py` | New file: all 12 behavioral contracts. |

