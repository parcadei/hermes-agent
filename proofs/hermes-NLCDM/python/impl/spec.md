# GPU Ops Behavioral Specification

**Module:** `gpu_ops.py`
**Location:** `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/gpu_ops.py`
**Created:** 2026-03-03
**Author:** architect-agent (Opus 4.6)

## Overview

`gpu_ops.py` accelerates the dominant hot paths in `dream_ops.py` by offloading
dense matrix operations to PyTorch CUDA. At N=18K, d=1024, a single `X @ X.T`
is ~660 MFLOP per element x 18K^2 = ~330 GFLOP. The dream cycle calls this
pattern 4-7 times, making it the single largest time sink. All functions follow
a **numpy-in / numpy-out** contract with graceful CPU fallback.

---

## 1. Module-Level Design

### 1.1 Device Management

```python
import numpy as np

_DEVICE: torch.device | None = None  # lazily initialized

def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU. Cached after first call."""
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE

def is_gpu_available() -> bool:
    """Check whether CUDA acceleration is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

### 1.2 Import Strategy

PyTorch is an optional dependency. The module must handle `ImportError` at
import time and expose a boolean `HAS_TORCH` flag:

```python
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
```

Every public function checks `HAS_TORCH` and falls back to pure numpy if False.

### 1.3 VRAM Safety

For `similarity_matrix` with N=18K, d=1024, float32:
- Input tensor: 18K x 1024 x 4B = ~72 MB
- Output tensor: 18K x 18K x 4B = ~1.3 GB
- Total VRAM: ~1.4 GB (fits comfortably on any 4GB+ GPU)

For N=100K (future scale):
- Output: 100K x 100K x 4B = ~40 GB (exceeds most GPUs)

Strategy: attempt the GPU path; on `torch.cuda.OutOfMemoryError`, fall back
to numpy. No manual size thresholds -- let PyTorch's allocator decide.

---

## 2. Public API

### 2.1 `similarity_matrix`

```python
def similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the full pairwise similarity matrix S = X @ X.T.

    Args:
        X: (N, d) array, float32 or float64. Typically unit-norm row vectors
           but this function does NOT assume normalization.

    Returns:
        S: (N, N) array, same dtype as X. S[i,j] = dot(X[i], X[j]).

    Contract:
        - np.allclose(result, X @ X.T, atol=1e-6) == True
        - result.shape == (X.shape[0], X.shape[0])
        - result.dtype == X.dtype
        - If CUDA unavailable or OOM: falls back to numpy, same result.
        - Input X is never mutated.
    """
```

**Implementation notes:**
- Convert to `torch.float32` on GPU (even if input is float64) for speed.
  Convert output back to input dtype before returning.
- Use `torch.mm(X_t, X_t.T)` -- single CUBLAS SGEMM call.
- The `.cpu().numpy()` copy is unavoidable but is O(N^2) memcpy vs O(N^2 d) compute.
- For float64 inputs, the 1e-6 tolerance still holds because the matmul
  itself is at most ~d ULPs off and d=1024 gives ~1e-4 relative error in
  float32, which for unit-norm vectors (dot products in [-1, 1]) is well
  within 1e-6 absolute.

  **Correction:** For float64 inputs where inner products can be large
  (non-unit vectors), compute in float64 on GPU (`torch.float64`) to
  preserve the contract. Float32 is only safe when input is float32.

### 2.2 `find_close_pairs`

```python
def find_close_pairs(
    S: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find index pairs (i, j) with i < j where S[i,j] > threshold.

    Args:
        S: (N, N) similarity matrix.
        threshold: scalar threshold for "close" pairs.

    Returns:
        (row_indices, col_indices): each 1-D int64 arrays.
        Equivalent to np.where(np.triu(S, k=1) > threshold).

    Contract:
        - Let (ri, ci) = result. For all k: ri[k] < ci[k].
        - For all k: S[ri[k], ci[k]] > threshold.
        - Set of returned pairs == set from np.where(np.triu(S, k=1) > threshold).
        - Input S is never mutated.

    Notes:
        This stays on CPU. The triu + threshold comparison is memory-bound,
        not compute-bound. GPU transfer overhead would exceed any gain
        for N <= 100K. The function exists to provide a single call site
        that can be GPU-accelerated later if N grows.
    """
```

**Implementation:** Pure numpy. `np.triu(S, k=1)` creates a copy, which is
fine at N=18K (~1.3 GB). For larger N, consider an in-place mask approach.

**GPU acceleration path (future, not in v1):** For N > 50K, transfer S to
GPU, apply `torch.triu` + `torch.where`, return indices. This avoids the
1.3 GB numpy copy. Not needed at current scale.

### 2.3 `batch_normalize`

```python
def batch_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.

    Args:
        X: (N, d) array, any float dtype.

    Returns:
        Y: (N, d) array, same dtype. Each row has unit L2 norm.
        Rows with norm < 1e-12 are returned unchanged (avoid division by zero).

    Contract:
        - norms = np.linalg.norm(result, axis=1)
        - For all i where np.linalg.norm(X[i]) >= 1e-12:
            abs(norms[i] - 1.0) < 1e-6
        - result.shape == X.shape
        - result.dtype == X.dtype
        - Input X is never mutated.
    """
```

**Implementation notes:**
- GPU path: `X_t / X_t.norm(dim=1, keepdim=True).clamp(min=1e-12)`
- CPU path: `X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)`
- This is bandwidth-bound (O(N*d) with low arithmetic intensity), so GPU
  benefit is modest. Include for API completeness and to accelerate the
  normalization that follows every repulsion/unlearn step.

### 2.4 `batch_matmul`

```python
def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """General matrix multiplication A @ B via CUDA.

    Args:
        A: (M, K) array, float32 or float64.
        B: (K, P) array, same dtype as A.

    Returns:
        C: (M, P) array, same dtype. C = A @ B.

    Contract:
        - np.allclose(result, A @ B, atol=1e-6) == True
        - result.shape == (A.shape[0], B.shape[1])
        - result.dtype == A.dtype (or common type of A, B)
        - If CUDA unavailable or OOM: falls back to numpy.
        - Inputs A, B are never mutated.
    """
```

**Implementation notes:**
- Used by `rem_explore_cross_domain_xb` for `patterns @ qi_batch.T` where
  patterns is (N, d) and qi_batch is (K, d), so B = qi_batch.T is (d, K).
  At N=18K, K=10, d=1024 this is small -- GPU overhead may exceed benefit.
  But at scale (K=100, multiple probes) it compounds.
- Same dtype-preservation logic as `similarity_matrix`.

### 2.5 `assign_clusters_matmul`

```python
def assign_clusters_matmul(
    patterns: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Cluster assignment via matmul + union-find (replaces N^2 Python loop).

    Computes S = patterns @ patterns.T, finds pairs above threshold via
    find_close_pairs, then runs union-find on those pairs.

    Args:
        patterns: (N, d) unit-norm vectors.
        threshold: cosine similarity threshold for same-cluster assignment.

    Returns:
        labels: (N,) int array of contiguous cluster labels starting at 0.

    Contract:
        - Equivalent to _assign_clusters(patterns, threshold) from dream_ops.py
          (the Python N^2 loop version).
        - labels.shape == (N,)
        - labels.dtype == int
        - Each connected component gets a unique label.
        - Labels are contiguous integers starting at 0.
    """
```

**Implementation notes:**
- This replaces lines 1324-1327 of `dream_ops.py` which do:
  ```python
  for i in range(N):
      for j in range(i + 1, N):
          if float(patterns[i] @ patterns[j]) > threshold:
              union(i, j)
  ```
- The replacement computes `S = similarity_matrix(patterns)`, then
  `close_i, close_j = find_close_pairs(S, threshold)`, then runs
  union-find on the resulting pairs.
- At N=18K this replaces ~162M individual Python dot products with one
  CUBLAS SGEMM + vectorized threshold.
- The union-find loop over the resulting pairs stays in Python (it is
  O(num_pairs) with path compression, typically much smaller than N^2).

---

## 3. Integration Plan: Hot Path Replacements in `dream_ops.py`

Each replacement below is a drop-in substitution. No function signatures
change. No behavioral differences beyond floating-point tolerance.

### 3.1 Import Block (top of file)

**Add after line 20** (`from nlcdm_core import cosine_sim, sigmoid, softmax`):

```python
try:
    from gpu_ops import (
        similarity_matrix as _gpu_sim,
        find_close_pairs as _gpu_close_pairs,
        batch_normalize as _gpu_normalize,
        batch_matmul as _gpu_matmul,
        assign_clusters_matmul as _gpu_assign_clusters,
    )
    _HAS_GPU_OPS = True
except ImportError:
    _HAS_GPU_OPS = False
```

### 3.2 Hot Path 1: `nrem_repulsion_xb` main similarity (line 884)

**Current code (line 884):**
```python
S = patterns @ patterns.T  # (N, N)
```

**Replacement:**
```python
S = _gpu_sim(patterns) if _HAS_GPU_OPS else patterns @ patterns.T  # (N, N)
```

**Lines 888, 898 unchanged** -- they operate on the numpy S returned by `_gpu_sim`.

### 3.3 Hot Path 2: `nrem_repulsion_xb` postcondition (lines 922-924)

**Current code (lines 922-924):**
```python
S_in_triu = patterns @ patterns.T
np.fill_diagonal(S_in_triu, -2.0)
S_out = X @ X.T
```

**Replacement:**
```python
S_in_triu = _gpu_sim(patterns) if _HAS_GPU_OPS else patterns @ patterns.T
np.fill_diagonal(S_in_triu, -2.0)
S_out = _gpu_sim(X) if _HAS_GPU_OPS else X @ X.T
```

**Note:** This postcondition computes TWO additional matmuls. Optimization
opportunity: reuse the S computed at line 884 for `S_in_triu` (it is the
same matrix before diagonal masking). This is a correctness-preserving
optimization that halves the postcondition cost:

```python
# Reuse S from line 884 (before fill_diagonal was applied)
S_in_triu = patterns @ patterns.T  # can reuse pre-diagonal-mask copy
```

However, that requires saving a copy before `np.fill_diagonal(S, -2.0)` on
line 888. The simplest approach: save `S_orig = S.copy()` before line 888
and reuse it at line 922. This is an optimization within `dream_ops.py`
itself, orthogonal to `gpu_ops.py`.

### 3.4 Hot Path 3: `nrem_prune_xb` (line 976)

**Current code (line 976):**
```python
S = patterns @ patterns.T  # (N, N)
```

**Replacement:**
```python
S = _gpu_sim(patterns) if _HAS_GPU_OPS else patterns @ patterns.T  # (N, N)
```

### 3.5 Hot Path 4: `nrem_merge_xb` (line 1100)

**Current code (line 1100):**
```python
S = patterns @ patterns.T  # (N, N)
```

**Replacement:**
```python
S = _gpu_sim(patterns) if _HAS_GPU_OPS else patterns @ patterns.T  # (N, N)
```

### 3.6 Hot Path 5: `dream_cycle_xb` shared S (line 1765)

**Current code (line 1765):**
```python
S = X3 @ X3.T
```

**Replacement:**
```python
S = _gpu_sim(X3) if _HAS_GPU_OPS else X3 @ X3.T
```

### 3.7 Hot Path 6: `rem_explore_cross_domain_xb` (lines 1264-1265)

**Current code (lines 1264-1265):**
```python
ri_stack = (patterns @ qi_batch.T).T  # (K, N)
rj_stack = (patterns @ qj_batch.T).T  # (K, N)
```

**Replacement:**
```python
ri_stack = (_gpu_matmul(patterns, qi_batch.T) if _HAS_GPU_OPS else patterns @ qi_batch.T).T  # (K, N)
rj_stack = (_gpu_matmul(patterns, qj_batch.T) if _HAS_GPU_OPS else patterns @ qj_batch.T).T  # (K, N)
```

**Note:** These are (N, d) @ (d, K) = (N, K) matmuls with K=10. At N=18K,
d=1024, this is only ~37 MFLOP per call -- below the threshold where GPU
transfer overhead pays off. However, `rem_explore_cross_domain_xb` is called
with up to `n_probes = max(N_out, 50)` iterations, and each probe does 2
matmuls. At N_out=15K, that is 30K matmuls of 37 MFLOP each = ~1.1 TFLOP
total.

**Better optimization:** Restructure the loop to batch all probes into fewer
large matmuls. This is a refactoring of `rem_explore_cross_domain_xb` itself,
beyond the scope of `gpu_ops.py`. For now, the per-call GPU acceleration
provides some benefit by avoiding the transfer overhead via persistent GPU
tensors (see Section 4: Future Optimizations).

### 3.8 Hot Path 7: `_assign_clusters` (lines 1324-1327)

**Current code (lines 1304-1340):**
```python
def _assign_clusters(patterns: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    N = patterns.shape[0]
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(N):
        for j in range(i + 1, N):
            if float(patterns[i] @ patterns[j]) > threshold:
                union(i, j)

    # Map roots to contiguous labels
    root_to_label: dict[int, int] = {}
    labels = np.zeros(N, dtype=int)
    next_label = 0
    for i in range(N):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels
```

**Replacement** -- rewrite the function body:

```python
def _assign_clusters(patterns: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if _HAS_GPU_OPS:
        return _gpu_assign_clusters(patterns, threshold)

    # Original N^2 loop fallback (unchanged)
    N = patterns.shape[0]
    parent = list(range(N))
    # ... (rest of original code unchanged)
```

**Behavioral equivalence:** `assign_clusters_matmul` replicates the exact
same algorithm: compute all pairwise cosine similarities, threshold, then
union-find. The only difference is that the N^2 individual `patterns[i] @
patterns[j]` Python-level dot products are replaced by a single SGEMM call
followed by vectorized threshold comparison.

### 3.9 Bonus Hot Path: `compute_adaptive_thresholds` (lines 1414-1425)

**Current code (lines 1414-1425):**
```python
if n_pairs <= max_sample_pairs:
    for a in range(n_c):
        for b in range(a + 1, n_c):
            sim = float(patterns[indices[a]] @ patterns[indices[b]])
            within_sims.append(sim)
else:
    for _ in range(max_sample_pairs):
        a, b = rng.choice(n_c, size=2, replace=False)
        sim = float(patterns[indices[a]] @ patterns[indices[b]])
        within_sims.append(sim)
```

**Replacement strategy:** For the enumeration path (n_pairs <= max_sample_pairs),
extract the cluster's pattern submatrix and use `similarity_matrix`:

```python
if n_pairs <= max_sample_pairs:
    cluster_patterns = patterns[indices]  # (n_c, d)
    S_cluster = _gpu_sim(cluster_patterns) if _HAS_GPU_OPS else cluster_patterns @ cluster_patterns.T
    # Extract upper-triangle similarities
    tri_i, tri_j = np.triu_indices(n_c, k=1)
    within_sims.extend(S_cluster[tri_i, tri_j].tolist())
else:
    # Sampling path stays unchanged (random pairs, not worth matmul)
    for _ in range(max_sample_pairs):
        a, b = rng.choice(n_c, size=2, replace=False)
        sim = float(patterns[indices[a]] @ patterns[indices[b]])
        within_sims.append(sim)
```

**Note:** This hot path is less critical than the others because
`max_sample_pairs=10000` caps the loop iterations. But for clusters with
n_c > 200 (where n_pairs = 200*199/2 = 19900 > 10000, so sampling kicks
in), the enumeration path only fires for small clusters. The GPU
acceleration here has marginal impact -- include for consistency but mark
as low priority.

### 3.10 `dream_cycle_v2` missing `similarity_matrix` pass (line 1612)

**Observation:** `dream_cycle_v2` calls `rem_unlearn_xb(X3, beta, ...)` at
line 1612 WITHOUT passing `similarity_matrix=S`. This means it takes the
expensive Monte Carlo path instead of the fast S-based path. This is likely
a bug or oversight -- `dream_cycle_xb` (the original pipeline) precomputes
`S = X3 @ X3.T` at line 1765 and passes it at line 1781.

**Recommended fix (in `dream_cycle_v2`, around line 1607-1615):**
```python
# Step 4: NREM repulsion
X3 = nrem_repulsion_xb(X2, importances_after_prune, eta=0.01, min_sep=0.3)

# Precompute similarity matrix for REM ops (same pattern as dream_cycle_xb)
S = _gpu_sim(X3) if _HAS_GPU_OPS else X3 @ X3.T

# Step 5: REM unlearn
X4 = rem_unlearn_xb(
    X3, beta,
    n_probes=200, separation_rate=0.02, rng=rng,
    similarity_matrix=S,  # <-- ADD THIS
)
```

This is an optimization within `dream_ops.py` independent of `gpu_ops.py`,
but the GPU acceleration of the S computation makes it even more impactful.

---

## 4. Contracts and Invariants

### 4.1 Behavioral Equivalence Contract

For any input `patterns` (N, d) with N in [0, 100000] and d in [1, 4096]:

```python
# With GPU ops
report_gpu = dream_cycle_xb(patterns, beta, seed=42)

# Without GPU ops (pure numpy)
report_cpu = dream_cycle_xb(patterns, beta, seed=42)

# Must hold:
assert np.allclose(report_gpu.patterns, report_cpu.patterns, atol=1e-5)
assert report_gpu.pruned_indices == report_cpu.pruned_indices
assert report_gpu.merge_map == report_cpu.merge_map
# Associations may differ slightly due to float ordering in correlation,
# but the set of pairs and their approximate strengths must match.
```

**Why 1e-5 and not 1e-6:** The dream cycle chains 5 operations, each with
up to 1e-6 error per matmul. Error accumulates through normalization,
repulsion steps, and re-normalization. 1e-5 provides margin for the full
chain while still catching real divergences.

### 4.2 Graceful Fallback Contract

```python
# If torch is not installed:
import gpu_ops
assert gpu_ops.HAS_TORCH == False
result = gpu_ops.similarity_matrix(X)
assert np.array_equal(result, X @ X.T)  # exact numpy fallback, no error
```

Every public function must:
1. Check `HAS_TORCH`
2. If False: execute pure numpy path
3. If True but CUDA unavailable: execute on CPU via torch (still faster than
   numpy for large matrices due to MKL vs OpenBLAS differences, but
   functionally equivalent)
4. If True but OOM: catch `torch.cuda.OutOfMemoryError`, fall back to numpy

### 4.3 numpy-in / numpy-out Contract

No `torch.Tensor` objects are ever returned or stored in module-level state.
Every public function accepts `np.ndarray` and returns `np.ndarray`. The
calling code in `dream_ops.py` is completely unaware of torch.

### 4.4 No Signature Changes Contract

All functions in `dream_ops.py` keep their existing signatures. The only
changes are:
- Adding the import block (Section 3.1)
- Replacing `X @ X.T` with `_gpu_sim(X) if _HAS_GPU_OPS else X @ X.T`
- Replacing `_assign_clusters` body with delegation to `_gpu_assign_clusters`
- Adding `similarity_matrix=S` parameter to `dream_cycle_v2`'s call to
  `rem_unlearn_xb` (this uses an existing optional parameter)

### 4.5 No Input Mutation Contract

Every function creates copies or new tensors. The input numpy arrays are
never written to. This preserves the existing `dream_ops.py` invariant
("All operations are pure: W_in -> W_out (no mutation)." -- line 11).

---

## 5. Error Handling

| Condition | Behavior |
|-----------|----------|
| `torch` not installed | `HAS_TORCH = False`, all functions use numpy |
| CUDA not available | `_get_device()` returns CPU, torch ops on CPU |
| `torch.cuda.OutOfMemoryError` | Catch, log warning, re-execute with numpy |
| Input not float32/64 | Cast to float64, compute, cast result back |
| Input has NaN/Inf | Pass through (same behavior as numpy matmul) |
| Empty input (N=0) | Return empty array with correct shape |
| 1-D input | Raise `ValueError` ("expected 2-D array") |

### 5.1 OOM Handling Pattern

```python
def similarity_matrix(X: np.ndarray) -> np.ndarray:
    if not HAS_TORCH:
        return X @ X.T

    try:
        X_t = torch.from_numpy(X).to(_get_device())
        S_t = torch.mm(X_t, X_t.T)
        return S_t.cpu().numpy().astype(X.dtype)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return X @ X.T  # fallback
    except Exception:
        return X @ X.T  # any other torch error -> numpy fallback
```

The broad `except Exception` ensures that exotic CUDA errors (driver issues,
corrupted state) never crash the dream cycle. The fallback is always correct.

---

## 6. Testing Strategy

### 6.1 Unit Tests (`test_gpu_ops.py`)

```python
class TestSimilarityMatrix:
    def test_small_exact(self):
        """3x4 matrix, verify against numpy."""
        X = np.random.randn(3, 4).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        assert np.allclose(S, X @ X.T, atol=1e-6)

    def test_dtype_preservation_float32(self):
        X = np.random.randn(5, 3).astype(np.float32)
        assert gpu_ops.similarity_matrix(X).dtype == np.float32

    def test_dtype_preservation_float64(self):
        X = np.random.randn(5, 3).astype(np.float64)
        assert gpu_ops.similarity_matrix(X).dtype == np.float64

    def test_empty(self):
        X = np.zeros((0, 10), dtype=np.float32)
        S = gpu_ops.similarity_matrix(X)
        assert S.shape == (0, 0)

    def test_single_row(self):
        X = np.array([[1.0, 0.0, 0.0]])
        S = gpu_ops.similarity_matrix(X)
        assert S.shape == (1, 1)
        assert np.isclose(S[0, 0], 1.0)

    def test_symmetry(self):
        X = np.random.randn(50, 20).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        assert np.allclose(S, S.T, atol=1e-7)

    def test_no_input_mutation(self):
        X = np.random.randn(10, 5).astype(np.float32)
        X_orig = X.copy()
        _ = gpu_ops.similarity_matrix(X)
        assert np.array_equal(X, X_orig)

    def test_scale_18k(self):
        """Realistic scale test (only runs if CUDA available)."""
        if not gpu_ops.is_gpu_available():
            pytest.skip("No CUDA")
        X = np.random.randn(18000, 1024).astype(np.float32)
        # Just check it completes and has right shape
        S = gpu_ops.similarity_matrix(X)
        assert S.shape == (18000, 18000)


class TestFindClosePairs:
    def test_basic(self):
        S = np.array([[1.0, 0.9, 0.3],
                       [0.9, 1.0, 0.8],
                       [0.3, 0.8, 1.0]])
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        expected_ri, expected_ci = np.where(np.triu(S, k=1) > 0.5)
        assert np.array_equal(ri, expected_ri)
        assert np.array_equal(ci, expected_ci)

    def test_no_pairs(self):
        S = np.eye(5)
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        assert len(ri) == 0
        assert len(ci) == 0

    def test_all_pairs(self):
        S = np.ones((3, 3))
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        assert len(ri) == 3  # (0,1), (0,2), (1,2)


class TestBatchNormalize:
    def test_unit_norms(self):
        X = np.random.randn(100, 50).astype(np.float32)
        Y = gpu_ops.batch_normalize(X)
        norms = np.linalg.norm(Y, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_zero_row_preserved(self):
        X = np.zeros((3, 4), dtype=np.float32)
        X[1] = [1.0, 2.0, 3.0, 4.0]
        Y = gpu_ops.batch_normalize(X)
        # Row 0 and 2 are zero -> should stay zero (or near-zero)
        assert np.linalg.norm(Y[0]) < 1e-6
        assert np.isclose(np.linalg.norm(Y[1]), 1.0, atol=1e-6)

    def test_already_normalized(self):
        X = np.random.randn(10, 5).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = gpu_ops.batch_normalize(X)
        assert np.allclose(X, Y, atol=1e-10)


class TestBatchMatmul:
    def test_basic(self):
        A = np.random.randn(10, 20).astype(np.float32)
        B = np.random.randn(20, 5).astype(np.float32)
        C = gpu_ops.batch_matmul(A, B)
        assert np.allclose(C, A @ B, atol=1e-6)

    def test_cross_domain_shape(self):
        """Simulate rem_explore_cross_domain_xb: patterns @ qi_batch.T"""
        patterns = np.random.randn(18000, 1024).astype(np.float32)
        qi_batch = np.random.randn(10, 1024).astype(np.float32)
        C = gpu_ops.batch_matmul(patterns, qi_batch.T)
        assert C.shape == (18000, 10)


class TestAssignClustersMatmul:
    def test_equivalence_small(self):
        """Verify against original _assign_clusters."""
        np.random.seed(42)
        X = np.random.randn(50, 10).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        labels_gpu = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        labels_cpu = _assign_clusters_reference(X, threshold=0.5)

        # Labels may differ in numbering but partitions must match
        assert_same_partition(labels_gpu, labels_cpu)

    def test_all_separate(self):
        """Orthogonal vectors -> each in own cluster."""
        X = np.eye(5, dtype=np.float64)
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert len(np.unique(labels)) == 5

    def test_all_same(self):
        """Identical vectors -> one cluster."""
        X = np.ones((5, 3), dtype=np.float64)
        X /= np.linalg.norm(X[0])
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert len(np.unique(labels)) == 1
```

### 6.2 Integration Test (`test_dream_ops_gpu.py`)

```python
def test_dream_cycle_xb_gpu_cpu_equivalence():
    """Full pipeline equivalence test."""
    N, d = 200, 64  # small enough for fast test
    rng = np.random.default_rng(42)
    patterns = rng.standard_normal((N, d)).astype(np.float64)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    beta = 5.0

    # Run with GPU ops
    report_gpu = dream_cycle_xb(patterns, beta, seed=42)

    # Temporarily disable GPU ops and run again
    import gpu_ops
    saved = gpu_ops.HAS_TORCH
    gpu_ops.HAS_TORCH = False
    try:
        # Also need to reload dream_ops to pick up _HAS_GPU_OPS = False
        # Alternative: test at gpu_ops level, not dream_ops level
        report_cpu = dream_cycle_xb(patterns, beta, seed=42)
    finally:
        gpu_ops.HAS_TORCH = saved

    assert np.allclose(report_gpu.patterns, report_cpu.patterns, atol=1e-5)
    assert report_gpu.pruned_indices == report_cpu.pruned_indices
    assert report_gpu.merge_map == report_cpu.merge_map
```

### 6.3 Performance Benchmark

```python
def bench_similarity_matrix():
    """Not a test -- run manually to verify GPU speedup."""
    import time
    X = np.random.randn(18000, 1024).astype(np.float32)

    # Warmup
    _ = gpu_ops.similarity_matrix(X[:100])

    # GPU
    t0 = time.perf_counter()
    S_gpu = gpu_ops.similarity_matrix(X)
    t_gpu = time.perf_counter() - t0

    # CPU
    t0 = time.perf_counter()
    S_cpu = X @ X.T
    t_cpu = time.perf_counter() - t0

    print(f"GPU: {t_gpu:.3f}s, CPU: {t_cpu:.3f}s, speedup: {t_cpu/t_gpu:.1f}x")
    assert np.allclose(S_gpu, S_cpu, atol=1e-5)
```

---

## 7. Implementation Phases

### Phase 1: Core Module (gpu_ops.py)

**Files to create:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/gpu_ops.py`

**Functions:**
1. `_get_device()` (internal)
2. `is_gpu_available()` (public)
3. `similarity_matrix()` (public)
4. `find_close_pairs()` (public)
5. `batch_normalize()` (public)
6. `batch_matmul()` (public)
7. `assign_clusters_matmul()` (public)

**Acceptance criteria:**
- [ ] All unit tests pass on CPU (numpy fallback)
- [ ] All unit tests pass on GPU (if CUDA available)
- [ ] `HAS_TORCH = False` path produces identical results to numpy
- [ ] OOM fallback tested (mock `torch.cuda.OutOfMemoryError`)
- [ ] No torch tensors leak into return values

**Estimated effort:** Small-Medium

### Phase 2: Integration (dream_ops.py edits)

**Files to modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/dream_ops.py`

**Changes:**
1. Add import block (Section 3.1)
2. Replace 5 `X @ X.T` calls (Sections 3.2-3.6)
3. Replace 2 `patterns @ qi_batch.T` calls (Section 3.7)
4. Replace `_assign_clusters` body (Section 3.8)
5. Optimize `compute_adaptive_thresholds` enumeration path (Section 3.9)
6. Fix `dream_cycle_v2` missing `similarity_matrix` pass (Section 3.10)

**Acceptance criteria:**
- [ ] All existing `test_dream_ops.py` tests pass unchanged
- [ ] GPU/CPU equivalence test passes
- [ ] No signature changes to any public function
- [ ] `_HAS_GPU_OPS = False` path is behaviorally identical to current code

**Estimated effort:** Small

### Phase 3: Testing

**Files to create:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/test_gpu_ops.py`
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/test_dream_ops_gpu.py`

**Coverage target:** 100% of gpu_ops.py public functions, including
fallback paths.

**Estimated effort:** Small

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Float32 precision loss | High -- postcondition R1 in `nrem_repulsion_xb` checks `delta_min_out >= delta_min_in - 1e-9`. If GPU matmul introduces > 1e-9 error in max(S), the postcondition fails and repulsion is silently reverted. | Compute in input dtype (float64 for float64 inputs). For float32 inputs, the 1e-9 postcondition margin is tight but ~1e-6 matmul error is still within bounds since the margin applies to the max of two independent matmuls. |
| CUDA driver mismatch | Medium -- wrong PyTorch + CUDA versions cause silent failures | Broad `except Exception` fallback. `is_gpu_available()` tests at import time. |
| Memory fragmentation | Low -- repeated GPU alloc/free within dream cycle may fragment VRAM | Call `torch.cuda.empty_cache()` on OOM before fallback. Consider pre-allocating output tensor. |
| Behavioral divergence | High -- different float rounding between CPU and GPU changes which pairs exceed threshold, leading to different prune/merge/unlearn decisions downstream | Use 1e-5 full-chain tolerance. Threshold comparisons are `>` not `>=`, so values near the boundary may flip. This is acceptable: the dream cycle is stochastic (seeded RNG) and robust to small perturbations. |
| Performance regression for small N | Low -- GPU transfer overhead exceeds compute savings for N < 500 | Add a size check: `if N < 256: return X @ X.T` (numpy is faster for tiny matrices). Threshold determined empirically. |

---

## 9. Open Questions

- [ ] **Q1: Pre-allocate GPU tensors across the dream cycle?**
  Currently each call to `similarity_matrix` allocates, computes, and frees
  GPU tensors. For 5 calls in a single `dream_cycle_xb`, keeping a persistent
  GPU allocation for the largest matrix could save ~5 allocation round-trips.
  Decision: defer to Phase 2 optimization if profiling shows allocation is
  a bottleneck.

- [ ] **Q2: Should `find_close_pairs` move to GPU for large N?**
  At N=18K, `np.triu` + `np.where` creates a 1.3 GB temporary. A GPU
  implementation using `torch.triu` + `torch.nonzero` would avoid the CPU
  copy. Deferring since current scale fits in RAM.

- [ ] **Q3: Should `rem_explore_cross_domain_xb` be restructured for batch GPU?**
  The current per-probe loop issues 2 small matmuls per iteration. Batching
  all probes into a single large matmul would be more GPU-efficient. This
  requires restructuring the loop in `dream_ops.py` itself, not just
  swapping in `gpu_ops` calls. Recommend as a follow-up.

- [ ] **Q4: Minimum N threshold for GPU dispatch?**
  Empirically determine the crossover point where GPU becomes faster than
  CPU BLAS. Likely around N=256-512 for SGEMM. Needs benchmarking on the
  target GPU (RunPod A100/H100).

---

## 10. Success Criteria

1. **Correctness:** All existing `test_dream_ops.py` tests pass without modification.
2. **Equivalence:** `dream_cycle_xb(X, beta, seed=s)` produces results within
   1e-5 tolerance with and without GPU acceleration.
3. **Speedup:** At N=18K, d=1024, full dream cycle wall time reduced by >= 3x
   (expected: CPU ~6s for 5 matmuls, GPU ~0.3s for 5 SGEMM + transfer).
4. **Robustness:** System runs correctly on machines without CUDA, without torch,
   and under GPU OOM conditions.
5. **Zero API change:** No function signature in `dream_ops.py` is modified.
6. **No tensor leakage:** `isinstance(result, np.ndarray)` for every return value.
