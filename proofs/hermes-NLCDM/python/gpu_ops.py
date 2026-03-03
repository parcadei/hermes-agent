"""GPU-accelerated matrix operations with numpy fallback.

Accelerates dominant hot paths in dream_ops.py by offloading dense matrix
operations to PyTorch CUDA. All functions follow a numpy-in / numpy-out
contract with graceful CPU fallback when torch or CUDA is unavailable.

Contracts:
    - numpy-in, numpy-out (no torch tensors leak)
    - No input mutation (never modify input arrays)
    - dtype preservation (float32 in -> float32 out, float64 in -> float64 out)
    - Behavioral equivalence: same results as direct numpy within 1e-6
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level device management
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

_DEVICE: object = None  # torch.device | None, lazily initialized


def _get_device() -> object:
    """Return CUDA device if available, else CPU. Cached after first call."""
    global _DEVICE
    if _DEVICE is None:
        if HAS_TORCH and torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif HAS_TORCH:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def is_gpu_available() -> bool:
    """Check whether CUDA acceleration is available."""
    try:
        import torch as _torch

        return _torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Dtype mapping helpers
# ---------------------------------------------------------------------------

_NUMPY_TO_TORCH = {
    np.float32: "torch.float32",
    np.float64: "torch.float64",
}


def _numpy_dtype_to_torch(dtype: np.dtype) -> object:
    """Map a numpy dtype to the corresponding torch dtype."""
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float64:
        return torch.float64
    # Default: float64 for safety
    return torch.float64


# ---------------------------------------------------------------------------
# 1. similarity_matrix
# ---------------------------------------------------------------------------


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
    if not HAS_TORCH:
        return np.array(X @ X.T, dtype=X.dtype)

    # Small matrices: numpy is faster (avoid GPU transfer overhead)
    if X.shape[0] < 256:
        return np.array(X @ X.T, dtype=X.dtype)

    try:
        device = _get_device()
        X_contig = np.ascontiguousarray(X)
        torch_dtype = _numpy_dtype_to_torch(X.dtype)
        X_t = torch.from_numpy(X_contig).to(device=device, dtype=torch_dtype)
        S_t = torch.mm(X_t, X_t.T)
        result = S_t.cpu().numpy()
        return result.astype(X.dtype, copy=False)
    except Exception:
        # Any torch error (OOM, driver, etc.) -> numpy fallback
        if HAS_TORCH:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return np.array(X @ X.T, dtype=X.dtype)


# ---------------------------------------------------------------------------
# 2. find_close_pairs
# ---------------------------------------------------------------------------


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
        for N <= 100K.
    """
    upper = np.triu(S, k=1)
    return np.where(upper > threshold)


# ---------------------------------------------------------------------------
# 3. batch_normalize
# ---------------------------------------------------------------------------


def batch_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.

    Args:
        X: (N, d) array, any float dtype.

    Returns:
        Y: (N, d) array, same dtype. Each row has unit L2 norm.
        Rows with norm < 1e-12 are returned as zero vectors.

    Contract:
        - norms = np.linalg.norm(result, axis=1)
        - For all i where np.linalg.norm(X[i]) >= 1e-12:
            abs(norms[i] - 1.0) < 1e-6
        - result.shape == X.shape
        - result.dtype == X.dtype
        - Input X is never mutated.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    return (X / safe_norms).astype(X.dtype, copy=False)


# ---------------------------------------------------------------------------
# 4. batch_matmul
# ---------------------------------------------------------------------------


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
    out_dtype = np.result_type(A.dtype, B.dtype)

    if not HAS_TORCH:
        return np.array(A @ B, dtype=out_dtype)

    # Small matrices: numpy is faster
    if A.shape[0] < 256 and B.shape[1] < 256:
        return np.array(A @ B, dtype=out_dtype)

    try:
        device = _get_device()
        torch_dtype = _numpy_dtype_to_torch(np.dtype(out_dtype))
        A_contig = np.ascontiguousarray(A)
        B_contig = np.ascontiguousarray(B)
        A_t = torch.from_numpy(A_contig).to(device=device, dtype=torch_dtype)
        B_t = torch.from_numpy(B_contig).to(device=device, dtype=torch_dtype)
        C_t = torch.mm(A_t, B_t)
        result = C_t.cpu().numpy()
        return result.astype(out_dtype, copy=False)
    except Exception:
        if HAS_TORCH:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return np.array(A @ B, dtype=out_dtype)


# ---------------------------------------------------------------------------
# 5. assign_clusters_matmul
# ---------------------------------------------------------------------------


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
        - labels.dtype is integer
        - Each connected component gets a unique label.
        - Labels are contiguous integers starting at 0.
    """
    n = patterns.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    # Step 1: Compute full similarity matrix
    S = similarity_matrix(patterns)

    # Step 2: Find all pairs above threshold (strict >)
    close_i, close_j = find_close_pairs(S, threshold)

    # Step 3: Union-find on the pairs
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in zip(close_i, close_j):
        union(int(i), int(j))

    # Step 4: Map roots to contiguous labels starting at 0
    root_to_label: dict[int, int] = {}
    labels = np.zeros(n, dtype=int)
    next_label = 0
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels
