"""Tests for gpu_ops.py -- GPU-accelerated matrix operations for dream_ops.

Written BEFORE implementation (TDD). All tests define expected behavior
of the numpy fallback path, which is the reference contract. GPU acceleration
must produce identical results within tolerance.

torch is NOT installed in the test environment, so all tests exercise
the numpy fallback codepath.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np


# ---------------------------------------------------------------------------
# Helper: import gpu_ops lazily (module does not exist yet)
# ---------------------------------------------------------------------------


def _import_gpu_ops():
    """Import gpu_ops, raising ImportError if it does not exist yet."""
    return importlib.import_module("gpu_ops")


def _assert_same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> None:
    """Assert two label arrays encode the same partition.

    Labels may differ in numbering (e.g. [0,0,1] vs [2,2,5]) but
    the groupings must be identical: for all i,j
        labels_a[i] == labels_a[j]  iff  labels_b[i] == labels_b[j]
    """
    assert labels_a.shape == labels_b.shape, (
        f"Shape mismatch: {labels_a.shape} vs {labels_b.shape}"
    )
    n = len(labels_a)
    for i in range(n):
        for j in range(i + 1, n):
            same_a = labels_a[i] == labels_a[j]
            same_b = labels_b[i] == labels_b[j]
            assert same_a == same_b, (
                f"Partition mismatch at ({i}, {j}): "
                f"labels_a[{i}]={labels_a[i]}, labels_a[{j}]={labels_a[j]}, "
                f"labels_b[{i}]={labels_b[i]}, labels_b[{j}]={labels_b[j]}"
            )


# Reference implementation of _assign_clusters (copied from dream_ops.py spec)
# for comparison in tests.
def _assign_clusters_reference(
    patterns: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Reference union-find clustering (pure Python N^2 loop)."""
    n = patterns.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if float(patterns[i] @ patterns[j]) > threshold:
                union(i, j)

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


# ===========================================================================
# 1. similarity_matrix(X) tests
# ===========================================================================


class TestSimilarityMatrix:
    """Tests for gpu_ops.similarity_matrix(X) -> X @ X.T."""

    def test_small_matrix_correctness(self):
        """5x3 matrix: result must equal X @ X.T within 1e-6."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 3)).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        expected = X @ X.T
        assert S.shape == (5, 5)
        np.testing.assert_allclose(S, expected, atol=1e-6)

    def test_identity_like_unit_vectors(self):
        """Unit vectors along axes: diagonal must be all 1s."""
        gpu_ops = _import_gpu_ops()
        X = np.eye(4, dtype=np.float64)
        S = gpu_ops.similarity_matrix(X)
        np.testing.assert_allclose(np.diag(S), np.ones(4), atol=1e-10)
        # Off-diagonal must be 0 for orthogonal unit vectors
        off_diag = S - np.diag(np.diag(S))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_single_vector(self):
        """Single vector: output shape must be (1, 1)."""
        gpu_ops = _import_gpu_ops()
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        S = gpu_ops.similarity_matrix(X)
        assert S.shape == (1, 1)
        expected_val = float(X[0] @ X[0])
        np.testing.assert_allclose(S[0, 0], expected_val, atol=1e-10)

    def test_empty_input(self):
        """Empty input (0, d): must return (0, 0) output."""
        gpu_ops = _import_gpu_ops()
        X = np.zeros((0, 10), dtype=np.float32)
        S = gpu_ops.similarity_matrix(X)
        assert S.shape == (0, 0)

    def test_dtype_preservation_float32(self):
        """Float32 input must produce float32 output."""
        gpu_ops = _import_gpu_ops()
        X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        assert S.dtype == np.float32

    def test_dtype_preservation_float64(self):
        """Float64 input must produce float64 output."""
        gpu_ops = _import_gpu_ops()
        X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float64)
        S = gpu_ops.similarity_matrix(X)
        assert S.dtype == np.float64

    def test_large_ish_matrix_accuracy(self):
        """500x128 matrix: verify numerical accuracy against numpy."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(123)
        X = rng.standard_normal((500, 128)).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        expected = X @ X.T
        np.testing.assert_allclose(S, expected, atol=1e-4)
        assert S.shape == (500, 500)

    def test_non_contiguous_input(self):
        """Sliced (non-contiguous) array must still work correctly."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(7)
        X_full = rng.standard_normal((10, 6)).astype(np.float64)
        # Take every other row -- non-contiguous
        X = X_full[::2]
        assert not X.flags["C_CONTIGUOUS"]
        S = gpu_ops.similarity_matrix(X)
        expected = np.ascontiguousarray(X) @ np.ascontiguousarray(X).T
        np.testing.assert_allclose(S, expected, atol=1e-10)

    def test_symmetry(self):
        """Similarity matrix must be symmetric: S == S.T."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 20)).astype(np.float32)
        S = gpu_ops.similarity_matrix(X)
        np.testing.assert_allclose(S, S.T, atol=1e-7)

    def test_no_input_mutation(self):
        """Input X must never be mutated."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(55)
        X = rng.standard_normal((10, 5)).astype(np.float32)
        X_orig = X.copy()
        _ = gpu_ops.similarity_matrix(X)
        np.testing.assert_array_equal(X, X_orig)


# ===========================================================================
# 2. find_close_pairs(S, threshold) tests
# ===========================================================================


class TestFindClosePairs:
    """Tests for gpu_ops.find_close_pairs(S, threshold)."""

    def test_known_3x3_one_pair(self):
        """3x3 matrix with exactly 1 pair above threshold."""
        gpu_ops = _import_gpu_ops()
        S = np.array(
            [
                [1.0, 0.6, 0.3],
                [0.6, 1.0, 0.2],
                [0.3, 0.2, 1.0],
            ]
        )
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        # Only pair (0, 1) with S=0.6 > 0.5
        assert len(ri) == 1
        assert ri[0] == 0
        assert ci[0] == 1

    def test_all_below_threshold_empty(self):
        """All pairs below threshold: must return empty arrays."""
        gpu_ops = _import_gpu_ops()
        S = np.eye(5)  # diagonal=1, off-diag=0
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        assert len(ri) == 0
        assert len(ci) == 0

    def test_all_above_threshold(self):
        """All pairs above threshold: returns all N*(N-1)/2 pairs."""
        gpu_ops = _import_gpu_ops()
        S = np.ones((3, 3))
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        assert len(ri) == 3  # (0,1), (0,2), (1,2)

    def test_diagonal_excluded(self):
        """Self-similarity on diagonal must be excluded."""
        gpu_ops = _import_gpu_ops()
        # Matrix where diagonal=10 (huge) but off-diagonal < threshold
        S = np.eye(4) * 10.0
        ri, ci = gpu_ops.find_close_pairs(S, threshold=5.0)
        assert len(ri) == 0  # no pairs, diagonal excluded

    def test_threshold_edge_exact(self):
        """Pair exactly at threshold must NOT be included (strict >)."""
        gpu_ops = _import_gpu_ops()
        threshold = 0.7
        S = np.array(
            [
                [1.0, threshold, 0.3],
                [threshold, 1.0, 0.2],
                [0.3, 0.2, 1.0],
            ]
        )
        ri, ci = gpu_ops.find_close_pairs(S, threshold=threshold)
        # S[0,1] == threshold exactly, strict > means NOT included
        assert len(ri) == 0

    def test_upper_triangle_only(self):
        """All returned pairs must have i < j."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        S = X @ X.T
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.0)
        for r, c in zip(ri, ci):
            assert r < c, f"Pair ({r}, {c}) violates i < j"

    def test_equivalence_with_numpy_triu(self):
        """Result must match np.where(np.triu(S, k=1) > threshold)."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(77)
        X = rng.standard_normal((15, 8)).astype(np.float64)
        S = X @ X.T
        threshold = 1.0
        ri, ci = gpu_ops.find_close_pairs(S, threshold=threshold)
        exp_ri, exp_ci = np.where(np.triu(S, k=1) > threshold)
        np.testing.assert_array_equal(ri, exp_ri)
        np.testing.assert_array_equal(ci, exp_ci)

    def test_no_input_mutation(self):
        """Input S must never be mutated."""
        gpu_ops = _import_gpu_ops()
        S = np.array(
            [
                [1.0, 0.9, 0.3],
                [0.9, 1.0, 0.8],
                [0.3, 0.8, 1.0],
            ]
        )
        S_orig = S.copy()
        _ = gpu_ops.find_close_pairs(S, threshold=0.5)
        np.testing.assert_array_equal(S, S_orig)


# ===========================================================================
# 3. batch_normalize(X) tests
# ===========================================================================


class TestBatchNormalize:
    """Tests for gpu_ops.batch_normalize(X)."""

    def test_output_row_norms(self):
        """All output rows must have unit L2 norm."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 32)).astype(np.float32)
        Y = gpu_ops.batch_normalize(X)
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_already_normalized_is_noop(self):
        """Already-normalized input: output should approximate input."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(11)
        X = rng.standard_normal((10, 5)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = gpu_ops.batch_normalize(X)
        np.testing.assert_allclose(Y, X, atol=1e-10)

    def test_zero_vector_row_no_nan(self):
        """Zero-vector rows must be handled gracefully (no NaN/inf)."""
        gpu_ops = _import_gpu_ops()
        X = np.zeros((3, 4), dtype=np.float32)
        X[1] = [1.0, 2.0, 3.0, 4.0]
        Y = gpu_ops.batch_normalize(X)
        # No NaN or inf anywhere
        assert not np.any(np.isnan(Y)), "NaN in output"
        assert not np.any(np.isinf(Y)), "Inf in output"
        # Row 0 and 2 (zero) should stay zero (norm < 1e-12 -> unchanged)
        np.testing.assert_allclose(Y[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(Y[2], 0.0, atol=1e-12)
        # Row 1 should be unit-norm
        assert abs(np.linalg.norm(Y[1]) - 1.0) < 1e-6

    def test_single_row(self):
        """Single row must work."""
        gpu_ops = _import_gpu_ops()
        X = np.array([[3.0, 4.0]], dtype=np.float64)
        Y = gpu_ops.batch_normalize(X)
        assert Y.shape == (1, 2)
        np.testing.assert_allclose(np.linalg.norm(Y[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(Y[0], [0.6, 0.8], atol=1e-10)

    def test_dtype_preservation_float32(self):
        """Float32 input must produce float32 output."""
        gpu_ops = _import_gpu_ops()
        X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
        Y = gpu_ops.batch_normalize(X)
        assert Y.dtype == np.float32

    def test_shape_preservation(self):
        """Output shape must match input shape."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 64)).astype(np.float64)
        Y = gpu_ops.batch_normalize(X)
        assert Y.shape == X.shape

    def test_no_input_mutation(self):
        """Input X must never be mutated."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(33)
        X = rng.standard_normal((5, 10)).astype(np.float32)
        X_orig = X.copy()
        _ = gpu_ops.batch_normalize(X)
        np.testing.assert_array_equal(X, X_orig)


# ===========================================================================
# 4. batch_matmul(A, B) tests
# ===========================================================================


class TestBatchMatmul:
    """Tests for gpu_ops.batch_matmul(A, B) -> A @ B."""

    def test_basic_matmul(self):
        """(5, 3) @ (3, 7) = (5, 7): verify vs np.matmul."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 3)).astype(np.float32)
        B = rng.standard_normal((3, 7)).astype(np.float32)
        C = gpu_ops.batch_matmul(A, B)
        expected = A @ B
        assert C.shape == (5, 7)
        np.testing.assert_allclose(C, expected, atol=1e-6)

    def test_scalar_like(self):
        """(1, d) @ (d, 1) = (1, 1): scalar-like result."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(0)
        A = rng.standard_normal((1, 10)).astype(np.float64)
        B = rng.standard_normal((10, 1)).astype(np.float64)
        C = gpu_ops.batch_matmul(A, B)
        assert C.shape == (1, 1)
        np.testing.assert_allclose(C, A @ B, atol=1e-10)

    def test_square_matrices(self):
        """Square (N, N) @ (N, N) = (N, N)."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(7)
        N = 20
        A = rng.standard_normal((N, N)).astype(np.float32)
        B = rng.standard_normal((N, N)).astype(np.float32)
        C = gpu_ops.batch_matmul(A, B)
        assert C.shape == (N, N)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)

    def test_non_contiguous_inputs(self):
        """Non-contiguous (sliced) inputs must work."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(3)
        A_full = rng.standard_normal((10, 8)).astype(np.float64)
        B_full = rng.standard_normal((8, 12)).astype(np.float64)
        # Take every other row/col -> non-contiguous
        A = A_full[::2]  # (5, 8)
        B = B_full[:, ::2]  # (8, 6)
        assert not A.flags["C_CONTIGUOUS"]
        C = gpu_ops.batch_matmul(A, B)
        expected = np.ascontiguousarray(A) @ np.ascontiguousarray(B)
        np.testing.assert_allclose(C, expected, atol=1e-10)

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(0)
        for dtype in [np.float32, np.float64]:
            A = rng.standard_normal((3, 4)).astype(dtype)
            B = rng.standard_normal((4, 5)).astype(dtype)
            C = gpu_ops.batch_matmul(A, B)
            assert C.dtype == dtype, f"Expected {dtype}, got {C.dtype}"

    def test_no_input_mutation(self):
        """Inputs A, B must never be mutated."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(22)
        A = rng.standard_normal((4, 6)).astype(np.float32)
        B = rng.standard_normal((6, 3)).astype(np.float32)
        A_orig = A.copy()
        B_orig = B.copy()
        _ = gpu_ops.batch_matmul(A, B)
        np.testing.assert_array_equal(A, A_orig)
        np.testing.assert_array_equal(B, B_orig)


# ===========================================================================
# 5. assign_clusters_matmul(patterns, threshold) tests
# ===========================================================================


class TestAssignClustersMatmul:
    """Tests for gpu_ops.assign_clusters_matmul(patterns, threshold)."""

    def test_three_tight_clusters(self):
        """3 tight clusters of 5 vectors each: must produce 3 distinct labels."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        d = 64
        clusters = []
        for _ in range(3):
            centroid = rng.standard_normal(d)
            centroid /= np.linalg.norm(centroid)
            # 5 vectors close to centroid (add small noise)
            group = centroid[None, :] + rng.standard_normal((5, d)) * 0.05
            group /= np.linalg.norm(group, axis=1, keepdims=True)
            clusters.append(group)
        patterns = np.vstack(clusters)

        labels = gpu_ops.assign_clusters_matmul(patterns, threshold=0.5)
        assert labels.shape == (15,)
        assert len(np.unique(labels)) == 3

        # Within each group of 5, labels should be identical
        assert len(np.unique(labels[0:5])) == 1
        assert len(np.unique(labels[5:10])) == 1
        assert len(np.unique(labels[10:15])) == 1

    def test_all_identical_one_cluster(self):
        """All identical vectors: must produce exactly 1 cluster."""
        gpu_ops = _import_gpu_ops()
        v = np.array([1.0, 0.0, 0.0])
        X = np.tile(v, (5, 1))
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert labels.shape == (5,)
        assert len(np.unique(labels)) == 1

    def test_all_orthogonal_n_clusters(self):
        """All orthogonal unit vectors: each in its own cluster (threshold=0.5)."""
        gpu_ops = _import_gpu_ops()
        X = np.eye(5, dtype=np.float64)
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert labels.shape == (5,)
        assert len(np.unique(labels)) == 5

    def test_contiguous_labels_from_zero(self):
        """Labels must be contiguous integers starting at 0."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 10)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        unique = np.unique(labels)
        expected_labels = np.arange(len(unique))
        np.testing.assert_array_equal(unique, expected_labels)

    def test_labels_dtype_int(self):
        """Labels must have integer dtype."""
        gpu_ops = _import_gpu_ops()
        X = np.eye(3, dtype=np.float64)
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_equivalence_with_dream_ops_assign_clusters(self):
        """Must match the original _assign_clusters from dream_ops.py."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 10)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        labels_gpu = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        labels_ref = _assign_clusters_reference(X, threshold=0.5)

        # Partitions must match (labels may differ in numbering)
        _assert_same_partition(labels_gpu, labels_ref)

    def test_equivalence_with_dream_ops_import(self):
        """Compare against actual dream_ops._assign_clusters import."""
        gpu_ops = _import_gpu_ops()
        from dream_ops import _assign_clusters

        rng = np.random.default_rng(77)
        X = rng.standard_normal((25, 8)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        labels_gpu = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        labels_cpu = _assign_clusters(X, threshold=0.5)

        _assert_same_partition(labels_gpu, labels_cpu)


# ===========================================================================
# 6. Integration equivalence: dream_cycle_xb baseline capture
# ===========================================================================


class TestDreamCycleEquivalence:
    """Capture numpy baseline of dream_cycle_xb.

    This test should PASS now (baseline capture on pure numpy) and
    continue passing after gpu_ops integration. It validates that
    dream_cycle_xb is deterministic with a fixed seed.
    """

    def test_dream_cycle_xb_deterministic_baseline(self):
        """dream_cycle_xb with fixed seed is reproducible."""
        from dream_ops import dream_cycle_xb

        rng = np.random.default_rng(42)
        N, d = 50, 32
        patterns = rng.standard_normal((N, d)).astype(np.float64)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
        beta = 5.0

        # Run twice with same seed -- must produce identical results
        report1 = dream_cycle_xb(patterns, beta, seed=42)
        report2 = dream_cycle_xb(patterns, beta, seed=42)

        np.testing.assert_array_equal(report1.patterns, report2.patterns)
        assert report1.pruned_indices == report2.pruned_indices
        assert report1.merge_map == report2.merge_map
        assert len(report1.associations) == len(report2.associations)

    def test_dream_cycle_xb_output_shape(self):
        """dream_cycle_xb output patterns have correct dimensionality."""
        from dream_ops import dream_cycle_xb

        rng = np.random.default_rng(0)
        N, d = 30, 16
        patterns = rng.standard_normal((N, d)).astype(np.float64)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
        beta = 5.0

        report = dream_cycle_xb(patterns, beta, seed=0)

        # Output patterns must have same dimension d
        assert report.patterns.shape[1] == d
        # Output may have fewer rows (pruning/merging)
        assert report.patterns.shape[0] <= N
        # Output patterns must be finite
        assert np.all(np.isfinite(report.patterns))

    def test_dream_cycle_xb_preserves_unit_norms(self):
        """Output patterns should be approximately unit-norm."""
        from dream_ops import dream_cycle_xb

        rng = np.random.default_rng(7)
        N, d = 40, 24
        patterns = rng.standard_normal((N, d)).astype(np.float64)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
        beta = 5.0

        report = dream_cycle_xb(patterns, beta, seed=7)
        norms = np.linalg.norm(report.patterns, axis=1)
        # After full dream cycle, norms should be close to 1
        # (repulsion, normalization steps keep them near unit)
        np.testing.assert_allclose(norms, 1.0, atol=0.1)


# ===========================================================================
# 7. Fallback behavior tests
# ===========================================================================


class TestFallbackBehavior:
    """Verify all functions work when torch is unavailable (numpy fallback)."""

    def test_has_torch_flag_false_when_no_torch(self):
        """HAS_TORCH must be False when torch is not installed."""
        gpu_ops = _import_gpu_ops()
        assert gpu_ops.HAS_TORCH is False

    def test_is_gpu_available_false_when_no_torch(self):
        """is_gpu_available() must return False when torch is not installed."""
        gpu_ops = _import_gpu_ops()
        assert gpu_ops.is_gpu_available() is False

    def test_similarity_matrix_fallback_exact(self):
        """Fallback similarity_matrix must return exact X @ X.T."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 4)).astype(np.float64)
        S = gpu_ops.similarity_matrix(X)
        expected = X @ X.T
        np.testing.assert_array_equal(S, expected)

    def test_find_close_pairs_fallback(self):
        """Fallback find_close_pairs must match numpy triu > threshold."""
        gpu_ops = _import_gpu_ops()
        S = np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.6],
                [0.3, 0.6, 1.0],
            ]
        )
        ri, ci = gpu_ops.find_close_pairs(S, threshold=0.5)
        exp_ri, exp_ci = np.where(np.triu(S, k=1) > 0.5)
        np.testing.assert_array_equal(ri, exp_ri)
        np.testing.assert_array_equal(ci, exp_ci)

    def test_batch_normalize_fallback(self):
        """Fallback batch_normalize must produce unit-norm rows."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 5)).astype(np.float64)
        Y = gpu_ops.batch_normalize(X)
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_batch_matmul_fallback_exact(self):
        """Fallback batch_matmul must return exact A @ B."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 6)).astype(np.float64)
        B = rng.standard_normal((6, 3)).astype(np.float64)
        C = gpu_ops.batch_matmul(A, B)
        expected = A @ B
        np.testing.assert_array_equal(C, expected)

    def test_assign_clusters_matmul_fallback(self):
        """Fallback assign_clusters_matmul must produce valid clustering."""
        gpu_ops = _import_gpu_ops()
        X = np.eye(3, dtype=np.float64)
        labels = gpu_ops.assign_clusters_matmul(X, threshold=0.5)
        assert labels.shape == (3,)
        assert len(np.unique(labels)) == 3

    def test_mock_torch_unavailable_all_functions(self):
        """With torch explicitly mocked as unavailable, all functions must work."""
        # Force reimport with torch mocked as missing
        with mock.patch.dict(sys.modules, {"torch": None}):
            # Remove gpu_ops from cache to force reimport
            sys.modules.pop("gpu_ops", None)
            try:
                gpu_ops = importlib.import_module("gpu_ops")
                assert gpu_ops.HAS_TORCH is False

                rng = np.random.default_rng(42)
                X = rng.standard_normal((5, 3)).astype(np.float32)

                # All functions must work without error
                S = gpu_ops.similarity_matrix(X)
                assert S.shape == (5, 5)

                ri, ci = gpu_ops.find_close_pairs(S, threshold=0.0)
                assert len(ri) > 0

                Y = gpu_ops.batch_normalize(X)
                norms = np.linalg.norm(Y, axis=1)
                np.testing.assert_allclose(norms, 1.0, atol=1e-6)

                A = rng.standard_normal((3, 4)).astype(np.float32)
                B = rng.standard_normal((4, 2)).astype(np.float32)
                C = gpu_ops.batch_matmul(A, B)
                assert C.shape == (3, 2)

                X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
                labels = gpu_ops.assign_clusters_matmul(X_norm, threshold=0.5)
                assert labels.shape == (5,)
            finally:
                # Clean up module cache
                sys.modules.pop("gpu_ops", None)

    def test_fallback_results_identical_to_numpy(self):
        """All fallback results must be numerically identical to direct numpy."""
        gpu_ops = _import_gpu_ops()
        rng = np.random.default_rng(12345)

        # similarity_matrix
        X = rng.standard_normal((20, 10)).astype(np.float64)
        S_gpu = gpu_ops.similarity_matrix(X)
        S_np = X @ X.T
        np.testing.assert_array_equal(S_gpu, S_np)

        # batch_matmul
        A = rng.standard_normal((10, 8)).astype(np.float64)
        B = rng.standard_normal((8, 6)).astype(np.float64)
        C_gpu = gpu_ops.batch_matmul(A, B)
        C_np = A @ B
        np.testing.assert_array_equal(C_gpu, C_np)

        # batch_normalize
        Y_gpu = gpu_ops.batch_normalize(X)
        norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y_np = X / norms
        np.testing.assert_array_equal(Y_gpu, Y_np)
