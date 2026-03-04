"""H3: _assign_clusters N^2 loop is actually slow.

Discriminative test: time the existing _assign_clusters (Python N^2 loop
with per-pair dot products) vs a matmul-based approach (compute full
similarity matrix, then union-find on pre-computed pairs).
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python")
from dream_ops import _assign_clusters  # noqa: E402


def _matmul_assign_clusters(patterns, threshold=0.5):
    """Matmul-based _assign_clusters: same logic, vectorized similarity."""
    N = patterns.shape[0]
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Vectorized: compute all similarities at once
    S = patterns @ patterns.T
    np.fill_diagonal(S, -2.0)
    close_i, close_j = np.where(np.triu(S, k=1) > threshold)

    for idx in range(len(close_i)):
        union(int(close_i[idx]), int(close_j[idx]))

    # Map roots to contiguous labels
    root_to_label = {}
    labels = np.zeros(N, dtype=int)
    next_label = 0
    for i in range(N):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels


def _time_fn(fn, warmup=1, trials=3):
    """Run fn warmup+trials times, return median of trials."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def test_h3_assign_clusters_n500():
    """At N=500: compare original loop vs matmul-based clustering."""
    rng = np.random.default_rng(42)
    N, d = 500, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    threshold = 0.5

    t_loop = _time_fn(lambda: _assign_clusters(X, threshold=threshold))
    t_matmul = _time_fn(lambda: _matmul_assign_clusters(X, threshold=threshold))

    # Verify correctness: both should produce same clustering
    labels_loop = _assign_clusters(X, threshold=threshold)
    labels_matmul = _matmul_assign_clusters(X, threshold=threshold)

    # Labels may differ in numbering, but clustering structure must match
    # Check: for all pairs, same-cluster in loop <=> same-cluster in matmul
    correct = True
    for i in range(N):
        for j in range(i + 1, N):
            same_loop = labels_loop[i] == labels_loop[j]
            same_matmul = labels_matmul[i] == labels_matmul[j]
            if same_loop != same_matmul:
                correct = False
                break
        if not correct:
            break

    speedup = t_loop / t_matmul if t_matmul > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"H3: _assign_clusters Performance (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Original (Python loop):    {t_loop*1000:8.2f} ms")
    print(f"  Matmul-based:              {t_matmul*1000:8.2f} ms")
    print(f"  Speedup:                   {speedup:.1f}x")
    print(f"  Results match:             {correct}")
    print(f"  Unique clusters (loop):    {len(set(labels_loop))}")
    print(f"  Unique clusters (matmul):  {len(set(labels_matmul))}")
    print(f"{'='*60}")

    return {
        "loop_ms": t_loop * 1000,
        "matmul_ms": t_matmul * 1000,
        "speedup": speedup,
        "correct": correct,
    }


def test_h3_assign_clusters_n1000():
    """At N=1000: the main scale target."""
    rng = np.random.default_rng(42)
    N, d = 1000, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    threshold = 0.5

    t_loop = _time_fn(lambda: _assign_clusters(X, threshold=threshold))
    t_matmul = _time_fn(lambda: _matmul_assign_clusters(X, threshold=threshold))

    labels_loop = _assign_clusters(X, threshold=threshold)
    labels_matmul = _matmul_assign_clusters(X, threshold=threshold)

    # Quick correctness check: count clusters
    n_clusters_loop = len(set(labels_loop))
    n_clusters_matmul = len(set(labels_matmul))

    speedup = t_loop / t_matmul if t_matmul > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"H3: _assign_clusters Performance (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Original (Python loop):    {t_loop*1000:8.2f} ms")
    print(f"  Matmul-based:              {t_matmul*1000:8.2f} ms")
    print(f"  Speedup:                   {speedup:.1f}x")
    print(f"  Clusters (loop):           {n_clusters_loop}")
    print(f"  Clusters (matmul):         {n_clusters_matmul}")
    print(f"{'='*60}")

    return {
        "loop_ms": t_loop * 1000,
        "matmul_ms": t_matmul * 1000,
        "speedup": speedup,
    }


def test_h3_assign_clusters_n2000():
    """At N=2000: test quadratic scaling."""
    rng = np.random.default_rng(42)
    N, d = 2000, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    threshold = 0.5

    t_loop = _time_fn(lambda: _assign_clusters(X, threshold=threshold), warmup=0, trials=1)
    t_matmul = _time_fn(lambda: _matmul_assign_clusters(X, threshold=threshold))

    speedup = t_loop / t_matmul if t_matmul > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"H3: _assign_clusters Performance (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Original (Python loop):    {t_loop*1000:8.2f} ms")
    print(f"  Matmul-based:              {t_matmul*1000:8.2f} ms")
    print(f"  Speedup:                   {speedup:.1f}x")
    print(f"{'='*60}")

    return {
        "loop_ms": t_loop * 1000,
        "matmul_ms": t_matmul * 1000,
        "speedup": speedup,
    }


if __name__ == "__main__":
    test_h3_assign_clusters_n500()
    print()
    test_h3_assign_clusters_n1000()
    print()
    test_h3_assign_clusters_n2000()
