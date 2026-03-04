"""H1: Numpy matmul is the actual bottleneck (not pair loops).

Discriminative test: time the matmul (S = X @ X.T), the pair-finding
(np.where on triu), and a typical pair loop separately for realistic N.
"""

import time
import numpy as np


def _time_it(fn, warmup=2, trials=5):
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


def test_h1_bottleneck_n1000():
    """At N=1000, d=128: measure matmul vs pair-find vs pair-loop."""
    rng = np.random.default_rng(42)
    N, d = 1000, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 1) Time matmul: S = X @ X.T
    def do_matmul():
        return X @ X.T

    t_matmul = _time_it(do_matmul)

    # Pre-compute S for the next two stages
    S = X @ X.T
    np.fill_diagonal(S, -2.0)
    threshold = 0.7

    # 2) Time pair-finding: np.where(np.triu(S, k=1) > threshold)
    def do_pair_find():
        return np.where(np.triu(S, k=1) > threshold)

    t_pair_find = _time_it(do_pair_find)

    # Find actual pairs for the loop test
    close_i, close_j = np.where(np.triu(S, k=1) > threshold)
    n_pairs = len(close_i)

    # 3) Time pair loop: the repulsion-style loop from nrem_repulsion_xb
    eta = 0.01

    def do_pair_loop():
        Y = X.copy()
        for idx in range(len(close_i)):
            i, j = int(close_i[idx]), int(close_j[idx])
            diff = Y[i] - Y[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-12:
                continue
            direction = diff / diff_norm
            Y[i] = Y[i] + eta * direction
            Y[j] = Y[j] - eta * direction
        return Y

    t_pair_loop = _time_it(do_pair_loop)

    # 4) Time the full nrem_repulsion_xb-style pipeline (matmul + find + loop)
    t_total = t_matmul + t_pair_find + t_pair_loop

    print(f"\n{'='*60}")
    print(f"H1: Bottleneck Analysis (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Matmul (X @ X.T):        {t_matmul*1000:8.2f} ms  ({t_matmul/t_total*100:5.1f}%)")
    print(f"  Pair-find (np.where):     {t_pair_find*1000:8.2f} ms  ({t_pair_find/t_total*100:5.1f}%)")
    print(f"  Pair-loop ({n_pairs} pairs):  {t_pair_loop*1000:8.2f} ms  ({t_pair_loop/t_total*100:5.1f}%)")
    print(f"  Total pipeline:           {t_total*1000:8.2f} ms")
    print(f"  Pairs found:              {n_pairs}")
    print(f"{'='*60}")

    # Verdict: which component dominates?
    components = {"matmul": t_matmul, "pair_find": t_pair_find, "pair_loop": t_pair_loop}
    bottleneck = max(components, key=components.get)
    print(f"  BOTTLENECK: {bottleneck} ({components[bottleneck]/t_total*100:.1f}% of total)")

    # The hypothesis says matmul is the bottleneck.
    # At N=1000, matmul is O(N^2*d), pair_find is O(N^2), pair_loop is O(P).
    # We report without asserting, since this is discriminative validation.

    return {
        "matmul_ms": t_matmul * 1000,
        "pair_find_ms": t_pair_find * 1000,
        "pair_loop_ms": t_pair_loop * 1000,
        "n_pairs": n_pairs,
        "bottleneck": bottleneck,
    }


def test_h1_bottleneck_n5000():
    """At N=5000, d=128: check if matmul dominance grows with N."""
    rng = np.random.default_rng(42)
    N, d = 5000, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 1) Time matmul
    def do_matmul():
        return X @ X.T

    t_matmul = _time_it(do_matmul, warmup=1, trials=3)

    # 2) Time pair-finding on the result
    S = X @ X.T
    np.fill_diagonal(S, -2.0)
    threshold = 0.7

    def do_pair_find():
        return np.where(np.triu(S, k=1) > threshold)

    t_pair_find = _time_it(do_pair_find, warmup=1, trials=3)

    close_i, close_j = np.where(np.triu(S, k=1) > threshold)
    n_pairs = len(close_i)

    # 3) Pair loop (cap at 500 pairs to keep test fast)
    cap = min(len(close_i), 500)
    ci, cj = close_i[:cap], close_j[:cap]
    eta = 0.01

    def do_pair_loop():
        Y = X.copy()
        for idx in range(cap):
            i, j = int(ci[idx]), int(cj[idx])
            diff = Y[i] - Y[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-12:
                continue
            direction = diff / diff_norm
            Y[i] = Y[i] + eta * direction
            Y[j] = Y[j] - eta * direction
        return Y

    t_pair_loop = _time_it(do_pair_loop, warmup=1, trials=3)
    t_total = t_matmul + t_pair_find + t_pair_loop

    print(f"\n{'='*60}")
    print(f"H1: Bottleneck Analysis (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Matmul (X @ X.T):        {t_matmul*1000:8.2f} ms  ({t_matmul/t_total*100:5.1f}%)")
    print(f"  Pair-find (np.where):     {t_pair_find*1000:8.2f} ms  ({t_pair_find/t_total*100:5.1f}%)")
    print(f"  Pair-loop ({n_pairs} total, {cap} capped): {t_pair_loop*1000:8.2f} ms  ({t_pair_loop/t_total*100:5.1f}%)")
    print(f"  Total pipeline:           {t_total*1000:8.2f} ms")
    print(f"  Pairs found:              {n_pairs}")
    print(f"{'='*60}")

    components = {"matmul": t_matmul, "pair_find": t_pair_find, "pair_loop": t_pair_loop}
    bottleneck = max(components, key=components.get)
    print(f"  BOTTLENECK: {bottleneck} ({components[bottleneck]/t_total*100:.1f}% of total)")

    return {
        "matmul_ms": t_matmul * 1000,
        "pair_find_ms": t_pair_find * 1000,
        "pair_loop_ms": t_pair_loop * 1000,
        "n_pairs": n_pairs,
        "bottleneck": bottleneck,
    }


def test_h1_bottleneck_clustered_n1000():
    """With clustered data that produces many pairs, measure the real breakdown.

    Random 128-d unit vectors have near-zero cosine similarity, so threshold=0.7
    finds 0 pairs. Real memory systems have clustered embeddings. This test
    creates 20 clusters of 50 vectors each (perturbed centroids) to produce
    realistic pair counts.
    """
    rng = np.random.default_rng(42)
    N, d = 1000, 128
    n_clusters = 20
    per_cluster = N // n_clusters

    # Generate clustered data
    centroids = rng.standard_normal((n_clusters, d))
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    X = np.empty((N, d))
    for c in range(n_clusters):
        # noise=0.05 gives realistic intra-cluster similarity (~0.7-0.85)
        # similar to real sentence embeddings within a topic
        noise = rng.standard_normal((per_cluster, d)) * 0.05
        X[c * per_cluster : (c + 1) * per_cluster] = centroids[c] + noise
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    threshold = 0.7

    # 1) Time matmul
    def do_matmul():
        return X @ X.T

    t_matmul = _time_it(do_matmul)

    # 2) Time pair-finding
    S = X @ X.T
    np.fill_diagonal(S, -2.0)

    def do_pair_find():
        return np.where(np.triu(S, k=1) > threshold)

    t_pair_find = _time_it(do_pair_find)

    close_i, close_j = np.where(np.triu(S, k=1) > threshold)
    n_pairs = len(close_i)

    # 3) Time pair loop (repulsion-style)
    eta = 0.01

    def do_pair_loop():
        Y = X.copy()
        for idx in range(len(close_i)):
            i, j = int(close_i[idx]), int(close_j[idx])
            diff = Y[i] - Y[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-12:
                continue
            direction = diff / diff_norm
            Y[i] = Y[i] + eta * direction
            Y[j] = Y[j] - eta * direction
        return Y

    t_pair_loop = _time_it(do_pair_loop)

    # 4) ALSO time the original Python N^2 loop (as in _assign_clusters)
    #    to show the cost WITHOUT pre-computed matmul
    def do_python_n2_dots():
        """Python loop computing individual dot products (the original approach)."""
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                sim = float(X[i] @ X[j])
                if sim > threshold:
                    count += 1
        return count

    t_python_n2 = _time_it(do_python_n2_dots, warmup=0, trials=1)

    t_vectorized_total = t_matmul + t_pair_find + t_pair_loop
    speedup_vs_python = t_python_n2 / t_vectorized_total if t_vectorized_total > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"H1: Bottleneck Analysis - CLUSTERED DATA (N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Matmul (X @ X.T):          {t_matmul*1000:8.2f} ms  ({t_matmul/t_vectorized_total*100:5.1f}%)")
    print(f"  Pair-find (np.where):       {t_pair_find*1000:8.2f} ms  ({t_pair_find/t_vectorized_total*100:5.1f}%)")
    print(f"  Pair-loop ({n_pairs} pairs): {t_pair_loop*1000:8.2f} ms  ({t_pair_loop/t_vectorized_total*100:5.1f}%)")
    print(f"  Vectorized total:           {t_vectorized_total*1000:8.2f} ms")
    print(f"  Python N^2 loop:            {t_python_n2*1000:8.2f} ms")
    print(f"  Speedup (vec vs python):    {speedup_vs_python:.1f}x")
    print(f"  Pairs found:                {n_pairs}")
    print(f"{'='*60}")

    components = {"matmul": t_matmul, "pair_find": t_pair_find, "pair_loop": t_pair_loop}
    bottleneck = max(components, key=components.get)
    print(f"  BOTTLENECK (vectorized):    {bottleneck} ({components[bottleneck]/t_vectorized_total*100:.1f}%)")
    print(f"  Python N^2 vs matmul alone: {t_python_n2/t_matmul:.1f}x slower")


if __name__ == "__main__":
    test_h1_bottleneck_n1000()
    print()
    test_h1_bottleneck_n5000()
    print()
    test_h1_bottleneck_clustered_n1000()
