"""Performance benchmarks: vectorized dream ops vs original."""
import time
import numpy as np
from dream_ops import rem_unlearn_xb, rem_explore_cross_domain_xb, dream_cycle_xb


def make_unit_vectors(n, dim=128, seed=42):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def make_clustered(n, n_clusters=5, dim=128, seed=42):
    rng = np.random.default_rng(seed)
    per_cluster = n // n_clusters
    centroids = rng.standard_normal((n_clusters, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    patterns = []
    labels = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            p = centroids[c] + 0.10 * rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            patterns.append(p)
            labels.append(c)
    return np.array(patterns), np.array(labels)


def bench_unlearn(n_values=[500, 1000, 2000]):
    print("\n=== rem_unlearn_xb: S-based vs MC ===")
    print(f"{'N':>6} {'MC (ms)':>10} {'S-based (ms)':>14} {'Speedup':>10}")
    print("-" * 45)
    for n in n_values:
        X = make_unit_vectors(n, seed=100+n)
        S = X @ X.T

        # MC path (similarity_matrix=None)
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        rem_unlearn_xb(X, beta=5.0, n_probes=200, rng=rng)
        mc_ms = (time.perf_counter() - t0) * 1000

        # S-based path
        t0 = time.perf_counter()
        rem_unlearn_xb(X, beta=5.0, n_probes=200, rng=np.random.default_rng(42), similarity_matrix=S)
        s_ms = (time.perf_counter() - t0) * 1000

        speedup = mc_ms / max(s_ms, 0.001)
        print(f"{n:>6} {mc_ms:>10.1f} {s_ms:>14.2f} {speedup:>9.1f}x")


def bench_explore(n_values=[500, 1000, 2000]):
    print("\n=== rem_explore_cross_domain_xb (batched) ===")
    print(f"{'N':>6} {'Time (ms)':>10} {'Associations':>14}")
    print("-" * 35)
    for n in n_values:
        X, labels = make_clustered(n, seed=200+n)
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        assoc = rem_explore_cross_domain_xb(X, labels, n_probes=max(n, 50), rng=rng)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>6} {elapsed_ms:>10.1f} {len(assoc):>14}")


def bench_dream_cycle(n_values=[500, 1000, 2000]):
    print("\n=== dream_cycle_xb total ===")
    print(f"{'N':>6} {'Time (ms)':>10}")
    print("-" * 20)
    for n in n_values:
        X, labels = make_clustered(n, seed=300+n)
        importances = np.full(n, 0.5)
        t0 = time.perf_counter()
        report = dream_cycle_xb(X, beta=5.0, importances=importances, labels=labels, seed=42)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>6} {elapsed_ms:>10.1f}")


def bench_s_precompute(n_values=[500, 1000, 2000, 5000]):
    print("\n=== S = X @ X.T precompute ===")
    print(f"{'N':>6} {'Time (ms)':>10} {'Memory (MB)':>12}")
    print("-" * 32)
    for n in n_values:
        X = make_unit_vectors(n, seed=400+n)
        t0 = time.perf_counter()
        S = X @ X.T
        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_mb = S.nbytes / 1024**2
        print(f"{n:>6} {elapsed_ms:>10.2f} {mem_mb:>12.1f}")


if __name__ == "__main__":
    bench_unlearn()
    bench_explore()
    bench_dream_cycle()
    bench_s_precompute()
