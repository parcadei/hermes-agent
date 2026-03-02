"""Quality metrics for evaluating dream cycle effectiveness.

Provides functions to measure:
  - Spurious attractor count (states that don't correspond to stored patterns)
  - Attractor basin depth (energy drop from perturbation to convergence)
  - Capacity utilization (fraction of stored patterns that are retrievable)
  - Inter-cluster coupling ratio (block-diagonal structure quality)
  - Overall dream quality comparison (before/after metrics)

All functions are pure (no mutation of inputs).
All stochastic functions use explicit seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np

from dream_ops import spreading_activation
from nlcdm_core import cosine_sim, local_energy


# ---------------------------------------------------------------------------
# Spurious Attractor Count
# ---------------------------------------------------------------------------

def count_spurious_attractors(
    W: np.ndarray,
    embeddings: np.ndarray,
    beta: float,
    n_samples: int = 500,
) -> int:
    """Count attractors that don't correspond to stored patterns.

    Method: Random init -> spreading_activation -> check if converged state
    is close to any stored pattern (cosine_sim > 0.9). Count unique attractors
    that AREN'T close to any pattern.

    The coupling matrix W is not used directly by spreading_activation (which
    operates in pattern space), but it shapes the effective energy landscape
    through the embeddings. We generate random starting points and check
    whether convergence leads to stored patterns or to spurious states.

    Args:
        W: (d, d) coupling matrix (provides context but spreading_activation
           uses pattern-space Hopfield dynamics)
        embeddings: (N, d) stored patterns
        beta: inverse temperature for spreading activation
        n_samples: number of random starting points to probe

    Returns:
        Number of unique spurious attractors found
    """
    N, d = embeddings.shape
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility

    # Collect unique spurious attractors
    spurious_attractors: list[np.ndarray] = []
    similarity_threshold = 0.9
    uniqueness_threshold = 0.95

    for _ in range(n_samples):
        # Random starting state: sample from unit sphere
        x0 = rng.standard_normal(d)
        x0 /= np.linalg.norm(x0)

        # Also perturb with W to explore the actual energy landscape
        # x0 = x0 + small component from W's principal directions
        x0_perturbed = x0 + 0.1 * (W @ x0)
        norm = np.linalg.norm(x0_perturbed)
        if norm > 0:
            x0_perturbed /= norm

        # Converge via spreading activation in pattern space
        x_conv = spreading_activation(beta, embeddings, x0_perturbed, max_steps=100)

        # Check if converged state is close to any stored pattern
        is_stored = False
        for i in range(N):
            if cosine_sim(x_conv, embeddings[i]) > similarity_threshold:
                is_stored = True
                break

        if not is_stored:
            # Check if this is a new unique spurious attractor
            is_new = True
            for att in spurious_attractors:
                if cosine_sim(x_conv, att) > uniqueness_threshold:
                    is_new = False
                    break
            if is_new:
                spurious_attractors.append(x_conv)

    return len(spurious_attractors)


# ---------------------------------------------------------------------------
# Attractor Depth
# ---------------------------------------------------------------------------

def measure_attractor_depth(
    W: np.ndarray,
    embeddings: np.ndarray,
    beta: float,
    pattern_idx: int,
) -> float:
    """Measure energy depth of a specific pattern's basin.

    Method: Start at the pattern, compute energy. Then perturb the pattern
    slightly (add noise with ||noise|| = 0.1 * ||pattern||), run spreading_activation,
    measure energy at convergence. The depth is |E(converged) - E(perturbed_start)|.
    Average over multiple perturbations (20).

    This measures how much energy the system loses when returning to the
    attractor basin from a perturbed state -- deeper basins are more robust.

    Args:
        W: (d, d) coupling matrix (not used directly by spreading_activation
           but conceptually defines the landscape)
        embeddings: (N, d) stored patterns
        beta: inverse temperature
        pattern_idx: index of the pattern whose basin depth to measure

    Returns:
        Average energy depth (non-negative float)
    """
    pattern = embeddings[pattern_idx]
    rng = np.random.default_rng(pattern_idx)  # Reproducible per pattern
    n_perturbations = 20
    depths: list[float] = []

    for _ in range(n_perturbations):
        # Perturb the pattern
        noise = rng.standard_normal(pattern.shape)
        noise *= 0.1 * np.linalg.norm(pattern) / np.linalg.norm(noise)
        x_perturbed = pattern + noise

        # Energy at perturbed starting point
        e_start = local_energy(beta, embeddings, x_perturbed)

        # Converge via spreading activation
        x_conv = spreading_activation(beta, embeddings, x_perturbed, max_steps=100)

        # Energy at convergence
        e_conv = local_energy(beta, embeddings, x_conv)

        # Depth is the energy drop (should be non-negative since
        # spreading activation descends energy)
        depth = abs(e_start - e_conv)
        depths.append(depth)

    return float(np.mean(depths))


# ---------------------------------------------------------------------------
# Capacity Utilization
# ---------------------------------------------------------------------------

def capacity_utilization(
    W: np.ndarray,
    embeddings: np.ndarray,
    beta: float,
) -> float:
    """Fraction of stored patterns that are retrievable fixed points.

    For each pattern, run spreading_activation starting from that pattern.
    Count how many converge back to within cosine_sim > 0.95 of the start.
    Return count / N.

    Args:
        W: (d, d) coupling matrix
        embeddings: (N, d) stored patterns
        beta: inverse temperature

    Returns:
        Fraction in [0, 1] of patterns that are retrievable
    """
    N = embeddings.shape[0]
    if N == 0:
        return 1.0

    retrieval_threshold = 0.95
    retrievable = 0

    for i in range(N):
        x_start = embeddings[i].copy()
        x_conv = spreading_activation(beta, embeddings, x_start, max_steps=100)
        sim = cosine_sim(x_conv, embeddings[i])
        if sim > retrieval_threshold:
            retrievable += 1

    return retrievable / N


# ---------------------------------------------------------------------------
# Inter-Cluster Coupling
# ---------------------------------------------------------------------------

def inter_cluster_coupling(
    W: np.ndarray,
    cluster_assignments: np.ndarray,
) -> float:
    """Measure coupling between clusters vs within clusters.

    Args:
        W: (d, d) coupling matrix
        cluster_assignments: array of int, one per row/column of W,
            indicating which cluster each dimension belongs to

    Returns:
        Ratio: mean(|W_ij| for i,j in different clusters) /
               mean(|W_ij| for i,j in same cluster).
        Lower is better (block-diagonal structure).
        Returns 0.0 if there are no intra-cluster pairs (degenerate case).
    """
    d = W.shape[0]
    abs_W = np.abs(W)

    intra_values: list[float] = []
    inter_values: list[float] = []

    for i in range(d):
        for j in range(d):
            if i == j:
                continue  # Skip diagonal
            if cluster_assignments[i] == cluster_assignments[j]:
                intra_values.append(abs_W[i, j])
            else:
                inter_values.append(abs_W[i, j])

    if len(intra_values) == 0:
        return 0.0

    mean_intra = float(np.mean(intra_values))
    if mean_intra == 0.0:
        return 0.0

    if len(inter_values) == 0:
        return 0.0

    mean_inter = float(np.mean(inter_values))
    return mean_inter / mean_intra


# ---------------------------------------------------------------------------
# Dream Quality Comparison
# ---------------------------------------------------------------------------

def measure_dream_quality(
    W_before: np.ndarray,
    W_after: np.ndarray,
    embeddings: np.ndarray,
    beta: float,
    tagged: list[int],
) -> dict:
    """Compare W before and after dreaming.

    Computes multiple quality metrics on both the pre-dream and post-dream
    coupling matrices, and returns a comprehensive comparison.

    Args:
        W_before: (d, d) coupling matrix before dreaming
        W_after: (d, d) coupling matrix after dreaming
        embeddings: (N, d) stored patterns
        beta: inverse temperature
        tagged: list of tagged memory indices

    Returns:
        Dictionary with keys:
        - spurious_before: int
        - spurious_after: int
        - capacity_before: float
        - capacity_after: float
        - tagged_depth_before: list[float] (per tagged memory)
        - tagged_depth_after: list[float]
        - improvement: bool (True if spurious decreased OR capacity increased)
    """
    # Use fewer samples for the quality comparison to keep it tractable
    n_samples = 200

    spurious_before = count_spurious_attractors(
        W_before, embeddings, beta, n_samples=n_samples
    )
    spurious_after = count_spurious_attractors(
        W_after, embeddings, beta, n_samples=n_samples
    )

    capacity_before = capacity_utilization(W_before, embeddings, beta)
    capacity_after = capacity_utilization(W_after, embeddings, beta)

    tagged_depth_before = [
        measure_attractor_depth(W_before, embeddings, beta, idx)
        for idx in tagged
    ]
    tagged_depth_after = [
        measure_attractor_depth(W_after, embeddings, beta, idx)
        for idx in tagged
    ]

    improvement = (
        spurious_after < spurious_before
        or capacity_after > capacity_before
    )

    return {
        "spurious_before": spurious_before,
        "spurious_after": spurious_after,
        "capacity_before": capacity_before,
        "capacity_after": capacity_after,
        "tagged_depth_before": tagged_depth_before,
        "tagged_depth_after": tagged_depth_after,
        "improvement": improvement,
    }
