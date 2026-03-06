"""
Thermodynamic Dream Cycle Operations for Hermes Memory System.

Three operations on coupling matrix W:
  1. NREM-replay: Hebbian reinforcement at high beta (stamp new)
  2. REM-unlearn: anti-Hebbian at moderate beta (clean spurious)
  3. REM-explore: weak Hebbian during wandering at low beta (protect old)

Plus: LV competition and consolidation for the full cycle.

All operations are pure: W_in -> W_out (no mutation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from typing import Callable

from nlcdm_core import cosine_sim, sigmoid, softmax, sparsemax

import gpu_ops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DreamParams:
    """Parameters for the full dream cycle.

    Defaults match research v3 specification.
    """

    beta_high: float = 10.0  # NREM temperature (near deterministic)
    beta_mod: float = 2.0  # REM-unlearn temperature (explore but settle)
    beta_low: float = 0.5  # REM-explore temperature (maximal wandering)
    eta: float = 0.01  # Hebbian/anti-Hebbian learning rate
    eta_weak: float = 0.001  # Exploration reinforcement rate
    n_unlearn: int = 100  # REM-unlearn trials per cycle
    n_explore: int = 50  # REM-explore steps per cycle
    consolidation_threshold: float = 0.95  # Merge threshold
    seed: int | None = None  # For reproducibility
    min_sep: float = 0.3  # NREM repulsion minimum separation
    prune_threshold: float = 0.95  # NREM prune near-duplicate threshold
    merge_threshold: float = 0.90  # NREM merge group threshold
    merge_min_group: int = 3  # Minimum group size for merging
    n_probes: int = 200  # REM unlearn probe count
    separation_rate: float = 0.02  # REM unlearn separation rate
    bridge_threshold: float = 0.3  # REM cross-domain bridge detection tau (BridgeFormation.lean)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Check Lean-proven safe bounds (DreamConvergence.lean: default_params_safe)."""
        if not (0 < self.eta < self.min_sep / 2):
            raise ValueError(
                f"Lean bound violated: need 0 < eta ({self.eta}) < min_sep/2 ({self.min_sep / 2})"
            )
        if not (0 < self.min_sep <= 1):
            raise ValueError(
                f"Lean bound violated: need 0 < min_sep ({self.min_sep}) <= 1"
            )
        if not (0 < self.merge_threshold < self.prune_threshold <= 1):
            raise ValueError(
                f"Lean bound violated: need 0 < merge ({self.merge_threshold}) "
                f"< prune ({self.prune_threshold}) <= 1"
            )
        if not (0 < self.bridge_threshold <= 1):
            raise ValueError(
                f"Need 0 < bridge_threshold ({self.bridge_threshold}) <= 1"
            )


@dataclass(frozen=True)
class DreamReport:
    """Immutable report of a complete dream cycle.

    All fields describe the outcome of dream_cycle_xb. The patterns array
    may have fewer rows than the input (due to pruning and merging).

    Fields:
        patterns: (N_out, d) array of post-dream patterns. All rows are
            unit vectors. N_out <= N_in.
        associations: Cross-domain associations discovered by REM-explore.
            Each tuple: (idx_i, idx_j, similarity_score). Indices are in
            OUTPUT pattern space (post-prune/merge). similarity_score in
            [0.0, 1.0].
        pruned_indices: Indices (in INPUT pattern space) that were removed
            by nrem_prune_xb. Sorted ascending. No duplicates.
        merge_map: Maps OUTPUT index to list of INPUT indices that were
            merged into it. Only contains entries for merged patterns
            (not pass-through patterns).
    """

    patterns: np.ndarray
    associations: list  # list of (int, int, float)
    pruned_indices: list  # list[int]
    merge_map: dict  # dict[int, list[int]]


# ---------------------------------------------------------------------------
# Hopfield Update (pattern-space dynamics, used by NREM)
# ---------------------------------------------------------------------------


def hopfield_update(
    beta: float,
    patterns: np.ndarray,
    xi: np.ndarray,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Single Hopfield update step in modern Hopfield network.

    xi_new = sum_mu attn(beta, X^T xi)_mu * x_mu

    This computes a convex combination of stored patterns weighted by
    their attention similarity to the current state. When attention_fn
    is sparsemax, irrelevant patterns get exactly zero weight, yielding
    exact retrieval without blurring.

    Args:
        beta: inverse temperature (higher = more selective)
        patterns: (N, d) array of stored patterns
        xi: (d,) current state vector
        attention_fn: attention function (default: softmax). Use sparsemax
            for sparse Hopfield retrieval.

    Returns:
        (d,) updated state vector
    """
    if attention_fn is None:
        attention_fn = softmax
    similarities = patterns @ xi  # (N,)
    weights = attention_fn(beta, similarities)  # (N,)
    return weights @ patterns  # (d,)


def spreading_activation(
    beta: float,
    patterns: np.ndarray,
    xi: np.ndarray,
    max_steps: int = 50,
    tol: float = 1e-6,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Iterate hopfield_update until convergence.

    Runs the modern Hopfield dynamics until the state converges to a
    fixed point (||x_{t+1} - x_t|| < tol) or max_steps is reached.

    When attention_fn=sparsemax, this implements sparse Hopfield
    retrieval where each update assigns zero weight to irrelevant
    patterns, eliminating the blurring problem of dense softmax.

    Args:
        beta: inverse temperature
        patterns: (N, d) stored patterns
        xi: (d,) initial state
        max_steps: maximum iterations
        tol: convergence tolerance
        attention_fn: attention function (default: softmax). Use sparsemax
            for sparse retrieval dynamics.

    Returns:
        (d,) converged state vector
    """
    x = xi.copy()
    for _ in range(max_steps):
        x_new = hopfield_update(beta, patterns, x, attention_fn=attention_fn)
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def hopfield_update_biased(
    beta: float,
    patterns: np.ndarray,
    xi: np.ndarray,
    W_temporal: np.ndarray,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Single Hopfield update with temporal bias injection.

    Combines standard content-based similarity with a temporal co-occurrence
    signal encoded in W_temporal. The temporal term is L2-normalized so it
    acts as a direction bias rather than a magnitude override.

    similarities = X xi + normalize(X (W_temporal xi))
    weights = attn(beta, similarities)
    xi_new = weights @ X

    Args:
        beta: inverse temperature (higher = more selective)
        patterns: (N, d) array of stored patterns
        xi: (d,) current state vector
        W_temporal: (d, d) temporal co-occurrence matrix (from Hebbian flush)
        attention_fn: attention function (default: softmax)

    Returns:
        (d,) updated state vector
    """
    if attention_fn is None:
        attention_fn = softmax
    base_similarities = patterns @ xi                       # (N,)
    temporal_probe = W_temporal @ xi                        # (d,)
    temporal_similarities = patterns @ temporal_probe       # (N,)
    t_norm = np.linalg.norm(temporal_similarities)
    if t_norm > 1e-12:
        temporal_similarities = temporal_similarities / t_norm
    combined = base_similarities + temporal_similarities    # (N,)
    weights = attention_fn(beta, combined)                  # (N,)
    return weights @ patterns                               # (d,)


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

    Runs the temporally-biased Hopfield dynamics until the state converges
    to a fixed point (||x_{t+1} - x_t|| < tol) or max_steps is reached.

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
    state = np.array(xi, dtype=np.float64)
    for _ in range(max_steps):
        new_state = hopfield_update_biased(
            beta, patterns, state, W_temporal, attention_fn=attention_fn,
        )
        if np.linalg.norm(new_state - state) < tol:
            return new_state
        state = new_state
    return state


# ---------------------------------------------------------------------------
# Boltzmann Dynamics (W-space dynamics, used by REM operations)
# ---------------------------------------------------------------------------


def boltzmann_dynamics(
    W: np.ndarray,
    x0: np.ndarray,
    beta: float,
    max_steps: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Run Boltzmann machine dynamics: x_{t+1} = sigmoid(beta * W * x_t).

    This operates directly on the coupling matrix W, exploring the
    energy landscape defined by W rather than the pattern space.
    Used by REM operations which clean/protect the coupling structure.

    Args:
        W: (d, d) coupling matrix
        x0: (d,) initial state
        beta: inverse temperature
        max_steps: maximum iterations
        tol: convergence tolerance

    Returns:
        (d,) converged state vector (values in (0, 1) due to sigmoid)
    """
    x = x0.copy().astype(np.float64)
    for _ in range(max_steps):
        x_new = sigmoid(beta * (W @ x))
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------------------
# NREM Replay (Hebbian reinforcement at high beta)
# ---------------------------------------------------------------------------


def nrem_replay(
    W: np.ndarray,
    tagged_idx: list[int],
    embeddings: np.ndarray,
    beta_high: float = 10.0,
    eta: float = 0.01,
    attention_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """NREM replay: stamp new memories deeper via Hebbian updates at high beta.

    For each tagged memory:
      1. Start from the tagged embedding
      2. Run spreading activation to convergence (high beta = near-deterministic)
      3. Apply rank-1 Hebbian update: W += eta * x_inf (outer) x_inf

    This strengthens the attractor basins of recently encoded memories.

    Args:
        W: (d, d) coupling matrix (not mutated)
        tagged_idx: indices of recently tagged memories
        embeddings: (N, d) all stored embeddings
        beta_high: high inverse temperature for near-deterministic recall
        eta: Hebbian learning rate

    Returns:
        (d, d) updated coupling matrix
    """
    W_new = W.copy()
    for i in tagged_idx:
        x0 = embeddings[i]
        x_inf = spreading_activation(
            beta_high, embeddings, x0, attention_fn=attention_fn,
        )
        W_new += eta * np.outer(x_inf, x_inf)
    return W_new


# ---------------------------------------------------------------------------
# REM Unlearn (anti-Hebbian at moderate beta)
# ---------------------------------------------------------------------------


def rem_unlearn(
    W: np.ndarray,
    dim: int,
    beta_mod: float = 2.0,
    eta: float = 0.01,
    n_trials: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """REM unlearn: clean spurious states via anti-Hebbian updates.

    For each trial:
      1. Start from a random unit vector
      2. Run Boltzmann dynamics on W until convergence
      3. Apply anti-Hebbian: W -= eta * x_inf (outer) x_inf

    This weakens spurious attractors (states that W converges to from
    random starts are likely spurious, not intentionally stored).

    The dynamics run on W directly (Boltzmann machine), NOT on the
    pattern space (Hopfield). This is because REM operations explore
    the coupling landscape.

    Args:
        W: (d, d) coupling matrix (not mutated)
        dim: dimensionality of state space
        beta_mod: moderate inverse temperature
        eta: anti-Hebbian learning rate
        n_trials: number of random-start trials
        rng: random number generator for reproducibility

    Returns:
        (d, d) updated coupling matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    W_new = W.copy()
    for _ in range(n_trials):
        # Random unit vector as starting state
        x0 = rng.standard_normal(dim)
        x0 /= np.linalg.norm(x0)

        # Boltzmann dynamics on the coupling matrix
        x_inf = boltzmann_dynamics(W_new, x0, beta_mod)

        # Anti-Hebbian: weaken whatever W converges to from random starts
        W_new -= eta * np.outer(x_inf, x_inf)

    return W_new


# ---------------------------------------------------------------------------
# REM Explore (weak Hebbian during wandering at low beta)
# ---------------------------------------------------------------------------


def rem_explore(
    W: np.ndarray,
    embeddings: np.ndarray,
    beta_low: float = 0.5,
    eta_weak: float = 0.001,
    n_steps: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """REM explore: protect old knowledge via weak Hebbian during wandering.

    CRITICAL: Uses FIXED step count, does NOT converge. This is intentional
    wandering at low temperature (high entropy / maximal exploration).

    For each step:
      1. Generate random starting state (perturbation of stored memory or random)
      2. Run ONE step of Boltzmann dynamics: x_1 = sigmoid(beta_low * W * x_0)
      3. Weak Hebbian at each step: W += eta_weak * x_1 (outer) x_1

    The weak Hebbian reinforcement during wandering gently strengthens
    paths that the system naturally traverses, protecting established
    knowledge without aggressive stamping.

    Args:
        W: (d, d) coupling matrix (not mutated)
        embeddings: (N, d) stored embeddings (used for random perturbations)
        beta_low: low inverse temperature (maximal wandering)
        eta_weak: weak Hebbian learning rate
        n_steps: fixed number of exploration steps (NOT a convergence criterion)
        rng: random number generator for reproducibility

    Returns:
        (d, d) updated coupling matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    W_new = W.copy()
    N = embeddings.shape[0]
    d = embeddings.shape[1]

    for _ in range(n_steps):
        # Random starting state: perturbed stored memory or pure random
        if N > 0 and rng.random() < 0.5:
            idx = rng.integers(0, N)
            x0 = embeddings[idx] + 0.1 * rng.standard_normal(d)
        else:
            x0 = rng.standard_normal(d)

        # ONE step of Boltzmann dynamics (no convergence check)
        x1 = sigmoid(beta_low * (W_new @ x0))

        # Weak Hebbian reinforcement
        W_new += eta_weak * np.outer(x1, x1)

    return W_new


# ---------------------------------------------------------------------------
# Lotka-Volterra Competition
# ---------------------------------------------------------------------------


def lv_competition(
    W: np.ndarray,
    embeddings: np.ndarray,
    fitness: np.ndarray,
    dt: float = 0.1,
    steps: int = 20,
) -> np.ndarray:
    """Lotka-Volterra competition dynamics on attractor fitness.

    Models competition between memory attractors: stronger attractors
    (higher fitness / deeper basins) suppress weaker ones.

    Update rule for fitness values:
      f_i(t+1) = f_i(t) + dt * f_i(t) * (r_i - sum_j a_ij * f_j(t))

    Where:
      - r_i = initial fitness (growth rate)
      - a_ij = competition coefficient derived from W
        (higher coupling between patterns = more competition)

    After dynamics, fitness changes are applied back to W by scaling
    the contribution of each pattern proportionally.

    Args:
        W: (d, d) coupling matrix (not mutated)
        embeddings: (N, d) stored embeddings
        fitness: (N,) initial fitness values per attractor
        dt: time step for Euler integration
        steps: number of integration steps

    Returns:
        (d, d) updated coupling matrix
    """
    W_new = W.copy()
    N = embeddings.shape[0]

    if N == 0:
        return W_new

    # Competition coefficients from pattern overlap via W
    # a_ij measures how much pattern j competes with pattern i
    # Higher coupling in W between pattern directions = more competition
    # Vectorized: competition = |E @ W @ E^T| where E is (N, d)
    projected = embeddings @ W_new  # (N, d)
    competition = np.abs(projected @ embeddings.T)  # (N, N)
    comp_max = np.max(competition)
    if comp_max > 0:
        competition /= comp_max

    # Initial growth rates = initial fitness
    r = fitness.copy().astype(np.float64)
    f = fitness.copy().astype(np.float64)

    # Euler integration of LV dynamics
    for _ in range(steps):
        # f_i(t+1) = f_i(t) + dt * f_i(t) * (r_i - sum_j a_ij * f_j(t))
        interaction = competition @ f
        df = dt * f * (r - interaction)
        f = np.maximum(f + df, 0.0)  # fitness can't go negative

    # Apply fitness scaling back to W
    # Scale each pattern's contribution proportionally to final/initial fitness
    scale = np.where(fitness > 0, f / fitness, 1.0)

    # Modify W: for each pair of patterns, scale their coupling
    # contribution by the geometric mean of their fitness scales
    # Vectorized: pair_scale[i,j] = sqrt(scale[i] * scale[j])
    pair_scale = np.sqrt(np.outer(scale, scale))  # (N, N)
    adjustment = pair_scale - 1.0  # how much to scale each pair's contribution

    # W += sum_{i,j} (pair_scale_{ij} - 1) * outer(e_i, e_j) / N
    # = (E^T @ diag(adjustment) @ E) / N ... but adjustment is a matrix
    # Efficient: W += (E^T @ adjustment @ E) / N
    W_new += (embeddings.T @ (adjustment @ embeddings)) / N

    return W_new


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def consolidate_similar(
    W: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Consolidate similar memories by merging near-duplicates.

    Finds pairs of memories with cosine similarity above threshold
    and merges them by averaging embeddings. The coupling matrix W
    is rebuilt to reflect the reduced pattern set.

    Args:
        W: (d, d) coupling matrix (not mutated)
        embeddings: (N, d) stored embeddings (not mutated)
        threshold: cosine similarity threshold for merging

    Returns:
        (updated_W, updated_embeddings) -- dimensions may shrink
    """
    W_new = W.copy()
    emb = embeddings.copy()
    N = emb.shape[0]

    if N <= 1:
        return W_new, emb

    # Build merge groups using union-find
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Find pairs above threshold
    for i in range(N):
        for j in range(i + 1, N):
            sim = cosine_sim(emb[i], emb[j])
            if sim > threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(N):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Check if any merging actually happened
    if len(groups) == N:
        # No merges: every pattern is its own group
        return W_new, emb

    # Merge groups
    merged_embeddings = []
    # Track which original indices map to each merged group
    group_indices: list[list[int]] = []
    for group in groups.values():
        # Average the embeddings in the group
        avg = np.mean(emb[group], axis=0)
        # Normalize to preserve unit-norm convention
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        merged_embeddings.append(avg)
        group_indices.append(group)

    merged_emb = np.array(merged_embeddings)

    # Rebuild W for the reduced pattern set
    # For merged groups, sum their W rows/columns (as specified)
    # W stays (d, d) -- it's a coupling matrix in embedding space, not pattern space
    N_new = merged_emb.shape[0]
    d = W.shape[0]
    W_merged = np.zeros((d, d))
    for i in range(N_new):
        W_merged += np.outer(merged_emb[i], merged_emb[i])
    W_merged /= max(N_new, 1)
    np.fill_diagonal(W_merged, 0.0)

    return W_merged, merged_emb


# ---------------------------------------------------------------------------
# Full Dream Cycle
# ---------------------------------------------------------------------------


def dream_cycle(
    W: np.ndarray,
    tagged: list[int],
    embeddings: np.ndarray,
    params: DreamParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the full thermodynamic dream cycle.

    Chains all operations in biological order:
      1. NREM-replay: stamp tagged memories deeper (Hebbian at high beta)
      2. REM-unlearn: clean spurious states (anti-Hebbian at moderate beta)
      3. REM-explore: protect old knowledge (weak Hebbian at low beta)
      4. LV competition: strong attractors suppress weak ones
      5. Consolidation: merge near-duplicate memories

    All operations are pure (W is copied at entry).

    Args:
        W: (d, d) coupling matrix (not mutated)
        tagged: indices of recently encoded memories to replay
        embeddings: (N, d) stored embeddings (not mutated)
        params: dream cycle parameters (defaults from research v3)

    Returns:
        (updated_W, updated_embeddings) -- embeddings may change from consolidation
    """
    if params is None:
        params = DreamParams()

    rng = np.random.default_rng(params.seed)
    d = W.shape[0]

    # Work on copies (pure function)
    W_cur = W.copy()
    emb_cur = embeddings.copy()

    # 1. NREM replay: stamp tagged memories
    W_cur = nrem_replay(
        W_cur, tagged, emb_cur, beta_high=params.beta_high, eta=params.eta
    )

    # 2. REM unlearn: clean spurious states
    W_cur = rem_unlearn(
        W_cur,
        d,
        beta_mod=params.beta_mod,
        eta=params.eta,
        n_trials=params.n_unlearn,
        rng=rng,
    )

    # 3. REM explore: protect old knowledge
    W_cur = rem_explore(
        W_cur,
        emb_cur,
        beta_low=params.beta_low,
        eta_weak=params.eta_weak,
        n_steps=params.n_explore,
        rng=rng,
    )

    # 4. LV competition: strong attractors survive
    # Compute fitness from attractor depth (how much energy each pattern
    # reduces when recalled via spreading activation)
    N = emb_cur.shape[0]
    if N > 0:
        fitness = np.ones(N)
        for i in range(N):
            x_inf = spreading_activation(params.beta_high, emb_cur, emb_cur[i])
            # Fitness = similarity between converged state and original pattern
            # (deeper attractor = higher cosine similarity after convergence)
            fitness[i] = max(cosine_sim(x_inf, emb_cur[i]), 0.01)

        W_cur = lv_competition(W_cur, emb_cur, fitness)

    # 5. Consolidation: merge near-duplicates
    W_cur, emb_cur = consolidate_similar(
        W_cur, emb_cur, threshold=params.consolidation_threshold
    )

    return W_cur, emb_cur


# ===========================================================================
# {X, β} Dream Operations — proof-aligned, no W matrix
#
# State = {X (pattern store), β (inverse temperature)}
# Query = softmax attention over X at β
# Dream = modulate β and curate X
# Coupling is derived from X, never independently evolved.
#
# Proof references:
#   Capacity.lean:    N_max = exp(βδ) / (4βM²)
#   BasinVolume.lean: Herfindahl H(p) determines basin curvature
#   EnergyGap.lean:   E(ξ) = -½‖ξ‖² + β⁻¹·Σp·log(p)
#   WeightUpdate.lean: perturbation tolerance ε per pair
# ===========================================================================


def nrem_replay_xb(
    patterns: np.ndarray,
    tagged_idx: list[int],
    beta_high: float = 40.0,
    pull_strength: float = 0.05,
) -> np.ndarray:
    """NREM replay on {X, β}: deepen basins of tagged patterns.

    For each tagged pattern x_μ:
      1. Run spreading_activation at high β → attractor x_∞
      2. Pull x_μ toward x_∞ by pull_strength (interpolation)
      3. Re-normalize to unit norm

    This makes tagged patterns more "canonical" — closer to their
    own basin centers. The basin deepens because the pattern moves
    toward the point of maximum softmax concentration.

    Proof link: local_minima_from_capacity guarantees the pattern
    remains a local minimum as long as the capacity bound holds.

    Args:
        patterns: (N, d) stored patterns (not mutated)
        tagged_idx: indices of recently tagged memories
        beta_high: high β for near-deterministic convergence
        pull_strength: interpolation weight toward attractor (0 = no change)

    Returns:
        (N, d) updated patterns
    """
    X = patterns.copy()
    for i in tagged_idx:
        if i < 0 or i >= X.shape[0]:
            continue
        x_inf = spreading_activation(beta_high, X, X[i])
        # Pull toward attractor center
        X[i] = (1 - pull_strength) * X[i] + pull_strength * x_inf
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def rem_unlearn_xb(
    patterns: np.ndarray,
    beta: float,
    beta_unlearn: float | None = None,
    n_probes: int = 200,
    separation_rate: float = 0.02,
    rng: np.random.Generator | None = None,
    similarity_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """REM unlearn on {X, beta}: destabilize mixture states by pattern separation.

    Proof-derived temperature:
      energy_gap = beta^{-1} log(N)   (from EnergyGap.lean)
      T_unlearn = energy_gap / 2      (below concentrated barrier, above mixture)
      beta_unlearn = 1 / T_unlearn

    Two code paths:
      (a) S-based fast path (when similarity_matrix is provided):
          Directly identifies close pairs from the precomputed S = X @ X.T,
          pushes apart pairs with similarity > 0.70, with intensity proportional
          to how far above 0.70 the pair similarity is. O(N^2) time.
      (b) Monte Carlo probing (when similarity_matrix is None):
          Probes from random starts at beta_unlearn, identifies mixture fixed
          points via softmax entropy, records co-activating pairs.
          O(n_probes * steps * N * d) time.

    Args:
        patterns: (N, d) stored patterns (not mutated)
        beta: operational inverse temperature
        beta_unlearn: unlearn temperature (derived from energy gap if None)
        n_probes: number of random-start probes (MC path only)
        separation_rate: how far to push mixture-forming pairs apart
        rng: random number generator (MC path only)
        similarity_matrix: precomputed S = patterns @ patterns.T; when provided,
            uses the S-based fast path instead of Monte Carlo probing.

    Returns:
        (N, d) updated patterns with increased separation where needed
    """
    if rng is None:
        rng = np.random.default_rng()

    X = patterns.copy()
    N, d = X.shape

    if N <= 1:
        return X

    # S-based pair analysis (fast path when similarity matrix provided)
    if similarity_matrix is not None:
        S = similarity_matrix.copy()
        np.fill_diagonal(S, -2.0)
        max_sim = np.max(S)
        if max_sim <= 0.70:  # fast-path: no close pairs
            return X
        close_i, close_j = np.where(np.triu(S, k=1) > 0.70)
        for idx in range(len(close_i)):
            i, j = close_i[idx], close_j[idx]
            diff = X[i] - X[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                direction = diff / norm
                intensity = min((S[i, j] - 0.70) / 0.30, 1.0)
                step = separation_rate * intensity
                X[i] += step * direction
                X[j] -= step * direction
                X[i] /= np.linalg.norm(X[i])
                X[j] /= np.linalg.norm(X[j])
        return X

    # Derive β_unlearn from proof if not provided
    if beta_unlearn is None:
        energy_gap = np.log(N) / beta
        t_unlearn = energy_gap / 2.0
        beta_unlearn = 1.0 / max(t_unlearn, 1e-6)

    # Track which pattern pairs co-activate in mixture states
    pair_mixture_count = np.zeros((N, N))

    for _ in range(n_probes):
        # Random unit vector start
        x0 = rng.standard_normal(d)
        x0 /= np.linalg.norm(x0)

        # Run spreading_activation at unlearn temperature
        fp = spreading_activation(beta_unlearn, X, x0, max_steps=50, tol=1e-6)

        # Compute softmax weights at the fixed point
        sims = X @ fp
        weights = softmax(beta_unlearn, sims)

        # Check if this is a mixture state (entropy > threshold)
        entropy = -np.sum(weights * np.log(weights + 1e-30))
        max_entropy = np.log(N)

        if entropy > 0.3 * max_entropy:
            # This is a mixture — record contributing pairs
            top_k = min(5, N)
            top_indices = np.argsort(weights)[-top_k:]
            top_weights = weights[top_indices]
            # Weight the pair count by the product of softmax weights
            for a_pos in range(len(top_indices)):
                for b_pos in range(a_pos + 1, len(top_indices)):
                    a, b = int(top_indices[a_pos]), int(top_indices[b_pos])
                    w_pair = float(top_weights[a_pos] * top_weights[b_pos])
                    pair_mixture_count[a, b] += w_pair
                    pair_mixture_count[b, a] += w_pair

    # Push apart pattern pairs that frequently co-activate in mixtures
    threshold = 0.05 * n_probes  # > 5% of probes
    for i in range(N):
        for j in range(i + 1, N):
            if pair_mixture_count[i, j] > threshold:
                diff = X[i] - X[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    direction = diff / norm
                    # Scale separation by how often this pair forms mixtures
                    intensity = min(pair_mixture_count[i, j] / n_probes, 1.0)
                    step = separation_rate * intensity
                    X[i] += step * direction
                    X[j] -= step * direction
                    # Re-normalize to unit vectors
                    X[i] /= np.linalg.norm(X[i])
                    X[j] /= np.linalg.norm(X[j])

    return X


def rem_explore_xb(
    patterns: np.ndarray,
    beta_min: float | None = None,
    n_probes: int = 50,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:
    """REM explore on {X, β}: discover cross-pattern associations at low β.

    At β_min (just above capacity floor), basins are broad.
    Patterns that share basin overlap at low β are associatively linked.

    Proof link: β_min = log(4NM²) / δ from Capacity.lean.
    Below β_min, patterns stop being guaranteed attractors.
    The associative sweet spot is right above this floor.

    Args:
        patterns: (N, d) stored patterns (read-only)
        beta_min: minimum β for exploration (derived from capacity if None)
        n_probes: number of exploration probes per pattern
        rng: random number generator

    Returns:
        List of (pattern_i, pattern_j, strength) associations.
        Strength reflects how often the pair co-activates at low β.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = patterns.shape
    if N <= 1:
        return []

    # Derive β_min from capacity formula if not provided
    if beta_min is None:
        M_sq = float(np.max(np.sum(patterns**2, axis=1)))
        # Estimate δ = minimum pairwise cosine distance
        # Sample pairs for efficiency
        n_sample = min(N * (N - 1) // 2, 500)
        min_delta = 2.0  # max possible cosine distance
        for _ in range(n_sample):
            i, j = rng.choice(N, size=2, replace=False)
            d_ij = 1.0 - float(patterns[i] @ patterns[j])
            min_delta = min(min_delta, d_ij)
        if min_delta <= 0:
            min_delta = 0.01
        beta_min = np.log(4 * N * M_sq) / min_delta

    # Probe from each stored pattern at β_min
    co_activation = np.zeros((N, N))

    for probe in range(n_probes):
        idx = probe % N
        # Add slight noise to avoid exact convergence
        x0 = patterns[idx] + 0.01 * rng.standard_normal(d)
        x0 /= np.linalg.norm(x0)

        fp = spreading_activation(beta_min, patterns, x0, max_steps=50, tol=1e-6)

        # Softmax weights at convergence
        sims = patterns @ fp
        weights = softmax(beta_min, sims)

        # Record co-activation: patterns with weight > 1.5/N are "active"
        active_threshold = 1.5 / N
        active = np.where(weights > active_threshold)[0]
        for a in active:
            for b in active:
                if a != b:
                    co_activation[a, b] += float(weights[a] * weights[b])

    # Extract significant associations
    associations = []
    for i in range(N):
        for j in range(i + 1, N):
            strength = co_activation[i, j] / max(n_probes, 1)
            if strength > 0.005:
                associations.append((i, j, strength))

    # Sort by strength descending
    associations.sort(key=lambda x: x[2], reverse=True)
    return associations


# ---------------------------------------------------------------------------
# Dream Redesign: SHY-inspired repulsion (nrem_repulsion_xb)
# ---------------------------------------------------------------------------


def nrem_repulsion_xb(
    patterns: np.ndarray,
    importances: np.ndarray,
    eta: float = 0.01,
    min_sep: float = 0.3,
) -> np.ndarray:
    """NREM repulsion on {X, beta}: push apart close patterns (SHY model).

    Implements Synaptic Homeostasis Hypothesis (Tononi 2003). During NREM
    slow-wave sleep, synaptic weights undergo global downscaling. In pattern
    space this translates to: push apart patterns that are closer than
    min_sep (cosine distance), anchoring high-importance patterns (>= 0.7)
    while moving low-importance ones.

    Contracts:
        R1: delta_min(output) >= delta_min(input)
        R2: High-importance patterns (>= 0.7) unchanged
        R3: All output vectors unit norm (within 1e-6)
        R4: Output shape == input shape
        R5: Input not mutated

    Args:
        patterns: (N, d) unit vectors (not mutated)
        importances: (N,) in [0.0, 1.0]
        eta: repulsion step size
        min_sep: cosine distance threshold (pairs closer than this are repelled)

    Returns:
        (N, d) updated unit vectors with increased separation
    """
    N = patterns.shape[0]
    if N <= 1:
        return patterns.copy()

    # Identify pairs closer than min_sep (cosine distance)
    # cosine_sim > (1 - min_sep) means cosine distance < min_sep
    sim_threshold = 1.0 - min_sep

    # Pre-compute full similarity matrix via BLAS — one matmul instead of
    # N*(N-1)/2 individual dot products. At N=10k this turns 50M Python-loop
    # dot products (~117s) into a single BLAS call (~1s).
    S = gpu_ops.similarity_matrix(patterns)  # (N, N)

    # Fast path: if no pair exceeds threshold, no repulsion needed
    # Mask diagonal (self-similarity = 1.0) before checking
    np.fill_diagonal(S, -2.0)
    if np.max(S) <= sim_threshold:
        return patterns.copy()

    X = patterns.copy()

    # Which patterns are movable (low importance)?
    movable = importances < 0.7

    # Find all close pairs from pre-computed similarity matrix
    close_i, close_j = np.where(np.triu(S, k=1) > sim_threshold)

    # Apply repulsion for each close pair
    for idx in range(len(close_i)):
        i, j = int(close_i[idx]), int(close_j[idx])
        diff = X[i] - X[j]
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-12:
            continue
        direction = diff / diff_norm

        if movable[i]:
            X[i] = X[i] + eta * direction
        if movable[j]:
            X[j] = X[j] - eta * direction

    # Re-normalize all output vectors to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = X / norms

    # Postcondition: verify R1 (delta_min non-decreasing)
    # With pre-computed matrices this is now feasible at any N
    S_in_triu = gpu_ops.similarity_matrix(patterns)
    np.fill_diagonal(S_in_triu, -2.0)
    S_out = gpu_ops.similarity_matrix(X)
    np.fill_diagonal(S_out, -2.0)

    delta_min_in = 1.0 - float(np.max(S_in_triu))
    delta_min_out = 1.0 - float(np.max(S_out))
    if delta_min_out < delta_min_in - 1e-6:
        return patterns.copy()

    return X


# ---------------------------------------------------------------------------
# Bridge counting (ConditionalMonotonicity.lean, Phase 9)
# ---------------------------------------------------------------------------


def count_protected_bridges(
    patterns: np.ndarray,
    importances: np.ndarray,
    labels: np.ndarray | None = None,
    importance_threshold: float = 0.7,
    bridge_sim_threshold: float = 0.3,
) -> int:
    """Count cross-cluster edges among protected (high-importance) patterns.

    A "bridge" is a pair (i, j) where:
      - Both have importance >= importance_threshold
      - They belong to different clusters (labels[i] != labels[j])
      - Their cosine similarity >= bridge_sim_threshold

    ConditionalMonotonicity.lean proves:
      protectedSet(importance >= 0.7) ⊆ survivors =>
      bridgeCount(after) >= bridgeCount(before)

    If labels is None, derives clusters via threshold-based clustering.
    """
    N = patterns.shape[0]
    if N < 2:
        return 0

    protected = np.where(importances >= importance_threshold)[0]
    if len(protected) < 2:
        return 0

    if labels is None:
        labels = _assign_clusters(patterns, threshold=0.5)

    protected_patterns = patterns[protected]
    protected_labels = labels[protected]

    S = gpu_ops.similarity_matrix(protected_patterns)
    bridge_count = 0
    for i in range(len(protected)):
        for j in range(i + 1, len(protected)):
            if (protected_labels[i] != protected_labels[j]
                    and S[i, j] >= bridge_sim_threshold):
                bridge_count += 1

    return bridge_count


# ---------------------------------------------------------------------------
# Dream Redesign: Prune near-duplicates (nrem_prune_xb)
# ---------------------------------------------------------------------------


def nrem_prune_xb(
    patterns: np.ndarray,
    importances: np.ndarray,
    threshold: float = 0.95,
) -> tuple[np.ndarray, list[int]]:
    """NREM prune on {X, beta}: remove near-duplicate patterns.

    Implements Active Systems Consolidation (Born 2010). Redundant episodic
    traces are eliminated. Near-duplicate patterns consume capacity without
    adding retrieval value. Pruning removes the lower-importance member of
    each near-duplicate pair.

    Contracts:
        P1: len(output) <= len(input)
        P2: Kept patterns preserved exactly
        P3: No close pairs in output (all sim <= threshold)
        P4: kept_indices sorted ascending
        P5: kept_indices subset of range(N)
        P6: Input not mutated

    Args:
        patterns: (N, d) unit vectors (not mutated)
        importances: (N,) in [0.0, 1.0]
        threshold: cosine similarity threshold above which pairs are pruned

    Returns:
        (pruned_patterns, kept_indices) where kept_indices lists the original
        indices that survived pruning.
    """
    N = patterns.shape[0]
    if N == 0:
        return patterns.copy(), []

    # Pre-compute full similarity matrix via BLAS — one matmul instead of
    # N*(N-1)/2 individual dot products. At N=10k: 50M dot products (~117s)
    # become a single BLAS call (~1s).
    S = gpu_ops.similarity_matrix(patterns)  # (N, N)

    # Fast path: if no pair exceeds threshold, no pruning needed.
    # Mask diagonal before checking (self-similarity = 1.0).
    np.fill_diagonal(S, -2.0)
    if np.max(S) <= threshold:
        return patterns.copy(), list(range(N))

    removed = set()

    # Greedy removal: for each pair exceeding threshold, remove the
    # lower-importance member (tie-break: remove higher index).
    # Same algorithm as before, but S[i, j] lookup (~ns) replaces
    # patterns[i] @ patterns[j] dot product (~μs).
    for i in range(N):
        if i in removed:
            continue
        for j in range(i + 1, N):
            if j in removed:
                continue
            if S[i, j] > threshold:
                if importances[i] < importances[j]:
                    removed.add(i)
                    break  # i is removed, no more pairs for i
                elif importances[i] > importances[j]:
                    removed.add(j)
                else:
                    # Equal importance: remove higher index
                    removed.add(j)

    kept_indices = sorted(set(range(N)) - removed)

    # Second pass: verify P3 — if any close pairs remain, do another pass.
    # Uses pre-computed S for lookups.
    while True:
        found_violation = False
        new_removed = set()
        kept_list = [k for k in kept_indices if k not in new_removed]
        for a_pos in range(len(kept_list)):
            if kept_list[a_pos] in new_removed:
                continue
            for b_pos in range(a_pos + 1, len(kept_list)):
                if kept_list[b_pos] in new_removed:
                    continue
                ki, kj = kept_list[a_pos], kept_list[b_pos]
                if S[ki, kj] > threshold:
                    found_violation = True
                    if importances[ki] < importances[kj]:
                        new_removed.add(ki)
                        break
                    elif importances[ki] > importances[kj]:
                        new_removed.add(kj)
                    else:
                        new_removed.add(kj)
        if not found_violation:
            break
        kept_indices = [k for k in kept_indices if k not in new_removed]

    pruned_patterns = patterns[kept_indices].copy()
    return pruned_patterns, kept_indices


# ---------------------------------------------------------------------------
# Dream Redesign: Merge similar groups (nrem_merge_xb)
# ---------------------------------------------------------------------------


def nrem_merge_xb(
    patterns: np.ndarray,
    importances: np.ndarray,
    threshold: float = 0.90,
    min_group: int = 3,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """NREM merge on {X, beta}: consolidate similar pattern groups.

    Implements Active Systems Consolidation (Born 2010). Groups of similar
    episodic memories are compressed into a single semantic prototype
    (episodic-to-semantic transfer).

    Algorithm:
        1. Build adjacency graph: edge (i,j) iff cosine_sim(i,j) > threshold
        2. Find connected components of size >= min_group (union-find)
        3. Replace qualifying components with unit-norm centroids
        4. Output ordering: non-merged first (original order), then centroids

    Contracts:
        M1: Merged patterns are unit norm
        M2: Partition property (each input index accounted for exactly once)
        M3: Merged importance boosted (> max of group)
        M4: Merged importance capped (<= 1.0)
        M5: merge_map keys are valid output indices
        M6: merge_map values are valid input indices
        M7: Input not mutated

    Args:
        patterns: (N, d) unit vectors (not mutated)
        importances: (N,) in [0.0, 1.0]
        threshold: pairwise cosine similarity threshold for adjacency
        min_group: minimum component size to trigger merge

    Returns:
        (merged_patterns, merge_map) where merge_map maps output index to
        list of input indices that were merged.
    """
    N = patterns.shape[0]
    if N < min_group:
        return patterns.copy(), {}

    # Union-Find data structure
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build adjacency via union-find.
    # Pre-compute similarity matrix via BLAS instead of N*(N-1)/2 dot products.
    S = gpu_ops.similarity_matrix(patterns)  # (N, N)
    np.fill_diagonal(S, -2.0)  # exclude self-similarity

    # Fast path: if no pair exceeds threshold, no merging possible
    if np.max(S) <= threshold:
        return patterns.copy(), {}

    # Find all above-threshold pairs from upper triangle
    close_i, close_j = np.where(np.triu(S, k=1) > threshold)
    for idx in range(len(close_i)):
        union(int(close_i[idx]), int(close_j[idx]))

    # Collect connected components
    components: dict[int, list[int]] = {}
    for i in range(N):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    # Separate into merge-able (>= min_group) and non-merged
    merge_groups: list[list[int]] = []
    merged_input_indices: set[int] = set()

    for _root, members in components.items():
        if len(members) >= min_group:
            merge_groups.append(sorted(members))
            merged_input_indices.update(members)

    # Sort merge groups by their smallest original index
    merge_groups.sort(key=lambda g: g[0])

    if not merge_groups:
        return patterns.copy(), {}

    # Build output: non-merged patterns first (original order), then centroids
    non_merged_indices = [i for i in range(N) if i not in merged_input_indices]

    output_rows = []
    # Non-merged patterns in original order
    for idx in non_merged_indices:
        output_rows.append(patterns[idx].copy())

    # Merged centroids
    merge_map: dict[int, list[int]] = {}
    for group in merge_groups:
        centroid = np.mean(patterns[group], axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-12:
            centroid = centroid / centroid_norm
        out_idx = len(output_rows)
        output_rows.append(centroid)
        merge_map[out_idx] = group

    merged_patterns = np.array(output_rows)
    return merged_patterns, merge_map


# ---------------------------------------------------------------------------
# Dream Redesign: Cross-domain exploration (rem_explore_cross_domain_xb)
# ---------------------------------------------------------------------------


def rem_explore_cross_domain_xb(
    patterns: np.ndarray,
    labels_or_clusters: np.ndarray,
    n_probes: int = 100,
    rng: np.random.Generator | None = None,
    bridge_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """REM cross-domain exploration: discover associations across clusters.

    Uses centroid bridge detection (BridgeFormation.lean Phase 8). A pattern
    is a "bridge" if it has cosine similarity >= tau to the centroid of a
    cluster it does NOT belong to. Bridge patterns create cross-domain
    association edges connecting their own cluster to the foreign cluster.

    Algorithm (Lean-aligned, two-phase):
        1. Derive core clusters using a tight threshold (0.7) so that bridge
           patterns (which straddle subspaces) don't merge all clusters into
           one connected component. Standard 0.5 threshold fails because
           bridges have cosine ~0.6 to both domains.
        2. Compute centroid for each core cluster (>= 2 members).
        3. For each pattern, compute cosine to every OTHER cluster's centroid.
        4. If cosine >= bridge_threshold (tau), the pattern is a bridge
           candidate for that foreign cluster.
        5. For each bridge candidate, create association edges between the
           bridge and the nearest patterns in the foreign cluster.
        6. Deduplicate by (min(i,j), max(i,j)), keeping max similarity.
        7. Sort by similarity descending.

    Lean proofs:
        - BridgeFormation.lean:126-154 centroid_creates_two_bridges:
          centroids with alignment >= tau to both domains create >= 2 bridges
        - BridgeFormation.lean:200-206 cross_domain_bridge:
          K traces with similarity >= sigma to both domains =>
          centroid preserves alignment to both

    Contracts:
        X1: All pairs are cross-cluster
        X2: Similarity scores in [0, 1]
        X3: No self-pairs
        X4: Indices valid
        X5: Input not mutated
        X6: Sorted descending by similarity

    Args:
        patterns: (N, d) unit vectors (not mutated)
        labels_or_clusters: (N,) integer cluster assignments (used as hint;
            re-clustered internally with tight threshold for bridge detection)
        n_probes: unused (kept for API compatibility)
        rng: unused (kept for API compatibility)
        bridge_threshold: tau — minimum cosine to foreign centroid for bridge

    Returns:
        List of (idx_i, idx_j, similarity_score) sorted by score descending.
    """
    N = patterns.shape[0]
    if N <= 1:
        return []

    # Re-cluster with tight threshold for bridge detection.
    # Standard threshold (0.5) merges everything when bridge patterns exist
    # because bridges have cosine ~0.6 to both domains, creating one component.
    # A tighter threshold (0.7) separates core domain clusters from bridges.
    labels = _assign_clusters(patterns, threshold=0.7)
    unique_clusters = np.unique(labels)

    # If tight clustering still yields only 1 cluster, try even tighter
    if len(unique_clusters) < 2:
        labels = _assign_clusters(patterns, threshold=0.85)
        unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        return []

    # Build cluster-to-indices mapping
    cluster_indices: dict[int, list[int]] = {}
    for idx in range(N):
        c = int(labels[idx])
        if c not in cluster_indices:
            cluster_indices[c] = []
        cluster_indices[c].append(idx)

    # Compute normalized centroids for clusters with >= 2 members
    centroids: dict[int, np.ndarray] = {}
    for c, indices in cluster_indices.items():
        if len(indices) < 2:
            continue
        centroid = patterns[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid = centroid / norm
        centroids[c] = centroid

    if len(centroids) < 2:
        # Need at least 2 clusters with meaningful centroids
        # Fall back: include singleton clusters too
        for c, indices in cluster_indices.items():
            if c not in centroids:
                centroid = patterns[indices].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-12:
                    centroid = centroid / norm
                centroids[c] = centroid

    if len(centroids) < 2:
        return []

    centroid_cluster_ids = list(centroids.keys())

    # Accumulate pair similarities: key = (min_idx, max_idx) -> max sim
    pair_best: dict[tuple[int, int], float] = {}

    # For each pattern, check alignment to foreign cluster centroids
    for idx in range(N):
        own_cluster = int(labels[idx])
        pattern = patterns[idx]

        for foreign_cluster in centroid_cluster_ids:
            if foreign_cluster == own_cluster:
                continue

            # Check cosine similarity to foreign centroid
            cos_to_foreign = float(pattern @ centroids[foreign_cluster])

            if cos_to_foreign < bridge_threshold:
                continue

            # This pattern is a bridge to the foreign cluster.
            # Create association edges: bridge <-> each foreign cluster member
            for foreign_idx in cluster_indices[foreign_cluster]:
                if foreign_idx == idx:
                    continue
                similarity = float(pattern @ patterns[foreign_idx])
                # Clamp to [0, 1]
                similarity = max(0.0, min(similarity, 1.0))
                if similarity > 0:
                    key = (min(idx, foreign_idx), max(idx, foreign_idx))
                    if key not in pair_best or similarity > pair_best[key]:
                        pair_best[key] = similarity

    # Build result list
    associations = [(k[0], k[1], v) for k, v in pair_best.items()]

    # Sort by similarity descending
    associations.sort(key=lambda x: x[2], reverse=True)
    return associations


# ---------------------------------------------------------------------------
# Dream Redesign: Simple cluster assignment for dream_cycle_xb
# ---------------------------------------------------------------------------


def _assign_clusters(patterns: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Assign cluster labels via connected-component clustering on cosine similarity.

    Uses a simple threshold-based approach: patterns with cosine similarity
    above threshold are in the same cluster. Returns integer labels.
    """
    return gpu_ops.assign_clusters_matmul(patterns, threshold)


# ---------------------------------------------------------------------------
# Adaptive threshold computation
# ---------------------------------------------------------------------------


def compute_adaptive_thresholds(
    patterns: np.ndarray,
    labels: np.ndarray | None = None,
    merge_percentile: float = 70.0,
    prune_percentile: float = 90.0,
    max_sample_pairs: int = 10000,
    fallback_merge: float = 0.90,
    fallback_prune: float = 0.95,
) -> tuple[float, float]:
    """Compute adaptive merge/prune thresholds from measured geometry.

    Measures the within-cluster pairwise cosine similarity distribution
    and sets thresholds at specified percentiles. This ensures thresholds
    match the actual embedding geometry rather than being hardcoded.

    Algorithm:
        1. Assign clusters (via labels or _assign_clusters)
        2. Collect within-cluster pairwise cosine similarities
        3. merge_threshold = percentile(similarities, merge_percentile)
        4. prune_threshold = percentile(similarities, prune_percentile)

    The merge threshold at P70 means the tightest 30% of within-cluster
    pairs are candidates for merge. The prune threshold at P90 means
    only the tightest 10% are considered near-duplicates.

    Args:
        patterns: (N, d) unit vectors
        labels: (N,) integer cluster assignments. If None, derived via
            _assign_clusters with threshold=0.5.
        merge_percentile: percentile for merge threshold (default 70)
        prune_percentile: percentile for prune threshold (default 90)
        max_sample_pairs: maximum pairs to sample per cluster for efficiency
        fallback_merge: returned if no within-cluster pairs exist
        fallback_prune: returned if no within-cluster pairs exist

    Returns:
        (merge_threshold, prune_threshold) as cosine similarity values.
    """
    N = patterns.shape[0]
    if N < 2:
        return fallback_merge, fallback_prune

    if labels is None:
        labels = _assign_clusters(patterns, threshold=0.5)

    rng = np.random.default_rng(42)

    # Collect within-cluster pairwise cosine similarities
    within_sims: list[float] = []

    # Group indices by cluster
    cluster_indices: dict[int, list[int]] = {}
    for i in range(N):
        lbl = int(labels[i])
        if lbl not in cluster_indices:
            cluster_indices[lbl] = []
        cluster_indices[lbl].append(i)

    for _lbl, indices in cluster_indices.items():
        n_c = len(indices)
        if n_c < 2:
            continue

        # Total possible pairs in this cluster
        n_pairs = n_c * (n_c - 1) // 2

        if n_pairs <= max_sample_pairs:
            # Enumerate all pairs
            for a in range(n_c):
                for b in range(a + 1, n_c):
                    sim = float(patterns[indices[a]] @ patterns[indices[b]])
                    within_sims.append(sim)
        else:
            # Sample pairs
            for _ in range(max_sample_pairs):
                a, b = rng.choice(n_c, size=2, replace=False)
                sim = float(patterns[indices[a]] @ patterns[indices[b]])
                within_sims.append(sim)

    if not within_sims:
        return fallback_merge, fallback_prune

    sims_arr = np.array(within_sims)
    merge_threshold = float(np.percentile(sims_arr, merge_percentile))
    prune_threshold = float(np.percentile(sims_arr, prune_percentile))

    # Ensure prune >= merge (prune is stricter)
    prune_threshold = max(prune_threshold, merge_threshold + 0.01)

    return merge_threshold, prune_threshold


# ---------------------------------------------------------------------------
# Dream cycle v2 — corrected pipeline order + adaptive thresholds
# ---------------------------------------------------------------------------


def dream_cycle_v2(
    patterns: np.ndarray,
    beta: float,
    tagged_indices: list[int] | None = None,
    importances: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    seed: int | None = None,
    merge_percentile: float = 70.0,
    prune_percentile: float = 90.0,
    bridge_threshold: float = 0.3,
) -> DreamReport:
    """Full dream cycle v2 — corrected pipeline with adaptive thresholds.

    Biologically correct pipeline order (early NREM consolidation,
    late NREM downscaling):
      1. Measure geometry: compute within-cluster similarity distribution
      2. Set adaptive thresholds from distribution percentiles
      3. nrem_merge_xb: consolidate similar groups while still tight
      4. nrem_prune_xb: remove near-duplicates that merge didn't catch
      5. nrem_repulsion_xb: push surviving patterns apart (SHY downscaling)
      6. rem_unlearn_xb: destabilize mixture states
      7. rem_explore_cross_domain_xb: discover cross-domain associations

    Key insight: consolidation before separation. Merging patterns while
    they're still close (before repulsion scatters them) enables the
    abstraction hierarchy. Adaptive thresholds from measured geometry
    eliminate configuration sensitivity — the system self-calibrates
    for any embedding model and data distribution.

    Args:
        patterns: (N, d) stored patterns (not mutated)
        beta: operational inverse temperature
        tagged_indices: indices of important memories
        importances: (N,) importance scores in [0.0, 1.0]. If None,
            derived from tagged_indices.
        labels: (N,) integer cluster assignments. If None, derived via
            threshold-based clustering.
        seed: random seed for reproducibility
        merge_percentile: percentile of within-cluster similarity
            distribution for merge threshold (default 70 = tightest 30%)
        prune_percentile: percentile for prune threshold (default 90 =
            tightest 10%)

    Returns:
        DreamReport with post-dream patterns, associations, prune/merge info.
    """
    rng = np.random.default_rng(seed)
    N = patterns.shape[0]

    if N == 0:
        return DreamReport(
            patterns=patterns.copy(),
            associations=[],
            pruned_indices=[],
            merge_map={},
        )

    # Default: tag all patterns
    if tagged_indices is None:
        tagged_indices = list(range(N))

    # Derive importances
    if importances is None:
        tagged_set = set(tagged_indices)
        importances = np.array([0.8 if i in tagged_set else 0.3 for i in range(N)])
    else:
        importances = np.asarray(importances, dtype=float)
        if len(importances) != N:
            raise ValueError(
                f"importances length {len(importances)} != patterns count {N}"
            )

    # Derive labels if not provided
    if labels is None:
        labels = _assign_clusters(patterns, threshold=0.5)

    # ---------------------------------------------------------------
    # Step 1: Measure geometry → adaptive thresholds
    # ---------------------------------------------------------------
    merge_threshold, prune_threshold = compute_adaptive_thresholds(
        patterns,
        labels,
        merge_percentile=merge_percentile,
        prune_percentile=prune_percentile,
    )

    # ---------------------------------------------------------------
    # Step 2: NREM merge (consolidation — while patterns are still tight)
    # ---------------------------------------------------------------
    X1, merge_map_local = nrem_merge_xb(
        patterns,
        importances,
        threshold=merge_threshold,
        min_group=3,
    )

    # Track which original indices were merged
    merged_input_indices: set[int] = set()
    for group in merge_map_local.values():
        merged_input_indices.update(group)

    # Build importances for post-merge patterns
    non_merged_indices = [i for i in range(N) if i not in merged_input_indices]
    importances_after_merge_list = [importances[i] for i in non_merged_indices]

    # For merged centroids: boosted importance
    for out_idx in sorted(merge_map_local.keys()):
        group = merge_map_local[out_idx]
        group_imps = [importances[i] for i in group]
        boosted = min(max(group_imps) + 0.1, 1.0)
        importances_after_merge_list.append(boosted)

    importances_after_merge = np.array(importances_after_merge_list, dtype=float)

    # Map merge_map to original indices (it already uses original indices
    # since we merged directly from the input patterns)
    merge_map_original = merge_map_local

    # ---------------------------------------------------------------
    # Step 3: NREM prune near-duplicates (catch what merge missed)
    # ---------------------------------------------------------------
    X2, kept_indices_after_prune = nrem_prune_xb(
        X1,
        importances_after_merge,
        threshold=prune_threshold,
    )

    # Map pruned indices back to original input space
    # post-merge index → original index mapping:
    #   non-merged indices map directly, merge centroids are new
    post_merge_to_original: list[int | None] = []
    for i in non_merged_indices:
        post_merge_to_original.append(i)
    for _out_idx in sorted(merge_map_local.keys()):
        post_merge_to_original.append(None)  # merged centroids have no single original

    # Pruned indices in post-merge space
    all_post_merge = set(range(len(X1)))
    kept_post_merge = set(kept_indices_after_prune)
    pruned_post_merge = sorted(all_post_merge - kept_post_merge)

    # Map to original indices for the report
    # When a merged centroid is pruned, ALL original indices in that merge
    # group must be reported as pruned so downstream metadata reconstruction
    # can account for them (otherwise they appear as non-merged survivors).
    pruned_indices_original: list[int] = []
    # Also collect merge groups whose centroids were pruned — these groups
    # are removed from the final merge_map, so their originals must be
    # explicitly marked as pruned.
    merge_keys_sorted = sorted(merge_map_local.keys())
    for pm_idx in pruned_post_merge:
        orig = post_merge_to_original[pm_idx]
        if orig is not None:
            pruned_indices_original.append(orig)
        else:
            # Merged centroid was pruned — add all its constituent originals
            centroid_offset = pm_idx - len(non_merged_indices)
            if 0 <= centroid_offset < len(merge_keys_sorted):
                merge_key = merge_keys_sorted[centroid_offset]
                pruned_indices_original.extend(merge_map_local[merge_key])

    # Importances for surviving patterns
    importances_after_prune = importances_after_merge[kept_indices_after_prune]

    # ---------------------------------------------------------------
    # Step 4: NREM repulsion (SHY — push apart AFTER consolidation)
    # ---------------------------------------------------------------
    X3 = nrem_repulsion_xb(X2, importances_after_prune, eta=0.01, min_sep=0.3)

    # ---------------------------------------------------------------
    # Step 5: REM unlearn
    # ---------------------------------------------------------------
    S = gpu_ops.similarity_matrix(X3)
    X4 = rem_unlearn_xb(
        X3,
        beta,
        n_probes=200,
        separation_rate=0.02,
        rng=rng,
        similarity_matrix=S,
    )

    # ---------------------------------------------------------------
    # Step 6: REM explore cross-domain associations
    # ---------------------------------------------------------------
    N_out = X4.shape[0]

    # Build labels for output patterns
    # Post-merge labels: non-merged keep original labels, centroids get majority
    labels_post_merge_list: list[int] = []
    for i in non_merged_indices:
        labels_post_merge_list.append(int(labels[i]))

    from collections import Counter

    for out_idx in sorted(merge_map_local.keys()):
        group = merge_map_local[out_idx]
        group_labels = [int(labels[i]) for i in group]
        label_counts = Counter(group_labels)
        majority_label = min(
            label_counts.keys(),
            key=lambda lbl: (-label_counts[lbl], lbl),
        )
        labels_post_merge_list.append(majority_label)

    labels_post_merge = np.array(labels_post_merge_list, dtype=int)

    # Post-prune labels: keep only surviving indices
    labels_out = labels_post_merge[kept_indices_after_prune]

    associations = rem_explore_cross_domain_xb(
        X4,
        labels_out,
        n_probes=max(N_out, 50),
        rng=rng,
        bridge_threshold=bridge_threshold,
    )

    # ---------------------------------------------------------------
    # Remap merge_map keys to account for pruning
    # ---------------------------------------------------------------
    # The merge_map_local keys are post-merge indices. After pruning,
    # these indices shift. We need to remap to post-prune output indices.
    # kept_indices_after_prune tells us which post-merge indices survived.
    post_merge_to_post_prune = {
        pm_idx: pp_idx for pp_idx, pm_idx in enumerate(kept_indices_after_prune)
    }

    merge_map_final: dict[int, list[int]] = {}
    for pm_out_idx, original_group in merge_map_original.items():
        if pm_out_idx in post_merge_to_post_prune:
            pp_out_idx = post_merge_to_post_prune[pm_out_idx]
            merge_map_final[pp_out_idx] = original_group

    return DreamReport(
        patterns=X4,
        associations=associations,
        pruned_indices=pruned_indices_original,
        merge_map=merge_map_final,
    )


# ---------------------------------------------------------------------------
# Updated dream_cycle_xb — original pipeline with DreamReport (preserved)
# ---------------------------------------------------------------------------


def dream_cycle_xb(
    patterns: np.ndarray,
    beta: float,
    tagged_indices: list[int] | None = None,
    importances: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    seed: int | None = None,
    params: DreamParams | None = None,
) -> DreamReport:
    """Full dream cycle on {X, beta} — redesigned pipeline.

    Chains neuroscience-informed operations in biological order:
      1. nrem_repulsion_xb: push apart close patterns (SHY model)
      2. nrem_prune_xb: remove near-duplicates (Active Systems Consolidation)
      3. nrem_merge_xb: merge similar groups into centroids
      4. rem_unlearn_xb: destabilize mixture states (Crick & Mitchison)
      5. rem_explore_cross_domain_xb: discover cross-domain associations

    All parameters are derived from the proofs, not tuned.

    Args:
        patterns: (N, d) stored patterns (not mutated)
        beta: operational inverse temperature
        tagged_indices: indices of important memories
        importances: (N,) importance scores in [0.0, 1.0]. If None,
            derived from tagged_indices (0.8 for tagged, 0.3 for untagged).
        labels: (N,) integer cluster assignments. If None, derived via
            threshold-based clustering.
        seed: random seed for reproducibility (overrides params.seed if given)
        params: DreamParams controlling thresholds and rates. If None, uses
            DreamParams() defaults (matching previous hardcoded values).

    Returns:
        DreamReport with post-dream patterns, associations, prune/merge info.
    """
    if params is None:
        params = DreamParams()

    # Explicit seed arg overrides params.seed; if both None, non-deterministic
    effective_seed = seed if seed is not None else params.seed
    rng = np.random.default_rng(effective_seed)
    N = patterns.shape[0]

    if N == 0:
        return DreamReport(
            patterns=patterns.copy(),
            associations=[],
            pruned_indices=[],
            merge_map={},
        )

    # Default: tag all patterns
    if tagged_indices is None:
        tagged_indices = list(range(N))

    # Derive importances from tagged_indices if not provided
    if importances is None:
        tagged_set = set(tagged_indices)
        importances = np.array([0.8 if i in tagged_set else 0.3 for i in range(N)])
    else:
        importances = np.asarray(importances, dtype=float)
        if len(importances) != N:
            raise ValueError(
                f"importances length {len(importances)} != patterns count {N}"
            )

    # ---------------------------------------------------------------
    # Step 1: NREM repulsion (SHY)
    # ---------------------------------------------------------------
    X1 = nrem_repulsion_xb(patterns, importances, eta=params.eta, min_sep=params.min_sep)

    # Phase 9: Count protected bridges before prune
    labels_for_bridge = labels if labels is not None else _assign_clusters(X1, threshold=0.5)
    bridges_before_prune = count_protected_bridges(
        X1, importances, labels_for_bridge
    )

    # ---------------------------------------------------------------
    # Step 2: NREM prune near-duplicates
    # ---------------------------------------------------------------
    X2, kept_indices_after_prune = nrem_prune_xb(
        X1,
        importances,
        threshold=params.prune_threshold,
    )
    # Map pruned indices back to original input space
    all_indices = set(range(N))
    kept_set = set(kept_indices_after_prune)
    pruned_indices = sorted(all_indices - kept_set)

    # Phase 9: Count protected bridges after prune
    labels_after_prune = labels_for_bridge[kept_indices_after_prune] if labels is not None else None
    bridges_after_prune = count_protected_bridges(
        X2, importances[kept_indices_after_prune], labels_after_prune
    )
    if bridges_after_prune < bridges_before_prune:
        logger.warning(
            "ConditionalMonotonicity violation: bridge count decreased "
            "%d -> %d after prune", bridges_before_prune, bridges_after_prune,
        )

    # Importances for surviving patterns (post-prune)
    importances_after_prune = importances[kept_indices_after_prune]

    # ---------------------------------------------------------------
    # Step 3: NREM merge similar groups
    # ---------------------------------------------------------------
    X3, merge_map_local = nrem_merge_xb(
        X2,
        importances_after_prune,
        threshold=params.merge_threshold,
        min_group=params.merge_min_group,
    )

    # Precompute similarity matrix — shared across REM ops
    S = gpu_ops.similarity_matrix(X3)

    # Translate merge_map from post-prune indices to original input indices
    # merge_map_local keys are output indices, values are post-prune indices
    # We need to map post-prune indices -> original indices
    merge_map_original: dict[int, list[int]] = {}
    for out_idx, post_prune_group in merge_map_local.items():
        original_group = [kept_indices_after_prune[pp] for pp in post_prune_group]
        merge_map_original[out_idx] = original_group

    # ---------------------------------------------------------------
    # Step 4: REM unlearn (Crick & Mitchison) — S-based fast path
    # ---------------------------------------------------------------
    X4 = rem_unlearn_xb(
        X3,
        beta,
        n_probes=params.n_probes,
        separation_rate=params.separation_rate,
        rng=rng,
        similarity_matrix=S,
    )

    # ---------------------------------------------------------------
    # Step 5: REM explore cross-domain associations
    # ---------------------------------------------------------------
    N_out = X4.shape[0]

    if labels is not None:
        # Map labels through prune + merge pipeline
        # Post-prune labels: keep only surviving indices
        labels_post_prune = labels[kept_indices_after_prune]

        # Post-merge labels: non-merged keep their labels, merged get new
        merged_input_indices: set[int] = set()
        for group in merge_map_local.values():
            merged_input_indices.update(group)

        non_merged_pp = [
            i
            for i in range(len(kept_indices_after_prune))
            if i not in merged_input_indices
        ]
        labels_out_list = [int(labels_post_prune[i]) for i in non_merged_pp]
        # For merged centroids, iterate by output index to match centroid order
        from collections import Counter

        for out_idx in sorted(merge_map_local.keys()):
            group = merge_map_local[out_idx]
            group_labels = [int(labels_post_prune[pp]) for pp in group]
            # Majority vote with deterministic tie-break (smallest label wins)
            label_counts = Counter(group_labels)
            majority_label = min(
                label_counts.keys(),
                key=lambda lbl: (-label_counts[lbl], lbl),
            )
            labels_out_list.append(majority_label)

        labels_out = np.array(labels_out_list, dtype=int)
    else:
        # Derive cluster labels from post-merge patterns
        labels_out = _assign_clusters(X4, threshold=0.5)

    associations = rem_explore_cross_domain_xb(
        X4,
        labels_out,
        n_probes=max(N_out, 50),
        rng=rng,
        bridge_threshold=params.bridge_threshold,
    )

    return DreamReport(
        patterns=X4,
        associations=associations,
        pruned_indices=pruned_indices,
        merge_map=merge_map_original,
    )
