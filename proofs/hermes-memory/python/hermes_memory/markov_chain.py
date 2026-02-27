"""Hermes Memory System — Stochastic Simulation and Spectral Analysis

Markov chain simulation for the 2-memory system, coupling constructions,
transition matrix discretization, and spectral gap computation.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la

from .core import soft_select, strength_update


def simulate_chain(
    alpha: float,
    beta: float,
    delta_t: float,
    S_max: float,
    T_temp: float,
    S0: tuple[float, float],
    n_steps: int,
    rng: Optional[random.Random] = None,
) -> list[tuple[float, float]]:
    """Simulate one trajectory of the stochastic 2-memory Markov chain."""
    if rng is None:
        rng = random.Random()

    trajectory: list[tuple[float, float]] = [S0]
    s1, s2 = S0
    decay = math.exp(-beta * delta_t)

    for _ in range(n_steps):
        # 1. Decay both strengths
        s1_d = s1 * decay
        s2_d = s2 * decay

        # 2. Soft selection: probability of picking memory 1
        q = soft_select(s1_d, s2_d, T_temp)

        # 3. Stochastic selection and update
        u = rng.random()
        if u < q:
            # Memory 1 selected
            s1_new = strength_update(alpha, s1_d, S_max)
            s2_new = s2_d
        else:
            # Memory 2 selected
            s1_new = s1_d
            s2_new = strength_update(alpha, s2_d, S_max)

        # 4. Clamp to [0, Smax]
        s1 = max(0.0, min(s1_new, S_max))
        s2 = max(0.0, min(s2_new, S_max))

        trajectory.append((s1, s2))

    return trajectory


def simulate_coupling(
    alpha: float,
    beta: float,
    delta_t: float,
    S_max: float,
    T_temp: float,
    n_steps: int,
    seed: int = 42,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], Optional[int]]:
    """Run two chains from opposite corners with shared randomness (coupling)."""
    rng = random.Random(seed)
    decay = math.exp(-beta * delta_t)

    # Chain A starts at origin, Chain B at (Smax, Smax)
    a1, a2 = 0.0, 0.0
    b1, b2 = S_max, S_max

    traj_a: list[tuple[float, float]] = [(a1, a2)]
    traj_b: list[tuple[float, float]] = [(b1, b2)]
    coupling_time: Optional[int] = None

    for step in range(1, n_steps + 1):
        # Shared random draw
        u = rng.random()

        # Advance chain A
        a1_d = a1 * decay
        a2_d = a2 * decay
        q_a = soft_select(a1_d, a2_d, T_temp)
        if u < q_a:
            a1_new = strength_update(alpha, a1_d, S_max)
            a2_new = a2_d
        else:
            a1_new = a1_d
            a2_new = strength_update(alpha, a2_d, S_max)
        a1 = max(0.0, min(a1_new, S_max))
        a2 = max(0.0, min(a2_new, S_max))

        # Advance chain B
        b1_d = b1 * decay
        b2_d = b2 * decay
        q_b = soft_select(b1_d, b2_d, T_temp)
        if u < q_b:
            b1_new = strength_update(alpha, b1_d, S_max)
            b2_new = b2_d
        else:
            b1_new = b1_d
            b2_new = strength_update(alpha, b2_d, S_max)
        b1 = max(0.0, min(b1_new, S_max))
        b2 = max(0.0, min(b2_new, S_max))

        traj_a.append((a1, a2))
        traj_b.append((b1, b2))

        # Check coupling
        if coupling_time is None:
            dist = max(abs(a1 - b1), abs(a2 - b2))
            if dist < 1e-3 * S_max:
                coupling_time = step

    return traj_a, traj_b, coupling_time


def build_transition_matrix(
    alpha: float,
    beta: float,
    delta_t: float,
    S_max: float,
    T_temp: float,
    n_grid: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize the 2-memory chain into an n_grid^2 x n_grid^2 transition matrix."""
    grid = np.linspace(0.0, S_max, n_grid)
    n_states = n_grid * n_grid
    P = np.zeros((n_states, n_states), dtype=np.float64)
    decay = math.exp(-beta * delta_t)

    for i1 in range(n_grid):
        for i2 in range(n_grid):
            s1 = grid[i1]
            s2 = grid[i2]
            row = i1 * n_grid + i2

            # Decay
            s1_d = s1 * decay
            s2_d = s2 * decay

            # Selection probability
            q = soft_select(s1_d, s2_d, T_temp)

            # Case 1: memory 1 selected (prob q)
            s1_sel = strength_update(alpha, s1_d, S_max)
            s2_sel = s2_d
            # Snap to nearest grid point
            j1_sel = int(np.argmin(np.abs(grid - s1_sel)))
            j2_sel = int(np.argmin(np.abs(grid - s2_sel)))
            col_sel = j1_sel * n_grid + j2_sel
            P[row, col_sel] += q

            # Case 2: memory 2 selected (prob 1-q)
            s1_not = s1_d
            s2_not = strength_update(alpha, s2_d, S_max)
            j1_not = int(np.argmin(np.abs(grid - s1_not)))
            j2_not = int(np.argmin(np.abs(grid - s2_not)))
            col_not = j1_not * n_grid + j2_not
            P[row, col_not] += 1.0 - q

    return P, grid


def spectral_analysis(P: NDArray[np.float64]) -> dict:
    """Compute spectral properties of transition matrix P."""
    # Eigenvalues of P^T (left eigenvectors of P become right eigenvectors of P^T)
    eigenvalues = la.eigvals(P.T)

    # Sort by magnitude descending
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]

    # Second largest eigenvalue magnitude
    lambda_2 = float(np.abs(eigenvalues[1]))
    spectral_gap = 1.0 - lambda_2

    # Stationary distribution: left eigenvector for eigenvalue ~1
    # Solve as right eigenvector of P^T
    evals, evecs = la.eig(P.T)
    # Find eigenvector corresponding to eigenvalue closest to 1
    idx = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, idx])

    # Normalize to sum to 1 (ensure non-negative)
    if pi.sum() < 0:
        pi = -pi
    pi = pi / pi.sum()

    # Mixing time
    mixing_time = 1.0 / spectral_gap if spectral_gap > 0 else float("inf")

    return {
        "eigenvalues": eigenvalues,
        "lambda_2": lambda_2,
        "spectral_gap": spectral_gap,
        "stationary": pi,
        "mixing_time": mixing_time,
    }
