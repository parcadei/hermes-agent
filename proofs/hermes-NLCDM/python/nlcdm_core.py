"""
hermes-NLCDM: NonLinear Coupled Dynamical Memory
=================================================
Core definitions shared across all phases.

Mirrors the Lean definitions in HermesNLCDM/Energy.lean.
Python side validates empirical/probabilistic properties;
Lean side proves deterministic conditional guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Smooth Weight Function (mirrors HermesNLCDM.Energy.smoothWeight)
# ---------------------------------------------------------------------------

def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """σ(x) = 1 / (1 + exp(-x)), numerically stable."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


@dataclass(frozen=True)
class WeightParams:
    """Smooth coupling weight parameters.

    τ_high: attractive threshold (above → positive coupling)
    τ_low:  repulsive threshold (below → negative coupling)
    k:      steepness of sigmoid transitions
    """
    tau_high: float = 0.65
    tau_low: float = -0.1
    k: float = 20.0

    def __post_init__(self):
        assert self.k > 0, "steepness must be positive"
        assert self.tau_low < self.tau_high, "thresholds must be ordered"


def smooth_weight(params: WeightParams, s: float | np.ndarray) -> float | np.ndarray:
    """W(s) = s·σ(k(s - τ_high)) - s·σ(k(τ_low - s))

    Three regimes:
      s >> τ_high  → W(s) ≈ s   (attractive)
      τ_low < s < τ_high → W(s) ≈ 0 (neutral)
      s << τ_low  → W(s) ≈ -s  (repulsive)
    """
    attractive = s * sigmoid(params.k * (s - params.tau_high))
    repulsive = s * sigmoid(params.k * (params.tau_low - s))
    return attractive - repulsive


# ---------------------------------------------------------------------------
# Memory State (mirrors HermesNLCDM.Energy.MemoryState)
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    """A single memory in the coupled dynamical system."""
    embedding: np.ndarray          # d-dimensional vector
    strength: float = 1.0          # ∈ [0, 1]
    importance: float = 0.5        # ∈ [0, 1]
    activation: float = 0.0        # ≥ 0

    def __post_init__(self):
        assert 0 <= self.strength <= 1
        assert 0 <= self.importance <= 1
        assert 0 <= self.activation


# ---------------------------------------------------------------------------
# Energy Function (mirrors HermesNLCDM.Energy)
# ---------------------------------------------------------------------------

def log_sum_exp(beta: float, z: np.ndarray) -> float:
    """lse(β, z) = β⁻¹ · log(Σ exp(β·z_i)), numerically stable."""
    z_scaled = beta * z
    z_max = np.max(z_scaled)
    return (1 / beta) * (z_max + np.log(np.sum(np.exp(z_scaled - z_max))))


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def local_energy(beta: float, patterns: np.ndarray, xi: np.ndarray) -> float:
    """E_local(ξ) = -lse(β, X^T ξ) + ½‖ξ‖²

    Args:
        beta: inverse temperature
        patterns: (N, d) array of stored patterns
        xi: (d,) query/state vector
    """
    similarities = patterns @ xi  # (N,)
    return -log_sum_exp(beta, similarities) + 0.5 * np.dot(xi, xi)


def coupling_energy(wp: WeightParams, x_i: np.ndarray, x_j: np.ndarray) -> float:
    """E_coupling(x_i, x_j) = -½ W(cos(x_i, x_j)) · ‖x_i‖ · ‖x_j‖"""
    s = cosine_sim(x_i, x_j)
    w = smooth_weight(wp, s)
    return -0.5 * w * np.linalg.norm(x_i) * np.linalg.norm(x_j)


def total_energy(
    beta: float,
    wp: WeightParams,
    patterns: np.ndarray,
    memories: np.ndarray,
) -> float:
    """E(X) = Σ_i E_local(x_i) + Σ_{i<j} E_coupling(x_i, x_j)

    Args:
        beta: inverse temperature
        wp: weight parameters
        patterns: (M, d) stored reference patterns
        memories: (N, d) current memory states
    """
    N = memories.shape[0]
    e_local = sum(local_energy(beta, patterns, memories[i]) for i in range(N))
    e_coupling = sum(
        coupling_energy(wp, memories[i], memories[j])
        for i in range(N) for j in range(i)
    )
    return e_local + e_coupling


# ---------------------------------------------------------------------------
# Softmax (mirrors HermesNLCDM.LocalMinima.softmax)
# ---------------------------------------------------------------------------

def softmax(beta: float, z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: p_i = exp(β z_i) / Σ exp(β z_j)"""
    z_scaled = beta * z
    z_scaled -= np.max(z_scaled)  # stability
    e = np.exp(z_scaled)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# Lean/Python Interface (Section 11C of synthesis)
# ---------------------------------------------------------------------------

@dataclass
class LeanPythonBridge:
    """Audit record connecting a Lean theorem to its empirical validation.

    Lean proves: condition C → property P
    Python shows: P(condition C) ≥ 1 - ε for operating range R
    """
    lean_theorem: str
    required_condition: str
    empirical_probability: float
    sample_size: int
    operating_range: str
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "lean_theorem": self.lean_theorem,
            "required_condition": self.required_condition,
            "empirical_probability": self.empirical_probability,
            "sample_size": self.sample_size,
            "operating_range": self.operating_range,
            "confidence_interval": list(self.confidence_interval),
        }
