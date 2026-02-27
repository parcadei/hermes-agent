"""Hermes Memory System — Core Definitions

Direct translation of 18 Lean 4 definitions into Python.
Each function corresponds to a `noncomputable def` in the formalization.

Lean source files:
  MemoryDynamics.lean  — retention, strengthUpdate, sigmoid, score, clamp01, importanceUpdate
  StrengthDecay.lean   — strengthDecay, combinedFactor, steadyStateStrength
  SoftSelection.lean   — softSelect
  NoveltyBonus.lean    — noveltyBonus, explorationWindow, boostedScore
  ComposedSystem.lean  — expectedStrengthUpdate, composedExpectedMap, composedContractionFactor
  ContractionWiring.lean — composedDomain (predicate only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


# ============================================================
# Section 1: Retention (Forgetting Curve)
#   retention(t, S) = exp(-t/S)
# ============================================================

def retention(t: float, S: float) -> float:
    """R(t) = e^(-t/S)"""
    return math.exp(-t / S)


# ============================================================
# Section 2: Strength Update (Discrete Dynamics)
#   strengthUpdate(α, S, Smax) = S + α·(Smax - S) = (1-α)·S + α·Smax
#   strengthIter(α, S₀, Smax, n) = Smax - (Smax - S₀)·(1-α)^n
# ============================================================

def strength_update(alpha: float, S: float, S_max: float) -> float:
    """S' = S + α·(Smax - S)"""
    return S + alpha * (S_max - S)


def strength_iter(alpha: float, S0: float, S_max: float, n: int) -> float:
    """n applications of strength_update, closed form: Smax - (Smax-S₀)·(1-α)^n"""
    return S_max - (S_max - S0) * (1.0 - alpha) ** n


# ============================================================
# Section 3: Sigmoid Function
#   sigmoid(x) = 1 / (1 + exp(-x))
# ============================================================

def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + e^(-x))"""
    # Numerically stable: avoid overflow for large negative x
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(x)
        return e / (1.0 + e)


# ============================================================
# Section 4: Scoring Function
#   score(w, rel, rec, imp, act) = w₁·rel + w₂·rec + w₃·imp + w₄·σ(act)
# ============================================================

@dataclass(frozen=True)
class ScoringWeights:
    """Weights satisfying w_i ≥ 0 and Σw_i = 1."""
    w1: float
    w2: float
    w3: float
    w4: float

    def __post_init__(self) -> None:
        for name in ("w1", "w2", "w3", "w4"):
            assert getattr(self, name) >= 0, f"{name} must be non-negative"
        assert abs(self.w1 + self.w2 + self.w3 + self.w4 - 1.0) < 1e-10, "weights must sum to 1"


def score(w: ScoringWeights, rel: float, rec: float, imp: float, act: float) -> float:
    """score = w₁·rel + w₂·rec + w₃·imp + w₄·σ(act)"""
    return w.w1 * rel + w.w2 * rec + w.w3 * imp + w.w4 * sigmoid(act)


# ============================================================
# Section 5: Feedback Loop
#   clamp01(x) = max(0, min(x, 1))
#   importanceUpdate(imp, δ, signal) = clamp01(imp + δ·signal)
# ============================================================

def clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(x, 1.0))


def importance_update(imp: float, delta: float, signal: float) -> float:
    """imp' = clamp(imp + δ·signal, 0, 1)"""
    return clamp01(imp + delta * signal)


# ============================================================
# Section 7: Strength Decay (Anti-Lock-in)
#   strengthDecay(β, S₀, t) = S₀ · e^(-β·t)
#   combinedFactor(α, β, Δ) = (1-α) · e^(-β·Δ)
#   steadyStateStrength(α, β, Δ, Smax) = α·Smax / (1 - combinedFactor)
# ============================================================

def strength_decay(beta: float, S0: float, t: float) -> float:
    """S(t) = S₀ · e^(-β·t)"""
    return S0 * math.exp(-beta * t)


def combined_factor(alpha: float, beta: float, delta_t: float) -> float:
    """γ = (1-α) · e^(-β·Δ)"""
    return (1.0 - alpha) * math.exp(-beta * delta_t)


def steady_state_strength(alpha: float, beta: float, delta_t: float, S_max: float) -> float:
    """S* = α·Smax / (1 - (1-α)·e^(-β·Δ))"""
    gamma = combined_factor(alpha, beta, delta_t)
    return alpha * S_max / (1.0 - gamma)


# ============================================================
# Section 8: Soft Selection (Anti-Thrashing)
#   softSelect(s₁, s₂, T) = σ((s₁ - s₂) / T)
# ============================================================

def soft_select(s1: float, s2: float, T: float) -> float:
    """P(select memory 1) = σ((s₁ - s₂) / T)"""
    return sigmoid((s1 - s2) / T)


# ============================================================
# Section 9: Novelty Bonus (Anti-Cold-Start)
#   noveltyBonus(N₀, γ, t) = N₀ · e^(-γ·t)
#   explorationWindow(N₀, γ, ε) = ln(N₀/ε) / γ
#   boostedScore(base, N₀, γ, t) = base + noveltyBonus(N₀, γ, t)
# ============================================================

def novelty_bonus(N0: float, gamma: float, t: float) -> float:
    """novelty(t) = N₀ · e^(-γ·t)"""
    return N0 * math.exp(-gamma * t)


def exploration_window(N0: float, gamma: float, epsilon: float) -> float:
    """W = ln(N₀/ε) / γ"""
    return math.log(N0 / epsilon) / gamma


def boosted_score(base_score: float, N0: float, gamma: float, t: float) -> float:
    """boosted = base_score + novelty(t)"""
    return base_score + novelty_bonus(N0, gamma, t)


# ============================================================
# Section 10-12: Composed System
#   expectedStrengthUpdate(α, β, Δ, Smax, q, S) = (1-qα)·e^(-βΔ)·S + qα·Smax
#   composedExpectedMap(selectProb, α, β, Δ, Smax, (S₁,S₂)) = (T(p,S₁), T(1-p,S₂))
#   composedContractionFactor(β, Δ, L, α, Smax) = e^(-βΔ) + L·α·Smax
# ============================================================

def expected_strength_update(
    alpha: float, beta: float, delta_t: float, S_max: float, q: float, S: float
) -> float:
    """E[S'] = (1 - qα)·e^(-βΔ)·S + qα·Smax"""
    e = math.exp(-beta * delta_t)
    return (1.0 - q * alpha) * e * S + q * alpha * S_max


def composed_expected_map(
    select_prob: Callable[[float, float], float],
    alpha: float, beta: float, delta_t: float, S_max: float,
    S: tuple[float, float],
) -> tuple[float, float]:
    """The composed mean-field map for a 2-memory system."""
    p = select_prob(S[0], S[1])
    return (
        expected_strength_update(alpha, beta, delta_t, S_max, p, S[0]),
        expected_strength_update(alpha, beta, delta_t, S_max, 1.0 - p, S[1]),
    )


def composed_contraction_factor(
    beta: float, delta_t: float, L: float, alpha: float, S_max: float
) -> float:
    """K = exp(-βΔ) + L·α·Smax"""
    return math.exp(-beta * delta_t) + L * alpha * S_max


# ============================================================
# Section 13: Domain predicate (ContractionWiring)
#   composedDomain(Smax) = [0, Smax]²
# ============================================================

def composed_domain_contains(S_max: float, S: tuple[float, float]) -> bool:
    """Check if S ∈ [0, Smax]²."""
    return 0 <= S[0] <= S_max and 0 <= S[1] <= S_max
