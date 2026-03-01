/-
  HermesNLCDM.Energy
  ==================
  Core energy function definitions for the coupled nonlinear dynamical memory system.

  Modern Hopfield energy (Ramsauer et al. 2020):
    E(ξ) = -lse(β, X^T ξ) + ½ ξ^T ξ + β⁻¹ log N + ½ M²

  Coupled system energy:
    E(X) = Σ_i E_local(x_i) + Σ_{i<j} E_coupling(x_i, x_j, W_ij)

  where:
    E_local  = individual memory energy (Hopfield + decay + importance)
    E_coupling = interaction energy via smooth weight function

  Weight function uses smooth sigmoid thresholds (not hard cutoffs):
    W(s) = s · σ(k(s - τ_high)) - s · σ(k(τ_low - s))

  where σ = sigmoid, k = steepness parameter.
-/

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

noncomputable section
open Real

namespace HermesNLCDM

/-! ## Smooth Weight Function

The coupling weight between memories i and j is determined by their
cosine similarity s_ij, passed through a smooth threshold function.

Three regimes:
- Attractive (s > τ_high): reinforcement, priming
- Neutral (τ_low < s < τ_high): independence
- Repulsive (s < τ_low): contradiction suppression
-/

/-- Sigmoid function σ(x) = 1 / (1 + exp(-x)) -/
def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- Sigmoid is bounded in (0, 1) -/
theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  positivity

theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one (by positivity)]
  linarith [exp_pos (-x)]

/-- Sigmoid is monotone increasing -/
theorem sigmoid_strictMono : StrictMono sigmoid := by
  intro a b hab
  unfold sigmoid
  have h1 : (0 : ℝ) < 1 + exp (-a) := by positivity
  have h2 : (0 : ℝ) < 1 + exp (-b) := by positivity
  have h3 : exp (-b) < exp (-a) := exp_lt_exp.mpr (neg_lt_neg hab)
  have h4 : 1 + exp (-b) < 1 + exp (-a) := by linarith
  exact div_lt_div_of_pos_left one_pos h2 h4

/-- Smooth weight function parameters -/
structure WeightParams where
  τ_high : ℝ       -- attractive threshold
  τ_low : ℝ        -- repulsive threshold
  k : ℝ            -- steepness (k ≥ k_min for smooth-to-sharp transition)
  k_pos : 0 < k
  thresholds_ordered : τ_low < τ_high

/-- Smooth coupling weight function W(s) = s·σ(k(s - τ_high)) - s·σ(k(τ_low - s))
    Attractive regime: s >> τ_high → W(s) ≈ s (positive coupling)
    Neutral regime: τ_low << s << τ_high → W(s) ≈ 0
    Repulsive regime: s << τ_low → W(s) ≈ -s (negative coupling) -/
def smoothWeight (p : WeightParams) (s : ℝ) : ℝ :=
  s * sigmoid (p.k * (s - p.τ_high)) - s * sigmoid (p.k * (p.τ_low - s))

/-- Sigmoid is continuous -/
theorem sigmoid_continuous : Continuous sigmoid := by
  unfold sigmoid
  apply Continuous.div continuous_const
  · exact continuous_const.add (Real.continuous_exp.comp continuous_neg)
  · intro x; exact ne_of_gt (by positivity)

/-- The smooth weight function is continuous (composition of continuous functions) -/
theorem smoothWeight_continuous (p : WeightParams) : Continuous (smoothWeight p) := by
  unfold smoothWeight
  apply Continuous.sub <;> apply Continuous.mul continuous_id
  · exact sigmoid_continuous.comp ((continuous_const.mul (continuous_id.sub continuous_const)))
  · exact sigmoid_continuous.comp ((continuous_const.mul (continuous_const.sub continuous_id)))

/-- W(0) = 0: zero similarity means zero coupling -/
theorem smoothWeight_zero (p : WeightParams) : smoothWeight p 0 = 0 := by
  simp [smoothWeight]

/-! ## Memory State and Pattern Space -/

/-- A memory state in d-dimensional embedding space -/
structure MemoryState (d : ℕ) where
  embedding : Fin d → ℝ    -- embedding vector
  strength : ℝ              -- memory strength (∈ [0, 1])
  importance : ℝ            -- importance score (∈ [0, 1])
  activation : ℝ            -- current activation level (≥ 0)
  strength_bounds : 0 ≤ strength ∧ strength ≤ 1
  importance_bounds : 0 ≤ importance ∧ importance ≤ 1
  activation_nonneg : 0 ≤ activation

/-- System configuration -/
structure SystemConfig (d N : ℕ) where
  β : ℝ                          -- inverse temperature (sharpness of softmax)
  β_pos : 0 < β
  weights : WeightParams          -- coupling weight parameters
  patterns : Fin N → Fin d → ℝ   -- stored patterns (embedding vectors)

/-- Cosine similarity between two vectors -/
def cosineSim {d : ℕ} (u v : Fin d → ℝ) : ℝ :=
  (∑ i, u i * v i) / (Real.sqrt (∑ i, u i ^ 2) * Real.sqrt (∑ i, v i ^ 2))

/-! ## Energy Function Components -/

/-- Log-sum-exp: lse(β, z) = β⁻¹ · log(Σ_i exp(β · z_i))
    This is the smooth maximum function used in Modern Hopfield. -/
def logSumExp (β : ℝ) {N : ℕ} (z : Fin N → ℝ) : ℝ :=
  β⁻¹ * Real.log (∑ i, exp (β * z i))

/-- Local energy for a single memory state (Modern Hopfield form):
    E_local(ξ, X) = -lse(β, X^T ξ) + ½ ‖ξ‖² -/
def localEnergy {d N : ℕ} (cfg : SystemConfig d N) (ξ : Fin d → ℝ) : ℝ :=
  -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) + (1/2) * ∑ i, ξ i ^ 2

/-- Coupling energy between two memory states:
    E_coupling(x_i, x_j) = -½ W(cos(x_i, x_j)) · ‖x_i‖ · ‖x_j‖ -/
def couplingEnergy {d : ℕ} (wp : WeightParams)
    (x_i x_j : Fin d → ℝ) : ℝ :=
  -(1/2) * smoothWeight wp (cosineSim x_i x_j) *
    Real.sqrt (∑ k, x_i k ^ 2) * Real.sqrt (∑ k, x_j k ^ 2)

/-- Total system energy for N coupled memories:
    E(X) = Σ_i E_local(x_i) + Σ_{i<j} E_coupling(x_i, x_j, W_ij) -/
def totalEnergy {d N M : ℕ} (cfg : SystemConfig d M)
    (memories : Fin N → Fin d → ℝ) : ℝ :=
  (∑ i, localEnergy cfg (memories i)) +
  ∑ i : Fin N, ∑ j : Fin N,
    if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0

end HermesNLCDM
