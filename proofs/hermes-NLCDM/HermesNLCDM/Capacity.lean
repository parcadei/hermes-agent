/-
  HermesNLCDM.Capacity
  ====================
  Phase 4: Conditional Capacity

  Main results:
  1. Capacity criterion: 4NβM² ≤ exp(βδ) implies Phase 1 conditions
  2. Tail bound and concentration under capacity
  3. Local minima from capacity (MAIN THEOREM)
  4. Coupling bound from pairwise monotonicity
  5. Coupled descent with pairwise condition

  The headline result: Modern Hopfield networks have exponential storage
  capacity N_max ~ exp(βδ)/(4βM²), with each stored pattern provably
  a local energy minimum.

  Reference: Ramsauer et al. 2020, Theorem 4 (exponential capacity)
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.LocalMinima
import HermesNLCDM.Dynamics
import HermesNLCDM.Coupling
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Capacity Criterion

  The key algebraic bridge: the exponential capacity bound
  N ≤ exp(βδ)/(4βM²) implies the logarithmic condition
  β·δ ≥ log(4NβM²) required by Phase 1.
-/

/-- Capacity criterion: 4NβM² ≤ exp(βδ) implies β·δ ≥ log(4NβM²).
    This bridges the capacity bound and the Phase 1 local minima condition. -/
theorem capacity_criterion {β δ M : ℝ} {N : ℕ} [NeZero N]
    (hβ : 0 < β) (hM_pos : 0 < M)
    (hcap : 4 * ↑N * β * M ^ 2 ≤ exp (β * δ)) :
    β * δ ≥ Real.log (4 * ↑N * β * M ^ 2) := by
  have hN_pos : (0 : ℝ) < ↑N := Nat.cast_pos.mpr (NeZero.pos N)
  have h_pos : 0 < 4 * ↑N * β * M ^ 2 := by positivity
  have h1 : Real.log (4 * ↑N * β * M ^ 2) ≤ Real.log (exp (β * δ)) :=
    Real.log_le_log h_pos hcap
  rw [Real.log_exp] at h1
  linarith

/-! ## Tail Bound Under Capacity -/

/-- Under the capacity condition, the softmax tail is bounded by 1/(4βM²). -/
theorem tail_bound_from_capacity {β δ M : ℝ} {N : ℕ} [NeZero N]
    (hβ : 0 < β) (hM_pos : 0 < M)
    (hcap : 4 * ↑N * β * M ^ 2 ≤ exp (β * δ)) :
    (↑N - 1) * exp (-β * δ) ≤ 1 / (4 * β * M ^ 2) := by
  have h4βM_pos : (0 : ℝ) < 4 * β * M ^ 2 := by positivity
  have hexp_neg_pos : (0 : ℝ) < exp (-β * δ) := exp_pos _
  rw [le_div_iff₀ h4βM_pos]
  -- Goal: (↑N - 1) * exp(-βδ) * (4βM²) ≤ 1
  have hprod : exp (β * δ) * exp (-β * δ) = 1 := by
    rw [← exp_add, show β * δ + -β * δ = 0 from by ring, exp_zero]
  have hmul := mul_le_mul_of_nonneg_right hcap (le_of_lt hexp_neg_pos)
  rw [hprod] at hmul
  -- hmul : 4 * ↑N * β * M ^ 2 * exp (-β * δ) ≤ 1
  nlinarith [mul_nonneg h4βM_pos.le hexp_neg_pos.le]

/-! ## Concentration Under Capacity -/

/-- Under the capacity condition, softmax concentrates with error ≤ 1/(4βM²). -/
theorem concentration_from_capacity {d N : ℕ} [NeZero N] {δ M : ℝ}
    (cfg : SystemConfig d N) (μ : Fin N)
    (hM_pos : 0 < M)
    (hSep : ∀ ν, ν ≠ μ →
      (∑ i, cfg.patterns μ i * cfg.patterns μ i) -
      (∑ i, cfg.patterns ν i * cfg.patterns μ i) ≥ δ)
    (hcap : 4 * ↑N * cfg.β * M ^ 2 ≤ exp (cfg.β * δ)) :
    1 - 1 / (4 * cfg.β * M ^ 2) ≤ softmax cfg.β
      (fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i) μ := by
  have htail := tail_bound_from_capacity cfg.β_pos hM_pos hcap
  have hconc := softmax_concentration cfg.β_pos
    (fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i) μ hSep
  linarith

/-! ## Local Minima from Capacity

  PHASE 4 MAIN THEOREM: The capacity condition 4NβM² ≤ exp(βδ)
  ensures each stored pattern is a local minimum. This gives
  exponential capacity: N_max ~ exp(βδ)/(4βM²).
-/

/-- PHASE 4 MAIN THEOREM: Under the capacity condition, each stored pattern
    is a local minimum with gradient ≈ 0 and positive definite Hessian. -/
theorem local_minima_from_capacity {d N : ℕ} [NeZero N] {δ M : ℝ}
    (cfg : SystemConfig d N) (μ : Fin N)
    (hM_pos : 0 < M)
    (hSep : ∀ ν, ν ≠ μ →
      (∑ i, cfg.patterns μ i * cfg.patterns μ i) -
      (∑ i, cfg.patterns ν i * cfg.patterns μ i) ≥ δ)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hcap : 4 * ↑N * cfg.β * M ^ 2 ≤ exp (cfg.β * δ)) :
    sqNorm (localEnergyGrad cfg (cfg.patterns μ)) ≤
      4 * ((↑N - 1) * exp (-cfg.β * δ) * M) ^ 2 ∧
    (∀ v : Fin d → ℝ, sqNorm v > 0 →
      let similarities := fun ν => ∑ i, cfg.patterns ν i * (cfg.patterns μ i)
      let p := softmax cfg.β similarities
      (∑ i, v i ^ 2) - cfg.β *
        ((∑ ν, p ν * (∑ i, cfg.patterns ν i * v i) ^ 2) -
         (∑ ν, p ν * (∑ i, cfg.patterns ν i * v i)) ^ 2) > 0) :=
  stored_patterns_are_local_minima cfg μ hSep hNorm
    (capacity_criterion cfg.β_pos hM_pos hcap)

/-! ## Coupling Bound from Pairwise Monotonicity

  The Phase 3 coupling bound hypothesis can be derived from a simpler
  per-pair condition: if each coupling term involving memory k does not
  increase after the update, the total coupling bound holds.
-/

/-- If the coupling energy between the updated memory k and every other
    memory does not exceed the original, the Phase 3 coupling bound holds.
    Uses coupling symmetry to handle both row-k and col-k terms. -/
theorem coupling_bound_from_pairwise {d N M_count : ℕ}
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (updated_k : Fin d → ℝ)
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights updated_k (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j)) :
    (∑ j : Fin N,
      if j.val < k.val then
        couplingEnergy cfg.weights updated_k (memories j)
      else 0) +
    (∑ i : Fin N,
      if k.val < i.val then
        couplingEnergy cfg.weights (memories i) updated_k
      else 0) ≤
    (∑ j : Fin N,
      if j.val < k.val then
        couplingEnergy cfg.weights (memories k) (memories j)
      else 0) +
    (∑ i : Fin N,
      if k.val < i.val then
        couplingEnergy cfg.weights (memories i) (memories k)
      else 0) := by
  apply add_le_add
  · -- Row-k: terms with j < k
    apply Finset.sum_le_sum; intro j _
    split_ifs with h
    · exact h_pairwise j (by intro heq; rw [heq] at h; omega)
    · exact le_refl _
  · -- Col-k: terms with k < i, use coupling symmetry
    apply Finset.sum_le_sum; intro i _
    split_ifs with h
    · rw [couplingEnergy_comm cfg.weights (memories i) updated_k,
          couplingEnergy_comm cfg.weights (memories i) (memories k)]
      exact h_pairwise i (by intro heq; rw [heq] at h; omega)
    · exact le_refl _

/-! ## Coupled Descent with Pairwise Condition -/

/-- Coordinate descent for the coupled system with a cleaner hypothesis:
    if each pairwise coupling involving k does not increase, total energy
    decreases. Combines Phase 2 local descent with Phase 3 coupling. -/
theorem coupled_descent_pairwise {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (k : Fin N)
    (updated_k : Fin d → ℝ)
    (h_local_descent : localEnergy cfg updated_k ≤ localEnergy cfg (memories k))
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights updated_k (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j)) :
    let updated : Fin N → Fin d → ℝ := fun i => if i = k then updated_k else memories i
    totalEnergy cfg updated ≤ totalEnergy cfg memories :=
  coupled_single_update_descent cfg memories k updated_k h_local_descent
    (coupling_bound_from_pairwise cfg memories k updated_k h_pairwise)

end HermesNLCDM
