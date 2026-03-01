/-
  HermesNLCDM.Dynamics
  ====================
  Phase 2: Single Update Convergence

  Main results:
  1. Energy descent: E(ξ_new) ≤ E(ξ_old) for one Hopfield update step
  2. Fixed point characterization (structure only, strict decrease deferred)
  3. Update preserves norm bounds

  The Modern Hopfield update rule:
    ξ_new = X · softmax(β, X^T ξ_old)

  This is the CCCP (concave-convex procedure) applied to E = E_concave + E_convex
  where E_concave = -lse(β, X^T ξ) and E_convex = ½‖ξ‖².

  The descent proof uses the Fenchel-Young equality for lse and Gibbs' inequality.

  Reference: Ramsauer et al. 2020, Theorem 3; Hopfield 1982, Theorem 1
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.LocalMinima
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Softmax Positivity -/

/-- Each softmax entry is strictly positive -/
theorem softmax_pos {β : ℝ} {N : ℕ} [NeZero N]
    (z : Fin N → ℝ) (μ : Fin N) : 0 < softmax β z μ := by
  unfold softmax; positivity

/-! ## Update Rule -/

/-- Modern Hopfield update rule: ξ_new = X · softmax(β, X^T ξ_old) -/
def hopfieldUpdate {d N : ℕ} (cfg : SystemConfig d N) (ξ : Fin d → ℝ) : Fin d → ℝ :=
  fun k =>
    let similarities : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ i
    ∑ μ, softmax cfg.β similarities μ * cfg.patterns μ k

/-- A state is a fixed point if the update doesn't change it -/
def isFixedPoint {d N : ℕ} (cfg : SystemConfig d N) (ξ : Fin d → ℝ) : Prop :=
  hopfieldUpdate cfg ξ = ξ

/-! ## Jensen's Inequality for Weighted Exponential -/

/-- Jensen's inequality for exp: exp(Σ w·x) ≤ Σ w·exp(x) for probability weights -/
theorem exp_weighted_sum_le {N : ℕ} (w : Fin N → ℝ) (x : Fin N → ℝ)
    (hw_nn : ∀ μ, 0 ≤ w μ) (hw_sum : ∑ μ, w μ = 1) :
    exp (∑ μ, w μ * x μ) ≤ ∑ μ, w μ * exp (x μ) := by
  have h := convexOn_exp.map_sum_le (s := Set.univ) (t := Finset.univ)
    (w := w) (p := x) (fun i _ => hw_nn i) hw_sum (fun i _ => Set.mem_univ _)
  simp_rw [smul_eq_mul] at h
  exact h

/-! ## Fenchel-Young Inequality (weak form) -/

/-- Weak Fenchel-Young: lse(β, z) ≥ Σ p_μ z_μ for any probability vector p -/
theorem lse_ge_weighted_sum {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) (p : Fin N → ℝ)
    (hp_nn : ∀ μ, 0 ≤ p μ) (hp_sum : ∑ μ, p μ = 1) :
    ∑ μ, p μ * z μ ≤ logSumExp β z := by
  unfold logSumExp
  have hβ_inv_pos : 0 < β⁻¹ := inv_pos.mpr hβ
  rw [show ∑ μ, p μ * z μ = β⁻¹ * (β * ∑ μ, p μ * z μ) from by
    rw [inv_mul_cancel_left₀ (ne_of_gt hβ)]]
  apply mul_le_mul_of_nonneg_left _ hβ_inv_pos.le
  have hS_pos : 0 < ∑ μ : Fin N, exp (β * z μ) := by positivity
  rw [Real.le_log_iff_exp_le hS_pos]
  calc exp (β * ∑ μ, p μ * z μ)
      = exp (∑ μ, p μ * (β * z μ)) := by
        congr 1; rw [Finset.mul_sum]; congr 1; ext μ; ring
    _ ≤ ∑ μ, p μ * exp (β * z μ) := exp_weighted_sum_le p _ hp_nn hp_sum
    _ ≤ ∑ μ, exp (β * z μ) := by
        apply Finset.sum_le_sum; intro μ _
        calc p μ * exp (β * z μ)
            ≤ 1 * exp (β * z μ) := by
              apply mul_le_mul_of_nonneg_right _ (le_of_lt (exp_pos _))
              calc p μ ≤ ∑ ν, p ν :=
                    Finset.single_le_sum (fun i _ => hp_nn i) (Finset.mem_univ μ)
                _ = 1 := hp_sum
          _ = exp (β * z μ) := one_mul _

/-! ## Log-Sum-Exp / Softmax Identity -/

/-- log(softmax_μ) = β z_μ - log(Σ exp(β z_ν)) -/
theorem log_softmax {β : ℝ} {N : ℕ} [NeZero N] (_hβ : 0 < β)
    (z : Fin N → ℝ) (μ : Fin N) :
    Real.log (softmax β z μ) = β * z μ - Real.log (∑ ν, exp (β * z ν)) := by
  unfold softmax
  have hS_pos : 0 < ∑ ν : Fin N, exp (β * z ν) := by positivity
  rw [Real.log_div (ne_of_gt (exp_pos _)) (ne_of_gt hS_pos), Real.log_exp]

/-- Σ p_μ log(p_μ) = β Σ p_μ z_μ - log S where p = softmax -/
theorem softmax_weighted_log {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) :
    ∑ μ, softmax β z μ * Real.log (softmax β z μ) =
      β * ∑ μ, softmax β z μ * z μ -
      Real.log (∑ ν, exp (β * z ν)) := by
  simp_rw [log_softmax hβ z, mul_sub, Finset.sum_sub_distrib]
  congr 1
  · rw [Finset.mul_sum]; congr 1; ext μ; ring
  · rw [← Finset.sum_mul, softmax_sum_one, one_mul]

/-- Fenchel-Young equality: lse(β, z) = Σ softmax_μ · z_μ - β⁻¹ Σ softmax_μ log(softmax_μ) -/
theorem lse_eq_softmax_sum {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) :
    logSumExp β z = ∑ μ, softmax β z μ * z μ -
      β⁻¹ * ∑ μ, softmax β z μ * Real.log (softmax β z μ) := by
  rw [softmax_weighted_log hβ z]
  unfold logSumExp
  field_simp
  ring

/-! ## Surrogate Minimizer -/

/-- The quadratic surrogate Q(ξ) = ½‖ξ‖² - <c, ξ> is minimized at ξ = c.
    Stated as: Q(c) ≤ Q(ξ) for all ξ, where c = X·p. -/
theorem surrogate_minimizer {d N : ℕ}
    (patterns : Fin N → Fin d → ℝ) (p : Fin N → ℝ)
    (ξ : Fin d → ℝ) :
    let c : Fin d → ℝ := fun k => ∑ μ, p μ * patterns μ k
    (-(∑ μ, p μ * ∑ i, patterns μ i * c i) + (1/2) * ∑ i, c i ^ 2) ≤
    (-(∑ μ, p μ * ∑ i, patterns μ i * ξ i) + (1/2) * ∑ i, ξ i ^ 2) := by
  intro c
  -- Σ_μ p_μ (Σ_i x_μ_i v_i) = Σ_i c_i v_i for any v
  have hswap : ∀ v : Fin d → ℝ, ∑ μ, p μ * ∑ i, patterns μ i * v i =
      ∑ i, c i * v i := by
    intro v; simp_rw [Finset.mul_sum]; rw [Finset.sum_comm]
    congr 1; ext i
    simp_rw [show ∀ x, p x * (patterns x i * v i) = (p x * patterns x i) * v i from fun x => by ring]
    rw [← Finset.sum_mul]
  rw [hswap ξ, hswap c]
  -- Goal: -(Σ c*c) + ½ Σ c² ≤ -(Σ c*ξ) + ½ Σ ξ²
  -- Note: c*c = c^2, so LHS = -½ Σ c²
  -- Suffices: 0 ≤ Σ ξ² - 2 Σ c*ξ + Σ c² = Σ(ξ-c)²
  have hsq : 0 ≤ ∑ i, (ξ i - c i) ^ 2 := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  have hexp : ∑ i, (ξ i - c i) ^ 2 = ∑ i, ξ i ^ 2 - 2 * ∑ i, c i * ξ i + ∑ i, c i ^ 2 := by
    have : ∀ i, (ξ i - c i) ^ 2 = ξ i ^ 2 - 2 * (c i * ξ i) + c i ^ 2 := fun i => by ring
    simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum]
  -- c*c = c^2
  have hcc : ∑ i, c i * c i = ∑ i, c i ^ 2 := by congr 1; ext i; ring
  linarith

/-! ## Entropy nonnegativity -/

/-- Σ p_μ log(p_μ) ≤ 0 for softmax (entropy is nonneg) -/
theorem softmax_entropy_nonneg {β : ℝ} {N : ℕ} [NeZero N] (_hβ : 0 < β)
    (z : Fin N → ℝ) :
    ∑ μ, softmax β z μ * Real.log (softmax β z μ) ≤ 0 := by
  apply Finset.sum_nonpos
  intro μ _
  have hp_pos := softmax_pos z μ (β := β)
  have hp_le : softmax β z μ ≤ 1 := by
    calc softmax β z μ ≤ ∑ ν, softmax β z ν :=
          Finset.single_le_sum (fun i _ => le_of_lt (softmax_pos z i)) (Finset.mem_univ μ)
      _ = 1 := softmax_sum_one z
  exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt hp_pos)
    (Real.log_nonpos (le_of_lt hp_pos) hp_le)

/-! ## Gibbs' Inequality and Full Fenchel-Young -/

/-- Gibbs' inequality: Σ q_μ log(p_μ) ≤ Σ q_μ log(q_μ) for probability vectors.
    Equivalently KL(q ‖ p) ≥ 0. Uses log x ≤ x - 1. -/
theorem gibbs_inequality {N : ℕ} [NeZero N]
    (q p : Fin N → ℝ)
    (hq_pos : ∀ μ, 0 < q μ) (hq_sum : ∑ μ, q μ = 1)
    (hp_pos : ∀ μ, 0 < p μ) (hp_sum : ∑ μ, p μ = 1) :
    ∑ μ, q μ * Real.log (p μ) ≤ ∑ μ, q μ * Real.log (q μ) := by
  suffices h : ∑ μ, q μ * (Real.log (p μ) - Real.log (q μ)) ≤ 0 by
    have heq : ∑ μ, q μ * (Real.log (p μ) - Real.log (q μ)) =
        ∑ μ, q μ * Real.log (p μ) - ∑ μ, q μ * Real.log (q μ) := by
      simp_rw [mul_sub, Finset.sum_sub_distrib]
    linarith
  -- log(p/q) ≤ p/q - 1
  have hlog_le : ∀ μ, Real.log (p μ / q μ) ≤ p μ / q μ - 1 :=
    fun μ => Real.log_le_sub_one_of_pos (div_pos (hp_pos μ) (hq_pos μ))
  calc ∑ μ, q μ * (Real.log (p μ) - Real.log (q μ))
      = ∑ μ, q μ * Real.log (p μ / q μ) := by
        congr 1; ext μ; rw [Real.log_div (ne_of_gt (hp_pos μ)) (ne_of_gt (hq_pos μ))]
    _ ≤ ∑ μ, q μ * (p μ / q μ - 1) :=
        Finset.sum_le_sum (fun μ _ =>
          mul_le_mul_of_nonneg_left (hlog_le μ) (le_of_lt (hq_pos μ)))
    _ = ∑ μ, (p μ - q μ) := by
        congr 1; ext μ
        rw [mul_sub, mul_div_cancel₀ _ (ne_of_gt (hq_pos μ)), mul_one]
    _ = ∑ μ, p μ - ∑ μ, q μ := by rw [← Finset.sum_sub_distrib]
    _ = 0 := by rw [hp_sum, hq_sum, sub_self]

/-- Full Fenchel-Young: lse(β, z) ≥ Σ q_μ z_μ - β⁻¹ Σ q_μ log(q_μ)
    for any probability vector q with positive entries -/
theorem lse_ge_weighted_sum_plus_entropy {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) (q : Fin N → ℝ)
    (hq_pos : ∀ μ, 0 < q μ) (hq_sum : ∑ μ, q μ = 1) :
    ∑ μ, q μ * z μ - β⁻¹ * ∑ μ, q μ * Real.log (q μ) ≤ logSumExp β z := by
  set p := softmax β z
  have hp_pos : ∀ μ, 0 < p μ := fun μ => softmax_pos z μ
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one z
  have hgibbs := gibbs_inequality q p hq_pos hq_sum hp_pos hp_sum
  -- Σ q log(p) = β Σ q z - log S
  have hq_log_p : ∑ μ, q μ * Real.log (p μ) =
      β * ∑ μ, q μ * z μ - Real.log (∑ ν, exp (β * z ν)) := by
    simp_rw [show ∀ μ, Real.log (p μ) = β * z μ - Real.log (∑ ν, exp (β * z ν)) from
      log_softmax hβ z]
    simp_rw [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hq_sum, one_mul]
    congr 1; rw [Finset.mul_sum]; congr 1; ext μ; ring
  -- From Gibbs + hq_log_p: β Σ q z - log S ≤ Σ q log(q)
  -- Rearrange: β Σ q z - Σ q log(q) ≤ log S
  -- Multiply by β⁻¹: Σ q z - β⁻¹ Σ q log(q) ≤ β⁻¹ log S = lse
  have key : β * ∑ μ, q μ * z μ - ∑ μ, q μ * Real.log (q μ) ≤
      Real.log (∑ ν, exp (β * z ν)) := by linarith [hq_log_p]
  have hβ_inv_pos : 0 < β⁻¹ := inv_pos.mpr hβ
  have key2 : β⁻¹ * (β * ∑ μ, q μ * z μ - ∑ μ, q μ * Real.log (q μ)) ≤
      β⁻¹ * Real.log (∑ ν, exp (β * z ν)) :=
    mul_le_mul_of_nonneg_left key hβ_inv_pos.le
  have hlhs : β⁻¹ * (β * ∑ μ, q μ * z μ - ∑ μ, q μ * Real.log (q μ)) =
      ∑ μ, q μ * z μ - β⁻¹ * ∑ μ, q μ * Real.log (q μ) := by
    rw [mul_sub, ← mul_assoc, inv_mul_cancel₀ (ne_of_gt hβ), one_mul]
  -- logSumExp β z = β⁻¹ * log S
  unfold logSumExp
  linarith

/-! ## Energy Descent Theorem

  Proof chain using the CCCP surrogate Q̃(ξ | ξ'):
  1. Q̃(ξ_old | ξ_old) = E(ξ_old)            [FY equality]
  2. E(ξ_new) ≤ Q̃(ξ_new | ξ_old)            [Full FY inequality]
  3. Q̃(ξ_new | ξ_old) ≤ Q̃(ξ_old | ξ_old)   [ξ_new minimizes surrogate]
-/

/-- PHASE 2 MAIN THEOREM: Energy descent for the Hopfield update rule.
    E(hopfieldUpdate(ξ)) ≤ E(ξ) -/
theorem energy_descent {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) :
    localEnergy cfg (hopfieldUpdate cfg ξ) ≤ localEnergy cfg ξ := by
  -- Unfold just localEnergy; keep hopfieldUpdate as a definition
  -- to avoid let-binding issues
  show -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * hopfieldUpdate cfg ξ i) +
    (1/2) * ∑ i, hopfieldUpdate cfg ξ i ^ 2 ≤
    -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) +
    (1/2) * ∑ i, ξ i ^ 2
  -- Set up notation
  set sim : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ i
  set p : Fin N → ℝ := softmax cfg.β sim
  -- ξ_new = hopfieldUpdate cfg ξ = X · p
  have hξ_new_eq : ∀ k, hopfieldUpdate cfg ξ k = ∑ μ, p μ * cfg.patterns μ k := by
    intro k; unfold hopfieldUpdate; simp only [sim, p]
  set ξ_new : Fin d → ℝ := fun k => ∑ μ, p μ * cfg.patterns μ k
  have hupd : hopfieldUpdate cfg ξ = ξ_new := funext hξ_new_eq
  rw [hupd]
  set sim_new : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ_new i
  have hp_pos : ∀ μ, 0 < p μ := fun μ => softmax_pos sim μ
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one sim
  -- Step 1: FY equality at ξ_old
  have step1 : -logSumExp cfg.β sim + (1/2) * ∑ i, ξ i ^ 2 =
      -(∑ μ, p μ * sim μ) + cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ) +
      (1/2) * ∑ i, ξ i ^ 2 := by
    linarith [lse_eq_softmax_sum cfg.β_pos sim]
  -- Step 2: Full FY at ξ_new gives E(ξ_new) ≤ Q̃(ξ_new)
  have step2 : -logSumExp cfg.β sim_new + (1/2) * ∑ i, ξ_new i ^ 2 ≤
      -(∑ μ, p μ * sim_new μ) + cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ) +
      (1/2) * ∑ i, ξ_new i ^ 2 := by
    linarith [lse_ge_weighted_sum_plus_entropy cfg.β_pos sim_new p hp_pos hp_sum]
  -- Step 3: Q̃(ξ_new) ≤ Q̃(ξ_old) by surrogate minimizer
  have step3 := surrogate_minimizer cfg.patterns p ξ
  -- Chain: E(ξ_new) ≤ Q̃(ξ_new) ≤ Q̃(ξ_old) = E(ξ_old)
  linarith

/-! ## Strict Surrogate Minimizer -/

/-- Strict surrogate minimizer: Q(c) < Q(ξ) when ξ ≠ c -/
theorem surrogate_minimizer_strict {d N : ℕ}
    (patterns : Fin N → Fin d → ℝ) (p : Fin N → ℝ)
    (ξ : Fin d → ℝ)
    (hne : (fun k => ∑ μ, p μ * patterns μ k) ≠ ξ) :
    let c : Fin d → ℝ := fun k => ∑ μ, p μ * patterns μ k
    (-(∑ μ, p μ * ∑ i, patterns μ i * c i) + (1/2) * ∑ i, c i ^ 2) <
    (-(∑ μ, p μ * ∑ i, patterns μ i * ξ i) + (1/2) * ∑ i, ξ i ^ 2) := by
  intro c
  have hswap : ∀ v : Fin d → ℝ, ∑ μ, p μ * ∑ i, patterns μ i * v i =
      ∑ i, c i * v i := by
    intro v; simp_rw [Finset.mul_sum]; rw [Finset.sum_comm]
    congr 1; ext i
    simp_rw [show ∀ x, p x * (patterns x i * v i) = (p x * patterns x i) * v i from fun x => by ring]
    rw [← Finset.sum_mul]
  rw [hswap ξ, hswap c]
  -- c ≠ ξ → ∃ i, c i ≠ ξ i → ∑ (ξ - c)² > 0
  have hne_pt : ∃ i, c i ≠ ξ i := Function.ne_iff.mp hne
  have hsq_pos : 0 < ∑ i, (ξ i - c i) ^ 2 := by
    obtain ⟨i, hi⟩ := hne_pt
    have hi_sq : 0 < (ξ i - c i) ^ 2 :=
      sq_pos_of_ne_zero (sub_ne_zero.mpr (Ne.symm hi))
    calc 0 < (ξ i - c i) ^ 2 := hi_sq
      _ ≤ ∑ j : Fin d, (ξ j - c j) ^ 2 :=
        Finset.single_le_sum (f := fun j => (ξ j - c j) ^ 2)
          (fun j _ => sq_nonneg _) (Finset.mem_univ i)
  have hexp : ∑ i, (ξ i - c i) ^ 2 = ∑ i, ξ i ^ 2 - 2 * ∑ i, c i * ξ i + ∑ i, c i ^ 2 := by
    have : ∀ i, (ξ i - c i) ^ 2 = ξ i ^ 2 - 2 * (c i * ξ i) + c i ^ 2 := fun i => by ring
    simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum]
  have hcc : ∑ i, c i * c i = ∑ i, c i ^ 2 := by congr 1; ext i; ring
  nlinarith

/-! ## Strict Energy Descent -/

/-- STRICT DESCENT: If ξ is not a fixed point, energy strictly decreases.
    E(hopfieldUpdate(ξ)) < E(ξ) when hopfieldUpdate(ξ) ≠ ξ -/
theorem energy_descent_strict {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (hne : ¬isFixedPoint cfg ξ) :
    localEnergy cfg (hopfieldUpdate cfg ξ) < localEnergy cfg ξ := by
  unfold isFixedPoint at hne
  show -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * hopfieldUpdate cfg ξ i) +
    (1/2) * ∑ i, hopfieldUpdate cfg ξ i ^ 2 <
    -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) +
    (1/2) * ∑ i, ξ i ^ 2
  set sim : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ i
  set p : Fin N → ℝ := softmax cfg.β sim
  have hξ_new_eq : ∀ k, hopfieldUpdate cfg ξ k = ∑ μ, p μ * cfg.patterns μ k := by
    intro k; unfold hopfieldUpdate; simp only [sim, p]
  set ξ_new : Fin d → ℝ := fun k => ∑ μ, p μ * cfg.patterns μ k
  have hupd : hopfieldUpdate cfg ξ = ξ_new := funext hξ_new_eq
  rw [hupd]
  set sim_new : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ_new i
  have hp_pos : ∀ μ, 0 < p μ := fun μ => softmax_pos sim μ
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one sim
  -- Step 2: Full FY at ξ_new gives E(ξ_new) ≤ Q̃(ξ_new)
  have step2 : -logSumExp cfg.β sim_new + (1/2) * ∑ i, ξ_new i ^ 2 ≤
      -(∑ μ, p μ * sim_new μ) + cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ) +
      (1/2) * ∑ i, ξ_new i ^ 2 := by
    linarith [lse_ge_weighted_sum_plus_entropy cfg.β_pos sim_new p hp_pos hp_sum]
  -- Step 3: Q̃(ξ_new) < Q̃(ξ_old) STRICTLY because ξ_new ≠ ξ
  have hne' : ξ_new ≠ ξ := by rwa [← hupd]
  have step3 := surrogate_minimizer_strict cfg.patterns p ξ hne'
  -- Chain: E(ξ_new) ≤ Q̃(ξ_new) < Q̃(ξ_old) = E(ξ_old)
  linarith [lse_eq_softmax_sum cfg.β_pos sim]

end HermesNLCDM
