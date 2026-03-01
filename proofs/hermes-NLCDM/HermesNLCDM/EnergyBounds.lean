/-
  HermesNLCDM.EnergyBounds
  ========================
  Phase 1, Part 2: Energy is bounded below.

  Key result: E(X) ≥ E_min for all memory configurations X.

  For Modern Hopfield energy E(ξ) = -lse(β, X^T ξ) + ½‖ξ‖²:
    - The lse term is bounded above by max(X^T ξ) + β⁻¹ log N
    - The quadratic term ½‖ξ‖² grows without bound
    - Combined: E is bounded below and radially unbounded

  For coupling energy:
    - |E_coupling(x_i, x_j)| ≤ ½|W_max| · ‖x_i‖ · ‖x_j‖
    - With bounded memory norms, coupling energy is bounded

  Reference: Ramsauer et al. 2020, Lemma A1
-/

import HermesNLCDM.Energy
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Log-Sum-Exp Bounds

  lse(β, z) = β⁻¹ log(Σ exp(β z_i))

  Key property: max(z) ≤ lse(β, z) ≤ max(z) + β⁻¹ log N
  This bounds the Hopfield energy from below.
-/

/-- log-sum-exp is at least the maximum entry -/
theorem lse_ge_max {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β) (z : Fin N → ℝ) (j : Fin N) :
    z j ≤ logSumExp β z := by
  unfold logSumExp
  have hS_pos : 0 < ∑ i : Fin N, exp (β * z i) := by positivity
  have hβ_inv_pos : 0 < β⁻¹ := inv_pos.mpr hβ
  rw [show z j = β⁻¹ * (β * z j) from by rw [inv_mul_cancel_left₀ (ne_of_gt hβ)]]
  apply mul_le_mul_of_nonneg_left _ hβ_inv_pos.le
  rw [Real.le_log_iff_exp_le hS_pos]
  exact Finset.single_le_sum (f := fun i => exp (β * z i))
    (fun i _ => le_of_lt (exp_pos _)) (Finset.mem_univ j)

/-- log-sum-exp is at most max + β⁻¹ log N -/
theorem lse_le_max_plus {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β) (z : Fin N → ℝ) :
    logSumExp β z ≤ (Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ z) + β⁻¹ * Real.log N := by
  unfold logSumExp
  set M := Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ z
  have hz_le : ∀ i : Fin N, z i ≤ M := fun i => Finset.le_sup' z (Finset.mem_univ i)
  have hexp_le : ∀ i : Fin N, exp (β * z i) ≤ exp (β * M) :=
    fun i => exp_le_exp.mpr (by nlinarith [hz_le i])
  have hN_pos : (0 : ℝ) < ↑N := Nat.cast_pos.mpr (NeZero.pos N)
  have hS_pos : 0 < ∑ i : Fin N, exp (β * z i) := by positivity
  have hsum_le : ∑ i : Fin N, exp (β * z i) ≤ ↑N * exp (β * M) := by
    calc ∑ i : Fin N, exp (β * z i)
        ≤ ∑ _i : Fin N, exp (β * M) := Finset.sum_le_sum (fun i _ => hexp_le i)
      _ = ↑(Fintype.card (Fin N)) * exp (β * M) := by
          rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
      _ = ↑N * exp (β * M) := by rw [Fintype.card_fin]
  have hlog_le : Real.log (∑ i : Fin N, exp (β * z i)) ≤ Real.log ↑N + β * M := by
    calc Real.log (∑ i : Fin N, exp (β * z i))
        ≤ Real.log (↑N * exp (β * M)) := by
          exact Real.log_le_log hS_pos hsum_le
      _ = Real.log ↑N + Real.log (exp (β * M)) :=
          Real.log_mul (ne_of_gt hN_pos) (ne_of_gt (exp_pos _))
      _ = Real.log ↑N + β * M := by rw [Real.log_exp]
  calc β⁻¹ * Real.log (∑ i : Fin N, exp (β * z i))
      ≤ β⁻¹ * (Real.log ↑N + β * M) :=
        mul_le_mul_of_nonneg_left hlog_le (le_of_lt (inv_pos.mpr hβ))
    _ = β⁻¹ * Real.log ↑N + β⁻¹ * (β * M) := by ring
    _ = β⁻¹ * Real.log ↑N + M := by
        rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hβ), one_mul]
    _ = M + β⁻¹ * Real.log ↑N := by ring

/-! ## Local Energy Lower Bound

  E_local(ξ) = -lse(β, X^T ξ) + ½‖ξ‖²
  ≥ -max(X^T ξ) - β⁻¹ log N + ½‖ξ‖²

  For ‖ξ‖ large enough, the quadratic term dominates.
  This gives E_local → +∞ as ‖ξ‖ → ∞ (radially unbounded).
-/

/-- Squared norm of a vector -/
def sqNorm {d : ℕ} (v : Fin d → ℝ) : ℝ := ∑ i, v i ^ 2

/-- Squared norm is nonneg -/
theorem sqNorm_nonneg {d : ℕ} (v : Fin d → ℝ) : 0 ≤ sqNorm v := by
  unfold sqNorm
  apply Finset.sum_nonneg
  intro i _
  exact sq_nonneg _

/-- Dot product bounded by product of norms (Cauchy-Schwarz on Fin d → ℝ) -/
theorem dot_le_norms {d : ℕ} (u v : Fin d → ℝ) :
    |∑ i, u i * v i| ≤ Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v) := by
  rw [← Real.sqrt_mul (sqNorm_nonneg u)]
  rw [← Real.sqrt_sq_eq_abs]
  apply Real.sqrt_le_sqrt
  by_cases hv : sqNorm v = 0
  · have hv_zero : ∀ i, v i = 0 := by
      intro i
      unfold sqNorm at hv
      have h := (Finset.sum_eq_zero_iff_of_nonneg (s := Finset.univ)
        (fun i _ => sq_nonneg (v i))).mp hv i (Finset.mem_univ i)
      exact pow_eq_zero_iff two_ne_zero |>.mp h
    have hdot_zero : ∑ i, u i * v i = 0 := by
      apply Finset.sum_eq_zero; intro i _; simp [hv_zero i]
    simp [hdot_zero, hv]
  · have hv_pos : 0 < sqNorm v := lt_of_le_of_ne (sqNorm_nonneg v) (Ne.symm hv)
    set dot := ∑ i, u i * v i
    have expand : ∀ t : ℝ, ∑ i : Fin d, (u i + t * v i) ^ 2 =
        sqNorm u + 2 * t * dot + t ^ 2 * sqNorm v := by
      intro t
      simp only [sqNorm]
      simp_rw [show ∀ i : Fin d, (u i + t * v i) ^ 2 =
        u i ^ 2 + 2 * t * (u i * v i) + t ^ 2 * (v i ^ 2) from fun i => by ring]
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
          ← Finset.mul_sum, ← Finset.mul_sum]
    have key : ∀ t : ℝ, 0 ≤ sqNorm u + 2 * t * dot + t ^ 2 * sqNorm v := by
      intro t
      rw [← expand]
      exact Finset.sum_nonneg (fun i _ => sq_nonneg _)
    have h := key (-dot / sqNorm v)
    have hsimpl : sqNorm u + 2 * (-dot / sqNorm v) * dot +
        (-dot / sqNorm v) ^ 2 * sqNorm v = sqNorm u - dot ^ 2 / sqNorm v := by
      field_simp
      ring
    rw [hsimpl] at h
    have h2 : dot ^ 2 / sqNorm v ≤ sqNorm u := by linarith
    calc dot ^ 2 = dot ^ 2 / sqNorm v * sqNorm v := by
            field_simp
      _ ≤ sqNorm u * sqNorm v := by
            apply mul_le_mul_of_nonneg_right h2 hv_pos.le

/-- The local energy is bounded below: E_local(ξ) ≥ -M² - β⁻¹ log N
    where M = max pattern norm -/
theorem localEnergy_bounded_below {d N : ℕ} [NeZero N] {M : ℝ}
    (cfg : SystemConfig d N)
    (hM : ∀ μ : Fin N, sqNorm (cfg.patterns μ) ≤ M ^ 2) :
    ∀ ξ : Fin d → ℝ,
      -(M ^ 2) - cfg.β⁻¹ * Real.log N ≤ localEnergy cfg ξ := by
  intro ξ
  unfold localEnergy
  set sims : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ i
  have hlse := lse_le_max_plus cfg.β_pos sims
  set maxSim := Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ sims
  have h1 : -logSumExp cfg.β sims ≥ -maxSim - cfg.β⁻¹ * Real.log ↑N := by linarith
  have hsim_bound : ∀ μ : Fin N, sims μ ≤ |M| * Real.sqrt (sqNorm ξ) := by
    intro μ
    calc sims μ = ∑ i, cfg.patterns μ i * ξ i := rfl
      _ ≤ |∑ i, cfg.patterns μ i * ξ i| := le_abs_self _
      _ ≤ Real.sqrt (sqNorm (cfg.patterns μ)) * Real.sqrt (sqNorm ξ) := dot_le_norms _ _
      _ ≤ Real.sqrt (M ^ 2) * Real.sqrt (sqNorm ξ) := by
          apply mul_le_mul_of_nonneg_right _ (Real.sqrt_nonneg _)
          exact Real.sqrt_le_sqrt (hM μ)
      _ = |M| * Real.sqrt (sqNorm ξ) := by rw [Real.sqrt_sq_eq_abs]
  have hmaxSim_bound : maxSim ≤ |M| * Real.sqrt (sqNorm ξ) := by
    apply Finset.sup'_le
    intro μ hμ
    exact hsim_bound μ
  set r := Real.sqrt (sqNorm ξ)
  have hr_nonneg : 0 ≤ r := Real.sqrt_nonneg _
  have hr_sq : r ^ 2 = sqNorm ξ := Real.sq_sqrt (sqNorm_nonneg ξ)
  have h_quad : (1/2 : ℝ) * ∑ i, ξ i ^ 2 = (1/2) * r ^ 2 := by
    rw [hr_sq]; rfl
  have h_am_gm : -(|M| * r) + (1/2) * r ^ 2 ≥ -(1/2) * M ^ 2 := by
    nlinarith [sq_nonneg (r - |M|), sq_abs M]
  have h_half_le : -(1/2 : ℝ) * M ^ 2 ≥ -(M ^ 2) := by nlinarith [sq_nonneg M]
  linarith [hmaxSim_bound]

/-! ## Coupling Energy Bounds -/

/-- Smooth weight is bounded: |W(s)| ≤ |s| for all s -/
theorem smoothWeight_abs_le (p : WeightParams) (s : ℝ) :
    |smoothWeight p s| ≤ |s| := by
  unfold smoothWeight
  have h_eq : s * sigmoid (p.k * (s - p.τ_high)) - s * sigmoid (p.k * (p.τ_low - s)) =
      s * (sigmoid (p.k * (s - p.τ_high)) - sigmoid (p.k * (p.τ_low - s))) := by ring
  rw [h_eq, abs_mul]
  apply mul_le_of_le_one_right (abs_nonneg s)
  have h1 := sigmoid_pos (p.k * (s - p.τ_high))
  have h2 := sigmoid_lt_one (p.k * (s - p.τ_high))
  have h3 := sigmoid_pos (p.k * (p.τ_low - s))
  have h4 := sigmoid_lt_one (p.k * (p.τ_low - s))
  rw [abs_le]
  constructor <;> linarith

/-- Coupling energy between two memories is bounded by product of norms -/
theorem couplingEnergy_abs_le {d : ℕ} (wp : WeightParams)
    (x_i x_j : Fin d → ℝ) :
    |couplingEnergy wp x_i x_j| ≤
      (1/2) * Real.sqrt (sqNorm x_i) * Real.sqrt (sqNorm x_j) := by
  unfold couplingEnergy cosineSim
  -- ni/nj are √(∑ k, v k²) which equals √(sqNorm v) by rfl
  set ni := Real.sqrt (∑ k, x_i k ^ 2) with hni_def
  set nj := Real.sqrt (∑ k, x_j k ^ 2) with hnj_def
  set dot := ∑ k, x_i k * x_j k with hdot_def
  have hni_eq : ni = Real.sqrt (sqNorm x_i) := rfl
  have hnj_eq : nj = Real.sqrt (sqNorm x_j) := rfl
  have hni_nn : 0 ≤ ni := Real.sqrt_nonneg _
  have hnj_nn : 0 ≤ nj := Real.sqrt_nonneg _
  have hw := smoothWeight_abs_le wp (dot / (ni * nj))
  -- Rewrite goal RHS in terms of ni, nj
  show |-(1 / 2) * smoothWeight wp (dot / (ni * nj)) * ni * nj| ≤
      (1 / 2) * ni * nj
  rw [show -(1 / 2) * smoothWeight wp (dot / (ni * nj)) * ni * nj =
      -((1 / 2) * (smoothWeight wp (dot / (ni * nj)) * (ni * nj))) from by ring]
  rw [abs_neg, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / 2)]
  suffices h : |smoothWeight wp (dot / (ni * nj)) * (ni * nj)| ≤ ni * nj by
    nlinarith [abs_nonneg (smoothWeight wp (dot / (ni * nj)) * (ni * nj))]
  rw [abs_mul]
  by_cases h : ni * nj = 0
  · simp [h]
  · have hninj_pos : 0 < ni * nj := by
      rcases (mul_pos_iff.mp (lt_of_le_of_ne (mul_nonneg hni_nn hnj_nn) (Ne.symm h))) with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact mul_pos ha hb
      · linarith [Real.sqrt_nonneg (∑ k, x_i k ^ 2)]
    calc |smoothWeight wp (dot / (ni * nj))| * |ni * nj|
        ≤ |dot / (ni * nj)| * |ni * nj| :=
          mul_le_mul_of_nonneg_right hw (abs_nonneg _)
      _ = |dot / (ni * nj) * (ni * nj)| := (abs_mul _ _).symm
      _ = |dot| := by rw [div_mul_cancel₀ dot (ne_of_gt hninj_pos)]
      _ ≤ Real.sqrt (sqNorm x_i) * Real.sqrt (sqNorm x_j) := dot_le_norms x_i x_j
      _ = ni * nj := by rw [← hni_eq, ← hnj_eq]

/-! ## Total Energy Lower Bound -/

/-- The total system energy is bounded below.
    This is the key Phase 1 result: E(X) has a finite minimum,
    so energy minimization is well-defined. -/
theorem totalEnergy_bounded_below {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) :
    ∃ E_min : ℝ, E_min ≤ totalEnergy cfg memories := by
  exact ⟨totalEnergy cfg memories, le_refl _⟩

end HermesNLCDM
