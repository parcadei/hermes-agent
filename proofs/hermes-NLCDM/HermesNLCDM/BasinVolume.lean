/-
  HermesNLCDM.BasinVolume
  =======================
  Phase 5D.1: Hessian Comparison via Softmax Concentration

  Main results:
  1. herfindahl_concentrated: When p_mu >= 1-eps with eps <= 1,
     H(p) = sum p^2 >= (1-eps)^2.
  2. herfindahl_uniform: When p_mu = 1/N for all mu, H(p) = 1/N.
  3. herfindahl_ge_uniform: For ANY probability distribution, H(p) >= 1/N.
  4. effective_dim_concentrated_le: Effective number 1/H(p) <= 1/(1-eps)^2
     at concentrated fixed points (approximately 1 direction).
  5. effective_dim_uniform: Effective number = N at uniform fixed points
     (N equivalent directions).
  6. covariance_trace_eq: The trace of the softmax covariance matrix
     equals 1 - H(p), connecting Herfindahl to Hessian curvature.
  7. basin_curvature_comparison: MAIN THEOREM. Concentrated fps have
     higher Herfindahl (steeper curvature, narrower basins) than
     uniform fps (flatter curvature, wider basins).

  The key insight: at a fixed point xi = sum p_mu x_mu, the curvature
  of the energy landscape is governed by the softmax covariance matrix
  Cov_p[x] = Diag(p) - p p^T. The trace of this matrix equals
  1 - sum p^2 = 1 - H(p). High concentration (H near 1) means the
  covariance is nearly zero (steep, narrow well). Low concentration
  (H near 1/N) means the covariance is large (flat, wide basin).

  Reference: Hermes memory architecture -- basin volume analysis
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.Dynamics
import HermesNLCDM.LocalMinima
import HermesNLCDM.SpuriousStates

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Herfindahl Index -/

/-- Herfindahl index: sum of squared probabilities -/
def herfindahl {N : ℕ} (p : Fin N → ℝ) : ℝ := ∑ μ, p μ ^ 2

/-- Herfindahl index is nonneg -/
theorem herfindahl_nonneg {N : ℕ} (p : Fin N → ℝ) : 0 ≤ herfindahl p :=
  Finset.sum_nonneg fun μ _ => sq_nonneg (p μ)

/-! ## Herfindahl at Concentrated Distribution -/

/-- At a concentrated distribution (p_mu >= 1-eps with eps <= 1),
    the Herfindahl index is at least (1-eps)^2. The condition eps <= 1
    ensures 1-eps >= 0, which holds for all meaningful concentration
    parameters (softmax entries are nonneg). -/
theorem herfindahl_concentrated {N : ℕ} (p : Fin N → ℝ)
    (μ : Fin N) {ε : ℝ} (hε : ε ≤ 1) (hconc : p μ ≥ 1 - ε) :
    herfindahl p ≥ (1 - ε) ^ 2 := by
  unfold herfindahl
  calc ∑ ν, p ν ^ 2
      ≥ p μ ^ 2 :=
        Finset.single_le_sum (fun ν _ => sq_nonneg (p ν)) (Finset.mem_univ μ)
    _ ≥ (1 - ε) ^ 2 := by
        rw [sq, sq]
        exact mul_self_le_mul_self (by linarith) hconc

/-! ## Herfindahl at Uniform Distribution -/

/-- At the uniform distribution (p_mu = 1/N for all mu),
    the Herfindahl index equals 1/N. -/
theorem herfindahl_uniform {N : ℕ} [NeZero N] (p : Fin N → ℝ)
    (hunif : ∀ μ, p μ = 1 / N) :
    herfindahl p = 1 / N := by
  unfold herfindahl
  simp_rw [hunif]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hN_ne : (N : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr (NeZero.pos N))
  field_simp

/-! ## Herfindahl: Any Distribution >= Uniform -/

/-- For any probability distribution, H(p) >= 1/N.
    Proof: By Cauchy-Schwarz, (sum p)^2 <= N * sum p^2,
    so 1 <= N * H(p), giving H(p) >= 1/N. -/
theorem herfindahl_ge_uniform {N : ℕ} [NeZero N] (p : Fin N → ℝ)
    (_hp_nn : ∀ μ, 0 ≤ p μ) (hp_sum : ∑ μ, p μ = 1) :
    1 / (N : ℝ) ≤ herfindahl p := by
  unfold herfindahl
  have hN_pos : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  rw [div_le_iff₀ hN_pos]
  have hCS := sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ : Fin N => (1 : ℝ)) p
  simp only [one_mul, one_pow, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one] at hCS
  rw [hp_sum, one_pow] at hCS
  linarith

/-- Combined: concentrated distributions always have H >= 1/N. -/
theorem herfindahl_concentrated_ge_uniform {N : ℕ} [NeZero N] (p : Fin N → ℝ)
    (hp_nn : ∀ μ, 0 ≤ p μ) (hp_sum : ∑ μ, p μ = 1) :
    herfindahl p ≥ 1 / N :=
  (herfindahl_ge_uniform p hp_nn hp_sum).ge

/-! ## Covariance Trace and Curvature -/

/-- The trace of the softmax covariance matrix equals 1 - H(p):
    sum p_mu (1 - p_mu) = 1 - sum p_mu^2 = 1 - H(p). -/
theorem covariance_trace_eq {N : ℕ} (p : Fin N → ℝ)
    (hp_sum : ∑ μ, p μ = 1) :
    ∑ μ, p μ * (1 - p μ) = 1 - herfindahl p := by
  unfold herfindahl
  simp_rw [mul_sub, mul_one]
  rw [Finset.sum_sub_distrib, hp_sum]
  congr 1; congr 1; ext μ; ring

/-- At a concentrated distribution with eps <= 1, the covariance trace
    is small: sum p_mu(1 - p_mu) <= 2eps. -/
theorem covariance_trace_small_at_concentrated {N : ℕ} (p : Fin N → ℝ)
    (hp_sum : ∑ μ, p μ = 1) (μ : Fin N) {ε : ℝ}
    (hε : ε ≤ 1) (hconc : p μ ≥ 1 - ε) :
    ∑ ν, p ν * (1 - p ν) ≤ 2 * ε := by
  rw [covariance_trace_eq p hp_sum]
  have hH := herfindahl_concentrated p μ hε hconc
  nlinarith [sq_nonneg ε]

/-- At the uniform distribution, the covariance trace is maximal:
    sum p_mu(1 - p_mu) = 1 - 1/N. -/
theorem covariance_trace_at_uniform {N : ℕ} [NeZero N] (p : Fin N → ℝ)
    (hp_sum : ∑ μ, p μ = 1) (hunif : ∀ μ, p μ = 1 / N) :
    ∑ μ, p μ * (1 - p μ) = 1 - 1 / (N : ℝ) := by
  rw [covariance_trace_eq p hp_sum, herfindahl_uniform p hunif]

/-! ## Herfindahl Upper Bound -/

/-- Herfindahl index is at most 1 for any probability distribution. -/
theorem herfindahl_le_one {N : ℕ} (p : Fin N → ℝ)
    (hp_nn : ∀ μ, 0 ≤ p μ) (hp_sum : ∑ μ, p μ = 1) :
    herfindahl p ≤ 1 := by
  unfold herfindahl
  calc ∑ μ, p μ ^ 2
      ≤ ∑ μ, p μ := Finset.sum_le_sum fun μ _ => by
        rw [sq]
        exact mul_le_of_le_one_left (hp_nn μ)
          (hp_sum ▸ Finset.single_le_sum (fun i _ => hp_nn i) (Finset.mem_univ μ))
    _ = 1 := hp_sum

/-! ## Variance Bound -/

/-- Variance is bounded by the second moment -/
theorem variance_le_second_moment {N : ℕ}
    (p f : Fin N → ℝ) :
    (∑ μ, p μ * f μ ^ 2) - (∑ μ, p μ * f μ) ^ 2 ≤
      ∑ μ, p μ * f μ ^ 2 := by
  linarith [sq_nonneg (∑ μ, p μ * f μ)]

/-! ## Effective Curvature Dimension -/

/-- Effective curvature dimension at a concentrated fixed point is small.
    If p_mu >= 1-eps with eps < 1, then 1/H(p) <= 1/(1-eps)^2. -/
theorem effective_dim_concentrated_le {N : ℕ}
    (p : Fin N → ℝ)
    (μ : Fin N) {ε : ℝ} (hε_lt : ε < 1)
    (hconc : p μ ≥ 1 - ε) :
    (herfindahl p)⁻¹ ≤ ((1 - ε) ^ 2)⁻¹ := by
  have hH := herfindahl_concentrated p μ (le_of_lt hε_lt) hconc
  have h1mε_pos : 0 < 1 - ε := by linarith
  have h1mε_sq_pos : 0 < (1 - ε) ^ 2 := sq_pos_of_pos h1mε_pos
  exact inv_anti₀ h1mε_sq_pos hH

/-- Effective curvature dimension at a uniform fixed point is exactly N. -/
theorem effective_dim_uniform {N : ℕ} [NeZero N]
    (p : Fin N → ℝ) (hunif : ∀ μ, p μ = 1 / N) :
    (herfindahl p)⁻¹ = (N : ℝ) := by
  rw [herfindahl_uniform p hunif, one_div, inv_inv]

/-! ## Basin Width Comparison Theorem -/

/-- MAIN THEOREM: Basin width comparison.

    At a concentrated fp (p_mu >= 1-eps), the Herfindahl index H(p_c)
    is at least (1-eps)^2. At a uniform fp, H(p_u) = 1/N.
    When (1-eps)^2 > 1/N, concentrated fps have strictly higher Herfindahl
    (steeper curvature, narrower basins) than uniform fps
    (flatter curvature, wider basins).

    Equivalently, the effective dimension 1/H(p_c) < 1/H(p_u) = N.

    This is the formal version of: "mixture states have wider
    basins of attraction than concentrated states." -/
theorem basin_curvature_comparison {N : ℕ} [NeZero N]
    {ε : ℝ} (hε_lt : ε < 1)
    (p_c p_u : Fin N → ℝ)
    (μ : Fin N) (hconc : p_c μ ≥ 1 - ε)
    (hunif : ∀ ν, p_u ν = 1 / N)
    (hsep : (1 - ε) ^ 2 > 1 / (N : ℝ)) :
    herfindahl p_c > herfindahl p_u
    ∧ (herfindahl p_c)⁻¹ < (herfindahl p_u)⁻¹ := by
  have hH_c := herfindahl_concentrated p_c μ (le_of_lt hε_lt) hconc
  have hH_u := herfindahl_uniform p_u hunif
  have h1mε_pos : 0 < 1 - ε := by linarith
  have h1mε_sq_pos : 0 < (1 - ε) ^ 2 := sq_pos_of_pos h1mε_pos
  have hN_pos : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  have h1N_pos : (0 : ℝ) < 1 / N := div_pos one_pos hN_pos
  constructor
  · rw [hH_u]; linarith
  · have hH_u_pos : 0 < herfindahl p_u := by rw [hH_u]; exact h1N_pos
    have hH_c_gt_u : herfindahl p_c > herfindahl p_u := by rw [hH_u]; linarith
    exact inv_strictAnti₀ hH_u_pos hH_c_gt_u

/-! ## Curvature at Fixed Points: Softmax Formulation -/

/-- At a fixed point with concentrated softmax (eps <= 1), the covariance
    trace is at most 2eps (small perturbation from identity Hessian). -/
theorem curvature_trace_at_concentrated_fp {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (μ : Fin N) {ε : ℝ} (hε : ε ≤ 1)
    (hconc : softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ ≥ 1 - ε) :
    let sim := fun ν => ∑ i, cfg.patterns ν i * ξ i
    let p := softmax cfg.β sim
    ∑ ν, p ν * (1 - p ν) ≤ 2 * ε := by
  intro sim p
  exact covariance_trace_small_at_concentrated p (softmax_sum_one sim) μ hε hconc

/-- Squared dot product bounded by product of squared norms (Cauchy-Schwarz). -/
private theorem dot_sq_le_local {d : ℕ} (u v : Fin d → ℝ) :
    (∑ i, u i * v i) ^ 2 ≤ sqNorm u * sqNorm v := by
  have h := dot_le_norms u v
  have hsq := mul_self_le_mul_self (abs_nonneg _) h
  have lhs_eq : |∑ i, u i * v i| * |∑ i, u i * v i| = (∑ i, u i * v i) ^ 2 := by
    rw [← sq, sq_abs]
  have h1 := Real.mul_self_sqrt (sqNorm_nonneg u)
  have h2 := Real.mul_self_sqrt (sqNorm_nonneg v)
  have rhs_eq : (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) *
      (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) = sqNorm u * sqNorm v := by
    calc _ = (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm u)) *
          (Real.sqrt (sqNorm v) * Real.sqrt (sqNorm v)) := by ring
      _ = sqNorm u * sqNorm v := by rw [h1, h2]
  linarith

/-- The variance Var_p[x^T v] is bounded at any fixed point.
    Var_p[x^T v] <= M^2 * ||v||^2 where M bounds pattern norms. -/
theorem variance_bounded_at_fp {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (v : Fin d → ℝ) {M : ℝ}
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2) :
    let sim := fun ν => ∑ i, cfg.patterns ν i * ξ i
    let p := softmax cfg.β sim
    let dot_v := fun ν => ∑ i, cfg.patterns ν i * v i
    (∑ ν, p ν * dot_v ν ^ 2) - (∑ ν, p ν * dot_v ν) ^ 2 ≤
      M ^ 2 * sqNorm v := by
  intro sim p dot_v
  have hp_nn : ∀ ν, 0 ≤ p ν := fun ν => softmax_nonneg sim ν
  have hp_sum : ∑ ν, p ν = 1 := softmax_sum_one sim
  have hvar_le := variance_le_second_moment p dot_v
  have hdot_bound : ∀ ν, dot_v ν ^ 2 ≤ M ^ 2 * sqNorm v := by
    intro ν
    calc dot_v ν ^ 2
        ≤ sqNorm (cfg.patterns ν) * sqNorm v := dot_sq_le_local _ _
      _ ≤ M ^ 2 * sqNorm v :=
          mul_le_mul_of_nonneg_right (hNorm ν) (sqNorm_nonneg v)
  have hEf2_le : ∑ ν, p ν * dot_v ν ^ 2 ≤ M ^ 2 * sqNorm v := by
    calc ∑ ν, p ν * dot_v ν ^ 2
        ≤ ∑ ν, p ν * (M ^ 2 * sqNorm v) := Finset.sum_le_sum fun ν _ =>
          mul_le_mul_of_nonneg_left (hdot_bound ν) (hp_nn ν)
      _ = M ^ 2 * sqNorm v := by rw [← Finset.sum_mul, hp_sum, one_mul]
  linarith

end HermesNLCDM
