/-
  HermesNLCDM.EnergyGap
  =====================
  Phase 5.9b: Energy Gap Between Concentrated and Mixture Fixed Points

  Main results:
  1. entropy_lower_bound: softmax entropy >= -log N
  2. energy_fixedPoint_lower: E(xi) >= -1/2 ||xi||^2 - beta^{-1} log N at fixed points
  3. energy_gap_from_norm_gap: norm gap between two fixed points implies energy gap
  4. concentrated_sqNorm_lower: lower bound on ||xi||^2 at concentrated fixed points
  5. energy_gap_concentrated: concentrated fps have lower energy than fps with small norm

  The key insight: the energy at a fixed point is E(xi) = -1/2 ||xi||^2 + beta^{-1} sum p log p.
  The entropy term satisfies -log N <= sum p log p <= 0.
  Concentrated fixed points have ||xi||^2 close to ||x_mu||^2 (large norm, near a pattern).
  Mixture fixed points have smaller ||xi||^2 (from Jensen averaging).
  Therefore concentrated fps sit at deeper energy wells.

  Reference: Hermes memory architecture -- energy landscape analysis
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.Dynamics
import HermesNLCDM.SpuriousStates

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Entropy Bounds

  For any probability distribution p, the entropy satisfies
  -log N <= sum p_mu log p_mu <= 0.
  The upper bound (entropy nonnegativity) is already in Dynamics.lean.
  Here we prove the lower bound.
-/

/-- Entropy lower bound: for softmax distribution, sum p log p >= -log N.
    Since each p_mu <= 1, we have log(p_mu) >= log(1/N) = -log N (when p_mu >= 1/N),
    but this needs care. Instead, use Jensen: sum p log p >= log(sum p * p) ... no.
    Direct proof: sum p log p >= (sum p) * log(1/N) = -log N by concavity of log. -/
theorem entropy_lower_bound {β : ℝ} {N : ℕ} [NeZero N] (_hβ : 0 < β)
    (z : Fin N → ℝ) :
    -(Real.log N) ≤ ∑ μ, softmax β z μ * Real.log (softmax β z μ) := by
  set p := softmax β z
  have hp_pos : ∀ μ, 0 < p μ := fun μ => softmax_pos z μ
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one z
  have hN_pos : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  -- Use Jensen's inequality for log (concave): sum p log p >= (sum p) * log(sum p^2 / sum p)
  -- Actually, use direct approach: each log(p_mu) >= -log N + log(N * p_mu)
  -- and sum p_mu * log(N * p_mu) >= 0 by sum p(log(Np)) >= 0 via Gibbs against uniform.
  -- Simpler: sum p*log(p) >= sum p * log(1/N) = -log N since p is "better" than uniform.
  -- This follows from Gibbs inequality with uniform distribution.
  -- Gibbs: sum p * log(q) <= sum p * log(p) for any dist q.
  -- Take q = uniform = 1/N for all mu.
  -- sum p * log(1/N) <= sum p * log(p)
  -- sum p * log(1/N) = (sum p) * log(1/N) = 1 * log(1/N) = -log N
  -- Therefore -log N <= sum p * log(p).
  have hunif_pos : ∀ μ : Fin N, (0 : ℝ) < (1 / N : ℝ) := by
    intro _; positivity
  have hunif_sum : ∑ _μ : Fin N, (1 / N : ℝ) = 1 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [mul_one_div, div_self (ne_of_gt hN_pos)]
  have hgibbs := gibbs_inequality p (fun _ => (1 : ℝ) / N) hp_pos hp_sum hunif_pos hunif_sum
  -- hgibbs : sum p * log(1/N) <= sum p * log(p)
  -- Simplify LHS: sum p * log(1/N) = log(1/N) = -log N
  have hlhs : ∑ μ, p μ * Real.log ((1 : ℝ) / N) = -Real.log N := by
    rw [← Finset.sum_mul, hp_sum, one_mul]
    rw [Real.log_div one_ne_zero (ne_of_gt hN_pos), Real.log_one, zero_sub]
  linarith

/-- Absolute entropy bound: |sum p log p| <= log N -/
theorem entropy_abs_bound {β : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) :
    |∑ μ, softmax β z μ * Real.log (softmax β z μ)| ≤ Real.log N := by
  rw [abs_le]
  constructor
  · linarith [entropy_lower_bound hβ z]
  · have h := softmax_entropy_nonneg hβ z
    have hN_ge : (1 : ℝ) ≤ N := by exact_mod_cast NeZero.pos N
    have hlog_nn : 0 ≤ Real.log N := Real.log_nonneg hN_ge
    linarith

/-! ## Energy Bounds at Fixed Points

  From energy_at_fixedPoint and the entropy bounds:
  -1/2 ||xi||^2 - beta^{-1} log N <= E(xi) <= -1/2 ||xi||^2
-/

/-- Lower bound on energy at a fixed point:
    E(xi) >= -1/2 ||xi||^2 - beta^{-1} log N -/
theorem energy_fixedPoint_lower {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) :
    -(1/2) * sqNorm ξ - cfg.β⁻¹ * Real.log N ≤ localEnergy cfg ξ := by
  have hE := energy_at_fixedPoint cfg ξ hfp
  have hent := entropy_lower_bound cfg.β_pos (fun μ => ∑ i, cfg.patterns μ i * ξ i)
  have hβ_inv_pos : 0 < cfg.β⁻¹ := inv_pos.mpr cfg.β_pos
  -- E = -1/2 ||xi||^2 + beta^{-1} * (sum p log p)
  -- sum p log p >= -log N
  -- beta^{-1} * (sum p log p) >= beta^{-1} * (-log N) = -beta^{-1} * log N
  have h : cfg.β⁻¹ * (-Real.log N) ≤
      cfg.β⁻¹ * ∑ μ, softmax cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) μ *
        Real.log (softmax cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) μ) :=
    mul_le_mul_of_nonneg_left hent hβ_inv_pos.le
  linarith

/-! ## Energy Gap from Norm Gap

  The central algebraic result: if two fixed points have different squared norms,
  the one with larger norm has lower energy (the entropy correction is bounded).
-/

/-- ENERGY GAP THEOREM (algebraic version):
    If two fixed points satisfy ||xi_a||^2 > ||xi_b||^2 + 2 beta^{-1} log N,
    then E(xi_a) < E(xi_b).

    Proof: E(xi_a) <= -1/2 ||xi_a||^2 (from entropy nonpositivity)
           E(xi_b) >= -1/2 ||xi_b||^2 - beta^{-1} log N (from entropy lower bound)
           The gap condition ensures the upper bound beats the lower bound. -/
theorem energy_gap_from_norm_gap {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N)
    (ξ_a ξ_b : Fin d → ℝ)
    (hfp_a : isFixedPoint cfg ξ_a)
    (hfp_b : isFixedPoint cfg ξ_b)
    (hgap : sqNorm ξ_a > sqNorm ξ_b + 2 * cfg.β⁻¹ * Real.log N) :
    localEnergy cfg ξ_a < localEnergy cfg ξ_b := by
  have hE_a := energy_at_fixedPoint_le cfg ξ_a hfp_a
  have hE_b := energy_fixedPoint_lower cfg ξ_b hfp_b
  -- E(xi_a) <= -1/2 ||xi_a||^2
  -- E(xi_b) >= -1/2 ||xi_b||^2 - beta^{-1} log N
  -- Need: -1/2 ||xi_a||^2 < -1/2 ||xi_b||^2 - beta^{-1} log N
  -- i.e., 1/2 ||xi_a||^2 > 1/2 ||xi_b||^2 + beta^{-1} log N
  -- i.e., ||xi_a||^2 > ||xi_b||^2 + 2 beta^{-1} log N -- this is hgap
  linarith

/-! ## Squared Norm Expansion and Perturbation

  For bounding ||xi||^2 at a concentrated fixed point, we expand:
  ||xi||^2 = ||x_mu + (xi - x_mu)||^2
           = ||x_mu||^2 + 2 <x_mu, xi - x_mu> + ||xi - x_mu||^2

  The cross term is bounded via Cauchy-Schwarz.
-/

/-- Squared norm expansion: ||a + b||^2 = ||a||^2 + 2<a,b> + ||b||^2 -/
theorem sqNorm_add_expansion {d : ℕ} (a b : Fin d → ℝ) :
    sqNorm (fun i => a i + b i) =
      sqNorm a + 2 * ∑ i, a i * b i + sqNorm b := by
  simp only [sqNorm]
  have : ∀ i : Fin d, (a i + b i) ^ 2 = a i ^ 2 + 2 * (a i * b i) + b i ^ 2 :=
    fun i => by ring
  simp_rw [this, Finset.sum_add_distrib, Finset.mul_sum]

/-- Rewriting ||xi||^2 using xi = x_mu + (xi - x_mu):
    ||xi||^2 = ||x_mu||^2 + 2<x_mu, xi-x_mu> + ||xi-x_mu||^2 -/
theorem sqNorm_eq_pattern_plus_diff {d : ℕ} (ξ x_mu : Fin d → ℝ) :
    sqNorm ξ = sqNorm x_mu + 2 * ∑ i, x_mu i * (ξ i - x_mu i) +
      sqNorm (fun i => ξ i - x_mu i) := by
  have key : ∀ i, ξ i = x_mu i + (ξ i - x_mu i) := fun i => by ring
  have h1 : sqNorm ξ = sqNorm (fun i => x_mu i + (ξ i - x_mu i)) := by
    congr 1; ext i; exact key i
  rw [h1, sqNorm_add_expansion]

/-- Dot product bound via Cauchy-Schwarz (squared form):
    (sum a_i * b_i)^2 <= sqNorm a * sqNorm b.
    This is dot_sq_le from LocalMinima.lean, restated here for accessibility. -/
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

/-- Lower bound on ||xi||^2 at a concentrated fixed point:
    ||xi||^2 >= ||x_mu||^2 - 2 sqrt(sqNorm x_mu * sqNorm(xi-x_mu)) -/
theorem concentrated_sqNorm_lower {d : ℕ} (ξ x_mu : Fin d → ℝ) :
    sqNorm ξ ≥ sqNorm x_mu - 2 * Real.sqrt (sqNorm x_mu * sqNorm (fun i => ξ i - x_mu i)) := by
  have hexp := sqNorm_eq_pattern_plus_diff ξ x_mu
  -- ||xi||^2 = ||x||^2 + 2<x, xi-x> + ||xi-x||^2
  -- >= ||x||^2 + 2<x, xi-x>     (since ||xi-x||^2 >= 0)
  -- >= ||x||^2 - 2|<x, xi-x>|   (since a >= -|a|)
  -- >= ||x||^2 - 2 sqrt(||x||^2 ||xi-x||^2)  (by Cauchy-Schwarz)
  have hdiff_nn : 0 ≤ sqNorm (fun i => ξ i - x_mu i) := sqNorm_nonneg _
  have hCS := dot_le_norms x_mu (fun i => ξ i - x_mu i)
  -- |<x, xi-x>| <= sqrt(||x||^2) * sqrt(||xi-x||^2)
  -- = sqrt(||x||^2 * ||xi-x||^2) by sqrt_mul
  rw [← Real.sqrt_mul (sqNorm_nonneg x_mu)] at hCS
  have habs : -(2 * ∑ i, x_mu i * (ξ i - x_mu i)) ≤
      2 * |∑ i, x_mu i * (ξ i - x_mu i)| := by
    linarith [neg_abs_le (∑ i, x_mu i * (ξ i - x_mu i))]
  linarith

/-- Simplified lower bound using pattern norm bound:
    If ||x_mu||^2 <= M^2 and ||xi - x_mu||^2 <= D, then
    ||xi||^2 >= ||x_mu||^2 - 2 * M * sqrt(D)

    where M = sqrt(M^2) = |M| (assuming M >= 0). -/
theorem concentrated_sqNorm_lower_bound {d : ℕ} {M D : ℝ}
    (hM : 0 ≤ M) (_hD : 0 ≤ D)
    (ξ x_mu : Fin d → ℝ)
    (hNorm : sqNorm x_mu ≤ M ^ 2)
    (hDiff : sqNorm (fun i => ξ i - x_mu i) ≤ D) :
    sqNorm ξ ≥ sqNorm x_mu - 2 * M * Real.sqrt D := by
  have h := concentrated_sqNorm_lower ξ x_mu
  -- Need: sqrt(||x||^2 * ||xi-x||^2) <= M * sqrt(D)
  -- ||x||^2 * ||xi-x||^2 <= M^2 * D
  -- so sqrt(||x||^2 * ||xi-x||^2) <= sqrt(M^2 * D) = M * sqrt(D)
  have hprod_le : sqNorm x_mu * sqNorm (fun i => ξ i - x_mu i) ≤ M ^ 2 * D :=
    mul_le_mul hNorm hDiff (sqNorm_nonneg _) (by positivity)
  have hsqrt_le : Real.sqrt (sqNorm x_mu * sqNorm (fun i => ξ i - x_mu i)) ≤
      Real.sqrt (M ^ 2 * D) :=
    Real.sqrt_le_sqrt hprod_le
  have hsqrt_eq : Real.sqrt (M ^ 2 * D) = M * Real.sqrt D := by
    rw [show M ^ 2 * D = M * M * D from by ring]
    rw [show M * M * D = M * (M * D) from by ring]
    rw [Real.sqrt_mul hM]
    rw [Real.sqrt_mul hM]
    -- Goal: sqrt(M) * (sqrt(M) * sqrt(D)) = M * sqrt(D)
    have hss : Real.sqrt M * Real.sqrt M = M := Real.mul_self_sqrt hM
    nlinarith [Real.sqrt_nonneg D]
  linarith

/-! ## Energy Gap for Concentrated Fixed Points

  Combining everything: at a concentrated fixed point xi_c with p_mu >= 1-eps,
  the squared norm is close to ||x_mu||^2. If ||x_mu||^2 is sufficiently larger
  than ||xi_m||^2 for any other fixed point xi_m, then E(xi_c) < E(xi_m).
-/

/-- MAIN THEOREM: Energy gap for concentrated fixed points.

    If xi_c is a concentrated fixed point (p_mu >= 1-eps) with pattern norm <= M,
    and xi_m is any fixed point with ||xi_m||^2 < ||x_mu||^2 - delta,
    where delta > 4M sqrt(M eps) + 2 beta^{-1} log N, arbitrarily simplified:
    where delta > 4 M^2 sqrt(eps) + 2 beta^{-1} log N,
    then E(xi_c) < E(xi_m).

    The proof chains:
    1. concentrated_fixedPoint_near_pattern: ||xi_c - x_mu||^2 <= 4 M^2 eps
    2. concentrated_sqNorm_lower_bound: ||xi_c||^2 >= ||x_mu||^2 - 2M sqrt(4M^2 eps)
       = ||x_mu||^2 - 4 M^2 sqrt(eps)
    3. energy_at_fixedPoint_le: E(xi_c) <= -1/2 ||xi_c||^2
    4. energy_fixedPoint_lower: E(xi_m) >= -1/2 ||xi_m||^2 - beta^{-1} log N
    5. Norm gap + entropy bounds give E(xi_c) < E(xi_m)
-/
theorem energy_gap_concentrated {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N)
    (ξ_c ξ_m : Fin d → ℝ)
    (hfp_c : isFixedPoint cfg ξ_c)
    (hfp_m : isFixedPoint cfg ξ_m)
    (μ : Fin N) {M ε : ℝ} (hM : 0 ≤ M) (hε_nn : 0 ≤ ε)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hconc : softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ_c i) μ ≥ 1 - ε)
    (hgap : sqNorm (cfg.patterns μ) - 4 * M ^ 2 * Real.sqrt ε >
        sqNorm ξ_m + 2 * cfg.β⁻¹ * Real.log N) :
    localEnergy cfg ξ_c < localEnergy cfg ξ_m := by
  -- Step 1: ||xi_c - x_mu||^2 <= 4M^2 eps
  have hDiff := concentrated_fixedPoint_near_pattern cfg ξ_c hfp_c μ hε_nn hNorm hconc
  -- Step 2: ||xi_c||^2 >= ||x_mu||^2 - 4M^2 sqrt(eps)
  -- Apply concentrated_sqNorm_lower_bound with D = 4*M^2*eps
  have hD_nn : 0 ≤ 4 * M ^ 2 * ε := by positivity
  have hSqNorm_lower := concentrated_sqNorm_lower_bound hM hD_nn ξ_c (cfg.patterns μ) (hNorm μ) hDiff
  -- Simplify sqrt(4*M^2*eps) = 2*M*sqrt(eps)
  have hsqrt_simp : Real.sqrt (4 * M ^ 2 * ε) = 2 * M * Real.sqrt ε := by
    rw [show 4 * M ^ 2 * ε = (2 * M) * ((2 * M) * ε) from by ring]
    rw [Real.sqrt_mul (by linarith : 0 ≤ 2 * M)]
    rw [Real.sqrt_mul (by linarith : 0 ≤ 2 * M)]
    -- Goal: sqrt(2*M) * (sqrt(2*M) * sqrt(eps)) = 2*M*sqrt(eps)
    have h2M := Real.mul_self_sqrt (by linarith : 0 ≤ 2 * M)
    nlinarith [Real.sqrt_nonneg ε]
  rw [hsqrt_simp] at hSqNorm_lower
  -- So: ||xi_c||^2 >= ||x_mu||^2 - 2*M*(2*M*sqrt(eps)) = ||x_mu||^2 - 4*M^2*sqrt(eps)
  -- Step 3: Upper bound E(xi_c) <= -1/2 ||xi_c||^2
  have hE_c := energy_at_fixedPoint_le cfg ξ_c hfp_c
  -- Step 4: Lower bound E(xi_m) >= -1/2 ||xi_m||^2 - beta^{-1} log N
  have hE_m := energy_fixedPoint_lower cfg ξ_m hfp_m
  -- Step 5: Chain the bounds
  -- E(xi_c) <= -1/2 ||xi_c||^2
  --         <= -1/2 (||x_mu||^2 - 4M^2 sqrt(eps))
  --         = -1/2 ||x_mu||^2 + 2M^2 sqrt(eps)
  -- E(xi_m) >= -1/2 ||xi_m||^2 - beta^{-1} log N
  -- Need: -1/2 ||x_mu||^2 + 2M^2 sqrt(eps) < -1/2 ||xi_m||^2 - beta^{-1} log N
  -- i.e., 1/2 ||x_mu||^2 - 2M^2 sqrt(eps) > 1/2 ||xi_m||^2 + beta^{-1} log N
  -- i.e., ||x_mu||^2 - 4M^2 sqrt(eps) > ||xi_m||^2 + 2 beta^{-1} log N
  -- This is exactly hgap.
  linarith

/-! ## Uniform Fixed Point Energy

  When all softmax weights are equal (p_mu = 1/N for all mu), the entropy
  reaches its maximum: sum p log p = -log N. This gives the energy at a
  uniform fixed point.
-/

/-- At a fixed point where all softmax weights equal 1/N, the entropy term
    equals -log N, giving E(xi) = -1/2 ||xi||^2 - beta^{-1} log N. -/
theorem energy_at_uniform_fixedPoint {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ)
    (hunif : ∀ μ, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ = 1 / N) :
    localEnergy cfg ξ = -(1/2) * sqNorm ξ - cfg.β⁻¹ * Real.log N := by
  -- Use energy_at_fixedPoint but expand the let bindings via show
  set sim := fun μ => ∑ i, cfg.patterns μ i * ξ i
  set p := softmax cfg.β sim
  have hN_pos : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  -- The energy identity with explicit p
  have hE : localEnergy cfg ξ = -(1/2) * sqNorm ξ +
      cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ) :=
    energy_at_fixedPoint cfg ξ hfp
  -- Compute sum p log p when p = 1/N
  have hent : ∑ μ, p μ * Real.log (p μ) = -Real.log N := by
    simp_rw [show ∀ μ, p μ = 1 / (N : ℝ) from hunif]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [Real.log_div one_ne_zero (ne_of_gt hN_pos), Real.log_one, zero_sub]
    field_simp
  rw [hent] at hE; linarith

end HermesNLCDM
