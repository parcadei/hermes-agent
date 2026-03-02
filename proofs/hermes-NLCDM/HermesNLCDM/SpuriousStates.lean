/-
  HermesNLCDM.SpuriousStates
  ==========================
  Phase 5.9: Spurious State Characterization

  Main results:
  1. fixedPoint_expansion: fixed points are softmax-weighted pattern combinations
  2. fixedPoint_similarity_identity: Σ p_μ (x_μ · ξ) = ‖ξ‖² at fixed points
  3. energy_at_fixedPoint: localEnergy(ξ) = -½‖ξ‖² + β⁻¹ Σ p log p
  4. fixedPoint_sqNorm_le: ‖ξ‖² ≤ Σ p_μ ‖x_μ‖² (Jensen)
  5. fixedPoint_sqNorm_bound: ‖ξ‖² ≤ M² under pattern norm bounds
  6. concentrated_fixedPoint_near_pattern: p_μ ≥ 1-ε → ‖ξ - x_μ‖² ≤ 4M²ε

  The key insight: every fixed point is a convex combination of stored patterns,
  and the energy at a fixed point decomposes cleanly into a norm term and an
  entropy term. Concentrated fixed points (near stored patterns) have high
  norm and low entropy → low energy. Spurious mixture states have reduced
  norm (from non-collinear averaging) and higher entropy → higher energy.

  The energy identity localEnergy(ξ) = -½‖ξ‖² + β⁻¹ Σ p log p makes
  the energy landscape completely transparent at fixed points.

  Reference: Hermes memory architecture — spurious state analysis
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.Dynamics

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Fixed Point Structure

  Every fixed point ξ of the Hopfield update is a softmax-weighted
  convex combination of stored patterns: ξ = Σ_μ p_μ x_μ where
  p = softmax(β, X^T ξ). This is immediate from the definition.
-/

/-- Fixed points are softmax-weighted combinations of stored patterns:
    ξ_k = Σ_μ softmax(β, X^T ξ)_μ · x_{μk} -/
theorem fixedPoint_expansion {d N : ℕ} (cfg : SystemConfig d N)
    (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) (k : Fin d) :
    ξ k = ∑ μ, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ *
      cfg.patterns μ k := by
  have h := congr_fun hfp k
  unfold hopfieldUpdate at h
  exact h.symm

/-! ## Similarity Identity at Fixed Points

  At a fixed point ξ = Σ p_μ x_μ, the weighted similarity sum equals
  the squared norm: Σ_μ p_μ (x_μ · ξ) = ‖ξ‖².

  Proof: Σ_μ p_μ (Σ_i x_{μi} ξ_i) = Σ_i ξ_i (Σ_μ p_μ x_{μi}) = Σ_i ξ_i² = ‖ξ‖²
  where the middle step uses ξ_i = Σ_μ p_μ x_{μi} (fixed point condition).
-/

/-- At a fixed point, the softmax-weighted similarity equals the squared norm:
    Σ_μ p_μ (x_μ · ξ) = ‖ξ‖² -/
theorem fixedPoint_similarity_identity {d N : ℕ} (cfg : SystemConfig d N)
    (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) :
    let sim := fun μ => ∑ i, cfg.patterns μ i * ξ i
    let p := softmax cfg.β sim
    ∑ μ, p μ * sim μ = sqNorm ξ := by
  intro sim p
  -- Unfold sim in the goal so simp_rw can see the inner sum
  show ∑ μ, p μ * (∑ i, cfg.patterns μ i * ξ i) = sqNorm ξ
  -- Swap sums: Σ_μ p_μ (Σ_i x_{μi} ξ_i) = Σ_i ξ_i (Σ_μ p_μ x_{μi})
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  -- Now goal: Σ_i Σ_μ p_μ * (x_{μi} * ξ_i) = Σ_i ξ_i²
  congr 1; ext i
  -- Factor out ξ_i: Σ_μ p_μ * (x_{μi} * ξ_i) = ξ_i * Σ_μ p_μ * x_{μi}
  simp_rw [show ∀ μ, p μ * (cfg.patterns μ i * ξ i) =
    ξ i * (p μ * cfg.patterns μ i) from fun μ => by ring]
  rw [← Finset.mul_sum]
  -- By fixed point: Σ_μ p_μ * x_{μi} = ξ_i
  have hfp_i : ∑ μ, p μ * cfg.patterns μ i = ξ i := by
    have h := congr_fun hfp i; unfold hopfieldUpdate at h; exact h
  rw [← hfp_i]; ring

/-! ## Energy Identity at Fixed Points

  Using the Fenchel-Young equality lse(β, z) = Σ p_μ z_μ - β⁻¹ Σ p_μ log(p_μ)
  and the similarity identity Σ p_μ z_μ = ‖ξ‖², we get:

  localEnergy(ξ) = -lse(β, X^T ξ) + ½‖ξ‖²
                 = -(‖ξ‖² - β⁻¹ Σ p log p) + ½‖ξ‖²
                 = -½‖ξ‖² + β⁻¹ Σ p log p

  Since Σ p log p ≤ 0 (entropy nonnegativity), we get:
  localEnergy(ξ) ≤ -½‖ξ‖²
-/

/-- PHASE 5.9 KEY THEOREM: Energy at a fixed point has a clean decomposition:
    localEnergy(ξ) = -½‖ξ‖² + β⁻¹ Σ p_μ log(p_μ)
    The energy is determined by the norm and the softmax entropy. -/
theorem energy_at_fixedPoint {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) :
    let sim := fun μ => ∑ i, cfg.patterns μ i * ξ i
    let p := softmax cfg.β sim
    localEnergy cfg ξ = -(1/2) * sqNorm ξ +
      cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ) := by
  intro sim p
  -- Use Fenchel-Young equality: lse = Σ p z - β⁻¹ Σ p log p
  have hFY := lse_eq_softmax_sum cfg.β_pos sim
  -- Use similarity identity: Σ p z = ‖ξ‖²
  have hSI := fixedPoint_similarity_identity cfg ξ hfp
  -- localEnergy = -lse + ½‖ξ‖²
  show -logSumExp cfg.β sim + (1/2) * sqNorm ξ =
    -(1/2) * sqNorm ξ + cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ)
  -- Substitute FY and SI
  rw [hFY]
  change -(∑ μ, p μ * sim μ - cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ)) +
    (1/2) * sqNorm ξ =
    -(1/2) * sqNorm ξ + cfg.β⁻¹ * ∑ μ, p μ * Real.log (p μ)
  rw [hSI]; ring

/-- At a fixed point, localEnergy ≤ -½‖ξ‖² (entropy term is non-positive) -/
theorem energy_at_fixedPoint_le {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) :
    localEnergy cfg ξ ≤ -(1/2) * sqNorm ξ := by
  have hE := energy_at_fixedPoint cfg ξ hfp
  have hent := softmax_entropy_nonneg cfg.β_pos
    (fun μ => ∑ i, cfg.patterns μ i * ξ i)
  have hβ_inv_pos : 0 < cfg.β⁻¹ := inv_pos.mpr cfg.β_pos
  -- β⁻¹ · (non-positive) ≤ 0
  have h := mul_nonpos_of_nonneg_of_nonpos hβ_inv_pos.le hent
  linarith

/-! ## Private Helpers (copies of private theorems from LocalMinima.lean) -/

/-- Weighted Cauchy-Schwarz: (Σ w_i a_i)² ≤ (Σ w_i)(Σ w_i a_i²) -/
private theorem weighted_sq_le_local {N : ℕ} (w a : Fin N → ℝ)
    (hw : ∀ i, 0 ≤ w i) :
    (∑ i, w i * a i) ^ 2 ≤ (∑ i, w i) * (∑ i, w i * a i ^ 2) := by
  set u : Fin N → ℝ := fun i => Real.sqrt (w i)
  set v : Fin N → ℝ := fun i => Real.sqrt (w i) * a i
  have huv : ∑ i, u i * v i = ∑ i, w i * a i := by
    congr 1; ext i; show Real.sqrt (w i) * (Real.sqrt (w i) * a i) = w i * a i
    rw [← mul_assoc, Real.mul_self_sqrt (hw i)]
  have huu : sqNorm u = ∑ i, w i := by
    simp only [sqNorm]; congr 1; ext i; show Real.sqrt (w i) ^ 2 = w i
    rw [sq, Real.mul_self_sqrt (hw i)]
  have hvv : sqNorm v = ∑ i, w i * a i ^ 2 := by
    simp only [sqNorm]; congr 1; ext i
    show (Real.sqrt (w i) * a i) ^ 2 = w i * a i ^ 2
    rw [mul_pow, sq (Real.sqrt (w i)), Real.mul_self_sqrt (hw i)]
  have h := dot_le_norms u v
  rw [huv] at h
  have h1 := mul_self_le_mul_self (abs_nonneg _) h
  have lhs : |∑ i, w i * a i| * |∑ i, w i * a i| = (∑ i, w i * a i) ^ 2 := by
    rw [← sq, sq_abs]
  have rhs : Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v) *
      (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) =
      (∑ i, w i) * (∑ i, w i * a i ^ 2) := by
    rw [show Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v) *
        (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) =
        (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm u)) *
        (Real.sqrt (sqNorm v) * Real.sqrt (sqNorm v)) from by ring,
      Real.mul_self_sqrt (sqNorm_nonneg u),
      Real.mul_self_sqrt (sqNorm_nonneg v), huu, hvv]
  rw [lhs, rhs] at h1; exact h1

/-- ‖a - b‖² ≤ 2(‖a‖² + ‖b‖²), so ≤ 4M² when both norms ≤ M² -/
private theorem sqNorm_sub_le_local {d : ℕ} {M : ℝ} (a b : Fin d → ℝ)
    (ha : sqNorm a ≤ M ^ 2) (hb : sqNorm b ≤ M ^ 2) :
    sqNorm (fun i => a i - b i) ≤ 4 * M ^ 2 := by
  have h : sqNorm (fun i => a i - b i) ≤ 2 * sqNorm a + 2 * sqNorm b := by
    simp only [sqNorm]
    have pw : ∀ i : Fin d, (a i - b i) ^ 2 ≤ 2 * a i ^ 2 + 2 * b i ^ 2 :=
      fun i => by nlinarith [sq_nonneg (a i + b i)]
    calc ∑ i, (a i - b i) ^ 2
        ≤ ∑ i, (2 * a i ^ 2 + 2 * b i ^ 2) :=
          Finset.sum_le_sum (fun i _ => pw i)
      _ = (∑ i, 2 * a i ^ 2) + (∑ i, 2 * b i ^ 2) :=
          Finset.sum_add_distrib
      _ = 2 * ∑ i, a i ^ 2 + 2 * ∑ i, b i ^ 2 := by
          rw [Finset.mul_sum, Finset.mul_sum]
  linarith

/-! ## Fixed Point Norm Bound

  Since ξ = Σ p_μ x_μ is a convex combination, by Jensen's inequality
  (via weighted Cauchy-Schwarz) we get ‖ξ‖² ≤ Σ p_μ ‖x_μ‖².
-/

/-- Jensen's inequality for fixed point norms: ‖ξ‖² ≤ Σ p_μ ‖x_μ‖² -/
theorem fixedPoint_sqNorm_le {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ) :
    let sim := fun μ => ∑ i, cfg.patterns μ i * ξ i
    let p := softmax cfg.β sim
    sqNorm ξ ≤ ∑ μ, p μ * sqNorm (cfg.patterns μ) := by
  intro sim p
  -- ‖ξ‖² = Σ_k ξ_k² = Σ_k (Σ_μ p_μ x_{μk})²
  -- By Jensen (convexity of t²): (Σ p_μ x_{μk})² ≤ Σ p_μ x_{μk}²
  -- Sum over k: Σ_k (Σ_μ p_μ x_{μk})² ≤ Σ_k Σ_μ p_μ x_{μk}² = Σ_μ p_μ ‖x_μ‖²
  show sqNorm ξ ≤ ∑ μ, p μ * sqNorm (cfg.patterns μ)
  have hp_nn : ∀ μ, 0 ≤ p μ := fun μ => softmax_nonneg sim μ
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one sim
  -- Per-coordinate Jensen: ξ_k² ≤ Σ_μ p_μ x_{μk}²
  have hcoord : ∀ k, ξ k ^ 2 ≤ ∑ μ, p μ * cfg.patterns μ k ^ 2 := by
    intro k
    -- Derive fixed point equation in p-terms
    have hfp_k : ξ k = ∑ μ, p μ * cfg.patterns μ k := by
      have h := congr_fun hfp k; unfold hopfieldUpdate at h; exact h.symm
    rw [hfp_k]
    -- (Σ p_μ x_{μk})² ≤ (Σ p_μ)(Σ p_μ x_{μk}²) by weighted Cauchy-Schwarz
    have h := weighted_sq_le_local p (fun μ => cfg.patterns μ k) hp_nn
    rw [hp_sum, one_mul] at h; exact h
  -- Sum over k: Σ_k ξ_k² ≤ Σ_k Σ_μ p_μ x_{μk}²
  calc sqNorm ξ = ∑ k, ξ k ^ 2 := rfl
    _ ≤ ∑ k, ∑ μ, p μ * cfg.patterns μ k ^ 2 :=
        Finset.sum_le_sum (fun k _ => hcoord k)
    _ = ∑ μ, p μ * ∑ k, cfg.patterns μ k ^ 2 := by
        rw [Finset.sum_comm]; congr 1; ext μ; rw [Finset.mul_sum]
    _ = ∑ μ, p μ * sqNorm (cfg.patterns μ) := rfl

/-- Fixed point norm is bounded by the maximum pattern norm:
    ‖ξ‖² ≤ M² when all ‖x_μ‖² ≤ M² -/
theorem fixedPoint_sqNorm_bound {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ)
    {M : ℝ} (hNorm : ∀ μ, sqNorm (cfg.patterns μ) ≤ M ^ 2) :
    sqNorm ξ ≤ M ^ 2 := by
  have hJ := fixedPoint_sqNorm_le cfg ξ hfp
  have sim := fun μ => ∑ i, cfg.patterns μ i * ξ i
  have p := softmax cfg.β sim
  have hp_nn : ∀ μ, 0 ≤ softmax cfg.β
    (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ :=
    fun μ => softmax_nonneg _ μ
  have hp_sum : ∑ μ, softmax cfg.β
    (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ = 1 := softmax_sum_one _
  calc sqNorm ξ
      ≤ ∑ μ, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ *
          sqNorm (cfg.patterns μ) := hJ
    _ ≤ ∑ μ, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ *
          M ^ 2 :=
        Finset.sum_le_sum (fun μ _ =>
          mul_le_mul_of_nonneg_left (hNorm μ) (hp_nn μ))
    _ = M ^ 2 := by rw [← Finset.sum_mul, hp_sum, one_mul]

/-! ## Concentrated Fixed Points are Near Stored Patterns

  If p_μ ≥ 1-ε at a fixed point, the fixed point is close to pattern μ.
  ‖ξ - x_μ‖² ≤ 4M²ε, using the same weighted Cauchy-Schwarz technique
  as the gradient bound in LocalMinima.lean.
-/

/-- Concentrated fixed points are near the dominant pattern:
    p_μ ≥ 1-ε implies ‖ξ - x_μ‖² ≤ 4M²ε -/
theorem concentrated_fixedPoint_near_pattern {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ) (hfp : isFixedPoint cfg ξ)
    (μ : Fin N) {M ε : ℝ} (hε_nn : 0 ≤ ε)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hconc : softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ i) μ ≥ 1 - ε) :
    sqNorm (fun k => ξ k - cfg.patterns μ k) ≤ 4 * M ^ 2 * ε := by
  set sim := fun ν => ∑ i, cfg.patterns ν i * ξ i
  set p := softmax cfg.β sim
  have hp_nn : ∀ ν, 0 ≤ p ν := fun ν => softmax_nonneg sim ν
  have hp_sum : ∑ ν, p ν = 1 := softmax_sum_one sim
  have hp_tail : 1 - p μ ≤ ε := by linarith
  have hp_le_one : p μ ≤ 1 := by
    calc p μ ≤ ∑ ν, p ν :=
          Finset.single_le_sum (f := p) (fun i _ => hp_nn i) (Finset.mem_univ μ)
      _ = 1 := hp_sum
  have hp_tail_nn : 0 ≤ 1 - p μ := by linarith
  -- ξ_k - x_{μk} = Σ_ν p_ν (x_{νk} - x_{μk}) since ξ = Σ p x and Σ p = 1
  have hdiff_eq : ∀ k, ξ k - cfg.patterns μ k =
      ∑ ν, p ν * (cfg.patterns ν k - cfg.patterns μ k) := by
    intro k
    -- Derive fixed point equation in p-terms
    have hfp_k : ξ k = ∑ ν, p ν * cfg.patterns ν k := by
      have h := congr_fun hfp k; unfold hopfieldUpdate at h; exact h.symm
    rw [hfp_k]
    simp_rw [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hp_sum, one_mul]
  -- Set w_ν = p_ν for ν ≠ μ, w_μ = 0 (μ-th term vanishes)
  set w : Fin N → ℝ := fun ν => if ν = μ then 0 else p ν
  have hw_nn : ∀ ν, 0 ≤ w ν := by
    intro ν; simp only [w]; split_ifs <;> linarith [hp_nn ν]
  have hw_sum : ∑ ν, w ν = 1 - p μ := by
    have h1 : w μ = 0 := if_pos rfl
    have h2 : ∀ ν, ν ≠ μ → w ν = p ν := fun ν hν => if_neg hν
    have h3 := Finset.add_sum_erase Finset.univ w (Finset.mem_univ μ)
    have h4 := Finset.add_sum_erase Finset.univ p (Finset.mem_univ μ)
    have h5 : (Finset.univ.erase μ).sum w = (Finset.univ.erase μ).sum p :=
      Finset.sum_congr rfl fun ν hν => h2 ν (Finset.ne_of_mem_erase hν)
    linarith [hp_sum]
  -- diff = Σ w_ν (x_ν - x_μ) (same as Σ p_ν (...) since μ-th term is 0)
  have hdiff_w : ∀ k, ξ k - cfg.patterns μ k =
      ∑ ν, w ν * (cfg.patterns ν k - cfg.patterns μ k) := by
    intro k; rw [hdiff_eq k]; congr 1; ext ν; simp only [w]
    split_ifs with h
    · rw [h, sub_self, mul_zero, zero_mul]
    · rfl
  -- Per-coordinate: (ξ_k - x_{μk})² ≤ (1-p_μ) · Σ_ν w_ν (x_{νk} - x_{μk})²
  -- by weighted Cauchy-Schwarz
  -- Sum over k and bound ‖x_ν - x_μ‖² ≤ 4M²
  -- Result: ‖ξ - x_μ‖² ≤ (1-p_μ) · 4M² · (1-p_μ) ≤ 4M²ε
  calc sqNorm (fun k => ξ k - cfg.patterns μ k)
      = ∑ k, (ξ k - cfg.patterns μ k) ^ 2 := rfl
    _ = ∑ k, (∑ ν, w ν * (cfg.patterns ν k - cfg.patterns μ k)) ^ 2 := by
        congr 1; ext k; rw [hdiff_w k]
    _ ≤ ∑ k, ((∑ ν, w ν) *
          ∑ ν, w ν * (cfg.patterns ν k - cfg.patterns μ k) ^ 2) := by
        apply Finset.sum_le_sum; intro k _
        exact weighted_sq_le_local w
          (fun ν => cfg.patterns ν k - cfg.patterns μ k) hw_nn
    _ = (1 - p μ) * ∑ k, ∑ ν,
          w ν * (cfg.patterns ν k - cfg.patterns μ k) ^ 2 := by
        simp_rw [hw_sum]; rw [Finset.mul_sum]
    _ = (1 - p μ) * ∑ ν, w ν *
          sqNorm (fun k => cfg.patterns ν k - cfg.patterns μ k) := by
        congr 1; simp only [sqNorm]; rw [Finset.sum_comm]
        congr 1; ext ν; rw [← Finset.mul_sum]
    _ ≤ (1 - p μ) * (4 * M ^ 2 * (1 - p μ)) := by
        apply mul_le_mul_of_nonneg_left _ hp_tail_nn
        calc ∑ ν, w ν * sqNorm (fun k => cfg.patterns ν k - cfg.patterns μ k)
            ≤ ∑ ν, w ν * (4 * M ^ 2) :=
              Finset.sum_le_sum (fun ν _ =>
                mul_le_mul_of_nonneg_left
                  (sqNorm_sub_le_local _ _ (hNorm ν) (hNorm μ)) (hw_nn ν))
          _ = 4 * M ^ 2 * ∑ ν, w ν := by rw [← Finset.sum_mul]; ring
          _ = 4 * M ^ 2 * (1 - p μ) := by rw [hw_sum]
    _ = 4 * M ^ 2 * (1 - p μ) ^ 2 := by ring
    _ ≤ 4 * M ^ 2 * ε := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        calc (1 - p μ) ^ 2 = (1 - p μ) * (1 - p μ) := sq (1 - p μ)
          _ ≤ ε * 1 :=
              mul_le_mul hp_tail (by linarith [hp_nn μ]) hp_tail_nn hε_nn
          _ = ε := mul_one ε

end HermesNLCDM
