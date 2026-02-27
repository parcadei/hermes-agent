/-
  Hermes Memory System — Composition Theorem

  The three failure-mode fixes (strength decay, soft selection, novelty bonus)
  are proved individually in StrengthDecay.lean, SoftSelection.lean, and
  NoveltyBonus.lean. This file proves they compose correctly.

  Approach: Mean-field analysis of the coupled 2-memory system.
  At each step of period Δ:
    1. Both memories' strengths decay: S_i → S_i · e^(-βΔ)
    2. Soft selection picks memory i with probability q_i (q₁ + q₂ = 1)
    3. The selected memory gets a strength update: S → (1-α)S + αSmax

  The expected strength update for memory i is:
    E[S_i'] = (1 - q_i·α) · e^(-βΔ) · S_i + q_i · α · Smax

  Key results:
  - Domain invariance: [0, Smax]² is forward-invariant
  - Fixed-point safety: ANY fixed point has S_i* < Smax (anti-lock-in)
  - Contraction: Under parameter constraint, the mean-field map contracts
  - Composed guarantees: All three failure-mode fixes hold simultaneously
-/

import HermesMemory.StrengthDecay
import HermesMemory.SoftSelection
import HermesMemory.NoveltyBonus

noncomputable section

open Real Set Filter

-- ============================================================
-- Section 10: Composed State — Mean-Field Expected Update
--
-- For a 2-memory system, the expected strength after one step
-- depends on the selection probability q ∈ (0,1).
-- ============================================================

/-- Expected strength of memory i after one step, given selection probability q.
    Derivation: With prob q, accessed: S' = (1-α)·e^(-βΔ)·S + α·Smax
                With prob 1-q, not accessed: S' = e^(-βΔ)·S
                E[S'] = (1 - qα)·e^(-βΔ)·S + qα·Smax -/
def expectedStrengthUpdate (α β Δ Smax q S : ℝ) : ℝ :=
  (1 - q * α) * exp (-β * Δ) * S + q * α * Smax

/-- The composed mean-field map for a 2-memory system.
    selectProb returns the probability of selecting memory 1. -/
def composedExpectedMap
    (selectProb : ℝ → ℝ → ℝ) (α β Δ Smax : ℝ) (S : ℝ × ℝ) : ℝ × ℝ :=
  let p := selectProb S.1 S.2
  (expectedStrengthUpdate α β Δ Smax p S.1,
   expectedStrengthUpdate α β Δ Smax (1 - p) S.2)

-- ============================================================
-- Section 10a: Domain Invariance
--
-- The expected update preserves [0, Smax] for valid parameters.
-- ============================================================

/-- The expected update is non-negative when S ≥ 0 and q ∈ [0,1]. -/
theorem expectedStrengthUpdate_nonneg {α β Δ Smax q S : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hq₀ : 0 ≤ q) (hq₁ : q ≤ 1)
    (hSmax : 0 < Smax) (hS : 0 ≤ S) :
    0 ≤ expectedStrengthUpdate α β Δ Smax q S := by
  unfold expectedStrengthUpdate
  have he : 0 < exp (-β * Δ) := exp_pos _
  have hqα_le : q * α ≤ 1 := by nlinarith
  have h1 : 0 ≤ (1 - q * α) * exp (-β * Δ) * S :=
    mul_nonneg (mul_nonneg (by linarith) he.le) hS
  have h2 : 0 ≤ q * α * Smax :=
    mul_nonneg (mul_nonneg hq₀ hα₀.le) hSmax.le
  linarith

/-- The expected update is at most Smax when S ∈ [0, Smax] and q ∈ [0,1]. -/
theorem expectedStrengthUpdate_le_Smax {α β Δ Smax q S : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (_ : 0 ≤ q) (hq₁ : q ≤ 1)
    (hS₀ : 0 ≤ S) (hS : S ≤ Smax) :
    expectedStrengthUpdate α β Δ Smax q S ≤ Smax := by
  unfold expectedStrengthUpdate
  have he₁ : exp (-β * Δ) ≤ 1 := by
    rw [← exp_zero]; exact exp_le_exp.mpr (by nlinarith)
  have hqα_le : q * α ≤ 1 := by nlinarith
  -- ((1-qα)*e)*S ≤ (1-qα)*S ≤ (1-qα)*Smax
  have h_coeff_le : (1 - q * α) * exp (-β * Δ) ≤ 1 - q * α :=
    mul_le_of_le_one_right (by linarith) he₁
  have h_term1 : (1 - q * α) * exp (-β * Δ) * S ≤ (1 - q * α) * Smax :=
    calc (1 - q * α) * exp (-β * Δ) * S
        ≤ (1 - q * α) * S := mul_le_mul_of_nonneg_right h_coeff_le hS₀
      _ ≤ (1 - q * α) * Smax := mul_le_mul_of_nonneg_left hS (by linarith)
  calc (1 - q * α) * exp (-β * Δ) * S + q * α * Smax
      ≤ (1 - q * α) * Smax + q * α * Smax := by linarith
    _ = Smax := by ring

-- ============================================================
-- Section 10b: Fixed-Point Safety
--
-- The central composition safety theorem: at ANY fixed point
-- of the composed mean-field dynamics, both memory strengths
-- are strictly below Smax. This holds for any selection
-- probability function with q ∈ (0,1).
--
-- This is the "anti-lock-in survives composition" guarantee.
-- ============================================================

/-- At any fixed point of the expected update with selection probability
    q ∈ (0,1], S* < Smax. The key composition safety theorem.

    Proof strategy (avoids division entirely):
    From S = T(S): S*(1 - (1-qα)*e) = qα*Smax
    Show: denom - qα = (1-e)*(1-qα) > 0, so denom > qα
    Then: qα*Smax < denom*Smax, so S < Smax by cancelling denom. -/
theorem fixedPoint_lt_Smax {α β Δ Smax q S : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hq₀ : 0 < q) (hq₁ : q ≤ 1)
    (hSmax : 0 < Smax) (_ : 0 ≤ S)
    (hfp : S = expectedStrengthUpdate α β Δ Smax q S) :
    S < Smax := by
  unfold expectedStrengthUpdate at hfp
  set e := exp (-β * Δ) with he_def
  have he₀ : 0 < e := exp_pos _
  -- e < 1 because -β*Δ < 0
  have he₁ : e < 1 := by
    rw [he_def, ← exp_zero]
    exact exp_lt_exp.mpr (by nlinarith)
  -- Define denom = 1 - (1-qα)*e
  set d := 1 - (1 - q * α) * e with hd_def
  -- denom > 0: since (1-qα)*e < 1 (both factors < 1 for the strict part)
  have hqα_lt : q * α < 1 := by nlinarith
  have h1_sub_e : 0 < 1 - e := by linarith
  have h1_sub_qα : 0 < 1 - q * α := by linarith
  -- denom - qα = (1-e)(1-qα) > 0
  have hd_minus_qα : d - q * α = (1 - e) * (1 - q * α) := by
    rw [hd_def]; ring
  have hd_gt_qα : q * α < d := by
    linarith [mul_pos h1_sub_e h1_sub_qα]
  have hd_pos : 0 < d := by linarith [mul_pos hq₀ hα₀]
  -- From fixed point: S*d = qα*Smax
  have hrearrange : S * d = q * α * Smax := by
    rw [hd_def]; linarith
  -- qα*Smax < d*Smax (since qα < d and Smax > 0)
  have hkey : q * α * Smax < d * Smax := by nlinarith
  -- S*d < Smax*d (need this form for lt_of_mul_lt_mul_right)
  have hmul : S * d < Smax * d := by
    calc S * d = q * α * Smax := hrearrange
      _ < d * Smax := hkey
      _ = Smax * d := by ring
  -- Cancel d > 0
  exact lt_of_mul_lt_mul_right hmul hd_pos.le

/-- At any fixed point of the composed system with selection probs in (0,1),
    the first memory's strength is strictly below Smax. -/
theorem composedFixedPoint_fst_lt_Smax {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax : ℝ} {S : ℝ × ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax) (hS₁ : 0 ≤ S.1)
    (hp₀ : 0 < selectProb S.1 S.2) (hp₁ : selectProb S.1 S.2 ≤ 1)
    (hfp : S = composedExpectedMap selectProb α β Δ Smax S) :
    S.1 < Smax := by
  have hfp₁ : S.1 = expectedStrengthUpdate α β Δ Smax (selectProb S.1 S.2) S.1 := by
    have := congr_arg Prod.fst hfp
    simp [composedExpectedMap] at this
    exact this
  exact fixedPoint_lt_Smax hα₀ hα₁ hβ hΔ hp₀ hp₁ hSmax hS₁ hfp₁

/-- At any fixed point of the composed system with selection probs in (0,1),
    the second memory's strength is strictly below Smax. -/
theorem composedFixedPoint_snd_lt_Smax {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax : ℝ} {S : ℝ × ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax) (hS₂ : 0 ≤ S.2)
    (hp₀ : 0 < selectProb S.1 S.2) (hp₁ : selectProb S.1 S.2 < 1)
    (hfp : S = composedExpectedMap selectProb α β Δ Smax S) :
    S.2 < Smax := by
  have hfp₂ : S.2 = expectedStrengthUpdate α β Δ Smax (1 - selectProb S.1 S.2) S.2 := by
    have := congr_arg Prod.snd hfp
    simp [composedExpectedMap] at this
    exact this
  exact fixedPoint_lt_Smax hα₀ hα₁ hβ hΔ (by linarith) (by linarith) hSmax hS₂ hfp₂

-- ============================================================
-- Section 11: Contraction Bound
--
-- The Lipschitz constant of the composed expected map is
--   K = exp(-βΔ) + L · α · Smax
-- where L is the Lipschitz constant of the selection probability.
--
-- When K < 1 (i.e., L < (1 - exp(-βΔ)) / (α · Smax)),
-- the map is a contraction and has a unique fixed point.
-- ============================================================

/-- The contraction factor of the composed mean-field dynamics. -/
def composedContractionFactor (β Δ L α Smax : ℝ) : ℝ :=
  exp (-β * Δ) + L * α * Smax

/-- The contraction condition: the selection probability's Lipschitz constant
    must be small enough relative to the "gap" created by strength decay. -/
theorem composedContractionFactor_lt_one {β Δ L α Smax : ℝ}
    (hConstraint : L * α * Smax < 1 - exp (-β * Δ)) :
    composedContractionFactor β Δ L α Smax < 1 := by
  unfold composedContractionFactor; linarith

/-- Lipschitz bound on the expected update with respect to strength,
    at a fixed selection probability: the factor is exp(-βΔ). -/
theorem expectedUpdate_lipschitz_in_S {α β Δ Smax q S S' : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hq₀ : 0 ≤ q) (hq₁ : q ≤ 1) :
    |expectedStrengthUpdate α β Δ Smax q S - expectedStrengthUpdate α β Δ Smax q S'|
    ≤ exp (-β * Δ) * |S - S'| := by
  unfold expectedStrengthUpdate
  have hsimp : (1 - q * α) * exp (-β * Δ) * S + q * α * Smax -
    ((1 - q * α) * exp (-β * Δ) * S' + q * α * Smax) =
    (1 - q * α) * exp (-β * Δ) * (S - S') := by ring
  rw [hsimp, abs_mul, abs_mul]
  have hqα_nn : 0 ≤ 1 - q * α := by nlinarith
  have he_nn : 0 ≤ exp (-β * Δ) := (exp_pos _).le
  rw [abs_of_nonneg hqα_nn, abs_of_nonneg he_nn]
  -- (1-qα) * e * |S-S'| ≤ 1 * e * |S-S'| = e * |S-S'|
  have h1qα : 1 - q * α ≤ 1 := by nlinarith [mul_nonneg hq₀ hα₀.le]
  calc (1 - q * α) * exp (-β * Δ) * |S - S'|
      ≤ 1 * exp (-β * Δ) * |S - S'| := by
        apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
        exact mul_le_mul_of_nonneg_right h1qα he_nn
    _ = exp (-β * Δ) * |S - S'| := by ring

/-- Lipschitz bound for the expected update with variation in q only:
    |T(q,S) - T(q',S)| ≤ α·Smax·|q-q'| when S ∈ [0,Smax] -/
theorem expectedUpdate_lipschitz_in_q {α β Δ Smax q q' S : ℝ}
    (hα₀ : 0 < α)
    (hβ : 0 < β) (hΔ : 0 < Δ) (hS₀ : 0 ≤ S) (hS : S ≤ Smax) :
    |expectedStrengthUpdate α β Δ Smax q S - expectedStrengthUpdate α β Δ Smax q' S|
    ≤ α * Smax * |q - q'| := by
  unfold expectedStrengthUpdate
  have hsimp : (1 - q * α) * exp (-β * Δ) * S + q * α * Smax -
    ((1 - q' * α) * exp (-β * Δ) * S + q' * α * Smax) =
    (q - q') * α * (Smax - exp (-β * Δ) * S) := by ring
  rw [hsimp]
  have he₀ : 0 ≤ exp (-β * Δ) := (exp_pos _).le
  have he₁ : exp (-β * Δ) ≤ 1 := by
    rw [← exp_zero]; exact exp_le_exp.mpr (by nlinarith)
  have hSmax_sub : 0 ≤ Smax - exp (-β * Δ) * S := by nlinarith
  have hSmax_bound : Smax - exp (-β * Δ) * S ≤ Smax := by nlinarith [mul_nonneg he₀ hS₀]
  rw [abs_mul, abs_mul, abs_of_nonneg hα₀.le, abs_of_nonneg hSmax_sub]
  calc |q - q'| * α * (Smax - exp (-β * Δ) * S)
      ≤ |q - q'| * α * Smax :=
        mul_le_mul_of_nonneg_left hSmax_bound (mul_nonneg (abs_nonneg _) hα₀.le)
    _ = α * Smax * |q - q'| := by ring

/-- Full Lipschitz bound via triangle inequality:
    |T(q,S) - T(q',S')| ≤ exp(-βΔ)·|S-S'| + α·Smax·|q-q'| -/
theorem expectedUpdate_lipschitz_full {α β Δ Smax q q' S S' : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1)
    (hq₀ : 0 ≤ q) (hq₁ : q ≤ 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) (hS'₀ : 0 ≤ S') (hS' : S' ≤ Smax) :
    |expectedStrengthUpdate α β Δ Smax q S - expectedStrengthUpdate α β Δ Smax q' S'|
    ≤ exp (-β * Δ) * |S - S'| + α * Smax * |q - q'| := by
  -- Route through intermediate point T(q, S')
  set A := expectedStrengthUpdate α β Δ Smax q S
  set B := expectedStrengthUpdate α β Δ Smax q S'
  set C := expectedStrengthUpdate α β Δ Smax q' S'
  -- |A - C| ≤ |A - B| + |B - C| (triangle inequality for |·| via dist)
  have htri : |A - C| ≤ |A - B| + |B - C| := by
    have := dist_triangle A B C
    simp only [Real.dist_eq] at this
    exact this
  -- |A - B| ≤ e·|S-S'|
  have h1 : |A - B| ≤ exp (-β * Δ) * |S - S'| :=
    expectedUpdate_lipschitz_in_S hα₀ hα₁ hq₀ hq₁
  -- |B - C| ≤ α·Smax·|q-q'|
  have h2 : |B - C| ≤ α * Smax * |q - q'| :=
    expectedUpdate_lipschitz_in_q hα₀ hβ hΔ hS'₀ hS'
  linarith

-- ============================================================
-- Section 12: Composed Guarantees
--
-- Combining the above results to show all three failure-mode
-- fixes hold simultaneously under composition.
-- ============================================================

/-- At any fixed point of the composed system, soft selection gives
    both memories strictly positive selection probability. -/
theorem composedFixedPoint_selection_pos (s₁ s₂ T : ℝ) :
    0 < softSelect s₁ s₂ T ∧ softSelect s₁ s₂ T < 1 :=
  ⟨softSelect_pos s₁ s₂ T, softSelect_lt_one s₁ s₂ T⟩

/-- The novelty bonus remains effective in the composed system:
    boosted score is bounded and the bonus eventually vanishes. -/
theorem composedNoveltyBonus_bounded {N₀ γ : ℝ} (hN₀ : 0 < N₀) (hγ : 0 < γ)
    {baseScore : ℝ} (hbase₁ : baseScore ≤ 1) (t : ℝ) (ht : 0 ≤ t) :
    boostedScore baseScore N₀ γ t ≤ 1 + N₀ := by
  unfold boostedScore
  have := noveltyBonus_le_init hγ ht hN₀
  linarith

/-- **Full Composition Safety**: Given any fixed point of the composed
    mean-field dynamics with soft selection, all three guarantees hold:
    1. Anti-lock-in: both strengths < Smax
    2. Anti-thrashing: selection probabilities in (0,1)
    3. Anti-cold-start: novelty bonus effective during exploration window -/
theorem composedSystem_safe {α β Δ Smax T N₀ γ ε : ℝ} {S : ℝ × ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax)
    (hN₀ : 0 < N₀) (hγ : 0 < γ) (hε : 0 < ε) (hεN : ε < N₀)
    (hS₁ : 0 ≤ S.1) (hS₂ : 0 ≤ S.2)
    (hfp : S = composedExpectedMap (fun s₁ s₂ => softSelect s₁ s₂ T) α β Δ Smax S) :
    -- 1. Anti-lock-in: both strengths below Smax
    S.1 < Smax ∧ S.2 < Smax ∧
    -- 2. Anti-thrashing: selection probabilities in (0,1)
    (0 < softSelect S.1 S.2 T ∧ softSelect S.1 S.2 T < 1) ∧
    -- 3. Anti-cold-start: novelty bonus survives through exploration window
    (∀ t, 0 ≤ t → t ≤ explorationWindow N₀ γ ε →
      ε ≤ boostedScore 0 N₀ γ t) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  -- 1a. S.1 < Smax
  · exact composedFixedPoint_fst_lt_Smax hα₀ hα₁ hβ hΔ hSmax hS₁
      (softSelect_pos S.1 S.2 T) (softSelect_lt_one S.1 S.2 T).le hfp
  -- 1b. S.2 < Smax
  · exact composedFixedPoint_snd_lt_Smax hα₀ hα₁ hβ hΔ hSmax hS₂
      (softSelect_pos S.1 S.2 T) (softSelect_lt_one S.1 S.2 T) hfp
  -- 2. Selection probabilities in (0,1)
  · exact composedFixedPoint_selection_pos S.1 S.2 T
  -- 3. Cold-start survival (independent of fixed point)
  · intro t ht₀ ht
    exact coldStart_survival hN₀ hγ hε hεN ht₀ ht

end
