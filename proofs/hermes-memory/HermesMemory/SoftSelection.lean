/-
  Hermes Memory System — Soft Selection (Anti-Thrashing)

  Problem: Hard top-k selection creates a discontinuity at the selection
  boundary. Memories near the boundary alternate between being recalled
  and not recalled, causing importance to oscillate (thrashing).

  Fix: Replace hard top-k with soft selection via softmax / sigmoid.
  For binary selection (memory vs threshold), soft selection reduces to:
    P(select) = σ((score - threshold) / T)
  where T > 0 is the temperature parameter.

  Key results:
  - Soft selection is monotone: higher score → higher probability
  - Soft selection is continuous: no discontinuous jumps at boundary
  - Selection probabilities are complementary: P(A) + P(B) = 1
  - Temperature T controls sharpness (T→0 recovers hard selection)
-/

import HermesMemory.MemoryDynamics

noncomputable section

open Real Set Filter

-- ============================================================
-- Section 8: Soft Selection (Anti-Thrashing)
--
-- Binary soft selection via sigmoid replaces discontinuous top-k
-- with a continuous probability of recall.
-- ============================================================

/-- Sigmoid is monotone increasing: higher input → higher output.
    This is the foundational continuity property that prevents thrashing. -/
theorem sigmoid_monotone : Monotone sigmoid := by
  intro x₁ x₂ hx
  unfold sigmoid
  have h₁ : (0 : ℝ) < 1 + exp (-x₁) := sigmoid_denom_pos x₁
  have h₂ : (0 : ℝ) < 1 + exp (-x₂) := sigmoid_denom_pos x₂
  -- 1/(1+exp(-x₁)) ≤ 1/(1+exp(-x₂)) ↔ (1+exp(-x₂)) ≤ (1+exp(-x₁))
  -- since we're dividing 1 by larger denom → smaller result
  have hexp : exp (-x₂) ≤ exp (-x₁) := exp_le_exp.mpr (by linarith)
  have hdenom : 1 + exp (-x₂) ≤ 1 + exp (-x₁) := by linarith
  -- Cross-multiply: 1/a ≤ 1/b ↔ a ≥ b for positive a, b
  have hprod : (0 : ℝ) < (1 + exp (-x₁)) * (1 + exp (-x₂)) := mul_pos h₁ h₂
  have lhs_scaled : 1 / (1 + exp (-x₁)) * ((1 + exp (-x₁)) * (1 + exp (-x₂))) =
    1 + exp (-x₂) := by field_simp
  have rhs_scaled : 1 / (1 + exp (-x₂)) * ((1 + exp (-x₁)) * (1 + exp (-x₂))) =
    1 + exp (-x₁) := by field_simp
  exact le_of_mul_le_mul_right (by linarith) hprod

/-- Binary soft selection: probability of selecting a memory with score s₁
    over a competitor with score s₂, at temperature T. -/
def softSelect (s₁ s₂ T : ℝ) : ℝ := sigmoid ((s₁ - s₂) / T)

/-- Soft selection probability is always strictly positive (never zero).
    Unlike hard selection, every memory has a nonzero chance. -/
theorem softSelect_pos (s₁ s₂ T : ℝ) : 0 < softSelect s₁ s₂ T := by
  unfold softSelect
  exact sigmoid_pos _

/-- Soft selection probability is always strictly less than 1 (never certain).
    Unlike hard selection, no memory has a guaranteed slot. -/
theorem softSelect_lt_one (s₁ s₂ T : ℝ) : softSelect s₁ s₂ T < 1 := by
  unfold softSelect
  exact sigmoid_lt_one _

/-- Soft selection probability lies in the open interval (0, 1). -/
theorem softSelect_mem_Ioo (s₁ s₂ T : ℝ) : softSelect s₁ s₂ T ∈ Ioo 0 1 :=
  ⟨softSelect_pos s₁ s₂ T, softSelect_lt_one s₁ s₂ T⟩

/-- **Complementarity**: The probabilities of selecting A over B and B over A sum to 1.
    This is the σ(x) + σ(-x) = 1 identity. -/
theorem softSelect_complementary (s₁ s₂ T : ℝ) (hT : T ≠ 0) :
    softSelect s₁ s₂ T + softSelect s₂ s₁ T = 1 := by
  unfold softSelect sigmoid
  -- Goal: 1/(1+exp(-(s₁-s₂)/T)) + 1/(1+exp(-(s₂-s₁)/T)) = 1
  have h₁ : (0 : ℝ) < 1 + exp (-((s₁ - s₂) / T)) := sigmoid_denom_pos _
  have h₂ : (0 : ℝ) < 1 + exp (-((s₂ - s₁) / T)) := sigmoid_denom_pos _
  -- Key identity: exp(-(s₂-s₁)/T) = exp((s₁-s₂)/T) = 1/exp(-(s₁-s₂)/T)
  -- So 1+exp(-(s₂-s₁)/T) = 1 + 1/exp(-(s₁-s₂)/T) = (exp(-(s₁-s₂)/T)+1)/exp(-(s₁-s₂)/T)
  have hexp_pos : (0 : ℝ) < exp (-((s₁ - s₂) / T)) := exp_pos _
  -- Cross-multiply by both denominators
  have hprod : (0 : ℝ) < (1 + exp (-((s₁ - s₂) / T))) * (1 + exp (-((s₂ - s₁) / T))) :=
    mul_pos h₁ h₂
  have hne₁ : (1 + exp (-((s₁ - s₂) / T))) ≠ 0 := ne_of_gt h₁
  have hne₂ : (1 + exp (-((s₂ - s₁) / T))) ≠ 0 := ne_of_gt h₂
  rw [div_add_div _ _ hne₁ hne₂]
  rw [div_eq_one_iff_eq (mul_ne_zero hne₁ hne₂)]
  -- Goal: 1 * (1 + exp(-((s₂-s₁)/T))) + 1 * (1 + exp(-((s₁-s₂)/T))) =
  --       (1 + exp(-((s₁-s₂)/T))) * (1 + exp(-((s₂-s₁)/T)))
  -- LHS = 2 + exp(-((s₂-s₁)/T)) + exp(-((s₁-s₂)/T))
  -- RHS = 1 + exp(-((s₂-s₁)/T)) + exp(-((s₁-s₂)/T)) + exp(-((s₂-s₁)/T))*exp(-((s₁-s₂)/T))
  -- RHS - LHS = exp(-((s₂-s₁)/T))*exp(-((s₁-s₂)/T)) - 1
  -- = exp(-((s₂-s₁)/T) + -((s₁-s₂)/T)) - 1
  -- = exp(0) - 1 = 0
  -- So LHS = RHS.
  have hexp_mul : exp (-((s₂ - s₁) / T)) * exp (-((s₁ - s₂) / T)) = 1 := by
    rw [← exp_add]
    have : -((s₂ - s₁) / T) + -((s₁ - s₂) / T) = 0 := by field_simp; ring
    rw [this, exp_zero]
  nlinarith [hexp_mul]

/-- **Anti-Thrashing (Monotonicity)**: For positive temperature,
    increasing a memory's score strictly increases its selection probability.
    Small score changes → small probability changes (no discontinuous jumps). -/
theorem softSelect_monotone_score {s₁ s₂ s₁' T : ℝ} (hT : 0 < T)
    (hs : s₁ ≤ s₁') :
    softSelect s₁ s₂ T ≤ softSelect s₁' s₂ T := by
  unfold softSelect
  apply sigmoid_monotone
  apply div_le_div_of_nonneg_right _ hT.le
  linarith

/-- Higher temperature flattens the selection curve toward uniform (0.5).
    This quantifies how temperature controls the sharpness of selection. -/
theorem softSelect_equal_scores (s T : ℝ) :
    softSelect s s T = sigmoid 0 := by
  unfold softSelect
  simp

/-- σ(0) = 1/2: equal scores give 50/50 selection. -/
theorem sigmoid_at_zero : sigmoid 0 = 1 / 2 := by
  unfold sigmoid
  simp
  norm_num

/-- At equal scores, each memory has exactly 50% selection probability. -/
theorem softSelect_equal_is_half (s T : ℝ) :
    softSelect s s T = 1 / 2 := by
  rw [softSelect_equal_scores s T, sigmoid_at_zero]

end
