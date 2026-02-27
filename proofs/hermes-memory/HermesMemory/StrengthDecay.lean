/-
  Hermes Memory System — Strength Decay (Anti-Lock-in)

  Problem: Without strength decay, S is monotonically non-decreasing.
  Early memories accumulate high S and permanently dominate recall.

  Fix: Add continuous strength decay dS/dt = -β·S between access events.
  Solution: S(t) = S₀ · e^(-β·t)

  Combined dynamics (access with period Δ, decay rate β):
    At each cycle: S decays by factor e^(-β·Δ), then strength update fires.
    Steady state: S* = α·Smax / (1 - (1-α)·e^(-β·Δ))

  Key result: S* < Smax — established memories plateau below maximum,
  leaving competitive room for newer memories.
-/

import HermesMemory.MemoryDynamics

noncomputable section

open Real Set Filter

-- ============================================================
-- Section 7: Strength Decay (Anti-Lock-in)
--
-- Continuous dynamics between events: dS/dt = -β·S
-- Solution: S(t) = S₀ · e^(-β·t)
-- ============================================================

/-- Continuous strength decay between access events: S(t) = S₀ · e^(-β·t). -/
def strengthDecay (β S₀ t : ℝ) : ℝ := S₀ * exp (-β * t)

/-- Decaying strength is always positive for positive initial strength. -/
theorem strengthDecay_pos {β S₀ : ℝ} (hS₀ : 0 < S₀) (t : ℝ) :
    0 < strengthDecay β S₀ t := by
  unfold strengthDecay
  exact mul_pos hS₀ (exp_pos _)

/-- At time zero, strength equals its initial value. -/
theorem strengthDecay_at_zero (β S₀ : ℝ) : strengthDecay β S₀ 0 = S₀ := by
  unfold strengthDecay; simp

/-- Strength is bounded by its initial value for positive decay rate and non-negative time. -/
theorem strengthDecay_le_init {β S₀ t : ℝ} (hβ : 0 < β) (ht : 0 ≤ t) (hS₀ : 0 < S₀) :
    strengthDecay β S₀ t ≤ S₀ := by
  unfold strengthDecay
  have hexp : exp (-β * t) ≤ 1 := by
    rw [← exp_zero]
    exact exp_le_exp.mpr (by nlinarith)
  calc S₀ * exp (-β * t) ≤ S₀ * 1 :=
        mul_le_mul_of_nonneg_left hexp hS₀.le
    _ = S₀ := mul_one S₀

/-- Strength decays monotonically over time for positive decay rate and initial strength. -/
theorem strengthDecay_antitone {β S₀ : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀) :
    Antitone (strengthDecay β S₀) := by
  intro t₁ t₂ ht
  unfold strengthDecay
  apply mul_le_mul_of_nonneg_left _ hS₀.le
  apply exp_le_exp.mpr
  nlinarith

/-- Without reinforcement, strength decays to zero. -/
theorem strengthDecay_tendsto_zero {β S₀ : ℝ} (hβ : 0 < β) :
    Tendsto (strengthDecay β S₀) atTop (nhds 0) := by
  unfold strengthDecay
  have h_lin : Tendsto (fun t : ℝ => -β * t) atTop atBot := by
    rw [Filter.tendsto_atBot]
    intro b
    rw [Filter.eventually_atTop]
    exact ⟨-b / β, fun t ht => by
      have hbt : β * (-b / β) = -b := by field_simp
      have hmul : β * (-b / β) ≤ β * t := mul_le_mul_of_nonneg_left ht hβ.le
      linarith⟩
  have h_exp : Tendsto (fun t : ℝ => exp (-β * t)) atTop (nhds 0) :=
    tendsto_exp_atBot.comp h_lin
  have h_const : Tendsto (fun _ : ℝ => S₀) atTop (nhds S₀) := tendsto_const_nhds
  have := h_const.mul h_exp
  rwa [mul_zero] at this

-- ============================================================
-- Section 7b: Combined Dynamics (Decay + Access)
--
-- Combined one-step map: after Δ time of decay, then access update.
-- S_{n+1} = strengthUpdate(α, S_n · e^(-β·Δ), Smax)
--         = (1-α) · e^(-β·Δ) · S_n + α · Smax
--
-- This is a linear affine map S ↦ γ·S + c with:
--   γ = (1-α) · e^(-β·Δ) ∈ (0, 1)
--   c = α · Smax > 0
--
-- Steady state: S* = c / (1 - γ) = α·Smax / (1 - (1-α)·e^(-β·Δ))
-- Key theorem: S* < Smax (anti-lock-in)
-- ============================================================

/-- The contraction factor for combined decay-and-update dynamics. -/
def combinedFactor (α β Δ : ℝ) : ℝ := (1 - α) * exp (-β * Δ)

/-- The contraction factor is strictly positive. -/
theorem combinedFactor_pos {α β Δ : ℝ} (hα₁ : α < 1) :
    0 < combinedFactor α β Δ := by
  unfold combinedFactor
  exact mul_pos (by linarith) (exp_pos _)

/-- The contraction factor is strictly less than 1
    (guarantees convergence of the combined iteration). -/
theorem combinedFactor_lt_one {α β Δ : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) :
    combinedFactor α β Δ < 1 := by
  unfold combinedFactor
  have h1mα_pos : 0 < 1 - α := by linarith
  have h1mα_lt : 1 - α < 1 := by linarith
  have hexp_pos : 0 < exp (-β * Δ) := exp_pos _
  have hexp_le : exp (-β * Δ) ≤ 1 := by
    rw [← exp_zero]; exact exp_le_exp.mpr (by nlinarith)
  calc (1 - α) * exp (-β * Δ) < 1 * exp (-β * Δ) :=
        mul_lt_mul_of_pos_right h1mα_lt hexp_pos
    _ = exp (-β * Δ) := one_mul _
    _ ≤ 1 := hexp_le

/-- The steady-state strength under combined decay-and-update dynamics.
    This is the fixed point of the affine map S ↦ γ·S + α·Smax. -/
def steadyStateStrength (α β Δ Smax : ℝ) : ℝ :=
  α * Smax / (1 - combinedFactor α β Δ)

/-- Helper: the denominator of the steady-state formula is positive. -/
theorem steadyState_denom_pos {α β Δ : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) :
    0 < 1 - combinedFactor α β Δ := by
  linarith [combinedFactor_lt_one hα₀ hα₁ hβ hΔ]

/-- The steady-state strength is strictly positive. -/
theorem steadyState_pos {α β Δ Smax : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) (hSmax : 0 < Smax) :
    0 < steadyStateStrength α β Δ Smax := by
  unfold steadyStateStrength
  exact div_pos (mul_pos hα₀ hSmax) (steadyState_denom_pos hα₀ hα₁ hβ hΔ)

/-- **Anti-Lock-in Theorem**: The steady-state strength is strictly less than Smax.
    This is the key result — established memories plateau below maximum strength,
    leaving competitive room for newer memories to enter recall. -/
theorem steadyState_lt_Smax {α β Δ Smax : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) (hSmax : 0 < Smax) :
    steadyStateStrength α β Δ Smax < Smax := by
  unfold steadyStateStrength
  have hdenom := steadyState_denom_pos hα₀ hα₁ hβ hΔ
  -- Goal: α * Smax / (1 - combinedFactor α β Δ) < Smax
  -- Multiply both sides by positive denominator:
  -- α * Smax < Smax * (1 - combinedFactor α β Δ)
  -- ↔ α < 1 - combinedFactor α β Δ
  -- ↔ combinedFactor α β Δ < 1 - α
  -- ↔ (1-α) * exp(-β*Δ) < (1-α)
  -- ↔ exp(-β*Δ) < 1  (since 1-α > 0)
  -- Which is true for β*Δ > 0.
  -- Strategy: cross-multiply using the positive denominator
  have key : α * Smax < Smax * (1 - combinedFactor α β Δ) := by
    -- Expand combinedFactor
    unfold combinedFactor
    have h1mα : 0 < 1 - α := by linarith
    have hexp_le : exp (-β * Δ) ≤ 1 := by
      rw [← exp_zero]; exact exp_le_exp.mpr (by nlinarith)
    have hexp_pos : 0 < exp (-β * Δ) := exp_pos _
    -- (1-α)*exp(-β*Δ) < 1-α since exp(-β*Δ) < 1
    -- But we only have ≤. For strict: exp(-β*Δ) ≠ 1.
    have hexp_ne : exp (-β * Δ) ≠ 1 := by
      intro h
      have heq : exp (-β * Δ) = exp 0 := by rwa [exp_zero]
      have h2 : 0 ≤ -β * Δ := exp_le_exp.mp (le_of_eq heq.symm)
      linarith [mul_pos hβ hΔ]
    have hexp_lt : exp (-β * Δ) < 1 := lt_of_le_of_ne hexp_le hexp_ne
    -- Now: (1-α)*exp(-β*Δ) < (1-α)
    have hcf_lt_1mα : (1 - α) * exp (-β * Δ) < (1 - α) * 1 :=
      mul_lt_mul_of_pos_left hexp_lt h1mα
    -- So: 1 - (1-α)*exp(-β*Δ) > 1 - (1-α) = α
    -- Thus: α < 1 - (1-α)*exp(-β*Δ) = denom
    have hα_lt_denom : α < 1 - (1 - α) * exp (-β * Δ) := by linarith
    -- Finally: α * Smax < Smax * denom
    nlinarith
  -- From key: α * Smax < Smax * denom, and denom > 0
  -- Conclude: α * Smax / denom < Smax
  -- Strategy: (a/d) * d < Smax * d, with d > 0, gives a/d < Smax
  have lhs_eq : α * Smax / (1 - combinedFactor α β Δ) *
    (1 - combinedFactor α β Δ) = α * Smax := by field_simp
  have rhs_eq : Smax * (1 - combinedFactor α β Δ) =
    Smax * (1 - combinedFactor α β Δ) := rfl
  have hmul_lt : α * Smax / (1 - combinedFactor α β Δ) *
    (1 - combinedFactor α β Δ) <
    Smax * (1 - combinedFactor α β Δ) := by linarith
  exact lt_of_mul_lt_mul_right hmul_lt hdenom.le

/-- The steady state is a fixed point of the combined dynamics:
    one period of decay followed by a strength update returns to S*. -/
theorem steadyState_is_fixpoint {α β Δ Smax : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1)
    (hβ : 0 < β) (hΔ : 0 < Δ) :
    strengthUpdate α (steadyStateStrength α β Δ Smax * exp (-β * Δ)) Smax =
    steadyStateStrength α β Δ Smax := by
  rw [strengthUpdate_alt]
  -- Goal: (1-α)*(S* * e) + α*Smax = S*
  -- Strategy: multiply both sides by (1 - γ) ≠ 0, use div_mul_cancel₀
  have hd : (1 - combinedFactor α β Δ) ≠ 0 :=
    ne_of_gt (steadyState_denom_pos hα₀ hα₁ hβ hΔ)
  have hS_mul : steadyStateStrength α β Δ Smax * (1 - combinedFactor α β Δ) = α * Smax := by
    unfold steadyStateStrength
    exact div_mul_cancel₀ (α * Smax) hd
  have hcf : combinedFactor α β Δ = (1 - α) * exp (-β * Δ) := rfl
  apply mul_right_cancel₀ hd
  rw [hS_mul]
  calc ((1 - α) * (steadyStateStrength α β Δ Smax * exp (-β * Δ)) + α * Smax) *
        (1 - combinedFactor α β Δ)
      = (1 - α) * exp (-β * Δ) *
          (steadyStateStrength α β Δ Smax * (1 - combinedFactor α β Δ)) +
        α * Smax * (1 - combinedFactor α β Δ) := by ring
    _ = (1 - α) * exp (-β * Δ) * (α * Smax) +
        α * Smax * (1 - combinedFactor α β Δ) := by rw [hS_mul]
    _ = combinedFactor α β Δ * (α * Smax) +
        α * Smax * (1 - combinedFactor α β Δ) := by rw [hcf]
    _ = α * Smax := by ring

end
