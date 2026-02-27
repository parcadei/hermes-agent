/-
  Hermes Memory System — Dynamical System Formalization

  Machine-verified properties of the memory retrieval and update dynamics.

  State variables per memory:
    R(t) : ℝ  — retention level ∈ [0, 1]
    S    : ℝ  — strength (decay time constant) ∈ [S_min, S_max]
    imp  : ℝ  — importance ∈ [0, 1]

  Continuous dynamics (between access events):
    dR/dt = -(1/S)·R  ⟹  R(t) = e^(-t/S)

  Discrete dynamics (at access events):
    R⁺ = 1  (retention reset)
    S⁺ = S + α·(S_max - S)  (strength growth)

  Readout:
    score = w₁·relevance + w₂·recency + w₃·importance + w₄·σ(activation)

  Feedback:
    imp⁺ = clamp(imp + δ·signal, 0, 1)
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import Mathlib.Data.Real.Basic
import Mathlib.Order.Interval.Set.Basic
import Mathlib.Order.Filter.AtTopBot.Tendsto

noncomputable section

open Real Set Filter

-- ============================================================
-- Section 1: Retention (Forgetting Curve)
--
-- R(t) = e^(-t/S) is the solution to dR/dt = -(1/S)·R
-- with initial condition R(0) = 1.
-- ============================================================

/-- Retention as a function of elapsed time `t` and strength `S`. -/
def retention (t S : ℝ) : ℝ := exp (-t / S)

/-- Retention is always strictly positive. -/
theorem retention_pos (t S : ℝ) : 0 < retention t S := by
  unfold retention
  exact exp_pos _

/-- At time zero, retention is 1 (memory just accessed). -/
theorem retention_at_zero (S : ℝ) (_ : S ≠ 0) : retention 0 S = 1 := by
  unfold retention
  simp

/-- Retention is at most 1 for non-negative time and positive strength. -/
theorem retention_le_one {t S : ℝ} (ht : 0 ≤ t) (hS : 0 < S) :
    retention t S ≤ 1 := by
  unfold retention
  rw [← exp_zero]
  apply exp_le_exp.mpr
  have : -t / S ≤ 0 := div_nonpos_of_nonpos_of_nonneg (by linarith) hS.le
  linarith

/-- Retention lies in [0, 1] for non-negative time and positive strength. -/
theorem retention_mem_Icc {t S : ℝ} (ht : 0 ≤ t) (hS : 0 < S) :
    retention t S ∈ Icc 0 1 :=
  ⟨le_of_lt (retention_pos t S), retention_le_one ht hS⟩

/-- Retention is monotonically decreasing in time (for fixed positive strength).
    Equivalently, more time elapsed ⟹ lower retention. -/
theorem retention_antitone {S : ℝ} (hS : 0 < S) :
    Antitone (fun t => retention t S) := by
  intro t₁ t₂ h
  unfold retention
  apply exp_le_exp.mpr
  apply div_le_div_of_nonneg_right _ hS.le
  linarith

/-- Stronger memories decay slower: for fixed positive time,
    retention increases with strength. -/
theorem retention_mono_strength {t S₁ S₂ : ℝ} (ht : 0 < t)
    (hS₁ : 0 < S₁) (hS₂ : 0 < S₂) (hS : S₁ ≤ S₂) :
    retention t S₁ ≤ retention t S₂ := by
  unfold retention
  apply exp_le_exp.mpr
  -- Need: -t / S₁ ≤ -t / S₂.
  -- Strategy: clear denominators by multiplying by S₁ * S₂ > 0.
  have h₁ : 0 < S₁ * S₂ := mul_pos hS₁ hS₂
  have lhs : -t / S₁ * (S₁ * S₂) = -t * S₂ := by field_simp
  have rhs : -t / S₂ * (S₁ * S₂) = -t * S₁ := by field_simp
  have hmul : -t * S₂ ≤ -t * S₁ := by nlinarith
  exact le_of_mul_le_mul_right (by linarith) h₁

/-- Without reinforcement, retention decays to zero. -/
theorem retention_tendsto_zero {S : ℝ} (hS : 0 < S) :
    Tendsto (fun t => retention t S) atTop (nhds 0) := by
  unfold retention
  -- exp(-t/S) → 0 = exp ∘ (t ↦ -t/S), where -t/S → -∞ and exp(-∞) → 0
  suffices key : Tendsto (fun t : ℝ => -t / S) atTop atBot from
    tendsto_exp_atBot.comp key
  rw [Filter.tendsto_atBot]
  intro b
  rw [Filter.eventually_atTop]
  exact ⟨-b * S, fun t ht => by
    have h1 : -t / S * S = -t := by field_simp
    have h2 : -t / S * S ≤ b * S := by linarith
    exact le_of_mul_le_mul_right h2 hS⟩

-- ============================================================
-- Section 2: Strength Update (Discrete Dynamics)
--
-- At each access event:  S' = S + α·(S_max - S)
-- This is equivalent to: S' = (1 - α)·S + α·S_max
-- Closed form: S_n = S_max - (S_max - S₀)·(1 - α)^n
-- ============================================================

/-- One step of the strength update. -/
def strengthUpdate (α S Smax : ℝ) : ℝ := S + α * (Smax - S)

/-- Alternative form of strength update. -/
theorem strengthUpdate_alt (α S Smax : ℝ) :
    strengthUpdate α S Smax = (1 - α) * S + α * Smax := by
  unfold strengthUpdate; ring

/-- Strength strictly increases at each access (when below max). -/
theorem strength_increases {α S Smax : ℝ} (hα : 0 < α) (hS : S < Smax) :
    S < strengthUpdate α S Smax := by
  unfold strengthUpdate
  linarith [mul_pos hα (sub_pos.mpr hS)]

/-- Updated strength stays at or below S_max. -/
theorem strength_le_max {α S Smax : ℝ} (hα : α ≤ 1) (hS : S ≤ Smax) :
    strengthUpdate α S Smax ≤ Smax := by
  rw [strengthUpdate_alt]
  nlinarith [sub_nonneg.mpr hα, sub_nonneg.mpr hS]

/-- Updated strength stays at or above S (strength never decreases). -/
theorem strength_nondecreasing {α S Smax : ℝ} (hα : 0 ≤ α) (hS : S ≤ Smax) :
    S ≤ strengthUpdate α S Smax := by
  unfold strengthUpdate
  linarith [mul_nonneg hα (sub_nonneg.mpr hS)]

/-- Iterated strength update: n applications of the update rule. -/
def strengthIter (α S₀ Smax : ℝ) : ℕ → ℝ
  | 0 => S₀
  | n + 1 => strengthUpdate α (strengthIter α S₀ Smax n) Smax

/-- Closed form for iterated strength. -/
theorem strengthIter_closed {α S₀ Smax : ℝ} (n : ℕ) :
    strengthIter α S₀ Smax n = Smax - (Smax - S₀) * (1 - α) ^ n := by
  induction n with
  | zero => simp [strengthIter]
  | succ n ih =>
    simp only [strengthIter]
    rw [ih, strengthUpdate_alt]
    ring

/-- Iterated strength converges to S_max. -/
theorem strengthIter_tendsto {α S₀ Smax : ℝ} (hα₀ : 0 < α) (hα₁ : α < 1) :
    Tendsto (strengthIter α S₀ Smax) atTop (nhds Smax) := by
  -- Closed form: Smax - (Smax - S₀) * (1-α)^n, where (1-α)^n → 0
  have heq : strengthIter α S₀ Smax = fun n => Smax - (Smax - S₀) * (1 - α) ^ n := by
    ext n; exact strengthIter_closed n
  rw [heq]
  have hr₀ : 0 ≤ 1 - α := by linarith
  have hr₁ : 1 - α < 1 := by linarith
  have h_geo : Tendsto (fun n : ℕ => (1 - α) ^ n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hr₀ hr₁
  have h_prod : Tendsto (fun n : ℕ => (Smax - S₀) * (1 - α) ^ n) atTop (nhds 0) := by
    have hc : Tendsto (fun _ : ℕ => Smax - S₀) atTop (nhds (Smax - S₀)) := tendsto_const_nhds
    have := hc.mul h_geo
    rwa [mul_zero] at this
  have h_const : Tendsto (fun _ : ℕ => Smax) atTop (nhds Smax) := tendsto_const_nhds
  have := h_const.sub h_prod
  rwa [sub_zero] at this

-- ============================================================
-- Section 3: Sigmoid Function
--
-- σ(x) = 1 / (1 + e^(-x))
-- Maps activation ∈ ℝ to a bounded (0, 1) signal.
-- ============================================================

/-- The sigmoid (logistic) function. -/
def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- The denominator of sigmoid is always positive. -/
theorem sigmoid_denom_pos (x : ℝ) : 0 < 1 + exp (-x) := by
  linarith [exp_pos (-x)]

/-- Sigmoid is always strictly positive. -/
theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  exact div_pos one_pos (sigmoid_denom_pos x)

/-- Sigmoid is strictly less than 1. -/
theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one (sigmoid_denom_pos x)]
  linarith [exp_pos (-x)]

/-- Sigmoid maps to the open interval (0, 1). -/
theorem sigmoid_mem_Ioo (x : ℝ) : sigmoid x ∈ Ioo 0 1 :=
  ⟨sigmoid_pos x, sigmoid_lt_one x⟩

/-- Sigmoid maps to the closed interval [0, 1]. -/
theorem sigmoid_mem_Icc (x : ℝ) : sigmoid x ∈ Icc 0 1 :=
  ⟨le_of_lt (sigmoid_pos x), le_of_lt (sigmoid_lt_one x)⟩

-- ============================================================
-- Section 4: Scoring Function (Readout)
--
-- score = w₁·relevance + w₂·recency + w₃·importance + w₄·σ(activation)
-- where wᵢ ≥ 0 and Σwᵢ = 1, and relevance, recency, importance ∈ [0, 1].
-- ============================================================

/-- Weights for the scoring function with non-negativity and sum-to-one constraints. -/
structure ScoringWeights where
  w₁ : ℝ  -- relevance
  w₂ : ℝ  -- recency
  w₃ : ℝ  -- importance
  w₄ : ℝ  -- activation
  hw₁ : 0 ≤ w₁
  hw₂ : 0 ≤ w₂
  hw₃ : 0 ≤ w₃
  hw₄ : 0 ≤ w₄
  hsum : w₁ + w₂ + w₃ + w₄ = 1

/-- The composite retrieval score. -/
def score (w : ScoringWeights) (rel rec imp act : ℝ) : ℝ :=
  w.w₁ * rel + w.w₂ * rec + w.w₃ * imp + w.w₄ * sigmoid act

/-- Score is non-negative when components are non-negative. -/
theorem score_nonneg (w : ScoringWeights) {rel rec imp act : ℝ}
    (hrel : 0 ≤ rel) (hrec : 0 ≤ rec) (himp : 0 ≤ imp) :
    0 ≤ score w rel rec imp act := by
  unfold score
  apply add_nonneg
  apply add_nonneg
  apply add_nonneg
  · exact mul_nonneg w.hw₁ hrel
  · exact mul_nonneg w.hw₂ hrec
  · exact mul_nonneg w.hw₃ himp
  · exact mul_nonneg w.hw₄ (le_of_lt (sigmoid_pos act))

/-- Score is at most 1 when components are in [0, 1]. -/
theorem score_le_one (w : ScoringWeights) {rel rec imp act : ℝ}
    (hrel : rel ∈ Icc 0 1) (hrec : rec ∈ Icc 0 1) (himp : imp ∈ Icc 0 1) :
    score w rel rec imp act ≤ 1 := by
  unfold score
  have hσ := sigmoid_mem_Icc act
  calc w.w₁ * rel + w.w₂ * rec + w.w₃ * imp + w.w₄ * sigmoid act
      ≤ w.w₁ * 1 + w.w₂ * 1 + w.w₃ * 1 + w.w₄ * 1 := by
        apply add_le_add
        apply add_le_add
        apply add_le_add
        · exact mul_le_mul_of_nonneg_left hrel.2 w.hw₁
        · exact mul_le_mul_of_nonneg_left hrec.2 w.hw₂
        · exact mul_le_mul_of_nonneg_left himp.2 w.hw₃
        · exact mul_le_mul_of_nonneg_left hσ.2 w.hw₄
    _ = w.w₁ + w.w₂ + w.w₃ + w.w₄ := by ring
    _ = 1 := w.hsum

/-- Score lies in [0, 1] — the main boundedness theorem. -/
theorem score_mem_Icc (w : ScoringWeights) {rel rec imp act : ℝ}
    (hrel : rel ∈ Icc 0 1) (hrec : rec ∈ Icc 0 1) (himp : imp ∈ Icc 0 1) :
    score w rel rec imp act ∈ Icc 0 1 :=
  ⟨score_nonneg w hrel.1 hrec.1 himp.1, score_le_one w hrel hrec himp⟩

-- ============================================================
-- Section 5: Feedback Loop (Algodonic Signal)
--
-- After each turn, importance is updated based on whether
-- the recalled memory was actually used in the response.
--
-- imp' = clamp(imp + δ·signal, 0, 1)
--
-- The clamp guarantees the invariant imp ∈ [0, 1] is maintained.
-- ============================================================

/-- Clamp a real number to the interval [0, 1]. -/
def clamp01 (x : ℝ) : ℝ := max 0 (min x 1)

/-- Importance update: imp' = clamp(imp + δ·signal, 0, 1). -/
def importanceUpdate (imp δ signal : ℝ) : ℝ :=
  clamp01 (imp + δ * signal)

/-- Clamp always produces a value in [0, 1]. -/
theorem clamp01_mem_Icc (x : ℝ) : clamp01 x ∈ Icc 0 1 := by
  constructor
  · exact le_max_left 0 (min x 1)
  · exact max_le (by norm_num) (min_le_right x 1)

/-- Updated importance is always in [0, 1]. -/
theorem importanceUpdate_mem_Icc (imp δ signal : ℝ) :
    importanceUpdate imp δ signal ∈ Icc 0 1 :=
  clamp01_mem_Icc _

-- ============================================================
-- Section 6: System Invariants
--
-- Composite properties that hold for the full dynamical system.
-- ============================================================

/-- The full system state at any point maintains bounded score,
    given that retention and importance are properly initialized
    and updated through the defined dynamics. -/
theorem system_score_bounded (w : ScoringWeights) {t S : ℝ}
    (ht : 0 ≤ t) (hS : 0 < S)
    {rel imp : ℝ} (hrel : rel ∈ Icc 0 1) (himp : imp ∈ Icc 0 1)
    (act : ℝ) :
    score w rel (retention t S) imp act ∈ Icc 0 1 :=
  score_mem_Icc w hrel (retention_mem_Icc ht hS) himp

/-- After a feedback update, the system still produces bounded scores. -/
theorem system_score_bounded_after_feedback (w : ScoringWeights) {t S : ℝ}
    (ht : 0 ≤ t) (hS : 0 < S)
    {rel imp : ℝ} (hrel : rel ∈ Icc 0 1) (_ : imp ∈ Icc 0 1)
    (δ signal act : ℝ) :
    score w rel (retention t S) (importanceUpdate imp δ signal) act ∈ Icc 0 1 :=
  score_mem_Icc w hrel (retention_mem_Icc ht hS) (importanceUpdate_mem_Icc imp δ signal)

end
