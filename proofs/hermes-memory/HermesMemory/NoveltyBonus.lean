/-
  Hermes Memory System — Novelty Bonus (Anti-Cold-Start)

  Problem: New memories start with low strength S and must compete with
  established memories that have accumulated high S through repeated access.
  If they're never recalled (because they score too low), they never get
  reinforced and decay to zero. Valuable memories die before getting a chance.

  Fix: Add a decaying novelty bonus to the score of recently created memories:
    novelty(t) = N₀ · e^(-γ·t)   where t = time since creation
    boosted_score = base_score + novelty(t)

  Key results:
  - Exploration window: for time W = ln(N₀/ε) / γ, the bonus exceeds ε
  - The bonus decays to zero, so it doesn't permanently distort ranking
  - New memories with zero base score still have boosted_score > 0
-/

import HermesMemory.MemoryDynamics

noncomputable section

open Real Set Filter

-- ============================================================
-- Section 9: Novelty Bonus (Anti-Cold-Start Starvation)
--
-- A decaying bonus added to new memories' scores to guarantee
-- they survive long enough to be evaluated by the recall system.
-- ============================================================

/-- Novelty bonus: decays exponentially from initial value N₀ with rate γ. -/
def noveltyBonus (N₀ γ t : ℝ) : ℝ := N₀ * exp (-γ * t)

/-- Novelty bonus is always strictly positive for positive initial bonus. -/
theorem noveltyBonus_pos {N₀ γ : ℝ} (hN₀ : 0 < N₀) (t : ℝ) :
    0 < noveltyBonus N₀ γ t := by
  unfold noveltyBonus
  exact mul_pos hN₀ (exp_pos _)

/-- At creation time, the novelty bonus equals its initial value. -/
theorem noveltyBonus_at_zero (N₀ γ : ℝ) : noveltyBonus N₀ γ 0 = N₀ := by
  unfold noveltyBonus; simp

/-- The novelty bonus is bounded by its initial value. -/
theorem noveltyBonus_le_init {N₀ γ t : ℝ} (hγ : 0 < γ) (ht : 0 ≤ t) (hN₀ : 0 < N₀) :
    noveltyBonus N₀ γ t ≤ N₀ := by
  unfold noveltyBonus
  have hexp : exp (-γ * t) ≤ 1 := by
    rw [← exp_zero]
    exact exp_le_exp.mpr (by nlinarith)
  calc N₀ * exp (-γ * t) ≤ N₀ * 1 :=
        mul_le_mul_of_nonneg_left hexp hN₀.le
    _ = N₀ := mul_one N₀

/-- The novelty bonus decays monotonically over time. -/
theorem noveltyBonus_antitone {N₀ γ : ℝ} (hγ : 0 < γ) (hN₀ : 0 < N₀) :
    Antitone (noveltyBonus N₀ γ) := by
  intro t₁ t₂ ht
  unfold noveltyBonus
  apply mul_le_mul_of_nonneg_left _ hN₀.le
  apply exp_le_exp.mpr
  nlinarith

/-- The novelty bonus decays to zero — it doesn't permanently distort scores. -/
theorem noveltyBonus_tendsto_zero {N₀ γ : ℝ} (hγ : 0 < γ) :
    Tendsto (noveltyBonus N₀ γ) atTop (nhds 0) := by
  unfold noveltyBonus
  have h_lin : Tendsto (fun t : ℝ => -γ * t) atTop atBot := by
    rw [Filter.tendsto_atBot]
    intro b
    rw [Filter.eventually_atTop]
    exact ⟨-b / γ, fun t ht => by
      have hbt : γ * (-b / γ) = -b := by field_simp
      have hmul : γ * (-b / γ) ≤ γ * t := mul_le_mul_of_nonneg_left ht hγ.le
      linarith⟩
  have h_exp : Tendsto (fun t : ℝ => exp (-γ * t)) atTop (nhds 0) :=
    tendsto_exp_atBot.comp h_lin
  have h_const : Tendsto (fun _ : ℝ => N₀) atTop (nhds N₀) := tendsto_const_nhds
  have := h_const.mul h_exp
  rwa [mul_zero] at this

-- ============================================================
-- Section 9b: Exploration Window Guarantee
--
-- The exploration window W = ln(N₀/ε) / γ is the time during which
-- the novelty bonus exceeds threshold ε. This guarantees new memories
-- get a chance to be recalled before the bonus fades.
-- ============================================================

/-- The guaranteed exploration window: time during which novelty bonus ≥ ε. -/
def explorationWindow (N₀ γ ε : ℝ) : ℝ := log (N₀ / ε) / γ

/-- The exploration window is positive when the bonus starts above the threshold. -/
theorem explorationWindow_pos {N₀ γ ε : ℝ} (hN₀ : 0 < N₀) (hγ : 0 < γ)
    (hε : 0 < ε) (hεN : ε < N₀) :
    0 < explorationWindow N₀ γ ε := by
  unfold explorationWindow
  apply div_pos _ hγ
  rw [Real.log_pos_iff (div_pos hN₀ hε).le]
  exact (one_lt_div hε).mpr hεN

/-- **Anti-Cold-Start Theorem**: Within the exploration window,
    the novelty bonus exceeds the threshold ε.
    This guarantees new memories survive long enough to be evaluated. -/
theorem noveltyBonus_above_threshold {N₀ γ ε t : ℝ}
    (hN₀ : 0 < N₀) (hγ : 0 < γ) (hε : 0 < ε) (_ : ε < N₀)
    (_ : 0 ≤ t) (ht : t ≤ explorationWindow N₀ γ ε) :
    ε ≤ noveltyBonus N₀ γ t := by
  unfold noveltyBonus explorationWindow at *
  -- Need: ε ≤ N₀ * exp(-γ*t)
  -- Given: t ≤ log(N₀/ε) / γ
  -- So: γ*t ≤ log(N₀/ε)
  -- So: -γ*t ≥ -log(N₀/ε)
  -- So: exp(-γ*t) ≥ exp(-log(N₀/ε)) = ε/N₀
  -- So: N₀ * exp(-γ*t) ≥ N₀ * ε/N₀ = ε  ✓
  have hN₀_ne : N₀ ≠ 0 := ne_of_gt hN₀
  have hε_ne : ε ≠ 0 := ne_of_gt hε
  have hγt : γ * t ≤ log (N₀ / ε) := by
    have := mul_le_mul_of_nonneg_left ht hγ.le
    rwa [mul_div_cancel₀ (log (N₀ / ε)) (ne_of_gt hγ)] at this
  have hexp : exp (-log (N₀ / ε)) ≤ exp (-γ * t) := by
    apply exp_le_exp.mpr; linarith
  have hexp_log : exp (-log (N₀ / ε)) = ε / N₀ := by
    rw [exp_neg, exp_log (div_pos hN₀ hε), inv_div]
  calc ε = N₀ * (ε / N₀) := by field_simp
    _ = N₀ * exp (-log (N₀ / ε)) := by rw [hexp_log]
    _ ≤ N₀ * exp (-γ * t) := mul_le_mul_of_nonneg_left hexp hN₀.le

/-- Boosted score: base score plus novelty bonus. -/
def boostedScore (baseScore N₀ γ t : ℝ) : ℝ := baseScore + noveltyBonus N₀ γ t

/-- Boosted score is at least the novelty bonus (base score ≥ 0). -/
theorem boostedScore_ge_novelty {baseScore N₀ γ : ℝ} (hbase : 0 ≤ baseScore) (t : ℝ) :
    noveltyBonus N₀ γ t ≤ boostedScore baseScore N₀ γ t := by
  unfold boostedScore
  linarith

/-- Boosted score is at least the base score (novelty ≥ 0). -/
theorem boostedScore_ge_base {N₀ γ : ℝ} (hN₀ : 0 < N₀) (baseScore t : ℝ) :
    baseScore ≤ boostedScore baseScore N₀ γ t := by
  unfold boostedScore
  linarith [@noveltyBonus_pos N₀ γ hN₀ t]

/-- Within the exploration window, even a memory with zero base score
    has a boosted score above the threshold. -/
theorem coldStart_survival {N₀ γ ε t : ℝ}
    (hN₀ : 0 < N₀) (hγ : 0 < γ) (hε : 0 < ε) (hεN : ε < N₀)
    (ht₀ : 0 ≤ t) (ht : t ≤ explorationWindow N₀ γ ε) :
    ε ≤ boostedScore 0 N₀ γ t := by
  unfold boostedScore
  simp
  exact noveltyBonus_above_threshold hN₀ hγ hε hεN ht₀ ht

end
