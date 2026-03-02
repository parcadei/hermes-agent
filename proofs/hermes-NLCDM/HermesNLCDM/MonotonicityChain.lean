/-
  HermesNLCDM.MonotonicityChain
  =============================
  Bridge Theorem: System safety is monotonically non-decreasing in time.

  This proves the central safety property connecting hermes-memory (V1)
  strength decay dynamics to hermes-NLCDM (V2) capacity theorem:

    As time increases, old patterns decay, importances drop,
    dream pruning removes weak patterns, N decreases,
    and the capacity margin exp(βδ)/(4βM²) − N grows.

  The proof is a 4-link monotonicity composition chain:
    Link 1: strengthDecay is Antitone in t           (restated from hermes-memory)
    Link 2: importance is Monotone in strength        (behavioral contract)
    Link 3: active_count is Monotone in importance    (pruning contract)
    Link 4: capacity_margin is Antitone in N          (algebraic)

  Composition:
    capacity_margin ∘ active_count ∘ importance ∘ decay
    = Antitone ∘ (Monotone ∘ Monotone ∘ Antitone)
    = Antitone ∘ Antitone
    = Monotone

  ⟹ capacity margin is Monotone in time (safety improves).

  Mathlib composition rules (from Mathlib.Order.Monotone.Defs):
    Monotone.comp_antitone : Monotone g → Antitone f → Antitone (g ∘ f)
    Antitone.comp          : Antitone g → Antitone f → Monotone (g ∘ f)
-/

import Mathlib.Order.Monotone.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import HermesNLCDM.Capacity

noncomputable section
open Real

namespace HermesNLCDM

/-! ## Link 1: Strength Decay (Antitone in time)

  S(t) = S₀ · e^(-β·t).
  Restated from hermes-memory/HermesMemory/StrengthDecay.lean.
  The original proof lives in the hermes-memory Lean project;
  we restate it here since the two are separate Lake packages.
-/

/-- Strength decay between access events: S(t) = S₀ · e^(-β·t).
    Direct translation from hermes-memory/core.py:strength_decay. -/
def bridgeStrengthDecay (β S₀ t : ℝ) : ℝ := S₀ * exp (-β * t)

/-- Decayed strength is always positive for positive initial strength. -/
theorem bridgeDecay_pos {β S₀ : ℝ} (hS₀ : 0 < S₀) (t : ℝ) :
    0 < bridgeStrengthDecay β S₀ t := by
  unfold bridgeStrengthDecay
  exact mul_pos hS₀ (exp_pos _)

/-- At time zero, decayed strength equals initial strength. -/
theorem bridgeDecay_at_zero (β S₀ : ℝ) :
    bridgeStrengthDecay β S₀ 0 = S₀ := by
  unfold bridgeStrengthDecay; simp

/-- **Link 1**: Strength decay is antitone in time.
    For positive decay rate β and initial strength S₀, strength is
    monotonically non-increasing as time progresses. -/
theorem bridgeDecay_antitone {β S₀ : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀) :
    Antitone (bridgeStrengthDecay β S₀) := by
  intro t₁ t₂ ht
  unfold bridgeStrengthDecay
  apply mul_le_mul_of_nonneg_left _ hS₀.le
  exact exp_le_exp.mpr (by nlinarith)

/-! ## Links 2–3: Importance and Active Count (Monotone, abstract)

  Link 2: The importance assigned to a memory is monotone in its strength.
  Stronger memories receive higher importance in the dream cycle.
  (Behavioral contract from dream-redesign-spec.md.)

  Link 3: The number of patterns surviving dream pruning is monotone in
  per-pattern importance. In dream_ops.py, nrem_prune_xb removes patterns
  below an importance threshold — higher importance means more survivors.

  Both are taken as hypotheses rather than proved from specific formulas,
  since the exact functions may evolve while the monotonicity contracts
  are stable architectural invariants.
-/

/-! ## Link 4: Capacity Margin (Antitone in N)

  capacity_margin(N) = N_max − N.
  Trivially antitone: more patterns means less headroom.
  Connects to Phase 4 (Capacity.lean): the exponential capacity bound
  4NβM² ≤ exp(βδ) is equivalent to N ≤ N_max where N_max = exp(βδ)/(4βM²).
-/

/-- Capacity margin: headroom between maximum capacity and current load.
    Positive margin ↔ capacity condition satisfied. -/
def capacityMargin (N_max : ℝ) (N : ℝ) : ℝ := N_max - N

/-- **Link 4**: Capacity margin is antitone in pattern count.
    More stored patterns → less headroom → smaller margin. -/
theorem capacityMargin_antitone {N_max : ℝ} :
    Antitone (capacityMargin N_max) := by
  intro N₁ N₂ hN
  unfold capacityMargin
  linarith

/-! ## Composed Links 1–3: Importance Over Time (Antitone)

  importance ∘ decay                   : Antitone (by Monotone.comp_antitone)
  active_count ∘ importance ∘ decay    : Antitone (by Monotone.comp_antitone)

  As time progresses, both importance and active pattern count
  are non-increasing.
-/

/-- Importance decreases over time: composing monotone importance
    with antitone strength decay yields an antitone function. -/
theorem importance_over_time_antitone
    {β S₀ : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance) :
    Antitone (importance ∘ bridgeStrengthDecay β S₀) :=
  h_imp_mono.comp_antitone (bridgeDecay_antitone hβ hS₀)

/-- Active pattern count decreases over time: composing monotone
    active_count with the antitone importance-over-time chain. -/
theorem active_count_over_time_antitone
    {β S₀ : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance)
    {active_count : ℝ → ℝ} (h_count_mono : Monotone active_count) :
    Antitone (active_count ∘ importance ∘ bridgeStrengthDecay β S₀) :=
  h_count_mono.comp_antitone (importance_over_time_antitone hβ hS₀ h_imp_mono)

/-! ## The Bridge Theorem: Safety Improves Over Time

  Composing all 4 links:
    capacity_margin ∘ active_count ∘ importance ∘ decay

  is **Monotone** in time (by Antitone.comp: Antitone ∘ Antitone = Monotone).

  Interpretation: as time progresses, the capacity margin grows.
  The system becomes safer because decaying patterns get pruned,
  reducing N and increasing headroom under the exponential capacity
  bound N_max = exp(βδ)/(4βM²).
-/

/-- **BRIDGE THEOREM**: System safety is monotonically non-decreasing in time.

    Given:
    - Strength decays exponentially (β > 0, S₀ > 0)
    - Importance is monotone in strength (behavioral contract)
    - Active pattern count is monotone in importance (pruning contract)
    - Capacity margin is antitone in pattern count (algebraic)

    Then: capacity margin is monotone in time — safety improves as
    old patterns decay and get pruned by the dream cycle.

    This is the formal bridge between hermes-memory V1 dynamics
    and hermes-NLCDM V2 capacity guarantees. -/
theorem bridge_safety_monotone
    {β S₀ N_max : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance)
    {active_count : ℝ → ℝ} (h_count_mono : Monotone active_count) :
    Monotone (capacityMargin N_max ∘ active_count ∘ importance ∘
      bridgeStrengthDecay β S₀) :=
  capacityMargin_antitone.comp
    (active_count_over_time_antitone hβ hS₀ h_imp_mono h_count_mono)

/-! ## Corollaries -/

/-- If the system starts safe (capacity margin ≥ 0 at time t₀),
    it remains safe for all future times t ≥ t₀. -/
theorem safety_preserved
    {β S₀ N_max : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance)
    {active_count : ℝ → ℝ} (h_count_mono : Monotone active_count)
    {t₀ t : ℝ} (ht : t₀ ≤ t)
    (h_safe : 0 ≤ (capacityMargin N_max ∘ active_count ∘ importance ∘
      bridgeStrengthDecay β S₀) t₀) :
    0 ≤ (capacityMargin N_max ∘ active_count ∘ importance ∘
      bridgeStrengthDecay β S₀) t :=
  le_trans h_safe (bridge_safety_monotone hβ hS₀ h_imp_mono h_count_mono ht)

/-- The full pipeline is antitone through the first three links:
    the number of active patterns is non-increasing over time.
    This is the "N decreases" intermediate result. -/
theorem pattern_count_nonincreasing
    {β S₀ : ℝ} (hβ : 0 < β) (hS₀ : 0 < S₀)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance)
    {active_count : ℝ → ℝ} (h_count_mono : Monotone active_count)
    {t₁ t₂ : ℝ} (ht : t₁ ≤ t₂) :
    (active_count ∘ importance ∘ bridgeStrengthDecay β S₀) t₂ ≤
    (active_count ∘ importance ∘ bridgeStrengthDecay β S₀) t₁ :=
  active_count_over_time_antitone hβ hS₀ h_imp_mono h_count_mono ht

/-- General monotonicity chain composition: given any alternating
    Antitone-Monotone-Monotone-Antitone chain of four functions,
    the full composition is Monotone.
    This abstracts the bridge theorem pattern for reuse. -/
theorem monotonicity_chain_AMMA
    {α : Type*} [Preorder α]
    {β₁ : Type*} [Preorder β₁]
    {β₂ : Type*} [Preorder β₂]
    {β₃ : Type*} [Preorder β₃]
    {γ : Type*} [Preorder γ]
    {f₁ : α → β₁} {f₂ : β₁ → β₂} {f₃ : β₂ → β₃} {f₄ : β₃ → γ}
    (h₁ : Antitone f₁) (h₂ : Monotone f₂) (h₃ : Monotone f₃) (h₄ : Antitone f₄) :
    Monotone (f₄ ∘ f₃ ∘ f₂ ∘ f₁) :=
  h₄.comp (h₃.comp_antitone (h₂.comp_antitone h₁))

end HermesNLCDM
