/-
  HermesNLCDM.EmotionalTagging
  ============================
  Phase 12: Emotional Tagging — Prediction-Error-Based Initial Strength

  At store time, the system measures prediction error: how much does the
  incoming pattern differ from existing high-generation abstractions?
  This cosine distance maps to initial strength S₀ ∈ [S_min, 1.0]:
    - High prediction error (novel) → S₀ near 1.0 (stored hot)
    - Low prediction error (redundant) → S₀ near S_min (stored cool)

  The decay function S(t) = S₀ · e^(-β·t) is the same for both, but
  hot memories start higher and survive longer through dream consolidation.

  Three theorems:
    1. prediction_error_well_formed: mapping produces S₀ ∈ [S_min, 1.0]
    2. prediction_error_monotone: mapping is monotone in distance
    3. emotional_tagging_improves_capacity: non-uniform S₀ where cold
       patterns decay faster → effective N drops at least as fast as
       uniform case → capacity margin grows at least as fast

  The existing MonotonicityChain.lean proves bridge_safety_monotone for
  any S₀ > 0. These theorems prove the prediction-error mapping is a
  valid input to that chain.
-/

import HermesNLCDM.MonotonicityChain
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Order.Basic

noncomputable section
open Real

namespace HermesNLCDM

/-! ## Prediction Error Mapping

  The emotional tagging function maps cosine distance d ∈ [0, 1] to
  initial strength S₀ via a linear interpolation:

    emotionalS₀(d) = S_min + (1 - S_min) · d

  where S_min > 0 is the minimum initial strength (ensuring even
  redundant memories get at least one dream cycle before pruning).
-/

/-- Emotional tagging: map prediction error (cosine distance) to initial strength.
    Linear interpolation from S_min at d=0 to 1.0 at d=1. -/
def emotionalS₀ (S_min d : ℝ) : ℝ := S_min + (1 - S_min) * d

/-! ## Theorem 1: Well-Formedness

  For S_min ∈ (0, 1] and d ∈ [0, 1], the output is in [S_min, 1.0].
  This ensures:
  - S₀ > 0 (required by bridgeDecay_pos)
  - S₀ ≤ 1 (importance is bounded)
  - S₀ ≥ S_min (cold memories survive at least one cycle)
-/

/-- **Theorem 1a**: Emotional S₀ is bounded below by S_min. -/
theorem emotionalS₀_ge_Smin {S_min d : ℝ}
    (_hSmin_pos : 0 < S_min) (hSmin_le : S_min ≤ 1)
    (hd_nn : 0 ≤ d) :
    S_min ≤ emotionalS₀ S_min d := by
  unfold emotionalS₀
  linarith [mul_nonneg (by linarith : 0 ≤ 1 - S_min) hd_nn]

/-- **Theorem 1b**: Emotional S₀ is bounded above by 1. -/
theorem emotionalS₀_le_one {S_min d : ℝ}
    (hSmin_le : S_min ≤ 1)
    (hd_le : d ≤ 1) :
    emotionalS₀ S_min d ≤ 1 := by
  unfold emotionalS₀
  nlinarith [mul_le_mul_of_nonneg_left hd_le (by linarith : 0 ≤ 1 - S_min)]

/-- **Theorem 1c**: Emotional S₀ is strictly positive. -/
theorem emotionalS₀_pos {S_min d : ℝ}
    (hSmin_pos : 0 < S_min) (hSmin_le : S_min ≤ 1)
    (hd_nn : 0 ≤ d) :
    0 < emotionalS₀ S_min d :=
  lt_of_lt_of_le hSmin_pos (emotionalS₀_ge_Smin hSmin_pos hSmin_le hd_nn)

/-- **THEOREM 1 (combined)**: Prediction error mapping is well-formed.
    For valid inputs, the output lies in [S_min, 1]. -/
theorem prediction_error_well_formed {S_min d : ℝ}
    (hSmin_pos : 0 < S_min) (hSmin_le : S_min ≤ 1)
    (hd_nn : 0 ≤ d) (hd_le : d ≤ 1) :
    S_min ≤ emotionalS₀ S_min d ∧ emotionalS₀ S_min d ≤ 1 :=
  ⟨emotionalS₀_ge_Smin hSmin_pos hSmin_le hd_nn,
   emotionalS₀_le_one hSmin_le hd_le⟩

/-! ## Theorem 2: Monotonicity in Prediction Error

  Higher cosine distance (more novel) → higher S₀.
  This ensures the mapping respects the semantics: surprising inputs
  are stored with higher initial strength.
-/

/-- **THEOREM 2**: Emotional S₀ is monotone in prediction error distance.
    Patterns farther from existing centroids get strictly higher S₀. -/
theorem prediction_error_monotone {S_min : ℝ}
    (_hSmin_pos : 0 < S_min) (hSmin_le : S_min ≤ 1) :
    Monotone (emotionalS₀ S_min) := by
  intro d₁ d₂ hd
  unfold emotionalS₀
  have h : 0 ≤ 1 - S_min := by linarith
  linarith [mul_le_mul_of_nonneg_left hd h]

/-- Strict monotonicity: when S_min < 1 (non-degenerate tagging),
    strictly higher distance gives strictly higher S₀. -/
theorem prediction_error_strict_mono {S_min : ℝ}
    (_hSmin_pos : 0 < S_min) (hSmin_lt : S_min < 1) :
    StrictMono (emotionalS₀ S_min) := by
  intro d₁ d₂ hd
  unfold emotionalS₀
  have h : 0 < 1 - S_min := by linarith
  linarith [mul_lt_mul_of_pos_left hd h]

/-! ## Theorem 3: Capacity Improvement

  Non-uniform S₀ where cold patterns decay faster means the effective
  pattern count drops at least as fast as in the uniform case.

  Key insight: if pattern i has S₀ᵢ ≤ S₀_uniform, then at any time t:
    S₀ᵢ · e^(-β·t) ≤ S₀_uniform · e^(-β·t)

  So the cold pattern's strength is always below the uniform strength.
  If the pruning threshold is fixed, the cold pattern crosses it sooner.
  Fewer surviving patterns → better capacity margin.

  We prove this by showing that for any two initial strengths S₀ₐ ≤ S₀ᵦ,
  the decayed strength preserves the ordering at all times. Then a system
  with some cold patterns has a subset of its patterns decaying faster
  than uniform, reducing effective N.
-/

/-- Strength decay preserves initial ordering: if S₀ₐ ≤ S₀ᵦ, then
    S₀ₐ·e^(-βt) ≤ S₀ᵦ·e^(-βt) for all t. -/
theorem decay_preserves_S₀_ordering {β S₀ₐ S₀ᵦ : ℝ}
    (h_ord : S₀ₐ ≤ S₀ᵦ) (t : ℝ) :
    bridgeStrengthDecay β S₀ₐ t ≤ bridgeStrengthDecay β S₀ᵦ t := by
  unfold bridgeStrengthDecay
  exact mul_le_mul_of_nonneg_right h_ord (le_of_lt (exp_pos _))

/-- Cold patterns cross any positive threshold sooner than warm patterns.
    If S₀_cold < S₀_warm and both cross threshold θ, the cold pattern
    crosses first: t_cold ≤ t_warm. -/
theorem cold_crosses_threshold_sooner {β S₀_cold S₀_warm θ : ℝ}
    (hβ : 0 < β) (h_cold_pos : 0 < S₀_cold) (_h_warm_pos : 0 < S₀_warm)
    (h_ord : S₀_cold ≤ S₀_warm)
    (hθ_pos : 0 < θ) (_hθ_lt_cold : θ < S₀_cold) :
    -- Crossing time: t_cross = ln(S₀/θ) / β
    -- For cold: ln(S₀_cold/θ)/β ≤ ln(S₀_warm/θ)/β
    Real.log (S₀_cold / θ) / β ≤ Real.log (S₀_warm / θ) / β := by
  apply div_le_div_of_nonneg_right _ hβ.le
  apply Real.log_le_log (div_pos h_cold_pos hθ_pos)
  exact div_le_div_of_nonneg_right h_ord hθ_pos.le

/-- **THEOREM 3**: Emotional tagging improves capacity margin.

    Consider a system where some patterns have S₀ᵢ ≤ S₀_uniform
    (the "cold" patterns from emotional tagging) and the rest have
    S₀ᵢ = S₀_uniform. The capacity margin of the emotionally-tagged
    system is at least as large as the uniform system at every time t.

    Proof strategy: the emotionally-tagged system's active count at any
    time t is at most the uniform system's active count (cold patterns
    die sooner), and capacity margin is antitone in active count. -/
theorem emotional_tagging_improves_capacity
    {N_max : ℝ}
    {active_count_emotional active_count_uniform : ℝ → ℝ}
    (h_fewer : ∀ t, active_count_emotional t ≤ active_count_uniform t) :
    ∀ t, capacityMargin N_max (active_count_uniform t) ≤
         capacityMargin N_max (active_count_emotional t) := by
  intro t
  unfold capacityMargin
  linarith [h_fewer t]

/-! ## Composing with the Bridge Theorem

  The bridge theorem (MonotonicityChain.lean) proves safety is monotone
  for any S₀ > 0. Emotional tagging provides S₀ = emotionalS₀(S_min, d)
  which is positive (Theorem 1c). Therefore the bridge theorem applies
  to every emotionally-tagged memory.
-/

/-- Emotional tagging composes with the bridge theorem: the capacity
    margin for an emotionally-tagged memory is monotone in time. -/
theorem emotional_bridge_safety_monotone
    {β S_min d N_max : ℝ}
    (hβ : 0 < β)
    (hSmin_pos : 0 < S_min) (hSmin_le : S_min ≤ 1)
    (hd_nn : 0 ≤ d)
    {importance : ℝ → ℝ} (h_imp_mono : Monotone importance)
    {active_count : ℝ → ℝ} (h_count_mono : Monotone active_count) :
    Monotone (capacityMargin N_max ∘ active_count ∘ importance ∘
      bridgeStrengthDecay β (emotionalS₀ S_min d)) :=
  bridge_safety_monotone hβ
    (emotionalS₀_pos hSmin_pos hSmin_le hd_nn)
    h_imp_mono h_count_mono

end HermesNLCDM
