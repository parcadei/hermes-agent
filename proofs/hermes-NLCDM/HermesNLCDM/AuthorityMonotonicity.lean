/-
  HermesNLCDM.AuthorityMonotonicity
  ==================================
  Phase 15: 4D Importance Composition — Authority Monotonicity

  The importance assigned to a memory is computed via headroom-fill
  composition across four dimensions:
    D1 (provenance): layer_seed floor based on cognitive layer
    D2 (content):    category boost fills headroom above seed
    D4 (authority):  user directive authority fills remaining headroom
    (D3 novelty applied externally via EmotionalTagging.lean)

  The core operation is the **headroom fill**:
    headroomFill(base, fill) = base + fill · (1 − base)

  Each dimension fills a fraction of remaining headroom above the
  previous floor, preserving all prior dimensions as floors:
    step 1: imp = seed + boost · (1 − seed)           (D1 × D2)
    step 2: imp += authority · (1 − imp)                (D4 fills remaining)

  Direct translation of coupled_engine.py:_resolve_layer_importance.

  This file proves:
    1. headroomFill well-formedness and monotonicity
    2. importance4D well-formedness (output ∈ [S_min, 1])
    3. importance4D monotonicity in each dimension independently
    4. Authority dominance: sufficient authority guarantees user
       memories beat non-user same-category memories
    5. Concrete parameter verification for NLCDM layer seeds
    6. Composition with MonotonicityChain Link 2
-/

import HermesNLCDM.MonotonicityChain

noncomputable section

namespace HermesNLCDM

/-! ## Headroom Fill: The Core Operation -/

/-- Headroom fill: base + fill fraction of remaining headroom to 1.0.
    Direct translation from coupled_engine.py:_resolve_layer_importance. -/
def headroomFill (base fill : ℝ) : ℝ := base + fill * (1 - base)

/-- Headroom fill preserves the base as a floor. -/
theorem headroomFill_ge_base {base fill : ℝ}
    (hf : 0 ≤ fill) (hb : base ≤ 1) :
    base ≤ headroomFill base fill := by
  unfold headroomFill
  linarith [mul_nonneg hf (by linarith : 0 ≤ 1 - base)]

/-- Headroom fill output is bounded above by 1. -/
theorem headroomFill_le_one {base fill : ℝ}
    (hb : base ≤ 1) (hf : fill ≤ 1) :
    headroomFill base fill ≤ 1 := by
  unfold headroomFill
  have h : 0 ≤ 1 - base := by linarith
  linarith [mul_le_mul_of_nonneg_right hf h]

/-- Headroom fill with fill = 0 returns the base unchanged. -/
theorem headroomFill_zero (base : ℝ) :
    headroomFill base 0 = base := by
  unfold headroomFill; ring

/-- Headroom fill with fill = 1 saturates to 1. -/
theorem headroomFill_one (base : ℝ) :
    headroomFill base 1 = 1 := by
  unfold headroomFill; ring

/-- Headroom fill is monotone in the base for fixed fill ≤ 1.
    Higher base → higher output (start from a higher floor). -/
theorem headroomFill_mono_base {fill : ℝ} (hf : fill ≤ 1) :
    Monotone (fun b => headroomFill b fill) := by
  intro b₁ b₂ hb
  unfold headroomFill
  nlinarith [mul_le_mul_of_nonneg_right hb (by linarith : 0 ≤ 1 - fill)]

/-- Headroom fill is monotone in the fill for fixed base ≤ 1.
    Higher fill → more headroom consumed → higher output. -/
theorem headroomFill_mono_fill {base : ℝ} (hb : base ≤ 1) :
    Monotone (fun f => headroomFill base f) := by
  intro f₁ f₂ hf
  unfold headroomFill
  linarith [mul_le_mul_of_nonneg_right hf (by linarith : 0 ≤ 1 - base)]

/-- Strict monotonicity in fill when base < 1 (non-degenerate). -/
theorem headroomFill_strict_mono_fill {base : ℝ} (hb : base < 1) :
    StrictMono (fun f => headroomFill base f) := by
  intro f₁ f₂ hf
  unfold headroomFill
  linarith [mul_lt_mul_of_pos_right hf (by linarith : 0 < 1 - base)]

/-- Headroom fill output is positive when base is positive. -/
theorem headroomFill_pos {base fill : ℝ}
    (hb_pos : 0 < base) (hf : 0 ≤ fill) (hb_le : base ≤ 1) :
    0 < headroomFill base fill :=
  lt_of_lt_of_le hb_pos (headroomFill_ge_base hf hb_le)

/-! ## 4D Importance Composition

  Two headroom fills composing seed (D1 provenance), boost (D2 content),
  and authority (D4 user directive). D3 (novelty) is applied externally
  by EmotionalTagging.lean at store time.

    importance4D(seed, boost, authority) =
      headroomFill(headroomFill(seed, boost), authority)
-/

/-- 4D importance: double headroom fill composing provenance seed,
    content boost, and authority fill.
    Direct translation of coupled_engine.py:_resolve_layer_importance. -/
def importance4D (seed boost authority : ℝ) : ℝ :=
  headroomFill (headroomFill seed boost) authority

/-! ### Well-Formedness -/

/-- 4D importance is bounded below by the layer seed (D1 floor never erased). -/
theorem importance4D_ge_seed {seed boost authority : ℝ}
    (hb_nn : 0 ≤ boost) (hb_le : boost ≤ 1)
    (hα_nn : 0 ≤ authority) (hs_le : seed ≤ 1) :
    seed ≤ importance4D seed boost authority := by
  unfold importance4D
  exact le_trans (headroomFill_ge_base hb_nn hs_le)
    (headroomFill_ge_base hα_nn (headroomFill_le_one hs_le hb_le))

/-- 4D importance is bounded above by 1. -/
theorem importance4D_le_one {seed boost authority : ℝ}
    (hs_le : seed ≤ 1) (hb_le : boost ≤ 1) (hα_le : authority ≤ 1) :
    importance4D seed boost authority ≤ 1 := by
  unfold importance4D
  exact headroomFill_le_one (headroomFill_le_one hs_le hb_le) hα_le

/-- **4D Well-Formedness**: output ∈ [S_min, 1] for valid inputs. -/
theorem importance4D_well_formed {S_min seed boost authority : ℝ}
    (hSmin : S_min ≤ seed) (hs_le : seed ≤ 1)
    (hb_nn : 0 ≤ boost) (hb_le : boost ≤ 1)
    (hα_nn : 0 ≤ authority) (hα_le : authority ≤ 1) :
    S_min ≤ importance4D seed boost authority ∧
    importance4D seed boost authority ≤ 1 :=
  ⟨le_trans hSmin (importance4D_ge_seed hb_nn hb_le hα_nn hs_le),
   importance4D_le_one hs_le hb_le hα_le⟩

/-- 4D importance is strictly positive when seed is positive. -/
theorem importance4D_pos {seed boost authority : ℝ}
    (hs_pos : 0 < seed) (hs_le : seed ≤ 1)
    (hb_nn : 0 ≤ boost) (hb_le : boost ≤ 1)
    (hα_nn : 0 ≤ authority) :
    0 < importance4D seed boost authority :=
  lt_of_lt_of_le hs_pos (importance4D_ge_seed hb_nn hb_le hα_nn hs_le)

/-! ### Per-Dimension Monotonicity -/

/-- **D1 Monotonicity**: 4D importance is monotone in seed (provenance).
    Higher layer seed → higher importance, preserving provenance ordering. -/
theorem importance4D_mono_seed {boost authority : ℝ}
    (hb : boost ≤ 1) (hα : authority ≤ 1) :
    Monotone (fun s => importance4D s boost authority) := by
  unfold importance4D
  exact (headroomFill_mono_base hα).comp (headroomFill_mono_base hb)

/-- **D2 Monotonicity**: 4D importance is monotone in boost (content).
    Higher category boost → higher importance within any layer. -/
theorem importance4D_mono_boost {seed authority : ℝ}
    (hs : seed ≤ 1) (hα : authority ≤ 1) :
    Monotone (fun b => importance4D seed b authority) := by
  unfold importance4D
  exact (headroomFill_mono_base hα).comp (headroomFill_mono_fill hs)

/-- **D4 Monotonicity**: 4D importance is monotone in authority.
    Higher authority fill → higher importance for user directives. -/
theorem importance4D_mono_authority {seed boost : ℝ}
    (hs : seed ≤ 1) (hb : boost ≤ 1) :
    Monotone (fun a => importance4D seed boost a) := by
  unfold importance4D
  exact headroomFill_mono_fill (headroomFill_le_one hs hb)

/-! ## Authority Dominance

  The central theorem: user_knowledge memories (lower seed S_u but with
  authority fill α) dominate non-user memories (higher seed S_p but no
  authority) when α is sufficiently large.

  The dominance threshold is: α · (1 − S_u) ≥ S_p − S_u.

  For NLCDM parameters S_u = 0.5 (user_knowledge) and S_p = 0.8
  (procedural), the threshold is 0.3 / 0.5 = 0.6. Our authority fills
  (correction = 0.9, instruction = 0.8, preference = 0.6) all meet it.

  Proof: RHS − LHS factors as (1 − boost) · [α · (1 − S_u) − (S_p − S_u)]
  which is ≥ 0 when the threshold condition holds and boost ≤ 1.
-/

/-- **Authority Dominance Theorem**: If authority fill α satisfies
    α · (1 − S_u) ≥ S_p − S_u, then user importance (seed S_u, authority α)
    dominates non-user importance (seed S_p, no authority) for any shared
    category boost. -/
theorem authority_dominance
    {S_u S_p boost authority : ℝ}
    (_hSu_lt : S_u < 1)
    (hb_le : boost ≤ 1)
    (hα_sufficient : S_p - S_u ≤ authority * (1 - S_u)) :
    importance4D S_p boost 0 ≤ importance4D S_u boost authority := by
  unfold importance4D headroomFill
  -- Goal: S_p + boost*(1-S_p) + 0*(1-(S_p+boost*(1-S_p)))
  --     ≤ S_u + boost*(1-S_u) + authority*(1-(S_u+boost*(1-S_u)))
  -- Factors as: (1-boost) · [α·(1-S_u) - (S_p-S_u)] ≥ 0
  nlinarith [mul_le_mul_of_nonneg_right hα_sufficient
    (by linarith : 0 ≤ 1 - boost)]

/-! ### Concrete Parameter Verification -/

/-- user_correction (0.5, 0.9, 0.9) = 0.995 dominates
    procedural_correction (0.8, 0.9, 0) = 0.98. -/
theorem user_correction_dominates :
    importance4D 0.8 0.9 0 ≤ importance4D 0.5 0.9 0.9 := by
  unfold importance4D headroomFill; norm_num

/-- user_instruction (0.5, 0.85, 0.8) = 0.985 dominates
    procedural_correction (0.8, 0.9, 0) = 0.98. -/
theorem user_instruction_dominates :
    importance4D 0.8 0.9 0 ≤ importance4D 0.5 0.85 0.8 := by
  unfold importance4D headroomFill; norm_num

/-- Authority threshold α ≥ 0.6 suffices for user (seed 0.5)
    to dominate procedural (seed 0.8) at any category boost. -/
theorem authority_threshold_06 {boost : ℝ}
    (hb_le : boost ≤ 1) :
    importance4D 0.8 boost 0 ≤ importance4D 0.5 boost 0.6 := by
  exact authority_dominance (by norm_num : (0.5 : ℝ) < 1) hb_le
    (by norm_num : (0.8 : ℝ) - 0.5 ≤ 0.6 * (1 - 0.5))

/-! ## Composition with MonotonicityChain Link 2

  The bridge theorem (MonotonicityChain.lean) takes `Monotone importance`
  as a hypothesis in Link 2: importance is monotone in strength.

  The 4D composition provides a valid importance function: when the
  seed is derived from strength via any monotone mapping, the composed
  function is monotone, satisfying Link 2.
-/

/-- The 4D importance, composed with a monotone strength-to-seed mapping,
    is monotone in strength — validating MonotonicityChain Link 2. -/
theorem importance4D_validates_link2
    {boost authority : ℝ}
    (hb : boost ≤ 1) (hα : authority ≤ 1)
    {strength_to_seed : ℝ → ℝ} (h_mono : Monotone strength_to_seed) :
    Monotone (fun S => importance4D (strength_to_seed S) boost authority) :=
  (importance4D_mono_seed hb hα).comp h_mono

/-- The 4D importance as a linear scaling factor of strength is monotone.
    This is the simplest Link 2 instantiation: importance(S) = I₄D · S. -/
theorem importance4D_scaling_monotone {seed boost authority : ℝ}
    (hs_pos : 0 < seed) (hs_le : seed ≤ 1)
    (hb_nn : 0 ≤ boost) (hb_le : boost ≤ 1)
    (hα_nn : 0 ≤ authority) :
    Monotone (fun S => importance4D seed boost authority * S) :=
  fun _ _ hS => mul_le_mul_of_nonneg_left hS
    (le_of_lt (importance4D_pos hs_pos hs_le hb_nn hb_le hα_nn))

end HermesNLCDM
