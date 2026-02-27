/-
  Hermes Memory System — Contraction Wiring

  Wires the Lipschitz bounds from ComposedSystem.lean into Mathlib's
  ContractingWith.fixedPoint (Banach fixed-point theorem) to obtain:

  1. EXISTENCE of a unique stationary state S* ∈ [0, Smax]²
  2. UNIQUENESS: no other fixed point exists in the domain
  3. CONVERGENCE: iterating the mean-field map from any S₀ ∈ [0, Smax]²
     converges to S* (global asymptotic stability)

  This establishes that the composed memory system's mean-field dynamics
  have a unique stable distribution — the deterministic analog of a
  stationary distribution for the underlying Markov chain.

  Approach:
  - Define the domain D = [0, Smax]² ⊆ ℝ × ℝ
  - Show D is closed (hence complete) and forward-invariant under T
  - Prove T is K-Lipschitz on D with K = exp(-βΔ) + L·α·Smax
  - When K < 1, apply ContractingWith on the restriction T|_D
  - Extract existence, uniqueness, convergence from Mathlib
-/

import HermesMemory.ComposedSystem
import Mathlib.Topology.MetricSpace.Contracting

noncomputable section

open Real Set Filter NNReal Function Topology

-- ============================================================
-- Section 13: Domain Setup
-- ============================================================

/-- The composed domain [0, Smax]² ⊆ ℝ × ℝ. -/
def composedDomain (Smax : ℝ) : Set (ℝ × ℝ) :=
  Icc 0 Smax ×ˢ Icc 0 Smax

theorem composedDomain_isClosed {Smax : ℝ} :
    IsClosed (composedDomain Smax) :=
  isClosed_Icc.prod isClosed_Icc

theorem composedDomain_isComplete {Smax : ℝ} :
    IsComplete (composedDomain Smax) :=
  composedDomain_isClosed.isComplete

theorem composedDomain_nonempty {Smax : ℝ} (hSmax : 0 ≤ Smax) :
    (composedDomain Smax).Nonempty :=
  ⟨(0, 0), ⟨left_mem_Icc.mpr hSmax, left_mem_Icc.mpr hSmax⟩⟩

-- ============================================================
-- Section 14: Forward Invariance (MapsTo)
-- ============================================================

/-- The composed mean-field map preserves the domain [0, Smax]². -/
theorem composedExpectedMap_mapsTo {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ) (hSmax : 0 < Smax)
    (hq_range : ∀ s₁ s₂, 0 ≤ selectProb s₁ s₂ ∧ selectProb s₁ s₂ ≤ 1) :
    MapsTo (composedExpectedMap selectProb α β Δ Smax)
           (composedDomain Smax) (composedDomain Smax) := by
  intro S hS
  obtain ⟨⟨hS₁₀, hS₁⟩, ⟨hS₂₀, hS₂⟩⟩ := hS
  set p := selectProb S.1 S.2
  have hp := hq_range S.1 S.2
  constructor
  · constructor
    · exact expectedStrengthUpdate_nonneg hα₀ hα₁ hp.1 hp.2 hSmax hS₁₀
    · exact expectedStrengthUpdate_le_Smax hα₀ hα₁ hβ hΔ hp.1 hp.2 hS₁₀ hS₁
  · have h1p₀ : 0 ≤ 1 - p := by linarith [hp.2]
    have h1p₁ : 1 - p ≤ 1 := by linarith [hp.1]
    constructor
    · exact expectedStrengthUpdate_nonneg hα₀ hα₁ h1p₀ h1p₁ hSmax hS₂₀
    · exact expectedStrengthUpdate_le_Smax hα₀ hα₁ hβ hΔ h1p₀ h1p₁ hS₂₀ hS₂

-- ============================================================
-- Section 15: Lipschitz Bound on Domain
--
-- On [0,Smax]², the composed map satisfies
--   dist(T(S), T(S')) ≤ K · dist(S, S')
-- where K = exp(-βΔ) + L·α·Smax and dist is the max (L∞) metric.
-- ============================================================

/-- Lipschitz bound for the composed map on the domain, in terms of
    the product (max) metric on ℝ × ℝ. -/
theorem composedExpectedMap_lipschitz_on_domain
    {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax L : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax) (_ : 0 ≤ L)
    (hLip : ∀ s₁ s₂ s₁' s₂',
      |selectProb s₁ s₂ - selectProb s₁' s₂'| ≤ L * max (|s₁ - s₁'|) (|s₂ - s₂'|))
    (hq_range : ∀ s₁ s₂, 0 ≤ selectProb s₁ s₂ ∧ selectProb s₁ s₂ ≤ 1)
    {S S' : ℝ × ℝ} (hS' : S' ∈ composedDomain Smax) :
    dist (composedExpectedMap selectProb α β Δ Smax S)
         (composedExpectedMap selectProb α β Δ Smax S')
    ≤ (exp (-β * Δ) + L * α * Smax) * dist S S' := by
  obtain ⟨⟨hS'₁₀, hS'₁⟩, ⟨hS'₂₀, hS'₂⟩⟩ := hS'
  set p := selectProb S.1 S.2
  set p' := selectProb S'.1 S'.2
  set K := exp (-β * Δ) + L * α * Smax
  set d := dist S S'
  have hp := hq_range S.1 S.2
  -- dist on product = max of component distances
  have hd_eq : d = max (dist S.1 S'.1) (dist S.2 S'.2) := Prod.dist_eq
  -- Convert component distances to absolute values
  have hd_abs : d = max |S.1 - S'.1| |S.2 - S'.2| := by
    rw [hd_eq]; simp only [Real.dist_eq]
  -- Lipschitz bound on |p - p'|
  have hp_lip : |p - p'| ≤ L * d := by
    calc |p - p'| ≤ L * max |S.1 - S'.1| |S.2 - S'.2| := hLip S.1 S.2 S'.1 S'.2
      _ = L * d := by rw [hd_abs]
  -- Component 1: |T₁(p,S₁) - T₁(p',S'₁)| ≤ K·d
  have hcomp1 : dist (expectedStrengthUpdate α β Δ Smax p S.1)
                     (expectedStrengthUpdate α β Δ Smax p' S'.1) ≤ K * d := by
    rw [Real.dist_eq]
    have h : |expectedStrengthUpdate α β Δ Smax p S.1 -
              expectedStrengthUpdate α β Δ Smax p' S'.1|
             ≤ exp (-β * Δ) * |S.1 - S'.1| + α * Smax * |p - p'| :=
      expectedUpdate_lipschitz_full hα₀ hα₁ hp.1 hp.2 hβ hΔ hS'₁₀ hS'₁
    have hS1_le : |S.1 - S'.1| ≤ d := by
      rw [hd_abs]; exact le_max_left _ _
    calc |expectedStrengthUpdate α β Δ Smax p S.1 -
          expectedStrengthUpdate α β Δ Smax p' S'.1|
        ≤ exp (-β * Δ) * |S.1 - S'.1| + α * Smax * |p - p'| := h
      _ ≤ exp (-β * Δ) * d + α * Smax * (L * d) := by gcongr
      _ = K * d := by ring
  -- Component 2: |T₂(1-p,S₂) - T₂(1-p',S'₂)| ≤ K·d
  have h1p₀ : 0 ≤ 1 - p := by linarith [hp.2]
  have h1p₁ : 1 - p ≤ 1 := by linarith [hp.1]
  have hcomp2 : dist (expectedStrengthUpdate α β Δ Smax (1 - p) S.2)
                     (expectedStrengthUpdate α β Δ Smax (1 - p') S'.2) ≤ K * d := by
    rw [Real.dist_eq]
    have h : |expectedStrengthUpdate α β Δ Smax (1 - p) S.2 -
              expectedStrengthUpdate α β Δ Smax (1 - p') S'.2|
             ≤ exp (-β * Δ) * |S.2 - S'.2| + α * Smax * |(1 - p) - (1 - p')| :=
      expectedUpdate_lipschitz_full hα₀ hα₁ h1p₀ h1p₁ hβ hΔ hS'₂₀ hS'₂
    -- |(1-p) - (1-p')| = |p' - p| = |p - p'|
    have hq_diff : |(1 - p) - (1 - p')| = |p - p'| := by
      have : (1 - p) - (1 - p') = -(p - p') := by ring
      rw [this, abs_neg]
    have hS2_le : |S.2 - S'.2| ≤ d := by
      rw [hd_abs]; exact le_max_right _ _
    calc |expectedStrengthUpdate α β Δ Smax (1 - p) S.2 -
          expectedStrengthUpdate α β Δ Smax (1 - p') S'.2|
        ≤ exp (-β * Δ) * |S.2 - S'.2| + α * Smax * |(1 - p) - (1 - p')| := h
      _ = exp (-β * Δ) * |S.2 - S'.2| + α * Smax * |p - p'| := by rw [hq_diff]
      _ ≤ exp (-β * Δ) * d + α * Smax * (L * d) := by gcongr
      _ = K * d := by ring
  -- Combine: max of both components ≤ K·d
  simp only [composedExpectedMap, Prod.dist_eq]
  exact max_le hcomp1 hcomp2

-- ============================================================
-- Section 16: ContractingWith Wiring
--
-- Package the Lipschitz bound as ContractingWith for Mathlib's
-- Banach fixed-point theorem.
-- ============================================================

/-- The restriction of the composed map to [0,Smax]² is a contraction
    when the contraction condition L·α·Smax < 1 - exp(-βΔ) holds. -/
theorem composedExpectedMap_contractingWith
    {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax L : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax) (hL : 0 ≤ L)
    (hLip : ∀ s₁ s₂ s₁' s₂',
      |selectProb s₁ s₂ - selectProb s₁' s₂'| ≤ L * max (|s₁ - s₁'|) (|s₂ - s₂'|))
    (hq_range : ∀ s₁ s₂, 0 ≤ selectProb s₁ s₂ ∧ selectProb s₁ s₂ ≤ 1)
    (hContraction : L * α * Smax < 1 - exp (-β * Δ))
    (hmapsTo : MapsTo (composedExpectedMap selectProb α β Δ Smax)
                      (composedDomain Smax) (composedDomain Smax)) :
    let K_val := exp (-β * Δ) + L * α * Smax
    let hK_nn : (0 : ℝ) ≤ K_val := by positivity
    ContractingWith ⟨K_val, hK_nn⟩
      (hmapsTo.restrict (composedExpectedMap selectProb α β Δ Smax)
                        (composedDomain Smax) (composedDomain Smax)) := by
  intro K_val hK_nn
  refine ⟨?_, ?_⟩
  · -- K < 1 as NNReal
    suffices h : K_val < 1 by exact_mod_cast h
    linarith
  · -- LipschitzWith K (restricted map)
    rw [lipschitzWith_iff_dist_le_mul]
    intro ⟨x, hx⟩ ⟨y, hy⟩
    -- Subtype distance = ambient distance
    simp only [Subtype.dist_eq]
    -- Unfold restrict to get the original function applied to .val
    exact composedExpectedMap_lipschitz_on_domain hα₀ hα₁ hβ hΔ hSmax hL hLip hq_range hy

-- ============================================================
-- Section 17: Banach Fixed-Point Theorem Application
--
-- The main stability theorem: existence, uniqueness, and
-- convergence of the stationary state.
-- ============================================================

/-- **Stationary State Theorem**: Under the contraction condition
    L·α·Smax < 1 - exp(-βΔ), the composed mean-field dynamics have
    a unique stationary state S* ∈ [0, Smax]², and iterates from any
    initial state in the domain converge to S*.

    This is the formal statement that the mean-field Markov chain has
    a unique stable distribution with global basin of attraction. -/
theorem stationaryState_exists_unique_convergent
    {selectProb : ℝ → ℝ → ℝ} {α β Δ Smax L : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax) (hL : 0 ≤ L)
    (hLip : ∀ s₁ s₂ s₁' s₂',
      |selectProb s₁ s₂ - selectProb s₁' s₂'| ≤ L * max (|s₁ - s₁'|) (|s₂ - s₂'|))
    (hq_range : ∀ s₁ s₂, 0 ≤ selectProb s₁ s₂ ∧ selectProb s₁ s₂ ≤ 1)
    (hContraction : L * α * Smax < 1 - exp (-β * Δ)) :
    ∃ S_star ∈ composedDomain Smax,
      -- Stationarity: T(S*) = S*
      IsFixedPt (composedExpectedMap selectProb α β Δ Smax) S_star ∧
      -- Uniqueness: any other fixed point in the domain equals S*
      (∀ S' ∈ composedDomain Smax,
        IsFixedPt (composedExpectedMap selectProb α β Δ Smax) S' → S' = S_star) ∧
      -- Global convergence: iterates from any starting point → S*
      (∀ S₀ ∈ composedDomain Smax,
        Tendsto (fun n => (composedExpectedMap selectProb α β Δ Smax)^[n] S₀)
          atTop (𝓝 S_star)) := by
  set f := composedExpectedMap selectProb α β Δ Smax
  set D := composedDomain Smax
  have hmapsTo := composedExpectedMap_mapsTo hα₀ hα₁ hβ hΔ hSmax hq_range
    (selectProb := selectProb)
  have hcw := composedExpectedMap_contractingWith hα₀ hα₁ hβ hΔ hSmax hL hLip hq_range
    hContraction hmapsTo
  -- Pick (0,0) ∈ D as starting point to obtain the fixed point
  have hx₀ : (0, 0) ∈ D := ⟨⟨le_refl 0, hSmax.le⟩, ⟨le_refl 0, hSmax.le⟩⟩
  have hedist₀ : edist ((0, 0) : ℝ × ℝ) (f (0, 0)) ≠ ⊤ := edist_ne_top _ _
  obtain ⟨S_star, hS_star_mem, hS_star_fp, hS_star_tendsto₀, _⟩ :=
    hcw.exists_fixedPoint' composedDomain_isComplete hmapsTo hx₀ hedist₀
  refine ⟨S_star, hS_star_mem, hS_star_fp, ?_, ?_⟩
  · -- Uniqueness: any fixed point in D equals S_star
    intro S' hS' hfpS'
    -- Both are fixed points of the contraction → must be equal
    have hfp_star : IsFixedPt (hmapsTo.restrict f D D) ⟨S_star, hS_star_mem⟩ :=
      Subtype.ext hS_star_fp.eq
    have hfp_S' : IsFixedPt (hmapsTo.restrict f D D) ⟨S', hS'⟩ :=
      Subtype.ext hfpS'.eq
    have := hcw.eq_or_edist_eq_top_of_fixedPoints hfp_S' hfp_star
    rcases this with h | h
    · exact congr_arg Subtype.val h
    · exact absurd h (edist_ne_top _ _)
  · -- Global convergence from any S₀ ∈ D
    intro S₀ hS₀
    have hedist : edist S₀ (f S₀) ≠ ⊤ := edist_ne_top _ _
    obtain ⟨y, hy_mem, hy_fp, hy_tendsto, _⟩ :=
      hcw.exists_fixedPoint' composedDomain_isComplete hmapsTo hS₀ hedist
    -- y must equal S_star by uniqueness (both are fixed points of the contraction)
    have hfp_y : IsFixedPt (hmapsTo.restrict f D D) ⟨y, hy_mem⟩ :=
      Subtype.ext hy_fp.eq
    have hfp_star : IsFixedPt (hmapsTo.restrict f D D) ⟨S_star, hS_star_mem⟩ :=
      Subtype.ext hS_star_fp.eq
    have := hcw.eq_or_edist_eq_top_of_fixedPoints hfp_y hfp_star
    rcases this with h | h
    · have heq : y = S_star := congr_arg Subtype.val h
      rw [heq] at hy_tendsto; exact hy_tendsto
    · exact absurd h (edist_ne_top _ _)

-- ============================================================
-- Section 18: Markov Chain Stability Interpretation
--
-- The mean-field map T(s) = E[S_{n+1} | S_n = s] represents the
-- expected one-step transition of the 2-memory Markov chain.
--
-- A fixed point S* = T(S*) means the expected next state equals
-- the current state — this IS a stationary expected state.
--
-- The contraction theorem gives:
-- (a) Such a state EXISTS and is UNIQUE
-- (b) Expected trajectories converge to it from ANY initial state
-- (c) Convergence is EXPONENTIAL (rate K^n)
--
-- These properties constitute GLOBAL ASYMPTOTIC STABILITY.
--
-- Connection to the stochastic Markov chain:
-- By the ODE method (Benveniste-Métivier-Priouret / Borkar),
-- the stochastic process concentrates around the deterministic
-- mean-field trajectory as the step size decreases. Combined
-- with contraction, the stochastic chain's stationary distribution
-- (which exists by compactness of [0,Smax]²) concentrates
-- around S*.
-- ============================================================

/-- **Capstone: Stable Stationary State with Safety Guarantees**

    The composed memory system has a unique globally stable stationary
    state that satisfies all three safety guarantees simultaneously:
    1. Anti-lock-in: both strengths strictly below Smax
    2. Anti-thrashing: selection probabilities in (0,1)
    3. Anti-cold-start: novelty bonus effective during exploration window
    4. Convergence: all trajectories converge to this state

    This is the full composition theorem: the three fixes compose
    correctly, produce a stable system with a unique attractor, and
    all individual safety guarantees survive composition. -/
theorem stableStationaryState_safe
    {α β Δ Smax T_temp N₀ γ ε L : ℝ}
    (hα₀ : 0 < α) (hα₁ : α < 1) (hβ : 0 < β) (hΔ : 0 < Δ)
    (hSmax : 0 < Smax)
    (hN₀ : 0 < N₀) (hγ : 0 < γ) (hε : 0 < ε) (hεN : ε < N₀)
    (hL : 0 ≤ L)
    (hLip : ∀ s₁ s₂ s₁' s₂',
      |softSelect s₁ s₂ T_temp - softSelect s₁' s₂' T_temp|
      ≤ L * max (|s₁ - s₁'|) (|s₂ - s₂'|))
    (hq_range : ∀ s₁ s₂, 0 ≤ softSelect s₁ s₂ T_temp ∧ softSelect s₁ s₂ T_temp ≤ 1)
    (hContraction : L * α * Smax < 1 - exp (-β * Δ)) :
    ∃ S_star ∈ composedDomain Smax,
      -- Stationarity
      composedExpectedMap (fun s₁ s₂ => softSelect s₁ s₂ T_temp) α β Δ Smax S_star = S_star ∧
      -- Anti-lock-in
      S_star.1 < Smax ∧ S_star.2 < Smax ∧
      -- Anti-thrashing
      (0 < softSelect S_star.1 S_star.2 T_temp ∧
       softSelect S_star.1 S_star.2 T_temp < 1) ∧
      -- Cold-start survival
      (∀ t, 0 ≤ t → t ≤ explorationWindow N₀ γ ε → ε ≤ boostedScore 0 N₀ γ t) ∧
      -- Global convergence
      (∀ S₀ ∈ composedDomain Smax,
        Tendsto (fun n => (composedExpectedMap (fun s₁ s₂ => softSelect s₁ s₂ T_temp)
          α β Δ Smax)^[n] S₀) atTop (𝓝 S_star)) := by
  set selectProb := fun s₁ s₂ => softSelect s₁ s₂ T_temp
  -- Get unique stationary state with convergence
  obtain ⟨S_star, hS_star_mem, hS_star_fp, _, hS_star_conv⟩ :=
    stationaryState_exists_unique_convergent hα₀ hα₁ hβ hΔ hSmax hL hLip hq_range hContraction
  have hS₁ : 0 ≤ S_star.1 := hS_star_mem.1.1
  have hS₂ : 0 ≤ S_star.2 := hS_star_mem.2.1
  -- The fixed-point equation in the form composedSystem_safe expects
  have hfp_eq : S_star = composedExpectedMap selectProb α β Δ Smax S_star :=
    hS_star_fp.eq.symm
  -- Apply the composed safety theorem
  have hsafe := composedSystem_safe hα₀ hα₁ hβ hΔ hSmax hN₀ hγ hε hεN hS₁ hS₂ hfp_eq
  obtain ⟨h_lt1, h_lt2, h_sel, h_cold⟩ := hsafe
  exact ⟨S_star, hS_star_mem, hS_star_fp.eq, h_lt1, h_lt2, h_sel, h_cold, hS_star_conv⟩

end
