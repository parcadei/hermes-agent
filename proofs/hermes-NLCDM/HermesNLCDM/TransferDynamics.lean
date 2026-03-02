/-
  HermesNLCDM.TransferDynamics
  ============================
  Phase 10: Cross-Domain Transfer via Bridge Patterns

  Main results:
  1. softmax_weight_lower_bound: high query-pattern alignment implies
     non-trivial softmax weight on that pattern
  2. hopfieldUpdate_dot: decompose ⟨T(ξ), target⟩ as softmax-weighted sum
  3. transfer_via_bridge: MAIN — query aligned with bridge pattern c gives
     non-trivial Hopfield update alignment with any target that c bridges to
  4. cross_domain_retrieval: corollary for cross-domain retrieval

  Mathematical framework:
  The Hopfield update is T(ξ) = Σ_μ p_μ · x_μ where p_μ = softmax(β · X^T ξ).
  When ξ has high dot product with bridge pattern c, p_c is large.
  Since c has high dot product with target pattern x_t,
  ⟨T(ξ), x_t⟩ ≥ p_c · ⟨c, x_t⟩ ≥ σ / (1 + (N-1)·exp(-βδ)).

  Reference: Hermes memory — OOD reasoning via cross-domain retrieval
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.Dynamics
import HermesNLCDM.BridgeFormation

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Softmax Weight from Alignment

  When the query ξ has alignment gap δ with pattern μ over all other patterns,
  the softmax weight p_μ is at least 1/(1 + (N-1)·exp(-β·δ)).
-/

/-- Softmax weight lower bound: when pattern μ has alignment gap ≥ δ over
    all other patterns, p_μ ≥ 1/(1 + (N-1)·exp(-β·δ)).

    Proof: bound the denominator by splitting into the μ-term and (N-1) copies
    of the largest non-μ term. -/
theorem softmax_weight_lower_bound {β : ℝ} {N : ℕ} [NeZero N]
    (hβ : 0 < β) (z : Fin N → ℝ) (μ : Fin N) {δ : ℝ} (_hδ : 0 < δ)
    (hgap : ∀ ν, ν ≠ μ → z μ - z ν ≥ δ) :
    softmax β z μ ≥ 1 / (1 + (↑N - 1) * exp (-β * δ)) := by
  unfold softmax
  have hexp_μ_pos : 0 < exp (β * z μ) := exp_pos _
  have hS_pos : 0 < ∑ ν, exp (β * z ν) := by positivity
  have hN_ge : (1 : ℝ) ≤ ↑N := Nat.one_le_cast.mpr (NeZero.pos N)
  have hNm1_nn : (0 : ℝ) ≤ ↑N - 1 := by linarith
  have hdenom_pos : (0 : ℝ) < 1 + (↑N - 1) * exp (-β * δ) := by
    linarith [mul_nonneg hNm1_nn (exp_pos (-β * δ)).le]
  -- Sufficient: Σ exp(β·z_ν) ≤ exp(β·z_μ) · (1 + (N-1)·exp(-βδ))
  rw [ge_iff_le, div_le_div_iff₀ hdenom_pos hS_pos, one_mul]
  have hbound : ∀ ν, ν ≠ μ → exp (β * z ν) ≤ exp (β * z μ) * exp (-β * δ) := by
    intro ν hνμ
    rw [← exp_add]
    exact exp_le_exp.mpr (by nlinarith [hgap ν hνμ])
  -- Bound: Σ exp(β z_ν) ≤ exp(β z_μ) + (N-1) · exp(β z_μ) · exp(-βδ)
  have hsum_bound : ∑ ν ∈ Finset.univ.erase μ, exp (β * z ν) ≤
      ∑ _ν ∈ Finset.univ.erase μ, exp (β * z μ) * exp (-β * δ) :=
    Finset.sum_le_sum (fun ν hν => hbound ν (Finset.ne_of_mem_erase hν))
  have hcard : (Finset.univ.erase μ).card = N - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ μ), Finset.card_univ, Fintype.card_fin]
  calc ∑ ν, exp (β * z ν)
      = exp (β * z μ) + ∑ ν ∈ Finset.univ.erase μ, exp (β * z ν) := by
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ μ)]
    _ ≤ exp (β * z μ) + ∑ _ν ∈ Finset.univ.erase μ, exp (β * z μ) * exp (-β * δ) := by
        linarith
    _ = exp (β * z μ) + ↑(N - 1) * (exp (β * z μ) * exp (-β * δ)) := by
        rw [Finset.sum_const, nsmul_eq_mul, hcard]
    _ = exp (β * z μ) + (↑N - 1) * (exp (β * z μ) * exp (-β * δ)) := by
        congr 1; congr 1
        rw [Nat.cast_sub (NeZero.pos N)]; simp
    _ = exp (β * z μ) * (1 + (↑N - 1) * exp (-β * δ)) := by ring

/-! ## Hopfield Update Dot Product Decomposition -/

/-- The dot product of the Hopfield update with any vector decomposes as
    a softmax-weighted sum of pattern-vector dot products.

    ⟨T(ξ), target⟩ = Σ_μ p_μ · ⟨x_μ, target⟩ -/
theorem hopfieldUpdate_dot {d N : ℕ} (cfg : SystemConfig d N)
    (ξ target : Fin d → ℝ) :
    dotSim (hopfieldUpdate cfg ξ) target =
      ∑ μ, softmax cfg.β (fun ν => ∑ j, cfg.patterns ν j * ξ j) μ *
           dotSim (cfg.patterns μ) target := by
  unfold hopfieldUpdate dotSim
  simp_rw [show ∀ i, (∑ μ, softmax cfg.β (fun ν => ∑ j, cfg.patterns ν j * ξ j) μ *
    cfg.patterns μ i) * target i =
    ∑ μ, softmax cfg.β (fun ν => ∑ j, cfg.patterns ν j * ξ j) μ *
    (cfg.patterns μ i * target i) from fun i => by rw [Finset.sum_mul]; congr 1; ext μ; ring]
  rw [Finset.sum_comm]
  congr 1; ext μ
  rw [← Finset.mul_sum]

/-! ## Transfer via Bridge Pattern — Main Theorem -/

/-- PHASE 10 MAIN THEOREM: Transfer via bridge pattern.

    If query ξ has alignment gap δ with bridge pattern c (index μ_c),
    and bridge pattern c has similarity ≥ σ with target,
    and all pattern-target similarities are non-negative,
    then ⟨T(ξ), target⟩ ≥ σ / (1 + (N-1)·exp(-βδ)).

    Quantitative behavior:
    - β → ∞: bound → σ (full transfer)
    - δ → ∞: bound → σ (bridge dominates)
    - N large: bound ≈ σ · exp(βδ) / N (diluted by many patterns) -/
theorem transfer_via_bridge {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ target : Fin d → ℝ)
    (μ_c : Fin N) {δ σ : ℝ} (hδ : 0 < δ) (hσ_nn : 0 ≤ σ)
    (hgap : ∀ ν, ν ≠ μ_c →
      (∑ i, cfg.patterns μ_c i * ξ i) - (∑ i, cfg.patterns ν i * ξ i) ≥ δ)
    (hbridge_sim : dotSim (cfg.patterns μ_c) target ≥ σ)
    (hothers_nn : ∀ ν, dotSim (cfg.patterns ν) target ≥ 0) :
    dotSim (hopfieldUpdate cfg ξ) target ≥
      σ / (1 + (↑N - 1) * exp (-cfg.β * δ)) := by
  rw [hopfieldUpdate_dot]
  set z := fun ν => ∑ j, cfg.patterns ν j * ξ j
  set p := softmax cfg.β z
  have hp_nn : ∀ μ, 0 ≤ p μ := fun μ => le_of_lt (softmax_pos z μ)
  have hpc_bound : p μ_c ≥ 1 / (1 + (↑N - 1) * exp (-cfg.β * δ)) :=
    softmax_weight_lower_bound cfg.β_pos z μ_c hδ hgap
  have hN_ge : (1 : ℝ) ≤ ↑N := Nat.one_le_cast.mpr (NeZero.pos N)
  have hNm1_nn : (0 : ℝ) ≤ ↑N - 1 := by linarith
  have hdenom_pos : (0 : ℝ) < 1 + (↑N - 1) * exp (-cfg.β * δ) := by
    linarith [mul_nonneg hNm1_nn (exp_pos (-cfg.β * δ)).le]
  calc ∑ μ, p μ * dotSim (cfg.patterns μ) target
      ≥ p μ_c * dotSim (cfg.patterns μ_c) target :=
        Finset.single_le_sum (fun μ _ => mul_nonneg (hp_nn μ) (hothers_nn μ))
          (Finset.mem_univ μ_c)
    _ ≥ (1 / (1 + (↑N - 1) * exp (-cfg.β * δ))) * σ := by
        exact mul_le_mul hpc_bound hbridge_sim hσ_nn (hp_nn μ_c)
    _ = σ / (1 + (↑N - 1) * exp (-cfg.β * δ)) := by ring

/-! ## Corollary: Cross-Domain Retrieval -/

/-- Cross-domain retrieval: if bridge pattern μ_c has similarity ≥ σ_t
    with a target-domain pattern, and query ξ is most aligned with μ_c
    by gap δ, then the Hopfield update of ξ has alignment ≥ σ_t / (1 + (N-1)·exp(-βδ))
    with the target pattern. -/
theorem cross_domain_retrieval {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (μ_c μ_t : Fin N) {δ σ_t : ℝ} (hδ : 0 < δ) (hσ_t_nn : 0 ≤ σ_t)
    (hgap : ∀ ν, ν ≠ μ_c →
      (∑ i, cfg.patterns μ_c i * ξ i) - (∑ i, cfg.patterns ν i * ξ i) ≥ δ)
    (hbridge_target : dotSim (cfg.patterns μ_c) (cfg.patterns μ_t) ≥ σ_t)
    (hothers_nn : ∀ ν, dotSim (cfg.patterns ν) (cfg.patterns μ_t) ≥ 0) :
    dotSim (hopfieldUpdate cfg ξ) (cfg.patterns μ_t) ≥
      σ_t / (1 + (↑N - 1) * exp (-cfg.β * δ)) :=
  transfer_via_bridge cfg ξ (cfg.patterns μ_t) μ_c hδ hσ_t_nn hgap hbridge_target hothers_nn

/-! ## Cosine Similarity Bound — Phase 10b

  The existing Phase 10 theorem bounds ⟨T(ξ), target⟩ (dot product).
  But the operationally meaningful quantity is cosine similarity:
    cos(T(ξ), target) = ⟨T(ξ), target⟩ / (‖T(ξ)‖ · ‖target‖)

  Key insight: T(ξ) = Σ p_μ · x_μ is a convex combination of stored patterns.
  For unit-vector patterns, ‖T(ξ)‖ ≤ 1, so:
    cosineSim(T(ξ), target) ≥ dotSim(T(ξ), target) ≥ σ/(1+(N-1)exp(-βδ))

  Empirical validation (test_h3_real_embeddings.py, Table 2):
    N=200:  dot=0.568, cos=0.905 — both well above theoretical bound
    N=5000: dot=0.295, cos=0.728 — cosine stays strong, dot collapses
    The bound formula gives ~0.001 at N=5000, which lower-bounds cosine too.

  Reference: Cosine rewrite motivated by H3 real-embedding experiments
-/

/-- Unit vector: ‖v‖² = Σ v_i² = 1 -/
def isUnitVec {d : ℕ} (v : Fin d → ℝ) : Prop := ∑ i, v i ^ 2 = 1

/-- dotSim v v = Σ v_i², relating dot product to norm squared. -/
theorem dotSim_eq_sum_sq {d : ℕ} (v : Fin d → ℝ) :
    dotSim v v = ∑ i, v i ^ 2 := by
  unfold dotSim; congr 1; ext i; rw [sq]

/-- Weighted mean square inequality (Jensen for x²):
    (Σ w_μ a_μ)² ≤ Σ w_μ a_μ²  for probability weights.
    Equivalently: E[X]² ≤ E[X²], i.e., variance ≥ 0. -/
theorem sq_weighted_mean_le {M : ℕ} (w a : Fin M → ℝ)
    (hw_nn : ∀ μ, 0 ≤ w μ) (hw_sum : ∑ μ, w μ = 1) :
    (∑ μ, w μ * a μ) ^ 2 ≤ ∑ μ, w μ * a μ ^ 2 := by
  -- Variance ≥ 0: 0 ≤ Σ w_μ (a_μ - m)² = Σ w_μ a_μ² - m²
  set m := ∑ μ, w μ * a μ with hm_def
  suffices h : m ^ 2 ≤ ∑ μ, w μ * a μ ^ 2 from h
  have hvar_nn : 0 ≤ ∑ μ, w μ * (a μ - m) ^ 2 :=
    Finset.sum_nonneg (fun μ _ => mul_nonneg (hw_nn μ) (sq_nonneg _))
  have hexp : ∑ μ, w μ * (a μ - m) ^ 2 = (∑ μ, w μ * a μ ^ 2) - m ^ 2 := by
    have step : ∀ μ, w μ * (a μ - m) ^ 2 =
        w μ * a μ ^ 2 + (-(2 * m) * (w μ * a μ) + m ^ 2 * w μ) := by
      intro μ; ring
    simp_rw [step]
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, hw_sum, ← hm_def]
    ring
  linarith

/-- Hopfield update norm bound: for unit-vector patterns, ‖T(ξ)‖² ≤ 1.

    T(ξ) = Σ p_μ x_μ is a convex combination of unit vectors.
    By Jensen applied per-coordinate:
      ‖T(ξ)‖² = Σ_k (Σ_μ p_μ x_μ(k))² ≤ Σ_k Σ_μ p_μ x_μ(k)²
                = Σ_μ p_μ ‖x_μ‖² = 1 -/
theorem hopfieldUpdate_normSq_le_one {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (hunit : ∀ μ, isUnitVec (cfg.patterns μ)) :
    ∑ k, (hopfieldUpdate cfg ξ k) ^ 2 ≤ 1 := by
  unfold hopfieldUpdate
  set sim := fun ν => ∑ j, cfg.patterns ν j * ξ j
  set p := softmax cfg.β sim
  have hp_nn : ∀ μ, 0 ≤ p μ := fun μ => le_of_lt (softmax_pos sim μ)
  have hp_sum : ∑ μ, p μ = 1 := softmax_sum_one sim
  -- Per-coordinate Jensen: (Σ p_μ x_μ(k))² ≤ Σ p_μ x_μ(k)²
  calc ∑ k, (∑ μ, p μ * cfg.patterns μ k) ^ 2
      ≤ ∑ k, ∑ μ, p μ * (cfg.patterns μ k) ^ 2 := by
        apply Finset.sum_le_sum; intro k _
        exact sq_weighted_mean_le p (fun μ => cfg.patterns μ k) hp_nn hp_sum
    _ = ∑ μ, p μ * ∑ k, (cfg.patterns μ k) ^ 2 := by
        rw [Finset.sum_comm]; congr 1; ext μ; rw [Finset.mul_sum]
    _ = ∑ μ, p μ * 1 := by
        congr 1; ext μ; congr 1; exact hunit μ
    _ = 1 := by simp [hp_sum]

/-- Cosine ≥ dot product when ‖u‖ ≤ 1 and ‖v‖ = 1.

    cosineSim u v = dot(u,v) / (‖u‖·‖v‖) = dot(u,v) / ‖u‖
    Since ‖u‖ ≤ 1: cosineSim ≥ dot(u,v) ≥ L.

    Requires ‖u‖ > 0 (non-degenerate Hopfield update). -/
theorem cosineSim_ge_of_dot_ge {d : ℕ} (u v : Fin d → ℝ)
    (hv_unit : isUnitVec v)
    (hu_normSq_le : ∑ i, u i ^ 2 ≤ 1)
    (hu_normSq_pos : 0 < ∑ i, u i ^ 2)
    {L : ℝ} (hL : 0 ≤ L) (hdot : dotSim u v ≥ L) :
    cosineSim u v ≥ L := by
  unfold cosineSim isUnitVec at *
  rw [hv_unit, Real.sqrt_one, mul_one]
  -- Goal: (Σ u_i * v_i) / sqrt(Σ u_i²) ≥ L
  have hsqrt_pos : 0 < Real.sqrt (∑ i, u i ^ 2) := Real.sqrt_pos.mpr hu_normSq_pos
  have hsqrt_le : Real.sqrt (∑ i, u i ^ 2) ≤ 1 :=
    Real.sqrt_le_one.mpr hu_normSq_le
  rw [ge_iff_le, le_div_iff₀ hsqrt_pos]
  unfold dotSim at hdot
  calc L * Real.sqrt (∑ i, u i ^ 2)
      ≤ L * 1 := by exact mul_le_mul_of_nonneg_left hsqrt_le hL
    _ = L := mul_one L
    _ ≤ ∑ i, u i * v i := hdot

/-! ## Main Cosine Transfer Theorems -/

/-- PHASE 10b MAIN THEOREM: Cosine transfer via bridge pattern.

    Under the same hypotheses as transfer_via_bridge, plus:
    - all patterns are unit vectors
    - the target is a unit vector
    - the Hopfield update is non-degenerate (‖T(ξ)‖ > 0)

    Then cosineSim(T(ξ), target) ≥ σ / (1 + (N-1)·exp(-βδ)).

    This is the cosine-similarity version of the Phase 10 bound.
    The bound formula is identical, but now lower-bounds the operationally
    meaningful quantity (direction alignment, scale-invariant). -/
theorem transfer_via_bridge_cosine {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ target : Fin d → ℝ)
    (μ_c : Fin N) {δ σ : ℝ} (hδ : 0 < δ) (hσ_nn : 0 ≤ σ)
    (hgap : ∀ ν, ν ≠ μ_c →
      (∑ i, cfg.patterns μ_c i * ξ i) - (∑ i, cfg.patterns ν i * ξ i) ≥ δ)
    (hbridge_sim : dotSim (cfg.patterns μ_c) target ≥ σ)
    (hothers_nn : ∀ ν, dotSim (cfg.patterns ν) target ≥ 0)
    (hunit : ∀ μ, isUnitVec (cfg.patterns μ))
    (htarget_unit : isUnitVec target)
    (hT_pos : 0 < ∑ k, (hopfieldUpdate cfg ξ k) ^ 2) :
    cosineSim (hopfieldUpdate cfg ξ) target ≥
      σ / (1 + (↑N - 1) * exp (-cfg.β * δ)) := by
  have hdot := transfer_via_bridge cfg ξ target μ_c hδ hσ_nn hgap hbridge_sim hothers_nn
  have hT_le := hopfieldUpdate_normSq_le_one cfg ξ hunit
  have hN_ge : (1 : ℝ) ≤ ↑N := Nat.one_le_cast.mpr (NeZero.pos N)
  have hNm1_nn : (0 : ℝ) ≤ ↑N - 1 := by linarith
  have hdenom_pos : (0 : ℝ) < 1 + (↑N - 1) * exp (-cfg.β * δ) := by
    linarith [mul_nonneg hNm1_nn (exp_pos (-cfg.β * δ)).le]
  have hbound_nn : 0 ≤ σ / (1 + (↑N - 1) * exp (-cfg.β * δ)) :=
    div_nonneg hσ_nn hdenom_pos.le
  exact cosineSim_ge_of_dot_ge _ _ htarget_unit hT_le hT_pos hbound_nn hdot

/-- Cosine cross-domain retrieval: corollary for unit-vector patterns. -/
theorem cross_domain_retrieval_cosine {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ : Fin d → ℝ)
    (μ_c μ_t : Fin N) {δ σ_t : ℝ} (hδ : 0 < δ) (hσ_t_nn : 0 ≤ σ_t)
    (hgap : ∀ ν, ν ≠ μ_c →
      (∑ i, cfg.patterns μ_c i * ξ i) - (∑ i, cfg.patterns ν i * ξ i) ≥ δ)
    (hbridge_target : dotSim (cfg.patterns μ_c) (cfg.patterns μ_t) ≥ σ_t)
    (hothers_nn : ∀ ν, dotSim (cfg.patterns ν) (cfg.patterns μ_t) ≥ 0)
    (hunit : ∀ μ, isUnitVec (cfg.patterns μ))
    (hT_pos : 0 < ∑ k, (hopfieldUpdate cfg ξ k) ^ 2) :
    cosineSim (hopfieldUpdate cfg ξ) (cfg.patterns μ_t) ≥
      σ_t / (1 + (↑N - 1) * exp (-cfg.β * δ)) :=
  transfer_via_bridge_cosine cfg ξ (cfg.patterns μ_t) μ_c hδ hσ_t_nn hgap
    hbridge_target hothers_nn hunit (hunit μ_t) hT_pos

end HermesNLCDM
