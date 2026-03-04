/-
  HermesNLCDM.HybridRetrieval
  ===========================
  Phase 13a: Hybrid Retrieval — Rank Preservation Under Multi-Signal Scoring

  Main results:
  1. cosine_rank_one: if m* has highest cosine similarity, cosine ranking places it first
  2. multisignal_rank_preservation: if signal coherence > 0, multi-signal preserves rank 1
  3. multisignal_rank_inversion: if coherence < 0, some wrong memory is ranked above m*

  Mathematical framework:
  The retrieval problem: given query q and memories m₁,...,mₙ with embeddings e₁,...,eₙ,
  rank memories by a scoring function and return the top-k.

  Two strategies:
  - Cosine-only: rank by cosineSim(q, eᵢ)
  - Multi-signal: rank by w₁·relevance + w₂·recency + w₃·importance + w₄·σ(activation)

  Signal coherence measures whether non-relevance signals reinforce or contradict
  the cosine ranking. When coherence > 0, multi-signal preserves the correct ranking.
  When coherence < 0, multi-signal inverts it.

  This gives a formal switching criterion for the _query() method:
  use cosine-only when coherence ≤ 0, multi-signal when coherence > 0.

  Reference: Hermes memory — hybrid retrieval switching criterion
-/

import HermesNLCDM.Energy
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Basic
import Mathlib.Order.Interval.Set.Basic

noncomputable section

open Real Set

namespace HermesNLCDM

-- ============================================================
-- Section 13.1: Restated Definitions from hermes-memory
--
-- The hermes-memory and hermes-NLCDM projects are separate Lake
-- packages. We restate the scoring definitions here, following
-- the precedent of MonotonicityChain.lean which restates
-- strengthDecay from hermes-memory.
-- ============================================================

/-- Sigmoid function σ(x) = 1 / (1 + e^(-x)).
    Restated from hermes-memory/HermesMemory/MemoryDynamics.lean. -/
def sigmoid' (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- Sigmoid is always strictly positive. -/
theorem sigmoid'_pos (x : ℝ) : 0 < sigmoid' x := by
  unfold sigmoid'
  exact div_pos one_pos (by linarith [exp_pos (-x)])

/-- Sigmoid is strictly less than 1. -/
theorem sigmoid'_lt_one (x : ℝ) : sigmoid' x < 1 := by
  unfold sigmoid'
  rw [div_lt_one (by linarith [exp_pos (-x)])]
  linarith [exp_pos (-x)]

/-- Sigmoid maps to [0, 1]. -/
theorem sigmoid'_mem_Icc (x : ℝ) : sigmoid' x ∈ Icc 0 1 :=
  ⟨le_of_lt (sigmoid'_pos x), le_of_lt (sigmoid'_lt_one x)⟩

-- ============================================================
-- Section 13.2: Multi-Signal Scoring
--
-- The composite retrieval score with 4 signals.
-- Weights satisfy wᵢ ≥ 0 and Σwᵢ = 1.
-- ============================================================

/-- Weights for the hybrid scoring function. -/
structure HybridWeights where
  w₁ : ℝ  -- relevance (cosine similarity)
  w₂ : ℝ  -- recency
  w₃ : ℝ  -- importance
  w₄ : ℝ  -- activation (via sigmoid)
  hw₁ : 0 ≤ w₁
  hw₂ : 0 ≤ w₂
  hw₃ : 0 ≤ w₃
  hw₄ : 0 ≤ w₄
  hsum : w₁ + w₂ + w₃ + w₄ = 1

/-- The non-relevance score: the part of the composite score that ignores
    cosine similarity. This is the "noise" that may help or hurt ranking. -/
def nonRelScore (w : HybridWeights) (rec imp act : ℝ) : ℝ :=
  w.w₂ * rec + w.w₃ * imp + w.w₄ * sigmoid' act

/-- The full multi-signal score. -/
def hybridScore (w : HybridWeights) (rel rec imp act : ℝ) : ℝ :=
  w.w₁ * rel + nonRelScore w rec imp act

/-- nonRelScore is non-negative when components are non-negative. -/
theorem nonRelScore_nonneg (w : HybridWeights) {rec imp act : ℝ}
    (hrec : 0 ≤ rec) (himp : 0 ≤ imp) :
    0 ≤ nonRelScore w rec imp act := by
  unfold nonRelScore
  apply add_nonneg
  apply add_nonneg
  · exact mul_nonneg w.hw₂ hrec
  · exact mul_nonneg w.hw₃ himp
  · exact mul_nonneg w.hw₄ (le_of_lt (sigmoid'_pos act))

/-- hybridScore is non-negative when components are non-negative. -/
theorem hybridScore_nonneg (w : HybridWeights) {rel rec imp act : ℝ}
    (hrel : 0 ≤ rel) (hrec : 0 ≤ rec) (himp : 0 ≤ imp) :
    0 ≤ hybridScore w rel rec imp act := by
  unfold hybridScore
  exact add_nonneg (mul_nonneg w.hw₁ hrel) (nonRelScore_nonneg w hrec himp)

/-- nonRelScore is at most (1 - w₁) when rec, imp ∈ [0, 1]. -/
theorem nonRelScore_le (w : HybridWeights) {rec imp act : ℝ}
    (hrec : rec ∈ Icc 0 1) (himp : imp ∈ Icc 0 1) :
    nonRelScore w rec imp act ≤ 1 - w.w₁ := by
  unfold nonRelScore
  have hσ := sigmoid'_mem_Icc act
  calc w.w₂ * rec + w.w₃ * imp + w.w₄ * sigmoid' act
      ≤ w.w₂ * 1 + w.w₃ * 1 + w.w₄ * 1 := by
        apply add_le_add
        apply add_le_add
        · exact mul_le_mul_of_nonneg_left hrec.2 w.hw₂
        · exact mul_le_mul_of_nonneg_left himp.2 w.hw₃
        · exact mul_le_mul_of_nonneg_left hσ.2 w.hw₄
    _ = w.w₂ + w.w₃ + w.w₄ := by ring
    _ = 1 - w.w₁ := by linarith [w.hsum]

/-- hybridScore is at most 1 when all components ∈ [0, 1]. -/
theorem hybridScore_le_one (w : HybridWeights) {rel rec imp act : ℝ}
    (hrel : rel ∈ Icc 0 1) (hrec : rec ∈ Icc 0 1) (himp : imp ∈ Icc 0 1) :
    hybridScore w rel rec imp act ≤ 1 := by
  unfold hybridScore
  calc w.w₁ * rel + nonRelScore w rec imp act
      ≤ w.w₁ * 1 + (1 - w.w₁) := by
        apply add_le_add
        · exact mul_le_mul_of_nonneg_left hrel.2 w.hw₁
        · exact nonRelScore_le w hrec himp
    _ = 1 := by ring

/-- hybridScore lies in [0, 1] — the main boundedness theorem. -/
theorem hybridScore_mem_Icc (w : HybridWeights) {rel rec imp act : ℝ}
    (hrel : rel ∈ Icc 0 1) (hrec : rec ∈ Icc 0 1) (himp : imp ∈ Icc 0 1) :
    hybridScore w rel rec imp act ∈ Icc 0 1 :=
  ⟨hybridScore_nonneg w hrel.1 hrec.1 himp.1,
   hybridScore_le_one w hrel hrec himp⟩

-- ============================================================
-- Section 13.3: Signal Coherence
--
-- Signal coherence measures whether non-relevance signals
-- reinforce or contradict the cosine ranking. It is the key
-- quantity that determines whether multi-signal re-ranking
-- helps or hurts retrieval quality.
-- ============================================================

/-- Signal inversion: how much the non-relevance signals favor
    memory j over the correct memory m*.
    Positive Δ means j has higher non-relevance score than m*. -/
def signalInversion (w : HybridWeights)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ) : ℝ :=
  nonRelScore w rec_j imp_j act_j - nonRelScore w rec_star imp_star act_star

/-- Signal coherence for a specific competitor j:
    positive means multi-signal preserves m* > j ranking.
    coherence_j = w₁ · (rel* - rel_j) - signalInversion

    When coherence_j > 0 for all j ≠ *, multi-signal preserves rank 1. -/
def signalCoherence_j (w : HybridWeights) (δ_rel_j : ℝ)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ) : ℝ :=
  w.w₁ * δ_rel_j - signalInversion w rec_star imp_star act_star rec_j imp_j act_j

-- ============================================================
-- Section 13.4: Rank Preservation Theorems
-- ============================================================

/-- **Cosine rank one**: if memory m* has strictly higher cosine similarity
    to the query than memory j, then the cosine-only ranking places m* above j.

    This is immediate: the scoring function IS cosine similarity.
    Formalized for completeness and to establish the pattern. -/
theorem cosine_rank_one {rel_star rel_j : ℝ}
    (hgap : rel_star > rel_j) :
    rel_star > rel_j := hgap

/-- **Multi-signal rank preservation**: if signal coherence is positive
    for a specific competitor j, then multi-signal also ranks m* above j.

    The key insight: hybridScore(m*) - hybridScore(j) =
      w₁·(rel* - rel_j) + (nonRelScore(m*) - nonRelScore(j))
    = w₁·δ_rel - signalInversion
    = coherence_j

    So coherence_j > 0 ⟹ hybridScore(m*) > hybridScore(j). -/
theorem multisignal_rank_preservation (w : HybridWeights)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    (hcoherence : signalCoherence_j w (rel_star - rel_j)
      rec_star imp_star act_star rec_j imp_j act_j > 0) :
    hybridScore w rel_star rec_star imp_star act_star >
    hybridScore w rel_j rec_j imp_j act_j := by
  unfold hybridScore
  unfold signalCoherence_j signalInversion at hcoherence
  -- Goal: w₁·rel* + NR* > w₁·rel_j + NR_j
  -- From hypothesis: w₁·(rel* - rel_j) - (NR_j - NR*) > 0
  -- i.e., w₁·rel* - w₁·rel_j - NR_j + NR* > 0
  -- i.e., (w₁·rel* + NR*) - (w₁·rel_j + NR_j) > 0
  linarith

/-- **Multi-signal rank inversion**: if signal coherence is negative
    for competitor j, then multi-signal ranks j above m*.

    This is the formal statement that negative coherence inverts ranking. -/
theorem multisignal_rank_inversion (w : HybridWeights)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    (hcoherence : signalCoherence_j w (rel_star - rel_j)
      rec_star imp_star act_star rec_j imp_j act_j < 0) :
    hybridScore w rel_star rec_star imp_star act_star <
    hybridScore w rel_j rec_j imp_j act_j := by
  unfold hybridScore
  unfold signalCoherence_j signalInversion at hcoherence
  linarith

/-- **Coherence equivalence**: multi-signal preserves ranking iff coherence > 0.

    This is the bidirectional version: coherence_j = hybridScore(m*) - hybridScore(j).
    Positive coherence ⟺ m* ranked above j. -/
theorem coherence_eq_score_gap (w : HybridWeights)
    (rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ) :
    signalCoherence_j w (rel_star - rel_j)
      rec_star imp_star act_star rec_j imp_j act_j =
    hybridScore w rel_star rec_star imp_star act_star -
    hybridScore w rel_j rec_j imp_j act_j := by
  unfold signalCoherence_j signalInversion hybridScore nonRelScore
  ring

-- ============================================================
-- Section 13.5: Cosine Dominance
--
-- When w₁ is large enough relative to the relevance gap,
-- multi-signal preserves ranking regardless of non-relevance signals.
-- This gives a sufficient condition for safe re-ranking.
-- ============================================================

/-- **Cosine dominance**: if the relevance gap exceeds the maximum possible
    signal inversion, multi-signal preserves ranking.

    The maximum signal inversion is bounded by (1 - w₁) (since nonRelScore
    ∈ [0, 1-w₁] when components ∈ [0,1]). So if w₁·δ_rel > (1 - w₁),
    i.e., δ_rel > (1 - w₁)/w₁, then coherence > 0 for all competitors. -/
theorem cosine_dominance (w : HybridWeights) (hw₁_pos : 0 < w.w₁)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    (_ : rel_star ∈ Icc 0 1) (_ : rel_j ∈ Icc 0 1)
    (hrec_star : rec_star ∈ Icc 0 1) (himp_star : imp_star ∈ Icc 0 1)
    (hrec_j : rec_j ∈ Icc 0 1) (himp_j : imp_j ∈ Icc 0 1)
    (hgap : rel_star - rel_j > (1 - w.w₁) / w.w₁) :
    hybridScore w rel_star rec_star imp_star act_star >
    hybridScore w rel_j rec_j imp_j act_j := by
  -- Strategy: show the score gap > 0 via coherence_eq_score_gap
  have hkey := coherence_eq_score_gap w rel_star rel_j
    rec_star imp_star act_star rec_j imp_j act_j
  -- Suffices to show signalCoherence_j > 0
  suffices hcoh : signalCoherence_j w (rel_star - rel_j)
      rec_star imp_star act_star rec_j imp_j act_j > 0 by
    linarith
  unfold signalCoherence_j signalInversion
  -- Need: w₁ · (rel* - rel_j) - (NR_j - NR*) > 0
  -- We know: NR_j - NR* ≤ (1 - w₁) - 0 = (1 - w₁)
  --   because NR_j ≤ 1 - w₁ and NR* ≥ 0
  -- And: w₁ · (rel* - rel_j) > w₁ · (1 - w₁)/w₁ = 1 - w₁
  have hNR_star_nn := nonRelScore_nonneg w hrec_star.1 himp_star.1
    (act := act_star)
  have hNR_j_le := nonRelScore_le w hrec_j himp_j (act := act_j)
  have hNR_bound : nonRelScore w rec_j imp_j act_j -
      nonRelScore w rec_star imp_star act_star ≤ 1 - w.w₁ := by
    linarith
  have hw₁_gap : w.w₁ * (rel_star - rel_j) > 1 - w.w₁ := by
    have h1 : (1 - w.w₁) / w.w₁ * w.w₁ = 1 - w.w₁ := by
      field_simp
    nlinarith
  linarith

-- ============================================================
-- Section 13.6: Pure Cosine Optimality
--
-- When w₁ = 1 (pure cosine), the hybrid score equals the cosine
-- score. This is the baseline: any deviation from w₁ = 1 requires
-- signal coherence to maintain ranking quality.
-- ============================================================

/-- When w₁ = 1, nonRelScore is zero (the other weights must all be 0). -/
theorem nonRelScore_zero_of_w1_eq_one (w : HybridWeights) (hw₁ : w.w₁ = 1)
    (rec imp act : ℝ) :
    nonRelScore w rec imp act = 0 := by
  unfold nonRelScore
  have h234 : w.w₂ + w.w₃ + w.w₄ = 0 := by linarith [w.hsum]
  have hw₂ : w.w₂ = 0 := le_antisymm (by linarith [w.hw₃, w.hw₄]) w.hw₂
  have hw₃ : w.w₃ = 0 := le_antisymm (by linarith [w.hw₂, w.hw₄]) w.hw₃
  have hw₄ : w.w₄ = 0 := le_antisymm (by linarith [w.hw₂, w.hw₃]) w.hw₄
  rw [hw₂, hw₃, hw₄]; ring

/-- When w₁ = 1, hybridScore equals cosine similarity. -/
theorem hybridScore_eq_cosine_of_w1_eq_one (w : HybridWeights) (hw₁ : w.w₁ = 1)
    (rel rec imp act : ℝ) :
    hybridScore w rel rec imp act = rel := by
  unfold hybridScore
  rw [nonRelScore_zero_of_w1_eq_one w hw₁, hw₁, one_mul, add_zero]

/-- When w₁ = 1, multi-signal ranking is identical to cosine ranking. -/
theorem pure_cosine_ranking (w : HybridWeights) (hw₁ : w.w₁ = 1)
    {rel_star rel_j : ℝ} (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hgap : rel_star > rel_j) :
    hybridScore w rel_star rec_star imp_star act_star >
    hybridScore w rel_j rec_j imp_j act_j := by
  rw [hybridScore_eq_cosine_of_w1_eq_one w hw₁,
      hybridScore_eq_cosine_of_w1_eq_one w hw₁]
  exact hgap

end HermesNLCDM
