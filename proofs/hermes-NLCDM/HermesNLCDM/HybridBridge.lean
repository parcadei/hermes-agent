/-
  HermesNLCDM.HybridBridge
  ========================
  Phase 13d: Connection to NLCDM Retrieval Bounds

  Main results:
  1. coherence_positive_general: coherence > 0 when δ_rel > SI/w₁
     (strengthens coherence_threshold by removing SI > 0 precondition)
  2. sufficient_for_multisignal: w₁·δ_rel > SI → coherence > 0
  3. nlcdm_bound_implies_hybrid_safety: NLCDM bound → hybrid preserves ranking
  4. nlcdm_bound_exceeds_threshold: NLCDM bound > SI/w₁ → multi-signal mode
  5. zero_inversion_coherence: at zero inversion, coherence = w₁·δ_rel
  6. w1_maximizes_gap_at_zero_inversion: larger w₁ → larger score gap
  7. pure_cosine_maximal_gap: w₁ = 1 gives coherence = δ_rel (maximal)

  Bridge between:
  - TransferDynamics.lean (Phase 10): transfer_via_bridge_cosine gives
    cosineSim(T(ξ), target) ≥ σ/(1+(N-1)·exp(-βδ)), establishing δ_rel > 0
  - HybridSwitching.lean (Phase 13b): switching_criterion uses δ_rel > 0

  Practical impact: formally explains MABench results:
  - V2 cosine-only (w₁ = 1): 56% SubEM — optimal at zero signal inversion
  - V1 4-signal re-ranking (w₁ ≈ 0.5): 11% SubEM — diluted relevance signal

  Gives the decision criterion for the Python _query() implementation:
  1. Compute δ_rel from cosine similarities
  2. Compute signalInversion from non-relevance scores
  3. If w₁ · δ_rel > signalInversion → use multi-signal
  4. Otherwise → use cosine-only

  Reference: Hermes memory — hybrid retrieval quality bounds
-/

import HermesNLCDM.HybridSwitching

noncomputable section

open Real Set

namespace HermesNLCDM

-- ============================================================
-- Section 13d.1: General Coherence Threshold
--
-- Strengthens coherence_threshold from HybridSwitching.lean by
-- removing the signalInversion > 0 precondition. The proof uses
-- the same technique (multiply δ_rel > SI/w₁ by w₁ > 0).
-- ============================================================

/-- **General coherence threshold**: coherence is positive when the
    relevance gap exceeds signalInversion/w₁.

    This strengthens coherence_threshold from HybridSwitching.lean:
    - When signalInversion ≤ 0 (signals help): automatically satisfied
      for any δ_rel > 0 with w₁ > 0
    - When signalInversion > 0 (signals hurt): requires δ_rel > SI/w₁

    In both cases, the conclusion is the same: coherence > 0. -/
theorem coherence_positive_general (w : HybridWeights)
    (hw₁ : 0 < w.w₁) {δ_rel : ℝ}
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hthresh : δ_rel > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j / w.w₁) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j > 0 := by
  unfold signalCoherence_j
  -- Need: w₁ * δ_rel - signalInversion > 0
  -- From hthresh: δ_rel > SI / w₁
  -- Multiply by w₁ > 0: w₁ * δ_rel > w₁ * (SI / w₁) = SI
  have h : w.w₁ * δ_rel > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j := by
    rw [gt_iff_lt] at hthresh ⊢
    calc signalInversion w rec_star imp_star act_star rec_j imp_j act_j
        = signalInversion w rec_star imp_star act_star rec_j imp_j act_j
          / w.w₁ * w.w₁ := by field_simp
      _ < δ_rel * w.w₁ := mul_lt_mul_of_pos_right hthresh hw₁
      _ = w.w₁ * δ_rel := by ring
  linarith

/-- **Direct coherence criterion**: w₁·δ_rel > signalInversion → coherence > 0.

    This avoids division by w₁ entirely. For the Python implementation:
    compute `w1 * delta_rel` and compare to `signal_inversion` directly. -/
theorem sufficient_for_multisignal (w : HybridWeights)
    {δ_rel : ℝ}
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (h : w.w₁ * δ_rel > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j > 0 := by
  unfold signalCoherence_j; linarith

-- ============================================================
-- Section 13d.2: NLCDM Bound → Hybrid Safety
--
-- The NLCDM transfer bound (TransferDynamics.lean) gives:
--   cosineSim(T(ξ), target) ≥ σ / (1 + (N-1)·exp(-βδ))
-- When this bound is positive, the correct memory has higher
-- cosine similarity → δ_rel > 0 → hybrid strategy is safe.
-- ============================================================

/-- **NLCDM bound implies hybrid safety**: if the NLCDM transfer bound
    guarantees a positive relevance gap, the hybrid strategy preserves
    the ranking of the correct memory.

    The `bound` parameter represents σ/(1+(N-1)·exp(-βδ)) from
    transfer_via_bridge_cosine in TransferDynamics.lean. When positive,
    it ensures δ_rel > 0, which is the prerequisite for
    switching_criterion.

    The bridge: NLCDM quality guarantee → δ_rel > 0 → hybrid safe. -/
theorem nlcdm_bound_implies_hybrid_safety (w : HybridWeights)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    {coherence : ℝ}
    {bound : ℝ} (hbound_pos : 0 < bound)
    (hgap : rel_star - rel_j ≥ bound)
    (hcoherence_pair : coherence > 0 →
      signalCoherence_j w (rel_star - rel_j)
        rec_star imp_star act_star rec_j imp_j act_j > 0) :
    hybridSwitchedScore w rel_star rec_star imp_star act_star coherence >
    hybridSwitchedScore w rel_j rec_j imp_j act_j coherence :=
  switching_criterion w (by linarith) hcoherence_pair

/-- **NLCDM bound exceeds threshold**: when the NLCDM-guaranteed relevance
    gap exceeds the coherence threshold, coherence is positive and the
    hybrid strategy uses multi-signal enrichment.

    This gives a computable condition: if the transfer bound B satisfies
    B > signalInversion/w₁, then for any δ_rel ≥ B, multi-signal is safe. -/
theorem nlcdm_bound_exceeds_threshold (w : HybridWeights) (hw₁ : 0 < w.w₁)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    {bound : ℝ} (_ : 0 < bound)
    (hthresh : bound > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j / w.w₁)
    {δ_rel : ℝ} (hδ : δ_rel ≥ bound) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j > 0 :=
  coherence_positive_general w hw₁ rec_star imp_star act_star
    rec_j imp_j act_j (by linarith)

-- ============================================================
-- Section 13d.3: Optimal Weights at Zero Signal Inversion
--
-- When non-relevance signals are balanced (signalInversion = 0),
-- coherence = w₁ · δ_rel. Larger w₁ → larger coherence → more
-- robust ranking. Pure cosine (w₁ = 1) is optimal.
--
-- This explains the MABench FactConsolidation observation:
-- recency/importance/activity are uncorrelated with factual
-- correctness, so signalInversion ≈ 0. Pure cosine gives 56%
-- SubEM while 4-signal re-ranking (w₁ ≈ 0.5) gives 11%.
-- ============================================================

/-- **Zero-inversion coherence**: when signalInversion = 0, coherence
    reduces to w₁ · δ_rel.

    At zero inversion, the non-relevance signals are balanced between
    correct and incorrect memories. The coherence depends only on how
    much weight is given to the cosine similarity signal. -/
theorem zero_inversion_coherence (w : HybridWeights) {δ_rel : ℝ}
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hinv : signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j = 0) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j = w.w₁ * δ_rel := by
  unfold signalCoherence_j; rw [hinv]; ring

/-- **Larger w₁ gives larger coherence at zero inversion**: for fixed
    δ_rel ≥ 0, the coherence w₁·δ_rel is monotone in w₁.

    Since coherence = score gap (by coherence_eq_score_gap), larger w₁
    gives a wider gap between correct and incorrect memories, making the
    ranking more robust to noise.

    Combined with pure_cosine_maximal_gap below: w₁ = 1 is optimal. -/
theorem w1_maximizes_gap_at_zero_inversion {w₁a w₁b : ℝ}
    (hw : w₁a ≤ w₁b) {δ_rel : ℝ} (hδ : 0 ≤ δ_rel) :
    w₁a * δ_rel ≤ w₁b * δ_rel :=
  mul_le_mul_of_nonneg_right hw hδ

/-- **Pure cosine gives maximal gap**: when w₁ = 1 and signalInversion = 0,
    the coherence equals δ_rel — the full cosine gap.

    No other weight configuration can achieve this: for w₁ < 1,
    coherence = w₁·δ_rel < δ_rel. So pure cosine is strictly optimal
    when non-relevance signals are uncorrelated with ground-truth relevance.

    Formally: w₁ = 1 ⟹ coherence = 1·δ_rel = δ_rel = cosine gap.

    This is the formal justification for using pure cosine retrieval
    in MABench FactConsolidation (and any task where recency, importance,
    and activity don't predict the correct answer). -/
theorem pure_cosine_maximal_gap (w : HybridWeights) (hw₁ : w.w₁ = 1)
    {δ_rel : ℝ}
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hinv : signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j = 0) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j = δ_rel := by
  rw [zero_inversion_coherence w rec_star imp_star act_star
    rec_j imp_j act_j hinv, hw₁, one_mul]

-- ============================================================
-- Section 13d.4: Negative Inversion (Signals Help)
--
-- When signalInversion < 0, the non-relevance signals favor the
-- correct memory. In this case, coherence > w₁·δ_rel > 0 for
-- any δ_rel > 0 with w₁ > 0. Multi-signal is always beneficial.
-- ============================================================

/-- **Negative inversion boosts coherence**: when non-relevance signals
    favor the correct memory (signalInversion < 0), coherence exceeds
    w₁·δ_rel.

    This is the favorable case for multi-signal re-ranking: recency,
    importance, and activity all point toward the correct answer.
    Multi-signal should always be used (coherence > 0). -/
theorem negative_inversion_boosts_coherence (w : HybridWeights)
    {δ_rel : ℝ} (_ : 0 ≤ δ_rel) (_ : 0 ≤ w.w₁)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hinv : signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j ≤ 0) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j ≥ w.w₁ * δ_rel := by
  unfold signalCoherence_j; linarith

/-- **Negative inversion guarantees coherence**: when signalInversion ≤ 0
    and δ_rel > 0 with w₁ > 0, coherence is strictly positive.

    In this regime, multi-signal re-ranking is always safe AND beneficial:
    the score gap is at least w₁·δ_rel > 0. -/
theorem negative_inversion_guarantees_coherence (w : HybridWeights)
    (hw₁ : 0 < w.w₁) {δ_rel : ℝ} (hδ : 0 < δ_rel)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (hinv : signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j ≤ 0) :
    signalCoherence_j w δ_rel rec_star imp_star act_star
      rec_j imp_j act_j > 0 := by
  have hge := negative_inversion_boosts_coherence w hδ.le hw₁.le
    rec_star imp_star act_star rec_j imp_j act_j hinv
  have hpos : w.w₁ * δ_rel > 0 := mul_pos hw₁ hδ
  linarith

end HermesNLCDM
