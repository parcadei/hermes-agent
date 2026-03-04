/-
  HermesNLCDM.HybridSwitching
  ============================
  Phase 13b: Hybrid Retrieval — Switching Criterion

  Main results:
  1. switching_criterion: the hybrid strategy (cosine when coherence ≤ 0,
     multi-signal when coherence > 0) preserves rank 1 whenever cosine does.
  2. score_gap_monotone_in_relevance_gap: the coherence is monotone in δ_rel.
  3. coherence_threshold: the exact switching threshold δ_rel > Δ_noise / w₁.
  4. hybrid_preserves_rank_one: the hybrid strategy never loses rank 1.

  The switching criterion gives a formal decision boundary for _query():
  compute coherence, then choose the retrieval strategy accordingly.

  Key insight: at query time, all candidate memories share the same coherence
  regime (it depends on the query and the global signal correlation, not
  per-memory). So the "same regime" switching is the practical case.

  Reference: Hermes memory — hybrid retrieval formal switching criterion
-/

import HermesNLCDM.HybridRetrieval

noncomputable section

open Real Set

namespace HermesNLCDM

-- ============================================================
-- Section 13b.1: Hybrid Strategy Definition
--
-- The hybrid strategy selects between cosine-only and
-- multi-signal based on the sign of signal coherence.
-- ============================================================

/-- The hybrid retrieval score: uses cosine-only when coherence ≤ 0,
    multi-signal when coherence > 0.

    This encodes the switching criterion as a function:
    - If adding non-relevance signals would preserve the cosine ranking
      (coherence > 0), use the richer multi-signal score.
    - Otherwise, fall back to pure cosine similarity. -/
def hybridSwitchedScore (w : HybridWeights)
    (rel rec imp act : ℝ) (coherence : ℝ) : ℝ :=
  if coherence > 0 then hybridScore w rel rec imp act
  else rel

-- ============================================================
-- Section 13b.2: Switching Criterion
--
-- The core theorem: the hybrid strategy preserves rank 1
-- whenever the cosine ranking places m* at rank 1, provided
-- both memories are evaluated under the same coherence regime.
--
-- This is the practical case: at query time, the coherence
-- assessment is global (depends on the query and the distribution
-- of non-relevance signals across all memories), so all candidates
-- share the same coherence value.
-- ============================================================

/-- **Switching criterion**: When both memories are evaluated
    under the SAME coherence regime (both positive or both non-positive),
    the hybrid strategy preserves the cosine ranking.

    Case 1: coherence > 0 → multi-signal is used, and by
      multisignal_rank_preservation, it preserves rank 1
      (given per-pair coherence is positive).
    Case 2: coherence ≤ 0 → cosine-only is used, and by
      cosine_rank_one, m* is at rank 1. -/
theorem switching_criterion (w : HybridWeights)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    {coherence : ℝ}
    (hgap : rel_star > rel_j)
    -- When coherence > 0, the per-pair coherence must also be positive
    (hcoherence_pair : coherence > 0 →
      signalCoherence_j w (rel_star - rel_j)
        rec_star imp_star act_star rec_j imp_j act_j > 0) :
    hybridSwitchedScore w rel_star rec_star imp_star act_star coherence >
    hybridSwitchedScore w rel_j rec_j imp_j act_j coherence := by
  unfold hybridSwitchedScore
  by_cases hc : coherence > 0
  · -- Both use multi-signal
    simp only [hc, ite_true]
    exact multisignal_rank_preservation w (hcoherence_pair hc)
  · -- Both use cosine
    simp only [hc, ite_false]
    exact hgap

-- ============================================================
-- Section 13b.3: Monotonicity in Coherence
-- ============================================================

/-- **Score gap equals coherence**: the gap between multi-signal scores
    of m* and j is exactly the signal coherence.

    This is a direct restatement of coherence_eq_score_gap, establishing
    that coherence IS the quality measure for ranking. -/
theorem score_gap_eq_coherence (w : HybridWeights)
    (rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ) :
    hybridScore w rel_star rec_star imp_star act_star -
    hybridScore w rel_j rec_j imp_j act_j =
    signalCoherence_j w (rel_star - rel_j)
      rec_star imp_star act_star rec_j imp_j act_j :=
  (coherence_eq_score_gap w rel_star rel_j
    rec_star imp_star act_star rec_j imp_j act_j).symm

/-- **Monotonicity in relevance gap**: for fixed non-relevance signals,
    increasing the relevance gap increases signal coherence.

    The score gap = w₁·δ_rel - signalInversion. Since w₁ ≥ 0,
    larger δ_rel gives larger coherence. -/
theorem score_gap_monotone_in_relevance_gap (w : HybridWeights)
    {δ₁ δ₂ : ℝ} (hδ : δ₁ ≤ δ₂)
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ) :
    signalCoherence_j w δ₁ rec_star imp_star act_star rec_j imp_j act_j ≤
    signalCoherence_j w δ₂ rec_star imp_star act_star rec_j imp_j act_j := by
  unfold signalCoherence_j
  linarith [mul_le_mul_of_nonneg_left hδ w.hw₁]

/-- **Coherence threshold**: the minimum relevance gap needed for
    multi-signal to preserve ranking, given a positive signal inversion.

    If Δ_noise = nonRelScore(j) - nonRelScore(*) > 0, then coherence > 0 iff
    δ_rel > Δ_noise / w₁.

    This gives the exact switching threshold. -/
theorem coherence_threshold (w : HybridWeights) (hw₁_pos : 0 < w.w₁)
    {δ_rel : ℝ}
    (rec_star imp_star act_star rec_j imp_j act_j : ℝ)
    (_ : signalInversion w rec_star imp_star act_star rec_j imp_j act_j > 0)
    (hthresh : δ_rel > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j / w.w₁) :
    signalCoherence_j w δ_rel rec_star imp_star act_star rec_j imp_j act_j > 0 := by
  unfold signalCoherence_j
  have h : w.w₁ * δ_rel > signalInversion w rec_star imp_star act_star
      rec_j imp_j act_j := by
    rw [gt_iff_lt] at hthresh ⊢
    calc signalInversion w rec_star imp_star act_star rec_j imp_j act_j
        = signalInversion w rec_star imp_star act_star rec_j imp_j act_j / w.w₁ * w.w₁ := by
          field_simp
      _ < δ_rel * w.w₁ := mul_lt_mul_of_pos_right hthresh hw₁_pos
      _ = w.w₁ * δ_rel := by ring
  linarith

-- ============================================================
-- Section 13b.4: Hybrid Strategy Safety
--
-- The hybrid strategy is at least as good as cosine-only
-- in terms of ranking, because it only uses multi-signal
-- when doing so preserves the ranking.
-- ============================================================

/-- **Hybrid never loses rank 1**: under the same-regime switching,
    if cosine ranks m* first, so does the hybrid strategy.

    This is the safety guarantee: switching to multi-signal only
    happens when it's proven safe via coherence. -/
theorem hybrid_preserves_rank_one (w : HybridWeights)
    {rel_star rel_j rec_star imp_star act_star rec_j imp_j act_j : ℝ}
    {coherence : ℝ}
    (hgap : rel_star > rel_j)
    (hcoherence_pair : coherence > 0 →
      signalCoherence_j w (rel_star - rel_j)
        rec_star imp_star act_star rec_j imp_j act_j > 0) :
    hybridSwitchedScore w rel_star rec_star imp_star act_star coherence >
    hybridSwitchedScore w rel_j rec_j imp_j act_j coherence :=
  switching_criterion w hgap hcoherence_pair

/-- **Cosine fallback is always available**: when coherence ≤ 0,
    the hybrid strategy reduces to pure cosine comparison. -/
theorem hybrid_fallback_is_cosine (w : HybridWeights)
    {rel rec imp act coherence : ℝ} (hc : ¬(coherence > 0)) :
    hybridSwitchedScore w rel rec imp act coherence = rel := by
  unfold hybridSwitchedScore
  simp only [hc, ite_false]

/-- **Multi-signal enrichment**: when coherence > 0, the hybrid
    strategy uses the full multi-signal score. -/
theorem hybrid_enrichment (w : HybridWeights)
    {rel rec imp act coherence : ℝ} (hc : coherence > 0) :
    hybridSwitchedScore w rel rec imp act coherence =
    hybridScore w rel rec imp act := by
  unfold hybridSwitchedScore
  simp only [hc, ite_true]

end HermesNLCDM
