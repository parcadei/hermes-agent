/-
  HermesNLCDM.Coupling
  ====================
  Phase 3: Pairwise Coupling

  Main results:
  1. Cosine similarity is symmetric: cos(u, v) = cos(v, u)
  2. Coupling energy is symmetric: E_coupling(u, v) = E_coupling(v, u)
  3. Coupled energy descent: if each memory's local energy decreases
     under its Hopfield update, the total energy (local + coupling) decreases
     when coupling terms are bounded.
  4. Positive coupling → shared basin: W_ij > 0 lowers the joint energy
     when memories are similar.
  5. Negative coupling → basin separation: W_ij < 0 raises the joint energy
     when memories are similar, pushing them apart.

  Reference: Slotine 2003, Wang & Slotine 2005 (contraction analysis),
             Ramsauer et al. 2020 (Modern Hopfield energy)
-/

import HermesNLCDM.Energy
import HermesNLCDM.Dynamics
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Cosine Similarity Symmetry -/

/-- The dot product ∑ u_i * v_i is commutative -/
theorem dot_comm {d : ℕ} (u v : Fin d → ℝ) :
    ∑ i, u i * v i = ∑ i, v i * u i := by
  congr 1; ext i; ring

/-- Cosine similarity is symmetric: cos(u, v) = cos(v, u) -/
theorem cosineSim_comm {d : ℕ} (u v : Fin d → ℝ) :
    cosineSim u v = cosineSim v u := by
  unfold cosineSim
  rw [dot_comm u v, mul_comm (Real.sqrt (∑ i, u i ^ 2)) (Real.sqrt (∑ i, v i ^ 2))]

/-! ## Coupling Energy Symmetry -/

/-- Coupling energy is symmetric: E_coupling(u, v) = E_coupling(v, u) -/
theorem couplingEnergy_comm {d : ℕ} (wp : WeightParams) (u v : Fin d → ℝ) :
    couplingEnergy wp u v = couplingEnergy wp v u := by
  unfold couplingEnergy
  rw [cosineSim_comm u v]
  ring

/-! ## Coupling Energy Sign Properties

  The sign of the coupling energy determines whether memories attract or repel:
  - W(s) > 0 (attractive): E_coupling < 0, lowering total energy when memories align
  - W(s) < 0 (repulsive): E_coupling > 0, raising total energy when memories align
  - W(s) = 0 (neutral):   E_coupling = 0, memories evolve independently
-/

/-- When the smooth weight is positive, coupling energy is nonpositive
    (attractive coupling lowers energy) -/
theorem couplingEnergy_nonpos_of_weight_pos {d : ℕ} (wp : WeightParams)
    (u v : Fin d → ℝ)
    (hw : 0 < smoothWeight wp (cosineSim u v)) :
    couplingEnergy wp u v ≤ 0 := by
  unfold couplingEnergy
  have h1 : 0 ≤ Real.sqrt (∑ k, u k ^ 2) := Real.sqrt_nonneg _
  have h2 : 0 ≤ Real.sqrt (∑ k, v k ^ 2) := Real.sqrt_nonneg _
  nlinarith [mul_nonneg h1 h2]

/-- When the smooth weight is negative, coupling energy is nonneg
    (repulsive coupling raises energy) -/
theorem couplingEnergy_nonneg_of_weight_neg {d : ℕ} (wp : WeightParams)
    (u v : Fin d → ℝ)
    (hw : smoothWeight wp (cosineSim u v) < 0) :
    0 ≤ couplingEnergy wp u v := by
  unfold couplingEnergy
  have h1 : 0 ≤ Real.sqrt (∑ k, u k ^ 2) := Real.sqrt_nonneg _
  have h2 : 0 ≤ Real.sqrt (∑ k, v k ^ 2) := Real.sqrt_nonneg _
  nlinarith [mul_nonneg h1 h2]

/-- When the smooth weight is zero, coupling energy vanishes -/
theorem couplingEnergy_zero_of_weight_zero {d : ℕ} (wp : WeightParams)
    (u v : Fin d → ℝ)
    (hw : smoothWeight wp (cosineSim u v) = 0) :
    couplingEnergy wp u v = 0 := by
  unfold couplingEnergy; rw [hw]; ring

/-! ## Positive Coupling → Shared Basin

  When W(cos(x_μ, x_ν)) > 0, memories μ and ν have positive coupling.
  The coupling energy E_coupling(x_μ, x_ν) < 0 creates a shared energy
  well: both memories minimize total energy by staying close to each other.
-/

/-- Positive coupling between two patterns means their coupling energy is negative,
    which deepens the total energy well — they share a basin of attraction -/
theorem shared_basin_of_positive_coupling {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N)
    (μ ν : Fin N)
    (hW_pos : 0 < smoothWeight cfg.weights
      (cosineSim (cfg.patterns μ) (cfg.patterns ν))) :
    couplingEnergy cfg.weights (cfg.patterns μ) (cfg.patterns ν) ≤ 0 :=
  couplingEnergy_nonpos_of_weight_pos cfg.weights _ _ hW_pos

/-! ## Negative Coupling → Basin Separation

  When W(cos(x_μ, x_ν)) < 0, memories μ and ν have negative coupling.
  The coupling energy E_coupling(x_μ, x_ν) > 0 raises the total energy
  when both memories are activated, effectively separating their basins.
-/

/-- Negative coupling between two patterns means their coupling energy is positive,
    which creates an energy barrier — they have separated basins of attraction -/
theorem separated_basins_of_negative_coupling {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N)
    (μ ν : Fin N)
    (hW_neg : smoothWeight cfg.weights
      (cosineSim (cfg.patterns μ) (cfg.patterns ν)) < 0) :
    0 ≤ couplingEnergy cfg.weights (cfg.patterns μ) (cfg.patterns ν) :=
  couplingEnergy_nonneg_of_weight_neg cfg.weights _ _ hW_neg

/-! ## Coupled System Total Energy Descent

  Main theorem: For the coupled system where each memory independently
  undergoes its Hopfield update, the total energy decreases when:
  1. Each local energy decreases (Phase 2: energy_descent)
  2. The coupling change is bounded

  The proof decomposes totalEnergy into local + coupling terms.
-/

/-- Decomposition: total energy = sum of local energies + coupling terms -/
theorem totalEnergy_decomp {d N M_count : ℕ} (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) :
    totalEnergy cfg memories =
      (∑ i, localEnergy cfg (memories i)) +
      ∑ i : Fin N, ∑ j : Fin N,
        if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0 := by
  rfl

/-- When all local energies decrease and coupling terms are frozen,
    total energy decreases -/
theorem totalEnergy_local_descent_frozen_coupling {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (updated : Fin N → Fin d → ℝ)
    (h_local : ∀ i, localEnergy cfg (updated i) ≤ localEnergy cfg (memories i))
    (h_coupling_frozen : ∀ i j, couplingEnergy cfg.weights (updated i) (updated j) =
      couplingEnergy cfg.weights (memories i) (memories j)) :
    totalEnergy cfg updated ≤ totalEnergy cfg memories := by
  rw [totalEnergy_decomp, totalEnergy_decomp]
  apply add_le_add
  · exact Finset.sum_le_sum (fun i _ => h_local i)
  · apply Finset.sum_le_sum; intro i _
    apply Finset.sum_le_sum; intro j _
    split_ifs with h
    · exact le_of_eq (h_coupling_frozen i j)
    · exact le_refl _

/-! ## Single-Memory Update Descent (Coordinate Descent)

  When we update a single memory k while freezing all others, the total
  energy change decomposes into:
  1. Local energy change at k: E_local(k_new) - E_local(k_old) ≤ 0 (Phase 2)
  2. Coupling changes involving k: Σ_j [E_coupling(k_new, j) - E_coupling(k_old, j)]

  The hypothesis h_coupling_bound captures that the coupling changes are bounded.
-/

/-- Helper: when i ≠ k and j ≠ k, the coupling term with updated memories
    equals the original (since both memories are unchanged) -/
private theorem coupling_unchanged_when_not_k {d N M_count : ℕ}
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (k : Fin N) (updated_k : Fin d → ℝ)
    (i j : Fin N) (hi : i ≠ k) (hj : j ≠ k) :
    let updated : Fin N → Fin d → ℝ := fun m => if m = k then updated_k else memories m
    couplingEnergy cfg.weights (updated i) (updated j) =
    couplingEnergy cfg.weights (memories i) (memories j) := by
  simp only [if_neg hi, if_neg hj]

/-- PHASE 3 MAIN THEOREM: Single-memory update descent for the coupled system.

    When we update memory k while freezing all other memories, the total energy
    decreases. This is the alternating minimization / coordinate descent approach.

    The local energy descent at k (Phase 2) plus the coupling bound ensures
    overall descent. -/
theorem coupled_single_update_descent {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (k : Fin N)
    (updated_k : Fin d → ℝ)
    (h_local_descent : localEnergy cfg updated_k ≤ localEnergy cfg (memories k))
    (h_coupling_bound :
      (∑ j : Fin N,
        if j.val < k.val then
          couplingEnergy cfg.weights updated_k (memories j)
        else 0) +
      (∑ i : Fin N,
        if k.val < i.val then
          couplingEnergy cfg.weights (memories i) updated_k
        else 0) ≤
      (∑ j : Fin N,
        if j.val < k.val then
          couplingEnergy cfg.weights (memories k) (memories j)
        else 0) +
      (∑ i : Fin N,
        if k.val < i.val then
          couplingEnergy cfg.weights (memories i) (memories k)
        else 0)) :
    let updated : Fin N → Fin d → ℝ := fun i =>
      if i = k then updated_k else memories i
    totalEnergy cfg updated ≤ totalEnergy cfg memories := by
  intro updated
  rw [totalEnergy_decomp, totalEnergy_decomp]
  -- Local energy sum decreases
  have h_local_sum : ∑ i, localEnergy cfg (updated i) ≤ ∑ i, localEnergy cfg (memories i) := by
    apply Finset.sum_le_sum; intro i _
    show localEnergy cfg (if i = k then updated_k else memories i) ≤ localEnergy cfg (memories i)
    split_ifs with h
    · rw [h]; exact h_local_descent
    · exact le_refl _
  -- Coupling sum: need to show updated coupling ≤ old coupling
  suffices h_coupling :
    ∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then couplingEnergy cfg.weights (updated i) (updated j) else 0) ≤
    ∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0) by
    linarith
  -- For each (i, j) with j < i, classify by whether i = k or j = k:
  -- Define f_new and f_old as the coupling summands
  set f_new : Fin N → Fin N → ℝ := fun i j =>
    if j.val < i.val then couplingEnergy cfg.weights (updated i) (updated j) else 0
  set f_old : Fin N → Fin N → ℝ := fun i j =>
    if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0
  -- Show that f_new(i,j) ≤ f_old(i,j) + correction, summing corrections = h_coupling_bound
  -- Actually, let's show the double sums are equal to decomposed forms and use the bound directly.
  -- Key insight: when neither i = k nor j = k, updated i = memories i and updated j = memories j
  -- So f_new(i,j) = f_old(i,j). The difference comes only from terms where i = k or j = k.
  -- Split the double sum into: terms where i = k, terms where j = k (and j < i so k < i),
  -- and terms where neither i = k nor j = k.
  -- We show the double sum difference equals the difference from h_coupling_bound.
  suffices h_diff :
    ∑ i, ∑ j, f_new i j - ∑ i, ∑ j, f_old i j ≤ 0 by linarith
  -- Combine the sums
  rw [show ∑ i, ∑ j, f_new i j - ∑ i, ∑ j, f_old i j =
      ∑ i, (∑ j, f_new i j - ∑ j, f_old i j) from by
    rw [← Finset.sum_sub_distrib]]
  rw [show ∑ i : Fin N, (∑ j, f_new i j - ∑ j, f_old i j) =
      ∑ i : Fin N, ∑ j, (f_new i j - f_old i j) from by
    congr 1; ext i; rw [← Finset.sum_sub_distrib]]
  -- For i ≠ k and j ≠ k: f_new i j - f_old i j = 0
  -- For i = k: f_new k j - f_old k j = contribution from updated_k vs memories k
  -- For j = k (with k < i): f_new i k - f_old i k = contribution from updated_k vs memories k
  -- Show each term of the double sum
  have h_term : ∀ i j : Fin N,
      f_new i j - f_old i j =
      (if j.val < i.val then
        couplingEnergy cfg.weights (updated i) (updated j) -
        couplingEnergy cfg.weights (memories i) (memories j)
      else 0) := by
    intro i j; simp only [f_new, f_old]; split_ifs <;> ring
  simp_rw [h_term]
  -- Now classify each term
  have h_unchanged : ∀ i j : Fin N, i ≠ k → j ≠ k → j.val < i.val →
      couplingEnergy cfg.weights (updated i) (updated j) =
      couplingEnergy cfg.weights (memories i) (memories j) := by
    intro i j hi hj _
    show couplingEnergy cfg.weights (if i = k then updated_k else memories i)
        (if j = k then updated_k else memories j) = _
    rw [if_neg hi, if_neg hj]
  -- For terms where i = k: updated i = updated_k, updated j = memories j (since j ≠ k because j < k = i implies j ≠ k)
  -- Wait, j < i and i = k means j.val < k.val, so j ≠ k
  have h_at_k_row : ∀ j : Fin N, j.val < k.val →
      couplingEnergy cfg.weights (updated k) (updated j) =
      couplingEnergy cfg.weights updated_k (memories j) := by
    intro j hj
    have hjk : j ≠ k := by intro h; rw [h] at hj; omega
    show couplingEnergy cfg.weights (if k = k then updated_k else memories k)
        (if j = k then updated_k else memories j) = _
    rw [if_pos rfl, if_neg hjk]
  -- For terms where j = k: i ≠ k (since k < i), updated i = memories i, updated j = updated_k
  have h_at_k_col : ∀ i : Fin N, k.val < i.val →
      couplingEnergy cfg.weights (updated i) (updated k) =
      couplingEnergy cfg.weights (memories i) updated_k := by
    intro i hi
    have hik : i ≠ k := by intro h; rw [h] at hi; omega
    show couplingEnergy cfg.weights (if i = k then updated_k else memories i)
        (if k = k then updated_k else memories k) = _
    rw [if_neg hik, if_pos rfl]
  -- The full double sum of differences = (row k differences) + (col k differences)
  -- because all other terms are 0
  -- Let's compute sum_i sum_j directly
  -- For each pair (i, j) with j < i:
  --   If i = k:  diff = E(updated_k, memories j) - E(memories k, memories j)
  --   If j = k:  diff = E(memories i, updated_k) - E(memories i, memories k)
  --   If i ≠ k ∧ j ≠ k: diff = 0
  -- Note: i = k and j = k can't both hold when j < i
  -- Rewrite the sum
  have h_sum_eq :
    ∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then
        couplingEnergy cfg.weights (updated i) (updated j) -
        couplingEnergy cfg.weights (memories i) (memories j)
      else 0) =
    (∑ j : Fin N,
      if j.val < k.val then
        couplingEnergy cfg.weights updated_k (memories j) -
        couplingEnergy cfg.weights (memories k) (memories j)
      else 0) +
    (∑ i : Fin N,
      if k.val < i.val then
        couplingEnergy cfg.weights (memories i) updated_k -
        couplingEnergy cfg.weights (memories i) (memories k)
      else 0) := by
    -- Extract the k-th row and k-th column terms
    -- For the outer sum over i, separate i = k from i ≠ k
    rw [show ∑ i : Fin N, ∑ j : Fin N,
        (if j.val < i.val then
          couplingEnergy cfg.weights (updated i) (updated j) -
          couplingEnergy cfg.weights (memories i) (memories j)
        else 0) =
      (∑ j : Fin N,
        (if j.val < k.val then
          couplingEnergy cfg.weights (updated k) (updated j) -
          couplingEnergy cfg.weights (memories k) (memories j)
        else 0)) +
      (∑ i ∈ Finset.univ.erase k, ∑ j : Fin N,
        (if j.val < i.val then
          couplingEnergy cfg.weights (updated i) (updated j) -
          couplingEnergy cfg.weights (memories i) (memories j)
        else 0)) from by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]]
    congr 1
    · -- Row k: updated k = updated_k, and for j < k, j ≠ k
      congr 1; ext j; split_ifs with h
      · rw [h_at_k_row j h]
      · rfl
    · -- Remaining rows: for i ≠ k, separate j = k from j ≠ k
      -- In each remaining row i (with i ≠ k), the only nonzero diff term is at j = k (if k < i)
      have h_remaining : ∀ i ∈ Finset.univ.erase k,
          ∑ j : Fin N,
            (if j.val < i.val then
              couplingEnergy cfg.weights (updated i) (updated j) -
              couplingEnergy cfg.weights (memories i) (memories j)
            else 0) =
          (if k.val < i.val then
            couplingEnergy cfg.weights (memories i) updated_k -
            couplingEnergy cfg.weights (memories i) (memories k)
          else 0) := by
        intro i hi
        have hik : i ≠ k := Finset.ne_of_mem_erase hi
        -- All terms with j ≠ k are 0 (both i ≠ k and j ≠ k)
        -- The only potentially nonzero term is j = k
        rw [show ∑ j : Fin N,
            (if j.val < i.val then
              couplingEnergy cfg.weights (updated i) (updated j) -
              couplingEnergy cfg.weights (memories i) (memories j)
            else 0) =
          (if k.val < i.val then
            couplingEnergy cfg.weights (updated i) (updated k) -
            couplingEnergy cfg.weights (memories i) (memories k)
          else 0) +
          ∑ j ∈ Finset.univ.erase k,
            (if j.val < i.val then
              couplingEnergy cfg.weights (updated i) (updated j) -
              couplingEnergy cfg.weights (memories i) (memories j)
            else 0) from by
          rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]]
        -- The remaining sum (j ≠ k, i ≠ k) is 0
        have h_rest_zero : ∑ j ∈ Finset.univ.erase k,
            (if j.val < i.val then
              couplingEnergy cfg.weights (updated i) (updated j) -
              couplingEnergy cfg.weights (memories i) (memories j)
            else 0) = 0 := by
          apply Finset.sum_eq_zero; intro j hj
          have hjk : j ≠ k := Finset.ne_of_mem_erase hj
          split_ifs with h
          · rw [h_unchanged i j hik hjk h]; ring
          · rfl
        rw [h_rest_zero, add_zero]
        split_ifs with h
        · rw [h_at_k_col i h]
        · rfl
      rw [Finset.sum_congr rfl h_remaining]
      -- Now we have ∑_{i ∈ univ.erase k} (if k < i then ... else 0)
      -- = ∑_{i ∈ univ} (if k < i then ... else 0) since the k-th term has ¬(k < k)
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]
      simp only [show ¬(k.val < k.val) from lt_irrefl _, if_false, zero_add]
  rw [h_sum_eq]
  -- Goal: (∑ j, if j<k then E_new_k_j - E_old_k_j else 0) +
  --       (∑ i, if k<i then E_new_i_k - E_old_i_k else 0) ≤ 0
  -- Split each ∑(if P then a-b else 0) = ∑(if P then a else 0) - ∑(if P then b else 0)
  have split_if_sub : ∀ (a b : Fin N → ℝ) (P : Fin N → Prop) [DecidablePred P],
      ∑ i : Fin N, (if P i then a i - b i else (0 : ℝ)) =
      (∑ i : Fin N, if P i then a i else 0) - (∑ i : Fin N, if P i then b i else 0) := by
    intro a b P _
    rw [← Finset.sum_sub_distrib]
    congr 1; ext i; split_ifs <;> ring
  rw [split_if_sub, split_if_sub]
  linarith

end HermesNLCDM
