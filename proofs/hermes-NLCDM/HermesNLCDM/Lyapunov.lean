/-
  HermesNLCDM.Lyapunov
  ====================
  Phase 6: Full System Lyapunov Function

  Main results:
  1. coordinateUpdate: update a single memory via Hopfield dynamics
  2. lyapunov_nonincreasing: totalEnergy doesn't increase under coordinate update
  3. lyapunov_strict_descent: totalEnergy strictly decreases at non-fixed memories
  4. isSystemEquilibrium: all memories are fixed points
  5. equilibrium_coordinate_fixed: coordinate update = identity at equilibrium
  6. not_equilibrium_energy_decreases: away from equilibrium, ∃ strict descent

  The key insight: totalEnergy is a Lyapunov function for coordinate descent.
  Combined with bounded below (Phase 1), this gives convergence guarantees.

  Reference: Hermes memory architecture — formal convergence of coupled dynamics
-/

import HermesNLCDM.Energy
import HermesNLCDM.Dynamics
import HermesNLCDM.Coupling
import HermesNLCDM.Capacity

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Coordinate Update

  Update a single memory by applying the Hopfield update rule.
  All other memories remain unchanged.
-/

/-- Coordinate update: apply Hopfield dynamics to memory k, keep others fixed -/
def coordinateUpdate {d N M_count : ℕ} (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) (k : Fin N) : Fin N → Fin d → ℝ :=
  fun i => if i = k then hopfieldUpdate cfg (memories k) else memories i

/-! ## Lyapunov Non-Increasing

  Total energy does not increase under coordinate update,
  given that pairwise coupling energies don't increase.
  Wraps coupled_descent_pairwise from Phase 4.
-/

/-- Total energy is non-increasing under coordinate update -/
theorem lyapunov_nonincreasing {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights (hopfieldUpdate cfg (memories k)) (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j)) :
    totalEnergy cfg (coordinateUpdate cfg memories k) ≤ totalEnergy cfg memories :=
  coupled_descent_pairwise cfg memories k (hopfieldUpdate cfg (memories k))
    (energy_descent cfg (memories k)) h_pairwise

/-! ## Lyapunov Strict Descent

  PHASE 6 KEY THEOREM: When memory k is NOT a fixed point,
  coordinate update at k STRICTLY decreases total energy.

  Proof: decompose totalEnergy into local sum + coupling sum.
  Local sum strictly decreases (one term strict at k, rest equal).
  Coupling sum doesn't increase (term-by-term from pairwise bound).
  Combine: strict + non-strict → strict.
-/

/-- PHASE 6 KEY THEOREM: Strict energy descent at non-fixed memories -/
theorem lyapunov_strict_descent {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (hne : ¬isFixedPoint cfg (memories k))
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights (hopfieldUpdate cfg (memories k)) (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j)) :
    totalEnergy cfg (coordinateUpdate cfg memories k) < totalEnergy cfg memories := by
  rw [totalEnergy_decomp cfg (coordinateUpdate cfg memories k),
      totalEnergy_decomp cfg memories]
  have h_local : (∑ i, localEnergy cfg (coordinateUpdate cfg memories k i)) <
      (∑ i, localEnergy cfg (memories i)) := by
    apply Finset.sum_lt_sum
    · intro i _
      simp only [coordinateUpdate]
      split_ifs with h
      · rw [h]; exact energy_descent cfg (memories k)
      · exact le_refl _
    · refine ⟨k, mem_univ k, ?_⟩
      simp [coordinateUpdate]
      exact energy_descent_strict cfg (memories k) hne
  have h_coupling : (∑ i, ∑ j, if j.val < i.val then
        couplingEnergy cfg.weights (coordinateUpdate cfg memories k i)
          (coordinateUpdate cfg memories k j) else 0) ≤
      (∑ i, ∑ j, if j.val < i.val then
        couplingEnergy cfg.weights (memories i) (memories j) else 0) := by
    apply Finset.sum_le_sum; intro i _
    apply Finset.sum_le_sum; intro j _
    split_ifs with h
    · -- j.val < i.val: case split on whether i or j equals k
      simp only [coordinateUpdate]
      by_cases hi : i = k <;> by_cases hj : j = k
      · -- i = k, j = k: impossible since j.val < i.val
        exfalso; omega
      · -- i = k, j ≠ k
        simp [hi, hj]; exact h_pairwise j hj
      · -- i ≠ k, j = k
        simp [hi, hj]
        rw [couplingEnergy_comm cfg.weights (memories i) (hopfieldUpdate cfg (memories k)),
            couplingEnergy_comm cfg.weights (memories i) (memories k)]
        exact h_pairwise i hi
      · -- neither equals k: unchanged
        simp [hi, hj]
    · exact le_refl _
  linarith

/-! ## System Equilibrium

  A system is at equilibrium when every memory is a fixed point.
  At equilibrium, coordinate updates are the identity.
-/

/-- System equilibrium: all memories are Hopfield fixed points -/
def isSystemEquilibrium {d N M_count : ℕ} (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) : Prop :=
  ∀ k : Fin N, isFixedPoint cfg (memories k)

/-- At equilibrium, coordinate update is the identity -/
theorem equilibrium_coordinate_fixed {d N M_count : ℕ} (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (h_eq : isSystemEquilibrium cfg memories) :
    coordinateUpdate cfg memories k = memories := by
  funext i
  simp only [coordinateUpdate]
  split_ifs with h
  · rw [h]; exact h_eq k
  · rfl

/-- PHASE 6 THEOREM: Away from equilibrium, some update strictly decreases energy -/
theorem not_equilibrium_energy_decreases {d N M_count : ℕ} [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (h_not_eq : ¬isSystemEquilibrium cfg memories)
    (h_pairwise_all : ∀ k : Fin N, ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights (hopfieldUpdate cfg (memories k)) (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j)) :
    ∃ k : Fin N, totalEnergy cfg (coordinateUpdate cfg memories k) < totalEnergy cfg memories := by
  unfold isSystemEquilibrium at h_not_eq
  push_neg at h_not_eq
  obtain ⟨k, hk⟩ := h_not_eq
  exact ⟨k, lyapunov_strict_descent cfg memories k hk (h_pairwise_all k)⟩

end HermesNLCDM
