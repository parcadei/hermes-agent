/-
  HermesNLCDM.Temperature
  =======================
  Phase 5.8: Temperature-Parametric Dynamics

  Main results:
  1. couplingEnergy_β_invariant: coupling energy doesn't depend on β
  2. temperature_perturbed_bound: bounded β change → bounded energy change
  3. temperature_perturbation_abs: |ΔE| ≤ N · δ under bounded local energy change
  4. annealing_descent: coordinate descent + small β change → net descent
  5. lyapunov_annealing: MAIN THEOREM — robust convergence under temperature annealing

  The key insight: couplingEnergy depends only on WeightParams (not β),
  so temperature changes affect only the local energy terms.
  If each local energy changes by at most δ, total energy changes by at most N·δ.
  Combined with Phase 6's strict descent, this gives robust convergence
  under simulated annealing (β-scheduling).

  Reference: Hermes memory architecture — convergence under temperature annealing
-/

import HermesNLCDM.Energy
import HermesNLCDM.Dynamics
import HermesNLCDM.Lyapunov

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Coupling Energy Independence from Temperature

  The coupling energy E_coupling(x_i, x_j) = -½ W(cos(x_i, x_j)) ‖x_i‖ ‖x_j‖
  depends only on the weight parameters and the memory vectors, not on β.
-/

/-- Coupling energy is invariant under temperature changes:
    it depends only on weight parameters and memory vectors, not on β -/
theorem couplingEnergy_β_invariant {d M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_w : cfg'.weights = cfg.weights)
    (x_i x_j : Fin d → ℝ) :
    couplingEnergy cfg'.weights x_i x_j = couplingEnergy cfg.weights x_i x_j := by
  rw [h_w]

/-! ## Temperature Perturbation Bound

  When β changes, only localEnergy terms are affected. If each
  localEnergy changes by at most δ, the total energy changes by at most N·δ.
  This is simpler than the weight perturbation (N² bound) because
  local energies contribute linearly (one per memory), not quadratically.
-/

/-- Bounded temperature perturbation implies bounded energy change.
    If each local energy changes by at most δ, total energy changes by at most N·δ.
    Note: the bound is N·δ (linear), not N²·δ, because local energies
    contribute one per memory (not one per pair). -/
theorem temperature_perturbed_bound {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_w : cfg'.weights = cfg.weights)
    (memories : Fin N → Fin d → ℝ) (δ : ℝ) (_hδ : 0 ≤ δ)
    (h_local_approx : ∀ i : Fin N,
      |localEnergy cfg' (memories i) - localEnergy cfg (memories i)| ≤ δ) :
    totalEnergy cfg' memories ≤ totalEnergy cfg memories + ↑N * δ := by
  -- Decompose total energy
  rw [totalEnergy_decomp cfg' memories, totalEnergy_decomp cfg memories]
  -- Coupling terms are equal (don't depend on β)
  have h_coupling : ∀ i j : Fin N,
      couplingEnergy cfg'.weights (memories i) (memories j) =
      couplingEnergy cfg.weights (memories i) (memories j) :=
    fun i j => couplingEnergy_β_invariant cfg cfg' h_w (memories i) (memories j)
  -- Rewrite coupling sums
  simp_rw [h_coupling]
  -- Suffices: local_sum' ≤ local_sum + N·δ
  have h_local_bound :
    ∑ i : Fin N, localEnergy cfg' (memories i) ≤
    (∑ i : Fin N, localEnergy cfg (memories i)) + ↑N * δ := by
    calc ∑ i : Fin N, localEnergy cfg' (memories i)
        ≤ ∑ i : Fin N, (localEnergy cfg (memories i) + δ) := by
          apply Finset.sum_le_sum; intro i _
          linarith [(abs_le.mp (h_local_approx i)).2]
      _ = (∑ i : Fin N, localEnergy cfg (memories i)) + ↑N * δ := by
          simp_rw [Finset.sum_add_distrib]
          congr 1
          simp [Finset.sum_const]
  linarith

/-! ## Absolute Value Temperature Perturbation Bound -/

/-- Symmetric temperature perturbation bound: |ΔE| ≤ N·δ -/
theorem temperature_perturbation_abs {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_w : cfg'.weights = cfg.weights)
    (memories : Fin N → Fin d → ℝ) (δ : ℝ) (hδ : 0 ≤ δ)
    (h_local_approx : ∀ i : Fin N,
      |localEnergy cfg' (memories i) - localEnergy cfg (memories i)| ≤ δ) :
    |totalEnergy cfg' memories - totalEnergy cfg memories| ≤ ↑N * δ := by
  rw [abs_le]
  constructor
  · -- Lower bound: swap configs
    have h_local_approx' : ∀ i : Fin N,
        |localEnergy cfg (memories i) - localEnergy cfg' (memories i)| ≤ δ := by
      intro i; rw [abs_sub_comm]; exact h_local_approx i
    have := temperature_perturbed_bound cfg' cfg h_w.symm memories δ hδ h_local_approx'
    linarith
  · -- Upper bound
    have := temperature_perturbed_bound cfg cfg' h_w memories δ hδ h_local_approx
    linarith

/-! ## Annealing Descent

  Coordinate descent + bounded temperature change → net descent
  when the descent gap exceeds the local energy perturbation bound.
-/

/-- Annealing descent: coordinate descent survives bounded temperature change
    when the descent exceeds the perturbation bound N·δ. -/
theorem annealing_descent {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_w : cfg'.weights = cfg.weights)
    (memories memories' : Fin N → Fin d → ℝ)
    (δ : ℝ) (hδ : 0 ≤ δ)
    (h_local_approx : ∀ i : Fin N,
      |localEnergy cfg' (memories' i) - localEnergy cfg (memories' i)| ≤ δ)
    (_h_descent : totalEnergy cfg memories' < totalEnergy cfg memories)
    (h_gap : totalEnergy cfg memories - totalEnergy cfg memories' > ↑N * δ) :
    totalEnergy cfg' memories' < totalEnergy cfg memories := by
  have h_perturb := temperature_perturbed_bound cfg cfg' h_w memories' δ hδ h_local_approx
  linarith

/-! ## Robust Lyapunov Convergence Under Annealing

  PHASE 5.8 MAIN THEOREM: Phase 6 convergence survives temperature changes.
-/

/-- PHASE 5.8 MAIN THEOREM: Robust Lyapunov convergence under temperature annealing.
    Coordinate descent at a non-fixed memory gives strict descent that
    survives bounded temperature change, provided the descent gap exceeds N·δ. -/
theorem lyapunov_annealing {d N M_count : ℕ} [NeZero M_count]
    (cfg cfg' : SystemConfig d M_count)
    (h_w : cfg'.weights = cfg.weights)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (hne : ¬isFixedPoint cfg (memories k))
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights (hopfieldUpdate cfg (memories k)) (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j))
    (δ : ℝ) (hδ : 0 ≤ δ)
    (h_local_approx : ∀ i : Fin N,
      |localEnergy cfg' ((coordinateUpdate cfg memories k) i) -
       localEnergy cfg ((coordinateUpdate cfg memories k) i)| ≤ δ)
    (h_gap : totalEnergy cfg memories -
      totalEnergy cfg (coordinateUpdate cfg memories k) > ↑N * δ) :
    totalEnergy cfg' (coordinateUpdate cfg memories k) < totalEnergy cfg memories :=
  annealing_descent cfg cfg' h_w memories (coordinateUpdate cfg memories k) δ hδ
    h_local_approx (lyapunov_strict_descent cfg memories k hne h_pairwise) h_gap

end HermesNLCDM
