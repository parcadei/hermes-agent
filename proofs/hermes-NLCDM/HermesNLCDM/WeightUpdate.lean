/-
  HermesNLCDM.WeightUpdate
  ========================
  Phase 5.7: Weight Perturbation and Robust Convergence

  Main results:
  1. localEnergy_weight_invariant: local energy doesn't depend on coupling weights
  2. weight_perturbed_bound: bounded weight change → bounded energy change
  3. weight_perturbation_abs: symmetric absolute-value bound on energy perturbation
  4. interleaved_descent: coordinate descent + small weight change → net descent
  5. lyapunov_robust: MAIN THEOREM — robust convergence under weight plasticity

  The key insight: localEnergy depends only on β and patterns (not weights),
  so weight perturbation affects only the coupling terms. If the coupling
  perturbation is bounded per pair by ε, the total energy perturbation is
  bounded by N²ε. Combined with Phase 6's strict descent, this gives
  robust convergence under online weight learning.

  Reference: Hermes memory architecture — robust convergence under weight plasticity
-/

import HermesNLCDM.Energy
import HermesNLCDM.Dynamics
import HermesNLCDM.Lyapunov

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Local Energy Independence from Weights

  The local energy term E_local(ξ) = -lse(β, Xᵀξ) + ½‖ξ‖² depends only on β
  and the pattern matrix X, not on the coupling weight parameters.
-/

/-- Local energy is invariant under weight changes: it depends only on β and patterns -/
theorem localEnergy_weight_invariant {d M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_β : cfg'.β = cfg.β) (h_pat : cfg'.patterns = cfg.patterns)
    (ξ : Fin d → ℝ) :
    localEnergy cfg' ξ = localEnergy cfg ξ := by
  unfold localEnergy logSumExp
  rw [h_β, h_pat]

/-! ## Weight Perturbation Bound

  When coupling weights change by at most ε per pair, the total energy
  changes by at most N²ε. The local sum cancels exactly, leaving only
  the coupling sum difference to bound.
-/

/-- Bounded weight perturbation implies bounded energy change.
    If each pairwise coupling energy changes by at most ε,
    total energy changes by at most N²ε. -/
theorem weight_perturbed_bound {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_β : cfg'.β = cfg.β) (h_pat : cfg'.patterns = cfg.patterns)
    (memories : Fin N → Fin d → ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (h_approx : ∀ i j : Fin N,
      |couplingEnergy cfg'.weights (memories i) (memories j) -
       couplingEnergy cfg.weights (memories i) (memories j)| ≤ ε) :
    totalEnergy cfg' memories ≤ totalEnergy cfg memories + ↑N * ↑N * ε := by
  -- Decompose total energy into local + coupling
  rw [totalEnergy_decomp cfg' memories, totalEnergy_decomp cfg memories]
  -- Rewrite local energies using weight invariance
  have h_local : ∀ i, localEnergy cfg' (memories i) = localEnergy cfg (memories i) :=
    fun i => localEnergy_weight_invariant cfg cfg' h_β h_pat (memories i)
  simp_rw [h_local]
  -- Suffices to show: coupling' ≤ coupling + N²ε
  have h_coupling :
    ∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then couplingEnergy cfg'.weights (memories i) (memories j) else 0) ≤
    (∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0)) +
    ↑N * ↑N * ε := by
    calc ∑ i : Fin N, ∑ j : Fin N,
          (if j.val < i.val then couplingEnergy cfg'.weights (memories i) (memories j) else 0)
        ≤ ∑ i : Fin N, ∑ j : Fin N,
          ((if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0) + ε) := by
          apply Finset.sum_le_sum; intro i _
          apply Finset.sum_le_sum; intro j _
          split_ifs with h
          · linarith [(abs_le.mp (h_approx i j)).2]
          · linarith
      _ = (∑ i : Fin N, ∑ j : Fin N,
            (if j.val < i.val then couplingEnergy cfg.weights (memories i) (memories j) else 0)) +
          ↑N * ↑N * ε := by
          simp_rw [Finset.sum_add_distrib]
          congr 1
          simp [Finset.sum_const]; ring
  linarith

/-! ## Absolute Value Perturbation Bound

  The symmetric version: |E(cfg') - E(cfg)| ≤ N²ε, obtained by
  applying the one-sided bound in both directions.
-/

/-- Symmetric perturbation bound: |ΔE| ≤ N²ε under bounded weight change -/
theorem weight_perturbation_abs {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_β : cfg'.β = cfg.β) (h_pat : cfg'.patterns = cfg.patterns)
    (memories : Fin N → Fin d → ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (h_approx : ∀ i j : Fin N,
      |couplingEnergy cfg'.weights (memories i) (memories j) -
       couplingEnergy cfg.weights (memories i) (memories j)| ≤ ε) :
    |totalEnergy cfg' memories - totalEnergy cfg memories| ≤ ↑N * ↑N * ε := by
  rw [abs_le]
  constructor
  · -- Lower bound: -(N²ε) ≤ E' - E, i.e., E ≤ E' + N²ε
    have h_approx' : ∀ i j : Fin N,
        |couplingEnergy cfg.weights (memories i) (memories j) -
         couplingEnergy cfg'.weights (memories i) (memories j)| ≤ ε := by
      intro i j; rw [abs_sub_comm]; exact h_approx i j
    have := weight_perturbed_bound cfg' cfg h_β.symm h_pat.symm memories ε hε h_approx'
    linarith
  · -- Upper bound: E' - E ≤ N²ε
    have := weight_perturbed_bound cfg cfg' h_β h_pat memories ε hε h_approx
    linarith

/-! ## Interleaved Descent

  When coordinate descent gives strict descent by some amount δ > N²ε,
  and weights are then perturbed by at most ε per pair, the net effect
  is still a strict decrease in total energy.
-/

/-- Interleaved descent: coordinate descent survives bounded weight perturbation
    when the descent exceeds the perturbation bound. -/
theorem interleaved_descent {d N M_count : ℕ}
    (cfg cfg' : SystemConfig d M_count)
    (h_β : cfg'.β = cfg.β) (h_pat : cfg'.patterns = cfg.patterns)
    (memories memories' : Fin N → Fin d → ℝ)
    (ε : ℝ) (hε : 0 ≤ ε)
    (h_approx : ∀ i j : Fin N,
      |couplingEnergy cfg'.weights (memories' i) (memories' j) -
       couplingEnergy cfg.weights (memories' i) (memories' j)| ≤ ε)
    (_h_descent : totalEnergy cfg memories' < totalEnergy cfg memories)
    (h_gap : totalEnergy cfg memories - totalEnergy cfg memories' > ↑N * ↑N * ε) :
    totalEnergy cfg' memories' < totalEnergy cfg memories := by
  have h_perturb := weight_perturbed_bound cfg cfg' h_β h_pat memories' ε hε h_approx
  linarith

/-! ## Robust Lyapunov Convergence

  PHASE 5.7 MAIN THEOREM: The Lyapunov convergence from Phase 6 is robust
  under bounded weight perturbation. If a memory is not a fixed point and
  the descent gap exceeds the perturbation bound N²ε, coordinate descent
  under the new weights still strictly decreases energy.
-/

/-- PHASE 5.7 MAIN THEOREM: Robust Lyapunov convergence under weight plasticity.
    Coordinate descent at a non-fixed memory gives strict descent that
    survives bounded weight perturbation, provided the descent gap exceeds N²ε. -/
theorem lyapunov_robust {d N M_count : ℕ} [NeZero M_count]
    (cfg cfg' : SystemConfig d M_count)
    (h_β : cfg'.β = cfg.β) (h_pat : cfg'.patterns = cfg.patterns)
    (memories : Fin N → Fin d → ℝ) (k : Fin N)
    (hne : ¬isFixedPoint cfg (memories k))
    (h_pairwise : ∀ j : Fin N, j ≠ k →
      couplingEnergy cfg.weights (hopfieldUpdate cfg (memories k)) (memories j) ≤
      couplingEnergy cfg.weights (memories k) (memories j))
    (ε : ℝ) (hε : 0 ≤ ε)
    (h_weight_approx : ∀ i j : Fin N,
      |couplingEnergy cfg'.weights
        ((coordinateUpdate cfg memories k) i)
        ((coordinateUpdate cfg memories k) j) -
       couplingEnergy cfg.weights
        ((coordinateUpdate cfg memories k) i)
        ((coordinateUpdate cfg memories k) j)| ≤ ε)
    (h_gap : totalEnergy cfg memories -
      totalEnergy cfg (coordinateUpdate cfg memories k) > ↑N * ↑N * ε) :
    totalEnergy cfg' (coordinateUpdate cfg memories k) < totalEnergy cfg memories :=
  interleaved_descent cfg cfg' h_β h_pat memories (coordinateUpdate cfg memories k) ε hε
    h_weight_approx (lyapunov_strict_descent cfg memories k hne h_pairwise) h_gap

end HermesNLCDM
