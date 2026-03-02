/-
  HermesNLCDM.BlockDiagonal
  =========================
  Phase 5.5: Block-Diagonal Decomposition — Conditions for Modularity

  Main results:
  1. Similarity graph: memories connected iff coupling weight ≠ 0
  2. Connected components induce a natural module partition
  3. Component partition satisfies inter-module neutrality (Phase 5)
  4. Bridge theorem: graph structure → energy decomposition
  5. Approximate block-diagonality: small cross-cut coupling → bounded error

  The key insight: the coupling matrix W is block-diagonal iff the
  similarity graph (edges where smoothWeight ≠ 0) is disconnected.
  Connected components give the natural modular decomposition.

  Reference: Hermes memory architecture — automatic module discovery via graph connectivity
-/

import HermesNLCDM.Energy
import HermesNLCDM.Modular
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected

noncomputable section
open Real Finset Classical

namespace HermesNLCDM

/-! ## Similarity Graph

  The similarity graph has memories as vertices and edges where the
  coupling weight is nonzero. This captures which memories interact.
-/

/-- The similarity graph: two memories are adjacent iff their coupling
    weight is nonzero. This is a SimpleGraph because:
    - Irreflexive: we exclude self-loops (i ≠ j required)
    - Symmetric: smoothWeight(cosineSim(u,v)) = smoothWeight(cosineSim(v,u))
      by cosineSim_comm -/
def similarityGraph {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) : SimpleGraph (Fin N) where
  Adj i j := i ≠ j ∧ smoothWeight wp (cosineSim (memories i) (memories j)) ≠ 0
  symm := by
    intro i j ⟨hij, hw⟩
    exact ⟨hij.symm, by rwa [cosineSim_comm] at hw⟩
  loopless := ⟨fun i ⟨hii, _⟩ => hii rfl⟩

/-! ## Non-Adjacent Vertices Have Zero Coupling

  The contrapositive of the graph definition: if two memories are NOT
  adjacent (and are distinct), their coupling weight is zero.
-/

/-- Non-adjacent distinct memories have zero coupling weight -/
theorem non_adj_weight_zero {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (i j : Fin N)
    (hij : i ≠ j)
    (h_not_adj : ¬ (similarityGraph wp memories).Adj i j) :
    smoothWeight wp (cosineSim (memories i) (memories j)) = 0 := by
  by_contra h
  exact h_not_adj ⟨hij, h⟩

/-! ## Reachability and Module Structure

  Two memories are in the same "module" iff they are connected by a path
  in the similarity graph. The connected components partition memories
  into modules where cross-component coupling vanishes.
-/

/-- If two memories are NOT reachable from each other in the similarity graph,
    their coupling weight is zero -/
theorem unreachable_weight_zero {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (i j : Fin N)
    (hij : i ≠ j)
    (h_unreach : ¬ (similarityGraph wp memories).Reachable i j) :
    smoothWeight wp (cosineSim (memories i) (memories j)) = 0 := by
  apply non_adj_weight_zero wp memories i j hij
  intro hadj
  exact h_unreach (SimpleGraph.Adj.reachable hadj)

/-! ## Connected Component Partition

  Any partition that separates unreachable vertices satisfies
  inter-module neutrality. Connected components are the coarsest
  such partition, but any refinement also works.
-/

/-- A module partition is compatible with the similarity graph if memories
    in different modules are not reachable from each other -/
def partitionCompatible {d N K : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (mp : ModulePartition N K) : Prop :=
  ∀ i j : Fin N, mp.assign i ≠ mp.assign j →
    ¬ (similarityGraph wp memories).Reachable i j

/-- PHASE 5.5 THEOREM: Any compatible partition satisfies inter-module neutrality.
    This bridges graph structure (connectivity) to energy decomposition (Phase 5). -/
theorem compatible_partition_neutral {d N K : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (mp : ModulePartition N K)
    (h_compat : partitionCompatible wp memories mp) :
    interModuleNeutral mp wp memories := by
  intro i j hij
  by_cases h_eq : i = j
  · exact absurd (h_eq ▸ rfl : mp.assign i = mp.assign i) (h_eq ▸ hij)
  · exact unreachable_weight_zero wp memories i j h_eq (h_compat i j hij)

/-! ## The Canonical Partition

  The connected components of the similarity graph give the canonical
  (coarsest) compatible partition. We define this using Mathlib's
  ConnectedComponent quotient type.
-/

/-- The canonical assignment: each memory maps to its connected component -/
def canonicalAssign {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (i : Fin N) :
    (similarityGraph wp memories).ConnectedComponent :=
  SimpleGraph.connectedComponentMk _ i

/-- Two memories get the same canonical assignment iff they are reachable -/
theorem canonicalAssign_eq_iff {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (i j : Fin N) :
    canonicalAssign wp memories i = canonicalAssign wp memories j ↔
    (similarityGraph wp memories).Reachable i j := by
  exact SimpleGraph.ConnectedComponent.eq

/-- Different canonical assignments imply zero coupling weight -/
theorem canonicalAssign_ne_weight_zero {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) (i j : Fin N)
    (h_diff : canonicalAssign wp memories i ≠ canonicalAssign wp memories j) :
    smoothWeight wp (cosineSim (memories i) (memories j)) = 0 := by
  have h_not_reach : ¬ (similarityGraph wp memories).Reachable i j :=
    (canonicalAssign_eq_iff wp memories i j).not.mp h_diff
  have hij : i ≠ j := by
    intro heq; subst heq
    exact h_not_reach (@SimpleGraph.Reachable.refl _ (similarityGraph wp memories) i)
  exact unreachable_weight_zero wp memories i j hij h_not_reach

/-! ## Connected Components Form a Finite Type

  Since Fin N is finite, the connected components of the similarity graph
  are also finite. This gives us the module count K.
-/

/-- Connected components of the similarity graph are finite -/
instance connectedComponent_fintype {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) :
    Fintype (similarityGraph wp memories).ConnectedComponent := by
  letI : DecidableEq (similarityGraph wp memories).ConnectedComponent := decEq _
  exact Fintype.ofSurjective (SimpleGraph.connectedComponentMk _)
    (fun c => Quot.inductionOn c (fun v => ⟨v, rfl⟩))

/-- Number of connected components (= number of modules) -/
def componentCount {d N : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ) : ℕ :=
  Fintype.card (similarityGraph wp memories).ConnectedComponent

/-! ## Canonical Module Partition from Graph Components

  Any equivalence between connected components and Fin K gives a
  ModulePartition. The canonical one uses Fintype.equivFin.
-/

/-- Build a ModulePartition from any equivalence between connected components and Fin K -/
def partitionFromComponents {d N K : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ)
    (equiv : (similarityGraph wp memories).ConnectedComponent ≃ Fin K) :
    ModulePartition N K where
  assign i := equiv (canonicalAssign wp memories i)

/-- The partition from connected components is always compatible with the graph -/
theorem partitionFromComponents_compatible {d N K : ℕ} (wp : WeightParams)
    (memories : Fin N → Fin d → ℝ)
    (equiv : (similarityGraph wp memories).ConnectedComponent ≃ Fin K) :
    partitionCompatible wp memories (partitionFromComponents wp memories equiv) := by
  intro i j h_diff_assign
  have h_diff_comp : canonicalAssign wp memories i ≠ canonicalAssign wp memories j := by
    intro h_eq
    exact h_diff_assign (by simp [partitionFromComponents, h_eq])
  exact (canonicalAssign_eq_iff wp memories i j).not.mp h_diff_comp

/-! ## Bridge Theorem: Graph Structure → Energy Decomposition

  The connected components of the similarity graph give a module partition
  that satisfies inter-module neutrality, which by Phase 5's main theorem
  implies total energy decomposes into per-module energies.
-/

/-- PHASE 5.5 BRIDGE THEOREM: The similarity graph's connected components
    induce a module partition under which total energy decomposes.

    E_total = Σ_m E_module(m)

    This connects the graph-theoretic structure (Phase 5.5) to the
    algebraic energy decomposition (Phase 5). -/
theorem graph_energy_decomposition {d N K M_count : ℕ} [NeZero K] [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (memories : Fin N → Fin d → ℝ)
    (equiv : (similarityGraph cfg.weights memories).ConnectedComponent ≃ Fin K) :
    totalEnergy cfg memories =
    ∑ m, moduleEnergy cfg (partitionFromComponents cfg.weights memories equiv) memories m := by
  apply totalEnergy_module_decomp
  exact compatible_partition_neutral cfg.weights memories _
    (partitionFromComponents_compatible cfg.weights memories equiv)

/-! ## Approximate Block-Diagonality

  When cross-module coupling is small but nonzero, the energy decomposition
  has bounded error. We define cross-module coupling and bound it.
-/

/-- Cross-module coupling: total coupling energy between memories in different modules -/
def crossModuleCoupling {d N K : ℕ} (wp : WeightParams)
    (mp : ModulePartition N K) (memories : Fin N → Fin d → ℝ) : ℝ :=
  ∑ i : Fin N, ∑ j : Fin N,
    if mp.assign i ≠ mp.assign j ∧ j.val < i.val
    then couplingEnergy wp (memories i) (memories j) else 0

/-- Cross-module coupling vanishes under exact inter-module neutrality -/
theorem cross_coupling_zero_of_neutral {d N K : ℕ}
    (wp : WeightParams) (mp : ModulePartition N K)
    (memories : Fin N → Fin d → ℝ)
    (h_neutral : interModuleNeutral mp wp memories) :
    crossModuleCoupling wp mp memories = 0 := by
  unfold crossModuleCoupling
  apply Finset.sum_eq_zero; intro i _
  apply Finset.sum_eq_zero; intro j _
  split_ifs with h
  · exact couplingEnergy_zero_of_weight_zero wp _ _ (h_neutral i j h.1)
  · rfl

/-- PHASE 5.5 THEOREM: Approximate block-diagonality bound.
    When each cross-module coupling term is bounded by ε,
    the total cross-module coupling is bounded by N² · ε.

    This quantifies how "close" the system is to perfectly modular:
    small cross-module coupling → small decomposition error. -/
theorem approx_block_diagonal_bound {d N K : ℕ} (wp : WeightParams)
    (mp : ModulePartition N K) (memories : Fin N → Fin d → ℝ)
    (ε : ℝ) (hε : 0 ≤ ε)
    (h_approx : ∀ i j : Fin N, mp.assign i ≠ mp.assign j →
      |couplingEnergy wp (memories i) (memories j)| ≤ ε) :
    |crossModuleCoupling wp mp memories| ≤ ↑N * ↑N * ε := by
  unfold crossModuleCoupling
  calc |∑ i : Fin N, ∑ j : Fin N, _|
      ≤ ∑ i : Fin N, |∑ j : Fin N, _| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i : Fin N, ∑ j : Fin N, |if mp.assign i ≠ mp.assign j ∧ j.val < i.val
          then couplingEnergy wp (memories i) (memories j) else 0| := by
        gcongr; exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin N, ∑ _j : Fin N, ε := by
        gcongr with i _ j _
        split_ifs with h
        · exact h_approx i j h.1
        · simp [hε]
    _ = ↑N * ↑N * ε := by simp [Finset.sum_const]; ring

end HermesNLCDM
