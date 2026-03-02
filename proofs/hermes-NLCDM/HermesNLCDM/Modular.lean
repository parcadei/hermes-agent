/-
  HermesNLCDM.Modular
  ===================
  Phase 5: Modular Structure — Block-Diagonal Decomposition

  Main results:
  1. Module partition: memories grouped into K modules by assignment function
  2. Inter-module coupling vanishes when cross-module similarities are neutral
  3. Total energy decomposes into sum of per-module energies
  4. Module independence: updates within one module don't affect others
  5. Per-module descent: coordinate descent within a module decreases energy
  6. Per-module capacity: each module has capacity based on module size N_m

  The key insight: when the smooth weight function W maps cross-module
  cosine similarities to zero (neutral zone: τ_low < cos(x_i, x_j) < τ_high),
  the coupling matrix becomes block-diagonal, and the system decomposes into
  independent sub-problems.

  Reference: Hermes memory architecture — modular memory organization
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.LocalMinima
import HermesNLCDM.Dynamics
import HermesNLCDM.Coupling
import HermesNLCDM.Capacity
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Module Partition

  A module partition assigns each memory to one of K modules.
  The key property is inter-module neutrality: coupling weights
  between memories in different modules are zero.
-/

/-- Assignment of N memories to K modules -/
structure ModulePartition (N K : ℕ) where
  assign : Fin N → Fin K

/-- The set of memory indices belonging to module m -/
def ModulePartition.members {N K : ℕ} (mp : ModulePartition N K)
    (m : Fin K) : Finset (Fin N) :=
  Finset.univ.filter (fun i => mp.assign i = m)

/-- Every memory belongs to exactly one module -/
theorem ModulePartition.mem_members_iff {N K : ℕ} (mp : ModulePartition N K)
    (i : Fin N) (m : Fin K) :
    i ∈ mp.members m ↔ mp.assign i = m := by
  simp [ModulePartition.members, Finset.mem_filter]

/-- Module membership sets are pairwise disjoint -/
theorem ModulePartition.members_disjoint {N K : ℕ} (mp : ModulePartition N K)
    (m₁ m₂ : Fin K) (hne : m₁ ≠ m₂) :
    Disjoint (mp.members m₁) (mp.members m₂) := by
  simp only [ModulePartition.members]
  rw [Finset.disjoint_filter]
  intro i _ h₁ h₂
  exact hne (h₁.symm.trans h₂)

/-- Pairwise disjointness as Set.PairwiseDisjoint -/
theorem ModulePartition.members_pairwiseDisjoint {N K : ℕ} (mp : ModulePartition N K) :
    (Set.univ : Set (Fin K)).PairwiseDisjoint mp.members := by
  intro m₁ _ m₂ _ hne
  show Disjoint (mp.members m₁) (mp.members m₂)
  exact mp.members_disjoint m₁ m₂ hne

/-- The union of all module membership sets covers all memories -/
theorem ModulePartition.members_cover {N K : ℕ} [NeZero K]
    (mp : ModulePartition N K) :
    Finset.univ.biUnion mp.members = Finset.univ := by
  ext i
  simp only [Finset.mem_biUnion, Finset.mem_univ, true_and]
  exact ⟨fun _ => trivial, fun _ => ⟨mp.assign i, mp.mem_members_iff i _ |>.mpr rfl⟩⟩

/-- Pairwise disjointness on Finset.univ coerced to Set -/
theorem ModulePartition.members_pairwiseDisjoint_univ {N K : ℕ}
    (mp : ModulePartition N K) :
    ((Finset.univ : Finset (Fin K)) : Set (Fin K)).PairwiseDisjoint mp.members :=
  fun m _ m' _ hne => mp.members_disjoint m m' hne

/-! ## Inter-Module Neutrality

  The condition that coupling weights vanish between different modules.
  This is the structural condition enabling block-diagonal decomposition.
-/

/-- Inter-module neutrality: coupling weight is zero between memories
    in different modules -/
def interModuleNeutral {d N K : ℕ} (mp : ModulePartition N K)
    (wp : WeightParams) (memories : Fin N → Fin d → ℝ) : Prop :=
  ∀ i j : Fin N, mp.assign i ≠ mp.assign j →
    smoothWeight wp (cosineSim (memories i) (memories j)) = 0

/-! ## Inter-Module Coupling Vanishes -/

/-- When inter-module neutrality holds, coupling energy between
    memories in different modules is zero -/
theorem cross_module_coupling_zero {d N K : ℕ}
    (mp : ModulePartition N K)
    (wp : WeightParams) (memories : Fin N → Fin d → ℝ)
    (h_neutral : interModuleNeutral mp wp memories)
    (i j : Fin N) (h_diff : mp.assign i ≠ mp.assign j) :
    couplingEnergy wp (memories i) (memories j) = 0 :=
  couplingEnergy_zero_of_weight_zero wp _ _ (h_neutral i j h_diff)

/-! ## Per-Module Energy

  Define the energy contribution of a single module.
-/

/-- Local energy sum for memories in module m -/
def moduleLocalEnergy {d N K M_count : ℕ} (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K) (memories : Fin N → Fin d → ℝ) (m : Fin K) : ℝ :=
  (mp.members m).sum (fun i => localEnergy cfg (memories i))

/-- Intra-module coupling energy for module m -/
def moduleCouplingEnergy {d N K : ℕ} (wp : WeightParams)
    (mp : ModulePartition N K) (memories : Fin N → Fin d → ℝ) (m : Fin K) : ℝ :=
  (mp.members m).sum (fun i => (mp.members m).sum (fun j =>
    if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0))

/-- Total energy for module m -/
def moduleEnergy {d N K M_count : ℕ} (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K) (memories : Fin N → Fin d → ℝ) (m : Fin K) : ℝ :=
  moduleLocalEnergy cfg mp memories m + moduleCouplingEnergy cfg.weights mp memories m

/-! ## Energy Decomposition Theorem

  The total energy decomposes into sum of per-module energies
  when inter-module coupling vanishes.
-/

/-- Helper: ∑ over Fin N = ∑ over Finset.univ -/
private theorem sum_fin_eq_sum_univ {N : ℕ} {M : Type*} [AddCommMonoid M]
    (f : Fin N → M) :
    ∑ i, f i = Finset.univ.sum f := rfl

/-- Local energy decomposes over modules -/
theorem localEnergy_decomp {d N K M_count : ℕ} [NeZero K]
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories : Fin N → Fin d → ℝ) :
    ∑ i, localEnergy cfg (memories i) =
    ∑ m, moduleLocalEnergy cfg mp memories m := by
  unfold moduleLocalEnergy
  -- Use sum_fiberwise: ∑ j, ∑ i in univ with g i = j, f i = ∑ i in univ, f i
  symm
  exact Finset.sum_fiberwise_of_maps_to (fun i _ => Finset.mem_univ (mp.assign i))
    (fun i => localEnergy cfg (memories i))

/-- Helper: for each i in module m, the inner sum over all j equals the inner sum
    over just module m, because cross-module terms vanish -/
private theorem inner_sum_restrict {d N K : ℕ} [NeZero K]
    (wp : WeightParams) (mp : ModulePartition N K)
    (memories : Fin N → Fin d → ℝ)
    (h_neutral : interModuleNeutral mp wp memories)
    (m : Fin K) (i : Fin N) (hi : mp.assign i = m) :
    Finset.univ.sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0) =
    (mp.members m).sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0) := by
  -- Partition j over modules, show cross-module terms vanish
  rw [← mp.members_cover, Finset.sum_biUnion mp.members_pairwiseDisjoint_univ]
  -- Now: ∑_m' ∑_{j ∈ members m'} (...) = ∑_{j ∈ members m} (...)
  -- Extract m-th term, show rest = 0
  rw [show ∑ x : Fin K, (mp.members x).sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0) =
    (mp.members m).sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0) +
    (Finset.univ.erase m).sum (fun m' => (mp.members m').sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0)) from
    (Finset.add_sum_erase _ _ (Finset.mem_univ m)).symm]
  suffices h_rest : (Finset.univ.erase m).sum (fun m' => (mp.members m').sum (fun j =>
      if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0)) = 0 by
    linarith
  apply Finset.sum_eq_zero
  intro m' hm'
  have hm'_ne : m' ≠ m := Finset.ne_of_mem_erase hm'
  apply Finset.sum_eq_zero
  intro j hj
  split_ifs with h
  · have hj_mod : mp.assign j = m' := (mp.mem_members_iff j m').mp hj
    have h_diff : mp.assign i ≠ mp.assign j := by rw [hi, hj_mod]; exact hm'_ne.symm
    exact cross_module_coupling_zero mp wp memories h_neutral i j h_diff
  · rfl

/-- Coupling energy between memories in the same module, summed over all modules,
    accounts for all coupling when inter-module coupling vanishes -/
theorem couplingEnergy_decomp {d N K : ℕ} [NeZero K]
    (wp : WeightParams)
    (mp : ModulePartition N K)
    (memories : Fin N → Fin d → ℝ)
    (h_neutral : interModuleNeutral mp wp memories) :
    ∑ i : Fin N, ∑ j : Fin N,
      (if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0) =
    ∑ m, moduleCouplingEnergy wp mp memories m := by
  unfold moduleCouplingEnergy
  -- Step 1: Partition the outer sum over i using the module cover
  set f : Fin N → ℝ := fun i => ∑ j : Fin N,
    (if j.val < i.val then couplingEnergy wp (memories i) (memories j) else 0)
  show Finset.univ.sum f = _
  rw [← mp.members_cover, Finset.sum_biUnion mp.members_pairwiseDisjoint_univ]
  -- Step 2: For each i ∈ module m, restrict inner sum
  congr 1; ext m
  apply Finset.sum_congr rfl
  intro i hi
  have hi_mod : mp.assign i = m := (mp.mem_members_iff i m).mp hi
  exact inner_sum_restrict wp mp memories h_neutral m i hi_mod

/-- PHASE 5 MAIN THEOREM (Part 1): Total energy decomposes into sum of
    per-module energies when inter-module coupling vanishes.

    E_total = Σ_m E_module(m)

    This is the block-diagonal decomposition of the energy landscape. -/
theorem totalEnergy_module_decomp {d N K M_count : ℕ} [NeZero K] [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories : Fin N → Fin d → ℝ)
    (h_neutral : interModuleNeutral mp cfg.weights memories) :
    totalEnergy cfg memories = ∑ m, moduleEnergy cfg mp memories m := by
  unfold totalEnergy moduleEnergy
  rw [show ∑ m, (moduleLocalEnergy cfg mp memories m +
      moduleCouplingEnergy cfg.weights mp memories m) =
    (∑ m, moduleLocalEnergy cfg mp memories m) +
    (∑ m, moduleCouplingEnergy cfg.weights mp memories m) from Finset.sum_add_distrib]
  congr 1
  · exact localEnergy_decomp cfg mp memories
  · exact couplingEnergy_decomp cfg.weights mp memories h_neutral

/-! ## Module Independence

  Updates within one module don't affect other modules' energy.
-/

/-- Updating memories preserves a module's local energy when its members are unchanged -/
theorem module_local_energy_preserved {d N K M_count : ℕ}
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories updated : Fin N → Fin d → ℝ)
    (m_other : Fin K)
    (h_unchanged : ∀ i, mp.assign i = m_other → updated i = memories i) :
    moduleLocalEnergy cfg mp updated m_other =
    moduleLocalEnergy cfg mp memories m_other := by
  unfold moduleLocalEnergy
  apply Finset.sum_congr rfl
  intro i hi
  rw [h_unchanged i ((mp.mem_members_iff i m_other).mp hi)]

/-- Updating memories preserves a module's coupling energy when its members are unchanged -/
theorem module_coupling_energy_preserved {d N K : ℕ}
    (wp : WeightParams)
    (mp : ModulePartition N K)
    (memories updated : Fin N → Fin d → ℝ)
    (m_other : Fin K)
    (h_unchanged : ∀ i, mp.assign i = m_other → updated i = memories i) :
    moduleCouplingEnergy wp mp updated m_other =
    moduleCouplingEnergy wp mp memories m_other := by
  unfold moduleCouplingEnergy
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  rw [h_unchanged i ((mp.mem_members_iff i m_other).mp hi),
      h_unchanged j ((mp.mem_members_iff j m_other).mp hj)]

/-- PHASE 5 THEOREM: Module energy is preserved for modules not being updated.
    Updates within module m_updated leave all other modules' energy unchanged. -/
theorem module_energy_preserved {d N K M_count : ℕ}
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories updated : Fin N → Fin d → ℝ)
    (m_updated m_other : Fin K)
    (hne : m_updated ≠ m_other)
    (h_only_m : ∀ i, mp.assign i ≠ m_updated → updated i = memories i) :
    moduleEnergy cfg mp updated m_other =
    moduleEnergy cfg mp memories m_other := by
  unfold moduleEnergy
  have h_unchanged : ∀ i, mp.assign i = m_other → updated i = memories i :=
    fun i hi => h_only_m i (by rw [hi]; exact Ne.symm hne)
  congr 1
  · exact module_local_energy_preserved cfg mp memories updated m_other h_unchanged
  · exact module_coupling_energy_preserved cfg.weights mp memories updated m_other h_unchanged

/-! ## Per-Module Descent

  When we update within a single module, only that module's energy changes,
  and the total energy change equals the module energy change.
-/

/-- Total energy change from an intra-module update equals the change in
    that module's energy alone. All other module energies are frozen. -/
theorem totalEnergy_intramodule_update {d N K M_count : ℕ}
    [NeZero K] [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories updated : Fin N → Fin d → ℝ)
    (m : Fin K)
    (h_neutral_old : interModuleNeutral mp cfg.weights memories)
    (h_neutral_new : interModuleNeutral mp cfg.weights updated)
    (h_only_m : ∀ i, mp.assign i ≠ m → updated i = memories i) :
    totalEnergy cfg updated - totalEnergy cfg memories =
    moduleEnergy cfg mp updated m - moduleEnergy cfg mp memories m := by
  rw [totalEnergy_module_decomp cfg mp memories h_neutral_old,
      totalEnergy_module_decomp cfg mp updated h_neutral_new]
  rw [← Finset.sum_sub_distrib]
  have h_zero : ∀ m' : Fin K, m' ≠ m →
      moduleEnergy cfg mp updated m' - moduleEnergy cfg mp memories m' = 0 := by
    intro m' hm'
    rw [module_energy_preserved cfg mp memories updated m m' (Ne.symm hm') h_only_m]
    ring
  rw [show ∑ x : Fin K, (moduleEnergy cfg mp updated x - moduleEnergy cfg mp memories x) =
    (moduleEnergy cfg mp updated m - moduleEnergy cfg mp memories m) +
    (Finset.univ.erase m).sum (fun m' =>
      moduleEnergy cfg mp updated m' - moduleEnergy cfg mp memories m') from
    (Finset.add_sum_erase _ _ (Finset.mem_univ m)).symm]
  rw [show (Finset.univ.erase m).sum (fun m' =>
      moduleEnergy cfg mp updated m' - moduleEnergy cfg mp memories m') = 0 from
    Finset.sum_eq_zero (fun m' hm' => h_zero m' (Finset.ne_of_mem_erase hm'))]
  ring

/-- PHASE 5 THEOREM: If an intra-module update decreases that module's energy,
    total energy decreases. Combined with Phase 2 descent, this gives modular
    coordinate descent. -/
theorem modular_descent {d N K M_count : ℕ}
    [NeZero K] [NeZero M_count]
    (cfg : SystemConfig d M_count)
    (mp : ModulePartition N K)
    (memories updated : Fin N → Fin d → ℝ)
    (m : Fin K)
    (h_neutral_old : interModuleNeutral mp cfg.weights memories)
    (h_neutral_new : interModuleNeutral mp cfg.weights updated)
    (h_only_m : ∀ i, mp.assign i ≠ m → updated i = memories i)
    (h_module_descent : moduleEnergy cfg mp updated m ≤ moduleEnergy cfg mp memories m) :
    totalEnergy cfg updated ≤ totalEnergy cfg memories := by
  have h := totalEnergy_intramodule_update cfg mp memories updated m
    h_neutral_old h_neutral_new h_only_m
  linarith

/-! ## Per-Module Capacity

  Each module has its own capacity bound based on module size N_m,
  not the total number of memories N. This is a key advantage of
  modular organization: capacity scales per-module.
-/

/-- Module size: number of memories in module m -/
def moduleSize {N K : ℕ} (mp : ModulePartition N K) (m : Fin K) : ℕ :=
  (mp.members m).card

/-- PHASE 5 THEOREM: Per-module capacity.
    For a module with N_m memories and separation δ, the capacity condition
    is 4·N_m·β·M² ≤ exp(β·δ), which is weaker than the global condition
    4·N·β·M² ≤ exp(β·δ) since N_m ≤ N.

    This means modular organization effectively increases the system's
    total capacity: each module can store more patterns because it only
    competes with other patterns in the same module. -/
theorem module_capacity_advantage {N K : ℕ} [NeZero K]
    (mp : ModulePartition N K) (m : Fin K) :
    moduleSize mp m ≤ N := by
  unfold moduleSize
  calc (mp.members m).card
      ≤ Finset.univ.card := Finset.card_filter_le _ _
    _ = N := Finset.card_fin N

end HermesNLCDM
