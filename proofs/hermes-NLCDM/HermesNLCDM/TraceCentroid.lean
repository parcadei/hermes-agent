/-
  HermesNLCDM.TraceCentroid
  =========================
  Phase 7: Reasoning Trace Centroid Energy Wells

  Main results:
  1. sqNorm_eq_sum: bridge between sqNorm and localEnergy's raw sum
  2. lse_mono_alignment: lse is monotone in pattern-query alignment
  3. trace_centroid_energy_well_strong: MAIN THEOREM — when a trace centroid has
     higher alignment with its nearest stored pattern than a random point does,
     by a margin exceeding β⁻¹ log N, then E(centroid) < E(random point).

  Empirical validation (from test_reasoning_trace_centroids.py):
    E(trace centroid) = -0.716, E(random interpolation) = -0.319, Δ = -0.397

  Mathematical framework:
  A "reasoning trace" is a unit vector r = α·x_s + (1-α)·x_t + noise, where x_s
  is a source-domain pattern and x_t is a target-domain pattern. The centroid
  c = normalize(mean(r_1, ..., r_K)) has high alignment with all K traces.

  When traces r_1,...,r_K are stored as additional patterns in the system config,
  the log-sum-exp at c is large (because c has high dot product with each r_k),
  making localEnergy(c) = -lse(β, X^T c) + ½‖c‖² low.

  Reference: Hermes memory architecture — OOD reasoning via memory consolidation
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.SpuriousStates
import HermesNLCDM.EnergyGap

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Bridge: sqNorm and localEnergy

  localEnergy uses `∑ i, ξ i ^ 2` directly, while other theorems use
  `sqNorm ξ = ∑ i, ξ i ^ 2`. We establish the identity so linarith
  can bridge between the two representations.
-/

/-- sqNorm ξ equals the raw sum of squares used in localEnergy. -/
theorem sqNorm_eq_sum {d : ℕ} (ξ : Fin d → ℝ) :
    sqNorm ξ = ∑ i, ξ i ^ 2 := rfl

/-- localEnergy expressed in terms of sqNorm. -/
theorem localEnergy_eq {d N : ℕ} (cfg : SystemConfig d N) (ξ : Fin d → ℝ) :
    localEnergy cfg ξ =
      -logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) +
        (1/2) * sqNorm ξ := by
  unfold localEnergy sqNorm; rfl

/-! ## Energy Comparison: Centroid vs Random Point

  The local energy E(ξ) = -lse(β, X^T ξ) + ½‖ξ‖².
  Lower energy = deeper well = better attractor.

  Key fact: lse is monotone in the input similarities. If ξ has higher dot
  products with stored patterns than ζ, then lse(β, X^T ξ) ≥ lse(β, X^T ζ)
  in the relevant components.

  We prove: if ξ has at least one stored pattern μ with ⟨x_μ, ξ⟩ > ⟨x_μ, ζ⟩
  by a margin exceeding β⁻¹ log N, AND ‖ξ‖² ≤ ‖ζ‖², then E(ξ) < E(ζ).
-/

/-- ENERGY MONOTONICITY IN ALIGNMENT:
    If ξ has higher similarity to every stored pattern than ζ does,
    then lse(β, X^T ξ) ≥ lse(β, X^T ζ).

    Proof: lse is monotone in each coordinate (since exp is monotone
    and log is monotone). -/
theorem lse_mono_alignment {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ ζ : Fin d → ℝ)
    (halign : ∀ μ, ∑ i, cfg.patterns μ i * ξ i ≥ ∑ i, cfg.patterns μ i * ζ i) :
    logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ξ i) ≥
    logSumExp cfg.β (fun μ => ∑ i, cfg.patterns μ i * ζ i) := by
  unfold logSumExp
  have hβ_inv_pos : 0 < cfg.β⁻¹ := inv_pos.mpr cfg.β_pos
  apply mul_le_mul_of_nonneg_left _ hβ_inv_pos.le
  apply Real.log_le_log
  · positivity
  · apply Finset.sum_le_sum
    intro μ _
    exact exp_le_exp.mpr (mul_le_mul_of_nonneg_left (halign μ) cfg.β_pos.le)

/-- TRACE CENTROID ENERGY WELL (main theorem):
    When ξ has higher alignment with pattern j than ζ does, AND j is the
    max-aligned pattern for ζ, AND the alignment gap exceeds β⁻¹ log N,
    AND ‖ξ‖² ≤ ‖ζ‖², then E(ξ) < E(ζ).

    The condition "j maximizes alignment for ζ" is natural when ζ is a random
    interpolation point and j is its nearest stored pattern.

    Proof outline:
      E(ξ) = -lse(X^T ξ) + ½‖ξ‖²
           ≤ -(X_j · ξ) + ½‖ξ‖²           (lse ≥ j-th entry)
           ≤ -(X_j · ζ + δ) + ½‖ζ‖²       (alignment gap + norm bound)
      E(ζ) = -lse(X^T ζ) + ½‖ζ‖²
           ≥ -(X_j · ζ + β⁻¹ log N) + ½‖ζ‖²  (lse ≤ max + β⁻¹ log N, j is max)
      E(ξ) - E(ζ) ≤ (β⁻¹ log N - δ) + ½(‖ξ‖² - ‖ζ‖²) < 0

    Empirical context: trace centroid achieves Δ = -0.397 energy gap. -/
theorem trace_centroid_energy_well_strong {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ ζ : Fin d → ℝ)
    (j : Fin N) {δ : ℝ}
    (halign_gap : ∑ i, cfg.patterns j i * ξ i ≥
                  ∑ i, cfg.patterns j i * ζ i + δ)
    (hδ : δ > cfg.β⁻¹ * Real.log N)
    (hnorm : sqNorm ξ ≤ sqNorm ζ)
    (hmax : ∀ μ, ∑ i, cfg.patterns μ i * ζ i ≤ ∑ i, cfg.patterns j i * ζ i) :
    localEnergy cfg ξ < localEnergy cfg ζ := by
  -- Rewrite in terms of sqNorm for clean arithmetic
  rw [localEnergy_eq, localEnergy_eq]
  set sim_ξ := fun μ => ∑ i, cfg.patterns μ i * ξ i
  set sim_ζ := fun μ => ∑ i, cfg.patterns μ i * ζ i
  -- (1) lse(sim_ξ) ≥ sim_ξ(j) ≥ sim_ζ(j) + δ
  have h1 : sim_ξ j ≤ logSumExp cfg.β sim_ξ :=
    lse_ge_max cfg.β_pos sim_ξ j
  -- (2) lse(sim_ζ) ≤ sup'(sim_ζ) + β⁻¹ log N ≤ sim_ζ(j) + β⁻¹ log N
  have h2 : logSumExp cfg.β sim_ζ ≤
      Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ sim_ζ + cfg.β⁻¹ * Real.log ↑N :=
    lse_le_max_plus cfg.β_pos sim_ζ
  have hsup_le : Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ sim_ζ ≤ sim_ζ j :=
    Finset.sup'_le _ _ (fun μ _ => hmax μ)
  have h3 : logSumExp cfg.β sim_ζ ≤ sim_ζ j + cfg.β⁻¹ * Real.log ↑N := by linarith
  -- Goal: -lse(sim_ξ) + ½ ‖ξ‖² < -lse(sim_ζ) + ½ ‖ζ‖²
  -- From (1): -lse(sim_ξ) ≤ -sim_ξ(j) ≤ -(sim_ζ(j) + δ)
  -- From (3): -lse(sim_ζ) ≥ -(sim_ζ(j) + β⁻¹ log N)
  -- LHS ≤ -(sim_ζ(j) + δ) + ½ ‖ξ‖²
  -- RHS ≥ -(sim_ζ(j) + β⁻¹ log N) + ½ ‖ζ‖²
  -- LHS - RHS ≤ (β⁻¹ log N - δ) + ½(‖ξ‖² - ‖ζ‖²) < 0 + 0 = 0
  linarith

/-- TRACE CENTROID ENERGY WELL (unit vector version):
    Specialization for unit vectors (‖ξ‖² = ‖ζ‖² = 1), which is the
    natural case for Hopfield patterns on the unit sphere.
    The norm condition becomes trivial. -/
theorem trace_centroid_energy_well_unit {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ ζ : Fin d → ℝ)
    (j : Fin N) {δ : ℝ}
    (hξ_unit : sqNorm ξ = 1) (hζ_unit : sqNorm ζ = 1)
    (halign_gap : ∑ i, cfg.patterns j i * ξ i ≥
                  ∑ i, cfg.patterns j i * ζ i + δ)
    (hδ : δ > cfg.β⁻¹ * Real.log N)
    (hmax : ∀ μ, ∑ i, cfg.patterns μ i * ζ i ≤ ∑ i, cfg.patterns j i * ζ i) :
    localEnergy cfg ξ < localEnergy cfg ζ :=
  trace_centroid_energy_well_strong cfg ξ ζ j halign_gap hδ (by rw [hξ_unit, hζ_unit]) hmax

/-! ## Centroid Alignment Bound

  The dot product ⟨mean(v_1,...,v_K), v_j⟩ = (1/K) Σ_k ⟨v_k, v_j⟩.
  When all pairwise similarities are ≥ σ, this is ≥ (1/K)(1 + (K-1)σ).
  For unit vectors with pairwise sim ≥ 0.9 and K=8:
    ⟨centroid, v_j⟩ ≥ (1/8)(1 + 7·0.9) = 0.9125 (before normalization)
    After normalization: even higher.
-/

/-- The dot product of a uniform mean with a constituent equals the average
    of all pairwise dots involving that constituent. -/
theorem dot_mean_eq_avg {d K : ℕ} [NeZero K]
    (v : Fin K → Fin d → ℝ) (j : Fin K) :
    ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i =
      (1 / (K : ℝ)) * ∑ k, ∑ i, v k i * v j i := by
  -- Factor out 1/K from each summand
  simp_rw [show ∀ i, ((1 : ℝ) / ↑K * ∑ k, v k i) * v j i =
    (1 / ↑K) * ((∑ k, v k i) * v j i) from fun i => by ring]
  rw [← Finset.mul_sum]
  congr 1
  -- Distribute multiplication over inner sum, then swap
  simp_rw [Finset.sum_mul]
  exact Finset.sum_comm

/-- If all pairwise dots are at least σ, the mean-constituent dot is at least σ.
    (The self-term ⟨v_j, v_j⟩ = ‖v_j‖² ≥ σ is included; for unit vectors
    the self-term is 1 ≥ σ, making the bound conservative.) -/
theorem dot_mean_ge_of_pairwise {d K : ℕ} [NeZero K]
    (v : Fin K → Fin d → ℝ) (j : Fin K) {σ : ℝ}
    (hpair : ∀ k, ∑ i, v k i * v j i ≥ σ) :
    ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i ≥ σ := by
  rw [dot_mean_eq_avg]
  have hK_pos : (0 : ℝ) < K := Nat.cast_pos.mpr (NeZero.pos K)
  have hK_inv_pos : (0 : ℝ) < 1 / K := by positivity
  have hsum_ge : ∑ k, ∑ i, v k i * v j i ≥ ↑K * σ := by
    calc ∑ k, ∑ i, v k i * v j i
        ≥ ∑ _k : Fin K, σ :=
          Finset.sum_le_sum (fun k _ => hpair k)
      _ = ↑(Fintype.card (Fin K)) * σ := by
          rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
      _ = ↑K * σ := by rw [Fintype.card_fin]
  calc (1 / ↑K) * ∑ k, ∑ i, v k i * v j i
      ≥ (1 / ↑K) * (↑K * σ) :=
        mul_le_mul_of_nonneg_left hsum_ge hK_inv_pos.le
    _ = σ := by field_simp

/-! ## Energy Monotonicity for Fixed Points

  If a fixed point ξ of the extended system (content + traces) has
  softmax concentration p_μ ≥ 1-ε on a trace centroid μ, then it
  sits at a deep energy well — directly from the existing Phase 5.9b
  energy_gap_concentrated theorem.

  This connects the reasoning trace story to the existing proof:
  dream consolidation merges similar traces into centroids, the centroids
  become stored patterns, and the energy gap theorem guarantees they
  are deeper wells than mixture states.
-/

/-- Corollary: trace centroids that become concentrated fixed points
    in the extended system have lower energy than mixture states,
    by direct application of Phase 5.9b.

    This is the connection between:
    - Phase 7 (this file): traces form centroids with high alignment
    - Phase 5.9b (EnergyGap.lean): concentrated fps have lower energy

    After dream consolidation, the trace centroid IS a stored pattern,
    and any query near it converges to a concentrated fixed point. -/
theorem trace_centroid_concentrated_gap {d N : ℕ} [NeZero N]
    (cfg : SystemConfig d N) (ξ_c ξ_m : Fin d → ℝ)
    (hfp_c : isFixedPoint cfg ξ_c) (hfp_m : isFixedPoint cfg ξ_m)
    (μ : Fin N) {M ε : ℝ} (hM : 0 ≤ M) (hε_nn : 0 ≤ ε)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hconc : softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * ξ_c i) μ ≥ 1 - ε)
    (hgap : sqNorm (cfg.patterns μ) - 4 * M ^ 2 * Real.sqrt ε >
        sqNorm ξ_m + 2 * cfg.β⁻¹ * Real.log N) :
    localEnergy cfg ξ_c < localEnergy cfg ξ_m :=
  energy_gap_concentrated cfg ξ_c ξ_m hfp_c hfp_m μ hM hε_nn hNorm hconc hgap

/-! ## Alignment Monotonicity under Centroid Construction

  When K similar traces are replaced by their centroid in the pattern store,
  the centroid's self-alignment is at least the minimum pairwise similarity.
  This guarantees lse at the centroid stays high after merge.
-/

/-- For unit vectors with minimum pairwise similarity σ ∈ [0,1],
    the squared norm of their mean is at least σ.

    Proof: ‖mean‖² = ⟨mean, mean⟩ = (1/K) Σ_j ⟨mean, v_j⟩ (bilinearity),
    and each ⟨mean, v_j⟩ ≥ σ by dot_mean_ge_of_pairwise.
    So ‖mean‖² ≥ (1/K) · K · σ = σ.

    This means the centroid "remembers" the similarity structure. -/
theorem sqNorm_mean_ge_of_pairwise {d K : ℕ} [NeZero K]
    (v : Fin K → Fin d → ℝ) {σ : ℝ}
    (_hσ_nn : 0 ≤ σ) (hσ_le : σ ≤ 1)
    (hunit : ∀ k, sqNorm (v k) = 1)
    (hpair : ∀ j k, j ≠ k → ∑ i, v j i * v k i ≥ σ) :
    sqNorm (fun i => (1 / (K : ℝ)) * ∑ k, v k i) ≥ σ := by
  -- Step 1: Each ⟨mean, v_j⟩ ≥ σ using dot_mean_ge_of_pairwise
  have hmean_dot : ∀ j, ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i ≥ σ := by
    intro j
    apply dot_mean_ge_of_pairwise
    intro k
    by_cases hjk : k = j
    · -- Self-term: ⟨v_j, v_j⟩ = sqNorm(v_j) = 1 ≥ σ
      rw [hjk]
      have huj := hunit j
      unfold sqNorm at huj
      have : ∑ i, v j i * v j i = ∑ i, v j i ^ 2 := by
        congr 1; ext i; ring
      linarith [this]
    · exact hpair k j hjk
  -- Step 2: Use dot_mean_eq_avg to rewrite sqNorm as (1/K) · Σ_j ⟨mean, v_j⟩
  -- Key identity: sqNorm(mean) = ⟨mean, mean⟩ = (1/K) Σ_j ⟨mean, v_j⟩
  -- This holds because mean = (1/K) Σ v_k, so the second factor expands.
  -- We prove this via a sufficiency argument: sqNorm(mean) ≥ (1/K) Σ_j σ = σ.
  --
  -- Direct approach: rewrite sqNorm as a double sum, then bound.
  have hK_pos : (0 : ℝ) < K := Nat.cast_pos.mpr (NeZero.pos K)
  -- sqNorm(mean) = Σ_j ⟨mean, (1/K)·v_j⟩ = (1/K) Σ_j ⟨mean, v_j⟩
  have hexpand : sqNorm (fun i => (1 / (K : ℝ)) * ∑ k, v k i) =
      (1 / (K : ℝ)) * ∑ j, ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i := by
    unfold sqNorm
    -- LHS: Σ_i ((1/K) Σ_k v k i)^2
    -- = Σ_i (1/K)(Σ_k v k i) · (1/K)(Σ_j v j i)
    -- = (1/K) · Σ_i (1/K)(Σ_k v k i) · (Σ_j v j i)
    -- = (1/K) · Σ_j Σ_i (1/K)(Σ_k v k i) · v j i   [distribute, swap]
    simp_rw [show ∀ i, ((1 : ℝ) / ↑K * ∑ k, v k i) ^ 2 =
      (1 / ↑K) * (((1 / ↑K) * ∑ k, v k i) * ∑ j, v j i) from fun i => by ring]
    rw [← Finset.mul_sum]
    congr 1
    simp_rw [Finset.mul_sum]
    exact Finset.sum_comm
  rw [hexpand]
  -- Goal: (1/K) · Σ_j ⟨mean, v_j⟩ ≥ σ, and each ⟨mean, v_j⟩ ≥ σ
  have hK_inv_pos : (0 : ℝ) < 1 / K := by positivity
  have hsum_ge : ∑ j, ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i ≥ ↑K * σ := by
    calc ∑ j, ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i
        ≥ ∑ _j : Fin K, σ :=
          Finset.sum_le_sum (fun j _ => hmean_dot j)
      _ = ↑(Fintype.card (Fin K)) * σ := by
          rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
      _ = ↑K * σ := by rw [Fintype.card_fin]
  calc (1 / ↑K) * ∑ j, ∑ i, ((1 / (K : ℝ)) * ∑ k, v k i) * v j i
      ≥ (1 / ↑K) * (↑K * σ) :=
        mul_le_mul_of_nonneg_left hsum_ge hK_inv_pos.le
    _ = σ := by field_simp

end HermesNLCDM
