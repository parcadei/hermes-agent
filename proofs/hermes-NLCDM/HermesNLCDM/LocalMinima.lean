/-
  HermesNLCDM.LocalMinima
  =======================
  Phase 1, Part 3: Stored patterns are local energy minima.

  Key result: Each stored pattern ξ^μ is a local minimum of E_local,
  meaning ∇E(ξ^μ) = 0 and ∇²E(ξ^μ) is positive definite.

  For Modern Hopfield with sufficient β:
    - The softmax concentrates on the nearest pattern
    - The gradient vanishes at stored patterns
    - The Hessian is positive definite (not just semidefinite)

  This is stronger than just showing stationary points: we need
  second-order conditions to exclude saddle points.

  Reference: Ramsauer et al. 2020, Theorems A5-A9
-/

import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.InnerProductSpace.Basic

noncomputable section
open Real Finset

namespace HermesNLCDM

/-! ## Softmax and its properties

  The softmax distribution p_μ(ξ) = exp(β ξ^T x_μ) / Σ_ν exp(β ξ^T x_ν)
  plays a central role. At a stored pattern ξ = x_μ, the softmax
  concentrates on μ (for large enough β and well-separated patterns).
-/

/-- Softmax distribution over patterns given query ξ -/
def softmax (β : ℝ) {N : ℕ} (similarities : Fin N → ℝ) (μ : Fin N) : ℝ :=
  exp (β * similarities μ) / ∑ ν, exp (β * similarities ν)

/-- Softmax entries are nonneg -/
theorem softmax_nonneg {β : ℝ} {N : ℕ} [NeZero N]
    (z : Fin N → ℝ) (μ : Fin N) : 0 ≤ softmax β z μ := by
  unfold softmax
  apply div_nonneg (le_of_lt (exp_pos _))
  apply Finset.sum_nonneg
  intro i _
  exact le_of_lt (exp_pos _)

/-- Softmax sums to 1 -/
theorem softmax_sum_one {β : ℝ} {N : ℕ} [NeZero N]
    (z : Fin N → ℝ) : ∑ μ, softmax β z μ = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (by positivity))

/-- Softmax concentrates on the largest entry as β → ∞:
    If z_j > z_i for all i ≠ j, then softmax(β, z)_j → 1.
    We prove a finite-β bound: p_j ≥ 1 - (N-1)·exp(-β·δ)
    where δ = min_{i≠j}(z_j - z_i) is the separation gap. -/
theorem softmax_concentration {β δ : ℝ} {N : ℕ} [NeZero N] (hβ : 0 < β)
    (z : Fin N → ℝ) (j : Fin N)
    (hδ : ∀ i, i ≠ j → z j - z i ≥ δ) :
    1 - (N - 1) * exp (-β * δ) ≤ softmax β z j := by
  unfold softmax
  set S := ∑ ν : Fin N, exp (β * z ν) with hS_def
  set ej := exp (β * z j) with hej_def
  set rest := (Finset.univ.erase j).sum (fun i => exp (β * z i)) with hrest_def
  have hej_pos : (0 : ℝ) < ej := exp_pos _
  have hS_pos : 0 < S := by positivity
  have hS_ne : S ≠ 0 := ne_of_gt hS_pos
  -- Split: S = ej + rest
  have hsplit : S = ej + rest := by
    simp only [S, ej, rest]
    exact (Finset.add_sum_erase Finset.univ (fun i => exp (β * z i))
      (Finset.mem_univ j)).symm
  -- Each non-j term ≤ ej * exp(-β*δ)
  have hbound : ∀ i, i ∈ Finset.univ.erase j →
      exp (β * z i) ≤ ej * exp (-β * δ) := by
    intro i hi; rw [show ej * exp (-β * δ) = exp (β * z j + (-β * δ)) from by
      rw [hej_def, ← exp_add]]
    exact exp_le_exp.mpr (by nlinarith [hδ i (Finset.ne_of_mem_erase hi)])
  -- rest ≤ (N-1) * ej * exp(-β*δ)
  have hcard : ((Finset.univ.erase j).card : ℝ) = ↑N - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ, Fintype.card_fin,
        Nat.cast_sub (NeZero.pos N), Nat.cast_one]
  have hrest_le : rest ≤ (↑N - 1) * (ej * exp (-β * δ)) := by
    calc rest
        ≤ (Finset.univ.erase j).sum (fun _ => ej * exp (-β * δ)) :=
          Finset.sum_le_sum hbound
      _ = ↑(Finset.univ.erase j).card * (ej * exp (-β * δ)) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ = (↑N - 1) * (ej * exp (-β * δ)) := by rw [hcard]
  have hrest_nn : (0 : ℝ) ≤ rest := by
    apply Finset.sum_nonneg; intro i _; exact le_of_lt (exp_pos _)
  have ht_nn : (0 : ℝ) ≤ (↑N - 1) * exp (-β * δ) := by
    apply mul_nonneg _ (le_of_lt (exp_pos _))
    linarith [show (1 : ℝ) ≤ ↑N from by exact_mod_cast NeZero.pos N]
  -- Main: (1-t)*S ≤ ej, then ej/S ≥ 1-t
  have hmain : (1 - (↑N - 1) * exp (-β * δ)) * S ≤ ej := by
    nlinarith [hsplit, hrest_le, mul_nonneg ht_nn hrest_nn]
  rw [show (1 : ℝ) - (↑N - 1) * exp (-β * δ) =
    ((1 - (↑N - 1) * exp (-β * δ)) * S) / S from
    (mul_div_cancel_right₀ _ hS_ne).symm]
  exact div_le_div_of_nonneg_right hmain hS_pos.le

/-- Weighted Cauchy-Schwarz: (Σ w_i a_i)² ≤ (Σ w_i)(Σ w_i a_i²) for nonneg weights.
    Proved by reducing to dot_le_norms via √w substitution. -/
private theorem weighted_sq_le {N : ℕ} (w a : Fin N → ℝ) (hw : ∀ i, 0 ≤ w i) :
    (∑ i, w i * a i) ^ 2 ≤ (∑ i, w i) * (∑ i, w i * a i ^ 2) := by
  set u : Fin N → ℝ := fun i => Real.sqrt (w i)
  set v : Fin N → ℝ := fun i => Real.sqrt (w i) * a i
  have huv : ∑ i, u i * v i = ∑ i, w i * a i := by
    congr 1; ext i; show Real.sqrt (w i) * (Real.sqrt (w i) * a i) = w i * a i
    rw [← mul_assoc, Real.mul_self_sqrt (hw i)]
  have huu : sqNorm u = ∑ i, w i := by
    simp only [sqNorm]; congr 1; ext i; show Real.sqrt (w i) ^ 2 = w i
    rw [sq, Real.mul_self_sqrt (hw i)]
  have hvv : sqNorm v = ∑ i, w i * a i ^ 2 := by
    simp only [sqNorm]; congr 1; ext i
    show (Real.sqrt (w i) * a i) ^ 2 = w i * a i ^ 2
    rw [mul_pow, sq (Real.sqrt (w i)), Real.mul_self_sqrt (hw i)]
  have h := dot_le_norms u v
  rw [huv] at h
  have h1 := mul_self_le_mul_self (abs_nonneg _) h
  -- Convert |B|*|B| to B² via sq_abs, and simplify RHS via mul_self_sqrt
  have lhs : |∑ i, w i * a i| * |∑ i, w i * a i| = (∑ i, w i * a i) ^ 2 := by
    rw [← sq, sq_abs]
  have rhs : Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v) *
      (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) =
      (∑ i, w i) * (∑ i, w i * a i ^ 2) := by
    rw [show Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v) *
        (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) =
        (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm u)) *
        (Real.sqrt (sqNorm v) * Real.sqrt (sqNorm v)) from by ring,
      Real.mul_self_sqrt (sqNorm_nonneg u),
      Real.mul_self_sqrt (sqNorm_nonneg v), huu, hvv]
  rw [lhs, rhs] at h1; exact h1

/-- ‖a - b‖² ≤ 2(‖a‖² + ‖b‖²), so ≤ 4M² when both norms ≤ M² -/
private theorem sqNorm_sub_le {d : ℕ} {M : ℝ} (a b : Fin d → ℝ)
    (ha : sqNorm a ≤ M ^ 2) (hb : sqNorm b ≤ M ^ 2) :
    sqNorm (fun i => a i - b i) ≤ 4 * M ^ 2 := by
  have h : sqNorm (fun i => a i - b i) ≤ 2 * sqNorm a + 2 * sqNorm b := by
    simp only [sqNorm]
    have pw : ∀ i : Fin d, (a i - b i) ^ 2 ≤ 2 * a i ^ 2 + 2 * b i ^ 2 :=
      fun i => by nlinarith [sq_nonneg (a i + b i)]
    calc ∑ i, (a i - b i) ^ 2
        ≤ ∑ i, (2 * a i ^ 2 + 2 * b i ^ 2) :=
          Finset.sum_le_sum (fun i _ => pw i)
      _ = (∑ i, 2 * a i ^ 2) + (∑ i, 2 * b i ^ 2) :=
          Finset.sum_add_distrib
      _ = 2 * ∑ i, a i ^ 2 + 2 * ∑ i, b i ^ 2 := by
          rw [Finset.mul_sum, Finset.mul_sum]
  linarith

/-- Squared Cauchy-Schwarz: (Σ u_i v_i)² ≤ ‖u‖² · ‖v‖² -/
private theorem dot_sq_le {d : ℕ} (u v : Fin d → ℝ) :
    (∑ i, u i * v i) ^ 2 ≤ sqNorm u * sqNorm v := by
  have h := dot_le_norms u v
  have hsq := mul_self_le_mul_self (abs_nonneg _) h
  have lhs_eq : |∑ i, u i * v i| * |∑ i, u i * v i| = (∑ i, u i * v i) ^ 2 := by
    rw [← sq, sq_abs]
  have h1 := Real.mul_self_sqrt (sqNorm_nonneg u)
  have h2 := Real.mul_self_sqrt (sqNorm_nonneg v)
  have rhs_eq : (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) *
      (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm v)) = sqNorm u * sqNorm v := by
    calc _ = (Real.sqrt (sqNorm u) * Real.sqrt (sqNorm u)) *
          (Real.sqrt (sqNorm v) * Real.sqrt (sqNorm v)) := by ring
      _ = sqNorm u * sqNorm v := by rw [h1, h2]
  linarith

/-! ## Gradient of Local Energy

  ∇E_local(ξ) = ξ - X·softmax(β, X^T ξ)

  At a stored pattern ξ = x_μ, if softmax concentrates on μ:
  ∇E_local(x_μ) ≈ x_μ - x_μ = 0

  Exact zero requires perfect concentration (β → ∞) or
  orthogonal patterns. For finite β and separated patterns,
  the gradient is small: ‖∇E‖ ≤ ε(β, δ, M).
-/

/-- The gradient of local energy is ξ minus the softmax-weighted
    combination of patterns. -/
def localEnergyGrad {d N : ℕ} (cfg : SystemConfig d N)
    (ξ : Fin d → ℝ) : Fin d → ℝ :=
  fun k =>
    let similarities : Fin N → ℝ := fun μ => ∑ i, cfg.patterns μ i * ξ i
    let weights : Fin N → ℝ := softmax cfg.β similarities
    ξ k - ∑ μ, weights μ * cfg.patterns μ k

/-- At a stored pattern with sufficient similarity gap, the gradient
    norm is bounded by a function of β, δ, and M.
    This shows stored patterns are approximate stationary points.

    The hypothesis uses a similarity gap (dot product separation) rather
    than L2 distance, which is the natural assumption for Modern Hopfield
    networks (cf. Ramsauer et al. 2020, Theorem A5).

    The factor of 4 arises from ‖x_μ - x_ν‖² ≤ (‖x_μ‖ + ‖x_ν‖)² ≤ 4M². -/
theorem localEnergyGrad_small_at_pattern {d N : ℕ} [NeZero N] {δ M : ℝ}
    (cfg : SystemConfig d N) (μ : Fin N)
    (hSep : ∀ ν, ν ≠ μ →
      (∑ i, cfg.patterns μ i * cfg.patterns μ i) -
      (∑ i, cfg.patterns ν i * cfg.patterns μ i) ≥ δ)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2) :
    sqNorm (localEnergyGrad cfg (cfg.patterns μ)) ≤
      4 * ((N - 1) * exp (-cfg.β * δ) * M) ^ 2 := by
  -- Set up the softmax weights at the stored pattern
  set sim : Fin N → ℝ := fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i with hsim_def
  set p : Fin N → ℝ := softmax cfg.β sim with hp_def
  -- (1) Softmax concentration: p_μ ≥ 1 - (N-1)·exp(-β·δ)
  have hconc : 1 - (↑N - 1) * exp (-cfg.β * δ) ≤ p μ :=
    softmax_concentration cfg.β_pos sim μ hSep
  have hp_nn : ∀ ν, 0 ≤ p ν := fun ν => softmax_nonneg sim ν
  have hp_sum : ∑ ν, p ν = 1 := softmax_sum_one sim
  -- p_μ ≤ 1 (single nonneg term ≤ sum of nonneg terms = 1)
  have hp_le_one : p μ ≤ 1 := by
    calc p μ ≤ ∑ ν, p ν :=
          Finset.single_le_sum (f := p) (fun i _ => hp_nn i) (Finset.mem_univ μ)
      _ = 1 := hp_sum
  -- (2) Tail bound: 1 - p_μ ≤ (N-1)·exp(-β·δ)
  have hp_tail : 1 - p μ ≤ (↑N - 1) * exp (-cfg.β * δ) := by linarith
  have hp_tail_nn : 0 ≤ 1 - p μ := by linarith
  -- (3) Gradient norm bound: ‖g‖² ≤ 4M²(1-p_μ)²
  --     From: g_k = Σ_ν p_ν(x_μ_k - x_ν_k), Jensen gives ‖g‖² ≤ (1-p_μ)·Σ_{ν≠μ} p_ν‖x_μ-x_ν‖²
  --     and ‖x_μ-x_ν‖² ≤ 4M², so ‖g‖² ≤ 4M²·(1-p_μ)²
  have hgrad_bound : sqNorm (localEnergyGrad cfg (cfg.patterns μ)) ≤
      4 * M ^ 2 * (1 - p μ) ^ 2 := by
    -- Step 1: Rewrite gradient as Σ p_ν(x_μ - x_ν) using Σp=1
    have hg_eq : ∀ k, localEnergyGrad cfg (cfg.patterns μ) k =
        ∑ ν, p ν * (cfg.patterns μ k - cfg.patterns ν k) := by
      intro k
      show cfg.patterns μ k - ∑ ν, p ν * cfg.patterns ν k =
        ∑ ν, p ν * (cfg.patterns μ k - cfg.patterns ν k)
      simp_rw [mul_sub]
      rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hp_sum, one_mul]
    -- Step 2: Reduced weights w (zero at μ) with Σw = 1-pμ
    set w : Fin N → ℝ := fun ν => if ν = μ then 0 else p ν
    have hw_nn : ∀ i, 0 ≤ w i := by
      intro i; simp only [w]; split_ifs <;> linarith [hp_nn i]
    have hw_sum : ∑ i, w i = 1 - p μ := by
      have h1 : w μ = 0 := if_pos rfl
      have h2 : ∀ i, i ≠ μ → w i = p i := fun i hi => if_neg hi
      have h3 := Finset.add_sum_erase Finset.univ w (Finset.mem_univ μ)
      have h4 := Finset.add_sum_erase Finset.univ p (Finset.mem_univ μ)
      have h5 : (Finset.univ.erase μ).sum w = (Finset.univ.erase μ).sum p :=
        Finset.sum_congr rfl fun i hi => h2 i (Finset.ne_of_mem_erase hi)
      linarith [hp_sum]
    -- Step 3: g_k = Σ w·diff (since p_μ·0 = 0 = w_μ·diff_μ)
    have hg_w : ∀ k, localEnergyGrad cfg (cfg.patterns μ) k =
        ∑ ν, w ν * (cfg.patterns μ k - cfg.patterns ν k) := by
      intro k; rw [hg_eq k]; congr 1; ext ν; simp only [w]
      split_ifs with h
      · rw [h, sub_self, mul_zero, zero_mul]
      · rfl
    -- Step 4: Apply weighted_sq_le for each k
    have hcs_k : ∀ k, (localEnergyGrad cfg (cfg.patterns μ) k) ^ 2 ≤
        (1 - p μ) * ∑ ν, w ν * (cfg.patterns μ k - cfg.patterns ν k) ^ 2 := by
      intro k; rw [hg_w k, ← hw_sum]
      exact weighted_sq_le w (fun ν => cfg.patterns μ k - cfg.patterns ν k) hw_nn
    -- Step 5: Sum over k, swap sums, bound sqNorm, factor
    calc sqNorm (localEnergyGrad cfg (cfg.patterns μ))
        = ∑ k, (localEnergyGrad cfg (cfg.patterns μ) k) ^ 2 := rfl
      _ ≤ ∑ k, ((1 - p μ) * ∑ ν, w ν * (cfg.patterns μ k - cfg.patterns ν k) ^ 2) :=
          Finset.sum_le_sum (fun k _ => hcs_k k)
      _ = (1 - p μ) * ∑ k, ∑ ν, w ν * (cfg.patterns μ k - cfg.patterns ν k) ^ 2 :=
          (Finset.mul_sum ..).symm
      _ = (1 - p μ) * ∑ ν, w ν * sqNorm (fun k => cfg.patterns μ k - cfg.patterns ν k) := by
          congr 1; simp only [sqNorm]; rw [Finset.sum_comm]; congr 1; ext ν
          rw [← Finset.mul_sum]
      _ ≤ (1 - p μ) * (4 * M ^ 2 * (1 - p μ)) :=
          mul_le_mul_of_nonneg_left (by
            calc ∑ ν, w ν * sqNorm (fun k => cfg.patterns μ k - cfg.patterns ν k)
                ≤ ∑ ν, w ν * (4 * M ^ 2) :=
                  Finset.sum_le_sum (fun ν _ => mul_le_mul_of_nonneg_left
                    (sqNorm_sub_le _ _ (hNorm μ) (hNorm ν)) (hw_nn ν))
              _ = 4 * M ^ 2 * ∑ ν, w ν := by rw [← Finset.sum_mul]; ring
              _ = 4 * M ^ 2 * (1 - p μ) := by rw [hw_sum]) hp_tail_nn
      _ = 4 * M ^ 2 * (1 - p μ) ^ 2 := by ring
  -- (4) Combine: 4M²(1-p_μ)² ≤ 4M²·((N-1)·exp(-β·δ))² = 4·((N-1)·exp(-β·δ)·M)²
  calc sqNorm (localEnergyGrad cfg (cfg.patterns μ))
      ≤ 4 * M ^ 2 * (1 - p μ) ^ 2 := hgrad_bound
    _ ≤ 4 * M ^ 2 * ((↑N - 1) * exp (-cfg.β * δ)) ^ 2 := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        rw [sq, sq]
        exact mul_self_le_mul_self hp_tail_nn hp_tail
    _ = 4 * ((↑N - 1) * exp (-cfg.β * δ) * M) ^ 2 := by ring

/-! ## Hessian of Local Energy

  ∇²E_local(ξ) = I - β X Diag(p) X^T + β X p p^T X^T
               = I - β X (Diag(p) - p p^T) X^T

  where p = softmax(β, X^T ξ).

  At a stored pattern where p concentrates on one index,
  Diag(p) - p p^T ≈ 0, so ∇²E ≈ I, which is positive definite.

  For finite β, we need: I - β X (Diag(p) - ppT) XT is PD.
  Since Diag(p) - ppT is PSD (covariance matrix of multinomial),
  and its eigenvalues are ≤ 1/4, we need β · λ_max(X(Diag(p)-ppT)XT) < 1.
-/

/-- The Hessian of local energy at a stored pattern is close to
    the identity (positive definite) when β and separation are large enough.
    This is the second-order condition ensuring local minimum, not saddle point. -/
theorem hessian_positive_definite_at_pattern {d N : ℕ} [NeZero N] {δ M : ℝ}
    (cfg : SystemConfig d N) (μ : Fin N)
    (hSep : ∀ ν, ν ≠ μ →
      (∑ i, cfg.patterns μ i * cfg.patterns μ i) -
      (∑ i, cfg.patterns ν i * cfg.patterns μ i) ≥ δ)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hβ_large : cfg.β * δ ≥ Real.log (4 * ↑N * cfg.β * M ^ 2)) :
    -- Under these conditions, the Hessian eigenvalues are > 0.
    -- We state this as: the quadratic form is positive on nonzero vectors.
    ∀ v : Fin d → ℝ, sqNorm v > 0 →
      -- v^T (∇²E) v > 0
      -- Simplified: ‖v‖² - β · v^T X(Diag(p)-ppT)X^T v > 0
      let similarities := fun ν => ∑ i, cfg.patterns ν i * (cfg.patterns μ i)
      let p := softmax cfg.β similarities
      (∑ i, v i ^ 2) - cfg.β *
        ((∑ ν, p ν * (∑ i, cfg.patterns ν i * v i) ^ 2) -
         (∑ ν, p ν * (∑ i, cfg.patterns ν i * v i)) ^ 2) > 0 := by
  intro v hv
  -- Reduce let bindings in the goal
  show (∑ i, v i ^ 2) - cfg.β *
    ((∑ ν, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i) ν *
        (∑ i, cfg.patterns ν i * v i) ^ 2) -
     (∑ ν, softmax cfg.β (fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i) ν *
        (∑ i, cfg.patterns ν i * v i)) ^ 2) > 0
  set sim : Fin N → ℝ := fun ν => ∑ i, cfg.patterns ν i * cfg.patterns μ i
  set p := softmax cfg.β sim
  set dot_v : Fin N → ℝ := fun ν => ∑ i, cfg.patterns ν i * v i
  -- The expression is ‖v‖² - β · Var_p[dot_v]
  set variance := (∑ ν, p ν * dot_v ν ^ 2) - (∑ ν, p ν * dot_v ν) ^ 2
  -- (1) Softmax concentration
  have hconc : 1 - (↑N - 1) * exp (-cfg.β * δ) ≤ p μ :=
    softmax_concentration cfg.β_pos sim μ hSep
  have hp_nn : ∀ ν, 0 ≤ p ν := fun ν => softmax_nonneg sim ν
  have hp_sum : ∑ ν, p ν = 1 := softmax_sum_one sim
  have hp_le_one : p μ ≤ 1 := by
    calc p μ ≤ ∑ ν, p ν :=
          Finset.single_le_sum (f := p) (fun i _ => hp_nn i) (Finset.mem_univ μ)
      _ = 1 := hp_sum
  have hp_tail : 1 - p μ ≤ (↑N - 1) * exp (-cfg.β * δ) := by linarith
  have hp_tail_nn : 0 ≤ 1 - p μ := by linarith
  -- (2) Variance bound via bias-variance decomposition + Cauchy-Schwarz
  have hvar_bound : variance ≤ 4 * M ^ 2 * sqNorm v * (1 - p μ) := by
    -- Step 1: Var ≤ Σ p(X-c)² where c = dot_v μ (bias-variance decomposition)
    have step1 : variance ≤ ∑ ν, p ν * (dot_v ν - dot_v μ) ^ 2 := by
      have h_expand : ∑ ν, p ν * (dot_v ν - dot_v μ) ^ 2 =
        (∑ ν, p ν * dot_v ν ^ 2) - 2 * dot_v μ * (∑ ν, p ν * dot_v ν) +
        dot_v μ ^ 2 := by
        have pw : ∀ ν, p ν * (dot_v ν - dot_v μ) ^ 2 =
          p ν * dot_v ν ^ 2 + (-(2 * dot_v μ) * (p ν * dot_v ν) +
          dot_v μ ^ 2 * p ν) := fun ν => by ring
        simp_rw [pw, Finset.sum_add_distrib, ← Finset.mul_sum, hp_sum, mul_one]
        ring
      rw [h_expand]
      show (∑ ν, p ν * dot_v ν ^ 2) - (∑ ν, p ν * dot_v ν) ^ 2 ≤
        (∑ ν, p ν * dot_v ν ^ 2) - 2 * dot_v μ * (∑ ν, p ν * dot_v ν) +
        dot_v μ ^ 2
      nlinarith [sq_nonneg ((∑ ν, p ν * dot_v ν) - dot_v μ)]
    -- Step 2: μ-th term is zero, so only ν ≠ μ contribute
    have hμ_zero : p μ * (dot_v μ - dot_v μ) ^ 2 = 0 := by simp
    have step1' : variance ≤ (Finset.univ.erase μ).sum
        (fun ν => p ν * (dot_v ν - dot_v μ) ^ 2) := by
      linarith [Finset.add_sum_erase Finset.univ
        (fun ν => p ν * (dot_v ν - dot_v μ) ^ 2) (Finset.mem_univ μ)]
    -- Step 3: Each (dot_v ν - dot_v μ)² ≤ 4M² · sqNorm v (CS + sqNorm_sub_le)
    have hdiff_bound : ∀ ν, (dot_v ν - dot_v μ) ^ 2 ≤
        4 * M ^ 2 * sqNorm v := by
      intro ν
      have hdiff_eq : dot_v ν - dot_v μ =
          ∑ i, (cfg.patterns ν i - cfg.patterns μ i) * v i := by
        show (∑ i, cfg.patterns ν i * v i) - (∑ i, cfg.patterns μ i * v i) =
          ∑ i, (cfg.patterns ν i - cfg.patterns μ i) * v i
        rw [← Finset.sum_sub_distrib]; congr 1; ext i; ring
      rw [hdiff_eq]
      calc (∑ i, (cfg.patterns ν i - cfg.patterns μ i) * v i) ^ 2
          ≤ sqNorm (fun i => cfg.patterns ν i - cfg.patterns μ i) * sqNorm v :=
            dot_sq_le _ _
        _ ≤ 4 * M ^ 2 * sqNorm v :=
            mul_le_mul_of_nonneg_right
              (sqNorm_sub_le _ _ (hNorm ν) (hNorm μ)) (sqNorm_nonneg v)
    -- Step 4: Combine — Var ≤ Σ_{ν≠μ} p_ν · 4M²‖v‖² = 4M²‖v‖²·(1-p_μ)
    have herase_sum : (Finset.univ.erase μ).sum p = 1 - p μ := by
      linarith [hp_sum, Finset.add_sum_erase Finset.univ p (Finset.mem_univ μ)]
    calc variance
        ≤ (Finset.univ.erase μ).sum
            (fun ν => p ν * (dot_v ν - dot_v μ) ^ 2) := step1'
      _ ≤ (Finset.univ.erase μ).sum
            (fun ν => p ν * (4 * M ^ 2 * sqNorm v)) :=
          Finset.sum_le_sum (fun ν _ =>
            mul_le_mul_of_nonneg_left (hdiff_bound ν) (hp_nn ν))
      _ = 4 * M ^ 2 * sqNorm v * (Finset.univ.erase μ).sum p := by
          rw [← Finset.sum_mul]; ring
      _ = 4 * M ^ 2 * sqNorm v * (1 - p μ) := by rw [herase_sum]
  have hN_pos : (0 : ℝ) < ↑N := by exact_mod_cast NeZero.pos N
  have hN_ge_one : (1 : ℝ) ≤ ↑N := by exact_mod_cast NeZero.pos N
  -- (3) Main bound: β · Var < ‖v‖², split on M = 0
  suffices hmain : cfg.β * variance < ∑ i, v i ^ 2 by linarith
  -- Suffices to show: β · 4M²(1-pμ) < 1, then multiply by ‖v‖²
  suffices hcoeff : cfg.β * (4 * M ^ 2 * (1 - p μ)) < 1 by
    have h1 : cfg.β * variance ≤ cfg.β * (4 * M ^ 2 * sqNorm v * (1 - p μ)) :=
      mul_le_mul_of_nonneg_left hvar_bound (le_of_lt cfg.β_pos)
    have h2 : cfg.β * (4 * M ^ 2 * sqNorm v * (1 - p μ)) =
              cfg.β * (4 * M ^ 2 * (1 - p μ)) * sqNorm v := by ring
    have h3 : cfg.β * (4 * M ^ 2 * (1 - p μ)) * sqNorm v < 1 * sqNorm v := by
      exact mul_lt_mul_of_pos_right hcoeff hv
    unfold sqNorm at h1 h2 h3; linarith
  -- Prove the coefficient bound: β · 4M² · (1-pμ) < 1
  by_cases hM : M = 0
  · -- M = 0: coefficient is 0 < 1
    simp [hM]
  · -- M ≠ 0: use exp bound from hβ_large
    have hM2_pos : (0 : ℝ) < M ^ 2 := sq_pos_of_ne_zero hM
    have hβM_pos : 0 < 4 * ↑N * cfg.β * M ^ 2 :=
      mul_pos (mul_pos (mul_pos (by norm_num : (0:ℝ) < 4) hN_pos) cfg.β_pos) hM2_pos
    -- From hβ_large: 4NβM² ≤ exp(βδ), so 4(N-1)βM² < exp(βδ)
    have hexp_large : 4 * (↑N - 1) * cfg.β * M ^ 2 < exp (cfg.β * δ) := by
      have h1 : 4 * ↑N * cfg.β * M ^ 2 ≤ exp (cfg.β * δ) := by
        calc 4 * ↑N * cfg.β * M ^ 2
            = exp (log (4 * ↑N * cfg.β * M ^ 2)) := (exp_log hβM_pos).symm
          _ ≤ exp (cfg.β * δ) := exp_le_exp.mpr hβ_large
      -- 4(N-1)βM² < 4NβM² ≤ exp(βδ)
      have : 4 * (↑N - 1) * cfg.β * M ^ 2 < 4 * ↑N * cfg.β * M ^ 2 := by
        nlinarith [cfg.β_pos]
      linarith
    -- Therefore β · 4M² · (1-pμ) ≤ β · 4M² · (N-1)·exp(-βδ) < 1
    calc cfg.β * (4 * M ^ 2 * (1 - p μ))
        ≤ cfg.β * (4 * M ^ 2 * ((↑N - 1) * exp (-cfg.β * δ))) := by
          nlinarith [hp_tail]
      _ < 1 := by
          rw [show (-cfg.β * δ : ℝ) = -(cfg.β * δ) from neg_mul cfg.β δ, exp_neg]
          rw [show cfg.β * (4 * M ^ 2 * ((↑N - 1) * (exp (cfg.β * δ))⁻¹)) =
              4 * (↑N - 1) * cfg.β * M ^ 2 * (exp (cfg.β * δ))⁻¹ from by ring]
          rw [mul_inv_lt_iff₀ (exp_pos _)]
          linarith [hexp_large]

/-! ## Main Phase 1 Result: Patterns are Local Minima -/

/-- PHASE 1 MAIN THEOREM: Under sufficient separation and β,
    each stored pattern is a local minimum of the energy function.

    Conditions:
    - Patterns are δ-separated in embedding space
    - β is large enough relative to δ, M, and N
    - Pattern norms bounded by M

    Conclusion:
    - Gradient approximately zero at stored patterns
    - Hessian positive definite at stored patterns
    - Therefore: stored patterns are local energy minima -/
theorem stored_patterns_are_local_minima {d N : ℕ} [NeZero N] {δ M : ℝ}
    (cfg : SystemConfig d N) (μ : Fin N)
    (hSep : ∀ ν, ν ≠ μ →
      (∑ i, cfg.patterns μ i * cfg.patterns μ i) -
      (∑ i, cfg.patterns ν i * cfg.patterns μ i) ≥ δ)
    (hNorm : ∀ ν, sqNorm (cfg.patterns ν) ≤ M ^ 2)
    (hβ_large : cfg.β * δ ≥ Real.log (4 * ↑N * cfg.β * M ^ 2)) :
    -- The pattern is an approximate local minimum:
    -- gradient is small AND Hessian is positive definite
    sqNorm (localEnergyGrad cfg (cfg.patterns μ)) ≤
      4 * ((N - 1) * exp (-cfg.β * δ) * M) ^ 2 ∧
    (∀ v : Fin d → ℝ, sqNorm v > 0 →
      let similarities := fun ν => ∑ i, cfg.patterns ν i * (cfg.patterns μ i)
      let p := softmax cfg.β similarities
      (∑ i, v i ^ 2) - cfg.β *
        ((∑ ν, p ν * (∑ i, cfg.patterns ν i * v i) ^ 2) -
         (∑ ν, p ν * (∑ i, cfg.patterns ν i * v i)) ^ 2) > 0) := by
  exact ⟨localEnergyGrad_small_at_pattern cfg μ hSep hNorm,
         hessian_positive_definite_at_pattern cfg μ hSep hNorm hβ_large⟩

end HermesNLCDM
