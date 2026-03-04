"""H2: Float32 is sufficient (no precision loss vs float64).

Discriminative test: compute similarity matrices in both dtypes,
measure max absolute difference. Check if it falls within the
contract tolerance (1e-5) and the postcondition margin in
nrem_repulsion_xb (1e-9).
"""

import numpy as np


def test_h2_similarity_matrix_precision():
    """Max abs diff between float32 and float64 similarity matrices."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 128))

    X_f32 = X.astype(np.float32)
    X_f64 = X.astype(np.float64)

    S_f32 = (X_f32 @ X_f32.T).astype(np.float64)
    S_f64 = X_f64 @ X_f64.T

    abs_diff = np.abs(S_f32 - S_f64)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    p99_diff = float(np.percentile(abs_diff, 99))

    print(f"\n{'='*60}")
    print("H2: Float32 vs Float64 Precision (N=500, d=128)")
    print(f"{'='*60}")
    print(f"  Max absolute difference:   {max_diff:.2e}")
    print(f"  Mean absolute difference:  {mean_diff:.2e}")
    print(f"  P99 absolute difference:   {p99_diff:.2e}")
    print(f"  Contract tolerance (1e-5): {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    print(f"  Postcondition margin (1e-9): {'SAFE' if max_diff < 1e-9 else 'VIOLATED'}")
    print(f"{'='*60}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "p99_diff": p99_diff,
        "contract_ok": max_diff < 1e-5,
        "postcondition_safe": max_diff < 1e-9,
    }


def test_h2_unit_normalized_precision():
    """Same test but with unit-normalized vectors (as used in dream_ops)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 128))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit norm

    X_f32 = X.astype(np.float32)
    X_f64 = X.astype(np.float64)

    S_f32 = (X_f32 @ X_f32.T).astype(np.float64)
    S_f64 = X_f64 @ X_f64.T

    abs_diff = np.abs(S_f32 - S_f64)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    # Check if the normalization step itself introduces error
    norms_f32 = np.linalg.norm(X_f32, axis=1)
    norm_deviation = float(np.max(np.abs(norms_f32 - 1.0)))

    print(f"\n{'='*60}")
    print("H2: Float32 vs Float64 Precision (unit-normalized, N=500, d=128)")
    print(f"{'='*60}")
    print(f"  Max sim difference:        {max_diff:.2e}")
    print(f"  Mean sim difference:       {mean_diff:.2e}")
    print(f"  Max norm deviation (f32):  {norm_deviation:.2e}")
    print(f"  Contract tolerance (1e-5): {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    print(f"  Postcondition margin (1e-9): {'SAFE' if max_diff < 1e-9 else 'VIOLATED'}")
    print(f"{'='*60}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "norm_deviation_f32": norm_deviation,
        "contract_ok": max_diff < 1e-5,
        "postcondition_safe": max_diff < 1e-9,
    }


def test_h2_repulsion_postcondition_margin():
    """Check if float32 error can flip the 1e-9 postcondition in nrem_repulsion_xb.

    The postcondition is: delta_min_out >= delta_min_in - 1e-9
    If float32 introduces > 1e-9 error in max(S), this could cause
    a valid repulsion to be falsely rejected.
    """
    rng = np.random.default_rng(42)
    N, d = 1000, 128
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Compute delta_min in both precisions
    S_f32 = (X.astype(np.float32) @ X.astype(np.float32).T).astype(np.float64)
    S_f64 = X @ X.T

    np.fill_diagonal(S_f32, -2.0)
    np.fill_diagonal(S_f64, -2.0)

    max_sim_f32 = float(np.max(S_f32))
    max_sim_f64 = float(np.max(S_f64))

    delta_min_f32 = 1.0 - max_sim_f32
    delta_min_f64 = 1.0 - max_sim_f64

    delta_min_diff = abs(delta_min_f32 - delta_min_f64)

    print(f"\n{'='*60}")
    print("H2: Repulsion Postcondition Margin Analysis (N=1000, d=128)")
    print(f"{'='*60}")
    print(f"  delta_min (f32):  {delta_min_f32:.10f}")
    print(f"  delta_min (f64):  {delta_min_f64:.10f}")
    print(f"  Difference:       {delta_min_diff:.2e}")
    print(f"  Margin (1e-9):    {'SAFE' if delta_min_diff < 1e-9 else 'AT RISK'}")
    print(f"{'='*60}")

    # If delta_min_diff > 1e-9, the float32 error could flip the postcondition
    if delta_min_diff >= 1e-9:
        print("  WARNING: Float32 precision difference exceeds 1e-9 margin!")
        print("           The postcondition guard may need widening to ~1e-5")
        print("           if switching to float32 computation.")

    return {
        "delta_min_f32": delta_min_f32,
        "delta_min_f64": delta_min_f64,
        "delta_min_diff": delta_min_diff,
        "margin_safe": delta_min_diff < 1e-9,
    }


def test_h2_large_scale_precision():
    """At N=2000, d=768 (realistic embedding size), check precision."""
    rng = np.random.default_rng(42)
    N, d = 2000, 768
    X = rng.standard_normal((N, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    X_f32 = X.astype(np.float32)
    X_f64 = X.astype(np.float64)

    S_f32 = (X_f32 @ X_f32.T).astype(np.float64)
    S_f64 = X_f64 @ X_f64.T

    abs_diff = np.abs(S_f32 - S_f64)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    print(f"\n{'='*60}")
    print(f"H2: Float32 vs Float64 Precision (unit-norm, N={N}, d={d})")
    print(f"{'='*60}")
    print(f"  Max sim difference:        {max_diff:.2e}")
    print(f"  Mean sim difference:       {mean_diff:.2e}")
    print(f"  Contract tolerance (1e-5): {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    print(f"{'='*60}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "contract_ok": max_diff < 1e-5,
    }


if __name__ == "__main__":
    test_h2_similarity_matrix_precision()
    test_h2_unit_normalized_precision()
    test_h2_repulsion_postcondition_margin()
    test_h2_large_scale_precision()
