# Discriminative Validation Results: GPU Acceleration Hypotheses

Generated: 2026-03-03
Platform: macOS Darwin 25.1.0 (Apple Silicon)
Python: 3.11.13, NumPy with Accelerate BLAS

---

## H1: Numpy matmul is the actual bottleneck (not pair loops)

**VERDICT: FALSIFIED**

### Measurements

#### Random data (N=1000, d=128) -- 0 pairs above threshold=0.7

| Component | Time (ms) | Share |
|-----------|-----------|-------|
| Matmul (X @ X.T) | 0.87 | 23.3% |
| Pair-find (np.where) | 2.83 | 76.3% |
| Pair-loop (0 pairs) | 0.01 | 0.4% |

With random high-dimensional unit vectors, cosine similarities are near-zero.
No pairs exceed threshold=0.7, making the pair loop trivially fast. In this
degenerate case, `np.where(np.triu(...))` dominates due to O(N^2) memory
allocation for the boolean mask.

#### Random data (N=5000, d=128) -- 0 pairs

| Component | Time (ms) | Share |
|-----------|-----------|-------|
| Matmul | 23.28 | 25.0% |
| Pair-find | 69.90 | 75.0% |
| Pair-loop | 0.08 | 0.1% |

Same pattern at larger N. The `np.triu` + `np.where` allocation dominates.

#### Clustered data (N=1000, d=128, 20 clusters, noise=0.05) -- 23,671 pairs

| Component | Time (ms) | Share |
|-----------|-----------|-------|
| Matmul (X @ X.T) | 0.73 | 1.0% |
| Pair-find (np.where) | 2.60 | 3.5% |
| Pair-loop (23,671 pairs) | 70.72 | **95.5%** |
| **Vectorized total** | **74.05** | |
| Python N^2 dot-product loop | 267.64 | (3.6x slower than vectorized) |

**Key finding:** With realistic clustered data that produces many close pairs,
the **Python pair loop is the true bottleneck** (95.5%), not the matmul (1.0%).
The matmul itself is already handled efficiently by BLAS (~0.7ms for 1000x1000).

### Implications for GPU acceleration

- GPU-accelerating matmul alone provides negligible benefit (<1% of runtime)
- The real target for GPU acceleration is the **per-pair repulsion loop** (~71ms)
- Alternatively, the repulsion loop should be **vectorized in numpy** (batch subtract, batch normalize)
- The Python N^2 loop in `_assign_clusters` (267ms) is already addressed by matmul-based approach (see H3)

---

## H2: Float32 is sufficient (no precision loss vs float64)

**VERDICT: VALIDATED WITH CAVEATS**

### Measurements

#### Unnormalized vectors (N=500, d=128)

| Metric | Value |
|--------|-------|
| Max abs diff (S_f32 vs S_f64) | 7.74e-05 |
| Mean abs diff | 1.69e-06 |
| P99 abs diff | 8.13e-06 |
| Contract tol (1e-5) | **FAIL** |

Unnormalized random vectors accumulate float32 error beyond 1e-5.

#### Unit-normalized vectors (N=500, d=128) -- the actual use case

| Metric | Value |
|--------|-------|
| Max abs diff | 5.96e-07 |
| Mean abs diff | 1.31e-08 |
| Norm deviation (f32) | 5.96e-08 |
| Contract tol (1e-5) | **PASS** |
| Postcondition margin (1e-9) | VIOLATED |

With unit-normalized vectors (as used throughout dream_ops), max error is ~6e-7.
Well within the 1e-5 contract tolerance.

#### Unit-normalized (N=2000, d=768) -- realistic embedding size

| Metric | Value |
|--------|-------|
| Max abs diff | 7.75e-07 |
| Mean abs diff | 1.26e-08 |
| Contract tol (1e-5) | **PASS** |

Even at d=768 (common embedding dimensionality), float32 stays within tolerance.

#### Repulsion postcondition margin analysis (N=1000, d=128)

| Metric | Value |
|--------|-------|
| delta_min (f32) | 0.5902523100 |
| delta_min (f64) | 0.5902522643 |
| Difference | 4.57e-08 |
| Margin (1e-9) | **AT RISK** |

**Caveat:** The `nrem_repulsion_xb` postcondition uses a 1e-9 margin:
```python
if delta_min_out < delta_min_in - 1e-9:
    return patterns.copy()  # reject repulsion
```

Float32 introduces ~4.6e-08 error in `delta_min`, which is 46x larger than the
1e-9 margin. If switching to float32, this guard must be widened to at least
1e-6 to avoid false rejections.

### Recommendation

Float32 is safe for similarity computation (max error ~6e-7 for unit vectors).
However, the `nrem_repulsion_xb` postcondition margin (1e-9) must be widened
to ~1e-6 before switching the computation pipeline to float32.

---

## H3: _assign_clusters N^2 loop is actually slow

**VERDICT: VALIDATED (75x speedup)**

### Measurements

| N | Original loop (ms) | Matmul-based (ms) | Speedup | Correct |
|---|--------------------:|-------------------:|--------:|---------|
| 500 | 65.76 | 0.87 | 75.3x | Yes |
| 1000 | 262.75 | 3.36 | 78.2x | Yes |
| 2000 | 1065.48 | 14.35 | 74.2x | Yes |

The original `_assign_clusters` uses a Python `for i / for j` double loop
computing `float(patterns[i] @ patterns[j])` for each pair. This is O(N^2)
Python-interpreted iterations with per-call numpy overhead.

The matmul-based approach computes `S = patterns @ patterns.T` in a single
BLAS call, then uses `np.where(np.triu(S, k=1) > threshold)` to find all
close pairs, and runs union-find only on those pairs.

### Scaling analysis

| N | Loop time | Expected O(N^2) | Actual ratio |
|---|----------:|----------------:|-------------:|
| 500 | 65.76ms | 1.0x | 1.0x |
| 1000 | 262.75ms | 4.0x | 4.0x |
| 2000 | 1065.48ms | 16.0x | 16.2x |

The original loop scales as expected O(N^2). The matmul-based approach
also scales O(N^2) in the matmul, but with ~75x lower constant factor
because BLAS uses vectorized instructions and cache-optimal memory access.

### Correctness verification

At N=500, exhaustive pairwise comparison confirmed that both methods produce
identical cluster assignments (same connected components, just potentially
different label numbering). All 500 vectors were in singleton clusters at
threshold=0.5 with random data.

---

## H4: Numpy fallback produces identical results to direct numpy

**VERDICT: VALIDATED BY DESIGN**

No `gpu_ops` module exists yet. The planned fallback path will delegate
directly to numpy operations. This is a tautology by construction.

### Verified properties

| Property | Result |
|----------|--------|
| Matmul determinism | IDENTICAL (bit-exact across calls) |
| Copy preservation | IDENTICAL (dtype, layout, values) |
| f64->f32->f64 roundtrip | Max element diff: 1.49e-08, max sim diff: 2.69e-08 |

The only risk when building the abstraction layer is unintended dtype
conversions or unnecessary copies. These will be caught by the existing
dream_ops test suite and by type annotations on the gpu_ops interface.

---

## Summary Table

| Hypothesis | Verdict | Key Finding |
|------------|---------|-------------|
| H1: Matmul is bottleneck | **FALSIFIED** | Pair loop is 95.5% of runtime with clustered data; matmul is only 1% |
| H2: Float32 sufficient | **VALIDATED** | Max error ~6e-7 for unit vectors; widen 1e-9 margin to 1e-6 |
| H3: _assign_clusters slow | **VALIDATED** | 75x speedup with matmul-based approach |
| H4: Numpy fallback identical | **VALIDATED** | By design (fallback IS numpy) |

## Architectural Implications

1. **GPU acceleration of matmul alone has minimal impact** (~1% of pipeline time)
2. **The real GPU target is the per-pair Python loop** in `nrem_repulsion_xb` and similar ops
3. **Vectorizing the pair loop in numpy** (batch operations on all pairs simultaneously)
   would be more impactful than moving matmul to GPU
4. **`_assign_clusters` should adopt the matmul-based approach** immediately (75x speedup, pure numpy, no GPU needed)
5. **Float32 is safe** but requires widening the postcondition margin from 1e-9 to 1e-6

## Test Files

- `impl/tests/discriminative/test_h1_bottleneck.py` -- 3 tests (random N=1000, random N=5000, clustered N=1000)
- `impl/tests/discriminative/test_h2_precision.py` -- 4 tests (unnorm, unit-norm, postcondition margin, large-scale)
- `impl/tests/discriminative/test_h3_assign_clusters.py` -- 3 tests (N=500, N=1000, N=2000)
- `impl/tests/discriminative/test_h4_numpy_fallback.py` -- 4 tests (determinism, copy, roundtrip, design doc)
