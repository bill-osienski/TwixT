# DRAFT — MLX bug report (not submitted)

**Status:** draft only. Submitting it, and where, is a separate decision for the user.
- **Its evidence:** ledger entry 27 (fixture operands) and **entry 30** (the synthetic probe, card
  `2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md`, evidence
  `docs/superpowers/evidence/search-distillation-d2-mlxrows/`).
- **Entry 31** (`MLX_ENABLE_TF32=0`) is added below the results.
- **Entry 30 is the first retained complete record of the probe, not its first execution.** The same frozen probe ran
  once before, unrecorded, during mutation controls (entry 28). Its stray example already showed a gross GPU failure.
  Entry 30's measurements are reproducible, but they are not an independent first test.
- **Snippet below:** not run on its own. Its X, W and b follow the probe's draws for case `fc_33600`, but it evaluates
  Y alone, while the probe evaluates four products jointly.
- Revised 2026-10-23 after entry 30: the title, the summary and the repeatability statement changed (the earlier draft
  said "very narrow operand", "first N − 32,736 rows" and "identical across three runs"; entry 30 contradicts all three).
- Revised again 2026-10-23 (wording): outputs earlier called "clean" or "correct" on the GPU were often only **free of
  gross errors**. Below, *within tolerance* (error ≤ 1e-4 × max|expected| + 1e-6) and *no gross error* (error ≤ 1e-2 ×
  max|expected|) are kept apart.

---

**Title:** Metal: float32 matmul returns wrong leading rows when the row count exceeds 32,768 (M5 Max, MLX 0.30.3)

**Environment** (recorded in both runs):
- MLX 0.30.3 (pip), NumPy 2.4.1, Python 3.14.9;
- macOS 27.0.1, Apple M5 Max (`applegpu_g17s`), default GPU device.

**Summary.**
- On the GPU, for X float32 of shape (N, K) and W of shape (F, K), the products below return grossly wrong values in
  a band of **leading rows** whenever N was above 32,768 in our tests, and in no row at N ≤ 32,768:
  - `X @ W.T + b` with (K, F) = (2, 64) and (64, 64);
  - `C @ W` with C (N, 64) and W (64, 2); with C (N, 1) and W (1, 64); with C (N, 64) and W (64, 64).
- **Band sizes measured:** 32 rows at N = 32,769; 64 at 32,800; 864 at 33,600; 992 at 33,728. With fixture
  operands (entry 27), 864 at 33,600 and 992 at 33,728 too.
- **Errors in the band** are of the order of the outputs themselves (the largest per output is 0.5 to 3.6 ×
  max|expected|).
- **Outside the band, these GPU outputs are still not within tolerance.** Every GPU evaluation of them, at every N
  tested (synthetic 32,704–33,728; fixtures 512–33,728), has elements beyond 1e-4 × max|expected|, up to about 1.2e-3 ×
  max|expected|: free of gross errors, but outside the tolerance (one exception: in one evaluation at 33,600, a row of
  the band fell just under the gross threshold, by inference).
- **The CPU device is within tolerance everywhere** (at most 3.9e-7 of max|expected|).
- **What stays within tolerance on the GPU at every N tested:** reductions over the rows (`C.T @ X`, `C.sum(0)`) and
  the (N, 64) @ (64, 1) product.
- **Where no row was grossly wrong** (but the products above were still outside tolerance): N = 32,704; 32,736;
  32,737; 32,767; 32,768 (synthetic), and 512, 2,048, 8,192, 16,384 and 32,768 (fixtures).
- **Repeatability:** the wrong rows are **not** bitwise repeatable within a process.
  - Three GPU evaluations of the same synthetic operands: the first differs from the second and third in every failing
    output (largest differences 2.5–6.5). The second and third agree on every reported statistic.
  - In the first evaluation, the band's rows of `X @ W.T + b` equal `b` alone (at 32,769, 33,600 and 33,728), as if the
    X·Wᵀ term were missing. In the later ones they hold other wrong values.
  - Outputs with no gross error repeat bitwise.
  - Across two separate processes, the first evaluation's band for one case (N = 32,769, rows 0–39 saved) was byte
    identical.
- **What did not change it** (fixtures, entry 27): `nn.Linear` or plain `@`; autodiff (`mx.vjp`) or explicit products;
  X contiguous or a transposed view; outputs evaluated jointly or one at a time.

**Minimal reproduction** (synthetic; the probe's case `fc_33600` operands, Y only; not run on its own):

```python
import numpy as np, mlx.core as mx

N = 33_600
rng = np.random.default_rng([20261023, 0, N])
X = rng.standard_normal((N, 2)).astype(np.float32)
W = (rng.standard_normal((64, 2)) / np.sqrt(2)).astype(np.float32)
b = rng.standard_normal(64).astype(np.float32)

expected = X.astype(np.float64) @ W.astype(np.float64).T + b
for dev in (mx.gpu, mx.cpu):
    mx.set_default_device(dev)
    for rep in range(3):
        y = mx.array(X) @ mx.array(W).T + mx.array(b)
        mx.eval(y)
        err = np.abs(np.array(y) - expected).max(axis=1)
        bad = np.flatnonzero(err > 1e-2 * np.abs(expected).max())
        print(dev, rep, "bad rows:", (bad.min(), bad.max(), bad.size) if bad.size else None)
# Probe (four products evaluated jointly), case fc_33600, Y on the GPU: bad rows [0, 864) in repeats 1-2; 863 of
# them in repeat 0, one row of the band falling just under the threshold (inferred). CPU: none.
```

**Results with synthetic operands** (entry 30; 27 cases, GPU three times and CPU once each, against NumPy float64;
gross = error above 1e-2 × max|expected|; "no gross" = no gross element but elements outside the 1e-4 tolerance in
every GPU evaluation; "within tol" = every element within it):

| N | fc 2→64: Y, dX | out 64→1: Y | out 64→1: dX | wide 64→64: Y, dX | every dW, db | CPU |
|---|---|---|---|---|---|---|
| 32,704 – 32,768 (5 sizes) | no gross | within tol | no gross | no gross | within tol | within tol |
| 32,769 | rows [0, 32) gross | within tol | rows [0, 32) gross | rows [0, 32) gross | within tol | within tol |
| 32,800 | rows [0, 64) gross | within tol | rows [0, 64) gross | rows [0, 64) gross | within tol | within tol |
| 33,600 | rows [0, 864) gross | within tol | rows [0, 864) gross | rows [0, 864) gross | within tol | within tol |
| 33,728 | rows [0, 992) gross | within tol | rows [0, 992) gross | rows [0, 992) gross | within tol | within tol |

In the rows marked gross, the other rows of the same outputs are also outside tolerance.

**With `MLX_ENABLE_TF32=0`** (ledger entry 31; a fresh process, same packages, the same 27 synthetic cases plus the
four original fixture cases, GPU three times and CPU once each):
- every GPU output is within the 1e-4 tolerance (Y and dX now at most 3.9e-7 × max|expected|);
- no gross rows at any size;
- every output is bitwise repeatable;
- the CPU's results, and the GPU products the setting does not gate in the 0.30.3 source (GEMV, split-K and the
  reductions), are bit-for-bit unchanged.

By that source reading, the setting chooses between the NAX kernel and the standard fused GEMM for float32
(`steel_matmul_regular_axpby`). The result is consistent with both the precision loss and the leading-row corruption
lying on the NAX path; no kernel was inspected.

**Not claimed:**
- the kernel or dispatch path responsible (the setting's effect points at the NAX path, but proves nothing about it);
- a rule for the band's size (N − 32,736 fits three sizes and fails at 32,769);
- a general rule for other shapes, devices or MLX versions;
- a safe row count.

**Attachments to consider:**
- the standalone script `mlx_matmul_rows_probe.py` (standard library, NumPy and MLX only);
- the probe's `environment.json` and `cases.jsonl`;
- one compact `.npz` (case fc_32769, Y: full operands, rows 0–39, expected rows and GPU repeat 0's rows).
