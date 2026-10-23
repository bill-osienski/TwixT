# Search distillation D2 — full-float32 setting validation card (MLX_ENABLE_TF32=0)

**Date:** 2026-10-23. **Status:** FROZEN before any run. **Gate:** `SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED` (the
fortieth), opened for ONE approved invocation by a one-line commit and restored by the wrapper.

**Approved:** preparation and one bounded validation invocation, capped at 10 minutes. **Not approved:** package
upgrades, chunking, persistent environment changes, model replay, training, held-out access, bug-report submission,
pushing, retries or any other setting.

## 1. Purpose and background

Does launching a fresh process with `MLX_ENABLE_TF32=0` remove the GPU defects recorded in ledger entries 27 and 30, for
the isolated linear operations? If it passes **both** sets below, it is a **candidate workaround for the isolated
operations only**. It would still need validating through the policy head and the full-model gradient path before
any training resumes.

**Background:**
- **MLX's precision page** says float32 matrix-multiplication operations may run at reduced precision on hardware with
  dedicated matrix units, and that `MLX_ENABLE_TF32=0` set at process launch keeps them in full float32.
- **The 0.30.3 source, as read:**
  - `mlx/utils.h`: `enable_tf32()` reads `MLX_ENABLE_TF32` once, into a function-local static, default 1.
  - `mlx/backend/metal/matmul.cpp`: `steel_matmul_regular_axpby` takes the NAX kernel when
    `is_nax_available() && (enable_tf32() || dtype != float32)`; otherwise the standard fused GEMM.
  - GEMV (when M or N is 1) and split-K are chosen before that test and are not gated by it.
- **The installed binary** (`.venv/.../mlx/lib/libmlx.dylib`) contains the strings `MLX_ENABLE_TF32` and
  `steel_gemm_fused_nax_`.
- **Entry 30's pattern:**
  - Outside the gross rows, on every GPU evaluation of every case, Y and dX exceeded the frozen tolerance, by up to
    about 1.2e-3 of scale.
  - dW, db and out's Y stayed within the tolerance everywhere, and so did the CPU.
  - The outputs within the tolerance are those this reading puts on ungated paths (GEMV, split-K, reduction).
- **This supports testing the setting. It does not establish the cause of the gross rows or of the tolerance
  failures.**

## 2. What runs (frozen)

- **The process:** the wrapper launches `python -B -m scripts.GPU.alphazero.search_distillation_d2_tf32off run` with
  `MLX_ENABLE_TF32=0` in that process's environment only. Nothing is installed or persisted.
- **The entry refuses** unless:
  - `MLX_ENABLE_TF32` is exactly `"0"` when it starts;
  - no `mlx` module is imported at that moment;
  - then the gate is open, every pin below matches, and no output exists.

  It records the variable's value at the start and at the end.
- **Same packages and application code:** MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, the M5 Max.

**Set A — synthetic:**
- The standalone probe `scripts/GPU/alphazero/mlx_matmul_rows_probe.py` (SHA-256 `1217bc8e48589b077bbe0962564ffb5c0298326f96de6658ae39a83cf739bb57`), **unchanged**.
- Its frozen SPEC: 27 cases (fc 2→64, out 64→1, wide 64→64 × nine row counts), the four products evaluated jointly,
  the GPU three times and the CPU once per case.
- It writes its own create-only outputs in `probe/` under the evidence directory. Its compact-example rule is
  unchanged, with the example directory `~/twixt-cache/search-distillation-d2/tf32off-<UTC stamp>/`.

**Set B — the four original saved cases** of entry 27:
- **Cases:** `fc_S0_original`, `fc_S21_original`, `out_S0_original`, `out_S21_original`, read from
  `~/twixt-cache/search-distillation-d2/linrepro-20261023T041634Z/<case>.npz`.
- **Integrity:**
  - each file's SHA-256 must equal the one in entry 27's committed
    `docs/superpowers/evidence/search-distillation-d2-linrepro/artifacts.jsonl` (that file pinned, SHA-256
    `668cae70a3b8ef85b8e9d90c857d37f98e58e52d38050162752235bb3e7abc88`), at the recorded path;
  - the saved float64 expected arrays must equal NumPy's recomputation from the saved operands, bit for bit.

  Either failure is VOID.
- **Implementations:** entry 27's three, unchanged: `linear_autodiff`, `explicit_autodiff` and `explicit_products`.
  They come from `search_distillation_d2_linrepro.compute` (module pinned, SHA-256
  `76fe5eff60a89aa6ca6e33f1e935d1dc617cd0d2f7152b18deab833cef53b8f0`), with a contiguous layout and joint
  evaluation.
- **Repeats:** each implementation on the GPU three times and on the CPU once.
- Only the saved operands are read: no network, no corpus, no held-out file.

**Order:** set A, then set B. If set A does not complete, the run is VOID and set B does not run.

## 3. Judgement (frozen)

Both sets use the probe's own `judge` and `scales` with its SPEC values:
- **Scale:** max|expected| for Y, dX and dW; for db, the absolute scale max_j Σ_i |C_ij|.
- **Outside the tolerance:** |got − expected| > 1e-4 × scale + 1e-6.
- **Gross:** |got − expected| > 1e-2 × scale.

**Three counts per set, over GPU evaluations,** reported separately:
- **T:** GPU evaluations (case × implementation × output × repeat) with **any element outside the tolerance**;
- **G:** GPU evaluations with **any gross element**;
- **R:** GPU (case × implementation × output) triples whose repeats 2 and 3 are **not bitwise identical** to repeat 1.

**The CPU's T, G and R** are reported alongside as a comparison. They do not enter the verdict.

**Verdict:**
- **A set PASSES** only if T = G = R = 0 on the GPU.
- **CANDIDATE** (for the isolated operations only): both sets pass.
- **NOT_A_WORKAROUND:** any GPU count is non-zero in either set. The record names each non-zero count and set.
  **Removing gross corruption alone (G = 0 with T > 0) is NOT_A_WORKAROUND.**
- **VOID:** the run did not complete.

**Also reported, outside the verdict:**
- **Tolerance-only rows:** for every Y and dX evaluation, the rows outside the tolerance that hold no gross element.
- **Comparisons with the earlier records**, for the predictions. For each case, implementation, device, output and
  repeat, it records whether the largest absolute error equals the one recorded before, exactly:
  - set A against entry 30's `cases.jsonl` (pinned, SHA-256
    `4ad1bdf538f3e79bd78a0fe0b237b92fa22a362acb7fbd760a83271dfe223d3d`);
  - set B against entry 27's `cases.jsonl` (pinned, SHA-256
    `35583162d8ce79f2fd01faf3b342e2a5050526386cdfa580cb25c3b239c2f617`).

  The groups are:
  - CPU;
  - **GPU ungated**: every dW and db, and out's Y;
  - **GPU gated**: every other output.

## 4. Evidence

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-tf32off/`, all create-only:
  - `progress.jsonl`;
  - `probe/`: the probe's `environment.json`, `progress.jsonl`, `cases.jsonl` and `record.json`, plus
    `artifacts.jsonl` if it saves an example;
  - `fixture_cases.jsonl`: one line per set-B case, flushed and fsynced as each finishes;
  - `record.json`, written on every exit the script survives: the setting at the start and end, both sets' counts,
    the verdict and the comparisons.
- **Compact failure evidence:**
  - every output's first 20 failing elements (index, observed, expected);
  - gross row ranges;
  - the largest absolute errors inside and outside them;
  - the probe's one compact example if a case has gross rows.
- **No full outputs are saved.** Set B saves no arrays: its operands are already in entry 27's directory.

## 5. Stops and outcome

- **A difference is an observation and never stops the run.**
- **COMPLETED (exit 0)** whatever the verdict.
- **VOID (exit 3):**
  - set A not completed;
  - a set-B fixture failing its integrity checks;
  - a non-finite result;
  - any exception;
  - SIGTERM (the wrapper's ceiling).
- **REFUSED (exit 4):** the setting is not `"0"` or MLX was already imported; the gate is closed; a pin does not
  match; an output exists.
- **USAGE (exit 2).**
- **No retry, and no other setting, whatever the result.**

## 6. Predictions on record (stated before the run)

1. **CPU unaffected:** every CPU largest absolute error equals the earlier record's (entry 30 for set A, entry 27 for
   set B), case by case.
2. **GPU ungated outputs unchanged:** every dW, every db and out's Y have the same largest absolute error as the
   earlier record's, in every repeat.
3. **No precision-only failures on the GPU:** in every GPU Y and dX evaluation of both sets, no row is outside the
   tolerance unless it holds a gross element (tolerance-only rows = 0).
4. **Gross rows:** **None stated in advance.**
5. **Repeatability:** **None stated in advance.**
6. **Verdict:** **None stated in advance.**

## 7. What this cannot establish

- The cause of either defect.
- That the setting works through the policy head, the full model's gradients, or training.
- Anything for another MLX version or machine, or for shapes and sizes outside these cases.
- A cost: the timings are recorded, but this is not a benchmark.
- Anything about measured playing strength.

## 8. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate:** the wrapper restores it on every exit and reads it back against the pre-open commit.
- **Cleanup:** surviving processes are reported. The linked worktree and its branch are removed after the evidence
  commit.
