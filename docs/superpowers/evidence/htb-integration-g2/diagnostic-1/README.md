# V2 Python-ORT failure — bounded diagnostic 1 (2026-11-09)

**Authorized:** the user, 2026-11-09 (ledger decision note of that date).
- **Cap:** one hour of execution; about 5½ minutes was used.
- **Not changed:** no package, no alternate export, no tolerance.
- **Unchanged and verified:** the student (sha256 `0e1cb621…`), the exported pair (`artifact_hashes.txt`: `bc68a274…` / `317d7a7c…`, made read-only before use) and the V2 FAIL record.

## Result: the cause is identified

**Python onnxruntime 1.24.4 (CPU) computes distinct Conv nodes with ONE set of parameters when their weight tensors
have the same shape and the first output filter (filter 0) is entirely zero.** It does this within one session, and
across sessions in the same process. **onnxruntime-node 1.23.2, the product's runtime, does not.**

The student has **four** 128×128 convolutions with an all-zero filter 0: `res_blocks.{0,1,2,3}.conv1`. calib020_0001
(c34b7ff) has **one**, `res_blocks.0.conv1`, so it has nothing to be conflated with, which is why August's parity
passed. (Both models' 30-channel `encoder_conv1` also has a zero filter 0, but no other conv shares its shape.)

The exported artifact is not at fault. Node ORT reads the same bytes and agrees with native MLX to 2.4e-5 (V2's S4).

## Configurations and results, in order

Inputs: five parity-corpus positions (`opening_00`, `early_mid_10`, `midgame_20`, `late_29`, `edge_red_to_move`).
- Their exact bytes are saved under `inputs/`, made by a process that imported neither MLX nor PyTorch.
- Each board's sha256 equals V2's recorded `encoding_sha256`.
- Comparison is against V2's recorded MLX and Node outputs (`scripts/score.py`).

### Full model (`runs/full_model_configurations.txt`, `runs/c*.json`; `scripts/ort_run.py`, run with `python -I`)

| run | configuration | result |
|---|---|---|
| c1 | NumPy + ORT only (MLX and PyTorch confirmed not loaded) | **still wrong**: logits 2.35–9.89 from both MLX and Node; values 0.02–1.70; **bit-identical to V2's `ort_py`** |
| c2 | single thread (intra = inter = 1) | still wrong; within 5.7e-6 of V2's `ort_py` |
| c3 | a c34b7ff session created first in the process | wrong, and **differently**: 4.1–11.6 from V2's `ort_py` |
| c4 | c3 plus single thread | identical to c3 (deterministic) |
| c5, c6, c7 | MLX / PyTorch / both imported before ORT | bit-identical to c1 |
| c8 | MLX imported after a session, then a new session | bit-identical to c1 |
| c9 | `session.disable_prepacking=1` | identical to c1 |
| c10, c11, c12 | CPU memory arena off / memory patterns off / both | identical to c1 |
| c13 | all graph optimizations off | identical to c1 |
| c14 | optimizations off and prepacking off | identical to c1 |

- **The parity tool's MLX / PyTorch imports are not the cause** (c1, c5–c8).
- **Session order matters** (c3, c4): process-level state is involved.
- **No ORT session option tried changes the result** (c9–c14).

### Layer by layer (`runs/layer_by_layer_opening_00.txt`; `scripts/py_compare.py`, `scripts/node_dump.mjs`)

A **diagnostic-only** copy of the graph exposes all 74 intermediate outputs. It was kept in scratch and is not
committed.
- Python and Node agree to ≤ 3e-6 through `relu_2`, the input to residual block 1.
- They first diverge at **`getitem_9`**, block 1's `conv1` (max difference 4.96).

Further checks:
- That conv's weights alone, in a one-node graph in a fresh Python process, are correct: 7.6e-7 from a float64 NumPy
  reference.
- In the full graph, Python's `getitem_9` equals the convolution of its correct input with **`res_blocks.0.conv1`'s
  weight and bias**, to 1.6e-6.

### Minimal reproduction (`runs/minimal_two_conv.txt`, `runs/minimal_two_conv_node.txt`; `scripts/mini.py`, `scripts/node_mini.mjs`)

- **Setup:** one fresh process per case; N parallel Convs on one random input.
- **Judged against:** a float64 NumPy reference, not another ORT session.

| case | Python ORT 1.24.4 |
|---|---|
| student `res_blocks.1.conv1` alone | correct |
| student blocks 0 + 1 `conv1`, either order | the first conv computed with the other's parameters |
| student blocks 0 + 1 + 2 `conv1` | the first two computed with block 2's parameters |
| two random dense convs | correct |
| two random convs, **filter 0 zero** (1, 8 or 73 leading zero filters) | conflated |
| two random convs, the last 73 filters zero | correct |
| two random convs, every third filter zero (filter 0 zero in one only) | correct |
| two random convs, 60% of elements zero | correct |

**onnxruntime-node 1.23.2 on the same saved graphs** (random leading-zero pair; student blocks 0 + 1) computes both
convs correctly: their outputs differ by 19.4 and 16.4, where Python's are bit-identical.

### Unsuccessful or contaminated checks, recorded

- **Shell slip:** the first attempt at c2–c7 passed its options as one word (zsh does not split `$args`), so nothing
  ran. It was rerun with `${=args}`.
- **Contaminated two-conv test:** a first in-process two-conv test used further ORT sessions as its reference. Those
  references were themselves conflated across sessions, which made "correct" and "conflated" indistinguishable. It
  was replaced by the fresh-process, NumPy-referenced test above.
- **Byte-prefix hypothesis:** byte prefixes were compared as a possible cache key. They do not explain it: the
  conflated weights share only 3 bytes, because `-0.0` and `0.0` differ in the sign bit.

## What is established, and what is not

**Established:**
- the trigger: same-shape Conv weights, each with an all-zero filter 0;
- the effect: one parameter set used for all of them, within a session and across sessions in one process;
- the scope: Python ORT 1.24.4 on this machine;
- the product's runtime (onnxruntime-node 1.23.2) is unaffected on the same graphs.

**Not established:**
- the mechanism inside ORT (some weight sharing keyed on content that an all-zero leading filter makes collide is the
  natural reading, but it was not inspected);
- which other ORT versions are affected;
- whether any other product of the exporter is affected. Shapes other than 128×128×3×3 were not tested.

## Proposed repair (for review; nothing applied)

1. **Requalify V2 with a Python ORT surface that does not have the defect.** This needs a package change, in an
   isolated environment, not the project `.venv`.
   - The first candidate is `onnxruntime==1.23.2` (Python), the product's own runtime version.
   - Before any measurement, that ORT must pass the minimal reproduction (the canary) above.
   - It needs a reviewed amendment naming the Python ORT version that S2 and S3 measure. Phase 2 §3 recorded 1.24.4
     as the environment of record.
   - The tolerances, corpus, exporter and artifact stay unchanged.
2. **Guard the product against importing the defect.**
   - `package.json` allows `onnxruntime-node ^1.20.0`; only the lockfile pins 1.23.2.
   - Add a model-free canary test to the server suite: two same-shape Convs with a zero filter 0 must produce
     different, correct outputs. Any runtime upgrade that brings this defect would then fail CI rather than silently
     corrupt the student's `hard` play.
   - Optionally, pin the dependency exactly.

## Next worthwhile discriminator

Run the canary (`scripts/mini.py`, the random cases need no model) under other Python ORT versions in a throwaway
environment, for example 1.23.2 and the newest release. That shows which versions are safe for the parity surface and
for any future product upgrade. It needs approval, because it installs packages outside the project.
