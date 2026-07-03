# Targeted Value Calibration v9 — `--train-value-head-and-final-block` Design

**Status:** DESIGN — awaiting user review, then writing-plans.
**Date:** 2026-07-03
**Supersedes-for-next-experiment:** v8 (`--train-value-head-only`, merged @ 0948a0d). v9 is the next hypothesis after the v8/v8b A-gate diagnostic below.

## Hypothesis

The v8 line established two facts:

1. **Full-trunk updates cause the nonlocal B/C/D drift.** Value-head-only training (v8) *preserved* guardrails B/C/D, which v6/v7 (full-network) broke. So the drift is a representation-level side effect of updating the whole trunk, not an inevitable cost of the calibration signal.
2. **Value-head-only is too constrained to move A.** On the 50 `black_predrop_correction` rows, the *raw* value-head output barely moved: BASE raw mean −0.2469 → v8 −0.2533 (Δ −0.0064), v8b −0.2433 (Δ +0.0035); severe raw overvalue only 20.0% → 14.0% (v8) / 16.0% (v8b). A did not fail because MCTS amplified an already-corrected raw value — A failed because a **shallow MLP readout on frozen features** (`value_head` is `fc1→fc2`, no conv, no BN) cannot substantially move the worst A raw values while the trunk is frozen.

**v9 hypothesis:** unfreeze the *smallest late representation slice* — the value head **plus the final residual block** `encoder.blocks[last]` — to give A enough representational flexibility to move the worst raw values, while keeping the rest of the trunk (stem + all earlier blocks + policy head) frozen to avoid the v6/v7-style nonlocal B/C/D drift.

**Interpretation of the result:**
- v9 passes A *and* holds B/C/D ⇒ the final block is the right amount of flexibility; the drift lives in the *earlier* trunk.
- v9 holds B/C/D but still fails A ⇒ one late block is still too constrained; consider v9b (last-2 blocks).
- v9 moves A but *breaks* B/C/D ⇒ even one late block is enough to induce the nonlocal drift; the guardrail damage is a late-representation phenomenon, and partial unfreeze is a dead end.

## Scope decisions (two made while user was away — flagged for review)

1. **Flag surface = boolean, final block only** (`--train-value-head-and-final-block`, `store_true`), mirroring v8's `--train-value-head-only`. *Rationale:* matches the name the user chose; matches the v8 precedent; and is **mechanically simpler** — a single `opt_main.update` on one block avoids the Adam double-step hazard that a multi-block loop would reintroduce (see MainModule docstring, trainer.py:214). Widening to last-N (v9b) is a deliberate future change that needs a wrapper module to preserve single-step semantics — explicitly NOT pre-built here (YAGNI). *If the user prefers a `--unfreeze-final-n-blocks N` count knob now, that changes the flag + verifier surface and the plan; revisit before writing-plans.*
2. **Verifier is strict on BN running stats everywhere, including the final block.** Allowed-to-change = `value_head.*` ∪ (`encoder.blocks.<last>.*` trainable, i.e. excluding `*.running_mean` / `*.running_var`). Every running stat in the whole network — including the final block's — must stay byte-identical, so a forgotten `--freeze-batchnorm-stats` is caught as a leak. This is faithful to what the guard actually does (the optimizer never touches BN buffers; only `--freeze-batchnorm-stats` keeps forward-pass tracking from moving them).

Everything else follows the v8 experiment definition unchanged: same v7 manifest, same v8 schedule (NOT v8b), weight 0.01, both `--freeze-batchnorm-stats` and the new flag, gates A/B/C/D vs `calib020_0001` with thresholds unchanged.

## Mechanism (the load-bearing part)

**Architecture recap (trainer.py `train_step`).** One gradient tree is computed over the whole network; grads are sliced into `main_grads = {"encoder": grads["encoder"], "policy_head": grads["policy_head"]}` and `value_grads = grads["value_head"]`, each clipped, then applied by two optimizers: `opt_main.update(main_module, main_grads)` (encoder+policy) and `opt_value.update(network.value_head, value_grads)`. v8 added a guard: when `train_value_head_only`, skip the `opt_main` call; `opt_value` always runs.

**v9 = v8's guard + one extra update.** When `train_value_head_and_final_block` is set:
- Skip the whole-trunk `opt_main.update(main_module, main_grads)` (exactly as v8 does), **then**
- Apply only the final block's already-clipped grads to the live block submodule:
  ```python
  last = len(network.encoder.blocks) - 1
  opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])
  ```

Why this is correct and minimal:
- **Reuses `opt_main`, no third optimizer, no `MainModule` change, no `train()` optimizer-creation change.** On the v9 path `opt_main` is otherwise unused, so its first-ever `update` initializes Adam state to the block's shape and stays consistent every step. `opt_main.state` is only ever consumed inside a shape-agnostic `mx.eval(...)` (trainer.py:1426) and is never checkpointed — verified — so repurposing it is safe.
- **Single `update` call per step ⇒ no double-step.** The MainModule wrapper existed to batch encoder+policy into one call; here we make exactly one call on one submodule, like `opt_value` already does on `value_head`.
- **Correct learning rate.** The block trains at `opt_main`'s `learning_rate` (trunk rate) — the same rate it would get in a full-trunk run — isolating the experiment's only variable to *which* params are applied. The value head keeps `opt_value`'s lower rate (`learning_rate × value_lr_scale`).
- **Magnitude-preserving clipping.** `main_grads` is still built over the full trunk and clipped by global norm at 1.0 *unchanged* (telemetry `main_gnorm` keeps its full-trunk meaning). Because `clip_grad_norm` scales every leaf by the same global factor, `main_grads["encoder"]["blocks"][last]` is exactly the block's contribution as it would be in a full-trunk step. No new clip path.
- **Structurally cannot leak.** `network.encoder.blocks[last]` is a real `ResBlock` submodule; descending into it sidesteps the list-vs-dict grad-tree reconstruction entirely. `policy_head`, the stem, and blocks `0..last-1` are never passed to any `update`, so they cannot move. BN `running_mean`/`running_var` are buffers (not in `trainable_parameters`, no backprop grad), so the optimizer never touches them; `--freeze-batchnorm-stats` (momentum=0 on all 14 BN modules) keeps the forward pass from moving them.

**Forward-compat note (do NOT build now):** last-N unfreeze (v9b) must NOT be a loop of N `opt_main.update` calls (that double-steps Adam N×). It would need a `FinalBlocksModule` wrapper holding the last-N blocks and a single `update` — a future design task. The mechanism is written for the singleton case; the block index is computed dynamically so the *location* logic already generalizes.

**Mutual exclusion.** `--train-value-head-only` and `--train-value-head-and-final-block` are mutually exclusive. Enforce with a `ValueError` in `train_step` (unit-testable without the CLI) and, for UX, an argparse-level check in `train.py`.

**Byte-identical when off.** Both flags default `False`; the new kwarg is appended last in `train_step`/`train` signatures; when neither flag is set the base `opt_main.update(main_module, main_grads)` path runs exactly as today. `alphazero_loss_batch` untouched; 7/10/14-tuple return arities unchanged. All pre-existing tests pass unmodified.

## Files

| File | Role |
|---|---|
| `scripts/GPU/alphazero/trainer.py` | `train_step` signature + v9 branch + mutual-exclusion guard; `train()` signature; two `train_step` call sites; startup print; checkpoint-JSON `"train_value_head_and_final_block"` + `"unfrozen_block_index"` |
| `scripts/GPU/alphazero/train.py` | argparse `--train-value-head-and-final-block` + mutual-exclusion check + plumb into `train(...)` |
| `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py` (create) | v9 tensor-diff acceptance verifier (new file; keeps the v8 verifier byte-stable) |
| `tests/test_train_value_head_and_final_block.py` (create) | trainer behavior + wiring tests |
| `tests/test_verify_value_head_and_final_block_checkpoint.py` (create) | verifier tests |

## Verifier (`verify_value_head_and_final_block_checkpoint.py`)

Compares two safetensors checkpoints. It auto-detects the final block index from the *base* checkpoint keys (`last = max n where "encoder.blocks.<n>." prefixes a key`) — it does not import the network.

Classification of every base tensor key `k`:
- **allowed-to-change** iff `k.startswith("value_head.")` OR (`k.startswith(f"encoder.blocks.{last}.")` AND NOT (`k.endswith(".running_mean")` or `k.endswith(".running_var")`)).
- **frozen** = everything else, including all BN running stats anywhere (stem, every block including the last, `policy_head.bn`).

`compare(base, cand) -> dict` with:
- `frozen_diffs: list[str]` — frozen tensors that changed (leaks).
- `value_head_deltas: dict[str,float]` — max-abs delta per `value_head.*` tensor (expect 4).
- `final_block_deltas: dict[str,float]` — max-abs delta per allowed final-block tensor (expect 8).
- `last_block_index: int`, `n_tensors: int`.
- Raises `ValueError` if base/candidate key sets differ (wrong architecture).

CLI `python -m scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint --base ... --candidate ...` exit codes:
- **0 PASS** — every frozen tensor byte-identical, value head changed, AND final block changed.
- **1 FAIL (leak)** — some frozen tensor changed (prints each leaking key; a running-stat leak means `--freeze-batchnorm-stats` was missing).
- **2 FAIL (no-op)** — no `value_head.*` tensor changed (training no-oped).
- **3 FAIL (v9 collapsed to v8)** — value head changed but no final-block tensor changed (the partial unfreeze never engaged — flag mis-plumbed). Distinct code so the operator sees this specific failure.

Optional `--last-block-index` override (defaults to auto-detect) for robustness, but auto-detect is the norm.

## Telemetry

Checkpoint JSON gains (next to `"freeze_batchnorm_stats"` / `"train_value_head_only"`):
- `"train_value_head_and_final_block": <bool>`
- `"unfrozen_block_index": <int or null>` — `last` when the flag is on, else null.

All existing calibration telemetry is unchanged (`calib_n_drawn_total`, `calib_policy_ce_avg_iter`, `n_teacher_retention_drawn=0` still expected — the manifest is value-only). The v9 flag changes *which params update*, not the loss or the draw counts.

## Tests

**`tests/test_train_value_head_and_final_block.py`** (real `train_step` on a tiny net, `create_network(hidden=64, n_blocks=2)` → `last=1`, BN running stats frozen in setup):
- `test_flag_on_only_value_head_and_final_block_change`: after 2 steps, changed keys ⊆ `{value_head.*, encoder.blocks.1.<trainable>}`; assert BOTH a `value_head.*` and an `encoder.blocks.1.*` tensor changed; assert `encoder.blocks.0.*`, `encoder.conv1.*`, `encoder.bn1.*`, `policy_head.*` did NOT change; assert no `*.running_mean/var` changed; arity == 7.
- `test_flag_off_default_trains_everything`: pins today's behavior (encoder+policy+value all move).
- `test_v9_and_v8_mutually_exclusive`: `train_step(..., train_value_head_only=True, train_value_head_and_final_block=True)` raises `ValueError`.
- `test_flag_on_with_calibration_batch_keeps_14_tuple`: v7-style masked teacher-mode batch still returns the 14-tuple under the flag.
- `test_train_loop_wiring_source_level` / `test_cli_flag_exists_and_plumbs`: both call sites forward the flag; checkpoint JSON key present; CLI flag + plumb strings present; argparse mutual-exclusion check present.

**`tests/test_verify_value_head_and_final_block_checkpoint.py`** (round-trip real `create_network` checkpoints via `save_weights`/`mx.load`):
- value head + final block change → exit 0.
- an earlier block (`encoder.blocks.0.conv1.weight`) changes → exit 1 (leak).
- `policy_head` tensor changes → exit 1.
- a final-block running stat (`encoder.blocks.<last>.bn1.running_mean`) changes → exit 1 (strict-BN check).
- identical checkpoints → exit 2.
- value head changed, final block unchanged → exit 3.
- different `n_blocks` (key-set mismatch) → `ValueError`.

## Global constraints (carried from v8)

- Python `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main after merge: **1318 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- Byte-identical when both flags absent/False. `alphazero_loss_batch` untouched.
- Do NOT touch: `alphazero_loss_batch`, `calibration_pool.py`, `mcts.py`, `continuation_extraction.py`, any builder/smoke, any manifest/checkpoint, `docs/post-game-analysis.md`, and the v8 verifier `verify_value_head_only_checkpoint.py`.
- Worktree; fresh worktree lacks gitignored game-log data → known 14F+6E in the whole-repo suite there; judge tasks file-scoped, authoritative suite on merged main. Per-task commits, FF-merge (no `--no-ff`, never force-push). Trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## Operator run (USER's, after merge)

1. **Train** — the v7 command with a new checkpoint dir and the two flags:
   `--checkpoint-dir checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001`, same v7 manifest (`logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`), same v8 schedule (`black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`), weight 0.01, `--freeze-batchnorm-stats --train-value-head-and-final-block`.
2. **Telemetry** (checkpoint JSON): `train_value_head_and_final_block=True`, `unfrozen_block_index=5`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_policy_ce_avg_iter=0.0`, `n_teacher_retention_drawn=0`.
3. **Tensor-diff acceptance** (the REAL proof), must exit 0:
   `.venv/bin/python -m scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint --base checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors --candidate checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`
   Exit 0 = all frozen tensors (every non-`value_head`/non-final-block tensor, plus ALL BN running stats incl. the final block's) byte-identical; the 4 value-head and 8 final-block tensors changed. Exit 3 specifically means the partial unfreeze never engaged.
4. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v9_value_head_and_final_block_v7_manifest_from_calib020_0001_gates_400s`. Thresholds unchanged from v8: A mean ≤ 0.0 and severe materially below 43.3%; B severe = 0.0 and over ≤ 11.1%; C severe ≤ 13.3%, over ≤ 33.3%, mean ≤ +0.099; D severe = 0.0 and mean ≤ 0.0. No promotion match unless all four pass.
5. **Ledger update** with the v9 row and the v8/v8b raw-A diagnostic (below).

## Ledger update to record (from the v8/v8b diagnostic)

> **v8/v8b raw-A diagnostic:** On the 50 `black_predrop_correction` rows, BASE raw mean was −0.2469, v8 was −0.2533, v8b was −0.2433. Severe raw overvalue improved only from 20.0% to 14.0% (v8) / 16.0% (v8b). Mean raw deltas were tiny (−0.0064 v8, +0.0035 v8b). Therefore A did not fail because MCTS amplified an already-corrected raw value; A failed because value-head-only could not substantially move the worst A raw values with the trunk frozen.
>
> **Conclusion:** v8 proved full-network drift was the main cause of B/C/D breakage (value-head-only preserved B/C/D), but value-head-only is too constrained to fix A. Next hypothesis is partial unfreeze: value head + the smallest late representation slice, starting with the final encoder/residual block (v9).
