# Targeted Value Calibration v8 — `--train-value-head-only` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--train-value-head-only` training flag that skips all encoder/policy-head parameter updates so only the 4 `value_head.*` tensors can change, plus a checkpoint tensor-diff verifier that proves it after a run.

**Architecture:** The trainer already computes one gradient tree and applies it with TWO optimizers (`opt_main.update(main_module, main_grads)` for encoder+policy, `opt_value.update(network.value_head, value_grads)` for the value head — trainer.py `train_step`). v8 is therefore a guard: when the flag is set, the `opt_main.update` call is skipped and everything else (loss, grads, value update, telemetry arity) is byte-identical. NO MLX `freeze()`/`unfreeze()` — that would change the grads-tree shape and break the `grads["encoder"]` slicing. A small standalone CLI compares two safetensors checkpoints and hard-fails if any tensor outside `value_head.*` differs.

**Tech Stack:** Python 3.14 / MLX (Apple Metal — NOTE: no `requires_grad`, no `named_parameters()`; params are nested dicts via `tree_flatten`), pytest.

**Requirements source:** user's v8 message of 2026-07-03 (experiment definition, flag name `--train-value-head-only` locked) + agreed mechanism revision (guard-the-update, tensor-diff acceptance test instead of name-matching).

## Global Constraints

- Python: always `.venv/bin/python` from the repo root; tests: `.venv/bin/python -m pytest <file> -v`; full suite baseline on main: 1309 passed.
- NEVER `sys.modules.pop("mlx")` (or any mlx submodule) in tests — native re-import SIGABRTs the suite.
- **Byte-identical when unused:** flag absent/False → `train_step` behavior, return arities (7/10/14-tuple), and all pre-existing tests pass UNMODIFIED. `alphazero_loss_batch` is untouched (no loss-math change of any kind).
- The value head is exactly 4 tensors: `value_head.fc1.weight (256,128)`, `value_head.fc1.bias (256,)`, `value_head.fc2.weight (1,256)`, `value_head.fc2.bias (1,)`. BatchNorm affine params live under `encoder.*`/`policy_head.*` (covered by the guard); BN *running stats* are frozen by the EXISTING `--freeze-batchnorm-stats` (momentum=0, `freeze_batchnorm_running_stats` trainer.py:61) — v8 runs use BOTH flags.
- Do NOT touch: `alphazero_loss_batch`, calibration_pool.py, mcts.py, continuation_extraction.py, any builder/smoke, any manifest/checkpoint, docs/post-game-analysis.md.
- One feature branch `feature/tvc-v8-train-value-head-only` (worktree; a fresh worktree lacks gitignored local game-log data → known 14F+6E in the whole-repo suite there; judge tasks on file-scoped runs, authoritative suite on merged main).
- Per-task commits, FF-merge to main (no `--no-ff`, never force-push). Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/trainer.py` (modify: `train_step` + `train()` signature + 2 call sites + startup print + JSON telemetry) | the flag's behavior |
| `scripts/GPU/alphazero/train.py` (modify: argparse + 1 plumb line) | CLI surface |
| `scripts/GPU/alphazero/verify_value_head_only_checkpoint.py` (create) | post-train tensor-diff verifier CLI |
| `tests/test_train_value_head_only.py` (create) | Task 1 tests |
| `tests/test_verify_value_head_only_checkpoint.py` (create) | Task 2 tests |

---

### Task 1: `--train-value-head-only` end-to-end (train_step guard + plumbing + telemetry)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (4 sites: `train_step` signature+guard; `train()` signature; the TWO `train_step(...)` call sites in the train loop; startup print after optimizer creation; checkpoint-JSON field)
- Modify: `scripts/GPU/alphazero/train.py` (argparse flag + plumb into the `train(...)` call)
- Test: `tests/test_train_value_head_only.py` (create)

**Interfaces:**
- Consumes: existing `train_step(network, main_module, opt_main, opt_value, batch, ...)`, `MainModule`, `create_network`, `freeze_batchnorm_running_stats` (all in trainer.py / network.py).
- Produces: `train_step(..., train_value_head_only: bool = False)`; `train(..., train_value_head_only: bool = False)`; CLI `--train-value-head-only` (store_true); checkpoint JSON key `"train_value_head_only"`. Task 2 is independent of these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_value_head_only.py`:

```python
"""--train-value-head-only: opt_main.update is skipped so ONLY value_head.*
tensors can change. Behavior tests run real train_step on a tiny net (with BN
running stats frozen, as v8 runs use BOTH flags); wiring that lives inside the
4000-line train loop is pinned source-level (precedent:
tests/test_trainer_teacher_mode_gate.py)."""
import re

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero import train as train_mod
from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
        ply=0, game_n_moves=10,
    )


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)     # v8 pairs the flags; keeps
    #  encoder.*.running_mean/var from moving via forward-pass tracking
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-3)
    return net, mm, opt_main, opt_value


def _params(net):
    return dict(tree_flatten(net.parameters()))


def _changed_keys(before, after):
    return sorted(k for k in before
                  if not bool(mx.array_equal(before[k], after[k]).item()))


def test_flag_on_only_value_head_changes():
    net, mm, opt_main, opt_value = _setup()
    before = _params(net)
    for _ in range(2):
        out = train_step(network=net, main_module=mm, opt_main=opt_main,
                         opt_value=opt_value, batch=[_pos() for _ in range(3)],
                         train_value_head_only=True)
    assert len(out) == 7                       # arity unchanged
    changed = _changed_keys(before, _params(net))
    assert changed, "value head must still train"
    assert all(k.startswith("value_head.") for k in changed), changed


def test_flag_off_default_trains_encoder_and_policy_too():
    net, mm, opt_main, opt_value = _setup()
    before = _params(net)
    for _ in range(2):
        train_step(network=net, main_module=mm, opt_main=opt_main,
                   opt_value=opt_value, batch=[_pos() for _ in range(3)])
    changed = _changed_keys(before, _params(net))
    assert any(k.startswith("encoder.") for k in changed)
    assert any(k.startswith("policy_head.") for k in changed)
    assert any(k.startswith("value_head.") for k in changed)


def test_flag_on_with_calibration_batch_keeps_14_tuple():
    """v8 trains WITH the v7 calibration manifest: the masked teacher-mode
    path must still return its 14-tuple under the flag."""
    net, mm, opt_main, opt_value = _setup()
    calib = [PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0, 0], outcome=-0.35, active_size=24,
        ply=20, game_n_moves=None)]
    out = train_step(network=net, main_module=mm, opt_main=opt_main,
                     opt_value=opt_value, batch=[_pos() for _ in range(3)],
                     calibration_positions=calib,
                     calibration_loss_weight=0.01,
                     calibration_teacher_policy_mask=np.zeros((1,), dtype=np.float32),
                     teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
                     train_value_head_only=True)
    assert len(out) == 14


def test_train_loop_wiring_source_level():
    src = open(trainer_mod.__file__).read()
    # both train_step call sites in the train loop forward the flag
    assert len(re.findall(r"train_value_head_only=train_value_head_only,", src)) == 2
    # checkpoint JSON records the run config
    assert '"train_value_head_only": train_value_head_only,' in src


def test_cli_flag_exists_and_plumbs():
    src = open(train_mod.__file__).read()
    assert '"--train-value-head-only"' in src
    assert "train_value_head_only=args.train_value_head_only," in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_value_head_only.py -v`
Expected: `test_flag_on_only_value_head_changes` / `test_flag_on_with_calibration_batch_keeps_14_tuple` FAIL with `TypeError: train_step() got an unexpected keyword argument 'train_value_head_only'`; the two source-level tests FAIL on the missing strings; `test_flag_off_default_trains_encoder_and_policy_too` PASSES already (it pins today's behavior — that is expected at RED).

- [ ] **Step 3: Implement**

**(a) `trainer.py` — `train_step` signature.** Locate `def train_step(` and add the new keyword parameter at the END of the parameter list (after `teacher_policy_kl_weight: float = 0.25,`):

```python
    teacher_policy_kl_weight: float = 0.25,           # v4
    train_value_head_only: bool = False,              # v8: skip opt_main.update
```

**(b) `trainer.py` — the guard.** Locate the update block at the end of `train_step`:

```python
    # Update REAL modules (guaranteed to mutate network)
    opt_main.update(main_module, main_grads)
    opt_value.update(network.value_head, value_grads)
```

Replace with:

```python
    # Update REAL modules (guaranteed to mutate network)
    if not train_value_head_only:
        opt_main.update(main_module, main_grads)
    # v8: with train_value_head_only, encoder+policy grads are computed and
    # clipped (telemetry unchanged) but never applied — only the value head
    # trains. BN running stats need --freeze-batchnorm-stats separately.
    opt_value.update(network.value_head, value_grads)
```

**(c) `trainer.py` — `train()` signature.** Locate `freeze_batchnorm_stats: bool = False,` in the `def train(` parameter list and add directly below it:

```python
    train_value_head_only: bool = False,
```

**(d) `trainer.py` — startup print.** Locate the two-optimizer creation:

```python
    opt_main = optim.Adam(learning_rate=learning_rate)
    value_lr = learning_rate * value_lr_scale
    opt_value = optim.Adam(learning_rate=value_lr)
```

and add directly after it:

```python
    if train_value_head_only:
        print("TRAIN VALUE HEAD ONLY: encoder+policy_head updates DISABLED "
              "(opt_main.update skipped; value head lr unchanged). Pair with "
              "--freeze-batchnorm-stats so BN running stats stay at base.")
```

**(e) `trainer.py` — BOTH `train_step(...)` call sites in the train loop.** There are exactly two (`_ret = train_step(` in the calibration branch and `... = train_step(` in the else branch). Add to EACH call's keyword arguments (e.g. after `value_grad_max_norm=value_grad_max_norm,`):

```python
                                train_value_head_only=train_value_head_only,
```

(Indentation must match each call site's existing kwargs; the else-branch call is indented differently from the calibration-branch call.)

**(f) `trainer.py` — checkpoint JSON.** Locate:

```python
            "freeze_batchnorm_stats": freeze_batchnorm_stats,
```

and add directly below it:

```python
            # v8: whether encoder/policy updates were skipped (value-head-only run).
            "train_value_head_only": train_value_head_only,
```

**(g) `train.py` — argparse flag.** Locate `parser.add_argument("--freeze-batchnorm-stats", action="store_true",` and add a sibling flag after that argument's full definition:

```python
    parser.add_argument("--train-value-head-only", action="store_true",
                        help="v8: freeze encoder+policy_head (skip opt_main updates); "
                             "only value_head.* tensors train. Pair with "
                             "--freeze-batchnorm-stats.")
```

**(h) `train.py` — plumb.** Locate `freeze_batchnorm_stats=args.freeze_batchnorm_stats,` in the `train(...)` call and add directly below it:

```python
        train_value_head_only=args.train_value_head_only,
```

- [ ] **Step 4: Run the new tests plus the trainer regression surface**

Run: `.venv/bin/python -m pytest tests/test_train_value_head_only.py tests/test_calibration_loss.py tests/test_training.py tests/test_trainer_teacher_mode_gate.py tests/test_value_calibration_sampling.py tests/test_calibration_cli_flags.py -v`
Expected: ALL PASS (pre-existing files unmodified).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/train.py tests/test_train_value_head_only.py
git commit -m "feat(training): --train-value-head-only — skip opt_main updates so only value_head.* tensors train (v8)"
```

---

### Task 2: checkpoint tensor-diff verifier CLI

**Files:**
- Create: `scripts/GPU/alphazero/verify_value_head_only_checkpoint.py`
- Test: `tests/test_verify_value_head_only_checkpoint.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1 (fully independent — compares two safetensors files; `network.save_weights` writes flat dotted keys that `mx.load` returns as a dict).
- Produces: `compare_value_head_only(base_path, candidate_path, prefix="value_head.") -> dict` with keys `frozen_diffs: list[str]`, `value_deltas: dict[str, float]` (max-abs delta per prefix tensor), `n_tensors: int`; CLI `python -m scripts.GPU.alphazero.verify_value_head_only_checkpoint --base ... --candidate ...` — exit 0 = PASS, exit 1 = a non-prefix tensor changed, exit 2 = NO prefix tensor changed (training no-oped).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_value_head_only_checkpoint.py`:

```python
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_only_checkpoint import (
    compare_value_head_only, main)


def _save(net, path):
    net.save_weights(str(path))
    return str(path)


@pytest.fixture()
def base_and_net(tmp_path):
    net = create_network(hidden=64, n_blocks=2)
    return _save(net, tmp_path / "base.safetensors"), net


def _bump_value_head(net):
    net.value_head.fc2.weight = net.value_head.fc2.weight + 0.01


def _bump_encoder(net):
    net.encoder.conv1.weight = net.encoder.conv1.weight + 0.01


def test_value_head_only_change_passes(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_only(base, cand)
    assert report["frozen_diffs"] == []
    assert max(report["value_deltas"].values()) > 0
    assert report["n_tensors"] == 44           # 64x2 arch: full flat tensor count
    assert main(["--base", base, "--candidate", cand]) == 0


def test_encoder_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    _bump_encoder(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_only(base, cand)
    assert "encoder.conv1.weight" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_identical_checkpoints_exit_2(tmp_path, base_and_net):
    base, net = base_and_net
    cand = _save(net, tmp_path / "cand.safetensors")
    assert main(["--base", base, "--candidate", cand]) == 2


def test_key_set_mismatch_raises(tmp_path, base_and_net):
    base, _ = base_and_net
    other = create_network(hidden=64, n_blocks=4)      # different key set
    cand = _save(other, tmp_path / "cand.safetensors")
    with pytest.raises(ValueError, match="key"):
        compare_value_head_only(base, cand)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_verify_value_head_only_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: ...verify_value_head_only_checkpoint`

- [ ] **Step 3: Implement `scripts/GPU/alphazero/verify_value_head_only_checkpoint.py`**

```python
"""v8 acceptance check: prove a --train-value-head-only run touched ONLY the
value head.

Compares two safetensors checkpoints tensor-by-tensor (network.save_weights
writes flat dotted keys; includes BatchNorm running stats, which must also be
byte-identical under --freeze-batchnorm-stats):
  exit 0  PASS: every non-value_head tensor byte-identical, value head changed
  exit 1  FAIL: some tensor outside the prefix changed (leak — run is invalid)
  exit 2  FAIL: NO value_head tensor changed (training no-oped)
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx


def compare_value_head_only(base_path: str, candidate_path: str,
                            prefix: str = "value_head.") -> dict:
    base = mx.load(str(base_path))
    cand = mx.load(str(candidate_path))
    if set(base) != set(cand):
        only_b = sorted(set(base) - set(cand))
        only_c = sorted(set(cand) - set(base))
        raise ValueError(
            f"checkpoint key sets differ (base-only {only_b[:3]}, "
            f"candidate-only {only_c[:3]}) — not the same architecture")
    frozen_diffs, value_deltas = [], {}
    for k in sorted(base):
        if k.startswith(prefix):
            delta = mx.abs(cand[k].astype(mx.float32)
                           - base[k].astype(mx.float32))
            value_deltas[k] = float(delta.max().item()) if delta.size else 0.0
        elif not bool(mx.array_equal(base[k], cand[k]).item()):
            frozen_diffs.append(k)
    return {"frozen_diffs": frozen_diffs, "value_deltas": value_deltas,
            "n_tensors": len(base)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a --train-value-head-only checkpoint changed ONLY "
                    "value_head.* tensors vs its base.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--prefix", default="value_head.")
    args = ap.parse_args(argv)
    report = compare_value_head_only(args.base, args.candidate, args.prefix)
    for k, d in sorted(report["value_deltas"].items()):
        print(f"{k}: max|delta| = {d:.3e}")
    if report["frozen_diffs"]:
        print(f"FAIL: {len(report['frozen_diffs'])} tensor(s) outside "
              f"{args.prefix!r} changed:")
        for k in report["frozen_diffs"]:
            print(f"  LEAK: {k}")
        return 1
    if not report["value_deltas"] or max(report["value_deltas"].values()) == 0.0:
        print(f"FAIL: no {args.prefix!r} tensor changed — training no-oped")
        return 2
    print(f"PASS: {report['n_tensors']} tensors; all non-{args.prefix!r} "
          f"byte-identical; value head trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_verify_value_head_only_checkpoint.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/verify_value_head_only_checkpoint.py tests/test_verify_value_head_only_checkpoint.py
git commit -m "feat(training): verify_value_head_only_checkpoint — tensor-diff acceptance check for v8 runs"
```

---

### Task 3: Full-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: 1281 + new tests passed with EXACTLY the known 14 failed + 6 errors (missing gitignored local game-log data). Authoritative check (1309 + new, 0 failures) happens on merged main before push. Do NOT merge red.

- [ ] **Step 2: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch).

---

## Operator run (USER's, after merge — from the locked v8 experiment definition)

1. **Train** — the v7 command with the new checkpoint dir and BOTH flags:
   `--checkpoint-dir checkpoints/alphazero-v8-value-head-only-v7-manifest-from-calib020-0001`, same v7 manifest (`logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`), same schedule (`black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`), weight 0.01, `--freeze-batchnorm-stats --train-value-head-only`.
2. **Telemetry** (checkpoint JSON): `train_value_head_only=True`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_n_drawn_per_step=8.0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`, `n_teacher_retention_drawn=0` (expected — policy-mask-derived).
3. **Tensor-diff acceptance** (the REAL proof):
   `.venv/bin/python -m scripts.GPU.alphazero.verify_value_head_only_checkpoint --base checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors --candidate checkpoints/alphazero-v8-value-head-only-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`
   Must exit 0: all non-`value_head.*` tensors (including BN running stats) byte-identical; the 4 value-head tensors changed.
4. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v8_value_head_only_v7_manifest_from_calib020_0001_gates_400s`. Pass/fail unchanged: A mean ≤ 0.0 and severe materially below 43.3%; B severe = 0.0 and over ≤ 11.1%; C severe ≤ 13.3%, over ≤ 33.3%, mean ≤ +0.099; D severe = 0.0 and mean ≤ 0.0. No promotion match unless all four pass.
5. **Ledger update** (v8 row; interpretation: pass ⇒ v6/v7 failures were representation drift; fail ⇒ even a pure value-head fit breaks the guardrails, pointing at the value head itself).
