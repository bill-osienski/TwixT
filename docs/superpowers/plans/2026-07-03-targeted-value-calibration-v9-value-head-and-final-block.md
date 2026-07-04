# Targeted Value Calibration v9 — `--train-value-head-and-final-block` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--train-value-head-and-final-block` training flag that trains only `value_head.*` plus the final residual block `encoder.blocks[last]` (partial unfreeze), plus a strict tensor-diff verifier that proves it, so we can test whether the smallest late representation slice moves gate A while preserving B/C/D.

**Architecture:** The trainer computes one gradient tree and applies it with two optimizers (`opt_main` for encoder+policy, `opt_value` for the value head). v8 added a guard skipping `opt_main.update` entirely. v9 extends that guard: when the new flag is set, skip the whole-trunk `opt_main.update`, then make one extra `opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])` — a single update on the live final-block submodule (no Adam double-step, trunk learning rate, magnitude-preserving because clipping stays global over the full trunk). `opt_value.update(network.value_head, value_grads)` always runs. NO MLX `freeze()`/`unfreeze()`. The two flags are mutually exclusive.

**Tech Stack:** Python 3.14 / MLX (Apple Metal — no `requires_grad`, no `named_parameters()`; params are nested dicts/lists via `tree_flatten`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block-design.md` (APPROVED — user confirmed both flagged decisions; do not redesign).

## Global Constraints

- Python: always `.venv/bin/python` from the repo root; tests: `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main: **1318 passed**.
- NEVER `sys.modules.pop("mlx")` (or any mlx submodule) in tests — native re-import SIGABRTs the suite.
- **This is MLX, not PyTorch:** no `requires_grad`, no `named_parameters()`, no `freeze()`/`unfreeze()`. The mechanism is ONLY: skip the whole-trunk `opt_main.update(main_module, main_grads)` and instead call `opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])` once. `opt_value.update(network.value_head, value_grads)` always runs. `alphazero_loss_batch` is untouched — no loss-math change of any kind; the 7/10/14-tuple return arities are byte-identical.
- **Byte-identical when unused:** both flags absent/False → `train_step` behavior, return arities, and all pre-existing tests pass UNMODIFIED.
- **The two flags are mutually exclusive:** enforce in BOTH places — a `ValueError` at the top of `train_step` (unit-testable) AND a `parser.error(...)` in `train.py` after `parse_args()` (CLI UX).
- The value head is exactly 4 tensors (`value_head.fc1.weight/bias`, `value_head.fc2.weight/bias`). The final residual block `encoder.blocks.<last>` has exactly 8 trainable tensors (`conv1.weight/bias`, `conv2.weight/bias`, `bn1.weight/bias`, `bn2.weight/bias`) plus 4 BN running stats (`bn1/bn2.running_mean/running_var`). The verifier allows the 4 value-head + 8 final-block trainable tensors to change; ALL BN running stats everywhere (incl. the final block's) must stay byte-identical. Do NOT pin the total tensor count (`n_tensors == 92`); assert `len(value_head_deltas) == 4` and `len(final_block_deltas) == 8`.
- The final block index is dynamic: `last = len(network.encoder.blocks) - 1` (production `n_blocks=6` → 5; tests use `n_blocks=2` → 1). The verifier auto-detects it as `max n` over `encoder.blocks.<n>.*` keys.
- Do NOT touch: `alphazero_loss_batch`, `calibration_pool.py`, `mcts.py`, `continuation_extraction.py`, any builder/smoke, any manifest/checkpoint, `docs/post-game-analysis.md`, and the v8 verifier `verify_value_head_only_checkpoint.py`.
- One feature branch `feature/tvc-v9-value-head-and-final-block` (worktree; a fresh worktree lacks gitignored local game-log data → known 14F+6E in the whole-repo suite there; judge tasks on file-scoped runs, authoritative suite on merged main).
- Per-task commits, FF-merge to main (no `--no-ff`, never force-push). Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/trainer.py` (modify: `train_step` mutual-exclusion guard + signature + three-way update branch; `train()` signature; startup print; 2 call sites; checkpoint JSON) | the flag's behavior |
| `scripts/GPU/alphazero/train.py` (modify: argparse flag + mutual-exclusion `parser.error` + 1 plumb line) | CLI surface |
| `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py` (create) | post-train tensor-diff verifier CLI |
| `tests/test_train_value_head_and_final_block.py` (create) | Task 1 tests |
| `tests/test_verify_value_head_and_final_block_checkpoint.py` (create) | Task 2 tests |

---

### Task 1: `--train-value-head-and-final-block` end-to-end (train_step branch + mutual-exclusion + plumbing + telemetry)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (6 sites: `train_step` signature; mutual-exclusion guard at top of `train_step`; the three-way update branch; `train()` signature; startup print; the TWO `train_step(...)` call sites; checkpoint-JSON fields)
- Modify: `scripts/GPU/alphazero/train.py` (argparse flag + mutual-exclusion `parser.error` + plumb into the `train(...)` call)
- Test: `tests/test_train_value_head_and_final_block.py` (create)

**Interfaces:**
- Consumes: existing `train_step(network, main_module, opt_main, opt_value, batch, ..., train_value_head_only=False)`, `MainModule`, `create_network`, `freeze_batchnorm_running_stats`, `PositionRecord` (trainer.py / network.py / self_play.py). The already-clipped `main_grads` is a dict `{"encoder": {..., "blocks": [<per-block grad dict>, ...]}, "policy_head": {...}}`; `main_grads["encoder"]["blocks"][last]` is the final block's grad subtree, co-structured with `network.encoder.blocks[last].trainable_parameters()`.
- Produces: `train_step(..., train_value_head_and_final_block: bool = False)`; `train(..., train_value_head_and_final_block: bool = False)`; CLI `--train-value-head-and-final-block` (store_true); checkpoint JSON keys `"train_value_head_and_final_block"` and `"unfrozen_block_index"`. Task 2 is independent of these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_value_head_and_final_block.py`:

```python
"""--train-value-head-and-final-block (v9): opt_main applies ONLY the final
residual block's grads, so only value_head.* and encoder.blocks[last].*
trainable tensors change. Behavior tests run real train_step on a tiny net
(BN running stats frozen, as v9 runs use BOTH flags); train-loop wiring is
pinned source-level (precedent: tests/test_train_value_head_only.py)."""
import re

import numpy as np
import mlx.core as mx
import pytest
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
    net = create_network(hidden=64, n_blocks=2)   # last block = blocks.1
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    import mlx.optimizers as optim
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-3)
    return net, mm, opt_main, opt_value


def _params(net):
    return dict(tree_flatten(net.parameters()))


def _changed_keys(before, after):
    return sorted(k for k in before
                  if not bool(mx.array_equal(before[k], after[k]).item()))


def test_flag_on_only_value_head_and_final_block_change():
    net, mm, opt_main, opt_value = _setup()
    last = len(net.encoder.blocks) - 1              # 1
    before = _params(net)
    for _ in range(2):
        out = train_step(network=net, main_module=mm, opt_main=opt_main,
                         opt_value=opt_value, batch=[_pos() for _ in range(3)],
                         train_value_head_and_final_block=True)
    assert len(out) == 7                            # arity unchanged
    changed = _changed_keys(before, _params(net))
    assert changed, "value head + final block must train"
    allowed = ("value_head.", f"encoder.blocks.{last}.")
    assert all(k.startswith(allowed) for k in changed), changed
    assert any(k.startswith("value_head.") for k in changed)
    assert any(k.startswith(f"encoder.blocks.{last}.") for k in changed)
    # earlier trunk + policy head frozen
    assert not any(k.startswith("encoder.blocks.0.") for k in changed)
    assert not any(k.startswith("encoder.conv1.") for k in changed)
    assert not any(k.startswith("encoder.bn1.") for k in changed)
    assert not any(k.startswith("policy_head.") for k in changed)
    # no BN running stats moved anywhere (incl. the final block's)
    assert not any(k.endswith(".running_mean") or k.endswith(".running_var")
                   for k in changed), changed


def test_flag_off_default_trains_everything():
    net, mm, opt_main, opt_value = _setup()
    before = _params(net)
    for _ in range(2):
        train_step(network=net, main_module=mm, opt_main=opt_main,
                   opt_value=opt_value, batch=[_pos() for _ in range(3)])
    changed = _changed_keys(before, _params(net))
    assert any(k.startswith("encoder.blocks.0.") for k in changed)
    assert any(k.startswith("policy_head.") for k in changed)
    assert any(k.startswith("value_head.") for k in changed)


def test_v9_and_v8_mutually_exclusive():
    net, mm, opt_main, opt_value = _setup()
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(network=net, main_module=mm, opt_main=opt_main,
                   opt_value=opt_value, batch=[_pos() for _ in range(3)],
                   train_value_head_only=True,
                   train_value_head_and_final_block=True)


def test_flag_on_with_calibration_batch_keeps_14_tuple():
    """v9 trains WITH the v7 calibration manifest: the masked teacher-mode
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
                     train_value_head_and_final_block=True)
    assert len(out) == 14


def test_train_loop_wiring_source_level():
    src = open(trainer_mod.__file__).read()
    assert len(re.findall(
        r"train_value_head_and_final_block=train_value_head_and_final_block,",
        src)) == 2
    assert ('"train_value_head_and_final_block": '
            'train_value_head_and_final_block,') in src
    assert '"unfrozen_block_index":' in src


def test_cli_flag_exists_plumbs_and_mutually_exclusive():
    src = open(train_mod.__file__).read()
    assert '"--train-value-head-and-final-block"' in src
    assert ("train_value_head_and_final_block="
            "args.train_value_head_and_final_block,") in src
    # argparse-level mutual exclusion
    assert ("args.train_value_head_only and "
            "args.train_value_head_and_final_block") in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_value_head_and_final_block.py -v`
Expected: `test_flag_on_only_value_head_and_final_block_change`, `test_v9_and_v8_mutually_exclusive`, `test_flag_on_with_calibration_batch_keeps_14_tuple` FAIL with `TypeError: train_step() got an unexpected keyword argument 'train_value_head_and_final_block'`; the two source-level tests FAIL on the missing strings; `test_flag_off_default_trains_everything` PASSES already (it pins today's behavior — expected at RED).

- [ ] **Step 3: Implement**

**(a) `trainer.py` — `train_step` signature.** Locate the parameter `train_value_head_only: bool = False,              # v8: skip opt_main.update` in the `train_step` signature and add directly below it:

```python
    train_value_head_only: bool = False,              # v8: skip opt_main.update
    train_value_head_and_final_block: bool = False,   # v9: only value head + final block
```

**(b) `trainer.py` — mutual-exclusion guard (fail fast).** Locate the first executable line of `train_step`'s body, `    calib_active = (`, and insert the guard directly above it:

```python
    if train_value_head_only and train_value_head_and_final_block:
        raise ValueError(
            "train_value_head_only and train_value_head_and_final_block are "
            "mutually exclusive")
    calib_active = (
```

**(c) `trainer.py` — the three-way update branch.** Locate the update block:

```python
    # Update REAL modules (guaranteed to mutate network)
    if not train_value_head_only:
        opt_main.update(main_module, main_grads)
    # v8: with train_value_head_only, encoder+policy grads are computed and
    # clipped (telemetry unchanged) but never applied — only the value head
    # trains. BN running stats need --freeze-batchnorm-stats separately.
    opt_value.update(network.value_head, value_grads)
```

Replace it with:

```python
    # Update REAL modules (guaranteed to mutate network)
    if train_value_head_only:
        # v8: encoder+policy grads are computed and clipped (telemetry
        # unchanged) but never applied — only the value head trains.
        pass
    elif train_value_head_and_final_block:
        # v9: apply ONLY the final residual block's (already-clipped) grads.
        # One opt_main.update on the live block submodule — no Adam double-step,
        # trunk learning rate, and structurally cannot touch policy_head, the
        # stem, or earlier blocks (they are never passed to any optimizer).
        last = len(network.encoder.blocks) - 1
        opt_main.update(network.encoder.blocks[last],
                        main_grads["encoder"]["blocks"][last])
    else:
        opt_main.update(main_module, main_grads)
    # BN running stats need --freeze-batchnorm-stats separately (v8/v9 pair the flags).
    opt_value.update(network.value_head, value_grads)
```

**(d) `trainer.py` — `train()` signature.** Locate `    train_value_head_only: bool = False,` in the `def train(` parameter list (the one whose next line is `) -> AlphaZeroNetwork:`) and add directly below it:

```python
    train_value_head_only: bool = False,
    train_value_head_and_final_block: bool = False,
```

**(e) `trainer.py` — startup print.** Locate the v8 startup print block:

```python
    if train_value_head_only:
        print("TRAIN VALUE HEAD ONLY: encoder+policy_head updates DISABLED "
              "(opt_main.update skipped; value head lr unchanged). Pair with "
              "--freeze-batchnorm-stats so BN running stats stay at base.")
```

and add directly after it:

```python
    if train_value_head_and_final_block:
        _vh_last = len(network.encoder.blocks) - 1
        print(f"TRAIN VALUE HEAD + FINAL BLOCK: only value_head.* and "
              f"encoder.blocks.{_vh_last}.* train (opt_main applies just the "
              f"final block; policy_head + earlier trunk frozen). Pair with "
              f"--freeze-batchnorm-stats so BN running stats stay at base.")
```

**(f) `trainer.py` — BOTH `train_step(...)` call sites.** There are exactly two lines reading `                                train_value_head_only=train_value_head_only,` (the calibration-branch call and the else-branch call). Add directly below EACH:

```python
                                train_value_head_only=train_value_head_only,
                                train_value_head_and_final_block=train_value_head_and_final_block,
```

(Both existing lines share the same indentation; match it.)

**(g) `trainer.py` — checkpoint JSON.** Locate:

```python
            # v8: whether encoder/policy updates were skipped (value-head-only run).
            "train_value_head_only": train_value_head_only,
```

and add directly below it:

```python
            # v9: whether only value head + final residual block trained, and
            # which block index was unfrozen (null when the flag is off).
            "train_value_head_and_final_block": train_value_head_and_final_block,
            "unfrozen_block_index": (
                len(network.encoder.blocks) - 1
                if train_value_head_and_final_block else None),
```

**(h) `train.py` — argparse flag.** Locate the `--train-value-head-only` argument definition (ends with `"--freeze-batchnorm-stats.")`) and add a sibling flag after it:

```python
    parser.add_argument("--train-value-head-and-final-block", action="store_true",
        help="v9: train only value_head.* plus the final residual block "
             "encoder.blocks[last] (skip the whole-trunk opt_main update; "
             "apply just the final block). Mutually exclusive with "
             "--train-value-head-only. Pair with --freeze-batchnorm-stats.")
```

**(i) `train.py` — mutual-exclusion check.** Locate `    _validate_closeout_td1_args(parser, args)` in `main()` and add directly below it:

```python
    if args.train_value_head_only and args.train_value_head_and_final_block:
        parser.error("--train-value-head-only and "
                     "--train-value-head-and-final-block are mutually exclusive")
```

**(j) `train.py` — plumb.** Locate `        train_value_head_only=args.train_value_head_only,` in the `train(...)` call and add directly below it:

```python
        train_value_head_and_final_block=args.train_value_head_and_final_block,
```

- [ ] **Step 4: Run the new tests plus the trainer regression surface**

Run: `.venv/bin/python -m pytest tests/test_train_value_head_and_final_block.py tests/test_train_value_head_only.py tests/test_calibration_loss.py tests/test_training.py tests/test_trainer_teacher_mode_gate.py tests/test_value_calibration_sampling.py tests/test_calibration_cli_flags.py -v`
Expected: ALL PASS (pre-existing files unmodified — `test_train_value_head_only.py` confirms the v8 path still works after the update-block rewrite).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/train.py tests/test_train_value_head_and_final_block.py
git commit -m "feat(training): --train-value-head-and-final-block — train only value head + final residual block (v9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: strict checkpoint tensor-diff verifier CLI

**Files:**
- Create: `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py`
- Test: `tests/test_verify_value_head_and_final_block_checkpoint.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1 (fully independent — compares two safetensors files; `network.save_weights` writes flat dotted keys that `mx.load` returns as a dict; `create_network` from network.py for the tests).
- Produces: `compare_value_head_and_final_block(base_path, candidate_path, last_block_index=None) -> dict` with keys `frozen_diffs: list[str]`, `value_head_deltas: dict[str, float]`, `final_block_deltas: dict[str, float]`, `last_block_index: int`, `n_tensors: int`; CLI `python -m scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint --base ... --candidate ...` — exit 0 = PASS, exit 1 = a frozen tensor changed (leak), exit 2 = no value-head tensor changed (no-op), exit 3 = value head changed but no final-block tensor changed (partial unfreeze never engaged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_value_head_and_final_block_checkpoint.py`:

```python
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint import (
    compare_value_head_and_final_block, main)


def _save(net, path):
    net.save_weights(str(path))
    return str(path)


@pytest.fixture()
def base_and_net(tmp_path):
    net = create_network(hidden=64, n_blocks=2)     # last block = blocks.1
    return _save(net, tmp_path / "base.safetensors"), net


def _bump_value_head(net):
    net.value_head.fc2.weight = net.value_head.fc2.weight + 0.01


def _bump_final_block(net):
    last = len(net.encoder.blocks) - 1
    b = net.encoder.blocks[last]
    b.conv1.weight = b.conv1.weight + 0.01


def _bump_early_block(net):
    net.encoder.blocks[0].conv1.weight = net.encoder.blocks[0].conv1.weight + 0.01


def _bump_policy(net):
    net.policy_head.conv.weight = net.policy_head.conv.weight + 0.01


def _bump_final_block_running_stat(net):
    last = len(net.encoder.blocks) - 1
    b = net.encoder.blocks[last]
    b.bn1.running_mean = b.bn1.running_mean + 0.01


def test_value_head_and_final_block_change_passes(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    _bump_final_block(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert report["frozen_diffs"] == []
    assert max(report["value_head_deltas"].values()) > 0
    assert max(report["final_block_deltas"].values()) > 0
    # 4 value-head tensors + 8 final-block trainable tensors (running stats
    # excluded) — the properties that matter; do NOT pin the total count.
    assert len(report["value_head_deltas"]) == 4
    assert len(report["final_block_deltas"]) == 8
    assert report["last_block_index"] == 1
    assert report["n_tensors"] > 0
    assert main(["--base", base, "--candidate", cand]) == 0


def test_early_block_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net); _bump_early_block(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert "encoder.blocks.0.conv1.weight" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_policy_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net); _bump_policy(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert any(k.startswith("policy_head.") for k in report["frozen_diffs"])
    assert main(["--base", base, "--candidate", cand]) == 1


def test_final_block_running_stat_change_fails_exit_1(tmp_path, base_and_net):
    """A forgotten --freeze-batchnorm-stats moves the final block's running
    stats; those are NOT in the allowed set, so they must leak → exit 1."""
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net)
    _bump_final_block_running_stat(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert "encoder.blocks.1.bn1.running_mean" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_identical_checkpoints_exit_2(tmp_path, base_and_net):
    base, net = base_and_net
    cand = _save(net, tmp_path / "cand.safetensors")
    assert main(["--base", base, "--candidate", cand]) == 2


def test_value_head_only_no_final_block_exit_3(tmp_path, base_and_net):
    """Value head moved but the final block did not — partial unfreeze never
    engaged (v9 collapsed to v8)."""
    base, net = base_and_net
    _bump_value_head(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert max(report["value_head_deltas"].values()) > 0
    assert max(report["final_block_deltas"].values()) == 0
    assert main(["--base", base, "--candidate", cand]) == 3


def test_key_set_mismatch_raises(tmp_path, base_and_net):
    base, _ = base_and_net
    other = create_network(hidden=64, n_blocks=4)   # different key set
    cand = _save(other, tmp_path / "cand.safetensors")
    with pytest.raises(ValueError, match="key"):
        compare_value_head_and_final_block(base, cand)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_verify_value_head_and_final_block_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: ...verify_value_head_and_final_block_checkpoint`

- [ ] **Step 3: Implement `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py`**

```python
"""v9 acceptance check: prove a --train-value-head-and-final-block run touched
ONLY the value head and the final residual block's trainable tensors.

Compares two safetensors checkpoints tensor-by-tensor. Allowed to change:
value_head.* (4 tensors) and encoder.blocks.<last>.* trainable tensors (8,
excluding BatchNorm running stats). Everything else — the stem, earlier
blocks, the policy head, and ALL BatchNorm running stats anywhere (including
the final block's) — must be byte-identical:
  exit 0  PASS: frozen set byte-identical; value head AND final block changed
  exit 1  FAIL: some frozen tensor changed (a running-stat leak means
          --freeze-batchnorm-stats was missing/ineffective — run is invalid)
  exit 2  FAIL: no value_head tensor changed (training no-oped)
  exit 3  FAIL: value head changed but no final-block tensor changed — the
          partial unfreeze never engaged (flag mis-plumbed; collapsed to v8)
"""
from __future__ import annotations

import argparse
import re
import sys

import mlx.core as mx


def _detect_last_block_index(keys) -> int:
    idxs = {int(m.group(1)) for k in keys
            if (m := re.match(r"encoder\.blocks\.(\d+)\.", k))}
    if not idxs:
        raise ValueError("no encoder.blocks.<n>.* tensors found — not an "
                         "AlphaZero encoder checkpoint")
    return max(idxs)


def _is_running_stat(key: str) -> bool:
    return key.endswith(".running_mean") or key.endswith(".running_var")


def compare_value_head_and_final_block(
        base_path: str, candidate_path: str,
        last_block_index: int | None = None) -> dict:
    base = mx.load(str(base_path))
    cand = mx.load(str(candidate_path))
    if set(base) != set(cand):
        only_b = sorted(set(base) - set(cand))
        only_c = sorted(set(cand) - set(base))
        raise ValueError(
            f"checkpoint key sets differ (base-only {only_b[:3]}, "
            f"candidate-only {only_c[:3]}) — not the same architecture")
    last = (_detect_last_block_index(base)
            if last_block_index is None else last_block_index)
    block_prefix = f"encoder.blocks.{last}."
    frozen_diffs, value_head_deltas, final_block_deltas = [], {}, {}
    for k in sorted(base):
        allowed_value = k.startswith("value_head.")
        allowed_block = k.startswith(block_prefix) and not _is_running_stat(k)
        if allowed_value or allowed_block:
            delta = mx.abs(cand[k].astype(mx.float32)
                           - base[k].astype(mx.float32))
            d = float(delta.max().item()) if delta.size else 0.0
            (value_head_deltas if allowed_value else final_block_deltas)[k] = d
        elif not bool(mx.array_equal(base[k], cand[k]).item()):
            frozen_diffs.append(k)
    return {"frozen_diffs": frozen_diffs,
            "value_head_deltas": value_head_deltas,
            "final_block_deltas": final_block_deltas,
            "last_block_index": last, "n_tensors": len(base)}


def _changed(deltas: dict) -> bool:
    return bool(deltas) and max(deltas.values()) > 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a --train-value-head-and-final-block checkpoint "
                    "changed ONLY value_head.* and the final residual block's "
                    "trainable tensors vs its base.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--last-block-index", type=int, default=None,
                    help="Override final-block index (default: auto-detect the "
                         "max encoder.blocks.<n> from the base checkpoint).")
    args = ap.parse_args(argv)
    report = compare_value_head_and_final_block(
        args.base, args.candidate, args.last_block_index)
    last = report["last_block_index"]
    for k, d in sorted(report["value_head_deltas"].items()):
        print(f"{k}: max|delta| = {d:.3e}")
    for k, d in sorted(report["final_block_deltas"].items()):
        print(f"{k}: max|delta| = {d:.3e}")
    if report["frozen_diffs"]:
        print(f"FAIL: {len(report['frozen_diffs'])} frozen tensor(s) changed "
              f"(allowed: value_head.* + encoder.blocks.{last}.* trainable):")
        for k in report["frozen_diffs"]:
            print(f"  LEAK: {k}")
        return 1
    if not _changed(report["value_head_deltas"]):
        print("FAIL: no value_head.* tensor changed — training no-oped")
        return 2
    if not _changed(report["final_block_deltas"]):
        print(f"FAIL: value head changed but no encoder.blocks.{last}.* tensor "
              f"changed — partial unfreeze never engaged (collapsed to v8)")
        return 3
    print(f"PASS: {report['n_tensors']} tensors; all frozen tensors "
          f"byte-identical; value head + final block (encoder.blocks.{last}) "
          f"trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_verify_value_head_and_final_block_checkpoint.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py tests/test_verify_value_head_and_final_block_checkpoint.py
git commit -m "feat(training): verify_value_head_and_final_block_checkpoint — strict tensor-diff acceptance check for v9 runs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Full-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored local game-log data). Authoritative check (1318 + new, 0 failures) happens on merged main before push. Do NOT merge red.

- [ ] **Step 2: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). Then hand the operator commands (below) back to the user. STOP after push — the operator run is the USER's.

---

## Operator run (USER's, after merge — from the locked v9 experiment definition)

1. **Train** — the v7 command with a new checkpoint dir and BOTH flags:
   `--checkpoint-dir checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001`, same v7 manifest (`logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`), same **v8 schedule** (`black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`), weight 0.01, `--freeze-batchnorm-stats --train-value-head-and-final-block`.
2. **Telemetry** (checkpoint JSON): `train_value_head_and_final_block=True`, `unfrozen_block_index=5`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_policy_ce_avg_iter=0.0`, `n_teacher_retention_drawn=0` (expected — policy-mask-derived).
3. **Tensor-diff acceptance** (the REAL proof), must exit 0:
   `.venv/bin/python -m scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint --base checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors --candidate checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`
   Exit 0 = all frozen tensors (every non-value_head/non-final-block tensor + ALL BN running stats incl. the final block's) byte-identical; the 4 value-head + 8 final-block tensors changed. Exit 3 specifically means the partial unfreeze never engaged.
4. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v9_value_head_and_final_block_v7_manifest_from_calib020_0001_gates_400s`. Thresholds unchanged: A mean ≤ 0.0 and severe materially below 43.3%; B severe = 0.0 and over ≤ 11.1%; C severe ≤ 13.3%, over ≤ 33.3%, mean ≤ +0.099; D severe = 0.0 and mean ≤ 0.0. No promotion match unless all four pass.
5. **Ledger update** (v9 row; interpretation: pass A + hold B/C/D ⇒ drift lives in the earlier trunk; hold B/C/D but miss A ⇒ one block too few → v9b last-2; move A but break B/C/D ⇒ partial unfreeze is a dead end).
