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
