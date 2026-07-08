"""v14 acceptance check: prove a --train-value-head-and-value-adapter run touched
ONLY value_head.* and the (new) value_adapter.* tensors.

Compares two safetensors checkpoints. The base has NO value_adapter.* keys; the
candidate adds exactly the value_adapter.* set. Allowed to change: value_head.*
and value_adapter.*. Everything the two checkpoints SHARE — the stem, ALL
residual blocks (including the final one), the policy head, and ALL BatchNorm
running stats anywhere — must be byte-identical:
  exit 0  PASS: shared frozen set byte-identical; value head AND the gate moved
  exit 1  FAIL: a shared frozen tensor changed (a running-stat leak means
          --freeze-batchnorm-stats was missing/ineffective — run is invalid)
  exit 2  FAIL: no value_head tensor changed, or the gate never left 0.0 — the
          value-path correction never engaged (no-op)
  exit 3  FAIL: candidate-only keys are not exactly value_adapter.*, the base has
          keys the candidate lacks, or the candidate has no value_adapter.* keys
          at all (wrong architecture / flag mis-plumbed)
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx

GATE_KEY = "value_adapter.gate"


def compare_value_head_and_adapter(base_path: str, candidate_path: str) -> dict:
    base = mx.load(str(base_path))
    cand = mx.load(str(candidate_path))
    base_keys, cand_keys = set(base), set(cand)
    new_keys = cand_keys - base_keys              # expected: exactly value_adapter.*
    missing = base_keys - cand_keys               # expected: empty
    unexpected_new = {k for k in new_keys if not k.startswith("value_adapter.")}
    adapter_keys = {k for k in new_keys if k.startswith("value_adapter.")}
    frozen_diffs, value_head_deltas, adapter_deltas = [], {}, {}
    for k in sorted(base_keys & cand_keys):
        if k.startswith("value_head."):
            delta = mx.abs(cand[k].astype(mx.float32) - base[k].astype(mx.float32))
            value_head_deltas[k] = float(delta.max().item()) if delta.size else 0.0
        elif not bool(mx.array_equal(base[k], cand[k]).item()):
            frozen_diffs.append(k)
    for k in sorted(adapter_keys):
        adapter_deltas[k] = float(mx.abs(cand[k].astype(mx.float32)).max().item())
    gate_abs = (float(mx.abs(cand[GATE_KEY]).max().item())
                if GATE_KEY in cand else 0.0)
    return {"frozen_diffs": frozen_diffs, "value_head_deltas": value_head_deltas,
            "adapter_deltas": adapter_deltas, "missing": sorted(missing),
            "unexpected_new": sorted(unexpected_new),
            "adapter_keys": sorted(adapter_keys), "gate_abs": gate_abs,
            "n_tensors": len(base_keys)}


def _changed(deltas: dict) -> bool:
    return bool(deltas) and max(deltas.values()) > 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a --train-value-head-and-value-adapter checkpoint "
                    "changed ONLY value_head.* and value_adapter.* vs its base.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args(argv)
    r = compare_value_head_and_adapter(args.base, args.candidate)
    for k, d in sorted(r["value_head_deltas"].items()):
        print(f"{k}: max|delta| = {d:.3e}")
    for k, d in sorted(r["adapter_deltas"].items()):
        print(f"{k}: max|abs| = {d:.3e}")
    if r["missing"] or r["unexpected_new"] or not r["adapter_keys"]:
        print(f"FAIL: unexpected key sets — missing-in-candidate={r['missing']}, "
              f"non-adapter-new={r['unexpected_new']}, "
              f"adapter-keys-present={bool(r['adapter_keys'])}")
        return 3
    if r["frozen_diffs"]:
        print(f"FAIL: {len(r['frozen_diffs'])} shared frozen tensor(s) changed "
              f"(allowed: value_head.* + value_adapter.*):")
        for k in r["frozen_diffs"]:
            print(f"  LEAK: {k}")
        return 1
    if not _changed(r["value_head_deltas"]) or r["gate_abs"] == 0.0:
        print(f"FAIL: value-path no-op (value_head changed="
              f"{_changed(r['value_head_deltas'])}, gate |abs|={r['gate_abs']:.3e}) "
              f"— the adapter correction never engaged")
        return 2
    print(f"PASS: {r['n_tensors']} base tensors; shared frozen set byte-identical; "
          f"value head + value_adapter (gate |abs|={r['gate_abs']:.3e}) trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
