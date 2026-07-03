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
