"""Deterministic v4 teacher-cache builder: read the v3 stratified manifest, run
the teacher checkpoint's RAW forward (LocalGPUEvaluator.infer — NO MCTS) over each
retention row, and append teacher_value / teacher_policy_json /
teacher_legal_moves_sha1 + loss_mode. Correction rows pass through with blank
teacher columns. See docs/superpowers/specs/2026-06-29-...-v4-teacher-retention-design.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .position_probe_cases import load_csv_manifest
from .goal_line_trigger_probe_cases import position_state
from .calibration_pool import legal_moves_sha1

CORRECTION_TAG = "black_predrop_correction"
# v4 columns appended to whatever the source manifest already carries. We preserve
# ALL source columns (don't drop future metadata — analysis IDs, probe scores,
# diagnostics) and only add/override these four.
NEW_COLUMNS = ["loss_mode", "teacher_value", "teacher_policy_json", "teacher_legal_moves_sha1"]


def _teacher_infer(state, evaluator):
    """Single-position RAW forward → (priors over legal_moves, value). No MCTS."""
    legal = state.legal_moves()
    board_chw = evaluator.build_input_tensor(state)
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)[None, ...]
    n = len(legal)
    rows = np.zeros((1, n), dtype=np.int32)
    cols = np.zeros((1, n), dtype=np.int32)
    mask = np.ones((1, n), dtype=np.float32)
    for j, (r, c) in enumerate(legal):
        rows[0, j], cols[0, j] = r, c
    priors, values = evaluator.infer(board_hwc, rows, cols, mask, state.active_size)
    return legal, priors[0][:n].astype(float).tolist(), float(values[0])


def build_rows(rows: list, evaluator) -> list:
    out = []
    for r in rows:
        row = dict(r)                            # preserve ALL source columns
        if r.get("tag") == CORRECTION_TAG:
            # Correction rows: hard value target stays; teacher columns blank.
            # IMPORTANT: do NOT touch target_black_value here — blanking it would
            # destroy the A correction target (-0.35).
            row["loss_mode"] = "hard_value"
            row["teacher_value"] = ""
            row["teacher_policy_json"] = ""
            row["teacher_legal_moves_sha1"] = ""
            out.append(row)
            continue
        replay = json.loads(Path(r["replay_path"]).read_text())
        state = position_state(replay, int(float(r["position_ply"])), r["side_to_move"])
        legal, policy, value = _teacher_infer(state, evaluator)
        row["loss_mode"] = "teacher_retention"
        row["teacher_value"] = repr(value)
        row["teacher_policy_json"] = json.dumps(policy)
        row["teacher_legal_moves_sha1"] = legal_moves_sha1(legal)
        row["target_black_value"] = ""          # RETENTION ONLY: blank stale v3 MCTS-root scalar
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the v4 teacher calibration manifest.")
    ap.add_argument("--source", required=True, help="v3 stratified manifest CSV")
    ap.add_argument("--teacher-checkpoint", required=True, help=".safetensors teacher")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    network = load_network_for_scoring(args.teacher_checkpoint)
    evaluator = LocalGPUEvaluator(network)
    out_rows = build_rows(rows, evaluator)
    # Preserve source column order; append any v4 columns not already present.
    base_columns = list(rows[0].keys()) if rows else []
    fieldnames = base_columns + [c for c in NEW_COLUMNS if c not in base_columns]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    n_ret = sum(1 for r in out_rows if r["loss_mode"] == "teacher_retention")
    print(f"wrote {len(out_rows)} rows ({n_ret} teacher_retention) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
