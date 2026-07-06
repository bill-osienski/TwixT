#!/usr/bin/env python3
"""v12: build the guardrail manifest.

Output = every hard_value row from the v7 manifest (A correction + severe-D,
kept for provenance though only A is scheduled) + one value-only guardrail
clone per B/C/D root-retention (`mcts_root_retention`) row. Each clone's
target_black_value is the parent's stm teacher_value converted to black
perspective (× sign). Pure copy + arithmetic — no reconstruction/MCTS; the
BASE anchor already lives in the parent's teacher_value. The old
retention/continuation/root_value rows are dropped (guardrail replaces
symmetric retention for B/C/D)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

SOURCE_TO_GUARDRAIL_TAG = {
    "goal_line_retention": "goal_line_guardrail_retention",
    "old_post_opening_retention": "old_post_opening_guardrail_retention",
    "red_predrop_retention": "red_predrop_guardrail_retention",
}


def make_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in SOURCE_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a B/C/D root tag: {tag}")
    tv = parent.get("teacher_value")
    if tv in (None, ""):
        raise ValueError(f"{parent.get('case_id')}: parent lacks teacher_value")
    sign = 1.0 if parent["side_to_move"] == "black" else -1.0
    target_black = float(tv) * sign
    row = dict(parent)
    row["case_id"] = f"{parent['case_id']}__guardrail"
    row["tag"] = SOURCE_TO_GUARDRAIL_TAG[tag]
    row["loss_mode"] = "asymmetric_guardrail_retention"
    row["target_black_value"] = repr(target_black)
    # value-only: blank every policy/root/continuation field
    for col in ("teacher_policy_json", "teacher_legal_moves_sha1",
                "root_visits_json", "root_legal_moves_sha1", "extra_moves_json",
                "continuation_side_to_move", "continuation_legal_moves_sha1",
                "continuation_depth", "continuation_parent_case_id",
                "continuation_source", "continuation_path_moves",
                "continuation_tree_visits", "continuation_tree_nn_value",
                "root_value_stm", "root_black_value", "root_sims",
                "root_base_checkpoint", "root_seed",
                "root_mcts_eval_batch_size", "root_mcts_stall_flush_sims"):
        if col in row:
            row[col] = ""
    # teacher_value kept as provenance (per spec)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv")
    ap.add_argument("--output",
                    default="logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv")
    args = ap.parse_args()
    with Path(args.input).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows if (r.get("loss_mode") or "hard_value") == "hard_value"]
    clones = [make_guardrail_clone(r) for r in rows
              if r.get("tag") in SOURCE_TO_GUARDRAIL_TAG]
    out_rows = kept + clones
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    counts = Counter(r["tag"] for r in out_rows)
    print(f"Wrote {args.output}: {len(out_rows)} rows "
          f"({len(kept)} hard_value kept, {len(clones)} guardrail clones)")
    for t in sorted(SOURCE_TO_GUARDRAIL_TAG.values()):
        print(f"  {t}: {counts.get(t, 0)}")


if __name__ == "__main__":
    main()
