#!/usr/bin/env python3
"""v12b: build the continuation guardrail manifest.

Output = every hard_value row from the v7 manifest (A correction + severe-D,
kept) + one value-only guardrail clone per B/C/D ROOT retention row (sign from
side_to_move, continuation fields blanked) + one value-only guardrail clone per
C/D CONTINUATION retention row (sign from continuation_side_to_move, continuation
reconstruction fields PRESERVED so the loader rebuilds the searched state). Pure
copy + arithmetic — the BASE anchor already lives in each parent's teacher_value.

Dropped (not cloned, not kept): the source root/continuation rows themselves,
goal_line_continuation_retention (B stays root-only), and
red_predrop_root_value_retention (D root already covered by the red_predrop root
guardrail)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT_TO_GUARDRAIL_TAG = {
    "goal_line_retention": "goal_line_guardrail_retention",
    "old_post_opening_retention": "old_post_opening_guardrail_retention",
    "red_predrop_retention": "red_predrop_guardrail_retention",
}
CONTINUATION_TO_GUARDRAIL_TAG = {
    "old_post_opening_continuation_retention":
        "old_post_opening_continuation_guardrail_retention",
    "red_predrop_continuation_retention":
        "red_predrop_continuation_guardrail_retention",
}

# policy/root/search-metadata columns blanked on EVERY guardrail clone (value-only)
_POLICY_ROOT_BLANK = (
    "teacher_policy_json", "teacher_legal_moves_sha1",
    "root_visits_json", "root_legal_moves_sha1",
    "root_value_stm", "root_black_value", "root_sims", "root_base_checkpoint",
    "root_seed", "root_mcts_eval_batch_size", "root_mcts_stall_flush_sims",
    "continuation_tree_visits", "continuation_tree_nn_value",
)
# continuation reconstruction/identity columns — blanked on ROOT clones only
_CONTINUATION_COLS = (
    "extra_moves_json", "continuation_side_to_move", "continuation_legal_moves_sha1",
    "continuation_depth", "continuation_parent_case_id", "continuation_source",
    "continuation_path_moves",
)


def _blank(row: dict, cols) -> None:
    for col in cols:
        if col in row:
            row[col] = ""


def _clone_base(parent: dict, new_tag: str, sign: float) -> dict:
    if parent.get("teacher_value") in (None, ""):
        raise ValueError(f"{parent.get('case_id')}: parent lacks teacher_value")
    row = dict(parent)
    row["case_id"] = f"{parent['case_id']}__guardrail"
    row["tag"] = new_tag
    row["loss_mode"] = "asymmetric_guardrail_retention"
    row["target_black_value"] = repr(float(parent["teacher_value"]) * sign)
    _blank(row, _POLICY_ROOT_BLANK)
    return row


def make_root_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in ROOT_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a B/C/D root tag: {tag}")
    sign = 1.0 if parent["side_to_move"] == "black" else -1.0
    row = _clone_base(parent, ROOT_TO_GUARDRAIL_TAG[tag], sign)
    _blank(row, _CONTINUATION_COLS)          # root clones carry no continuation state
    return row


def make_continuation_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in CONTINUATION_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a C/D continuation tag: {tag}")
    if parent.get("extra_moves_json") in (None, ""):
        raise ValueError(
            f"{parent.get('case_id')}: continuation guardrail clone requires "
            "non-empty extra_moves_json (it must reconstruct a continuation state; "
            "depth-0/root_value rows do not belong here)")
    side = parent.get("continuation_side_to_move")
    if side in (None, ""):
        raise ValueError(
            f"{parent.get('case_id')}: continuation row lacks continuation_side_to_move")
    sign = 1.0 if side == "black" else -1.0    # CONTINUATION side, not root side
    # _clone_base blanks policy/root cols but leaves _CONTINUATION_COLS intact
    return _clone_base(parent, CONTINUATION_TO_GUARDRAIL_TAG[tag], sign)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv")
    ap.add_argument("--output",
                    default="logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv")
    args = ap.parse_args()
    with Path(args.input).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows if (r.get("loss_mode") or "hard_value") == "hard_value"]
    root_clones, cont_clones = [], []
    for r in rows:
        mode = r.get("loss_mode") or "hard_value"
        tag = r.get("tag")
        if mode == "mcts_root_retention" and tag in ROOT_TO_GUARDRAIL_TAG:
            root_clones.append(make_root_guardrail_clone(r))
        elif (mode == "searched_continuation_retention"
              and tag in CONTINUATION_TO_GUARDRAIL_TAG):
            cont_clones.append(make_continuation_guardrail_clone(r))
        # else dropped: source root/cont rows, goal_line_continuation, red_predrop_root_value
    out_rows = kept + root_clones + cont_clones
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    counts = Counter(r["tag"] for r in out_rows)
    print(f"Wrote {args.output}: {len(out_rows)} rows "
          f"({len(kept)} hard_value kept, {len(root_clones)} root guardrail, "
          f"{len(cont_clones)} continuation guardrail)")
    for t in sorted(list(ROOT_TO_GUARDRAIL_TAG.values())
                    + list(CONTINUATION_TO_GUARDRAIL_TAG.values())):
        print(f"  {t}: {counts.get(t, 0)}")


if __name__ == "__main__":
    main()
