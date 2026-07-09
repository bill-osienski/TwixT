#!/usr/bin/env python3
"""v11: surgical value-only B root-preservation clones.

Reads the v7 manifest, clones the two B goal_line root-retention rows into
depth-0 value-only `root_value` preservation rows (no policy CE, no hard
target), appends them, and writes the v11 manifest. Pure CSV copy: the raw
eval-mode `teacher_value` anchor already lives in the parent rows, and a
depth-0 root_value clone's `continuation_legal_moves_sha1` equals the parent's
`root_legal_moves_sha1` by construction (empty extra_moves keeps the state at
the root), so no board reconstruction or MCTS/inference is needed. The clones
validate at training load via calibration_pool (which reconstructs the root and
checks the sha) and train value-only (has_policy_target=False)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

BLOCKER_CASES = {
    "game_000015_ply_19",
    "game_000327_ply_63",
}
SOURCE_TAG = "goal_line_retention"
TARGET_TAG = "goal_line_root_value_retention"


def require_columns(fieldnames: list[str]) -> None:
    required = {
        "case_id",
        "tag",
        "loss_mode",
        "side_to_move",
        "teacher_value",
        "teacher_value_source",
        "target_black_value",
        "teacher_policy_json",
        "root_visits_json",
        "root_legal_moves_sha1",
        "continuation_source",
        "continuation_depth",
        "extra_moves_json",
        "continuation_side_to_move",
        "continuation_legal_moves_sha1",
        "continuation_parent_case_id",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"Missing required manifest columns: {missing}")


def make_root_value_clone(parent: dict[str, str]) -> dict[str, str]:
    if parent["tag"] != SOURCE_TAG:
        raise ValueError(f"Expected source tag {SOURCE_TAG}, got {parent['tag']}")
    if not parent.get("teacher_value"):
        raise ValueError(f"Parent {parent.get('case_id')} has blank teacher_value")
    if not parent.get("root_legal_moves_sha1"):
        raise ValueError(f"Parent {parent.get('case_id')} has blank root_legal_moves_sha1")
    row = dict(parent)
    row["case_id"] = f"{parent['case_id']}__root_value"
    row["tag"] = TARGET_TAG
    row["loss_mode"] = "searched_continuation_retention"
    row["continuation_source"] = "root_value"
    row["continuation_depth"] = "0"
    row["extra_moves_json"] = "[]"
    row["continuation_side_to_move"] = parent["side_to_move"]
    row["continuation_legal_moves_sha1"] = parent["root_legal_moves_sha1"]
    row["teacher_value_source"] = "base_raw_root_clone"
    # Link the clone back to its root row — matches the D emitter
    # (_root_value_row) and lets the coverage diagnostic join clone->parent.
    row["continuation_parent_case_id"] = parent["case_id"]
    # Make the clone value-only.
    row["target_black_value"] = ""
    row["teacher_policy_json"] = ""
    row["root_visits_json"] = ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv",
    )
    ap.add_argument(
        "--output",
        default="logs/eval/targeted_calibration_v11_surgical_root_value_from_v10_nearmiss.csv",
    )
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    with inp.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        require_columns(fieldnames)
        rows = list(reader)
    existing_ids = {r["case_id"] for r in rows}
    clones = []
    for case_id in sorted(BLOCKER_CASES):
        matches = [
            r for r in rows
            if r["case_id"] == case_id and r["tag"] == SOURCE_TAG
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"Expected exactly one {SOURCE_TAG} row for {case_id}, found {len(matches)}"
            )
        clone = make_root_value_clone(matches[0])
        if clone["case_id"] in existing_ids:
            raise SystemExit(f"Clone already exists: {clone['case_id']}")
        clones.append(clone)
    new_rows = rows + clones
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    counts = Counter(r["tag"] for r in new_rows)
    print(f"Wrote {out}")
    print(f"Input rows:  {len(rows)}")
    print(f"Output rows: {len(new_rows)}")
    print(f"Added rows:  {len(clones)}")
    print(f"{TARGET_TAG}: {counts[TARGET_TAG]}")
    for c in clones:
        print(
            "clone",
            c["case_id"],
            "teacher_value=",
            c["teacher_value"],
            "sha=",
            c["continuation_legal_moves_sha1"],
        )


if __name__ == "__main__":
    main()
