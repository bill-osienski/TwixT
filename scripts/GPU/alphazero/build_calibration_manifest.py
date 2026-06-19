"""Deterministic builder for the post-opening sharp-drop calibration TRAIN
manifest.

Reads the loss-replay analyzer's review-queue CSV, keeps only black-loss
post-opening sharp-drop rows, excludes the frozen probe game_idx set, and
writes a manifest in the same schema the probe loader expects
(position_probe_cases.load_csv_manifest). The frozen 30 probe games are the
EVAL set and must never appear here (see design §4 invariant).

position_ply = drop_ply - 2  (black's decision point, two plies before the
collapse) — matches how the frozen probe manifest was derived.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

OUTPUT_COLUMNS = [
    "case_rank", "source_rank", "game_idx", "case_id", "replay_path",
    "position_ply", "drop_ply", "side_to_move", "a_color", "winner", "n_moves",
    "initial_a_value", "final_a_value", "largest_a_value_drop",
    "largest_drop_phase", "collapse_type",
]


def derive_case(row: dict, case_rank: int) -> dict:
    """Map one review-queue row to a probe-manifest case dict."""
    game_idx = int(row["game_idx"])
    drop_ply = int(float(row["largest_drop_ply"]))
    position_ply = drop_ply - 2
    return {
        "case_rank": case_rank,
        "source_rank": int(row["rank"]),
        "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": row["replay_path"],
        "position_ply": position_ply,
        "drop_ply": drop_ply,
        "side_to_move": row["a_color"],
        "a_color": row["a_color"],
        "winner": row["winner"],
        "n_moves": int(float(row["n_moves"])),
        "initial_a_value": row["initial_a_value"],
        "final_a_value": row["final_a_value"],
        "largest_a_value_drop": row["largest_a_value_drop"],
        "largest_drop_phase": row["largest_drop_phase"],
        "collapse_type": row["collapse_type"],
    }


def select_calibration_cases(queue_rows: list, holdout_game_idxs: set) -> list:
    """Filter to black-loss post-opening sharp-drop rows, drop the holdout,
    preserve the analyzer's (rank) order, and re-rank 1..N.

    Rows whose decision point would be negative (drop_ply < 2) are skipped.
    """
    kept = []
    for row in queue_rows:
        if row["collapse_type"] != "sharp_value_drop":
            continue
        if row["largest_drop_phase"] != "post_opening":
            continue
        if row["a_color"] != "black":
            continue
        if row["winner"] != "red":
            continue
        if int(row["game_idx"]) in holdout_game_idxs:
            continue
        if int(float(row["largest_drop_ply"])) - 2 < 0:
            continue
        kept.append(row)
    return [derive_case(r, i + 1) for i, r in enumerate(kept)]


def load_holdout_game_idxs(frozen_manifest_path) -> set:
    """Read the frozen probe manifest CSV; return its set of game_idx ints."""
    with Path(frozen_manifest_path).open(newline="") as f:
        return {int(r["game_idx"]) for r in csv.DictReader(f)}


def write_manifest(cases: list, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for case in cases:
            w.writerow({k: case[k] for k in OUTPUT_COLUMNS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the post-opening sharp-drop calibration train manifest."
    )
    p.add_argument("--queue", required=True,
                   help="analyzer manual_review_queue.csv (regenerate wide).")
    p.add_argument("--holdout-manifest", required=True,
                   help="frozen probe manifest CSV whose game_idx column is excluded.")
    p.add_argument("--out", required=True, help="output train manifest CSV path.")
    args = p.parse_args(argv)

    for path in (args.queue, args.holdout_manifest):
        if not Path(path).exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2

    with Path(args.queue).open(newline="") as f:
        queue_rows = list(csv.DictReader(f))

    holdout = load_holdout_game_idxs(args.holdout_manifest)
    cases = select_calibration_cases(queue_rows, holdout)
    write_manifest(cases, args.out)
    print(f"wrote {len(cases)} calibration train cases -> {args.out} "
          f"(excluded {len(holdout)} holdout games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
