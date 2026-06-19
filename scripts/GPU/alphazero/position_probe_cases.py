"""Pure helpers for generic fixed-position probes.

Input manifests are CSV files with at least:
  game_idx, case_id, replay_path, position_ply, side_to_move

The probe reconstructs each replay at position_ply and evaluates the side-to-move
position across checkpoints. For black-defense probes, lower black root value
means better danger recognition.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median

from .goal_line_trigger_probe_cases import position_state

OVERVALUE_THRESHOLD = 0.25
SEVERE_OVERVALUE_THRESHOLD = 0.50

REQUIRED_CASE_KEYS = ("game_idx", "case_id", "replay_path", "position_ply", "side_to_move")


def load_csv_manifest(path):
    path = Path(path)
    with path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"manifest has no cases: {path}")

    cases = []
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_CASE_KEYS if k not in r or r[k] == ""]
        if missing:
            raise ValueError(f"case {i}: missing keys {missing}")

        case = dict(r)
        case["game_idx"] = int(case["game_idx"])
        case["position_ply"] = int(float(case["position_ply"]))
        case["case_id"] = case["case_id"] or f"game_{case['game_idx']:06d}_ply_{case['position_ply']:03d}"

        # Preserve optional numeric metadata where present.
        for k in [
            "case_rank", "drop_ply", "n_moves",
            "initial_a_value", "final_a_value", "largest_a_value_drop",
        ]:
            if k in case and case[k] not in ("", None):
                try:
                    case[k] = int(float(case[k])) if k in {"case_rank", "drop_ply", "n_moves"} else float(case[k])
                except ValueError:
                    pass

        cases.append(case)

    return {
        "schema_version": 1,
        "name": path.stem,
        "source": str(path),
        "num_cases": len(cases),
        "cases": cases,
    }


def case_id(case):
    return case.get("case_id") or f"game_{case['game_idx']:06d}_ply_{case['position_ply']:03d}"


def summarize(black_values, top1_shares):
    n = len(black_values)
    if n == 0:
        raise ValueError("no cases to summarize")
    over = sum(1 for v in black_values if v >= OVERVALUE_THRESHOLD)
    severe = sum(1 for v in black_values if v >= SEVERE_OVERVALUE_THRESHOLD)
    return {
        "num_cases": n,
        "mean_black_root_value": mean(black_values),
        "median_black_root_value": median(black_values),
        "black_overvalue_rate": over / n,
        "severe_black_overvalue_rate": severe / n,
        "mean_top1_share": mean(top1_shares),
        "median_top1_share": median(top1_shares),
    }
