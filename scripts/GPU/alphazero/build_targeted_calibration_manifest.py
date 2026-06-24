"""Deterministic builder for the Targeted Value Calibration v2 mixed manifest.

Correction rows (hard target) + retention rows (anchored to a checkpoint's own
probe_black_root_value) are merged into one CSV the calibration pool can load.
See docs/superpowers/specs/2026-06-23-targeted-value-calibration-v2-design.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def resolve_anchor_rows(rows: list, anchor_label: str) -> list:
    """Rows whose checkpoint == anchor_label (exact); else the unique set whose
    checkpoint endswith ':' + anchor_label. Raise on ambiguous or missing."""
    exact = [r for r in rows if r.get("checkpoint") == anchor_label]
    if exact:
        return exact
    suffix = [r for r in rows if str(r.get("checkpoint", "")).endswith(":" + anchor_label)]
    labels = sorted({r["checkpoint"] for r in suffix})
    if len(labels) == 1:
        return suffix
    if len(labels) > 1:
        raise ValueError(
            f"ambiguous anchor label {anchor_label!r}; candidates: {labels}; "
            f"pass an exact --*-anchor-label")
    raise ValueError(f"no checkpoint matches anchor label {anchor_label!r}")


UNIFIED_COLUMNS = [
    "case_rank", "tag", "source", "source_rank", "target_black_value", "weight_scale",
    "game_idx", "case_id", "replay_path", "position_ply", "side_to_move",
    "anchor_checkpoint", "drop_ply", "largest_drop_phase", "collapse_type",
]


def _unified_row(**kw) -> dict:
    row = {c: "" for c in UNIFIED_COLUMNS}
    row.update(kw)
    extra = set(kw) - set(UNIFIED_COLUMNS)
    if extra:
        raise KeyError(f"unknown unified columns: {sorted(extra)}")
    return row


def _read_csv(path) -> list:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _ply_key(replay_path: str, position_ply) -> tuple:
    return (replay_path, str(int(float(position_ply))))


def _validate_target_str(value: str, case_id: str) -> str:
    t = float(value)
    if not math.isfinite(t) or not (-1.0 <= t <= 1.0):
        raise ValueError(f"target_black_value {t!r} out of [-1,1] (case {case_id!r})")
    return value


def correction_rows(manifest_path, target: float, weight: float) -> list:
    target_str = _validate_target_str(str(target), "correction-target")
    out = []
    for r in _read_csv(manifest_path):
        out.append(_unified_row(
            tag="black_predrop_correction",
            source=Path(manifest_path).name,
            source_rank=r.get("case_rank", ""),
            target_black_value=target_str,
            weight_scale=str(weight),
            game_idx=r["game_idx"], case_id=r["case_id"], replay_path=r["replay_path"],
            position_ply=r["position_ply"], side_to_move=r["side_to_move"],
            drop_ply=r.get("drop_ply", ""),
            largest_drop_phase=r.get("largest_drop_phase", ""),
            collapse_type=r.get("collapse_type", "")))
    return out


def assert_no_holdout_overlap(correction: list, holdout_path) -> None:
    holdout_keys = {_ply_key(r["replay_path"], r["position_ply"])
                    for r in _read_csv(holdout_path)}
    leaks = [r for r in correction
             if _ply_key(r["replay_path"], r["position_ply"]) in holdout_keys]
    if leaks:
        raise ValueError(
            f"correction train leaks {len(leaks)} frozen-eval positions: "
            f"{[r['case_id'] for r in leaks]}")


def position_probe_retention_rows(cases_path, anchor_label, tag, weight) -> list:
    rows = resolve_anchor_rows(_read_csv(cases_path), anchor_label)
    seen, out = set(), []
    for r in rows:
        cid = r["case_id"]
        if cid in seen:
            raise ValueError(f"{tag}: duplicate case_id {cid!r} for anchor {anchor_label!r}")
        seen.add(cid)
        out.append(_unified_row(
            tag=tag, source=Path(cases_path).name,
            source_rank=r.get("case_rank", ""),
            target_black_value=_validate_target_str(r["probe_black_root_value"], cid),
            weight_scale=str(weight),
            game_idx=r["game_idx"], case_id=cid, replay_path=r["replay_path"],
            position_ply=r["position_ply"], side_to_move=r["side_to_move"],
            anchor_checkpoint=r["checkpoint"], drop_ply=r.get("drop_ply", ""),
            largest_drop_phase=r.get("largest_drop_phase", ""),
            collapse_type=r.get("collapse_type", "")))
    return out


def goal_line_retention_rows(cases_path, candidates_path, anchor_label, tag, weight) -> list:
    cases = resolve_anchor_rows(_read_csv(cases_path), anchor_label)
    index = {}
    for c in _read_csv(candidates_path):
        key = (str(int(c["game_idx"])), str(int(float(c["prev_black_ply"]))))
        if key in index:
            raise ValueError(f"goal-line candidates: duplicate join key {key}")
        index[key] = c
    out = []
    for r in cases:
        cid = r["case_id"]
        key = (str(int(r["game_idx"])), str(int(float(r["position_ply"]))))
        match = index.get(key)
        if match is None:
            raise ValueError(f"goal-line join: no candidate for case {cid!r} key {key}")
        replay_path = match["replay_path"]
        if not Path(replay_path).exists():
            raise ValueError(f"goal-line join: replay_path missing on disk: {replay_path}")
        if r["side_to_move"] != "black":
            raise ValueError(f"goal-line join: side_to_move {r['side_to_move']!r} != black ({cid!r})")
        out.append(_unified_row(
            tag=tag, source=Path(cases_path).name, source_rank=r.get("rank", ""),
            target_black_value=_validate_target_str(r["probe_black_root_value"], cid),
            weight_scale=str(weight),
            game_idx=r["game_idx"], case_id=cid, replay_path=replay_path,
            position_ply=r["position_ply"], side_to_move=r["side_to_move"],
            anchor_checkpoint=r["checkpoint"]))
    return out


def assign_case_rank(rows: list) -> list:
    for i, r in enumerate(rows, start=1):
        r["case_rank"] = i
    return rows


def tag_stats(rows: list) -> dict:
    stats: dict = {}
    for r in rows:
        s = stats.setdefault(r["tag"], {"n": 0, "weight_mass": 0.0, "targets": []})
        s["n"] += 1
        s["weight_mass"] += float(r["weight_scale"])
        s["targets"].append(float(r["target_black_value"]))
    return stats


def print_tag_stats(stats: dict) -> None:
    for tag in sorted(stats):
        s = stats[tag]
        t = s["targets"]
        print(f"{tag}: n={s['n']}, weight_mass={s['weight_mass']:.1f}, "
              f"target mean={sum(t)/len(t):+.3f} min={min(t):+.3f} max={max(t):+.3f}")


def write_manifest(rows: list, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in UNIFIED_COLUMNS})


def validate_rows(rows: list) -> None:
    """Final guard on the assembled manifest before writing (catches adapter bugs)."""
    for r in rows:
        _validate_target_str(r["target_black_value"], r["case_id"])
        w = float(r["weight_scale"])
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"weight_scale {w!r} invalid (case {r['case_id']!r})")
        for k in ("replay_path", "position_ply", "side_to_move"):
            if not r[k]:
                raise ValueError(f"missing {k} (case {r['case_id']!r})")


def build(args) -> list:
    rows = correction_rows(args.correction_manifest, args.correction_target, 1.0)
    assert_no_holdout_overlap(rows, args.correction_holdout_manifest)
    rows += position_probe_retention_rows(
        args.red_predrop_cases, args.red_predrop_anchor_label,
        "red_predrop_retention", args.retention_weight)
    rows += position_probe_retention_rows(
        args.old_post_opening_cases, args.old_post_opening_anchor_label,
        "old_post_opening_retention", args.retention_weight)
    rows += goal_line_retention_rows(
        args.goal_line_cases, args.goal_line_candidates, args.goal_line_anchor_label,
        "goal_line_retention", args.retention_weight)
    validate_rows(rows)
    assign_case_rank(rows)
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the Targeted Value Calibration v2 manifest.")
    p.add_argument("--correction-manifest", required=True)
    p.add_argument("--correction-holdout-manifest", required=True)
    p.add_argument("--red-predrop-cases", required=True)
    p.add_argument("--red-predrop-anchor-label", default="0001")
    p.add_argument("--old-post-opening-cases", required=True)
    p.add_argument("--old-post-opening-anchor-label", default="0001")
    p.add_argument("--goal-line-cases", required=True)
    p.add_argument("--goal-line-candidates", required=True)
    p.add_argument("--goal-line-anchor-label", default="0001")
    p.add_argument("--correction-target", type=float, default=-0.35)
    p.add_argument("--retention-weight", type=float, default=0.5)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = build(args)
    write_manifest(rows, args.out)
    print_tag_stats(tag_stats(rows))
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
