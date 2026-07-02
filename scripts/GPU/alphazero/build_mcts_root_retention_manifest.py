"""Deterministic v5 root-retention manifest builder.

For each retention row of a v3-style stratified source manifest, computes TWO
targets against the BASE checkpoint:
  1. teacher_value  — raw single-position forward in EVAL-mode BatchNorm
     (matches the training-path eval-mode calibration forward, so the value
     term starts ~0 at gate-0), via the shared _teacher_infer.
  2. root_visits_json / root_value_stm — a 400-sim MCTS search using the GATE
     probes' exact loader (_default_evaluator_factory: NO eval(), train-mode
     BatchNorm, batch=1 sync search) and per-family gate seeds, so BASE root
     values reproduce the gate CSVs by construction.

Correction rows (tag == black_predrop_correction) pass through with all new
columns blank and target_black_value PRESERVED. Retention rows blank
target_black_value (stale v3 MCTS-root scalar) and teacher_policy_json.

See the v5 section of docs/2026-06-26-targeted-value-calibration-experiment-
ledger-v3f-v4-overlap-updated.md and the plan doc for the two-evaluator split.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from .position_probe_cases import load_csv_manifest
from .goal_line_trigger_probe_cases import position_state
from .build_teacher_calibration_manifest import _teacher_infer
from .calibration_pool import legal_moves_sha1

CORRECTION_TAG = "black_predrop_correction"
GOAL_LINE_TAG = "goal_line_retention"
NEW_COLUMNS = [
    "loss_mode", "teacher_value",
    "root_value_stm", "root_black_value", "root_visits_json",
    "root_legal_moves_sha1", "root_sims", "root_base_checkpoint", "root_seed",
    "root_mcts_eval_batch_size", "root_mcts_stall_flush_sims",
    # blanked-on-purpose teacher columns (kept so a v4-source manifest can't
    # leak stale raw-prior targets through the v5 contamination guard):
    "teacher_policy_json", "teacher_legal_moves_sha1",
]


def row_seed(tag: str, game_idx: int, position_ply: int,
             pos_base_seed: int, goal_base_seed: int) -> int:
    """Replicate the gate probes' per-case rng seeds exactly.

    goal_line_retention rows came from eval_goal_line_trigger_probe
    (seed = base ^ game_idx); every other retention family came from
    eval_position_probe (seed = base ^ game_idx ^ position_ply).
    """
    if tag == GOAL_LINE_TAG:
        return goal_base_seed ^ int(game_idx)
    return pos_base_seed ^ int(game_idx) ^ int(position_ply)


def dense_normalized_visits(counts: dict, legal, case_id: str) -> list[float]:
    """Dense visit vector aligned to legal order, normalized to sum 1.0.

    INTENTIONALLY hand-normalized instead of reusing MCTS.get_policy_target:
    that helper (mcts.py:1229-1245) returns a DICT {move: prob}, not a dense
    vector aligned to the manifest legal order that root_legal_moves_sha1
    pins, and it silently falls back to UNIFORM when total visits == 0. Here
    zero total visits is a build FAILURE (loud), never a uniform fallback —
    a uniform target would silently anchor garbage.
    """
    total = float(sum(counts.get(m, 0) for m in legal))
    if total <= 0:
        raise ValueError(f"{case_id}: root search returned zero total visits")
    return [counts.get(m, 0) / total for m in legal]


def _to_black(value_stm: float, side_to_move: str) -> float:
    if side_to_move == "black":
        return float(value_stm)
    if side_to_move == "red":
        return float(-value_stm)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


def build_rows(rows: list, raw_evaluator, search_fn, *, sims: int,
               base_checkpoint: str, pos_base_seed: int, goal_base_seed: int,
               eval_batch_size: int, stall_flush_sims: int) -> list[dict]:
    out = []
    for r in rows:
        row = dict(r)                            # preserve ALL source columns
        for c in NEW_COLUMNS:
            row[c] = ""
        if r.get("tag") == CORRECTION_TAG:
            # A-correction: hard target stays; every retention column blank.
            row["loss_mode"] = "hard_value"
            out.append(row)
            continue
        cid = r.get("case_id")
        replay = json.loads(Path(r["replay_path"]).read_text())
        ply = int(float(r["position_ply"]))
        side = r["side_to_move"]
        state = position_state(replay, ply, side)
        legal = state.legal_moves()

        # (1) raw eval-mode value anchor (matches training-path eval forward).
        _, _, raw_value = _teacher_infer(state, raw_evaluator)

        # (2) gate-faithful BASE root search.
        seed = row_seed(r.get("tag", ""), r["game_idx"], ply,
                        pos_base_seed, goal_base_seed)
        counts, root_value_stm = search_fn(state, seed)
        dense = dense_normalized_visits(counts, legal, cid)

        row["loss_mode"] = "mcts_root_retention"
        row["teacher_value"] = repr(float(raw_value))
        row["root_value_stm"] = repr(float(root_value_stm))
        row["root_black_value"] = repr(_to_black(root_value_stm, side))
        row["root_visits_json"] = json.dumps(dense)
        row["root_legal_moves_sha1"] = legal_moves_sha1(legal)
        row["root_sims"] = str(sims)
        row["root_base_checkpoint"] = base_checkpoint
        row["root_seed"] = str(seed)
        row["root_mcts_eval_batch_size"] = str(eval_batch_size)
        row["root_mcts_stall_flush_sims"] = str(stall_flush_sims)
        row["target_black_value"] = ""      # blank stale v3 MCTS-root scalar
        out.append(row)
    return out


def _gate_rows_for_checkpoint(rows: list, checkpoint_label: str) -> list:
    """Rows whose checkpoint == label (exact); else the unique set whose
    checkpoint endswith ':' + label. Raise if the label matches nothing.

    Mirrors resolve_anchor_rows in build_targeted_calibration_manifest.py."""
    exact = [r for r in rows if r.get("checkpoint") == checkpoint_label]
    if exact:
        return exact
    suffix = [r for r in rows if str(r.get("checkpoint", "")).endswith(":" + checkpoint_label)]
    if suffix:
        return suffix
    raise ValueError(f"no gate CSV row matches --gate-checkpoint-label {checkpoint_label!r}")


def cross_check_gate_values(out_rows: list, gate_csv_paths: list, tol: float,
                            checkpoint_label: str | None = None) -> dict:
    """Builder sanity gate: for every retention row whose case_id appears in a
    gate cases CSV, the recomputed root_black_value must match the gate's
    probe_black_root_value within tol. Proves the search config/seeds/BN mode
    reproduce the gate setup. Raises on any mismatch.

    Gate cases CSVs contain one row per (checkpoint x case_id); when
    checkpoint_label is given, only rows for that checkpoint (exact match, or
    the unique 'parent:label' suffix form) are kept. Without a label, a
    case_id that appears under more than one distinct checkpoint is ambiguous
    and raises rather than silently keeping whichever row sorts last."""
    all_rows = []
    for path in gate_csv_paths:
        with open(path, newline="") as f:
            all_rows.extend(csv.DictReader(f))
    if checkpoint_label is not None:
        all_rows = _gate_rows_for_checkpoint(all_rows, checkpoint_label)

    gate = {}
    gate_checkpoint = {}
    for r in all_rows:
        cid = r["case_id"]
        ckpt = r.get("checkpoint")
        seen_ckpt = gate_checkpoint.get(cid)
        if seen_ckpt is not None and seen_ckpt != ckpt:
            raise ValueError(
                f"{cid}: ambiguous gate row — matched checkpoints "
                f"{sorted({seen_ckpt, ckpt})}; pass --gate-checkpoint-label "
                f"to select BASE's rows")
        gate_checkpoint[cid] = ckpt
        gate[cid] = float(r["probe_black_root_value"])
    checked, unmatched, errors = 0, 0, []
    for row in out_rows:
        if row.get("loss_mode") != "mcts_root_retention":
            continue
        cid = row["case_id"]
        if cid not in gate:
            unmatched += 1
            continue
        checked += 1
        got = float(row["root_black_value"])
        want = gate[cid]
        if abs(got - want) > tol:
            errors.append(f"{cid}: recomputed {got:+.4f} vs gate {want:+.4f}")
    if errors:
        raise ValueError(
            "gate cross-check FAILED (wrong seeds / BN mode / config?): "
            + "; ".join(errors))
    return {"checked": checked, "unmatched": unmatched}


def output_fieldnames(base_columns: list, out_rows: list) -> list:
    """Source column order first, then NEW_COLUMNS, then any remaining keys
    build_rows introduced (e.g. target_black_value when the source lacked it)."""
    fields = list(base_columns) + [c for c in NEW_COLUMNS if c not in base_columns]
    for row in out_rows[:1]:
        fields += [k for k in row.keys() if k not in fields]
    return fields


def _real_search_fn(base_checkpoint: str, sims: int,
                    eval_batch_size: int, stall_flush_sims: int):
    """Gate-faithful search factory. Heavy imports deferred: MLX loads here,
    NOT at module import (tests run with fakes). Uses the gate probes' exact
    loader (_default_evaluator_factory: no eval(), compile=True)."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(base_checkpoint)
    cfg = cfg_from(EvalConfig(mcts_sims=sims,
                              mcts_eval_batch_size=eval_batch_size,
                              mcts_stall_flush_sims=stall_flush_sims))

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search(state, add_noise=False)

    return search_fn


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the v5 mcts-root-retention manifest.")
    ap.add_argument("--source", required=True, help="v3-style stratified manifest CSV")
    ap.add_argument("--base-checkpoint", required=True, help=".safetensors BASE (= teacher)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--gate-cases-csv", action="append", default=[],
                    help="gate cases CSV (repeatable) for the root-value cross-check; "
                         "STRONGLY recommended")
    ap.add_argument("--gate-tolerance", type=float, default=1e-3)
    ap.add_argument("--gate-checkpoint-label", default=None,
                    help="checkpoint label of BASE inside the gate cases CSVs "
                         "(short_id, e.g. 0001, or its disambiguated parent:short "
                         "form); required when the CSVs contain multiple "
                         "checkpoints")
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    # Raw anchor evaluator: EVAL-mode BN (running stats) — matches the
    # training-path eval-mode calibration forward, so gate-0 value term ~ 0.
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    network.eval()
    raw_evaluator = LocalGPUEvaluator(network)
    # Root search evaluator: the GATE loader, by construction (separate load;
    # do NOT share the eval()'d network above).
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows = build_rows(rows, raw_evaluator, search_fn, sims=args.sims,
                          base_checkpoint=args.base_checkpoint,
                          pos_base_seed=args.position_probe_base_seed,
                          goal_base_seed=args.goal_line_base_seed,
                          eval_batch_size=args.eval_batch_size,
                          stall_flush_sims=args.stall_flush_sims)
    if args.gate_cases_csv:
        stats = cross_check_gate_values(out_rows, args.gate_cases_csv,
                                        args.gate_tolerance,
                                        checkpoint_label=args.gate_checkpoint_label)
        print(f"gate cross-check PASS: {stats['checked']} matched, "
              f"{stats['unmatched']} retention rows without a gate row")
    else:
        print("WARNING: no --gate-cases-csv given; root targets NOT cross-checked "
              "against the gate CSVs")

    base_columns = list(rows[0].keys()) if rows else []
    fieldnames = output_fieldnames(base_columns, out_rows)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    n_ret = sum(1 for r in out_rows if r["loss_mode"] == "mcts_root_retention")
    print(f"wrote {len(out_rows)} rows ({n_ret} mcts_root_retention) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
