"""Capture the shipped-FPU selected move for every frozen A/B/C/D case.

v17 plan Task 1 supplement. The four canonical probe CSVs record
`probe_black_root_value` and `probe_top1_share` but NOT which move the search
picked, so design §9's "with exact selected moves" clause has no baseline to
compare against. This re-runs the SAME harness, checkpoint, configuration and
per-case seed that produced those CSVs, and keeps the visit counts that
`eval_position_probe.evaluate_case` / `eval_goal_line_trigger_probe.evaluate_case`
discard.

SHIPPED BASELINE ONLY. No v17 field exists yet, no candidate is evaluated, and
no gate verdict is emitted. This is baseline authentication under design §9
("authenticated into the protocol before any MCTS source edit"), NOT an
A/B/C/D acceptance run under §1.2.

Authentication: each case's recomputed black value must reproduce the frozen
CSV value within 1e-6. That check is what proves this module's search glue is
equivalent to the original harness -- the glue exists only because the harness
returns `(black_value, top1_share)` and drops `counts`.

Run:
    .venv/bin/python -m scripts.GPU.alphazero.capture_v17_abcd_selected_moves
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
from .mcts import MCTS
from .position_probe_cases import position_state

CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CHECKPOINT_ROW_FILTER = "0001"
OUT = "logs/eval/fpu_v17_baseline_policy_mass/prechange_abcd_selected_moves.json"

# design §2.4 -- the frozen batching triple. These are also the CLI defaults of
# both probe harnesses, i.e. the configuration the frozen CSVs were produced at.
MCTS_SIMS, EVAL_BATCH_SIZE, STALL_FLUSH_SIMS, PENDING_VIRTUAL_VISITS = 400, 14, 48, 8

# Per-gate: cases CSV, replay source, and the harness's own seed rule.
# A/C/D come from eval_position_probe.py  (base seed 20260616,
#   rng = base ^ game_idx ^ position_ply; replay_path is carried on each row).
# B comes from eval_goal_line_trigger_probe.py (base seed 20260614,
#   rng = base ^ game_idx; replay_path lives in the manifest, joined on
#   (game_idx, position_ply)).
GATES = {
    "A": {"cases": "logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/position_probe_cases.csv",
          "manifest": None, "base_seed": 20260616, "seed_rule": "base ^ game_idx ^ position_ply"},
    "B": {"cases": "logs/eval/black_predrop_calib010_goal_line/goal_line_trigger_probe_cases.csv",
          "manifest": "logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json",
          "base_seed": 20260614, "seed_rule": "base ^ game_idx"},
    "C": {"cases": "logs/eval/calib020_post_opening_sweep/position_probe_cases.csv",
          "manifest": None, "base_seed": 20260616, "seed_rule": "base ^ game_idx ^ position_ply"},
    "D": {"cases": "logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv",
          "manifest": None, "base_seed": 20260616, "seed_rule": "base ^ game_idx ^ position_ply"},
}
TOLERANCE = 1e-6


def sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def mcts_config():
    return cfg_from(EvalConfig(mcts_sims=MCTS_SIMS,
                               mcts_eval_batch_size=EVAL_BATCH_SIZE,
                               mcts_stall_flush_sims=STALL_FLUSH_SIMS))


def load_gate_cases(gate):
    """Frozen rows for one gate, each carrying a usable `replay_path`."""
    spec = GATES[gate]
    rows = [r for r in csv.DictReader(open(spec["cases"]))
            if r["checkpoint"] == CHECKPOINT_ROW_FILTER]
    if spec["manifest"]:
        man = json.loads(Path(spec["manifest"]).read_text())["cases"]
        by_key = {(int(c["game_idx"]), int(c["position_ply"])): c for c in man}
        for r in rows:
            src = by_key[(int(r["game_idx"]), int(r["position_ply"]))]
            assert src["side_to_move"] == r["side_to_move"], r
            r["replay_path"] = src["replay_path"]
    return rows


def selected_move_for(evaluator, row, cfg, base_seed, seed_rule):
    """Mirror of the harness `evaluate_case`, but keeps the visit counts.

    Tie-break follows design §7.0: greatest completed visit count, then the
    canonical move comparator. Row-major encoding makes ordering by (row, col)
    identical to ordering by encoded move id.
    """
    replay = json.loads(Path(row["replay_path"]).read_text())
    state = position_state(replay, int(row["position_ply"]), row["side_to_move"])
    seed = (base_seed ^ int(row["game_idx"]) ^ int(row["position_ply"])
            if "position_ply" in seed_rule else base_seed ^ int(row["game_idx"]))
    counts, root_value = MCTS(evaluator, cfg, random.Random(seed)).search(
        state, add_noise=False)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"empty search counts for {row}")
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_move, top_visits = ranked[0]
    tied = sum(1 for _m, v in ranked if v == top_visits) - 1
    black_value = root_value if state.to_move == "black" else -root_value
    return {
        "selected_move": [int(top_move[0]), int(top_move[1])],
        "selected_move_visits": int(top_visits),
        "tied_with_top": tied,
        "top_share_repr": repr(top_visits / total),
        "recomputed_black_value_repr": repr(float(black_value)),
        "seed": seed,
    }


def capture():
    cfg = mcts_config()
    assert (cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits) \
        == (EVAL_BATCH_SIZE, STALL_FLUSH_SIMS, PENDING_VIRTUAL_VISITS), cfg
    evaluator = _default_evaluator_factory(CHECKPOINT)
    gates, mismatches, ties = {}, [], 0
    for gate, spec in GATES.items():
        rows = load_gate_cases(gate)
        cases = []
        for row in rows:
            out = selected_move_for(evaluator, row, cfg, spec["base_seed"],
                                    spec["seed_rule"])
            frozen = float(row["probe_black_root_value"])
            delta = abs(float(out["recomputed_black_value_repr"]) - frozen)
            if delta >= TOLERANCE:
                mismatches.append((gate, row["case_id"], frozen, delta))
            ties += out["tied_with_top"] > 0
            cases.append({"case_id": row["case_id"],
                          "game_idx": int(row["game_idx"]),
                          "position_ply": int(row["position_ply"]),
                          "side_to_move": row["side_to_move"],
                          "frozen_black_value_repr": repr(frozen),
                          "abs_delta_vs_frozen_repr": repr(delta),
                          **out})
        gates[gate] = {"cases_source": spec["cases"],
                       "cases_source_sha1": sha1(spec["cases"]),
                       "manifest": spec["manifest"],
                       "base_seed": spec["base_seed"],
                       "seed_rule": spec["seed_rule"],
                       "n": len(cases),
                       "max_abs_delta_vs_frozen_repr":
                           repr(max(float(c["abs_delta_vs_frozen_repr"]) for c in cases)),
                       "cases": cases}
        print(f"  {gate}: {len(cases)} cases, max |delta| vs frozen = "
              f"{gates[gate]['max_abs_delta_vs_frozen_repr']}")
    if mismatches:
        raise SystemExit(f"AUTHENTICATION FAILED, {len(mismatches)} case(s) "
                         f"outside {TOLERANCE}: {mismatches[:5]}")
    return {
        "record_kind": "v17_prechange_abcd_selected_moves",
        "schema_version": 1,
        "experiment": "fpu_v17_baseline_policy_mass",
        "scope": "SHIPPED BASELINE ONLY. No v17 field exists, no candidate is "
                 "evaluated, and no gate verdict is emitted. Baseline "
                 "authentication under design section 9; NOT an A/B/C/D "
                 "acceptance run under section 1.2.",
        "scientific_interpretation": "This artifact may be used ONLY as the "
                                     "selected-move half of the Stage-4 "
                                     "baseline reproduction check. It may not "
                                     "select or tune a coefficient.",
        "checkpoint": CHECKPOINT,
        "checkpoint_sha1": sha1(CHECKPOINT),
        "mcts": {"n_simulations": MCTS_SIMS,
                 "batching_triple": [EVAL_BATCH_SIZE, STALL_FLUSH_SIMS,
                                     PENDING_VIRTUAL_VISITS],
                 "add_noise": False,
                 "tie_break": "design section 7.0 -- greatest completed visit "
                              "count, then canonical move order",
                 "positions_with_a_tie_at_the_top": ties},
        "authentication": {"rule": f"every recomputed black value within "
                                   f"{TOLERANCE} of the frozen CSV value",
                           "cases_checked": sum(g["n"] for g in gates.values()),
                           "mismatches": 0},
        "source_sha1s": {p: sha1(p) for p in [
            "scripts/GPU/alphazero/eval_position_probe.py",
            "scripts/GPU/alphazero/eval_goal_line_trigger_probe.py",
            "scripts/GPU/alphazero/position_probe_cases.py",
            "scripts/GPU/alphazero/goal_line_trigger_probe_cases.py",
            "scripts/GPU/alphazero/mcts.py",
            "scripts/GPU/alphazero/eval_runner.py",
        ]},
        "gates": gates,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    record = capture()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, sort_keys=True, indent=1)
        f.write("\n")
    print(f"wrote {args.out}  sha1={sha1(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
