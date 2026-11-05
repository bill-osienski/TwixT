"""H-TB memory budgets and revised fit projection (cards §15-§16; the user's reviews, 2026-11-04 and 2026-11-05).

Pure arithmetic over the COMMITTED timing records (`evidence/htb-timing/{label2,fit}/record.json`) and the per-unit
allowances below. Nothing here runs a model, reads the corpus or times anything. Every allowance is at or above what
was measured, and the totals are CONSERVATIVE SUMS: every phase is counted as if it were still resident (no freed
memory is assumed returned to the system). A less conservative per-phase peak is reported beside each sum.

Provenance of the per-unit figures (2026-11-04):
- measured with `tracemalloc` on the tests' fake games and FakeTeacher (the real structures; no model):
  label row as `run_label` holds it 44,803 B (mean 500 legal moves); compact row metadata 193 B;
  selection peak 358-370 B per position; one built example 235,973 B (T-fit measured 246,540 B on real rows);
- measured on 50 train-arm corpus files: parsed JSON is 2.00x the file bytes;
- the corpus's 41,400 game files total 2,513,232,495 B (file sizes);
- NOT measured, allowances only: the interpreter/library baselines, MLX's working buffers for one update.
- the longest C0-eligible game has 341 plies (the committed C0 inventory's eligible plies_per_game max).

All of these are ESTIMATES; peak memory has not been measured.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from . import htb_pilot as H
from . import htb_timing as T

CORPUS_BYTES = 2_513_232_495
PARSED_PER_RAW = 2.5            # measured 2.00
LABEL_ROW_B = 56_000            # measured 44,803
META_ROW_B = 300                # measured 193
SELECT_B_PER_POSITION = 500     # measured 358-370 (peak)
POOL_B_PER_MOVE = 200           # a move's (row, col) tuple and its player string, plus list slots
EXAMPLE_MARGIN = 1.25           # on T-fit's measured bytes per built example
BASELINE_FIT_B = 1_500_000_000      # Python, NumPy, MLX: allowance, not measured
BASELINE_LABEL_B = 3_000_000_000    # Python, NumPy, TensorFlow and the teacher model: allowance, not measured
HIDDEN, BOARD = 128, 24
WORKING_TENSORS = 200           # feature-map-sized buffers for one update (forward keeps about 50): allowance
GIB = 1 << 30
MAX_GAME_PLIES = 341            # C0 inventory: eligible plies_per_game max
COMPACT_ROW_B = H.PADDED * 4 + 3 * 8 + 4 + 1    # float32 target, three float64 values, int32 ply, filled flag
KEY_INDEX_B = 200               # LabelArrays' key -> row map, per row: allowance


def _gb(b: float) -> float:
    return round(b / 1e9, 3)


def fit_budget(fit_rec: Mapping[str, Any], n_rows: int = H.N_TRAIN + H.N_VAL) -> Dict[str, Any]:
    """The batched fit (card §15): retained data, temporary examples, optimizer state, gradients, working buffers."""
    m, pool = fit_rec["measurements"], fit_rec["pool_counts"]
    positions = pool["train"]["positions"] + pool["val"]["positions"]
    nets = m["networks_bytes"]                                   # reference + student parameters
    feature = 2 * H.FROZEN_RECIPE.rows_per_batch * BOARD * BOARD * HIDDEN * 4
    terms = {
        "baseline_process": BASELINE_FIT_B,
        "corpus_authentication_parsed": CORPUS_BYTES * PARSED_PER_RAW,
        "pool_move_lists": positions * POOL_B_PER_MOVE,
        "compact_labels": n_rows * (H.PADDED * 4 + 8 + META_ROW_B),
        "label_shard_reads": 2 * H.SHARD_ROWS * (H.PADDED * 4 + 8 * 4),
        "temporary_examples": H.SCORE_CHUNK * m["example_bytes_traced"] * EXAMPLE_MARGIN,
        "networks_optimizer_gradients_snapshots": 4 * nets,     # params N; Adam m+v N; grads N/2; snapshots N; +N/2
        "working_buffers": WORKING_TENSORS * feature,
    }
    total = sum(terms.values())
    auth = sum(terms[k] for k in ("baseline_process", "corpus_authentication_parsed", "pool_move_lists",
                                  "compact_labels"))
    train = total - terms["corpus_authentication_parsed"]
    return {"terms_bytes": {k: int(v) for k, v in terms.items()}, "terms_gb": {k: _gb(v) for k, v in terms.items()},
            "conservative_sum_bytes": int(total), "phase_peak_bytes": int(max(auth, train)),
            "proposed_memory_required_bytes": int(math.ceil(total / GIB) * GIB),
            "feature_map_bytes": feature,
            "as_built_estimate_bytes": fit_rec["memory_estimate"]["estimate_bytes"],
            "available_bytes_observed": [m["memory_before"]["available_bytes"], m["memory_after"]["available_bytes"]],
            "physical_bytes": m["memory_before"]["physical_bytes"]}


def label_budget(label_rec: Mapping[str, Any], fit_rec: Mapping[str, Any],
                 n_train: int = H.N_TRAIN, n_val: int = H.N_VAL, compact: bool = True) -> Dict[str, Any]:
    """`run_label`. compact=True (card §16): an arm's labels in LabelArrays, filled as each game is labelled, so only
    one game's label rows are ever held whole. compact=False: the previous path (ledger 2026-11-04), which held an
    arm's labelled rows with their legal-move lists until its shards were written."""
    pool = label_rec["pool_counts"]
    positions = pool["train"]["positions"] + pool["val"]["positions"]
    terms = {
        "baseline_process_with_tensorflow": BASELINE_LABEL_B,
        "corpus_authentication_parsed": CORPUS_BYTES * PARSED_PER_RAW,
        "pool_move_lists": positions * POOL_B_PER_MOVE,
        "selection_peak": positions * SELECT_B_PER_POSITION,
        "selection_record_and_json": 2 * (n_train + n_val) * META_ROW_B,
        "shard_buffer": H.SHARD_ROWS * (H.PADDED * 4 + 8 * 4),
    }
    if compact:
        terms["compact_labels_of_the_larger_arm"] = max(n_train, n_val) * (COMPACT_ROW_B + META_ROW_B + KEY_INDEX_B)
        terms["one_games_label_rows_and_replays"] = 2 * MAX_GAME_PLIES * LABEL_ROW_B
    else:
        terms["labelled_rows_of_the_larger_arm"] = max(n_train, n_val) * LABEL_ROW_B
    total = sum(terms.values())
    m = fit_rec["measurements"]
    return {"terms_bytes": {k: int(v) for k, v in terms.items()}, "terms_gb": {k: _gb(v) for k, v in terms.items()},
            "compact": compact, "conservative_sum_bytes": int(total),
            "phase_peak_bytes": int(total - terms["corpus_authentication_parsed"]),
            "proposed_memory_required_bytes": int(math.ceil(total / GIB) * GIB),
            "available_bytes_observed": [m["memory_before"]["available_bytes"], m["memory_after"]["available_bytes"]]}


def batched_fit_projection(fit_rec: Mapping[str, Any], cfg: T.TimingConfig) -> Dict[str, Any]:
    """The full fit with examples built per batch and per scoring chunk. Rebuilding is priced by a PLANNING ESTIMATE
    per row: T-fit's lazy board rebuild (prefix replay, board, mirror) plus its whole up-front example time per row.
    Derived, not measured, and not a demonstrated bound (the user's correction, 2026-11-05: timings on other rows,
    ply mixes or machine states can exceed it). The 871 s board-only figure is NOT this."""
    m = fit_rec["measurements"]
    r = m["lazy_board_s_per_row"] + m["examples_s_per_row"]
    rows_built_in_updates = cfg.full_updates * cfg.rows_per_batch
    base = T.project_fit(m, cfg)["terms"]
    terms = {
        "corpus_auth_s": base["corpus_auth_s"],
        "label_reload_s": base["label_reload_s"],
        "model_loads_s": base["model_loads_s"],
        "start_check_s": cfg.start_rows * (m["score_s_per_row_network"] + r),
        "validation_scoring_s": 2 * cfg.n_val * (m["score_s_per_row_network"] + r),
        "updates_s": (m["updates"]["warmup_s"] + (cfg.full_updates - m["updates"]["warmup_updates"])
                      * m["updates"]["steady_mean_s"] + rows_built_in_updates * r),
        "save_s": base["save_s"],
    }
    return {"terms": {k: round(v, 1) for k, v in terms.items()}, **T._ceiling(sum(terms.values())),
            "rebuild_s_per_row_estimate": r, "rows_built_in_updates": rows_built_in_updates,
            "replaces": {"examples_s": base["examples_s"]},
            "basis": "derived from the committed T-fit record; the per-row rebuild is a planning estimate, not a "
                     "measurement or a demonstrated bound"}
