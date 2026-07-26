"""Read-only postmortem for the rejected policy-mass ``r0`` control gate.

This tool deliberately has no candidate-grid or frozen-split mode.  It replays
only the 80 already-selected tuning rows under the two already-run control
configurations, authenticates them against the persisted controls artifact,
and writes a paired table for the 40 tuning controls.  It never writes inside
the production root or changes any source artifact.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from . import diagnose_fpu_policy_mass as dfpm
from . import fpu_provenance


PRODUCTION_ROOT = Path(
    "logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000")
DIAGNOSTIC_ROOT = Path(
    "logs/eval/fpu_v16_policy_mass_v2/diagnostic/"
    "production_v2_b400amend_4000g_seed20300000/tuning")
DEFAULT_OUT_DIR = DIAGNOSTIC_ROOT / "postmortem"
SCHEMA_VERSION = 1
EXPECTED_TUNING_ROWS = 80
EXPECTED_TUNING_CONTROLS = 40
EXPECTED_LOWER_PRIOR_FLIPS = 11


POSTMORTEM_FIELDNAMES = [
    "canonical_position_sha1", "game_idx", "position_ply", "phase", "ply_bucket",
    "branching_band", "side", "control_flip_to_lower_prior",
    "absolute_off_selected_move_id", "absolute_off_selected_move",
    "r0_selected_move_id", "r0_selected_move",
    "absolute_off_selected_move_prior", "r0_selected_move_prior",
    "absolute_off_selected_move_prior_rank", "r0_selected_move_prior_rank",
    "absolute_off_q_parent_final", "r0_q_parent_final",
    "absolute_off_root_value_stm", "r0_root_value_stm",
    "root_value_delta_r0_minus_absolute_off",
    "absolute_off_top_share", "r0_top_share", "top_share_delta_r0_minus_absolute_off",
    "absolute_off_effective_children", "r0_effective_children",
    "effective_children_delta_r0_minus_absolute_off",
    "absolute_off_reply_count", "r0_reply_count", "reply_count_delta_r0_minus_absolute_off",
    "absolute_off_collapsed", "r0_collapsed", "collapse_transition",
    "absolute_off_lock_in", "r0_lock_in", "lock_in_transition",
    "absolute_off_explored_mass", "r0_explored_mass",
    "absolute_off_explored_mass_at_stabilization", "r0_explored_mass_at_stabilization",
    "absolute_off_stabilization_sim", "r0_stabilization_sim",
]


def canonical_json(value: Any) -> str:
    """Stable JSON suitable for byte-identity checks; intentionally no time."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def move_label(move_id: int | None) -> str | None:
    if move_id is None:
        return None
    from .mcts import decode_move
    row, col = decode_move(move_id)
    return f"({row},{col})"


def _delta(new: float, old: float) -> float:
    # Keeps ordinary decimal diagnostics readable without hiding meaningful
    # 400-simulation precision (the raw paired values remain in the row).
    return round(float(new) - float(old), 15)


def _transition(before: bool, after: bool, *, positive: str, negative: str) -> str:
    if not before and after:
        return f"un{negative}_to_{positive}"
    if before and not after:
        return f"{negative}_to_un{negative}"
    return positive if after else f"un{negative}"


def _search_features(search_out, observer: dfpm.FpuTraceObserver) -> dict:
    """Extract fields missing from the original controls CSV from a fresh,
    deterministic replay of the same root.  The existing diagnostic's feature
    extraction remains the source of truth for gate-visible metrics."""
    _counts, _root_value, root = search_out
    features = dfpm._position_features(search_out, observer)
    leader = dfpm._leader_child(root)
    trace = features["trace"]
    features.update({
        "selected_move_id": None if leader is None else leader.move,
        "selected_move": None if leader is None else move_label(leader.move),
        "selected_move_prior": trace["selected_move_prior"],
        "selected_move_prior_rank": trace["selected_move_prior_rank"],
        "q_parent_final": float(root.q_value),
        "lock_in": dfpm.lock_in_event(trace),
        "explored_mass": trace["explored_mass"],
        "explored_mass_at_stabilization": trace["explored_mass_at_stabilization"],
        "stabilization_sim": trace["stabilization_sim"],
    })
    return features


def paired_control_row(manifest_row: Mapping[str, str], off: Mapping[str, Any],
                       r0: Mapping[str, Any]) -> dict:
    """The durable, per-control paired record.  This is pure so its crucial
    lower-prior predicate and deltas have a no-GPU test contract."""
    lower_prior = (
        off["selected_move_id"] != r0["selected_move_id"]
        and r0["selected_move_prior"] is not None
        and off["selected_move_prior"] is not None
        and r0["selected_move_prior"] < off["selected_move_prior"]
    )
    return {
        "canonical_position_sha1": manifest_row["canonical_position_sha1"],
        "game_idx": manifest_row["game_idx"],
        "position_ply": manifest_row["position_ply"],
        "phase": manifest_row["ply_bucket"],
        "ply_bucket": manifest_row["ply_bucket"],
        "branching_band": manifest_row["branching_band"],
        "side": manifest_row["side"],
        "control_flip_to_lower_prior": lower_prior,
        "absolute_off_selected_move_id": off["selected_move_id"],
        "absolute_off_selected_move": off["selected_move"],
        "r0_selected_move_id": r0["selected_move_id"],
        "r0_selected_move": r0["selected_move"],
        "absolute_off_selected_move_prior": off["selected_move_prior"],
        "r0_selected_move_prior": r0["selected_move_prior"],
        "absolute_off_selected_move_prior_rank": off["selected_move_prior_rank"],
        "r0_selected_move_prior_rank": r0["selected_move_prior_rank"],
        "absolute_off_q_parent_final": off["q_parent_final"],
        "r0_q_parent_final": r0["q_parent_final"],
        "absolute_off_root_value_stm": off["root_value_stm"],
        "r0_root_value_stm": r0["root_value_stm"],
        "root_value_delta_r0_minus_absolute_off": _delta(r0["root_value_stm"], off["root_value_stm"]),
        "absolute_off_top_share": off["top_share"], "r0_top_share": r0["top_share"],
        "top_share_delta_r0_minus_absolute_off": _delta(r0["top_share"], off["top_share"]),
        "absolute_off_effective_children": off["effective_children"],
        "r0_effective_children": r0["effective_children"],
        "effective_children_delta_r0_minus_absolute_off": _delta(r0["effective_children"], off["effective_children"]),
        "absolute_off_reply_count": off["replies"], "r0_reply_count": r0["replies"],
        "reply_count_delta_r0_minus_absolute_off": int(r0["replies"]) - int(off["replies"]),
        "absolute_off_collapsed": off["collapsed"], "r0_collapsed": r0["collapsed"],
        "collapse_transition": _transition(off["collapsed"], r0["collapsed"],
                                             positive="collapsed", negative="collapsed"),
        "absolute_off_lock_in": off["lock_in"], "r0_lock_in": r0["lock_in"],
        "lock_in_transition": _transition(off["lock_in"], r0["lock_in"],
                                            positive="locked", negative="locked"),
        "absolute_off_explored_mass": off["explored_mass"], "r0_explored_mass": r0["explored_mass"],
        "absolute_off_explored_mass_at_stabilization": off["explored_mass_at_stabilization"],
        "r0_explored_mass_at_stabilization": r0["explored_mass_at_stabilization"],
        "absolute_off_stabilization_sim": off["stabilization_sim"],
        "r0_stabilization_sim": r0["stabilization_sim"],
    }


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict:
    groups: Dict[str, list] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: {"flipped": sum(r["control_flip_to_lower_prior"] for r in group),
                   "total": len(group),
                   "rate": sum(r["control_flip_to_lower_prior"] for r in group) / len(group)}
            for name, group in sorted(groups.items())}


def _cohort_metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    if not rows:
        return {"n": 0}
    mean = lambda field: sum(float(r[field]) for r in rows) / len(rows)
    return {
        "n": len(rows),
        "mean_absolute_off_q_parent_final": mean("absolute_off_q_parent_final"),
        "mean_r0_q_parent_final": mean("r0_q_parent_final"),
        "mean_root_value_delta": mean("root_value_delta_r0_minus_absolute_off"),
        "mean_top_share_delta": mean("top_share_delta_r0_minus_absolute_off"),
        "mean_effective_children_delta": mean("effective_children_delta_r0_minus_absolute_off"),
        "mean_reply_count_delta": mean("reply_count_delta_r0_minus_absolute_off"),
        "mean_absolute_off_prior_rank": mean("absolute_off_selected_move_prior_rank"),
        "mean_r0_prior_rank": mean("r0_selected_move_prior_rank"),
    }


def summarize_controls(rows: Sequence[Mapping[str, Any]]) -> dict:
    flipped = [r for r in rows if r["control_flip_to_lower_prior"]]
    retained = [r for r in rows if not r["control_flip_to_lower_prior"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "n_controls": len(rows),
        "n_lower_prior_flips": len(flipped),
        "flip_rate": len(flipped) / len(rows) if rows else 0.0,
        "by_phase": _counts(rows, "phase"),
        "by_side": _counts(rows, "side"),
        "by_branching_band": _counts(rows, "branching_band"),
        "by_game": _counts(rows, "game_idx"),
        "flipped_vs_nonflipped": {
            "flipped": _cohort_metrics(flipped),
            "nonflipped": _cohort_metrics(retained),
        },
        "interpretation_boundary": (
            "All values are measured paired associations on the consumed tuning controls; "
            "they explain the rejection but do not reopen, relax, or rescue the rejected formula."
        ),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_tuning_rows(manifest: Path) -> tuple[list[dict], list[dict]]:
    all_rows = _read_csv(manifest)
    tuning = [row for row in all_rows if row["split"] == "tuning"]
    controls = [row for row in tuning if row["role"] == "control"]
    if len(tuning) != EXPECTED_TUNING_ROWS or len(controls) != EXPECTED_TUNING_CONTROLS:
        raise ValueError(f"expected {EXPECTED_TUNING_ROWS} tuning rows / {EXPECTED_TUNING_CONTROLS} controls, got {len(tuning)} / {len(controls)}")
    if any(row["split"] == "frozen_check" for row in tuning):
        raise ValueError("frozen_check row entered tuning selection")
    return tuning, controls


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSTMORTEM_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value))


def _immutable_hashes(manifest: Path, source_index: Path, controls_cases: Path,
                      replay_paths: Iterable[str]) -> dict:
    return {
        "manifest_sha1": _sha1(manifest),
        "source_index_sha1": _sha1(source_index),
        "controls_cases_sha1": _sha1(controls_cases),
        "replay_data_sha1": fpu_provenance.replay_data_sha1(list(replay_paths)),
    }


def run_postmortem(args) -> dict:
    manifest = Path(args.manifest)
    source_index = Path(args.source_jsonl)
    controls_cases = Path(args.controls_cases)
    gate_path = Path(args.controls_gate)
    config_path = Path(args.dev_corpus_config)
    checkpoint = args.checkpoint
    out_dir = Path(args.out_dir)
    if PRODUCTION_ROOT in out_dir.resolve().parents or out_dir.resolve() == PRODUCTION_ROOT.resolve():
        raise ValueError("postmortem output must not be written under the immutable production root")

    tuning_rows, control_rows = _load_tuning_rows(manifest)
    gate = json.loads(gate_path.read_text())
    if gate.get("r0_qualified") is not False or gate.get("mode") != "tuning":
        raise ValueError("expected the recorded rejected tuning controls gate")
    expected = gate["fingerprint"]["selection_context"]
    from .build_fpu_dev_corpus import load_game_index
    # `_reconstruct_state` is the shared production helper and deliberately
    # indexes this mapping with ``int(row["game_idx"])``.
    replay_by_game = {int(row["game_idx"]): row["replay_path"] for row in load_game_index(str(source_index))}
    replay_paths = list(replay_by_game.values())
    immutable_before = _immutable_hashes(manifest, source_index, controls_cases, replay_paths)
    if immutable_before["manifest_sha1"] != expected["dev_manifest_sha1"]:
        raise ValueError("manifest SHA-1 does not match controls fingerprint")
    if immutable_before["source_index_sha1"] != expected["source_index_sha1"]:
        raise ValueError("source-index SHA-1 does not match controls fingerprint")
    if immutable_before["replay_data_sha1"] != expected["replay_data_sha1"]:
        raise ValueError("replay-data SHA-1 does not match controls fingerprint")
    if _sha1(config_path) != expected["dev_corpus_config_sha1"]:
        raise ValueError("dev-corpus config SHA-1 does not match controls fingerprint")
    if dfpm._checkpoint_identity(checkpoint) != expected["checkpoint_identity"]:
        raise ValueError("checkpoint identity does not match controls fingerprint")
    current_sources = fpu_provenance.source_file_sha1s(dfpm.RESULT_DETERMINING_SOURCES)
    if current_sources != expected["source_file_sha1s"]:
        raise ValueError("result-determining source hashes do not match controls fingerprint")

    evaluator, base_cfg = dfpm._make_evaluator_and_base_cfg(
        checkpoint, args.eval_batch_size, args.stall_flush_sims)
    if dataclasses.asdict(base_cfg) != expected["base_mcts_config"]:
        raise ValueError("effective MCTS config does not match controls fingerprint")

    all_features: dict[str, dict[str, dict]] = {dfpm.ABSOLUTE_OFF.label: {}, dfpm.R0.label: {}}
    for row in tuning_rows:
        state = dfpm._reconstruct_state(row, replay_by_game)
        seed = dfpm._run_seed(args.seed_base, int(row["game_idx"]), int(float(row["position_ply"])))
        sha = row["canonical_position_sha1"]
        for config in (dfpm.ABSOLUTE_OFF, dfpm.R0):
            search_out, observer = dfpm._search_position(
                evaluator, dfpm._config_for(base_cfg, config), state, seed)
            all_features[config.label][sha] = _search_features(search_out, observer)

    persisted = _read_csv(controls_cases)
    if len(persisted) != 2 * EXPECTED_TUNING_ROWS:
        raise ValueError(f"expected {2 * EXPECTED_TUNING_ROWS} persisted controls rows, got {len(persisted)}")
    recomputed = {dfpm.ABSOLUTE_OFF.label: {}, dfpm.R0.label: {}}
    for row in tuning_rows:
        sha = row["canonical_position_sha1"]
        for label in recomputed:
            recomputed[label][sha] = dfpm._controls_case_row(row, label, all_features[label][sha])
    verified_count, verified_rows_sha1 = dfpm.verify_recomputed_controls(persisted, recomputed)

    paired_rows = [paired_control_row(
        row, all_features[dfpm.ABSOLUTE_OFF.label][row["canonical_position_sha1"]],
        all_features[dfpm.R0.label][row["canonical_position_sha1"]]) for row in control_rows]
    paired_rows.sort(key=lambda row: (int(row["game_idx"]), int(float(row["position_ply"]))))
    summary = summarize_controls(paired_rows)
    if summary["n_lower_prior_flips"] != EXPECTED_LOWER_PRIOR_FLIPS:
        raise RuntimeError(f"expected exactly {EXPECTED_LOWER_PRIOR_FLIPS} lower-prior flips, got {summary['n_lower_prior_flips']}")

    immutable_after = _immutable_hashes(manifest, source_index, controls_cases, replay_paths)
    if immutable_after != immutable_before:
        raise RuntimeError("an immutable input changed during the read-only postmortem")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "scope": "tuning-only read-only postmortem; no frozen_check row or nonzero coefficient",
        "manifest_sha1": immutable_before["manifest_sha1"],
        "replay_data_sha1": immutable_before["replay_data_sha1"],
        "source_index_sha1": immutable_before["source_index_sha1"],
        "controls_cases_sha1": immutable_before["controls_cases_sha1"],
        "controls_gate_sha1": _sha1(gate_path),
        "checkpoint_identity": expected["checkpoint_identity"],
        "effective_mcts_config": dataclasses.asdict(base_cfg),
        "source_file_sha1s": current_sources,
        "postmortem_module_sha1": _sha1(Path(__file__)),
        "add_noise": False,
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
        "n_tuning_rows_replayed": len(tuning_rows),
        "n_tuning_controls_reported": len(paired_rows),
        "n_frozen_check_rows_participated": 0,
        "persisted_controls_rows_verified": verified_count,
        "persisted_controls_verified_rows_sha1": verified_rows_sha1,
        "immutable_inputs_before": immutable_before,
        "immutable_inputs_after": immutable_after,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "control_flip_pairs.csv"
    summary_path = out_dir / "summary.json"
    provenance_path = out_dir / "provenance.json"
    report_path = out_dir / "report.md"
    _write_csv(table_path, paired_rows)
    _write_json(summary_path, summary)
    _write_json(provenance_path, provenance)
    report_path.write_text(
        "# Policy-mass r0 tuning-control postmortem\n\n"
        f"Measured result: {summary['n_lower_prior_flips']}/{summary['n_controls']} lower-prior flips "
        f"({summary['flip_rate']:.1%}); the frozen rejection remains final.\n\n"
        "The paired CSV contains every tuning control. Summary comparisons are descriptive "
        "associations only, not a causal rescue or a new candidate result.\n")
    artifact_hashes = {path.name: _sha1(path) for path in (table_path, summary_path, provenance_path, report_path)}
    _write_json(out_dir / "artifact_hashes.json", artifact_hashes)
    return {"summary": summary, "provenance": provenance, "artifact_hashes": artifact_hashes}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(PRODUCTION_ROOT / "fpu_dev_corpus_v2_manifest.csv"))
    parser.add_argument("--source-jsonl", default=str(PRODUCTION_ROOT / "calib020_0001_vs_0379_4000g_w4_seed20300000_games.jsonl"))
    parser.add_argument("--controls-cases", default=str(DIAGNOSTIC_ROOT / "controls_cases.csv"))
    parser.add_argument("--controls-gate", default=str(DIAGNOSTIC_ROOT / "controls_gate.json"))
    parser.add_argument("--dev-corpus-config", default=str(PRODUCTION_ROOT / "fpu_dev_corpus_v2_config.json"))
    parser.add_argument("--checkpoint", default="checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed-base", type=int, default=20260711)
    parser.add_argument("--eval-batch-size", type=int, default=14)
    parser.add_argument("--stall-flush-sims", type=int, default=48)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    result = run_postmortem(_parse_args(argv))
    summary = result["summary"]
    print(f"[fpu-postmortem] verified 160 persisted rows; lower-prior flips="
          f"{summary['n_lower_prior_flips']}/{summary['n_controls']} ({summary['flip_rate']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
