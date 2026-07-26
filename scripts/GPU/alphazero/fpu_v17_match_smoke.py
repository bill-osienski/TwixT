"""v17 same-checkpoint match-runner smoke driver (design §5.4, plan Task 8C).

Runs the two frozen four-game blocks with `calib020_0001` on both sides:

    block 1  shipped vs shipped,  seeds [20309100, 20309104)
    block 2  shipped vs r=0.35,   seeds [20309104, 20309108)

TOOLING INTEGRITY ONLY. Every artifact carries `run_kind=tooling_smoke` and
`scientific_interpretation_forbidden=true`; scores and game outcomes have no
scientific meaning and there is no 50% expectation.

This is a committed module rather than a scratch script because it is
RESULT-DETERMINING: it chooses the agent configs, the seeds and the colour
schedule, so its own SHA-1 belongs in the protocol's `source_files` alongside
the search and match modules it drives.

SMOKE ONLY. This driver hardcodes the tooling-smoke labels and the two §5.4
blocks, so it CANNOT run a scientific strength match (Task 14) as written:
`labels_for` would have to stop defaulting to `tooling_smoke`, the block table
and seed range would have to come from the strength stage, and the clean-tree
and output-path rules for a scientific run kind would have to be honoured.
Reusing it for Task 14 requires separate authorization and implementation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from . import fpu_v17_protocol as proto
from . import fpu_v17_provenance as prov
from .eval_checkpoint_match import run_match
from .eval_runner import EvalConfig, build_pairing_tasks, cfg_from

__all__ = ["CKPT", "V17_FIELD", "BLOCKS", "OUTPUT_DIR", "eval_config",
           "agent_config", "labels_for", "build_blocks", "driver_sha1",
           "negative_call_site_proof", "run_block", "run_smoke", "main"]

CKPT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
V17_FIELD = "fpu_shipped_policy_mass_reduction"
CANDIDATE = 0.35
RUN_KIND = "tooling_smoke"
OUTPUT_DIR = f"{prov.OUTPUT_ROOT}/smoke_8c"

# Design §5.4, verbatim. Data, not control flow.
BLOCKS = (
    {"name": "shipped_vs_shipped", "base_seed": 20309100, "games": 4,
     "a": ("shipped_a", 0.0), "b": ("shipped_b", 0.0)},
    {"name": "shipped_vs_r035", "base_seed": 20309104, "games": 4,
     "a": ("shipped", 0.0), "b": (f"r{CANDIDATE}", CANDIDATE)},
)


def eval_config(**overrides) -> EvalConfig:
    """The frozen match settings, including the §2.4 batching triple."""
    base = dict(board_size=prov.BOARD_SIZE, mcts_sims=prov.MCTS_SIMS,
                mcts_eval_batch_size=prov.BATCHING[0],
                mcts_stall_flush_sims=prov.BATCHING[1],
                selection_mode="opening_temperature", opening_temp_plies=20,
                temp_high=1.0, temp_low=0.1, max_moves=280)
    base.update(overrides)
    return EvalConfig(**base)


def agent_config(base, coefficient: float):
    return dataclasses.replace(base, **{V17_FIELD: coefficient})


def labels_for(run_kind: str = RUN_KIND) -> Dict[str, Any]:
    """The run labels stamped onto every artifact this driver emits."""
    prov.validate_run_kind(run_kind)
    return {"run_kind": run_kind,
            "scientific_interpretation_forbidden": not prov.is_scientific(run_kind)}


def driver_sha1() -> str:
    """This file's SHA-1 -- recorded in the protocol so the artifact names the
    exact driver bytes that produced it."""
    with open(__file__, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def build_blocks(base) -> List[Dict[str, Any]]:
    """Resolve the block table into concrete (id, config) agent pairs."""
    out = []
    for blk in BLOCKS:
        (a_id, a_r), (b_id, b_r) = blk["a"], blk["b"]
        out.append({**blk,
                    "a_id": a_id, "a_cfg": agent_config(base, a_r), "a_r": a_r,
                    "b_id": b_id, "b_cfg": agent_config(base, b_r), "b_r": b_r})
    return out


def negative_call_site_proof(evaluator_factory) -> Dict[str, Any]:
    """Prove the REAL call site refuses an invalid batching triple.

    Exercises `run_match` exactly as the smoke does -- same arguments, same
    `config_validator` -- but with a base config whose batching is wrong. The
    two agents still AGREE with that base, so the agent-consistency check is
    satisfied and only the frozen-triple validator can catch it.

    Returns evidence: that it refused, that the evaluator factory was never
    called, and that NOTHING was written -- checked over a clean temporary
    directory rather than one path, so the companion `_games.jsonl` and any
    replay output are covered too. `evaluator_factory` must record its calls in
    a `.calls` list.
    """
    import tempfile
    wrong = eval_config(mcts_eval_batch_size=16, mcts_stall_flush_sims=16)
    base = cfg_from(wrong)
    tmp = tempfile.mkdtemp(prefix="v17_negative_proof_")
    out = os.path.join(tmp, "must_not_exist.json")
    replay_dir = os.path.join(tmp, "replays")
    evaluator_factory.calls.clear()
    refused: Optional[str] = None
    try:
        run_match(a_ckpt=CKPT, b_ckpt=CKPT, games=2, base_seed=20309100,
                  config=wrong, workers=1, output=out,
                  replay_dir=replay_dir,
                  evaluator_factory=evaluator_factory,
                  a_mcts=agent_config(base, 0.0),
                  b_mcts=agent_config(base, CANDIDATE),
                  a_agent="shipped", b_agent="candidate",
                  allow_differ={V17_FIELD},
                  config_validator=prov.validate_batching,
                  labels=labels_for())
    except prov.ProtocolViolation as exc:
        refused = str(exc)
    written = sorted(
        os.path.relpath(os.path.join(d, f), tmp)
        for d, _subdirs, files in os.walk(tmp) for f in files)
    return {
        "refused": refused is not None,
        "reason": refused,
        "evaluator_loads": len(evaluator_factory.calls),
        "files_written": written,          # complete, not just the summary path
        "output_written": bool(written),
        "checked_paths": [os.path.basename(out), "replays/", "*_games.jsonl"],
        "invalid_triple": [16, 16, prov.BATCHING[2]],
    }


def run_block(blk, config: EvalConfig, out_dir: str, *,
              evaluator_factory=None) -> Dict[str, Any]:
    """Run one four-game block and return its summary."""
    labels = labels_for()
    for cfg in (blk["a_cfg"], blk["b_cfg"]):
        prov.validate_batching(cfg)          # before anything loads
    out = os.path.join(out_dir, f"{blk['name']}.json")
    summary = run_match(
        a_ckpt=CKPT, b_ckpt=CKPT, games=blk["games"],
        base_seed=blk["base_seed"], config=config, workers=1, output=out,
        pairing_id=blk["name"], evaluator_factory=evaluator_factory,
        a_mcts=blk["a_cfg"], b_mcts=blk["b_cfg"],
        a_agent=blk["a_id"], b_agent=blk["b_id"],
        allow_differ={V17_FIELD}, config_validator=prov.validate_batching,
        labels=labels)
    return summary


def tasks_for(blk, pairing_index: int = 0):
    return build_pairing_tasks(
        blk["name"], CKPT, CKPT, games=blk["games"],
        base_seed=blk["base_seed"], pairing_index=pairing_index,
        a_mcts=blk["a_cfg"], b_mcts=blk["b_cfg"],
        a_agent=blk["a_id"], b_agent=blk["b_id"])


def build_smoke_protocol() -> Dict[str, Any]:
    """One protocol covering the whole §5.4 match-smoke seed range."""
    return proto.build_protocol(
        run_kind=RUN_KIND, coefficient=CANDIDATE,
        base_seed=prov.MATCH_SMOKE_SEEDS[0], games=prov.MATCH_SMOKE_SEEDS[1],
        board_size=prov.BOARD_SIZE, checkpoints={"both_sides": CKPT},
        source_files=("scripts/GPU/alphazero/mcts.py",
                      "scripts/GPU/alphazero/eval_runner.py",
                      "scripts/GPU/alphazero/eval_checkpoint_match.py",
                      "scripts/GPU/alphazero/eval_summary.py",
                      "scripts/GPU/alphazero/fpu_v17_match_smoke.py"),
        extra={"design_section": "5.4",
               "blocks": [b["name"] for b in BLOCKS],
               "driver_sha1": driver_sha1(),
               "scientific_interpretation_forbidden": True})


def run_smoke(out_dir: str = OUTPUT_DIR, *, evaluator_factory=None) -> Dict[str, Any]:
    """Emit the protocol/config, run both blocks, write the stamped report."""
    os.makedirs(out_dir, exist_ok=True)
    config = eval_config()
    base = cfg_from(config)
    prov.validate_batching(base)

    protocol = build_smoke_protocol()
    derived = proto.derive_config(protocol)
    proto.emit(os.path.join(out_dir, "protocol.json"), protocol)
    proto.emit(os.path.join(out_dir, "config.json"), derived)
    proto.verify_config_matches(protocol, derived)

    report: Dict[str, Any] = {
        "artifact_kind": "match_runner_tooling_smoke",
        **labels_for(),
        "design_section": "5.4",
        "protocol_sha1": proto.protocol_sha1(protocol),
        "driver_sha1": driver_sha1(),
        "frozen_design_sha1": prov.FROZEN_DESIGN_SHA1,
        "note": ("Tooling integrity only. Scores and outcomes have no "
                 "scientific meaning; there is no 50% expectation."),
        "blocks": {},
    }
    for blk in build_blocks(base):
        summary = run_block(blk, config, out_dir,
                            evaluator_factory=evaluator_factory)
        report["blocks"][blk["name"]] = {
            "base_seed": blk["base_seed"], "games": blk["games"],
            "agents": {blk["a_id"]: blk["a_r"], blk["b_id"]: blk["b_r"]},
            "a_score_rate": summary["a_score_rate"],
            "state_caps": summary["state_caps"],
            "avg_plies": summary["avg_plies"],
        }
    dest = os.path.join(out_dir, "match_runner_smoke_report.json")
    with open(dest, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUTPUT_DIR)
    args = ap.parse_args(argv)
    report = run_smoke(args.out_dir)
    print(json.dumps({"run_kind": report["run_kind"],
                      "blocks": sorted(report["blocks"]),
                      "driver_sha1": report["driver_sha1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
