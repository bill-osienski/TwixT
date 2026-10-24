"""SEARCH DISTILLATION D2 -- CORRECTED-CONFIGURATION TRIAL, STEP 1: the 12 fresh held-out games, in ONE fresh process
launched with MLX_ENABLE_TF32=0. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md` (§2, §4), pinned below by sha256,
with the reused modules pinned by hash. Nothing runs on import, and importing this module loads no MLX. `run` refuses
unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless
`search_distillation_d2_heldout_authorization.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED` is open (read at call time).

WHAT THE RUN DOES: 12 fresh self-play games of calib020_0001 at Stage 2's exact configuration (Stage 1's: 1,600
simulations, train()'s other MCTS defaults, root noise on, mirror off, both goal-completion diagnostics off; the same
loader, evaluation mode and evaluator), all HELD OUT, in ONE capture phase. A COPY of Stage 2's generation loop
(search_distillation_d2_generate._execute); the diff: one phase of 12 slots, every slot held out, its own directories
and manifest, the version check and the setting recorded at start and end. Slot s = game id s, seed plan.base + s. The
phase is one finalized capture directory OUTSIDE the repository, apart from the original corpus; its context.json,
receipts.jsonl and finalized.json are copied into the evidence directory. Every record the Stage 0 reader rebuilds must
equal the learner PositionRecord play_game returned.

The set is ELIGIBLE (status COMPLETED, heldout.json written) only if all 12 games finalize. The first capture failure,
check failure, exception or SIGTERM (the wrapper's 2-hour ceiling) stops it as a VOID: no retry, replacement or top-up,
and the trial is closed. The games are used ONLY for the trial's baseline and final check.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_heldout run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import platform
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from . import search_distillation_d2_heldout_authorization as GATE
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-heldout-record/1"
MANIFEST_KIND = "search-distillation-d2-heldout/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md"
CARD_SHA256 = "9f2c60967792bc8d970f00472b50bbc91f8c3ab599acb7c59aa417815dc10062"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-heldout"
PLAN = "docs/superpowers/evidence/search-distillation-d2-heldout-run/plan.json"
HELDOUT_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"     # outside the repository
SETTING, SETTING_VALUE = "MLX_ENABLE_TF32", "0"
VERSIONS = {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.9"}
#: the reused modules, pinned (card §8)
CODE = {"measure": "scripts/GPU/alphazero/search_distillation_d2_measure.py",
        "generate": "scripts/GPU/alphazero/search_distillation_d2_generate.py",
        "distill": "scripts/GPU/alphazero/search_distillation_d2.py",
        "capture_serial": "scripts/GPU/alphazero/training_capture_serial.py",
        "capture": "scripts/GPU/alphazero/training_replay_capture.py"}
PINNED = {CODE["measure"]: "c2ecd81d7939a3879895b8bdfb36221c5cedecc616c345a37d4ef26f03b30669",
          CODE["generate"]: "38b0aba68d9cef583f5e8af118ba1ffc6361bf8851fcdbafa90d268de4cab347",
          CODE["distill"]: "bc8fd0c80fc533f50273ebda60c81375c5131d068a70344e07afef9a73617be9",
          CODE["capture_serial"]: "89d86be91f424e73c8b464935454a61c5f347f0a2161d7f78fb6fba37b21c4f0",
          CODE["capture"]: "b58765b13f798133fb685ae4599f4b32d9f262d7659a402f44c10600f8974f32"}
FROZEN = {"games": 12, "n_simulations": 1600, "phase_bytes": 100 << 20}
PHASE_DIR = "phase_00"
EXIT_DONE, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated, Seams = M.Void, M.Terminated, M.Seams


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, an output exists, or a bad
    argument."""


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def versions() -> Dict[str, str]:
    import mlx.core as mx
    import numpy as np
    return {"mlx": mx.__version__, "numpy": np.__version__, "python": platform.python_version()}


def _execute(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, heldout_dir: pathlib.Path,
             rec: Dict[str, Any], log, versions_of: Callable[[], Dict[str, str]]) -> None:
    import numpy as np

    from . import search_distillation_d2 as D
    from . import training_capture_serial as S
    from . import training_replay_capture as C

    rec["versions"] = versions_of()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    n_slots = spec["games"]
    plan = seams.read_plan()
    if plan.get("n_slots") != n_slots or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {n_slots} slots from an int base: {plan}")
    bad = [plan["base"] + s for s in range(n_slots) if not seams.seed_ok(plan["base"] + s)]
    if bad:
        raise Void("seeds", f"{len(bad)} seeds not ACCOUNTED, available and schedulable, first {bad[0]}")
    rec["seeds"] = {"base": plan["base"], "n_slots": n_slots}
    if seams.mirror_prob() != 0.0:
        raise Void("mirror", f"mirror augmentation is {seams.mirror_prob()}; the card needs TWIXT_MIRROR_PROB=0")
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}

    network = seams.load_network()
    network.eval()
    rec["modes_at_load"] = M._require_eval(network, "after load")
    evaluator = seams.make_evaluator(network)
    digest = S.network_state_digest(network)
    rec["model_digest_before"] = digest
    cfg = M.mcts_config_for(spec)
    code = S.code_identity(ROOT)
    rec["games"] = []
    log("loaded")

    slots = range(n_slots)
    context = S.build_context(run_id=f"search-distillation-d2:heldout-tf32off:{heldout_dir.name}", phase_id=PHASE_DIR,
                              seeds={s: plan["base"] + s for s in slots}, model_id=M.MODEL_ID, model_digest=digest,
                              model_source={"checkpoint": M.CHECKPOINT, "sha1": sha1}, mcts_config=cfg, code=code,
                              phase_bytes=spec["phase_bytes"], play_kwargs=M.DIAGNOSTICS_OFF)
    rec["configuration"] = context["configuration"]
    rec["modules"] = seams.loaded_modules()

    def timed(evaluator_, **kw):
        slot = kw["game_id"]
        log("game_start", game=slot, start=kw["start_player"])
        t0 = time.monotonic()
        game = seams.play_game(evaluator_, **kw)
        secs = time.monotonic() - t0
        g = {"game": slot, "seed": plan["base"] + slot, "held_out": True, "start_player": kw["start_player"],
             "plies": game.n_moves, "winner": game.winner, "draw_reason": game.draw_reason,
             "seconds": round(secs, 3), "seconds_per_ply": round(secs / max(game.n_moves, 1), 4)}
        rec["games"].append(g)
        log("game_end", **g)
        return game

    phase_dir = heldout_dir / PHASE_DIR
    out = S.run_serial_capture(evaluator=evaluator, network=network, capture_dir=str(phase_dir), context=context,
                               mcts_config=cfg, play_game=timed)
    _ctx, records, ids = D.load_records(str(phase_dir))
    learner = [pos for g in out["games"] for pos in g.positions]
    if len(records) != len(learner):
        raise Void("reader", f"{len(records)} rebuilt records for {len(learner)} learner rows")
    misaligned = [i for i, (r, q) in enumerate(zip(records, learner))
                  if not (np.array_equal(r.board_tensor, q.board_tensor) and r.to_move == q.to_move
                          and list(r.legal_moves) == list(q.legal_moves) and r.visit_counts == list(q.visit_counts)
                          and r.outcome == q.outcome)]
    if misaligned:
        raise Void("alignment", f"{len(misaligned)} records differ from the learner rows")
    rows = [r for env in C.load_phase(str(phase_dir))[1] for r in env["rows"]]
    meta = out_dir / "phases" / PHASE_DIR
    meta.mkdir(parents=True)
    for f in C.PHASE_FILES:
        shutil.copyfile(phase_dir / f, meta / f)
    finalized = (phase_dir / "finalized.json").read_bytes()
    rec["phase"] = {**out["stats"], "dir": PHASE_DIR, "games": list(slots), "context_sha256": out["context_sha256"],
                    "finalized_sha256": M._sha(finalized), "records": len(records),
                    "rows_with_inherited_child_visits": sum(sum(r["child_visits_before"]) > 0 for r in rows)}
    log("phase_end", records=len(records))
    del out, records, learner, rows

    rec["model_digest_after"] = S.network_state_digest(network)
    rec["modes_at_end"] = M._require_eval(network, "after the last game")
    if rec["model_digest_after"] != digest:
        raise Void("model", "the network changed during generation")
    games = rec["games"]
    rec["totals"] = {"games": len(games), "plies": sum(g["plies"] for g in games),
                     "game_seconds": round(sum(g["seconds"] for g in games), 3),
                     "records": rec["phase"]["records"], "bytes": rec["phase"]["bytes"]}
    manifest = {"kind": MANIFEST_KIND, "heldout_dir": str(heldout_dir), "eligible": True, "model_digest": digest,
                "seeds": rec["seeds"], "setting": {SETTING: os.environ.get(SETTING)},
                "phase": {k: rec["phase"][k] for k in ("dir", "games", "context_sha256", "finalized_sha256",
                                                       "records")},
                "held_out_games": list(slots)}
    rec["heldout_sha256"] = M._write_json_new(out_dir / "heldout.json", manifest)


def conclude(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, heldout_dir: pathlib.Path,
             head: str | None = None, versions_of: Callable[[], Dict[str, str]] = versions) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 2's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    heldout_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "heldout_dir": str(heldout_dir), "status": None, "written_by": "script",
                           "eligible": False, "strength_verdict": None, "setting": {SETTING: os.environ.get(SETTING)},
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable},
                           "loadavg_before": os.getloadavg()}
    log_fh = open(out_dir / "progress.jsonl", "x")

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - t0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start", setting=rec["setting"])
        _execute(spec, seams, out_dir, heldout_dir, rec, log, versions_of)
        rec.update(status="COMPLETED", eligible=True)
        code = EXIT_DONE
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    seams = M._real_seams(checkpoint)
    seams.read_plan = lambda: json.loads((ROOT / PLAN).read_text())
    return seams


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the frozen spec, the pinned card and code, the real sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED is closed: generation is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    heldout = HELDOUT_PARENT / f"heldout-tf32off-{stamp}"
    if not HELDOUT_PARENT.is_dir() or os.path.lexists(heldout):
        raise Refused(f"{HELDOUT_PARENT} is not a directory, or {heldout} exists")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, heldout, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "heldout_dir": rec["heldout_dir"], "totals": rec.get("totals")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_heldout run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
