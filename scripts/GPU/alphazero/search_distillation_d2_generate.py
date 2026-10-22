"""SEARCH DISTILLATION D2 -- STAGE 2: corpus generation. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-stage2-card.md`, pinned below by sha256. Nothing runs on
import. `run` refuses unless `search_distillation_d2_generate_authorization.SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED`
is open, read at call time.

WHAT THE RUN DOES: 120 fresh self-play games of calib020_0001 in 12 ordered phases of 10, at Stage 1's exact
configuration (search_distillation_d2_measure: 1,600 simulations, train()'s other defaults, root noise on, mirror
off, both goal-completion diagnostics off; the same loader, evaluation mode and evaluator). Slot s = game id s, seed
plan.base + s, phase s // 10; slot % 10 == 9 is HELD OUT (12 games), the rest are TRAINING (108). Each phase is one
finalized capture directory under the corpus root OUTSIDE the repository; its context.json, receipts.jsonl and
finalized.json (which bind every game file by sha256) are copied into the evidence directory as it finalizes.
After each phase: every record the Stage 0 reader rebuilds must equal the learner PositionRecord play_game returned.

The corpus is ELIGIBLE FOR TRAINING only if all 12 phases finalize (status COMPLETED, corpus.json written). The
first capture failure, check failure, exception or SIGTERM (the wrapper's 12-hour ceiling) stops the stage as a VOID:
no retry, replacement or top-up; completed and partial phases stay on disk as diagnostics.

    python -m scripts.GPU.alphazero.search_distillation_d2_generate run --checkpoint <absolute path to calib020_0001>
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import signal
import sys
import threading
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

from . import search_distillation_d2_generate_authorization as GATE
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-generate-record/1"
CORPUS_KIND = "search-distillation-d2-corpus/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-stage2-card.md"
CARD_SHA256 = "73bbee34e73dff6e0ebebc3e706b5df95ac1918ade87622eacec956ba941205b"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-generate"
PLAN = "docs/superpowers/evidence/search-distillation-d2-generate-run/plan.json"
CORPUS_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"     # outside the repository
FROZEN = {"phases": 12, "games_per_phase": 10, "n_simulations": 1600, "phase_bytes": 84 << 20}
EXIT_DONE, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated, Seams = M.Void, M.Terminated, M.Seams


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, evidence exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED is closed: generation is not authorized")


def held_out(slot: int) -> bool:
    return slot % 10 == 9


def phase_slots(spec: Mapping[str, int], p: int) -> range:
    n = spec["games_per_phase"]
    return range(p * n, (p + 1) * n)


def _execute(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, corpus_dir: pathlib.Path,
             rec: Dict[str, Any], log) -> None:
    import numpy as np

    from . import search_distillation_d2 as D
    from . import training_capture_serial as S
    from . import training_replay_capture as C

    n_slots = spec["phases"] * spec["games_per_phase"]
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
    rec["phases"], rec["games"] = [], []
    log("loaded")

    for p in range(spec["phases"]):
        slots = phase_slots(spec, p)
        context = S.build_context(run_id=f"search-distillation-d2:stage2:{corpus_dir.name}", phase_id=f"phase_{p:02d}",
                                  seeds={s: plan["base"] + s for s in slots}, model_id=M.MODEL_ID,
                                  model_digest=digest, model_source={"checkpoint": M.CHECKPOINT, "sha1": sha1},
                                  mcts_config=cfg, code=code, phase_bytes=spec["phase_bytes"],
                                  play_kwargs=M.DIAGNOSTICS_OFF)
        if p == 0:
            rec["configuration"] = context["configuration"]
            rec["modules"] = seams.loaded_modules()

        def timed(evaluator_, **kw):
            slot = kw["game_id"]
            log("game_start", phase=p, game=slot, start=kw["start_player"])
            t0 = time.monotonic()
            game = seams.play_game(evaluator_, **kw)
            secs = time.monotonic() - t0
            g = {"phase": p, "game": slot, "seed": plan["base"] + slot, "held_out": held_out(slot),
                 "start_player": kw["start_player"], "plies": game.n_moves, "winner": game.winner,
                 "draw_reason": game.draw_reason, "seconds": round(secs, 3),
                 "seconds_per_ply": round(secs / max(game.n_moves, 1), 4)}
            rec["games"].append(g)
            log("game_end", **g)
            return game

        phase_dir = corpus_dir / f"phase_{p:02d}"
        out = S.run_serial_capture(evaluator=evaluator, network=network, capture_dir=str(phase_dir),
                                   context=context, mcts_config=cfg, play_game=timed)
        _ctx, records, ids = D.load_records(str(phase_dir))
        learner = [pos for g in out["games"] for pos in g.positions]
        if len(records) != len(learner):
            raise Void("reader", f"phase {p}: {len(records)} rebuilt records for {len(learner)} learner rows")
        misaligned = [i for i, (r, q) in enumerate(zip(records, learner))
                      if not (np.array_equal(r.board_tensor, q.board_tensor) and r.to_move == q.to_move
                              and list(r.legal_moves) == list(q.legal_moves) and r.visit_counts == list(q.visit_counts)
                              and r.outcome == q.outcome)]
        if misaligned:
            raise Void("alignment", f"phase {p}: {len(misaligned)} records differ from the learner rows")
        rows = [r for env in C.load_phase(str(phase_dir))[1] for r in env["rows"]]
        meta = out_dir / "phases" / f"phase_{p:02d}"
        meta.mkdir(parents=True)
        for f in C.PHASE_FILES:
            shutil.copyfile(phase_dir / f, meta / f)
        finalized = (phase_dir / "finalized.json").read_bytes()
        rec["phases"].append({**out["stats"], "phase": p, "dir": phase_dir.name, "games": list(slots),
                              "context_sha256": out["context_sha256"], "finalized_sha256": M._sha(finalized),
                              "records": len(records), "rows_with_inherited_child_visits":
                                  sum(sum(r["child_visits_before"]) > 0 for r in rows)})
        log("phase_end", phase=p, records=len(records))
        del out, records, learner, rows

    rec["model_digest_after"] = S.network_state_digest(network)
    rec["modes_at_end"] = M._require_eval(network, "after the last game")
    if rec["model_digest_after"] != digest:
        raise Void("model", "the network changed during generation")
    games = rec["games"]
    rec["totals"] = {"games": len(games), "plies": sum(g["plies"] for g in games),
                     "game_seconds": round(sum(g["seconds"] for g in games), 3),
                     "records": sum(ph["records"] for ph in rec["phases"]),
                     "bytes": sum(ph["bytes"] for ph in rec["phases"])}
    corpus = {"kind": CORPUS_KIND, "corpus_dir": str(corpus_dir), "eligible_for_training": True,
              "model_digest": digest, "seeds": rec["seeds"],
              "phases": [{k: ph[k] for k in ("phase", "dir", "games", "context_sha256", "finalized_sha256",
                                              "records")} for ph in rec["phases"]],
              "training_games": [s for s in range(n_slots) if not held_out(s)],
              "held_out_games": [s for s in range(n_slots) if held_out(s)]}
    rec["corpus_sha256"] = M._write_json_new(out_dir / "corpus.json", corpus)


def conclude(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, corpus_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 1's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    corpus_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "corpus_dir": str(corpus_dir), "status": None, "written_by": "script",
                           "eligible_for_training": False, "strength_verdict": None,
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
        log("start")
        _execute(spec, seams, out_dir, corpus_dir, rec, log)
        rec.update(status="COMPLETED", eligible_for_training=True)
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
    """THE PUBLIC ENTRY. No seam: the frozen spec, the real sources, the pinned card."""
    _require_gate()
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corpus = CORPUS_PARENT / f"corpus-{stamp}"
    if os.path.lexists(corpus):
        raise Refused(f"{corpus} exists")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, corpus, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "corpus_dir": rec["corpus_dir"], "totals": rec.get("totals")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_generate run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
