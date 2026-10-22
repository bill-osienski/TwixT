"""SEARCH DISTILLATION D2 -- STAGE 1: the two-game live measurement. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-stage1-card.md`, pinned below by sha256. Nothing runs on
import. `run` refuses unless `search_distillation_d2_measure_authorization.SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED`
is open, read at call time.

WHAT THE RUN DOES (card sections 2-3): reads the seed plan (two slots, every seed ACCOUNTED, available and not
test-only); requires mirror augmentation off; verifies calib020_0001 by SHA-1, loads it and puts it in evaluation
mode (no module training); checks every loaded repository module is committed and from this tree; binds the live
digest in a capture context and plays the two games through training_capture_serial.run_serial_capture at the
frozen configuration (1,600 simulations, train()'s other defaults, both goal-completion diagnostics off), timing
each game; then requires evaluation mode again and reopens the finalized phase through the Stage 0 reader.
MEASUREMENT ONLY: no score, no strength claim; the games are never training or evaluation material.

The record is written inside the boundary on every exit. SIGTERM (the wrapper's 30-minute ceiling, or a signal it
forwards) becomes a VOID at step "terminated"; a run killed before it can write gets its record from the wrapper.

    python -m scripts.GPU.alphazero.search_distillation_d2_measure run --checkpoint <absolute path to calib020_0001>
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from . import search_distillation_d2_measure_authorization as GATE

ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORD_KIND = "search-distillation-d2-measure-record/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-stage1-card.md"
CARD_SHA256 = "04218c03813e58826b67412188f664552249083dd8055f73f7fd028c7a6d72cf"
CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CHECKPOINT_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
MODEL_ID = "calib020_0001"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-measure"
PLAN = "docs/superpowers/evidence/search-distillation-d2-measure-run/plan.json"
FROZEN = {"games": 2, "n_simulations": 1600, "phase_bytes": 20 << 20}
DIAGNOSTICS_OFF = {"goal_completion_emit_enabled": False, "goal_completion_record_enabled": False}
EXIT_DONE, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2


class Void(Exception):
    def __init__(self, step: str, detail: str):
        super().__init__(f"[{step}] {detail}")
        self.step, self.detail = step, detail


class Terminated(BaseException):
    """SIGTERM inside the record boundary (not an Exception, so only conclude catches it)."""


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, evidence exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED is closed: the measurement is not authorized")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json_new(path: pathlib.Path, obj: Any) -> str:
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return _sha(data)


def mcts_config_for(spec: Mapping[str, int]):
    """Card section 3: n_simulations from the spec, every other field the dataclass default (= train()'s)."""
    from .mcts import MCTSConfig
    cfg = MCTSConfig(n_simulations=spec["n_simulations"])
    defaults = (cfg.eval_batch_size, cfg.pending_virtual_visits, cfg.stall_flush_sims, cfg.c_puct, cfg.fpu_value,
                cfg.dirichlet_alpha, cfg.dirichlet_eps, cfg.temp_threshold_ply, cfg.temp_high, cfg.temp_low)
    if defaults != (14, 8, 16, 1.5, 0.0, 0.3, 0.25, 20, 1.0, 0.1):
        raise Void("configuration", f"MCTSConfig defaults are not train()'s: {defaults}")
    return cfg


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_plan: Callable[[], Mapping[str, Any]]
    seed_ok: Callable[[int], bool]
    mirror_prob: Callable[[], float]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]
    make_evaluator: Callable[[Any], Any]
    loaded_modules: Callable[[], Dict[str, str]]
    play_game: Callable[..., Any]


def _modes(network) -> Dict[str, Any]:
    import mlx.nn as nn
    mods = network.named_modules()
    return {"modules": len(mods), "modules_in_training_mode": sum(bool(m.training) for _, m in mods),
            "batchnorm": sum(isinstance(m, nn.BatchNorm) for _, m in mods)}


def _require_eval(network, when: str) -> Dict[str, Any]:
    m = _modes(network)
    if m["modules_in_training_mode"] or not m["batchnorm"]:
        raise Void("mode", f"{when}: the network is not wholly in evaluation mode ({m})")
    return m


def _execute(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    from . import search_distillation_d2 as D
    import numpy as np

    from . import training_capture_serial as S
    from . import training_replay_capture as C

    plan = seams.read_plan()
    if plan.get("n_slots") != spec["games"] or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {spec['games']} slots from an int base: {plan}")
    seeds = {g: plan["base"] + g for g in range(spec["games"])}
    bad = [s for s in seeds.values() if not seams.seed_ok(s)]
    if bad:
        raise Void("seeds", f"seeds not ACCOUNTED, available and schedulable: {bad}")
    rec["seeds"] = {str(g): s for g, s in seeds.items()}
    if seams.mirror_prob() != 0.0:
        raise Void("mirror", f"mirror augmentation is {seams.mirror_prob()}; the card needs TWIXT_MIRROR_PROB=0")
    sha1 = seams.checkpoint_sha1()
    if sha1 != CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": CHECKPOINT, "sha1": sha1}

    t = time.monotonic()
    network = seams.load_network()
    network.eval()
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["modes_at_load"] = _require_eval(network, "after load")
    evaluator = seams.make_evaluator(network)
    digest = S.network_state_digest(network)
    rec["model_digest_before"] = digest
    rec["value_adapter"] = network.value_adapter is not None
    log("loaded", seconds=rec["load_seconds"])

    cfg = mcts_config_for(spec)
    context = S.build_context(run_id="search-distillation-d2:stage1-measure", phase_id="measure", seeds=seeds,
                              model_id=MODEL_ID, model_digest=digest,
                              model_source={"checkpoint": CHECKPOINT, "sha1": sha1}, mcts_config=cfg,
                              code=S.code_identity(ROOT), phase_bytes=spec["phase_bytes"],
                              play_kwargs=DIAGNOSTICS_OFF)
    rec["configuration"] = context["configuration"]
    rec["modules"] = seams.loaded_modules()                  # after every import the run makes before playing

    games = []

    def timed(evaluator_, **kw):
        log("game_start", game=kw["game_id"], start=kw["start_player"])
        t0 = time.monotonic()
        game = seams.play_game(evaluator_, **kw)
        secs = time.monotonic() - t0
        games.append({"game": kw["game_id"], "seed": seeds[kw["game_id"]], "start_player": kw["start_player"],
                      "plies": game.n_moves, "winner": game.winner, "draw_reason": game.draw_reason,
                      "seconds": round(secs, 3), "seconds_per_ply": round(secs / max(game.n_moves, 1), 4)})
        rec["games"] = games
        log("game_end", **games[-1])
        return game

    capture_dir = out_dir / "capture"
    out = S.run_serial_capture(evaluator=evaluator, network=network, capture_dir=str(capture_dir), context=context,
                               mcts_config=cfg, play_game=timed)
    rec["capture"] = {**out["stats"], "context_sha256": out["context_sha256"]}
    rec["model_digest_after"] = S.network_state_digest(network)
    rec["modes_at_end"] = _require_eval(network, "after the last game")
    t = time.monotonic()
    _ctx, records, _ids = D.load_records(str(capture_dir))
    secs_read = round(time.monotonic() - t, 3)
    learner = [p for g in out["games"] for p in g.positions]           # the rows train() would have trained on
    if len(records) != len(learner) or len(records) != sum(g["plies"] for g in games):
        raise Void("reader", f"the Stage 0 reader rebuilt {len(records)} records for {len(learner)} learner rows")
    misaligned = [i for i, (r, p) in enumerate(zip(records, learner))
                  if not (np.array_equal(r.board_tensor, p.board_tensor) and r.to_move == p.to_move
                          and list(r.legal_moves) == list(p.legal_moves) and r.visit_counts == list(p.visit_counts)
                          and r.outcome == p.outcome)]
    if misaligned:
        raise Void("alignment", f"{len(misaligned)} rebuilt records differ from the learner rows, first {misaligned[0]}")
    _ctx, envelopes = C.load_phase(str(capture_dir))
    rows = [r for env in envelopes for r in env["rows"]]
    rec["stage0_reader"] = {"records": len(records), "seconds": secs_read, "aligned_with_learner_rows": True,
                            "rows_with_inherited_child_visits": sum(sum(r["child_visits_before"]) > 0 for r in rows),
                            "rows": len(rows)}
    plies = sum(g["plies"] for g in games)
    secs = sum(g["seconds"] for g in games)
    rec["totals"] = {"plies": plies, "game_seconds": round(secs, 3), "seconds_per_ply": round(secs / plies, 4)}


def conclude(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json is written on every exit. SIGTERM inside it raises Terminated, recorded as a
    VOID at step "terminated"; while the record is written SIGTERM is ignored; the previous handler is restored."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "written_by": "script", "strength_verdict": None,
                           "limitation": "measurement only; the two games are never training or evaluation material",
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
        _execute(spec, seams, out_dir, rec, log)
        rec["status"], code = "COMPLETED", EXIT_DONE
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # card section 6: any exception is a VOID, and it is recorded
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
            _write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    from . import e4_screen_reference as REG
    from . import self_play as sp

    def read_plan():
        return json.loads((ROOT / PLAN).read_text())

    def seed_ok(s):
        return REG.seed_is_accounted(s) and not REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s)

    def load_network():
        from .probe_eval import load_network_for_scoring
        return load_network_for_scoring(str(checkpoint), verbose=False)[0]

    def make_evaluator(network):
        from .local_evaluator import LocalGPUEvaluator
        return LocalGPUEvaluator(network)                 # as train() builds it

    def loaded_modules():
        ours = {n: m.__file__ for n, m in list(sys.modules.items())
                if n.split(".")[0] in ("scripts", "tests") and getattr(m, "__file__", None)}
        outside = sorted(n for n, f in ours.items() if not pathlib.Path(f).resolve().is_relative_to(ROOT))
        if outside:
            raise Void("modules", f"repository modules loaded from outside this tree: {outside}")
        rel = sorted({str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values()})
        if _git("status", "--porcelain", "--untracked-files=all", "--", *rel):
            raise Void("modules", "a loaded repository module is uncommitted or differs from HEAD")
        return {p: _sha((ROOT / p).read_bytes()) for p in rel}

    return Seams(read_plan=read_plan, seed_ok=seed_ok, mirror_prob=lambda: sp._MIRROR_PROB,
                 checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                 load_network=load_network, make_evaluator=make_evaluator, loaded_modules=loaded_modules,
                 play_game=sp.play_game)


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the frozen spec, the real sources, the pinned card."""
    _require_gate()
    if _sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, head=_git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "totals": rec.get("totals")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_measure run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
