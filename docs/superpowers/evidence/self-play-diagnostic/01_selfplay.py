"""SELF-PLAY DIAGNOSTIC -- four self-play games of calib020_0001 in evaluation mode, played only to find one
concrete, verifiable mistake. DIAGNOSTIC MATERIAL: never held-out evaluation (closing retires the block). No score,
no strength claim.

A copy of evidence/inference-mode-match/01_match.py; the diff: one corrected evaluator (eval_runner's factory,
net.eval()) plays BOTH sides -- evaluation mode keeps no per-call state, so one network is shared -- the slot count
(4), no legacy copy and no score, the titles. Agents `self_a`/`self_b` are labels only: the same checkpoint,
evaluator, budget and readout, so the runner's colour alternation changes nothing but the label.
Settings: Stage B's PROTOCOL unchanged -- 24x24, 400 simulations each side, 280-ply cap, the opening-temperature
readout (1.0 for 20 plies, then 0.1), no root noise, one game at a time. Game g has seed plan.base + g, the block
registered in ACCOUNTED from self-play-diagnostic-seed-survey. Each game goes through
eval_runner._play_and_build_result (per-ply checks, colour binding, replay with root statistics per ply).

Checks (any failure stops the run): the plan is 4 slots; every seed ACCOUNTED, available and not test-only; the
checkpoint hash; the repository modules loaded are committed, unmodified and from this tree; the network and all
its BatchNorm layers in evaluation mode after the load and again after the last game; the runner's own checks.
The diagnosis is a separate read-only step (03_scan.py) over the committed replays.

One-shot and create-only (progress.jsonl, record.json, games/); run ONLY through 02_wrapper.sh (the 1,200 s
ceiling and the outer status live there). Exit 0 COMPLETED / 1 a repeat run / 2 STOP / 3 STOP_ERROR / 4 REFUSED.

    python -B <this file> --checkpoint <absolute path to calib020_0001>
"""
import dataclasses
import gc
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
GAMES = 4
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
T0 = time.monotonic()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def modes(net):
    """Training flags only (copied from search-profile/06_bn_mode.py); reads no running statistic."""
    import mlx.nn as nn

    mods = net.named_modules()
    return {"network_training": bool(net.training), "modules": len(mods),
            "modules_in_training_mode": sum(bool(m.training) for _, m in mods),
            "batchnorm": [{"name": n, "training": bool(m.training)} for n, m in mods if isinstance(m, nn.BatchNorm)]}


def check_modes(ev, when):
    m = modes(ev.network)
    if m["network_training"] or not m["batchnorm"] or any(x["training"] for x in m["batchnorm"]):
        raise Stop(f"{when}: the network is not wholly in evaluation mode")
    return m


def main(rec, argv, log):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.search_distillation_stage_b import PROTOCOL
    import mlx.core as mx
    import numpy

    rec["environment"] = {"python": sys.version, "platform": platform.platform(),
                          "mlx": mx.__version__, "numpy": numpy.__version__, "executable": sys.executable}
    if (PROTOCOL["board_size"], PROTOCOL["max_moves"], PROTOCOL["mcts_sims"]) != (24, 280, 400):
        raise Refused("PROTOCOL is not 24x24 / 280 plies / 400 simulations")
    plan = json.loads((HERE / "plan.json").read_text())
    if plan.get("n_slots") != GAMES or type(plan.get("base")) is not int:
        raise Refused(f"the plan is not exactly {GAMES} slots from an int base")
    for slot in range(GAMES):
        s = plan["base"] + slot
        if not REG.seed_is_accounted(s) or REG.seed_is_unavailable(s) or REG.seed_is_test_only(s):
            raise Refused(f"seed slot {slot} is not ACCOUNTED, available and schedulable")
    sha1 = hashlib.sha1(ckpt.read_bytes()).hexdigest()
    if sha1 != SHA1:
        raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
    rec["checkpoint"] = {"path": str(ckpt), "sha1_verified": sha1}

    t = time.monotonic()
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["modes_at_load"] = check_modes(ev, "after load")
    log("loaded", seconds=rec["load_seconds"], modules_in_training_mode=rec["modes_at_load"]["modules_in_training_mode"])

    # Repository modules loaded so far (this script included) must be committed, unmodified, and from THIS tree.
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] in ("scripts", "tests")) and getattr(m, "__file__", None)}
    outside = sorted(n for n, f in ours.items() if not pathlib.Path(f).resolve().is_relative_to(ROOT))
    if outside:
        raise Refused(f"repository modules loaded from outside this tree: {outside}")
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")

    cfg = ER.EvalConfig(**PROTOCOL)
    rd = ER.readout_from_eval_config(cfg)
    a, b = ER.AgentSpec("self_a", str(ckpt), rd), ER.AgentSpec("self_b", str(ckpt), rd)
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)]}
    tasks = ER.build_agent_pairing_tasks("self-play-diagnostic", a, b, GAMES, base_seed=plan["base"])
    rec["games"] = []
    for task in tasks:
        slot = task.seed - plan["base"]
        log("game_start", game=task.game_idx, slot=slot, red=task.red_agent.agent_id)
        t = time.monotonic()
        r = ER._play_and_build_result(task, ev, ev, cfg, True, str(HERE / "games"))
        secs = round(time.monotonic() - t, 3)
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs})
        log("game_end", game=task.game_idx, slot=slot, seconds=secs, n_moves=r.n_moves, reason=r.reason)
        mx.eval()
        gc.collect()
        mx.clear_cache()
    rec["modes_at_end"] = check_modes(ev, "after the last game")
    return "COMPLETED"


if __name__ == "__main__":
    log_fh = (HERE / "progress.jsonl").open("x")          # a second run stops here, exit 1

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - T0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    rec = {"kind": "self-play-diagnostic-record/1", "script": "docs/superpowers/evidence/self-play-diagnostic/01_selfplay.py",
           "head": git("rev-parse", "HEAD"), "strength_verdict": None,
           "limitation": "diagnostic material only; never held-out evaluation; no score", "loadavg_before": os.getloadavg()}
    log("start", head=rec["head"])
    try:
        verdict = main(rec, sys.argv, log)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        from scripts.GPU.alphazero.eval_integrity import IntegrityError
        verdict = "STOP" if isinstance(e, (Stop, IntegrityError)) else "STOP_ERROR"
        rec["reason"] = f"{type(e).__name__}: {e}"
    rec["verdict"], rec["total_seconds"] = verdict, round(time.monotonic() - T0, 3)
    rec["loadavg_after"] = os.getloadavg()
    log("end", verdict=verdict)
    with (HERE / "record.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        print(f"COMPLETED: {len(rec['games'])} games, plies {[g['n_moves'] for g in rec['games']]}, "
              f"reasons {[g['reason'] for g in rec['games']]}")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
