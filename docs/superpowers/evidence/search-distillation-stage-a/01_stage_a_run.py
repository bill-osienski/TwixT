"""STAGE A: 80 colour-reversed pairs (160 games), calib020_0001 at 1,600 vs 400 simulations.
Card: docs/superpowers/2026-10-11-search-distillation-stage-a-card.md. A copy of the A1 run script
(evidence/search-distillation-a1/01_a1_run.py); its diff is limited to: 160 games and slots, the paths,
outcome-blind progress (game_end carries no winner, score or termination reason), a digest of each game's
ordered move history, and validate_result_set over the whole schedule before COMPLETED.

This script computes NO verdict; 03_stage_a_analysis.py does, from the committed record. One-shot and
create-only; run ONLY through 02_stage_a_wrapper.sh (the 24 h ceiling and the outer status live there).
Exit 0 COMPLETED / 1 a repeat run / 2 STOP_INTEGRITY / 3 STOP_ERROR / 4 REFUSED.

    python -B <this file> --checkpoint <absolute path to calib020_0001>
"""
import dataclasses
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
EXIT = {"COMPLETED": 0, "STOP_INTEGRITY": 2, "STOP_ERROR": 3, "REFUSED": 4}
N_GAMES = 160                                              # 80 pairs
T0 = time.monotonic()
log_fh = (HERE / "progress.jsonl").open("x")              # a second run stops here, exit 1


def log(event, **kw):
    log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - T0, 3), **kw}) + "\n")
    log_fh.flush()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.eval_integrity import validate_result_set
    from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL
    import mlx.core
    import numpy

    rec["environment"] = {"python": sys.version, "platform": platform.platform(),
                          "mlx": mlx.core.__version__, "numpy": numpy.__version__, "executable": sys.executable}
    plan = json.loads((HERE / "plan.json").read_text())
    if plan.get("n_slots") != N_GAMES or type(plan.get("base")) is not int:
        raise Refused(f"the plan is not exactly {N_GAMES} slots from an int base")
    for slot in range(N_GAMES):
        s = plan["base"] + slot
        if not REG.seed_is_accounted(s) or REG.seed_is_unavailable(s):
            raise Refused(f"seed slot {slot} is not ACCOUNTED and available")
    sha1 = hashlib.sha1(ckpt.read_bytes()).hexdigest()
    if sha1 != SHA1:
        raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
    rec["checkpoint"] = {"path": str(ckpt), "sha1_verified": sha1}

    t = time.monotonic()
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    log("loaded", seconds=rec["load_seconds"])

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

    def factory(path):
        assert path == str(ckpt), path
        return ev

    cfg = ER.EvalConfig(**PROTOCOL)
    rd = ER.readout_from_eval_config(cfg)
    hi = ER.AgentSpec("sims1600", str(ckpt), rd, mcts_sims=1600)
    lo = ER.AgentSpec("sims400", str(ckpt), rd, mcts_sims=400)
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(hi), dataclasses.asdict(lo)]}
    tasks = ER.build_agent_pairing_tasks("stage-a", hi, lo, N_GAMES, base_seed=plan["base"])
    rec["games"], results = [], []
    for task in tasks:
        slot = task.seed - plan["base"]
        log("game_start", game=task.game_idx, slot=slot, red=task.red_agent.agent_id)
        t = time.monotonic()
        [r] = ER.run_game_tasks([task], 1, cfg, evaluator_factory=factory,
                                replay_dir=str(HERE / "replays"))
        secs = round(time.monotonic() - t, 3)
        results.append(r)
        moves = json.loads(pathlib.Path(r.replay_path).read_text())["moves"]
        # the ordered move history, as the replay records it
        history = [[m["ply"], m["player"], m["row"], m["col"]] for m in moves]
        transcript = hashlib.sha256(json.dumps(history, separators=(",", ":")).encode()).hexdigest()
        visits = {c: sorted({m["root_total_visits"] for m in moves if m["player"] == c}) for c in ("red", "black")}
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs, "root_total_visits": visits,
                             "transcript_sha256": transcript})
        log("game_end", game=task.game_idx, slot=slot, seconds=secs, n_moves=r.n_moves,       # outcome-blind
            red_mcts_sims=r.red_mcts_sims, black_mcts_sims=r.black_mcts_sims)
    validate_result_set(results, tasks, "sims1600", "sims400")
    rec["result_set_check"] = "validate_result_set passed"
    return "COMPLETED"


if __name__ == "__main__":
    rec = {"kind": "search-distillation-stage-a-record/1",
           "card": "docs/superpowers/2026-10-11-search-distillation-stage-a-card.md",
           "head": git("rev-parse", "HEAD"), "strength_verdict": None}
    log("start", head=rec["head"])
    try:
        verdict = main(rec, sys.argv)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception as e:
        from scripts.GPU.alphazero.eval_integrity import IntegrityError
        verdict = "STOP_INTEGRITY" if isinstance(e, IntegrityError) else "STOP_ERROR"
        rec["reason"] = f"{type(e).__name__}: {e}"
    rec["verdict"], rec["total_seconds"] = verdict, round(time.monotonic() - T0, 3)
    log("end", verdict=verdict)
    with (HERE / "record.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    sys.exit(EXIT[verdict])
