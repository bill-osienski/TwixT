"""MIRROR-AVERAGING MATCH -- 200 simulations with mirror averaging vs 400 ordinary simulations. An exploratory
gameplay screen: not promotion evidence and not a strength verdict.

Both agents are calib020_0001 (SHA-1 pinned), each from its OWN eval_runner factory load (evaluation mode checked
at load and after the last game; two distinct network objects):
* averaged -- MirrorAveragingEvaluator (scripts/GPU/alphazero/mirror_averaging.py, checked at bf6d03ad/3b11645a):
  each leaf evaluation averages the policy distributions and values of the position and its left-right reflection
  (two underlying network calls); 200 simulations;
* ordinary -- the plain evaluator; 400 simulations.
APPROXIMATELY INFERENCE-BUDGET MATCHED, not equal wall time: a nonterminal search can need 402 underlying network
evaluations for averaging vs 401 ordinarily (root expansion included); terminal simulations reduce either count.
Settings otherwise: Stage B's PROTOCOL unchanged -- 24x24, 280-ply cap, the opening-temperature readout (1.0 for 20
plies, then 0.1), no root noise, one game at a time. 16 colour-reversed pairs = 32 games from eval_runner's agent
builder (averaged red in even games), game g with seed plan.base + g from the block registered in ACCOUNTED from
mirror-averaging-match-seed-survey. Each game goes through eval_runner._play_and_build_result (per-ply checks
against each mover's own budget, colour binding, replay), MLX buffers released between games.

COST, recorded separately per agent and per game, read-only: search calls (moves), leaf infer calls the search
made, underlying network infer calls and the positions they evaluated (the network evaluations), seconds inside
the underlying infer, and seconds inside search_with_root (wrapped at the class, keyed by the MCTS's evaluator).

Checks (any failure stops the run; no score is reported): the plan is 32 slots; every seed ACCOUNTED, available and
not test-only; the checkpoint hash; the loaded repository modules committed, unmodified and from this tree; both
networks and all their BatchNorm layers in evaluation mode at load and after the last game; the runner's own
checks; every search's agent identified.
FROZEN SCREENING RULE (operational, not statistical proof): the averaged agent's points over the 32 games >= 24
(mean pair score >= 0.75) -> CONSIDER_CONFIRMATORY; below -> STOP_CONFIGURATION (no budget, weight or orientation
change). An incomplete cohort gets no score. The unclipped Hoeffding 95% interval (half-width 0.3395) is
descriptive and CONDITIONAL on the assumed independence of the 16 pairs.

One-shot and create-only (progress.jsonl, record.json, games/); run ONLY through 02_wrapper.sh (the 5,390 s ceiling
plus 10 s kill grace and the outer status live there). Exit 0 COMPLETED / 1 a repeat run / 2 STOP / 3 STOP_ERROR /
4 REFUSED.

    python -B <this file> --checkpoint <absolute path to calib020_0001>
"""
import dataclasses
import gc
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
PAIRS, GAMES = 16, 32
SIMS = {"averaged": 200, "ordinary": 400}
POINTS_BAR = 24                                            # of 32: mean pair score >= 0.75
HALF_WIDTH = math.sqrt(math.log(2 / 0.05) / (2 * PAIRS))   # 0.33953...: pair means in [0, 1]
AVERAGED, ORDINARY = "averaged", "ordinary"
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
T0 = time.monotonic()
COST_KEYS = ("searches", "search_seconds", "leaf_infer_calls", "network_infer_calls", "network_positions",
             "network_seconds")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def modes(net):
    """Training flags only (copied from inference-mode-match/01_match.py); reads no running statistic."""
    import mlx.nn as nn

    mods = net.named_modules()
    return {"network_training": bool(net.training), "modules": len(mods),
            "modules_in_training_mode": sum(bool(m.training) for _, m in mods),
            "batchnorm": [{"name": n, "training": bool(m.training)} for n, m in mods if isinstance(m, nn.BatchNorm)]}


def check_modes(bases, when):
    m = {k: modes(ev.network) for k, ev in bases.items()}
    bad = [k for k in bases if m[k]["network_training"] or not m[k]["batchnorm"]
           or any(x["training"] for x in m[k]["batchnorm"])]
    if bad or bases[AVERAGED].network is bases[ORDINARY].network:
        raise Stop(f"{when}: not in evaluation mode {bad}, or the network copies are shared")
    return m


def instrument(bases, agents):
    """Read-only cost counters. bases: agent -> the plain evaluator whose network calls are counted;
    agents: agent -> the evaluator the agent's search uses. Returns (cost, tally) with cost[agent][key]."""
    from scripts.GPU.alphazero.mcts import MCTS

    cost = {k: dict.fromkeys(COST_KEYS, 0) for k in bases}
    for k, base in bases.items():
        real = base.infer

        def net_infer(boards, *a, _k=k, _real=real, **kw):
            t = time.perf_counter()
            out = _real(boards, *a, **kw)
            c = cost[_k]
            c["network_seconds"] += time.perf_counter() - t
            c["network_infer_calls"] += 1
            c["network_positions"] += int(boards.shape[0])
            return out

        base.infer = net_infer
    for k, ev in agents.items():
        if ev is not bases[k]:
            real_leaf = ev.infer

            def leaf(*a, _k=k, _real=real_leaf, **kw):
                cost[_k]["leaf_infer_calls"] += 1
                return _real(*a, **kw)

            ev.infer = leaf
    owner = {id(ev): k for k, ev in agents.items()}
    real_search = MCTS.search_with_root

    def search_with_root(self, *a, **kw):
        k = owner.get(id(self.evaluator))
        if k is None:
            raise Stop("a search ran with an unidentified evaluator")
        t = time.perf_counter()
        out = real_search(self, *a, **kw)
        cost[k]["search_seconds"] += time.perf_counter() - t
        cost[k]["searches"] += 1
        return out

    MCTS.search_with_root = search_with_root
    return cost


def finish_cost(cost):
    """The ordinary agent's search calls the network directly: its leaf calls ARE its network calls."""
    out = {k: dict(v) for k, v in cost.items()}
    out[ORDINARY]["leaf_infer_calls"] = out[ORDINARY]["network_infer_calls"]
    for v in out.values():
        v["search_seconds"], v["network_seconds"] = round(v["search_seconds"], 3), round(v["network_seconds"], 3)
    return out


def summary(results):
    if sorted(r.game_idx for r in results) != list(range(GAMES)):
        raise Stop(f"incomplete cohort: {len(results)} of {GAMES} games -- no score")
    s = {r.game_idx: r.red_score if r.red_agent_id == AVERAGED else r.black_score for r in results}
    pairs = [(s[2 * k] + s[2 * k + 1]) / 2 for k in range(PAIRS)]
    points = sum(s.values())
    mean = sum(pairs) / PAIRS
    return {"averaged_wins": sum(r.winner_agent_id == AVERAGED for r in results),
            "ordinary_wins": sum(r.winner_agent_id == ORDINARY for r in results),
            "draws": sum(r.winner is None for r in results),
            "caps": sum(r.reason == "state_cap" for r in results),
            "board_full": sum(r.reason == "board_full" for r in results),
            "averaged_points": points, "points_bar": POINTS_BAR,
            "screening_outcome": "CONSIDER_CONFIRMATORY" if points >= POINTS_BAR else "STOP_CONFIGURATION",
            "screening_note": "operational screening threshold, not statistical proof of superiority",
            "pair_scores": pairs, "mean_pair_score": mean, "half_width": HALF_WIDTH,
            "interval_unclipped": [mean - HALF_WIDTH, mean + HALF_WIDTH],
            "interval_note": "descriptive; pairs are the sampling unit; conditional on the assumed independence of "
                             "the 16 pairs, which distinct seeds and colour reversal do not themselves establish"}


def main(rec, argv, log):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
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
    bases = {AVERAGED: ER._default_evaluator_factory(str(ckpt)), ORDINARY: ER._default_evaluator_factory(str(ckpt))}
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["modes_at_load"] = check_modes(bases, "after load")
    agents = {AVERAGED: MirrorAveragingEvaluator(bases[AVERAGED]), ORDINARY: bases[ORDINARY]}
    log("loaded", seconds=rec["load_seconds"],
        modules_in_training_mode={k: v["modules_in_training_mode"] for k, v in rec["modes_at_load"].items()})

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
    a = ER.AgentSpec(AVERAGED, str(ckpt), rd, mcts_sims=SIMS[AVERAGED])
    b = ER.AgentSpec(ORDINARY, str(ckpt), rd, mcts_sims=SIMS[ORDINARY])
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)]}
    cost = instrument(bases, agents)
    tasks = ER.build_agent_pairing_tasks("mirror-averaging", a, b, GAMES, base_seed=plan["base"])
    results, rec["games"] = [], []
    for task in tasks:
        slot = task.seed - plan["base"]
        log("game_start", game=task.game_idx, slot=slot, red=task.red_agent.agent_id)
        before = finish_cost(cost)
        t = time.monotonic()
        r = ER._play_and_build_result(task, agents[task.red_agent.agent_id], agents[task.black_agent.agent_id],
                                      cfg, True, str(HERE / "games"))
        secs = round(time.monotonic() - t, 3)
        after = finish_cost(cost)
        game_cost = {k: {x: round(after[k][x] - before[k][x], 3) for x in COST_KEYS} for k in after}
        results.append(r)
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs, "cost": game_cost})
        log("game_end", game=task.game_idx, slot=slot, seconds=secs, n_moves=r.n_moves, reason=r.reason)
        mx.eval()
        gc.collect()
        mx.clear_cache()
    rec["modes_at_end"] = check_modes(bases, "after the last game")
    rec["cost"] = finish_cost(cost)
    rec["screen"] = summary(results)
    return "COMPLETED"


if __name__ == "__main__":
    log_fh = (HERE / "progress.jsonl").open("x")          # a second run stops here, exit 1

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - T0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    rec = {"kind": "mirror-averaging-match-record/1", "script": "docs/superpowers/evidence/mirror-averaging-match/01_match.py",
           "head": git("rev-parse", "HEAD"), "strength_verdict": None,
           "limitation": "exploratory screen; not promotion evidence; approximately inference-budget matched",
           "loadavg_before": os.getloadavg()}
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
        s = rec["screen"]
        print(f"COMPLETED: averaged {s['averaged_wins']}-{s['draws']}-{s['ordinary_wins']} ordinary, caps {s['caps']}; "
              f"averaged points {s['averaged_points']}/{GAMES} (bar {POINTS_BAR}) -> {s['screening_outcome']}; "
              f"mean pair score {s['mean_pair_score']:.4f} +/- {HALF_WIDTH:.4f} (descriptive, assumes independence)")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
