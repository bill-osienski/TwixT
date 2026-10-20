"""ROOT-HALVING MATCH -- the root-only sequential-halving variant vs ordinary search, 400 simulations each. An
exploratory gameplay screen: not promotion evidence and not a strength verdict.

Both agents are calib020_0001 (SHA-1 pinned), each from its OWN eval_runner factory load (evaluation mode checked
at load and after the last game; two distinct network objects):
* halving -- scripts/GPU/alphazero/root_sequential_halving.RootSequentialHalvingMCTS (frozen variant, ledger
  Post-closure entry 15): ordinary search and readout for plies 0-19; from ply 20 its root allocation and its own
  final choice (last_choice), which replaces the readout;
* ordinary -- mcts.MCTS with the readout.
This measures root allocation PLUS its final decision rule against ordinary search with its readout, not
allocation alone. Settings otherwise: Stage B's PROTOCOL unchanged -- 24x24, 280-ply cap, the opening-temperature
readout (1.0 for 20 plies, then 0.1), no root noise, one game at a time. 16 colour-reversed pairs = 32 games from
eval_runner's agent builder (halving red in even games), game g with seed plan.base + g from a block registered in
ACCOUNTED.

THE GAME LOOP (play_game / play_and_build) is a copy of eval_runner.play_eval_game / _play_and_build_result with
three differences only: each colour's MCTS class comes from its agent; the move is the search's last_choice when it
supplies one (readout_overrode_leader False), else eval_readout.select exactly as before; and an optional on_search
callback times each search. Separate search/readout RNG streams, per-ply budget checks against the mover's own
budget, colour binding and replay records are unchanged; 03_loop_check.py checks the copy against the original.

COST, per agent and per game, read-only: searches, halving searches (last_choice supplied), seconds inside
search_with_root, network infer calls, positions evaluated and seconds inside the network infer.

Checks (any failure stops the run; no score is reported): the plan is 32 slots; every seed ACCOUNTED, available and
not test-only; the checkpoint hash; the loaded repository modules committed, unmodified and from this tree; both
networks and all their BatchNorm layers in evaluation mode at load and after the last game; the loop's own checks.
FROZEN SCREENING RULE (operational, not statistical proof): the halving agent's points over the 32 games >= 24
(mean pair score >= 0.75) -> ADVANCE_TO_PRICED_STRENGTH_COMPARISON; below -> STOP_CONFIGURATION. No timing veto. An
incomplete cohort gets no score. Pairs are the statistical unit; the unclipped Hoeffding 95% interval (half-width
0.3395) is descriptive and CONDITIONAL on the assumed independence of the 16 pairs. Prediction on record (ledger
entry 15): fails the screen.

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
import random
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
PAIRS, GAMES, SIMS = 16, 32, 400
POINTS_BAR = 24                                            # of 32: mean pair score >= 0.75
HALF_WIDTH = math.sqrt(math.log(2 / 0.05) / (2 * PAIRS))   # 0.33953...: pair means in [0, 1]
HALVING, ORDINARY = "halving", "ordinary"
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
T0 = time.monotonic()
COST_KEYS = ("searches", "halving_searches", "search_seconds", "network_infer_calls", "network_positions",
             "network_seconds")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def play_game(red_eval, black_eval, config, seed, capture=False, red_readout=None, black_readout=None,
              red_sims=None, black_sims=None, red_cls=None, black_cls=None, on_search=None):
    """eval_runner.play_eval_game, copied; the three differences are marked DIFF."""
    from scripts.GPU.alphazero import eval_readout
    from scripts.GPU.alphazero.eval_integrity import validate_ply
    from scripts.GPU.alphazero.eval_replay import ply_record
    from scripts.GPU.alphazero.eval_runner import cfg_from, readout_from_eval_config, root_child_stats
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS

    red_rd = red_readout or readout_from_eval_config(config)
    black_rd = black_readout or readout_from_eval_config(config)
    red_n = config.mcts_sims if red_sims is None else red_sims
    black_n = config.mcts_sims if black_sims is None else black_sims
    mcts_red = (red_cls or MCTS)(red_eval, cfg_from(config, red_n), random.Random(seed ^ 0xA5A5A5))        # DIFF
    mcts_black = (black_cls or MCTS)(black_eval, cfg_from(config, black_n), random.Random(seed ^ 0x5A5A5A))  # DIFF
    readout_rng_red = random.Random(seed ^ 0xC3C3C3)
    readout_rng_black = random.Random(seed ^ 0x3C3C3C)
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    records = [] if capture else None
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        is_red = state.to_move == "red"
        mcts = mcts_red if is_red else mcts_black
        rdt = red_rd if is_red else black_rd
        rng = readout_rng_red if is_red else readout_rng_black
        t = time.perf_counter()                                                                     # DIFF
        counts, root_value, root = mcts.search_with_root(state, add_noise=False)
        choice = getattr(mcts, "last_choice", None)                                                 # DIFF
        if on_search is not None:                                                                   # DIFF
            on_search(is_red, time.perf_counter() - t, choice is not None)
        top2 = eval_readout.top_two(root_child_stats(counts, root))
        validate_ply(ply, red_n if is_red else black_n, root.visit_count,
                     root_value, top2)
        if choice is None:                                                                          # DIFF
            move, overrode = eval_readout.select(counts, ply, rdt, rng, top2=top2)
        else:
            move, overrode = choice, False
        if capture:
            records.append(ply_record(ply, state.to_move, move, counts,
                                      root_value, top2=top2,
                                      overrode_leader=overrode))
        state = state.apply_move(move)
        ply += 1
    winner = state.winner()
    if winner is not None:
        reason = "win"
    elif ply >= config.max_moves:
        reason = "state_cap"
    elif not state.legal_moves():
        reason = "board_full"
    else:
        reason = "unknown_error"
    return winner, reason, ply, records


def play_and_build(task, red, black, config, capture, replay_dir, classes=None, on_search=None):
    """eval_runner._play_and_build_result, copied; it calls play_game with each agent's MCTS class (DIFF)."""
    from scripts.GPU.alphazero.eval_integrity import validate_game_binding
    from scripts.GPU.alphazero.eval_replay import build_replay_dict, write_replay
    from scripts.GPU.alphazero.eval_runner import agent_sims, make_result

    classes = classes or {}
    red_rd = task.red_agent.readout if task.red_agent is not None else None
    black_rd = task.black_agent.readout if task.black_agent is not None else None
    winner, reason, nm, records = play_game(
        red, black, config, task.seed, capture=capture,
        red_readout=red_rd, black_readout=black_rd,
        red_sims=agent_sims(task.red_agent, config),
        black_sims=agent_sims(task.black_agent, config),
        red_cls=classes.get(task.red_agent.agent_id), black_cls=classes.get(task.black_agent.agent_id),
        on_search=on_search)
    result = make_result(task, winner, reason, nm, config=config)
    validate_game_binding(result, task)
    if records is not None:
        result.replay_path = write_replay(
            replay_dir,
            build_replay_dict(result, task.seed, config.board_size, records))
    return result


def modes(net):
    """Training flags only (copied from mirror-averaging-match/01_match.py); reads no running statistic."""
    import mlx.nn as nn

    mods = net.named_modules()
    return {"network_training": bool(net.training), "modules": len(mods),
            "modules_in_training_mode": sum(bool(m.training) for _, m in mods),
            "batchnorm": [{"name": n, "training": bool(m.training)} for n, m in mods if isinstance(m, nn.BatchNorm)]}


def check_modes(bases, when):
    m = {k: modes(ev.network) for k, ev in bases.items()}
    bad = [k for k in bases if m[k]["network_training"] or not m[k]["batchnorm"]
           or any(x["training"] for x in m[k]["batchnorm"])]
    if bad or bases[HALVING].network is bases[ORDINARY].network:
        raise Stop(f"{when}: not in evaluation mode {bad}, or the network copies are shared")
    return m


def instrument(bases):
    """Read-only network counters on each agent's own evaluator. Returns cost[agent][key]."""
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
    return cost


def rounded(cost):
    return {k: {x: round(v[x], 3) for x in COST_KEYS} for k, v in cost.items()}


def summary(results):
    if sorted(r.game_idx for r in results) != list(range(GAMES)):
        raise Stop(f"incomplete cohort: {len(results)} of {GAMES} games -- no score")
    s = {r.game_idx: r.red_score if r.red_agent_id == HALVING else r.black_score for r in results}
    pairs = [(s[2 * k] + s[2 * k + 1]) / 2 for k in range(PAIRS)]
    points = sum(s.values())
    mean = sum(pairs) / PAIRS
    return {"halving_wins": sum(r.winner_agent_id == HALVING for r in results),
            "ordinary_wins": sum(r.winner_agent_id == ORDINARY for r in results),
            "draws": sum(r.winner is None for r in results),
            "caps": sum(r.reason == "state_cap" for r in results),
            "board_full": sum(r.reason == "board_full" for r in results),
            "halving_points": points, "points_bar": POINTS_BAR,
            "screening_outcome": ("ADVANCE_TO_PRICED_STRENGTH_COMPARISON" if points >= POINTS_BAR
                                  else "STOP_CONFIGURATION"),
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
    from scripts.GPU.alphazero.mcts import MCTS
    from scripts.GPU.alphazero.root_sequential_halving import RootSequentialHalvingMCTS
    from scripts.GPU.alphazero.search_distillation_stage_b import PROTOCOL
    import mlx.core as mx
    import numpy

    rec["environment"] = {"python": sys.version, "platform": platform.platform(),
                          "mlx": mx.__version__, "numpy": numpy.__version__, "executable": sys.executable}
    if (PROTOCOL["board_size"], PROTOCOL["max_moves"], PROTOCOL["mcts_sims"]) != (24, 280, SIMS):
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
    bases = {HALVING: ER._default_evaluator_factory(str(ckpt)), ORDINARY: ER._default_evaluator_factory(str(ckpt))}
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["modes_at_load"] = check_modes(bases, "after load")
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
    a = ER.AgentSpec(HALVING, str(ckpt), rd, mcts_sims=SIMS)
    b = ER.AgentSpec(ORDINARY, str(ckpt), rd, mcts_sims=SIMS)
    classes = {HALVING: RootSequentialHalvingMCTS, ORDINARY: MCTS}
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)],
                       "classes": {k: f"{v.__module__}.{v.__qualname__}" for k, v in classes.items()}}
    cost = instrument(bases)
    tasks = ER.build_agent_pairing_tasks("root-halving", a, b, GAMES, base_seed=plan["base"])
    results, rec["games"] = [], []
    for task in tasks:
        slot = task.seed - plan["base"]
        colour = {True: task.red_agent.agent_id, False: task.black_agent.agent_id}

        def on_search(is_red, secs, halving, _c=colour):
            c = cost[_c[is_red]]
            c["searches"] += 1
            c["halving_searches"] += halving
            c["search_seconds"] += secs

        log("game_start", game=task.game_idx, slot=slot, red=task.red_agent.agent_id)
        before = {k: dict(v) for k, v in cost.items()}
        t = time.monotonic()
        r = play_and_build(task, bases[task.red_agent.agent_id], bases[task.black_agent.agent_id], cfg, True,
                           str(HERE / "games"), classes, on_search)
        secs = round(time.monotonic() - t, 3)
        game_cost = {k: {x: round(cost[k][x] - before[k][x], 3) for x in COST_KEYS} for k in cost}
        if game_cost[ORDINARY]["halving_searches"] or any(game_cost[k]["searches"] == 0 for k in cost):
            raise Stop(f"game {task.game_idx}: a search was attributed to the wrong agent")
        results.append(r)
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs, "cost": game_cost})
        log("game_end", game=task.game_idx, slot=slot, seconds=secs, n_moves=r.n_moves, reason=r.reason)
        mx.eval()
        gc.collect()
        mx.clear_cache()
    rec["modes_at_end"] = check_modes(bases, "after the last game")
    rec["cost"] = rounded(cost)
    rec["screen"] = summary(results)
    return "COMPLETED"


if __name__ == "__main__":
    log_fh = (HERE / "progress.jsonl").open("x")          # a second run stops here, exit 1

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - T0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    rec = {"kind": "root-halving-match-record/1", "script": "docs/superpowers/evidence/root-halving-match/01_match.py",
           "head": git("rev-parse", "HEAD"), "strength_verdict": None,
           "limitation": "exploratory screen; not promotion evidence; root allocation plus its decision rule",
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
        print(f"COMPLETED: halving {s['halving_wins']}-{s['draws']}-{s['ordinary_wins']} ordinary, caps {s['caps']}; "
              f"halving points {s['halving_points']}/{GAMES} (bar {POINTS_BAR}) -> {s['screening_outcome']}; "
              f"mean pair score {s['mean_pair_score']:.4f} +/- {HALF_WIDTH:.4f} (descriptive, assumes independence)")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
