"""PLAY-PATH TIMING ON THIS MACHINE -- ms per simulation and seconds per 400-simulation game for calib020_0001.
(prepared 2026-10-21; committing it and running it are separate approvals)

Re-measures, on the machine that runs it, the timings forward-looking documents rely on. The old figures stay true
for the old machine (Apple M3 Pro); this run adds new ones beside them.
* The network: calib020_0001 (SHA-1 pinned), through eval_runner's own factory -- eval mode since c8cb2aa6,
  compile=True, the default (Metal) device. 0 modules in training mode is checked after the load.
* The search: MCTS(evaluator, cfg_from(EvalConfig(**PROTOCOL), 400), random.Random()) and
  search_with_root(state, add_noise=False), one MCTS per colour per game, as eval_runner.play_eval_game builds
  them. No setting is changed. NO SEED: random.Random() is unseeded; the moves come from the record, so the stream
  only orders ties inside a search.
* The positions: every position (plies 0 .. n-1) of ONE committed colour-reversed pair of inference-mode-match
  games (games 2k and 2k+1): the pair whose total length is closest to the median pair total, ties to the lowest k.
  Each recorded move is replayed and must be legal, by the recorded player, with the recorded n_legal. No game is
  played and no move is chosen.
* Timing: one warm-up infer before any search (as search-profile/01_profile.py); every search timed (wall and
  process CPU); load average before and after each game. Statistics are given with and without the first search.
* The record: MEASURED figures; DERIVED figures, labelled as derived (the methods note's budget hours at entry 15's
  mean plies per game, counted from its committed records; the pilot's label hours for 1,600 x 1,600 and
  400 x 3,200 simulations, assuming the per-simulation cost does not depend on the simulation count); the old
  machine's figures beside them as REFERENCE; this machine's chip, cores, memory, GPU cores and macOS version.
  The pilot's generation line (200 x 205 s) is NOT derived: Stage B's G phase (eval_runner.run_game_tasks, a400 vs
  b400, mean 57 plies) implied ~3.6 s per ply against ~1.1 s per ply in the match games on the same path, an
  unexplained discrepancy, so it stays OPEN.
Refuses if another pytest or the other project's guarded suite is running: this run records first timings.
  --checkpoint PATH   absolute path to calib020_0001 (checkpoints/ is not in a linked worktree)
  --dry               preflight only: pair, replay, SHA-1, PROTOCOL, hardware, contention. No MLX, no model, no file.
  --out DIR           the real run. DIR must exist and hold no record.json; writes DIR/record.json, create-only.
Exit 0 COMPLETED / 3 STOP_ERROR / 4 REFUSED.
"""
import hashlib
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.GPU.alphazero import search_distillation_stage_b as B  # noqa: E402
from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTS  # noqa: E402

SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
SIMS = 400
SOURCE = "docs/superpowers/evidence/inference-mode-match/games"
ENTRY15 = "docs/superpowers/evidence/root-halving-match/games"
N_GAMES = 32
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}
BUDGET_GAMES = {"60%": 1018, "55%": 4070}                   # methods note (2026-10-20): conservative Hoeffding budgets
LABELS = {"training": (1600, 1600), "held_out": (400, 3200)}   # the pilot estimate: positions x simulations each
REFERENCE_OLD_MACHINE = {                                   # true for the old machine; never overwritten
    "machine": "Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GPU cores, 36 GB",
    "ms_per_simulation_profile_b8cf2adf": 3.16,
    "profile_note": "three hand-built positions, BatchNorm in TRAINING mode (before c8cb2aa6): not a baseline",
    "s_per_match_game_entry15": 66.6,
    "budget_hours": {"60%": 19, "55%": 75},
    "stage_b_G_s_per_game": 205, "stage_b_G_mean_plies": 57,
    "pilot_hours": {"generation_200_x_205_s": 11.4, "training_labels_at_4.07_ms": 2.9, "held_out_labels_at_3.75_ms": 1.3},
}


class Refused(Exception):
    pass


def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True).stdout.strip()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def load_games(rel):
    games = sorted((json.loads(p.read_text()) for p in (ROOT / rel).glob("game_*.json")), key=lambda g: g["game_idx"])
    if [g["game_idx"] for g in games] != list(range(N_GAMES)):
        raise Refused(f"{rel}: game_idx is not 0..{N_GAMES - 1}")
    return games


def choose_pair(games):
    """The colour-reversed pair (2k, 2k+1) whose total plies is closest to the median pair total; ties -> lowest k."""
    totals = [len(games[2 * k]["moves"]) + len(games[2 * k + 1]["moves"]) for k in range(N_GAMES // 2)]
    median = statistics.median(totals)
    k = min(range(len(totals)), key=lambda i: (abs(totals[i] - median), i))
    return k, totals, median


def replay(game):
    """The states BEFORE each recorded move; each move legal, by the recorded player, with the recorded n_legal."""
    if game["board_size"] != 24:
        raise Refused(f"game {game['game_idx']} is not 24x24")
    s, out = B.replay_state([]), []
    for i, m in enumerate(game["moves"]):
        legal = s.legal_moves()
        move = (int(m["row"]), int(m["col"]))
        if m["ply"] != i or m["player"] != s.to_move or len(legal) != m["n_legal"] or move not in set(legal):
            raise Refused(f"game {game['game_idx']} ply {i}: the replay does not reproduce the record")
        out.append(s)
        s = s.apply_move(move)
    return out


def contention():
    """Other pytest runs or the other project's guarded suite, by full command line."""
    mine = os.getpid()
    rows = (line.strip().split(None, 1) for line in sh("ps", "-axo", "pid=,command=").splitlines() if line.strip())
    return [f"{pid} {cmd[:120]}" for pid, cmd in rows
            if int(pid) != mine and ("-m pytest" in cmd or "guarded_suite" in cmd)]


def hardware():
    keys = ("machdep.cpu.brand_string", "hw.ncpu", "hw.nperflevels", "hw.perflevel0.name", "hw.perflevel0.physicalcpu",
            "hw.perflevel1.name", "hw.perflevel1.physicalcpu", "hw.memsize")
    hw = {k: sh("sysctl", "-n", k) for k in keys}
    sp = sh("system_profiler", "SPHardwareDataType", "SPDisplaysDataType")
    hw["system_profiler"] = [line.strip() for line in sp.splitlines()
                             if any(t in line for t in ("Model Name", "Model Identifier", "Chip", "Total Number of Cores",
                                                        "Memory:", "Chipset Model"))]
    hw["macos"] = {k: sh("sw_vers", f"-{k}") for k in ("productName", "productVersion", "buildVersion")}
    hw["platform"] = platform.platform()
    return hw


def preflight(ckpt):
    """Everything checkable without MLX or the model."""
    if (B.PROTOCOL["board_size"], B.PROTOCOL["max_moves"], B.PROTOCOL["mcts_sims"]) != (24, 280, SIMS):
        raise Refused(f"PROTOCOL is not 24x24 / 280 plies / {SIMS} simulations")
    busy = contention()
    if busy:
        raise Refused(f"other heavy compute is running: {busy}")
    games = load_games(SOURCE)
    k, totals, median = choose_pair(games)
    pair = [{"game_idx": g["game_idx"], "states": replay(g)} for g in (games[2 * k], games[2 * k + 1])]
    entry15 = load_games(ENTRY15)
    plies15 = statistics.mean(len(g["moves"]) for g in entry15)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    sha1 = hashlib.sha1(ckpt.read_bytes()).hexdigest()
    if sha1 != SHA1:
        raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
    choice = {"source": SOURCE, "pair": k, "games": [2 * k, 2 * k + 1], "pair_totals": totals, "median_total": median,
              "rule": "pair total closest to the median pair total; ties -> lowest k"}
    return pair, choice, plies15, sha1


def warmup(ev, state):
    """ONE infer on the first position's root, packed as _expand_batch packs a single leaf (as 01_profile.py)."""
    import numpy as np
    legal = state.legal_moves()
    board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in legal]], dtype=np.int32)
    cols = np.array([[c for _, c in legal]], dtype=np.int32)
    mask = np.ones((1, len(legal)), dtype=np.float32)
    t = time.perf_counter()
    ev.infer(board, rows, cols, mask, state.active_size)
    return time.perf_counter() - t


def stats(ms):
    s = sorted(ms)
    return {"n": len(s), "mean": statistics.mean(s), "median": statistics.median(s),
            "p90": s[min(len(s) - 1, int(0.9 * len(s)))], "min": s[0], "max": s[-1]}


def measure(rec, pair, ckpt, plies15):
    import mlx.core
    import numpy
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["environment"].update({"mlx": mlx.core.__version__, "numpy": numpy.__version__,
                               "default_device": str(mlx.core.default_device())})
    t = time.perf_counter()
    ev = _default_evaluator_factory(str(ckpt))
    rec["load_s"] = time.perf_counter() - t
    training = sum(bool(m.training) for m in ev.network.modules())
    rec["modules_training_after_load"] = training
    if training:
        raise RuntimeError(f"{training} modules in training mode after the factory load")
    rec["warmup_infer_s"] = warmup(ev, pair[0]["states"][0])
    cfg = cfg_from(EvalConfig(**B.PROTOCOL), SIMS)
    rec["games"], searches = [], []
    for g in pair:
        mcts = {"red": MCTS(ev, cfg, random.Random()), "black": MCTS(ev, cfg, random.Random())}   # unseeded
        before, g0, per = os.getloadavg(), time.perf_counter(), []
        for s in g["states"]:
            legal = s.legal_moves()
            w0, c0 = time.perf_counter_ns(), time.process_time_ns()
            counts, _value, root = mcts[s.to_move].search_with_root(s, add_noise=False)
            wall, cpu = time.perf_counter_ns() - w0, time.process_time_ns() - c0
            if root.visit_count != SIMS or sum(counts.values()) != SIMS or sorted(counts) != sorted(legal):
                raise RuntimeError(f"game {g['game_idx']} ply {s.ply}: not {SIMS} simulations over the legal list")
            per.append({"ply": s.ply, "to_move": s.to_move, "legal": len(legal), "wall_ms": wall / 1e6,
                        "process_cpu_ms": cpu / 1e6})
        searches += per
        rec["games"].append({"game_idx": g["game_idx"], "plies": len(per), "loop_wall_s": time.perf_counter() - g0,
                             "search_wall_s": sum(p["wall_ms"] for p in per) / 1e3,
                             "loadavg_before": before, "loadavg_after": os.getloadavg(), "searches": per})
    per_sim = [p["wall_ms"] / SIMS for p in searches]
    ms_mean = statistics.mean(per_sim[1:])
    s_per_ply = sum(p["wall_ms"] for p in searches[1:]) / 1e3 / (len(searches) - 1)
    rec["measured"] = {
        "ms_per_simulation_all": stats(per_sim), "ms_per_simulation_excluding_first": stats(per_sim[1:]),
        "first_search_ms": searches[0]["wall_ms"], "s_per_ply_excluding_first": s_per_ply,
        "s_per_recorded_game": {g["game_idx"]: g["search_wall_s"] for g in rec["games"]},
        "process_cpu_share": sum(p["process_cpu_ms"] for p in searches) / sum(p["wall_ms"] for p in searches)}
    s_game15 = s_per_ply * plies15
    rec["derived"] = {
        "note": "DERIVED from the measured figures above, not measured: assumptions stated per line",
        "entry15_mean_plies": plies15,
        "s_per_game_at_entry15_plies": s_game15,
        "budget_hours": {k: n * s_game15 / 3600 for k, n in BUDGET_GAMES.items()},
        "budget_assumption": "search time only, at entry 15's mean game length; game-loop overhead not included",
        "pilot_label_hours": {k: n * sims * ms_mean / 3.6e6 for k, (n, sims) in LABELS.items()},
        "label_assumption": "per-simulation cost at 400 simulations applies at 1,600 and 3,200 simulations",
        "pilot_generation": "OPEN -- not derived: Stage B's G phase implied ~3.6 s/ply vs ~1.1 s/ply in match games "
                            "on the same path (unexplained)"}


def main(argv):
    dry = "--dry" in argv
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else pathlib.Path("")
    out = None if dry else (pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None)
    if not dry and (out is None or not out.is_dir() or (out / "record.json").exists()):
        print("REFUSED: --out must name an existing directory without record.json (or pass --dry)", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "play-path-timing/1", "script": "docs/superpowers/evidence/play-path-timing/01_timing.py",
           "reference_old_machine": REFERENCE_OLD_MACHINE, "strength_verdict": None}
    t0 = time.perf_counter()
    try:
        pair, choice, plies15, sha1 = preflight(ckpt)
        hw = hardware()
        if dry:
            print(json.dumps({"choice": choice, "plies": [len(g["states"]) for g in pair], "entry15_mean_plies": plies15,
                              "sha1_verified": sha1, "hardware": hw}, indent=1))
            print(f"DRY OK in {time.perf_counter() - t0:.2f} s; no MLX, no model, nothing written")
            return EXIT["COMPLETED"]
        rec.update({"head": git("rev-parse", "HEAD"), "dirty_paths": git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(ckpt), "sha1_verified": sha1}, "protocol": dict(B.PROTOCOL),
                    "sims_per_position": SIMS, "seed": "none: MCTS gets an unseeded random.Random()",
                    "choice": choice, "hardware": hw,
                    "environment": {"python": sys.version, "executable": sys.executable}})
        measure(rec, pair, ckpt, plies15)
        rec["contention_after"] = contention()
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    rec["total_seconds"] = time.perf_counter() - t0
    if not dry:
        with open(out / "record.json", "x") as fh:
            fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        if code == 0:
            m = rec["measured"]["ms_per_simulation_excluding_first"]
            print(f"COMPLETED: {m['n']} timed searches, mean {m['mean']:.2f} ms/simulation, "
                  f"{rec['derived']['s_per_game_at_entry15_plies']:.1f} s per game at entry 15's length")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
