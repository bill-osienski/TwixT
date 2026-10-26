"""MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- C1, THE COMPOSITION-AND-COST CHECK, in ONE fresh process launched
with MLX_ENABLE_TF32=0. BUILT; run only through its wrapper (TERM at 890 s, KILL 10 s later: the card's 15 minutes).

Card: `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §4 and §5 C1, pinned below
by sha256, with the reused code and the panel's game files. Nothing runs on import, and importing this module loads no
MLX. `run` refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless
`mirror_halving_c1_authorization.MIRROR_HALVING_C1_AUTHORIZED` is open (read at call time). No game is played, no
schedulable seed is drawn, and nothing is trained.

THE PANEL: every position (plies 0 .. n-1) of entry 19's pair, inference-mode-match games 0 and 1 (104 positions: 40
before ply 20, 64 from it); each recorded move is replayed and must be legal, by the recorded player, with the
recorded n_legal. THE CHECK RNG: every search gets a fresh random.Random(s), s = TEST_ONLY_SEED_INTERVALS[0][0]
(membership verified), so the two arms of every comparison start from identical RNG states.

IN ORDER, on two separate eval-mode loads of calib020_0001 (one per side):
1. entry 10's averaging correctness check (docs/superpowers/evidence/mirror-averaging/01_check.py), run unchanged
   (its own load, reference and 1e-5 tolerance);
2. control (a), plies < 20: the candidate equals averaging alone (MCTS over the averaging wrapper) at 200 -- visits,
   root value and no last_choice, bitwise;
3. control (b), plies >= 20: halving over a pass-through wrapper equals halving over the plain evaluator at 200 --
   visits, root value and last_choice, bitwise. (2 and 3 are the card's named identity controls; they are the only
   component-only searches this entry makes.)
4. COST: one warm-up search per side on the panel's first position (excluded), then one search per side on each
   position, the incumbent first on even indices and the candidate first on odd. Totals (card §4): underlying network
   positions and search_with_root seconds per side; each guard a RATIO OF TOTALS (evaluations <= 1.05, search
   seconds <= 1.10). The candidate must use its halving choice at every panel position from ply 20.
VERDICT: PASS only if every check holds; otherwise STOP, naming each failed check. The measurements price later runs
(DERIVED projections, labelled); they change no budget or parameter. A failure to complete, an exception or SIGTERM is
a VOID. One shot; no retry.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.mirror_halving_c1 run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from . import mirror_halving_c1_authorization as GATE
from . import mirror_halving_combination as K
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "mirror-halving-c1-record/1"
CARD = K.CARD
CARD_SHA256 = "34d2924fb565641d6bfb350904f00626849e99b1881002003219f962a94bd221"
EVIDENCE_DIR = "docs/superpowers/evidence/mirror-halving-c1"
SETTING, SETTING_VALUE = "MLX_ENABLE_TF32", "0"
VERSIONS = {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.9"}
ENTRY10 = "docs/superpowers/evidence/mirror-averaging/01_check.py"
PANEL_SOURCE = "docs/superpowers/evidence/inference-mode-match/games"
PANEL_GAMES = {0: "b2e17263247af080b3ec0093582ea30391f367bcd1045687da608f87d9dc38bf",
               1: "edfb72cbe2d869804d38e14d8b10960577884dd2c407725de6dfba0bd671d869"}
PANEL_SIZE = {"positions": 104, "before_from_ply": 40, "from_ply": 64}
PINNED = {"scripts/GPU/alphazero/mirror_halving_combination.py": "907ce4342f95c28f30f71dfaf50406994a93cfaf70929c89f0a1c0bedca1a176",
          "scripts/GPU/alphazero/mirror_averaging.py": "1f23ac6adeb6ea0a33b7fe5500edeb2c9929a9781089106d5925a99817340aef",
          "scripts/GPU/alphazero/root_sequential_halving.py":
              "0bec4ef1914b25541f5247c5dd6c0002cea171b2902c629e52cfc034e9402790",
          ENTRY10: "3cad00901d3c71d7e345a560c489d0701b36cc77166c6f5b5175caa4eb327401",
          **{f"{PANEL_SOURCE}/game_{g:06d}.json": sha for g, sha in PANEL_GAMES.items()}}
#: DERIVED projections only: entry 15's mean game length, as entry 19 used it (an assumption, not a measurement here)
PLIES_PER_GAME_ASSUMED = 59.8
PROJECTED_GAMES = {"screen": 2 * K.SCREEN["pairs"], "strength": 2 * K.STRENGTH["pairs"]}
STRENGTH_REPRICE_HOURS = 30.0
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = 0, 2, 3, 4
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, an output exists, or a bad
    argument."""


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    versions: Callable[[], Dict[str, str]]
    checkpoint_sha1: Callable[[], str]
    load_evaluator: Callable[[], Any]
    require_eval: Callable[[Any, str], Dict[str, Any]]
    entry10_check: Callable[[], Dict[str, Any]]
    loaded_modules: Callable[[], Dict[str, str]]
    read_bytes: Callable[[str], bytes]


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def panel(read_bytes: Callable[[str], bytes]) -> List[Any]:
    """The states BEFORE each recorded move of games 0 and 1, every file the pinned bytes and every move replayed."""
    from .game.twixt_state import TwixtState

    states = []
    for g, sha in PANEL_GAMES.items():
        rel = f"{PANEL_SOURCE}/game_{g:06d}.json"
        raw = read_bytes(rel)
        if M._sha(raw) != sha:
            raise Void("panel", f"{rel} is not the pinned bytes")
        game = json.loads(raw)
        if game["board_size"] != 24:
            raise Void("panel", f"{rel} is not 24x24")
        s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
        for i, m in enumerate(game["moves"]):
            move, legal = (int(m["row"]), int(m["col"])), s.legal_moves()
            if m["ply"] != i or m["player"] != s.to_move or len(legal) != m["n_legal"] or move not in set(legal):
                raise Void("panel", f"{rel} ply {i}: the replay does not reproduce the record")
            states.append(s)
            s = s.apply_move(move)
    before = sum(s.ply < K.VARIANT["FROM_PLY"] for s in states)
    if (len(states), before, len(states) - before) != tuple(PANEL_SIZE.values()):
        raise Void("panel", f"{len(states)} positions ({before} before ply 20), not {PANEL_SIZE}")
    return states


def _same(a, b, choice: bool) -> Dict[str, Any]:
    """Bitwise comparison of two searches: visit counts, root value and (for halving) last_choice."""
    (m1, c1, v1, _r1, _s1), (m2, c2, v2, _r2, _s2) = a, b
    row = {"visits_equal": c1 == c2, "root_value_equal": v1 == v2,
           "last_choice": [getattr(m1, "last_choice", None), getattr(m2, "last_choice", None)]}
    if choice:
        row["choice_equal"] = row["last_choice"][0] == row["last_choice"][1] and row["last_choice"][0] is not None
    else:
        row["choice_equal"] = row["last_choice"][0] is None
    row["equal"] = row["visits_equal"] and row["root_value_equal"] and row["choice_equal"]
    return row


def _execute(seams: Seams, rec: Dict[str, Any], log) -> None:
    from . import e4_screen_reference as REF
    from . import eval_runner as ER
    from . import root_sequential_halving as RSH
    from .external_learning_stage3 import PROTOCOL
    from .mcts import MCTS
    from .mirror_averaging import MirrorAveragingEvaluator

    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    seed = REF.TEST_ONLY_SEED_INTERVALS[0][0]
    if not REF.seed_is_test_only(seed):
        raise Void("check_rng", f"{seed} is not in TEST_ONLY_SEED_INTERVALS")
    rec["check_rng"] = {"seed": seed, "rule": "a fresh random.Random(seed) per search, both arms; test-only band"}
    rec["variant"] = K.check_variant(RSH)
    config = ER.EvalConfig(**PROTOCOL)
    if (config.board_size, config.max_moves, config.mcts_sims) != (24, 280, K.INCUMBENT_SIMS):
        raise Void("protocol", "PROTOCOL is not 24x24 / 280 plies / 400 simulations")
    states = panel(seams.read_bytes)
    rec["panel"] = {"source": PANEL_SOURCE, "games": sorted(PANEL_GAMES), **PANEL_SIZE}
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint_sha1"] = sha1
    checks: Dict[str, bool] = {}

    log("entry10_start")
    e10 = seams.entry10_check()
    rec["entry10"] = e10
    checks["entry10_correctness"] = e10.get("verdict") == "COMPLETED" and e10.get("answer") == "PASS"
    log("entry10_end", answer=e10.get("answer"), verdict=e10.get("verdict"))

    t = time.monotonic()
    base = {K.CANDIDATE: seams.load_evaluator(), K.INCUMBENT: seams.load_evaluator()}
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    if base[K.CANDIDATE] is base[K.INCUMBENT] or base[K.CANDIDATE].network is base[K.INCUMBENT].network:
        raise Void("load", "the two sides share an evaluator or a network")
    rec["modes_at_load"] = {k: seams.require_eval(ev, "after load") for k, ev in base.items()}
    rec["modules"] = seams.loaded_modules()
    cost = {k: K.new_cost() for k in base}
    for k, ev in base.items():
        K.instrument(ev, cost[k])
    cand = base[K.CANDIDATE]
    cfg200 = ER.cfg_from(config, K.CANDIDATE_SIMS)
    make = {K.CANDIDATE: lambda r: K.candidate_search(cand, config, r),
            K.INCUMBENT: lambda r: K.incumbent_search(base[K.INCUMBENT], config, r)}
    log("loaded", seconds=rec["load_seconds"])

    rows_a, rows_b = [], []
    for i, s in enumerate(states):
        if s.ply < K.VARIANT["FROM_PLY"]:
            a = K.timed_search(make[K.CANDIDATE], s, seed)
            b = K.timed_search(lambda r: MCTS(MirrorAveragingEvaluator(cand), cfg200, r), s, seed)
            rows_a.append({"index": i, "ply": s.ply, **_same(a, b, choice=False)})
        else:
            a = K.timed_search(lambda r: RSH.RootSequentialHalvingMCTS(K.PassThroughEvaluator(cand), cfg200, r), s,
                               seed)
            b = K.timed_search(lambda r: RSH.RootSequentialHalvingMCTS(cand, cfg200, r), s, seed)
            rows_b.append({"index": i, "ply": s.ply, **_same(a, b, choice=True)})
    rec["control_a"], rec["control_b"] = rows_a, rows_b
    checks["control_a_identical"] = len(rows_a) == PANEL_SIZE["before_from_ply"] and all(r["equal"] for r in rows_a)
    checks["control_b_identical"] = len(rows_b) == PANEL_SIZE["from_ply"] and all(r["equal"] for r in rows_b)
    log("controls_end", a=sum(r["equal"] for r in rows_a), b=sum(r["equal"] for r in rows_b))

    for k in (K.CANDIDATE, K.INCUMBENT):                      # warm-up: one search per side, excluded
        K.timed_search(make[k], states[0], seed)
    before = {k: dict(v) for k, v in cost.items()}
    rows = []
    for i, s in enumerate(states):
        row = {"index": i, "ply": s.ply}
        for k in ((K.INCUMBENT, K.CANDIDATE) if i % 2 == 0 else (K.CANDIDATE, K.INCUMBENT)):
            n0 = cost[k]["network_positions"]
            m, _counts, _value, _root, secs = K.timed_search(make[k], s, seed)
            halving = getattr(m, "last_choice", None) is not None
            cost[k]["searches"] += 1
            cost[k]["search_seconds"] += secs
            cost[k]["halving_searches"] += halving
            row[k] = {"seconds": round(secs, 6), "network_positions": cost[k]["network_positions"] - n0,
                      "halving": halving}
        rows.append(row)
    totals = {k: K.delta(cost[k], before[k]) for k in cost}
    rec["cost_positions"], rec["cost_totals"] = rows, totals
    g = K.guards(totals[K.CANDIDATE], totals[K.INCUMBENT])
    rec["guards"] = g
    checks["evaluations_within_guard"] = g["evaluations_within"]
    checks["search_seconds_within_guard"] = g["search_seconds_within"]
    checks["halving_at_every_position_from_ply_20"] = (
        totals[K.CANDIDATE]["halving_searches"] == PANEL_SIZE["from_ply"]
        and totals[K.INCUMBENT]["halving_searches"] == 0)
    rec["modes_at_end"] = {k: seams.require_eval(ev, "at the end") for k, ev in base.items()}
    per = {k: totals[k]["search_seconds"] / totals[k]["searches"] for k in totals}
    game_s = PLIES_PER_GAME_ASSUMED / 2 * (per[K.CANDIDATE] + per[K.INCUMBENT])
    rec["derived"] = {"label": "DERIVED: search seconds only, at an ASSUMED 59.8 plies per game (entry 15's mean)",
                      "seconds_per_search": per, "search_seconds_per_game": game_s,
                      "search_hours": {k: n * game_s / 3600 for k, n in PROJECTED_GAMES.items()}}
    rec["derived"]["strength_over_reprice_hours"] = rec["derived"]["search_hours"]["strength"] > STRENGTH_REPRICE_HOURS
    rec["checks"] = checks
    rec["failed_checks"] = [k for k, ok in checks.items() if not ok]
    rec["verdict"] = "PASS" if not rec["failed_checks"] else "STOP"
    log("cost_end", guards={k: g[k] for k in ("evaluations_ratio", "search_seconds_ratio")}, verdict=rec["verdict"])


def conclude(seams: Seams, out_dir: pathlib.Path, head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "head": head,
                           "status": None, "verdict": None, "written_by": "script", "strength_verdict": None,
                           "setting": {SETTING: os.environ.get(SETTING)},
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
        _execute(seams, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", verdict=None,
                   void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", verdict=None, void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", verdict=None, void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", verdict=None, void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec.get("verdict"))
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _versions() -> Dict[str, str]:
    import mlx.core as mx
    import numpy as np
    return {"mlx": mx.__version__, "numpy": np.__version__, "python": platform.python_version()}


def _entry10(checkpoint: pathlib.Path) -> Dict[str, Any]:
    """Entry 10's check, unchanged: its own module from its own file, its main() with its own record."""
    spec = importlib.util.spec_from_file_location("mirror_averaging_entry10_check", ROOT / ENTRY10)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sub: Dict[str, Any] = {"kind": "mirror-averaging-check/1", "script": ENTRY10, "tolerance": mod.TOL}
    try:
        sub["verdict"] = mod.main(sub, ["--checkpoint", str(checkpoint)])
    except Exception as e:  # noqa: BLE001 -- its REFUSED / STOP / error ends the check, recorded, not raised
        sub["verdict"], sub["reason"] = type(e).__name__, str(e)
    return sub


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    from . import eval_runner as ER

    return Seams(versions=_versions,
                 checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                 load_evaluator=lambda: ER._default_evaluator_factory(str(checkpoint)),
                 require_eval=lambda ev, when: M._require_eval(ev.network, when),
                 entry10_check=lambda: _entry10(checkpoint),
                 loaded_modules=M._real_seams(checkpoint).loaded_modules,
                 read_bytes=lambda rel: (ROOT / rel).read_bytes())


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the pinned card, code and panel, the real sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.MIRROR_HALVING_C1_AUTHORIZED is not True:
        raise Refused("MIRROR_HALVING_C1_AUTHORIZED is closed: C1 is not authorized")
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
    code, rec = conclude(_real_seams(ckpt), out, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "verdict": rec.get("verdict"),
                      "failed_checks": rec.get("failed_checks"), "guards": rec.get("guards"),
                      "derived": rec.get("derived")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.mirror_halving_c1 run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
