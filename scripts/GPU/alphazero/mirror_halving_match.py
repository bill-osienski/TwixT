"""MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- S, THE GROSS-REGRESSION SCREEN, and T, THE STRENGTH COMPARISON, each
in ONE fresh process launched with MLX_ENABLE_TF32=0. BUILT with both gates closed; their wrappers, plans and seed
copies follow C1, whose measurements set the ceilings (card §5 B, §6).

Card: `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §2, §4 and §5 S/T, pinned
below by sha256, with the reused code. Nothing runs on import, and importing this module loads no MLX. `run` refuses
unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless that kind's gate
(`mirror_halving_screen_authorization` / `mirror_halving_strength_authorization`) is open (read at call time).

ADMISSION: the screen only after C1's committed record is COMPLETED with verdict PASS under this card; the strength
comparison only after that AND the screen's committed record is COMPLETED with verdict CONTINUE under this card.

THE GAMES: the candidate (halving over the averaging wrapper, 200 simulations) against the unchanged incumbent (MCTS,
400), each from its OWN eval-mode load of calib020_0001; Stage B's protocol (24x24, 280-ply cap, opening-temperature
readout, no root noise); colour-reversed pairs from eval_runner's agent builder (candidate red in even games), game g
with seed plan.base + g from a block registered in ACCOUNTED, through the shared copied loop
(`mirror_halving_combination.play_and_build`). Per game: cost per side, and the incumbent never supplies a halving
choice. A `game_start` event (game = slot) is logged BEFORE each game, so the closing derivation exposes an
interrupted game; each game's row is appended to games.jsonl as it ends.

VERDICTS (mirror_halving_combination): the screen STOPs below a mean pair score of 0.40, else CONTINUE (no strength
claim); the strength comparison PASSes only if the exact Hoeffding lower bound exceeds 0.5 AND both cost guards hold
over the whole run (above the bound with a guard failed: NOT_COMPUTE_MATCHED); otherwise NON_PASS. An incomplete cohort
gets no score: any failure, exception or SIGTERM is a VOID. One shot each; no retry, top-up or extension.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.mirror_halving_match run screen|strength --checkpoint <path>
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from . import mirror_halving_c1 as C1
from . import mirror_halving_combination as K
from . import mirror_halving_screen_authorization as SCREEN_GATE
from . import mirror_halving_strength_authorization as STRENGTH_GATE
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "mirror-halving-match-record/1"
CARD, CARD_SHA256 = C1.CARD, C1.CARD_SHA256
SETTING, SETTING_VALUE, VERSIONS = C1.SETTING, C1.SETTING_VALUE, C1.VERSIONS
KINDS = {"screen": {"pairs": K.SCREEN["pairs"], "gate": (SCREEN_GATE, "MIRROR_HALVING_SCREEN_AUTHORIZED"),
                    "evidence": "docs/superpowers/evidence/mirror-halving-screen",
                    "plan": "docs/superpowers/evidence/mirror-halving-screen-run/plan.json"},
         "strength": {"pairs": K.STRENGTH["pairs"], "gate": (STRENGTH_GATE, "MIRROR_HALVING_STRENGTH_AUTHORIZED"),
                      "evidence": "docs/superpowers/evidence/mirror-halving-strength",
                      "plan": "docs/superpowers/evidence/mirror-halving-strength-run/plan.json"}}
ADMISSION = {"screen": [(C1.EVIDENCE_DIR, "verdict", "PASS")],
             "strength": [(C1.EVIDENCE_DIR, "verdict", "PASS"),
                          (KINDS["screen"]["evidence"], "screen_verdict", "CONTINUE")]}
PINNED = {rel: sha for rel, sha in C1.PINNED.items() if not rel.startswith(C1.PANEL_SOURCE) and rel != C1.ENTRY10}
PINNED["scripts/GPU/alphazero/mirror_halving_c1.py"] = "b23379e59bd31bf51715f41debf028910efbf37f306457bba96902d048fbbde8"
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = 0, 2, 3, 4
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, an output exists, admission, or a
    bad argument."""


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    versions: Callable[[], Dict[str, str]]
    checkpoint_sha1: Callable[[], str]
    load_evaluator: Callable[[], Any]
    require_eval: Callable[[Any, str], Dict[str, Any]]
    loaded_modules: Callable[[], Dict[str, str]]
    read_plan: Callable[[str], Mapping[str, Any]]
    seed_ok: Callable[[int], bool]
    play_and_build: Callable[..., Any]
    between_games: Callable[[], None]


def admitted(kind: str, read_record: Callable[[str], Mapping[str, Any]]) -> Dict[str, str]:
    """{record: verdict} for every earlier step this kind needs, or Refused."""
    seen = {}
    for rel, field, want in ADMISSION[kind]:
        try:
            rec = read_record(rel)
        except FileNotFoundError:
            raise Refused(f"{rel}/record.json does not exist: {kind} is not admitted") from None
        if rec.get("status") != "COMPLETED" or rec.get(field) != want or rec.get("card_sha256") != CARD_SHA256:
            raise Refused(f"{rel}/record.json is not COMPLETED with {field} {want} under this card: {kind} is not "
                          f"admitted")
        seen[rel] = want
    return seen


def _execute(kind: str, seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    from . import eval_runner as ER
    from . import root_sequential_halving as RSH
    from .external_learning_stage3 import PROTOCOL
    from .mcts import MCTS
    from .mirror_averaging import MirrorAveragingEvaluator

    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    rec["variant"] = K.check_variant(RSH)
    config = ER.EvalConfig(**PROTOCOL)
    if (config.board_size, config.max_moves, config.mcts_sims) != (24, 280, K.INCUMBENT_SIMS):
        raise Void("protocol", "PROTOCOL is not 24x24 / 280 plies / 400 simulations")
    pairs, games = KINDS[kind]["pairs"], 2 * KINDS[kind]["pairs"]
    plan = seams.read_plan(KINDS[kind]["plan"])
    if plan.get("n_slots") != games or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {games} slots from an int base: {plan}")
    bad = [plan["base"] + s for s in range(games) if not seams.seed_ok(plan["base"] + s)]
    if bad:
        raise Void("seeds", f"{len(bad)} seeds not ACCOUNTED, available and schedulable, first {bad[0]}")
    rec["seeds"] = {"base": plan["base"], "n_slots": games}
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint_sha1"] = sha1

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
    evaluators = {K.CANDIDATE: MirrorAveragingEvaluator(base[K.CANDIDATE]), K.INCUMBENT: base[K.INCUMBENT]}
    classes = {K.CANDIDATE: RSH.RootSequentialHalvingMCTS, K.INCUMBENT: MCTS}
    rd = ER.readout_from_eval_config(config)
    a = ER.AgentSpec(K.CANDIDATE, M.CHECKPOINT, rd, mcts_sims=K.CANDIDATE_SIMS)
    b = ER.AgentSpec(K.INCUMBENT, M.CHECKPOINT, rd, mcts_sims=K.INCUMBENT_SIMS)
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(config),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)],
                       "classes": {k: f"{v.__module__}.{v.__qualname__}" for k, v in classes.items()},
                       "evaluators": {k: type(v).__name__ for k, v in evaluators.items()}}
    tasks = ER.build_agent_pairing_tasks(f"mirror-halving-{kind}", a, b, games, base_seed=plan["base"])
    log("loaded", seconds=rec["load_seconds"])

    results = []
    with open(out_dir / "games.jsonl", "x") as games_fh:
        for task in tasks:
            colour = {True: task.red_agent.agent_id, False: task.black_agent.agent_id}

            def on_search(is_red, secs, halving, _c=colour):
                c = cost[_c[is_red]]
                c["searches"] += 1
                c["halving_searches"] += halving
                c["search_seconds"] += secs

            log("game_start", game=task.game_idx, slot=task.seed - plan["base"], red=task.red_agent.agent_id)
            before = {k: dict(v) for k, v in cost.items()}
            t = time.monotonic()
            r = seams.play_and_build(task, evaluators[task.red_agent.agent_id], evaluators[task.black_agent.agent_id],
                                     config, True, str(out_dir / "games"), classes, on_search)
            game_cost = {k: K.delta(cost[k], before[k]) for k in cost}
            if game_cost[K.INCUMBENT]["halving_searches"] or any(game_cost[k]["searches"] == 0 for k in cost):
                raise Void("attribution", f"game {task.game_idx}: a search was attributed to the wrong agent")
            row = {**dataclasses.asdict(r), "slot": task.seed - plan["base"],
                   "seconds": round(time.monotonic() - t, 3), "cost": game_cost}
            games_fh.write(json.dumps(row) + "\n")
            games_fh.flush()
            os.fsync(games_fh.fileno())
            results.append(r)
            log("game_end", game=task.game_idx, n_moves=r.n_moves, reason=r.reason)
            seams.between_games()
    rec["modes_at_end"] = {k: seams.require_eval(ev, "at the end") for k, ev in base.items()}
    rec["cost_totals"] = cost
    g = K.guards(cost[K.CANDIDATE], cost[K.INCUMBENT])
    rec["guards"] = g
    scores = K.pair_scores(results, pairs)
    rec["pair_scores"] = scores
    rec["games_summary"] = {"candidate_wins": sum(r.winner_agent_id == K.CANDIDATE for r in results),
                            "incumbent_wins": sum(r.winner_agent_id == K.INCUMBENT for r in results),
                            "draws": sum(r.winner is None for r in results),
                            "state_caps": sum(r.reason == "state_cap" for r in results)}
    if kind == "screen":
        rec["screen"] = K.screen_verdict(scores)
        rec["screen_verdict"] = rec["screen"]["verdict"]
    else:
        rec["strength"] = K.strength_verdict(scores, g)
        rec["strength_verdict"] = rec["strength"]["verdict"]


def conclude(kind: str, seams: Seams, out_dir: pathlib.Path, admission: Mapping[str, str],
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "run": kind, "card": CARD, "card_sha256": CARD_SHA256, "head": head,
                           "admission": dict(admission), "status": None, "written_by": "script",
                           "screen_verdict": None, "strength_verdict": None,
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
        _execute(kind, seams, out_dir, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", screen_verdict=None, strength_verdict=None,
                   void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", screen_verdict=None, strength_verdict=None,
                   void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", screen_verdict=None, strength_verdict=None,
                   void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", screen_verdict=None, strength_verdict=None,
                   void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], screen=rec.get("screen_verdict"), strength=rec.get("strength_verdict"))
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _between_games() -> None:
    import gc

    import mlx.core as mx
    mx.eval()
    gc.collect()
    mx.clear_cache()


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    from . import e4_screen_reference as REG
    from . import eval_runner as ER

    def seed_ok(s):
        return REG.seed_is_accounted(s) and not REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s)

    return Seams(versions=C1._versions,
                 checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                 load_evaluator=lambda: ER._default_evaluator_factory(str(checkpoint)),
                 require_eval=lambda ev, when: M._require_eval(ev.network, when),
                 loaded_modules=M._real_seams(checkpoint).loaded_modules,
                 read_plan=lambda rel: json.loads((ROOT / rel).read_text()),
                 seed_ok=seed_ok, play_and_build=K.play_and_build, between_games=_between_games)


def _read_record(rel: str) -> Mapping[str, Any]:
    """A committed record: it must exist and be unchanged from HEAD."""
    path = f"{rel}/record.json"
    if not (ROOT / path).is_file():
        raise FileNotFoundError(path)
    if M._git("status", "--porcelain", "--untracked-files=all", "--", path):
        raise Refused(f"{path} is uncommitted or differs from HEAD")
    return json.loads((ROOT / path).read_text())


def run(kind: str, checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the gate, the pinned card and code, admission, the real sources."""
    if kind not in KINDS:
        raise Refused(f"unknown run {kind!r}")
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    gate, name = KINDS[kind]["gate"]
    if getattr(gate, name) is not True:
        raise Refused(f"{name} is closed: the {kind} is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    admission = admitted(kind, _read_record)
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / KINDS[kind]["evidence"]
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    code, rec = conclude(kind, _real_seams(ckpt), out, admission, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "run": kind,
                      "screen_verdict": rec.get("screen_verdict"), "strength_verdict": rec.get("strength_verdict"),
                      "guards": rec.get("guards")}))
    return code


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 4 or argv[0] != "run" or argv[1] not in KINDS or argv[2] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.mirror_halving_match run screen|strength "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[1], argv[3])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
