"""THE H-TB CONFIRMATION TEST -- the WHOLE H-TB student in the proven combination against the original proven
combination, 110 fresh colour-reversed pairs (220 games). BUILT with its gate closed; nothing here has run.

Plan: `docs/superpowers/2026-11-06-htb-strength-plan.md`, pinned below by sha256, filling in the pilot card's §9.2 and
§9.3 (`docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md`, its pin unchanged). Nothing runs on import,
and importing this module loads no MLX. `run strength` refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx`
module is loaded when it starts, the gate is open (read at call time), the plan, card and frozen modules (the screen's
module among them) are the pinned bytes, and the committed fit AND screen evidence admit it.

REUSED FROM THE SCREEN (`htb_screen`, the reviewed module that ran once; pinned here, never modified): the student and
reference identities, the fit admission, the seams and their real wiring (the match's loader), the identity check,
the background watch and the opening probe. The loop is the screen's text, which is the cycle-2 screen's (a test
compares all three).

THE RUN (plan §§2-7), in ONE fresh process under its wrapper's hard ceiling:
* versions; the PROTOCOL (24x24, 280 plies); the plan: exactly 220 slots of `htb-strength-plan/1` from an int base,
  every seed ACCOUNTED, available and schedulable, and the block disjoint from the screen's (read from the screen's
  committed plan); calib020_0001's SHA-1; the student file's sha256;
* both sides through the match's loader, digests the fit's at load AND at the end, evaluation mode at load and at the
  end, no shared network, an opening probe that must find their priors different;
* THE RULE (plan §6, card §9.2): lower = mean - sqrt(ln 20 / 220) over the 110 pair scores; PASS only if lower > 0.5 AND
  the evaluations ratio <= 1.05 AND the search-seconds ratio <= 1.10; otherwise NON_PASS, every reason listed.

Any failure, exception or SIGTERM inside the record boundary is a VOID with record.json written by the script. One
shot; no retry, top-up or extension.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_strength run strength --checkpoint <abs path to calib020_0001>
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from . import combination_policy_pilot as CPP
from . import htb_screen as HS
from . import htb_strength_authorization as GATE
from . import mirror_halving_combination as K
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "htb-strength-record/1"
CARD, CARD_SHA256 = HS.CARD, HS.CARD_SHA256                # the pilot card, its pin unchanged
PLAN_DOC = "docs/superpowers/2026-11-06-htb-strength-plan.md"
PLAN_DOC_SHA256 = "92e355a87c5686814aca372da12816364cd025b31b77464332be6b680963d49e"
SETTING, SETTING_VALUE = HS.SETTING, HS.SETTING_VALUE
VERSIONS = dict(HS.VERSIONS)
#: the reviewed components this run reuses, imported and never modified: everything the screen pinned, the screen's
#: module itself (the one that ran, entry 53) and the screen's plan
PINNED = {**HS.PINNED,
          "scripts/GPU/alphazero/htb_screen.py": "5d583cbb2de9101a2fc08bf2e487e052f383da4e3ca70ac6687db91eb405bc92",
          HS.PLAN_DOC: HS.PLAN_DOC_SHA256}
GATE_NAME = "HTB_STRENGTH_AUTHORIZED"
EVIDENCE = "docs/superpowers/evidence/htb-strength"
RUN_DIR = "docs/superpowers/evidence/htb-strength-run"
PLAN = f"{RUN_DIR}/plan.json"
PLAN_KIND = "htb-strength-plan/1"
STUDENT, REFERENCE_DIGEST = HS.STUDENT, HS.REFERENCE_DIGEST
#: card §9.2: 110 fresh pairs; the one-sided Hoeffding bound at alpha = 0.05, sqrt(ln(1 / alpha) / (2 n)) = sqrt(ln 20 / 220)
STRENGTH = {"pairs": 110, "alpha": 0.05, "null": 0.5}
HALF_WIDTH = math.sqrt(math.log(20.0) / 220.0)
GUARDS = dict(K.GUARDS)                                    # card §9.3: evaluations <= 1.05, search seconds <= 1.10
CANDIDATE, REFERENCE = HS.CANDIDATE, HS.REFERENCE
BOARD = HS.BOARD
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = HS.EXIT_COMPLETED, HS.EXIT_USAGE, HS.EXIT_VOID, HS.EXIT_REFUSED
Void, Terminated = M.Void, M.Terminated
_watch = HS._watch
opening_probe = HS.opening_probe
_identities = HS._identities
ScreenSeams = HS.ScreenSeams


class Refused(Exception):
    """A precondition outside the record boundary: nothing is loaded, played or written."""


# ─────────────────────────────── the rule (plan §6) ───────────────────────────────

def strength_verdict(pairs: Sequence[float], g: Mapping[str, Any]) -> Dict[str, Any]:
    """PASS only if mean - sqrt(ln 20 / 220) > 0.5 AND both guards hold over the whole run; otherwise NON_PASS, with
    every reason that applies."""
    if len(pairs) != STRENGTH["pairs"]:
        raise ValueError(f"{len(pairs)} pairs, not the confirmation test's {STRENGTH['pairs']}")
    if g.get("guards") != GUARDS:
        raise ValueError(f"the guards are not the card's {GUARDS}: {g.get('guards')}")
    mean = math.fsum(pairs) / len(pairs)
    lower = mean - HALF_WIDTH
    reasons = []
    if not lower > STRENGTH["null"]:
        reasons.append(f"the lower bound {lower} (mean {mean} - {HALF_WIDTH}) is not above {STRENGTH['null']}")
    if g["evaluations_within"] is not True:
        reasons.append(f"evaluations ratio {g['evaluations_ratio']} exceeds {GUARDS['evaluations']}")
    if g["search_seconds_within"] is not True:
        reasons.append(f"search-seconds ratio {g['search_seconds_ratio']} exceeds {GUARDS['search_seconds']}")
    return {"pairs": len(pairs), "mean_pair_score": mean, "half_width": HALF_WIDTH, "lower_bound": lower,
            "alpha": STRENGTH["alpha"], "one_sided": True, "null": STRENGTH["null"],
            "evaluations_ratio": g["evaluations_ratio"], "search_seconds_ratio": g["search_seconds_ratio"],
            "guards": dict(GUARDS), "reasons": reasons, "verdict": "NON_PASS" if reasons else "PASS",
            "note": ("against the original proven combination at 200 simulations only; pairs assumed independent; "
                     "never pooled with the screen; a NON_PASS is not evidence of no advantage")}


# ─────────────────────────────── admission (plan §4) ───────────────────────────────

SCREEN_OUTER_OK = HS.OUTER_OK


def screen_admission(read_record: Callable[[str], Mapping[str, Any]],
                     read_outer: Callable[[str], Mapping[str, Any]]) -> Dict[str, Any]:
    """The committed screen record and its wrapper's outer.json must show the ONE screen COMPLETED and CONTINUE under
    the pinned card and screen plan, with cleanup confirmed, on exactly the pinned student and reference."""
    try:
        rec = read_record(HS.EVIDENCE)
        outer = read_outer(HS.RUN_DIR)
    except FileNotFoundError as e:
        raise Refused(f"{e} does not exist: the confirmation test is not admitted") from None
    faults = []
    want = {"kind": HS.RECORD_KIND, "status": "COMPLETED", "verdict": "CONTINUE", "card_sha256": CARD_SHA256,
            "plan_doc_sha256": HS.PLAN_DOC_SHA256, "written_by": "script"}
    faults += [f"{k} is {rec.get(k)!r}, not {v!r}" for k, v in want.items() if rec.get(k) != v]
    if (rec.get("screen") or {}).get("verdict") != "CONTINUE":
        faults.append(f"the screen's own verdict is not CONTINUE: {(rec.get('screen') or {}).get('verdict')!r}")
    s = rec.get("student") or {}
    if s != {"path": str(STUDENT["path"]), "sha256": STUDENT["sha256"], "state_digest": STUDENT["state_digest"]}:
        faults.append(f"the screen's student is not the pinned one: {s}")
    ids = {CANDIDATE: STUDENT["state_digest"], REFERENCE: REFERENCE_DIGEST}
    for key in ("digests", "digests_at_end"):
        if rec.get(key) != ids:
            faults.append(f"the screen's {key} are not the pinned identities: {rec.get(key)}")
    for key, val in SCREEN_OUTER_OK:
        if outer.get(key) != val or type(outer.get(key)) is not type(val):
            faults.append(f"the screen's outer.json {key} is {outer.get(key)!r}, not {val!r}")
    if faults:
        raise Refused(f"the screen does not admit the confirmation test: {'; '.join(faults)}")
    return {f"{HS.EVIDENCE}/record.json": "CONTINUE", f"{HS.RUN_DIR}/outer.json": "COMPLETED"}


def admission(read_record, read_outer, read_bytes) -> Dict[str, Any]:
    """The fit (the screen's own admission, unchanged), then the screen."""
    return {**HS.fit_admission(read_record, read_outer, read_bytes), **screen_admission(read_record, read_outer)}


# ─────────────────────────────── the confirmation test (plan §§2-7) ───────────────────────────────

def _strength(seams: ScreenSeams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    """Plan §§2-7 through the screen's loop (the cycle-2 screen's), both sides the combination; the candidate the whole
    student; 110 pairs on a fresh block disjoint from the screen's."""
    from . import eval_runner as ER
    from . import root_sequential_halving as RSH
    from .external_learning_stage3 import PROTOCOL
    from .mirror_averaging import MirrorAveragingEvaluator

    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the plan's pinned {VERSIONS}")
    rec["variant"] = K.check_variant(RSH)
    config = ER.EvalConfig(**PROTOCOL)
    if (config.board_size, config.max_moves) != (BOARD, 280):
        raise Void("protocol", "PROTOCOL is not 24x24 / 280 plies")
    pairs, games = STRENGTH["pairs"], 2 * STRENGTH["pairs"]
    plan = seams.read_plan(PLAN)
    if plan.get("kind") != PLAN_KIND or plan.get("n_slots") != games or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {games} slots of {PLAN_KIND} from an int base: {plan}")
    bad = [plan["base"] + s for s in range(games) if not seams.seed_ok(plan["base"] + s)]
    if bad:
        raise Void("seeds", f"{len(bad)} seeds not ACCOUNTED, available and schedulable, first {bad[0]}")
    rec["seeds"] = {"base": plan["base"], "n_slots": games}
    screen = seams.read_plan(HS.PLAN)                    # the screen's games are never pooled: its block is not this one
    lo, n = screen.get("base"), screen.get("n_slots")
    if screen.get("kind") != HS.PLAN_KIND or type(lo) is not int or type(n) is not int or \
            (plan["base"] < lo + n and lo < plan["base"] + games):
        raise Void("seeds", f"the block overlaps the screen's, or the screen's plan is unreadable: {screen}")
    rec["screen_block_excluded"] = [lo, lo + n]
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint_sha1"] = sha1
    sha = seams.student_sha256()
    if sha != STUDENT["sha256"]:
        raise Void("candidate", f"the student file's sha256 {sha} is not the pinned one")
    rec["student"] = {"path": str(STUDENT["path"]), "sha256": sha, "state_digest": STUDENT["state_digest"]}
    _watch(seams, rec, log, "before load")

    t = time.monotonic()
    base = {CANDIDATE: seams.load_candidate(), REFERENCE: seams.load_reference()}
    if base[CANDIDATE] is base[REFERENCE] or base[CANDIDATE].network is base[REFERENCE].network:
        raise Void("load", "the two sides share an evaluator or a network")
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["digests"] = _identities(seams, base, "after load")
    rec["modes_at_load"] = {k: seams.require_eval(ev, "after load") for k, ev in base.items()}
    rec["modules"] = seams.loaded_modules()
    rec["opening_probe"] = opening_probe(base[CANDIDATE], base[REFERENCE])
    if not rec["opening_probe"]["max_prior_abs_diff"] > 0.0:
        raise Void("candidate", "the candidate's opening priors equal the reference's")
    cost = {k: K.new_cost() for k in base}
    for k, ev in base.items():
        K.instrument(ev, cost[k])
    evaluators = {k: MirrorAveragingEvaluator(ev) for k, ev in base.items()}
    classes = {k: RSH.RootSequentialHalvingMCTS for k in base}
    rd = ER.readout_from_eval_config(config)
    a = ER.AgentSpec(CANDIDATE, str(STUDENT["path"]), rd, mcts_sims=K.CANDIDATE_SIMS)
    b = ER.AgentSpec(REFERENCE, M.CHECKPOINT, rd, mcts_sims=K.CANDIDATE_SIMS)
    rec["protocol"] = {"PROTOCOL": dict(PROTOCOL), "eval_config": dataclasses.asdict(config),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)],
                       "classes": {k: f"{v.__module__}.{v.__qualname__}" for k, v in classes.items()},
                       "evaluators": {k: type(v).__name__ for k, v in evaluators.items()}}
    tasks = ER.build_agent_pairing_tasks(pathlib.Path(EVIDENCE).name, a, b, games, base_seed=plan["base"])
    log("loaded", seconds=rec["load_seconds"])

    # ── THE LOOP: htb_screen._screen's = combination_policy_pilot._screen's, unchanged (the tests compare the text) ──
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
            if any(c["searches"] == 0 for c in game_cost.values()) or \
                    sum(c["searches"] for c in game_cost.values()) != r.n_moves:
                raise Void("attribution", f"game {task.game_idx}: the searches do not account for its moves")
            row = {**dataclasses.asdict(r), "slot": task.seed - plan["base"],
                   "seconds": round(time.monotonic() - t, 3), "cost": game_cost}
            games_fh.write(json.dumps(row) + "\n")
            games_fh.flush()
            os.fsync(games_fh.fileno())
            results.append(r)
            log("game_end", game=task.game_idx, n_moves=r.n_moves, reason=r.reason)
            seams.between_games()
            _watch(seams, rec, log, f"after game {task.game_idx}")
    _watch(seams, rec, log, "before the verdict")
    # ── end of the loop ──

    rec["modes_at_end"] = {k: seams.require_eval(ev, "at the end") for k, ev in base.items()}
    rec["digests_at_end"] = _identities(seams, base, "at the end")         # the candidate stays frozen
    rec["cost_totals"] = cost
    g = K.guards(cost[CANDIDATE], cost[REFERENCE])
    rec["guards"] = g
    scores = K.pair_scores(results, pairs)
    rec["pair_scores"] = scores
    rec["games_summary"] = {"candidate_wins": sum(r.winner_agent_id == CANDIDATE for r in results),
                            "reference_wins": sum(r.winner_agent_id == REFERENCE for r in results),
                            "draws": sum(r.winner is None for r in results),
                            "state_caps": sum(r.reason == "state_cap" for r in results)}
    rec["strength"] = strength_verdict(scores, g)
    rec["verdict"] = rec["strength"]["verdict"]


def _real_strength_seams(checkpoint: pathlib.Path) -> ScreenSeams:   # pragma: no cover - loads models; never in tests
    """The screen's real seams, unchanged: the match's loader for both sides, its loop and checks."""
    return HS._real_screen_seams(checkpoint)


# ─────────────────────────────── the record boundary and the public entry ───────────────────────────────

def conclude(execute: Callable[[pathlib.Path, Dict[str, Any], Callable[..., None]], None], out_dir: pathlib.Path,
             admission: Mapping[str, str], head: Optional[str] = None) -> Tuple[int, Dict]:
    """THE RECORD BOUNDARY (htb_screen.conclude's, with this run's kind, run name and plan): record.json on every exit
    this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "run": "strength", "card": CARD, "card_sha256": CARD_SHA256,
                           "plan_doc": PLAN_DOC, "plan_doc_sha256": PLAN_DOC_SHA256, "head": head,
                           "admission": dict(admission), "status": None, "verdict": None, "written_by": "script",
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
        execute(out_dir, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", verdict=None, void={"step": "terminated",
                                                      "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
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


def _preflight() -> None:
    """The setting in a fresh process, then the pinned plan, card and reused modules."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if CPP._mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    for rel, sha in ((PLAN_DOC, PLAN_DOC_SHA256), (CARD, CARD_SHA256), *PINNED.items()):
        if not (ROOT / rel).is_file() or M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the gate, the pins, admission, real sources."""
    _preflight()
    if getattr(GATE, GATE_NAME) is not True:
        raise Refused(f"{GATE_NAME} is closed: the confirmation test is not authorized")
    try:
        adm = admission(CPP._read_record, CPP._read_outer, CPP._read_committed_bytes)
    except HS.Refused as e:
        raise Refused(str(e)) from None
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    if not pathlib.Path(STUDENT["path"]).is_file():
        raise Refused(f"{STUDENT['path']} is not a file")
    out = ROOT / EVIDENCE
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    seams = _real_strength_seams(ckpt)
    code, rec = conclude(lambda o, r, lg: _strength(seams, o, r, lg), out, adm, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "run": "strength",
                      "verdict": rec.get("verdict"), "strength": rec.get("strength")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_strength run strength --checkpoint <abs path>"
    try:
        if len(argv) == 4 and argv[:2] == ["run", "strength"] and argv[2] == "--checkpoint":
            return run(argv[3])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(usage, file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
