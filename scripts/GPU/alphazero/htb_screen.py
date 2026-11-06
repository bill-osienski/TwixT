"""THE H-TB SCREEN -- the WHOLE H-TB student in the proven combination against the original proven combination, 32
colour-reversed pairs (64 games). BUILT with its gate closed; nothing here has run.

Plan: `docs/superpowers/2026-11-06-htb-screen-plan.md`, pinned below by sha256, filling in the pilot card's §9.1 and
§9.3 (`docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md`, its pin unchanged). Nothing runs on import,
and importing this module loads no MLX. `run screen` refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module
is loaded when it starts, the gate is open (read at call time), the plan, card and frozen modules are the pinned bytes,
and the committed fit evidence admits it and names exactly the pinned student.

THE RUN (plan §§2-6), in ONE fresh process under its wrapper's hard ceiling:
* versions (Python 3.14.10, MLX 0.30.3, NumPy 2.4.1); the PROTOCOL (24x24, 280 plies); the plan: exactly 64 slots from
  an int base, every seed ACCOUNTED, available and schedulable; calib020_0001's SHA-1; the student file's sha256;
* both sides through the match's loader (`eval_runner._default_evaluator_factory`): the candidate is the student file
  loaded WHOLE (every parameter and BatchNorm statistic; no head installed), the reference calib020_0001; their state
  digests must be the fit's (student `586dca4f...`, reference `ae0e71fc...`) at load AND at the end; no shared network;
  evaluation mode at load and at the end; an opening probe that must find their priors different;
* the cycle-2 screen's loop, unchanged (`combination_policy_pilot._screen`; a test compares the source text): both sides
  MirrorAveragingEvaluator + RootSequentialHalvingMCTS at 200 simulations (halving from ply 20), the candidate red in
  even games, game g on seed plan.base + g; `game_start` logged before each game; background_compute read before
  loading, after every game and before the verdict;
* THE RULE (plan §5): STOP if the mean pair score is below 0.50, or the evaluations ratio exceeds 1.05, or the
  search-seconds ratio exceeds 1.10 (ratios of the run's totals, candidate over reference); otherwise CONTINUE. A
  screen only: no strength claim.

Any failure, exception or SIGTERM inside the record boundary is a VOID with record.json written by the script. One
shot; no retry, top-up or extension.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_screen run screen --checkpoint <abs path to calib020_0001>
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import combination_policy_pilot as CPP
from . import htb_pilot as H
from . import htb_screen_authorization as GATE
from . import mirror_halving_combination as K
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "htb-screen-record/1"
CARD, CARD_SHA256 = H.CARD, H.CARD_SHA256                  # the pilot card, its pin unchanged
PLAN_DOC = "docs/superpowers/2026-11-06-htb-screen-plan.md"
PLAN_DOC_SHA256 = "4201dbec2c4adfd3c2a728f0759cdb658b5fee0f7fddc9ad3c2dd0b307558912"
SETTING, SETTING_VALUE = CPP.SETTING, CPP.SETTING_VALUE
VERSIONS = {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.10"}
#: the reviewed components this run reuses, imported and never modified: the cycle-2 screen's module (its loop and
#: probe), the frozen combination, averaging and halving modules, and the pilot module whose constants bind the fit
PINNED = {**CPP.PINNED,
          "scripts/GPU/alphazero/combination_policy_pilot.py":
          "47d2e05e64288f6040aecaedfee63f7ffd46e75d28c149d100ee8391b0f00aab",
          "scripts/GPU/alphazero/htb_pilot.py": "6c707c3f399cff5539eaaf8849a16fce9b93e1aeedfed12c0bbf6986e107e423"}
GATE_NAME = "HTB_SCREEN_AUTHORIZED"
EVIDENCE = "docs/superpowers/evidence/htb-screen"
RUN_DIR = "docs/superpowers/evidence/htb-screen-run"
PLAN = f"{RUN_DIR}/plan.json"
PLAN_KIND = "htb-screen-plan/1"
FIT_EVIDENCE, FIT_RUN = H.EVIDENCE_FIT, "docs/superpowers/evidence/htb-fit-run"
#: the candidate (plan §2): exactly the student the ONE fit saved -- these bytes, with this state digest
STUDENT = {"path": H.CACHE / "student.safetensors",
           "sha256": "0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5",
           "state_digest": "586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b"}
#: the reference (plan §3): calib020_0001 as the fit loaded it
REFERENCE_DIGEST = "ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b"
SCREEN = {"pairs": 32, "stop_below": 0.50}                 # card §9.1
GUARDS = dict(K.GUARDS)                                    # card §9.3: evaluations <= 1.05, search seconds <= 1.10
CANDIDATE, REFERENCE = K.CANDIDATE, CPP.REFERENCE
BOARD = CPP.BOARD
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = CPP.EXIT_COMPLETED, CPP.EXIT_USAGE, CPP.EXIT_VOID, CPP.EXIT_REFUSED
Void, Terminated = M.Void, M.Terminated
_watch = CPP._watch
opening_probe = CPP.opening_probe


class Refused(Exception):
    """A precondition outside the record boundary: nothing is loaded, played or written."""


# ─────────────────────────────── the rule (plan §5) ───────────────────────────────

def screen_verdict(pairs: Sequence[float], g: Mapping[str, Any]) -> Dict[str, Any]:
    """STOP below a mean pair score of 0.50 or on either failed compute guard; otherwise CONTINUE. Every reason that
    applies is listed. No strength claim either way."""
    if len(pairs) != SCREEN["pairs"]:
        raise ValueError(f"{len(pairs)} pairs, not the screen's {SCREEN['pairs']}")
    if g.get("guards") != GUARDS:
        raise ValueError(f"the guards are not the card's {GUARDS}: {g.get('guards')}")
    mean = math.fsum(pairs) / len(pairs)
    reasons = []
    if mean < SCREEN["stop_below"]:
        reasons.append(f"mean pair score {mean} is below {SCREEN['stop_below']}")
    if g["evaluations_within"] is not True:
        reasons.append(f"evaluations ratio {g['evaluations_ratio']} exceeds {GUARDS['evaluations']}")
    if g["search_seconds_within"] is not True:
        reasons.append(f"search-seconds ratio {g['search_seconds_ratio']} exceeds {GUARDS['search_seconds']}")
    return {"pairs": len(pairs), "mean_pair_score": mean, "stop_below": SCREEN["stop_below"],
            "evaluations_ratio": g["evaluations_ratio"], "search_seconds_ratio": g["search_seconds_ratio"],
            "guards": dict(GUARDS), "stop_reasons": reasons, "verdict": "STOP" if reasons else "CONTINUE",
            "note": "a screen only: not a strength result; never pooled with strength games"}


# ─────────────────────────────── admission (plan §2) ───────────────────────────────

OUTER_OK = (("inner_exit", 0), ("outer_exit", 0), ("outcome", "COMPLETED"), ("gate_restored", True),
            ("survivors", ""), ("record_by", "script"))


def fit_admission(read_record: Callable[[str], Mapping[str, Any]], read_outer: Callable[[str], Mapping[str, Any]],
                  read_bytes: Callable[[str], bytes]) -> Dict[str, Any]:
    """The committed fit record and its wrapper's outer.json must show the ONE fit COMPLETED and ELIGIBLE under the
    pinned card, with cleanup confirmed, naming exactly the pinned student and reference. Refused otherwise."""
    try:
        fit = read_record(FIT_EVIDENCE)
        outer = read_outer(FIT_RUN)
        label_raw = read_bytes(f"{H.EVIDENCE_LABEL}/record.json")
    except FileNotFoundError as e:
        raise Refused(f"{e} does not exist: the screen is not admitted") from None
    faults = []
    want = {"kind": "htb-fit-record/1", "status": "COMPLETED", "card_sha256": CARD_SHA256}
    faults += [f"{k} is {fit.get(k)!r}, not {v!r}" for k, v in want.items() if fit.get(k) != v]
    v = fit.get("verdict") or {}
    if v.get("verdict") != "ELIGIBLE" or any(v.get(f) is not True for f in ("F1", "F2", "F3")):
        faults.append(f"the verdict is not ELIGIBLE with F1-F3 all true: {v}")
    s = fit.get("student") or {}
    if s.get("path") != str(STUDENT["path"]) or s.get("sha256") != STUDENT["sha256"] or \
            s.get("state_digest") != STUDENT["state_digest"]:
        faults.append(f"the record's student is not the pinned one: {s}")
    if (fit.get("fit") or {}).get("reference_digest") != REFERENCE_DIGEST:
        faults.append("the fit's reference digest is not calib020_0001's")
    if fit.get("label_record_sha256") != hashlib.sha256(label_raw).hexdigest():
        faults.append("the fit's label record is not the committed one")
    for key, val in OUTER_OK:
        if outer.get(key) != val or type(outer.get(key)) is not type(val):
            faults.append(f"outer.json {key} is {outer.get(key)!r}, not {val!r}")
    if faults:
        raise Refused(f"the fit does not admit the screen: {'; '.join(faults)}")
    return {f"{FIT_EVIDENCE}/record.json": "ELIGIBLE", f"{FIT_RUN}/outer.json": "COMPLETED"}


# ─────────────────────────────── the screen (plan §§2-6) ───────────────────────────────

@dataclasses.dataclass
class ScreenSeams:
    """What the screen reads from outside this module. TEST seams; `run` passes `_real_screen_seams` only."""
    versions: Callable[[], Dict[str, str]]
    read_plan: Callable[[str], Mapping[str, Any]]
    seed_ok: Callable[[int], bool]
    checkpoint_sha1: Callable[[], str]
    student_sha256: Callable[[], str]
    load_candidate: Callable[[], Any]                    # the student file, whole
    load_reference: Callable[[], Any]                    # calib020_0001
    digest: Callable[[Any], str]                         # a network -> its state digest (named-tensors/1)
    require_eval: Callable[[Any, str], Dict[str, Any]]
    loaded_modules: Callable[[], Dict[str, str]]
    background: Callable[[], Dict[str, Any]]
    play_and_build: Callable[..., Any]
    between_games: Callable[[], None]


def _identities(seams: ScreenSeams, base: Mapping[str, Any], when: str) -> Dict[str, str]:
    got = {k: seams.digest(ev.network) for k, ev in base.items()}
    if got[CANDIDATE] != STUDENT["state_digest"]:
        raise Void("candidate", f"{when}: the candidate's state digest {got[CANDIDATE]} is not the student's")
    if got[REFERENCE] != REFERENCE_DIGEST:
        raise Void("reference", f"{when}: the reference's state digest {got[REFERENCE]} is not calib020_0001's")
    return got


def _screen(seams: ScreenSeams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    """Plan §§2-6 through the cycle-2 screen's loop, both sides the combination; the candidate the whole student."""
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
    pairs, games = SCREEN["pairs"], 2 * SCREEN["pairs"]
    plan = seams.read_plan(PLAN)
    if plan.get("kind") != PLAN_KIND or plan.get("n_slots") != games or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {games} slots of {PLAN_KIND} from an int base: {plan}")
    bad = [plan["base"] + s for s in range(games) if not seams.seed_ok(plan["base"] + s)]
    if bad:
        raise Void("seeds", f"{len(bad)} seeds not ACCOUNTED, available and schedulable, first {bad[0]}")
    rec["seeds"] = {"base": plan["base"], "n_slots": games}
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

    # ── THE LOOP: combination_policy_pilot._screen's, unchanged (tests/test_htb_screen.py compares the text) ──
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
    rec["screen"] = screen_verdict(scores, g)
    rec["verdict"] = rec["screen"]["verdict"]


def _real_screen_seams(checkpoint: pathlib.Path) -> ScreenSeams:   # pragma: no cover - loads models; never in tests
    from . import background_compute as BG
    from . import e4_screen_reference as REG
    from . import eval_runner as ER
    from . import mirror_halving_match as MHM

    def seed_ok(s):
        return REG.seed_is_accounted(s) and not REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s)

    student = pathlib.Path(STUDENT["path"])
    return ScreenSeams(versions=CPP._versions, read_plan=lambda rel: json.loads((ROOT / rel).read_text()),
                       seed_ok=seed_ok, checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                       student_sha256=lambda: hashlib.sha256(student.read_bytes()).hexdigest(),
                       load_candidate=lambda: ER._default_evaluator_factory(str(student)),    # the match's loader
                       load_reference=lambda: ER._default_evaluator_factory(str(checkpoint)),
                       digest=H._digest, require_eval=lambda ev, when: M._require_eval(ev.network, when),
                       loaded_modules=M._real_seams(checkpoint).loaded_modules, background=BG.observe,
                       play_and_build=K.play_and_build, between_games=MHM._between_games)


# ─────────────────────────────── the record boundary and the public entry ───────────────────────────────

def conclude(execute: Callable[[pathlib.Path, Dict[str, Any], Callable[..., None]], None], out_dir: pathlib.Path,
             admission: Mapping[str, str], head: Optional[str] = None) -> Tuple[int, Dict]:
    """THE RECORD BOUNDARY (combination_policy_pilot.conclude's, with this run's kind and pins): record.json on every
    exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "run": "screen", "card": CARD, "card_sha256": CARD_SHA256,
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
        raise Refused(f"{GATE_NAME} is closed: the screen is not authorized")
    admission = fit_admission(CPP._read_record, CPP._read_outer, CPP._read_committed_bytes)
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    if not pathlib.Path(STUDENT["path"]).is_file():
        raise Refused(f"{STUDENT['path']} is not a file")
    out = ROOT / EVIDENCE
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    seams = _real_screen_seams(ckpt)
    code, rec = conclude(lambda o, r, lg: _screen(seams, o, r, lg), out, admission, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "run": "screen", "verdict": rec.get("verdict"),
                      "screen": rec.get("screen")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_screen run screen --checkpoint <abs path>"
    try:
        if len(argv) == 4 and argv[:2] == ["run", "screen"] and argv[2] == "--checkpoint":
            return run(argv[3])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(usage, file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
