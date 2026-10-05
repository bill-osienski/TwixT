"""THE TWO-AGENT MATCH RUNNER (R1): the tactical candidate against the unchanged incumbent.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §3 R1, §7.1, §11 A1.2-A1.3,
§12 A2.4. Adjacent colour-reversed pairs from the empty official board: in Arm A the candidate
plays red, in Arm B black. Each game builds ONE FRESH agent per colour, and the SEED FOLLOWS THE
COLOUR -- the pair's bundle {red, black} -- never the role. 24x24, cap 280, the winner first.

EXACT REPLAY (§7.1). Until its first override each game IS the incumbent-only game G0, move for
move. So the two games of a pair must agree up to the earlier first override; there, the
overriding game's BASELINE must be the other game's move; and a pair in which nothing fired must
be move-identical. Any breach STOPS the segment: the determinism premise has failed.

BLIND. The record holds every move, every candidate witness and the terminal type, so outcomes
are derivable from the evidence alone -- by design. Nothing outcome-bearing reaches the console
or the journal: no winner, no score, no ply count, no terminal type. Nothing here scores a game.

RUN CONTRACT (the H5 studies'): pure refusals; the parent's create-only CLAIM with a nonce; the
child in its own session under the parent's hard watchdog (process-group SIGKILL at the segment
limit); a hard SIGALRM limit per game (15 min) and for the segment (4 h, setup included); an
fsynced journal; record.json on every exit, the completion check inside the recorded boundary;
watchdog.json; no automatic retry. Gate: `candidate_match_authorization`, created closed.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
import pathlib
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import candidate_match_authorization as AUTH
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_tactics as T
from . import tactical_candidate as TC
from . import twixtbot_g3_reference as G3

MODULE = "scripts.GPU.alphazero.candidate_match_runner"
BOARD, PLY_CAP = 24, 280                   # tests pin both to the tactics card's
COLOURS = ("red", "black")
OTHER = {"red": "black", "black": "red"}
ARMS = {"A": "red", "B": "black"}          # the CANDIDATE's colour in each arm
MODES = ("fixture", "pilot", "study")      # "fixture": synthetic plans, never a public entry
GAME_DEADLINE_S = 900.0                    # §12 A2.4: 15 min per game, hard
SEGMENT_DEADLINE_S = 14_400.0              # §12 A2.4: 4 h per segment, setup included
STUDY_CEILING_S = 80 * 3600.0              # §12 A2.2: across all study segments
MARGIN_S = 60.0
PLAN_DIR = "tactical-candidate-plans"
CLAIM_FILE, WATCHDOG_FILE, CLAIM_ENV = "claim.json", "watchdog.json", "CANDIDATE_MATCH_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5
CARD = "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"


class MatchRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class MatchUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class MatchStop(Exception):
    """A check failed during the run: recorded, never retried."""


class MatchTimeout(MatchStop):
    """A hard limit fired."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ─────────────────────────────── the plan ───────────────────────────────

def check_plan(plan: Mapping[str, Any], *, mode: str) -> None:
    """Whole pairs, a seed per colour, every id and seed unique; synthetic seeds negative,
    research seeds positive AND accounted in the shared registry, none spent."""
    if mode not in MODES:
        raise MatchRefused(f"unknown mode {mode!r}; one of {list(MODES)}")
    if not isinstance(plan, Mapping) or plan.get("mode") != mode:
        raise MatchRefused(f"the plan is not a {mode} plan")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise MatchRefused("a plan is one or more whole pairs; none given")
    ids, seeds = [], []
    for p in pairs:
        s = p.get("seeds") if isinstance(p, Mapping) else None
        if not isinstance(s, Mapping) or set(s) != set(COLOURS):
            raise MatchRefused(f"pair {p!r}: seeds must be exactly one per colour")
        ids.append(p.get("pair_id"))
        seeds += [s["red"], s["black"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise MatchRefused("pair ids must be non-empty and unique")
    if any(type(x) is not int for x in seeds):
        raise MatchRefused("every seed must be an int")
    if len(set(seeds)) != len(seeds):
        raise MatchRefused("a seed repeats: that is an integrity fault, not a replicate")
    if mode == "fixture":
        if any(x >= 0 for x in seeds):
            raise MatchRefused("a fixture plan uses NEGATIVE seeds only; research seeds are positive")
        return
    if any(x < 0 for x in seeds):
        raise MatchRefused(f"a {mode} plan refuses a negative (synthetic) seed")
    faults = R4.seed_registry_faults([{"seed": x} for x in seeds])
    if faults:
        raise MatchRefused(f"seeds not registered for use, or spent: {faults[:5]}")


# ─────────────────────────────── agents and one game ───────────────────────────────

def production_builder(config, evaluator, identity: Mapping[str, Any]) -> Callable:
    """The ONE construction path: G3's seeded reference agent, its seed bound to the COLOUR; the
    candidate is that agent behind the enabled wrapper (§1), the incumbent the bare agent."""
    def make(colour: str, seed: int, role: str):
        task = {"seed": seed, "reference": identity["reference"],
                "reference_sha1": identity["reference_sha1"], "anchor_colour": OTHER[colour]}
        agent = G3.build_reference_agent(task=task, evaluator=evaluator, colour=colour,
                                         config=config, capture=role == "candidate")
        return TC.TacticalCandidate(agent, enabled=True) if role == "candidate" else agent
    return make


@contextlib.contextmanager
def _alarm(seconds: float, what: str):
    """A HARD limit: SIGALRM interrupts the blocking call itself (as d1_probe's supervisor).
    Fails closed: refuses where it cannot arm, or with nothing left (setitimer(0) disarms)."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise MatchRefused("a hard limit cannot be armed here (needs SIGALRM on the main thread)")
    if seconds <= 0:
        raise MatchTimeout(f"{what}: no time remained to arm the hard limit")

    def _fire(_signum, _frame):
        raise MatchTimeout(f"{what}: the hard limit of {seconds:.1f} s was exceeded")
    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _compact(w: Mapping[str, Any]) -> Dict[str, Any]:
    """One candidate decision (card §3 R1): the full class only where the rule was active."""
    c = w["class"]
    cls = c if w["rule"] != "inactive" else (c and {"class": c["class"], "T_void": c["T_void"]})
    stat = lambda s: [*s["move"], s["visits"], s["q_root_mover"]]   # noqa: E731
    return {"ply": w["ply"], "rule": w["rule"], "class": cls, "reason": w["reason"],
            "override": w["override"], "unsearched": w["unsearched"], "baseline": stat(w["baseline"]),
            "final": stat(w["final"]), "scan_s": w["scan_s"], "search_s": w["search_s"]}


def game_digest(moves: Sequence, terminal: Mapping[str, Any]) -> str:
    return _sha({"moves": [list(m) for m in moves], "terminal": dict(terminal)})


def play_game(make: Callable, pair: Mapping[str, Any], arm: str, *, limit_s: float,
              clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """One game from the empty official board under a HARD limit. Fresh agents; the seed
    follows the colour. The winner is checked before the cap; no move is asked for after."""
    cand = ARMS[arm]
    roles = {c: "candidate" if c == cand else "incumbent" for c in COLOURS}
    t0 = clock()
    with _alarm(limit_s, f"game {pair['pair_id']}-{arm}"):
        agents = {c: make(c, pair["seeds"][c], roles[c]) for c in COLOURS}
        state, moves = T.replay([]), []
        while state.winner() is None and state.ply < PLY_CAP:
            if clock() - t0 > limit_s:
                raise MatchTimeout(f"game {pair['pair_id']}-{arm}: {limit_s:.1f} s passed at ply "
                                   f"{state.ply}")
            move = _t(agents[state.to_move](state))
            if not state.is_valid_placement(*move):
                raise MatchStop(f"game {pair['pair_id']}-{arm}: the illegal move {list(move)} "
                                f"at ply {state.ply}")
            moves.append(list(move))
            state = state.apply_move(move)
    witnesses = agents[cand].witnesses
    own = [i for i in range(len(moves)) if i % 2 == (0 if cand == "red" else 1)]
    if [w["ply"] for w in witnesses] != own or \
            any(w["final"]["move"] != moves[w["ply"]] for w in witnesses):
        raise MatchStop(f"game {pair['pair_id']}-{arm}: the candidate's witnesses are not its "
                        "decisions in this game")
    terminal = {"reason": "win" if state.winner() is not None else "cap", "ply": state.ply}
    return {"pair_id": pair["pair_id"], "arm": arm, "candidate": cand, "roles": roles,
            "seeds": dict(pair["seeds"]), "moves": moves, "terminal": terminal,
            "witnesses": [_compact(w) for w in witnesses], "digest": game_digest(moves, terminal),
            "seconds": clock() - t0}


def first_override(game: Mapping[str, Any]) -> Optional[int]:
    return min((w["ply"] for w in game["witnesses"] if w["override"]), default=None)


def check_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """§7.1, checked: both games are G0 until the earlier first override."""
    pid = a["pair_id"]
    if (a["arm"], b["arm"], b["pair_id"], a["seeds"]) != ("A", "B", pid, b["seeds"]):
        raise MatchStop(f"pair {pid}: not one A-then-B pair on one seed bundle")
    fa, fb = first_override(a), first_override(b)
    out = {"pair_id": pid, "fired": fa is not None or fb is not None,
           "first_override": {"A": fa, "B": fb}}
    if not out["fired"]:
        if a["moves"] != b["moves"] or a["terminal"] != b["terminal"]:
            raise MatchStop(f"pair {pid}: nothing fired, yet the two games differ -- the "
                            "exact-replay premise (§7.1) failed")
        return {**out, "agree_through": len(a["moves"])}
    k = min(x for x in (fa, fb) if x is not None)
    if a["moves"][:k] != b["moves"][:k]:
        raise MatchStop(f"pair {pid}: the games differ before the first override (ply {k})")
    g, other = (a, b) if fa == k else (b, a)
    base = next(w for w in g["witnesses"] if w["ply"] == k)["baseline"][:2]
    if k >= len(other["moves"]) or other["moves"][k] != base:
        raise MatchStop(f"pair {pid}: at ply {k} the overriding game's baseline {base} is not "
                        "the other game's move")
    return {**out, "agree_through": k}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def _file_sha(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _run(plan: Mapping[str, Any], config, identity: Mapping[str, Any], loader: Callable,
         d: pathlib.Path, *, mode: str, builder: Optional[Callable] = None,
         clock: Callable[[], float] = time.monotonic, wall_s: float = SEGMENT_DEADLINE_S - MARGIN_S,
         game_s: float = GAME_DEADLINE_S) -> Dict[str, Any]:
    """The segment. `builder(evaluator) -> make` is a TEST seam; no public entry passes one."""
    check_plan(plan, mode=mode)
    if not _unconsumed(d):
        raise MatchRefused(f"{d} does not hold exactly an unconsumed claim")
    t0 = clock()
    left = lambda: wall_s - (clock() - t0)    # noqa: E731
    log, step, games, pairs = None, "claim", [], []
    status, failure, interrupt = "COMPLETE", None, None
    try:
        log = CC._Log(d / "log.jsonl")
        step = "model_load"
        with _alarm(left(), "the segment (model load)"):
            evaluator = loader()
        make = builder(evaluator) if builder is not None else \
            production_builder(config, evaluator, identity)
        log.done("model_load", {"incumbent_identity": dict(identity)})
        for p in plan["pairs"]:
            got = []
            for arm in ARMS:
                step = f"game:{p['pair_id']}-{arm}"
                if left() <= 0:
                    raise MatchTimeout(f"the segment's {wall_s:.0f} s passed before {step}")
                g = play_game(make, p, arm, limit_s=min(game_s, left()), clock=clock)
                games.append(g)
                got.append(g)
                log.done("game", {"pair_id": g["pair_id"], "arm": arm, "digest": g["digest"],
                                  "overrides": sum(w["override"] for w in g["witnesses"]),
                                  "seconds": round(g["seconds"], 3)})
            step = f"replay:{p['pair_id']}"
            pc = check_pair(*got)
            pairs.append(pc)
            log.done("pair", {"pair_id": pc["pair_id"], "fired": pc["fired"],
                              "first_override": pc["first_override"]})
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    finally:
        if log is not None:
            log.close()
    if status == "COMPLETE":                     # inside the recorded boundary
        step = "completion_check"
        try:
            want = [(p["pair_id"], arm) for p in plan["pairs"] for arm in ARMS]
            got_ = [(g["pair_id"], g["arm"]) for g in games]
            if got_ != want or [x["pair_id"] for x in pairs] != [p["pair_id"] for p in plan["pairs"]]:
                status, failure = "INCOMPLETE", {
                    "failing_step": "completion_check", "exception_type": None,
                    "reason": "the games are not the plan",
                    "missing": [list(k) for k in want if k not in got_],
                    "duplicated": [list(k) for k, n in collections.Counter(got_).items() if n > 1],
                    "extra": [list(k) for k in got_ if k not in want]}
        except (KeyboardInterrupt, SystemExit) as e:
            status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            status, failure = "STOP", _failure(d, step, e)
    record = {"card": CARD, "mode": mode, "status": status, "failure": failure,
              "plan_sha256": _sha(plan), "plan": plan, "incumbent_identity": dict(identity),
              "wrapper_sha256": _file_sha(TC.__file__), "m1_sha256": _file_sha(T.__file__),
              "board": BOARD, "ply_cap": PLY_CAP, "game_deadline_s": game_s, "deadline_s": wall_s,
              "elapsed_s": clock() - t0, "games": games, "pairs": pairs,
              "note": "nothing here scores a game; outcomes are derivable from the moves, by design"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── the parent ───────────────────────────────

def _unconsumed(d: pathlib.Path) -> bool:
    if not d.is_dir():
        return False
    names = set(os.listdir(d))
    return CLAIM_FILE in names and names <= {CLAIM_FILE, *CHILD_STREAMS.values()}


def _claim(d: pathlib.Path, extra: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    import secrets
    try:
        os.mkdir(d)
    except FileExistsError:
        raise MatchRefused(f"{d} is occupied: each segment is one-shot") from None
    nonce = secrets.token_hex(16)
    claim = {"claimed_monotonic": time.monotonic(),
             "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_pid": os.getpid(),
             "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(), **extra}
    with open(d / CLAIM_FILE, "x", encoding="utf-8") as fh:
        json.dump(claim, fh, indent=1, sort_keys=True)
    return claim, nonce


def _accept_claim(d: pathlib.Path) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise MatchRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise MatchRefused("the claim's nonce does not match: this process was not started by "
                           "the parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
    """watchdog.json: statuses, streams and the journal by hash -- never an outcome."""
    rec, present = None, (d / "record.json").exists()
    if present:
        try:
            rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
        except ValueError:
            rec = None
    status = ("TERMINATED" if watch["terminated"] else
              "EXITED_WITHOUT_RECORD" if rec is None else
              "EXITED_ABNORMALLY" if watch["rc"] != 0 and rec.get("status") != "INTERRUPTED"
              else rec.get("status"))
    rc, streams = watch["rc"], {}
    for s, name in sorted(CHILD_STREAMS.items()):
        raw = (d / name).read_bytes() if (d / name).exists() else None
        streams[s] = None if raw is None else {
            "file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    summary = {"card": CARD, "status": status, "cause": watch["cause"],
               "limit_s": watch["limit_s"], "elapsed_s": watch["elapsed_s"],
               "child_pid": watch["child_pid"], "child_rc": rc,
               "child_exit": "never started" if watch["child_pid"] is None else
               f"killed by signal {CC._signal_name(-rc)}" if rc < 0 else f"exit status {rc}",
               "child_streams": streams, "signal": watch["signal"],
               "survivors_killed": watch["survivors_killed"], "journal": CC._journal(d),
               "record_present": present, "record_readable": rec is not None,
               "record_status": None if rec is None else rec.get("status")}
    with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


# ─────────────────────────────── the public entries ───────────────────────────────

def _gate() -> None:
    if not AUTH.CANDIDATE_MATCH_AUTHORIZED:
        raise MatchUnauthorized("the candidate match is UNAUTHORIZED. Nothing was opened, read, "
                                "built, loaded, played or written.")


def _role(role: str, segment: int) -> None:
    if role not in ("pilot", "study") or type(segment) is not int or segment < 0 or \
            (role == "pilot" and segment != 0):
        raise MatchRefused(f"no {role} segment {segment}: the pilot is segment 0; study segments "
                           "count from 0")


def out_dir(role: str, segment: int) -> pathlib.Path:
    name = "tactical-candidate-pilot" if role == "pilot" else f"tactical-candidate-study-s{segment}"
    return pathlib.Path(R4.EVIDENCE_ROOT) / name


def plan_path(role: str, segment: int) -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / PLAN_DIR / f"{role}-s{segment}.json"


def _committed_plan(role: str, segment: int) -> Tuple[Dict[str, Any], str]:
    path = plan_path(role, segment)
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise MatchRefused(f"the plan must be committed before any model load: {e}") from None
    raw = path.read_bytes()
    plan = json.loads(raw)
    check_plan(plan, mode=role)
    return plan, hashlib.sha256(raw).hexdigest()


def _segment_limit(role: str, segment: int) -> float:
    """§12 A2.4: 4 h per segment. A study segment needs every earlier segment committed and
    COMPLETE, and starts only if the 80 h ceiling holds even should this one run its full limit."""
    if role == "pilot":
        return SEGMENT_DEADLINE_S
    used = 0.0
    for k in range(segment):
        d = out_dir("study", k)
        paths = [str(d / "record.json"), str(d / WATCHDOG_FILE)]
        try:
            F.check_durable(paths, _fixture=False)
        except F.H4AnalysisRefused as e:
            raise MatchRefused(f"study segment {k} must be committed first: {e}") from None
        w = json.loads((d / WATCHDOG_FILE).read_text(encoding="utf-8"))
        if (w.get("status"), w.get("record_status")) != ("COMPLETE", "COMPLETE"):
            raise MatchRefused(f"study segment {k} is {w.get('status')}: no later segment runs")
        used += float(w["elapsed_s"])
    if used + SEGMENT_DEADLINE_S > STUDY_CEILING_S:
        raise MatchRefused(f"{used / 3600:.2f} h used: another {SEGMENT_DEADLINE_S / 3600:.0f} h "
                           f"segment could pass the {STUDY_CEILING_S / 3600:.0f} h ceiling")
    return SEGMENT_DEADLINE_S


def _child_argv(role: str, segment: int) -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed", "--role", role, "--segment", str(segment)]


def run(role: str, segment: int) -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals, THE CLAIM, the child under the hard watchdog;
    `watchdog.json` on every exit. Returns statuses only."""
    _gate()
    _role(role, segment)
    out = out_dir(role, segment)
    if os.path.lexists(out):
        raise MatchRefused(f"{out} is occupied: each segment is one-shot")
    _plan, sha = _committed_plan(role, segment)
    limit = _segment_limit(role, segment)
    D._identity(R4.frozen_argmax_config())
    claim, nonce = _claim(out, {"plan_sha256": sha, "role": role, "segment": segment})
    watch = CC._watch(_child_argv(role, segment),
                      limit_s=limit - (time.monotonic() - claim["claimed_monotonic"]),
                      env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                      cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                      stderr_path=out / CHILD_STREAMS["stderr"])
    summary = _settle(out, watch)
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def run_claimed(role: str, segment: int) -> Dict[str, Any]:
    """The CHILD: the gate, the committed plan (durable, the claim's sha, this role and segment),
    the identity, the claim; then the model load and the games."""
    _gate()
    _role(role, segment)
    out = out_dir(role, segment)
    plan, sha = _committed_plan(role, segment)
    config = R4.frozen_argmax_config()
    identity = D._identity(config)
    claim = _accept_claim(out)
    if (claim.get("plan_sha256"), claim.get("role"), claim.get("segment")) != (sha, role, segment):
        raise MatchRefused("the claim is for another plan, role or segment")
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(plan, config, identity, lambda: R4.load_production_evaluator(identity), out,
                mode=role, wall_s=SEGMENT_DEADLINE_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on. Statuses only."""
    import argparse
    ap = argparse.ArgumentParser(prog="candidate_match_runner",
                                 description="Tactical candidate vs incumbent. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("run", "run-claimed"))
    ap.add_argument("--role", choices=("pilot", "study"), required=True)
    ap.add_argument("--segment", type=int, default=0)
    a = ap.parse_args(argv)
    if not AUTH.CANDIDATE_MATCH_AUTHORIZED:
        print("the candidate match is UNAUTHORIZED. Nothing was opened, read, built, loaded, "
              "played or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "run":
            res = run(a.role, a.segment)
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed(a.role, a.segment)["status"]}))
    except (MatchRefused, MatchUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
