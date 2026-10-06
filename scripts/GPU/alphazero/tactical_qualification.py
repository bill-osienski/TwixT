"""THE TACTICAL CANDIDATE QUALIFICATION RUN: production-path qualification, never strength.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §17 A7.

BINDINGS FIRST (A7.1). Before any model load -- in the parent before its claim, in the child
before the load -- the published corpus is accepted by `committed_corpus()`, the code the corpus
recorded is the current code, R1's runner and the fixture modules equal their pinned hashes, and
the prior-position file (the boundary positions) is the pinned one.

THE CALLS (A7.2), exactly: the five pinned incumbent reproductions; the 100 corpus positions, each
decided by a fresh plain incumbent and a fresh candidate -- identical search, configuration and
both random streams, and the M1 readout as specified; four boundaries (a winning 280th move; a
void threat at ply 279; ply 280 and an 8x8 board refused before any search). Every agent comes
from R1's `production_builder` with seed -1 for the colour to move.

RECORDING (A7.3): one-shot destination; the parent's 1 h hard limit (process-group SIGKILL); a
soft per-call child alarm; every completed call durable before the next; record.json on every exit
with the completion check and the verdict inside the boundary; watchdog.json from the parent on
every exit; no retry. Gate: `tactical_candidate_authorization`, created closed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import candidate_match_runner as CM
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_defence_factors as X
from . import h5_incumbent_tactics as T
from . import tactical_candidate as TC
from . import tactical_candidate_authorization as AUTH
from . import tactical_corpus as TQ
from .game.twixt_state import TwixtState

MODULE = "scripts.GPU.alphazero.tactical_qualification"
CARD = "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"
LABEL = "TACTICAL CANDIDATE QUALIFICATION"
SEED = -1                                  # the seed of the pinned fingerprints; synthetic, never research
CODE_PINS = {                              # §17 A7.1 -- reused or read here, pinned at implementation
    "candidate_match_runner.py": "a53e7ac857f045a281af85d7dd268445e2419e1c71fd2a0458092a5c2c7e21b6",
    "h5_incumbent_defence.py": "d3328c324e3e9779a3d9e0213a67b459ae1a2e6d6c3ce1757d7644629619e5af",
    "h5_incumbent_defence_factors.py": "098a86c67386157b9b15e77ef931f54800b90ff9ec1d4e596286a0713e0bf997"}
FIXTURES = ("E1", "E2", "E3", "CD1", "CD2")
BOUNDARIES = ("own-win-279", "threat-279", "ply-280", "board-8")
HARD_LIMIT_S, MARGIN_S, CALL_S = 3600.0, 60.0, 120.0
OUT_NAME = "tactical-candidate-qualification"
OBSERVATIONS = "observations.jsonl"
CLAIM_FILE, WATCHDOG_FILE, CLAIM_ENV = "claim.json", "watchdog.json", "TACTICAL_QUALIFICATION_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class QualRefused(Exception):
    """A binding, input or claim failed: nothing further runs."""


class QualUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, loaded, decided or written."""


class QualStop(Exception):
    """A qualification check failed: recorded, never retried."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ─────────────────────────────── bindings and the call list (§17 A7.1, A7.2) ─────────────────

def bindings() -> Dict[str, Any]:
    """§17 A7.1: refused unless the corpus, the code and the prior file are the bound ones."""
    try:
        corpus, sha = TQ.committed_corpus()
    except TQ.CorpusRefused as e:
        raise QualRefused(f"the corpus is not accepted: {e}") from None
    here = pathlib.Path(TQ.__file__).parent
    for name, want in sorted(corpus["code"].items()):
        if CM._file_sha(str(here / name)) != want:
            raise QualRefused(f"{name} is not the code the corpus recorded")
    for name, want in sorted(CODE_PINS.items()):
        if CM._file_sha(str(here / name)) != want:
            raise QualRefused(f"{name} is not the pinned code")
    raw = TQ.PRIOR_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TQ.PRIOR_SHA256:
        raise QualRefused("the prior-position file is not the pinned one")
    return {"corpus": corpus, "corpus_sha256": sha, "prior": json.loads(raw)}


def plan_of(corpus: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """§17 A7.2: 5 reproductions, every corpus position in its published order, 4 boundaries."""
    return ([{"id": f"repro:{f}", "kind": "repro", "fixture": f} for f in FIXTURES] +
            [{"id": f"corpus:{i:03d}", "kind": "corpus", "index": i} for i in range(len(corpus["selected"]))] +
            [{"id": f"boundary:{b}", "kind": "boundary", "name": b} for b in BOUNDARIES])


# ─────────────────────────────── one call ───────────────────────────────

def _decide(make: Callable, config, colour: str, role: str, state: TwixtState) -> Dict[str, Any]:
    """A FRESH agent from the production builder, seed -1, one decision, its search kept."""
    agent = make(colour, SEED, role)
    inner = agent.agent if isinstance(agent, TC.TacticalCandidate) else agent
    box = D.retain_search(inner.mcts)
    move = _t(agent(state))
    if len(box["results"]) != 1:
        raise QualStop(f"{len(box['results'])} searches for one {role} decision, expected 1")
    if (inner.seed, inner.colour) != (SEED, colour):
        raise QualStop(f"the {role} was built with seed {inner.seed} as {inner.colour}")
    counts, root_value, root = box["results"][0]
    return {"agent": agent, "move": move, "counts": counts, "root_value": root_value,
            "fp": D.fingerprint(root, counts, root_value), "search": inner.mcts.rng.getstate(),
            "readout": inner.readout_rng.getstate(), "config_diff": sorted(X.config_diff(inner, config))}


def _equivalent(make: Callable, config, state: TwixtState, need: Callable) -> Tuple[Dict, Dict, Dict]:
    """§17 A7.2: the candidate's search IS the incumbent's; one witness, its baseline the move."""
    inc = _decide(make, config, state.to_move, "incumbent", state)
    cand = _decide(make, config, state.to_move, "candidate", state)
    need("configuration (incumbent)", inc["config_diff"] == [])
    need("configuration (candidate)", cand["config_diff"] == [])
    need("counts", inc["counts"] == cand["counts"])
    need("tie set", D.tie_set(inc["counts"]) == D.tie_set(cand["counts"]))
    need("tree fingerprint", inc["fp"] == cand["fp"])
    need("root value", inc["root_value"] == cand["root_value"])
    need("search stream", inc["search"] == cand["search"])
    need("readout stream", inc["readout"] == cand["readout"])
    ws = cand["agent"].witnesses
    need("one witness", len(ws) == 1)
    need("the baseline is the incumbent's move", _t(ws[0]["baseline"]["move"]) == inc["move"])
    return inc, cand, ws[0]


def _readout(state: TwixtState, inc: Dict, cand: Dict, w: Mapping[str, Any], need: Callable) -> None:
    """§17 A7.2: M1 as specified -- the baseline when inactive or admissible, else the most-visited
    admissible move; inside the admissible set whenever the rule is active."""
    c = T.classify(state, semantics="protocol")
    need("the witness class", w["class"] == c)
    if c is None or c["class"] in TC.INACTIVE:
        expect = inc["move"]
    else:
        A = [_t(m) for m in (c["W"] if c["class"] == "OWN_WIN" else c["D"])]
        counts = inc["counts"]
        expect = inc["move"] if inc["move"] in A else min(A, key=lambda m: (-counts.get(m, 0), m))
        need("an active rule ends inside the admissible set", cand["move"] in A)
    need("the candidate's move follows M1", cand["move"] == expect)
    need("the witness records the move", _t(w["final"]["move"]) == cand["move"] and
         w["override"] == (cand["move"] != inc["move"]))


def _refused_before_search(make: Callable, state: TwixtState, why: type, need: Callable) -> None:
    """§17 A7.2: a candidate, enabled and disabled, refuses `state` before ANY search."""
    for enabled in (True, False):
        tag = "enabled" if enabled else "disabled"
        cand = make(state.to_move, SEED, "candidate")
        agent = cand if enabled else TC.TacticalCandidate(cand.agent, enabled=False)
        entered, real = [], cand.agent.mcts.search_with_root
        cand.agent.mcts.search_with_root = lambda *a, _e=entered, _r=real, **k: (_e.append(1),
                                                                                 _r(*a, **k))[1]
        streams = (cand.agent.mcts.rng.getstate(), cand.agent.readout_rng.getstate())
        try:
            agent(state)
            refused = False
        except why:
            refused = True
        need(f"refused ({tag})", refused)
        need(f"no search ({tag})", entered == [] and
             (cand.agent.mcts.rng.getstate(), cand.agent.readout_rng.getstate()) == streams)


def observe(call: Mapping[str, Any], make: Callable, config, bound: Mapping[str, Any]) -> Dict[str, Any]:
    """One call of the list. Every check is named; the first failure STOPS."""
    passed: List[str] = []

    def need(name: str, ok: bool) -> None:
        if not ok:
            raise QualStop(f"{call['id']}: check '{name}' failed")
        passed.append(name)
    row: Dict[str, Any] = {"id": call["id"], "kind": call["kind"]}
    if call["kind"] == "repro":
        p = {q["id"]: q for q in D.positions(D._frozen())}[call["fixture"]]
        state = CC.state_of(p["moves"])
        inc = _decide(make, config, state.to_move, "incumbent", state)
        need("configuration", inc["config_diff"] == [])
        need("the recorded move", inc["move"] == _t(p["recorded"]))
        need("the pinned fingerprint", inc["fp"] == X.PINNED_FINGERPRINTS[call["fixture"]])
        row.update(move=list(inc["move"]), fingerprint=inc["fp"])
    elif call["kind"] == "corpus":
        e = bound["corpus"]["selected"][call["index"]]
        state = T.replay(e["moves"])
        need("the corpus position", (T.state_digest(state), T.classify(state, semantics="protocol")) ==
             (e["state_digest"], e["class_dict"]))
        inc, cand, w = _equivalent(make, config, state, need)
        need("the witness class is the recorded class", w["class"] == e["class_dict"])
        _readout(state, inc, cand, w, need)
        row.update(cls=e["class"], stream=e["stream"], ply=e["ply"], rule=w["rule"], override=w["override"],
                   unsearched=w["unsearched"], baseline=w["baseline"]["move"], final=w["final"]["move"],
                   fingerprint=inc["fp"], scan_s=w["scan_s"], search_s=w["search_s"])
    elif call["name"] == "own-win-279":
        state = T.replay(bound["prior"]["pinned"]["OWN_WIN_279"])
        inc, cand, w = _equivalent(make, config, state, need)
        _readout(state, inc, cand, w, need)
        need("the 280th move wins", state.apply_move(cand["move"]).winner() == state.to_move)
        row.update(rule=w["rule"], final=w["final"]["move"])
    elif call["name"] == "threat-279":
        state = T.replay(bound["prior"]["pinned"]["THREAT_279"])
        inc, cand, w = _equivalent(make, config, state, need)
        _readout(state, inc, cand, w, need)
        need("the void threat leaves the rule inactive", w["rule"] == "inactive" and bool(w["class"]["T_void"]))
        row.update(rule=w["rule"], final=w["final"]["move"])
    elif call["name"] == "ply-280":
        s = T.replay(bound["prior"]["pinned"]["OWN_WIN_279"])
        state = s.apply_move(next(m for m in sorted(s.legal_moves()) if s.apply_move(m).winner() is None))
        need("a non-winning position at ply 280", (state.ply, state.winner()) == (280, None))
        _refused_before_search(make, state, TC.CandidateRefused, need)
    elif call["name"] == "board-8":
        _refused_before_search(make, TwixtState(active_size=8, to_move="red"), T.TacticsRefused, need)
    else:
        raise QualStop(f"{call['id']}: no such call")
    row["checks"] = passed
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§17 A7.3, descriptive only: per class, the rule outcomes, the unsearched choices, timing."""
    out = {}
    for cls in T.CLASSES:
        rs = [r for r in rows if r.get("cls") == cls]
        out[cls] = {"positions": len(rs), "rules": dict(collections.Counter(r["rule"] for r in rs)),
                    "overrides": sum(r["override"] for r in rs), "unsearched": sum(r["unsearched"] for r in rs),
                    "streams": dict(collections.Counter(r["stream"] for r in rs)),
                    "median_scan_s": statistics.median(r["scan_s"] for r in rs) if rs else None,
                    "median_search_s": statistics.median(r["search_s"] for r in rs) if rs else None}
    return {"per_class": out, "note": "production-path qualification: no strength claim"}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def _run(plan: Sequence[Mapping[str, Any]], config, identity: Mapping[str, Any], loader: Callable,
         d: pathlib.Path, bound: Mapping[str, Any], *, clock: Callable[[], float] = time.monotonic,
         wall_s: float = HARD_LIMIT_S - MARGIN_S, call_s: float = CALL_S) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise QualRefused(f"{d} does not hold exactly an unconsumed claim")
    t0 = clock()
    left = lambda: wall_s - (clock() - t0)    # noqa: E731
    log, obs, step, rows = None, None, "claim", []
    status, failure, interrupt, summary = "COMPLETE", None, None, None
    try:
        log = CC._Log(d / "log.jsonl")
        step = "model_load"
        with CM._alarm(left(), "the qualification (model load)"):
            evaluator = loader()
        make = CM.production_builder(config, evaluator, identity)
        log.done("model_load", {"incumbent_identity": dict(identity)})
        obs = CM._Partial(d / OBSERVATIONS)
        for call in plan:
            step = call["id"]
            if left() <= 0:
                raise QualStop(f"the {wall_s:.0f} s deadline passed before {step}")
            log.done("call_start", {"id": call["id"]})
            with CM._alarm(min(call_s, left()), f"call {call['id']}"):
                row = observe(call, make, config, bound)
            obs(row)                             # durable before the next call
            rows.append(row)
            log.done("call", {"id": call["id"], "checks": len(row["checks"])})
        obs.close()
        obs = None
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    finally:
        if obs is not None:
            obs.close()
        if log is not None:
            log.close()
    if status == "COMPLETE":                     # inside the recorded boundary
        step = "completion_check"
        try:
            if [r["id"] for r in rows] != [c["id"] for c in plan]:
                status, failure = "INCOMPLETE", {
                    "failing_step": "completion_check", "exception_type": None,
                    "reason": "the observations are not the call list",
                    "missing": [c["id"] for c in plan if c["id"] not in {r["id"] for r in rows}]}
            else:
                step = "interpretation"
                summary = summarize(rows)
        except (KeyboardInterrupt, SystemExit) as e:
            status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            status, failure, summary = "STOP", _failure(d, step, e), None
    record = {"label": LABEL, "card": CARD, "status": status,
              "verdict": "QUALIFIED" if status == "COMPLETE" else "NOT_QUALIFIED", "failure": failure,
              "corpus_sha256": bound["corpus_sha256"], "plan_sha256": _sha(list(plan)), "calls": len(plan),
              "observed": len(rows), "incumbent_identity": dict(identity), "seed": SEED,
              "observations_sha256": CM._file_sha(str(d / OBSERVATIONS)) if (d / OBSERVATIONS).exists() else None,
              "summary": summary, "deadline_s": wall_s, "call_s": call_s, "elapsed_s": clock() - t0,
              "note": "production-path qualification on the frozen corpus; NOT strength"}
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
        raise QualRefused(f"{d} is occupied: the qualification is one-shot") from None
    nonce = secrets.token_hex(16)
    claim = {"claimed_monotonic": time.monotonic(),
             "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_pid": os.getpid(), "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(), **extra}
    with open(d / CLAIM_FILE, "x", encoding="utf-8") as fh:
        json.dump(claim, fh, indent=1, sort_keys=True)
    return claim, nonce


def _accept_claim(d: pathlib.Path) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise QualRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise QualRefused("the claim's nonce does not match: this process was not started by the "
                          "parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
    """watchdog.json: statuses, the verdict as recorded, streams, journal, record and observations
    by hash."""
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
        streams[s] = None if raw is None else {"file": name, "bytes": len(raw),
                                               "sha256": hashlib.sha256(raw).hexdigest()}
    summary = {"label": LABEL, "card": CARD, "status": status, "cause": watch["cause"],
               "limit_s": watch["limit_s"], "elapsed_s": watch["elapsed_s"], "child_pid": watch["child_pid"],
               "child_rc": rc, "child_exit": "never started" if watch["child_pid"] is None else
               f"killed by signal {CC._signal_name(-rc)}" if rc < 0 else f"exit status {rc}",
               "child_streams": streams, "signal": watch["signal"], "survivors_killed": watch["survivors_killed"],
               "journal": CC._journal(d), "record_present": present, "record_readable": rec is not None,
               "record_status": None if rec is None else rec.get("status"),
               "verdict": "QUALIFIED" if status == "COMPLETE" and rec.get("verdict") == "QUALIFIED"
               else "NOT_QUALIFIED",
               "record_sha256": CM._file_sha(str(d / "record.json")) if present else None,
               "observations_sha256": CM._file_sha(str(d / OBSERVATIONS)) if (d / OBSERVATIONS).exists()
               else None}
    with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


def _record_settle_failure(d: pathlib.Path, e: BaseException) -> None:
    """A settlement that failed or was interrupted still leaves watchdog.json (as §16 A6.3)."""
    if (d / WATCHDOG_FILE).exists():
        return
    try:
        with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
            json.dump({"label": LABEL, "card": CARD, "verdict": "NOT_QUALIFIED",
                       "status": "INTERRUPTED" if isinstance(e, (KeyboardInterrupt, SystemExit))
                       else "SETTLE_FAILED",
                       "failure": {"exception_type": type(e).__name__, "reason": str(e)},
                       "record_present": (d / "record.json").exists()}, fh, indent=1, sort_keys=True)
    except Exception:                            # noqa: BLE001 -- nothing more can be recorded
        pass


# ─────────────────────────────── the public entries ───────────────────────────────

def _gate() -> None:
    if not AUTH.TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED:
        raise QualUnauthorized("the qualification run is UNAUTHORIZED. Nothing was opened, read, "
                               "loaded, decided or written.")


def out_dir() -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / OUT_NAME


def _child_argv() -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed"]


def run() -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Bindings (pure), THE CLAIM, the child under the 1 h hard
    limit; watchdog.json on every exit."""
    _gate()
    out = out_dir()
    if os.path.lexists(out):
        raise QualRefused(f"{out} is occupied: the qualification is one-shot")
    bound = bindings()
    plan = plan_of(bound["corpus"])
    D._identity(R4.frozen_argmax_config())
    claim, nonce = _claim(out, {"corpus_sha256": bound["corpus_sha256"], "plan_sha256": _sha(plan)})
    try:
        watch = CC._watch(_child_argv(),
                          limit_s=HARD_LIMIT_S - (time.monotonic() - claim["claimed_monotonic"]),
                          env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                          cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                          stderr_path=out / CHILD_STREAMS["stderr"])
        summary = _settle(out, watch)
    except BaseException as e:
        _record_settle_failure(out, e)
        raise
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def run_claimed() -> Dict[str, Any]:
    """The CHILD: the gate, the bindings, the claim (nonce, corpus and call-list hashes); then the
    single model load and the calls."""
    _gate()
    out = out_dir()
    bound = bindings()
    plan = plan_of(bound["corpus"])
    config = R4.frozen_argmax_config()
    identity = D._identity(config)
    claim = _accept_claim(out)
    if (claim.get("corpus_sha256"), claim.get("plan_sha256")) != (bound["corpus_sha256"], _sha(plan)):
        raise QualRefused("the claim is for another corpus or call list")
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(plan, config, identity, lambda: R4.load_production_evaluator(identity), out, bound,
                wall_s=HARD_LIMIT_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="tactical_qualification",
                                 description="Tactical candidate qualification run. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("run", "run-claimed"))
    a = ap.parse_args(argv)
    if not AUTH.TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED:
        print("the qualification run is UNAUTHORIZED. Nothing was opened, read, loaded, decided or "
              "written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "run":
            res = run()
            print(json.dumps({k: res[k] for k in ("status", "record_status", "verdict", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed()["status"]}))
    except (QualRefused, QualUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
