"""H5 INCUMBENT DEFENCE MECHANISM STUDY -- EXPLORATORY.

Card: `docs/superpowers/2026-10-02-t1j-h5-incumbent-defence-mechanism-card.md`, with §9 A1.
🔴 THE CARD IS THE AUTHORITY. Five frozen DIAGNOSTIC FIXTURES, not a strength evaluation;
final-tree observations only; every causal reading is a hypothesis.

At each fixture the unchanged incumbent decides twice from fresh agents with the same
seed -- capture off, then capture on -- each through a study-local pass-through on its
own `mcts.search_with_root` that keeps THE returned objects (A1.1). The decision counts
only if configuration, both streams, the 400-simulation budget, the counts, the tie set,
the tree fingerprint and the move are identical, and the reader changed nothing (A1.3).
For each saving move the reader records A (excluded), B/B0 (never visited), C (visited,
not returned) or D (returned); at the episodes also R0/R1/R2, whether the search saw the
refutation of its returned move. Every value names its perspective (A1.2).

The run is the established contract (cap-completion card §7-§10): the parent claims and
watches, the child records on every exit. Gate read first at every public entry.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import eval_runner as ER
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence_authorization as AUTH
from . import h5_runner as H5
from . import h5_weakness_review as WR
from . import twixtbot_g3_reference as G3
from .mcts import decode_move, encode_move

LABEL = "EXPLORATORY"
CARD = "docs/superpowers/2026-10-02-t1j-h5-incumbent-defence-mechanism-card.md"
MODULE = "scripts.GPU.alphazero.h5_incumbent_defence"
FROZEN_SET = "docs/superpowers/evidence/2026-10-02-t1j-h5-cap-completion-EXPLORATORY/00_frozen_set.json"
FROZEN_SHA256 = "28c03f9f0be9accdb691b9325941da6227b856c17ddeff8e1cac5c91972e7cc6"
#: card §6: the five fixtures, in this order
POSITION_IDS = ("E1", "E2", "E3", "CD1", "CD2")
#: §9 A1.6: the controls' saving sets, derived by the frozen set's rule and PINNED
CONTROL_SAVING = {"CD1": ((16, 9), (17, 9)), "CD2": ((12, 4), (12, 5), (13, 5))}
SEED = -1                               # the Stage-2 convention; no research seed
HARD_LIMIT_S, MARGIN_S = 1800.0, 60.0
CLAIM_FILE, WATCHDOG_FILE, CLAIM_ENV = "claim.json", "watchdog.json", "H5_INCUMBENT_DEFENCE_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class DefenceRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class DefenceUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class DefenceStop(Exception):
    """A recorded STOP: a preservation check failed, or the deadline passed."""


# ─────────────────────────────── the fixtures (card §6, §9 A1.6) ───────────────────────────────

def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def derived_saving(state) -> List[Tuple[int, int]]:
    """The frozen set's rule: the mover's immediate wins, plus every move that leaves the
    opponent no immediate win (the weakness review's exact reduction)."""
    return sorted(set(CC.immediate_wins(state)) |
                  {_t(m) for m in WR.saving_moves_of(state, CC.threats_against(state))})


def positions(frozen: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The five fixtures from the committed frozen set, each re-checked by our engine."""
    out = []
    for e in POSITION_IDS[:3]:
        ep = frozen["episodes"][e]
        out.append({"id": e, "moves": ep["incumbent_position"]["moves"],
                    "W": [_t(w) for w in ep["W"]], "saving": [_t(s) for s in ep["saving"]],
                    "recorded": _t(ep["incumbent_recorded"]), "expect": "miss"})
    for c in POSITION_IDS[3:]:
        ct = frozen["controls"][c]
        out.append({"id": c, "moves": ct["position"]["moves"], "W": [_t(w) for w in ct["threat"]],
                    "saving": [_t(s) for s in CONTROL_SAVING[c]], "recorded": _t(ct["recorded"]),
                    "expect": "block"})
    for p in out:
        state = CC.state_of(p["moves"])
        if CC.threats_against(state) != sorted(p["W"]):
            raise DefenceRefused(f"{p['id']}: the opponent's immediate wins are "
                                 f"{CC.threats_against(state)}, not {p['W']}")
        if derived_saving(state) != sorted(p["saving"]):
            raise DefenceRefused(f"{p['id']}: the saving moves are {derived_saving(state)}, "
                                 f"not the pinned {p['saving']}")
        if CC.acceptable(state, p["recorded"], "defend") != (p["expect"] == "block"):
            raise DefenceRefused(f"{p['id']}: the recorded move does not {p['expect']}")
    return out


# ─────────────────────────────── the pass-through (§9 A1.1) ───────────────────────────────

def retain_search(mcts) -> Dict[str, List[Any]]:
    """Install on THIS instance a pass-through around its `search_with_root`: the same
    arguments in, THE returned tuple out, any exception propagating unchanged (nothing is
    caught). Each returned tuple is kept in `box["results"]`. No class is edited."""
    box: Dict[str, List[Any]] = {"results": []}
    real = mcts.search_with_root

    def search_with_root(*args, **kwargs):
        result = real(*args, **kwargs)
        box["results"].append(result)
        return result
    mcts.search_with_root = search_with_root
    return box


# ─────────────────────────────── the fingerprint (§9 A1.3) ───────────────────────────────

def _fx(v) -> Optional[str]:
    return None if v is None else float(v).hex()


def fingerprint(root, counts: Mapping, root_value) -> str:
    """sha256 over a canonical walk of EVERY node -- move id, visits, value sum, network
    value, every prior and raw prior IN INSERTION ORDER, whether priors IS priors_raw, the
    child keys -- children in move-id order; then the counts table and the root value."""
    h = hashlib.sha256()
    stack = [((), root)]
    while stack:
        path, n = stack.pop()
        h.update(json.dumps([list(path), n.move, n.visit_count, _fx(n.value_sum),
                             _fx(n.nn_value),
                             None if n.priors is None else [[k, _fx(p)] for k, p in n.priors.items()],
                             None if n.priors_raw is None else
                             [[k, _fx(p)] for k, p in n.priors_raw.items()],
                             n.priors is n.priors_raw, sorted(n.children)]).encode())
        for k in sorted(n.children, reverse=True):
            stack.append((path + (k,), n.children[k]))
    h.update(json.dumps([sorted([list(m), int(c)] for m, c in counts.items()),
                         _fx(root_value)]).encode())
    return h.hexdigest()


# ─────────────────────────────── the reader (card §2, §9 A1.2) ───────────────────────────────

def node_stat(parent, mid: int, depth: int) -> Dict[str, Any]:
    """One child of `parent`, `depth` plies below the root. `q_node` is the child's stored
    mean, for the player to move AT the child; `q_root_mover` is the same for the root's
    mover (sign by depth parity). Defined only for a visited child."""
    child = parent.children.get(mid)
    visits = 0 if child is None else child.visit_count
    q = float(child.q_value) if visits > 0 else None
    return {"move": list(decode_move(mid)),
            "prior": None if not parent.priors or mid not in parent.priors
            else float(parent.priors[mid]),
            "visits": visits, "q_node": q,
            "q_root_mover": None if q is None else q * (-1) ** depth,
            "q_note": None if q is not None else "undefined: never visited"}


def category(root, mid: int, returned: int) -> str:
    """Card §2.1: A excluded, B never visited (B0 with a prior of exactly 0.0), C visited
    and not returned, D returned."""
    if not root.priors or mid not in root.priors:
        return "A"
    if mid == returned:
        return "D"
    child = root.children.get(mid)
    if child is None or child.visit_count == 0:
        return "B0" if root.priors[mid] == 0.0 else "B"
    return "C"


def refutation(root, m: int, w: int) -> Dict[str, Any]:
    """Card §2.2 with §9 A1.2: R0 the returned move's node never expanded; R1 expanded but
    W never selected under it; R2 W visited under it -- each such terminal visit adds +1
    to m's value sum (the opponent's perspective) and -1 at the root."""
    mnode = root.children.get(m)
    if mnode is None or not mnode.is_expanded:
        return {"category": "R0", "m": node_stat(root, m, 1), "W": None,
                "W_note": "unavailable: the returned move's node was never expanded"}
    wnode = mnode.children.get(w)
    cat = "R1" if wnode is None or wnode.visit_count == 0 else "R2"
    return {"category": cat, "m": node_stat(root, m, 1), "W": node_stat(mnode, w, 2),
            "W_note": None}


def tie_set(counts: Mapping) -> List[List[int]]:
    top = max(counts.values())
    return [list(m) for m in sorted(m for m, c in counts.items() if c == top)]


def read(root, counts: Mapping, root_value, move, position: Mapping[str, Any]) -> Dict[str, Any]:
    """Everything the card records, read off the final tree. Reads ONLY."""
    returned = encode_move(*move)
    priors = root.priors or {}
    ranked = sorted(priors.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {mid: i + 1 for i, (mid, _p) in enumerate(ranked)}
    saving = [{**node_stat(root, encode_move(*s), 1), "category": category(root, encode_move(*s), returned),
               "prior_rank": rank.get(encode_move(*s))} for s in position["saving"]]
    visited = sorted((mid for mid, c in root.children.items() if c.visit_count > 0),
                     key=lambda mid: (-root.children[mid].visit_count, mid))
    return {
        "returned": {**node_stat(root, returned, 1), "prior_rank": rank.get(returned)},
        "saving": saving,
        "refutation": [refutation(root, returned, encode_move(*w)) for w in position["W"]]
        if position["expect"] == "miss" else None,
        "refutation_note": None if position["expect"] == "miss" else
        "not applicable: the returned move is expected to defend",
        "tie_set": tie_set(counts),
        "root_q_node": float(root_value),
        "root_q_note": "the root's mean, for the root's mover (the incumbent)",
        "root_nn_value": None if root.nn_value is None else float(root.nn_value),
        "root_nn_note": "the network's value for the player to move at the root",
        "candidates": len(priors),
        "root_priors": [[list(decode_move(mid)), float(p)] for mid, p in ranked],
        "root_children_visited": [node_stat(root, mid, 1) for mid in visited],
    }


# ─────────────────────────────── one decision, and its preservation (§5, §9 A1) ───────────────

def decide(build: Callable, state, *, capture: bool) -> Dict[str, Any]:
    """One decision from a FRESH agent, its search kept by the pass-through."""
    agent = build(state.to_move, capture)
    box = retain_search(agent.mcts)
    move = _t(agent(state))
    if len(box["results"]) != 1:
        raise DefenceStop(f"{len(box['results'])} searches for one decision, expected 1")
    counts, root_value, root = box["results"][0]
    if capture and (agent.last_capture["counts"] is not counts
                    or agent.last_capture["root"] is not root):
        raise DefenceStop("capture reports objects other than the ones the search returned")
    return {"agent": agent, "move": move, "counts": counts, "root_value": root_value,
            "root": root, "search_rng": agent.mcts.rng.getstate(),
            "readout_rng": agent.readout_rng.getstate()}


def observe(build: Callable, config, position: Mapping[str, Any]) -> Dict[str, Any]:
    """Capture off, then on; every §5 / §9 A1 check, or a STOP naming the position and the
    check; then the reading, shown not to have touched either tree or stream."""
    state = CC.state_of(position["moves"])
    off = decide(build, state, capture=False)
    on = decide(build, state, capture=True)
    passed: List[str] = []

    def need(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise DefenceStop(f"{position['id']}: preservation check '{name}' failed. {detail}")
        passed.append(name)

    for tag, r in (("off", off), ("on", on)):
        a = r["agent"]
        G3._check_config(a.config)
        need(f"configuration ({tag})", a.config is config and a.mcts.config == ER.cfg_from(config)
             and a.seed == SEED and a.colour == state.to_move)
        need(f"budget ({tag})", r["root"].visit_count == config.mcts_sims,
             f"root visits {r['root'].visit_count}")
    need("move", off["move"] == on["move"] == _t(position["recorded"]),
         f"off {off['move']}, on {on['move']}, recorded {position['recorded']}")
    need("search stream", off["search_rng"] == on["search_rng"])
    need("readout stream", off["readout_rng"] == on["readout_rng"])
    need("counts", off["counts"] == on["counts"])
    need("tie set", tie_set(off["counts"]) == tie_set(on["counts"]))
    fp = {tag: fingerprint(r["root"], r["counts"], r["root_value"])
          for tag, r in (("off", off), ("on", on))}
    need("tree fingerprint off == on", fp["off"] == fp["on"])
    readings = {}
    for tag, r in (("off", off), ("on", on)):
        streams = (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate())
        readings[tag] = read(r["root"], r["counts"], r["root_value"], r["move"], position)
        need(f"reader left the tree unchanged ({tag})",
             fingerprint(r["root"], r["counts"], r["root_value"]) == fp[tag])
        need(f"reader drew from no stream ({tag})",
             (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate()) == streams)
    need("readings agree", readings["off"] == readings["on"])
    return {"position": position["id"], "expect": position["expect"], "move": list(off["move"]),
            "fingerprint": fp["off"], "checks": passed, "reading": readings["off"]}


# ─────────────────────────────── the incumbent (unchanged) ───────────────────────────────

def _identity(config) -> Dict[str, Any]:
    """PURE: H5's bound incumbent identity, read off THE config object and checked against
    the H5 study manifest. No model is loaded."""
    identity = R4.frozen_incumbent_identity(config, design=H5.STUDY_DESIGN)
    R4.check_incumbent_identity(identity, config, design=H5.STUDY_DESIGN)
    raw = (pathlib.Path(R4.REPO_ROOT) / CC.MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CC.MANIFEST_SHA256:
        raise DefenceRefused("the H5 study manifest does not hash to its pin")
    if identity != json.loads(raw)["segments"][0]["incumbent_identity"]:
        raise DefenceRefused("the incumbent identity is not H5's bound identity")
    return identity


def tasks_for(identity: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The Stage-2 fixture tasks: seed -1 for the incumbent in either colour."""
    return {"red": R4.make_schedule([("h5-incumbent-defence-red", SEED, SEED - 2)],
                                    mode="fixture", reference=identity)[0],
            "black": R4.make_schedule([("h5-incumbent-defence-black", SEED - 2, SEED)],
                                      mode="fixture", reference=identity)[1]}


def study_builder(config, evaluator, tasks: Mapping[str, Mapping[str, Any]]) -> Callable:
    """`production_incumbent_build`'s construction, with the existing `capture` opt-in."""
    from . import e4_screen_reference as REF

    def build(colour: str, capture: bool):
        task = tasks[colour]
        return G3.build_reference_agent(task=task, evaluator=evaluator,
                                        colour=REF.reference_colour(task), config=config,
                                        capture=capture)
    return build


# ─────────────────────────────── the child's run (card §8, §9 A1.7) ───────────────────────────

def _unconsumed(d: pathlib.Path) -> bool:
    if not d.is_dir():
        return False
    names = set(os.listdir(d))
    return CLAIM_FILE in names and names <= {CLAIM_FILE, *CHILD_STREAMS.values()}


def _run(fixtures: Sequence[Mapping[str, Any]], config, identity: Mapping[str, Any],
         loader: Callable, d: pathlib.Path, *, clock: Callable[[], float] = time.monotonic,
         wall_s: float = HARD_LIMIT_S - MARGIN_S) -> Dict[str, Any]:
    """On a CLAIMED directory: the model load, then each fixture; the journal fsynced
    after each step; `record.json` on EVERY exit."""
    from . import d1_probe as D1
    if not _unconsumed(d):
        raise DefenceRefused(f"{d} does not hold exactly an unconsumed claim")
    deadline = D1.Deadline(limit_s=wall_s, clock=clock).start()
    log, step, rows = None, "claim", []
    status, failure, interrupt = "COMPLETE", None, None

    def due(next_step: str) -> None:
        if deadline.elapsed() > wall_s:
            raise DefenceStop(f"the {wall_s:.0f} s deadline passed before {next_step}")

    try:
        log = CC._Log(d / "log.jsonl")
        step = "model_load"
        with D1._supervisor(deadline):
            build = study_builder(config, loader(), tasks_for(identity))
            log.done("model_load", {"incumbent_identity": identity})
            for p in fixtures:
                step = f"observe:{p['id']}"
                due(step)
                row = observe(build, config, p)
                rows.append(row)
                log.done("position", row)
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt = "INTERRUPTED", e
        failure = {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
                   **CC._store(d, "failure", CC.carried(e))}
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status = "STOP"
        failure = {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
                   **CC._store(d, "failure", CC.carried(e))}
    finally:
        if log is not None:
            log.close()
    got, want = collections.Counter(r["position"] for r in rows), collections.Counter(POSITION_IDS)
    if status == "COMPLETE" and got != want:
        status, failure = "INCOMPLETE", {
            "failing_step": "completion_check", "exception_type": None,
            "reason": "the observed positions are not the five fixtures",
            "missing": sorted(want - got), "duplicated": sorted(k for k, n in got.items() if n > 1),
            "extra": sorted(set(got) - set(want))}
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "interpretable": status == "COMPLETE",
              "incumbent_identity": dict(identity), "seed": SEED,
              "frozen_set_sha256": FROZEN_SHA256, "rows": rows,
              "deadline_s": wall_s, "elapsed_s": deadline.elapsed(),
              "note": "what the fresh-agent search did at five diagnostic fixtures; the "
                      "in-game trees may differ and were not observed; every causal "
                      "reading is a hypothesis; not a strength evaluation"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── the parent (cap-completion card §9-§10) ───────────────────────

def _claim(d: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    import secrets
    try:
        os.mkdir(d)                              # THE CLAIM: create-only
    except FileExistsError:
        raise DefenceRefused(f"{d} is occupied: the study is one-shot") from None
    nonce = secrets.token_hex(16)
    claim = {"claimed_monotonic": time.monotonic(),
             "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_pid": os.getpid(),
             "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest()}
    with open(d / CLAIM_FILE, "x", encoding="utf-8") as fh:
        json.dump(claim, fh, indent=1, sort_keys=True)
    return claim, nonce


def _accept_claim(d: pathlib.Path) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise DefenceRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise DefenceRefused("the claim's nonce does not match: this process was not "
                             "started by the parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
    """The parent's own record on EVERY exit; the child's files are left as written."""
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
    journal = CC._journal(d)
    summary = {"label": LABEL, "card": CARD, "status": status, "cause": watch["cause"],
               "limit_s": watch["limit_s"], "elapsed_s": watch["elapsed_s"],
               "child_pid": watch["child_pid"], "child_rc": rc,
               "child_exit": "never started" if watch["child_pid"] is None else
               f"killed by signal {CC._signal_name(-rc)}" if rc < 0 else f"exit status {rc}",
               "child_streams": streams, "signal": watch["signal"],
               "survivors_killed": watch["survivors_killed"],
               "journal": journal, "journal_note": None if journal is not None else
               "no journal: the child stopped before opening it",
               "record_present": present, "record_readable": rec is not None,
               "record_status": None if rec is None else rec.get("status"),
               "note": "the journal and any record are exactly as the child wrote them"}
    with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


# ─────────────────────────────── the public entries ───────────────────────────────

def _gate() -> None:
    if not AUTH.H5_INCUMBENT_DEFENCE_AUTHORIZED:
        raise DefenceUnauthorized("the H5 incumbent defence study is UNAUTHORIZED. Nothing "
                                  "was opened, read, built, loaded, decided or written.")


def _evidence_dir(out_dir: str) -> pathlib.Path:
    p = pathlib.Path(out_dir).resolve()
    if "EXPLORATORY" not in p.name or not R4._inside(p, pathlib.Path(R4.EVIDENCE_ROOT)):
        raise DefenceRefused(f"{out_dir}: the evidence directory must be under the evidence "
                             f"tree and named EXPLORATORY")
    return p


def _frozen() -> Dict[str, Any]:
    path = pathlib.Path(R4.REPO_ROOT) / FROZEN_SET
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise DefenceRefused(f"the frozen set is not durable: {e}") from None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise DefenceRefused("the frozen set does not hash to its pin")
    return json.loads(raw)


def _child_argv(out: pathlib.Path) -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed", "--out-dir", str(out)]


def run(out_dir: str) -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals, THE CLAIM, then the child in its own
    session under the hard watchdog; `watchdog.json` on every exit. No seam is accepted."""
    _gate()
    out = _evidence_dir(out_dir)
    if os.path.lexists(out):
        raise DefenceRefused(f"{out_dir} is occupied: the study is one-shot")
    positions(_frozen())
    _identity(R4.frozen_argmax_config())
    claim, nonce = _claim(out)
    watch = CC._watch(_child_argv(out),
                      limit_s=HARD_LIMIT_S - (time.monotonic() - claim["claimed_monotonic"]),
                      env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                      cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                      stderr_path=out / CHILD_STREAMS["stderr"])
    summary = _settle(out, watch)
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def run_claimed(out_dir: str) -> Dict[str, Any]:
    """The CHILD, started only by `run` with the claim's nonce: the gate and the pure
    checks again, the claim, then the model load and the five fixtures."""
    _gate()
    out = _evidence_dir(out_dir)
    fixtures = positions(_frozen())
    config = R4.frozen_argmax_config()
    identity = _identity(config)
    claim = _accept_claim(out)
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(fixtures, config, identity, lambda: R4.load_production_evaluator(identity), out,
                wall_s=HARD_LIMIT_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_incumbent_defence",
                                 description="H5 incumbent defence study -- EXPLORATORY. "
                                             "NOT AUTHORIZED.")
    ap.add_argument("step", choices=("run", "run-claimed"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if not AUTH.H5_INCUMBENT_DEFENCE_AUTHORIZED:
        print("the H5 incumbent defence study is UNAUTHORIZED. Nothing was opened, read, "
              "built, loaded, decided or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "run":
            res = run(a.out_dir)
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed(a.out_dir)["status"]}))
    except (DefenceRefused, DefenceUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
