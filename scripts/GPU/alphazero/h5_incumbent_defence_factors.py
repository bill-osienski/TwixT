"""H5 INCUMBENT DEFENCE -- CONTROLLED ONE-FACTOR STUDY -- EXPLORATORY.

Card: `docs/superpowers/2026-10-02-t1j-h5-incumbent-defence-controlled-study-card.md`,
with §9 A1. 🔴 THE CARD IS THE AUTHORITY. Five frozen DIAGNOSTIC FIXTURES; not a strength
evaluation. The absolute-FPU levels are diagnostic probes, NEVER candidate settings
(§9 A1.5); the policy-mass rules are not tested.

Arms, in this order, each changing ONE factor from the unchanged incumbent:
  B-repro  seed -1, frozen config: must reproduce `912b601f` (move + fingerprint) -- else STOP
  B-seed   seeds -3, -5, frozen config
  F        seed -1, MCTSConfig.fpu_value in {-0.20, -0.50, -1.0}      (whole tree)
  S        seed -1, budget in {1600, 6400} (EvalConfig.mcts_sims == MCTSConfig.n_simulations)
  R        post-tree on the B-repro trees: ties by prior, ties by the root mover's value
Every decision runs capture off then on from fresh agents; checks are ARM-SPECIFIC
(§9 A1.3). Refutation is recomputed after the ACTUAL returned move (§9 A1.4).

`h5_incumbent_defence` is reused by import and not edited. The run follows the established
contract: the parent claims and watches; the child records on every exit.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import eval_runner as ER
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_defence_factors_authorization as AUTH
from .mcts import encode_move

LABEL = "EXPLORATORY"
CARD = "docs/superpowers/2026-10-02-t1j-h5-incumbent-defence-controlled-study-card.md"
MODULE = "scripts.GPU.alphazero.h5_incumbent_defence_factors"
#: §9 A1.6: the ONLY output directory, a subdirectory of the evidence root
OUT_NAME = "t1j-h5-incumbent-defence-factors-EXPLORATORY"
#: §9 A1.3: the fingerprints `912b601f` recorded at seed -1 -- B-repro must reproduce them
PINNED_FINGERPRINTS = {
    "E1": "109ad0a992682f64288bedbd81fe536accc90fb33f296167056a4153575aebb5",
    "E2": "99b8281cf5768abf01f49043273f239d7b6c7f49ff574a2f9b2b4b44c9f799de",
    "E3": "87c82ac0230085d0755187d8db48035fb189be12c485cbf5c17ddeff2b6a16d9",
    "CD1": "514d983a50dec80ea6df7d7dbe19c168c2c54533899d2f6a046f556956ab79c8",
    "CD2": "e9a563aaca14c98a1f7d217b20640275dd19c80796ec35ced2e137cc6ecb279f",
}
BASE_SEED, SEED_ARMS = -1, (-3, -5)
FPU_LEVELS, BUDGETS = (-0.20, -0.50, -1.0), (1600, 6400)
#: §9 A1.1: the six frozen cells -- True = the full sweep PERSISTS, False = it breaks
PREDICTED_SWEEP = {-0.20: {"E1": True, "E2": True}, -0.50: {"E1": False, "E2": True},
                   -1.0: {"E1": False, "E2": False}}
HARD_LIMIT_S, MARGIN_S = 1800.0, 60.0
CLAIM_FILE, WATCHDOG_FILE = "claim.json", "watchdog.json"
CLAIM_ENV = "H5_INCUMBENT_DEFENCE_FACTORS_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class FactorRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class FactorUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class FactorStop(Exception):
    """A recorded STOP: an arm-specific check failed, or the deadline passed."""


# ─────────────────────────────── the arms (card §2-§3, §9 A1.3) ───────────────────────────────

@dataclasses.dataclass(frozen=True)
class Arm:
    name: str                       # B-repro | B-seed | F | S
    seed: int
    fpu: Optional[float] = None     # F only
    budget: int = 400               # S changes it

    @property
    def label(self) -> str:
        return (f"{self.name}" + (f"@seed{self.seed}" if self.name == "B-seed" else "") +
                (f"@fpu{self.fpu}" if self.fpu is not None else "") +
                (f"@sims{self.budget}" if self.name == "S" else ""))

    def allowed_diff(self) -> set:
        """§9 A1.3: the ONLY configuration fields that may differ from the frozen ones."""
        return {"F": {"MCTSConfig.fpu_value"},
                "S": {"EvalConfig.mcts_sims", "MCTSConfig.n_simulations"}}.get(self.name, set())


def plan() -> List[Arm]:
    """The frozen order: B-repro first (a failure STOPs before every other arm)."""
    return ([Arm("B-repro", BASE_SEED)] + [Arm("B-seed", s) for s in SEED_ARMS] +
            [Arm("F", BASE_SEED, fpu=f) for f in FPU_LEVELS] +
            [Arm("S", BASE_SEED, budget=b) for b in BUDGETS])


def tasks_for_seed(identity: Mapping[str, Any], seed: int) -> Dict[str, Dict[str, Any]]:
    """`D.tasks_for` at an arbitrary synthetic seed (seed -1 gives D's tasks exactly)."""
    return {"red": R4.make_schedule([("h5-incumbent-defence-red", seed, seed - 2)],
                                    mode="fixture", reference=identity)[0],
            "black": R4.make_schedule([("h5-incumbent-defence-black", seed - 2, seed)],
                                      mode="fixture", reference=identity)[1]}


def arm_builder(arm: Arm, config, evaluator, identity: Mapping[str, Any]) -> Callable:
    """Study-local construction: the production builder with THE frozen config, then
    exactly the arm's factor applied. The production builder and `_check_config` are
    untouched."""
    base = D.study_builder(config, evaluator, tasks_for_seed(identity, arm.seed))

    def build(colour: str, capture: bool):
        agent = base(colour, capture)
        if arm.name == "F":
            agent.mcts.config = dataclasses.replace(agent.mcts.config, fpu_value=arm.fpu)
        elif arm.name == "S":
            agent.config = dataclasses.replace(agent.config, mcts_sims=arm.budget)
            agent.mcts.config = dataclasses.replace(agent.mcts.config,
                                                    n_simulations=arm.budget)
        return agent
    return build


def config_diff(agent, config) -> set:
    """Every field, type-strictly, in which the agent's EvalConfig / MCTSConfig differ
    from the frozen ones."""
    out = set()
    for tag, got, want in (("EvalConfig", agent.config, config),
                           ("MCTSConfig", agent.mcts.config, ER.cfg_from(config))):
        if type(got) is not type(want):
            out.add(f"{tag}.<type>")
            continue
        for f in dataclasses.fields(want):
            a, b = getattr(got, f.name), getattr(want, f.name)
            if type(a) is not type(b) or a != b:
                out.add(f"{tag}.{f.name}")
    return out


# ─────────────────────────────── refutation after the returned move (§9 A1.4) ─────────────────

def refutation_after(root, state, move) -> Dict[str, Any]:
    """TERMINAL if the returned move ends the game; NEUTRALIZED if it leaves the opponent no
    immediate win; otherwise R0/R1/R2 for EACH of the opponent's immediate wins after it --
    a set that may differ from the frozen W."""
    after = state.apply_move(D._t(move))
    if after.winner() is not None:
        return {"status": "TERMINAL", "winner": after.winner(), "refutation_set": [],
                "refutations": []}
    wins = CC.immediate_wins(after)
    if not wins:
        return {"status": "NEUTRALIZED", "winner": None, "refutation_set": [],
                "refutations": []}
    m = encode_move(*D._t(move))
    return {"status": "OPEN", "winner": None, "refutation_set": [list(w) for w in wins],
            "refutations": [{"W": list(w), **D.refutation(root, m, encode_move(*w))}
                            for w in wins]}


# ─────────────────────────────── the sweep and the readout variants ───────────────────────────

def sweep(root, counts: Mapping) -> Dict[str, Any]:
    """§5.1 / §9 A1.1: a FULL sweep is max root-child visits == 1 with every visited child
    in the tie set."""
    visits = [c.visit_count for c in root.children.values() if c.visit_count > 0]
    tie = D.tie_set(counts)
    top = max(visits) if visits else 0
    return {"visited_children": len(visits), "max_visits": top, "tie_set_size": len(tie),
            "histogram": {str(k): v for k, v in sorted(collections.Counter(visits).items())},
            "full_sweep": top == 1 and len(tie) == len(visits)}


def readout_variants(root, counts: Mapping, saving: Sequence) -> Dict[str, Any]:
    """§3.3: the R factor, post-tree -- the visit argmax's TIE broken by prior (descending)
    or by the root mover's value of the visited child (descending), then by (row, col).
    Recorded, not tested: both were seen at E1/E2 during review."""
    top = max(counts.values())
    tied = sorted(m for m, c in counts.items() if c == top)

    def stat(m):
        child = root.children.get(encode_move(*m))
        prior = float(root.priors[encode_move(*m)])
        q_root = None if child is None or child.visit_count == 0 else -float(child.q_value)
        return prior, q_root
    by_prior = min(tied, key=lambda m: (-stat(m)[0], m))
    valued = [m for m in tied if stat(m)[1] is not None]
    by_value = min(valued, key=lambda m: (-stat(m)[1], m)) if valued else None
    save = {D._t(s) for s in saving}
    return {"tie_set_size": len(tied),
            "R-prior": {"move": list(by_prior), "defends": by_prior in save},
            "R-value": None if by_value is None else
            {"move": list(by_value), "defends": by_value in save},
            "note": "recorded, not tested: contaminated at E1/E2 (§3.3)"}


# ─────────────────────────────── one arm at one fixture (§9 A1.3) ─────────────────────────────

def observe(arm: Arm, build: Callable, config, position: Mapping[str, Any]) -> Dict[str, Any]:
    """Capture off, then on; the ARM-SPECIFIC checks, or a STOP naming arm, fixture and
    check; then the reading, with refutation recomputed after the returned move."""
    state = CC.state_of(position["moves"])
    off = D.decide(build, state, capture=False)
    on = D.decide(build, state, capture=True)
    passed: List[str] = []

    def need(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise FactorStop(f"{arm.label} {position['id']}: check '{name}' failed. {detail}")
        passed.append(name)

    for tag, r in (("off", off), ("on", on)):
        a = r["agent"]
        diff = config_diff(a, config)
        need(f"configuration ({tag})", diff == arm.allowed_diff(),
             f"differs in {sorted(diff)}, allowed {sorted(arm.allowed_diff())}")
        need(f"factor value ({tag})", a.mcts.config.fpu_value == (arm.fpu or 0.0) and
             a.config.mcts_sims == a.mcts.config.n_simulations == arm.budget)
        need(f"seed and colour ({tag})", a.seed == arm.seed and a.colour == state.to_move)
        need(f"budget ({tag})", r["root"].visit_count == arm.budget,
             f"root visits {r['root'].visit_count}, declared {arm.budget}")
    need("move off == on", off["move"] == on["move"], f"off {off['move']}, on {on['move']}")
    need("search stream", off["search_rng"] == on["search_rng"])
    need("readout stream", off["readout_rng"] == on["readout_rng"])
    need("counts", off["counts"] == on["counts"])
    need("tie set", D.tie_set(off["counts"]) == D.tie_set(on["counts"]))
    fp = {tag: D.fingerprint(r["root"], r["counts"], r["root_value"])
          for tag, r in (("off", off), ("on", on))}
    need("tree fingerprint off == on", fp["off"] == fp["on"])
    if arm.name == "B-repro":
        need("B-repro move", off["move"] == D._t(position["recorded"]),
             f"returned {off['move']}, recorded {position['recorded']}")
        need("B-repro fingerprint", fp["off"] == PINNED_FINGERPRINTS[position["id"]],
             f"{fp['off'][:16]}… vs pinned {PINNED_FINGERPRINTS[position['id']][:16]}…")
    readings = {}
    for tag, r in (("off", off), ("on", on)):
        streams = (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate())
        reading = D.read(r["root"], r["counts"], r["root_value"], r["move"],
                         {**position, "expect": "block"})   # D's own refutation is NOT used
        reading.pop("refutation_note")
        reading["refutation"] = refutation_after(r["root"], state, r["move"])
        reading["sweep"] = sweep(r["root"], r["counts"])
        if arm.name == "B-repro":
            reading["readout_variants"] = readout_variants(r["root"], r["counts"],
                                                           position["saving"])
        readings[tag] = reading
        need(f"reader left the tree unchanged ({tag})",
             D.fingerprint(r["root"], r["counts"], r["root_value"]) == fp[tag])
        need(f"reader drew from no stream ({tag})",
             (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate()) == streams)
    need("readings agree", readings["off"] == readings["on"])
    return {"arm": arm.label, "arm_name": arm.name, "seed": arm.seed, "fpu": arm.fpu,
            "budget": arm.budget, "position": position["id"], "expect": position["expect"],
            "move": list(off["move"]),
            "defends": D._t(off["move"]) in {D._t(s) for s in position["saving"]},
            "fingerprint": fp["off"], "checks": passed, "reading": readings["off"]}


# ─────────────────────────────── interpretation (§5.2-§5.3 with §9 A1.1) ──────────────────────

def interpret(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The frozen rule, from COMPLETE rows only. 'consistent with the H-FPU-SWEEP conjecture
    at E1 and E2' iff B-repro passed (all five rows present), B-seed shows a FULL sweep at
    E1 and E2 at -3 and -5, and all SIX cells match. Repair is per fixture and level, with
    the controls' status; nothing is ranked or selected."""
    at = {(r["arm"], r["position"]): r for r in rows}
    problems = []
    if not all(("B-repro", p) in at for p in D.POSITION_IDS):
        problems.append("B-repro incomplete")
    for s in SEED_ARMS:
        for p in ("E1", "E2"):
            r = at.get((Arm("B-seed", s).label, p))
            if r is None or not r["reading"]["sweep"]["full_sweep"]:
                problems.append(f"B-seed {s} {p}: no full sweep")
    cells = {}
    for f, want in PREDICTED_SWEEP.items():
        for p, persists in want.items():
            r = at.get((Arm("F", BASE_SEED, fpu=f).label, p))
            got = None if r is None else r["reading"]["sweep"]["full_sweep"]
            cells[f"fpu{f} {p}"] = {"predicted_persists": persists, "observed_persists": got,
                                    "match": got == persists}
            if got != persists:
                problems.append(f"cell fpu{f} {p}: predicted "
                                f"{'persists' if persists else 'breaks'}, observed "
                                f"{'missing' if got is None else ('persists' if got else 'breaks')}")
    repair = {}
    for arm in plan():
        here = {p: at.get((arm.label, p)) for p in D.POSITION_IDS}
        if any(v is None for v in here.values()):
            continue
        harm = [p for p in ("CD1", "CD2") if not here[p]["defends"]]
        repair[arm.label] = {
            "defends": {p: here[p]["defends"] for p in D.POSITION_IDS},
            "control_harm": harm,
            "repairs": [p for p in ("E1", "E2", "E3") if here[p]["defends"]],
            "note": "repair with control harm" if harm and any(
                here[p]["defends"] for p in ("E1", "E2", "E3")) else None}
    return {"verdict": "consistent with the H-FPU-SWEEP conjecture at E1 and E2"
            if not problems else "not supported as stated",
            "problems": problems, "cells": cells, "repair": repair,
            "note": "a heuristic conjecture (§9 A1.2); consistency at these fixtures under "
                    "fresh-agent search, not cause; no level is a candidate setting"}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    """A recorded failure: the step, the actual exception, and what it carried."""
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def _run(fixtures: Sequence[Mapping[str, Any]], config, identity: Mapping[str, Any],
         loader: Callable, d: pathlib.Path, *, arms: Optional[Sequence[Arm]] = None,
         clock: Callable[[], float] = time.monotonic,
         wall_s: float = HARD_LIMIT_S - MARGIN_S) -> Dict[str, Any]:
    """On a CLAIMED directory: one model load, then every arm at every fixture in the frozen
    order; the journal fsynced after each observation; `record.json` on EVERY exit."""
    from . import d1_probe as D1
    arms = list(plan() if arms is None else arms)
    if not _unconsumed(d):
        raise FactorRefused(f"{d} does not hold exactly an unconsumed claim")
    deadline = D1.Deadline(limit_s=wall_s, clock=clock).start()
    log, step, rows = None, "claim", []
    status, failure, interrupt = "COMPLETE", None, None

    def due(next_step: str) -> None:
        if deadline.elapsed() > wall_s:
            raise FactorStop(f"the {wall_s:.0f} s deadline passed before {next_step}")

    try:
        log = CC._Log(d / "log.jsonl")
        step = "model_load"
        with D1._supervisor(deadline):
            evaluator = loader()
            log.done("model_load", {"incumbent_identity": identity})
            for arm in arms:
                build = arm_builder(arm, config, evaluator, identity)
                for p in fixtures:
                    step = f"observe:{arm.label}:{p['id']}"
                    due(step)
                    row = observe(arm, build, config, p)
                    rows.append(row)
                    log.done("observation", row)
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    finally:
        if log is not None:
            log.close()
    #: the completion check and the interpretation sit inside their OWN recorded
    #: boundary: an exception here still leaves `record.json`, every row and the journal.
    interpretation = None
    if status == "COMPLETE":
        step = "completion_check"
        try:
            want = collections.Counter((a.label, p) for a in plan() for p in D.POSITION_IDS)
            got = collections.Counter((r["arm"], r["position"]) for r in rows)
            if got != want:
                status, failure = "INCOMPLETE", {
                    "failing_step": "completion_check", "exception_type": None,
                    "reason": "the observations are not the frozen plan",
                    "missing": sorted(list(k) for k in want - got),
                    "duplicated": sorted(list(k) for k, n in got.items() if n > 1),
                    "extra": sorted(list(k) for k in set(got) - set(want))}
            else:
                step = "interpretation"
                interpretation = interpret(rows)
        except (KeyboardInterrupt, SystemExit) as e:
            status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            status, failure = "STOP", _failure(d, step, e)
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "interpretable": status == "COMPLETE",
              "interpretation": interpretation,     # set only on the COMPLETE path
              "interpretation_note": None if status == "COMPLETE" else
              f"not interpreted: the run is {status}",
              "incumbent_identity": dict(identity), "frozen_set_sha256": D.FROZEN_SHA256,
              "pinned_fingerprints": dict(PINNED_FINGERPRINTS), "rows": rows,
              "deadline_s": wall_s, "elapsed_s": deadline.elapsed(),
              "note": "diagnostic fixtures under fresh-agent search; absolute-FPU levels are "
                      "probes, not settings; not a strength evaluation"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── the parent: claim, watchdog, settle ──────────────────────────

def _unconsumed(d: pathlib.Path) -> bool:
    if not d.is_dir():
        return False
    names = set(os.listdir(d))
    return CLAIM_FILE in names and names <= {CLAIM_FILE, *CHILD_STREAMS.values()}


def _claim(d: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    import secrets
    try:
        os.mkdir(d)                              # THE CLAIM: create-only
    except FileExistsError:
        raise FactorRefused(f"{d} is occupied: the study is one-shot") from None
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
        raise FactorRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise FactorRefused("the claim's nonce does not match: this process was not "
                            "started by the parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
    """The parent's own record on EVERY exit; the child's files are left as written; a
    killed run is TERMINATED, never COMPLETE."""
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
    if not AUTH.H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED:
        raise FactorUnauthorized("the H5 incumbent defence factor study is UNAUTHORIZED. "
                                 "Nothing was opened, read, built, loaded, decided or written.")


def out_dir() -> pathlib.Path:
    """§9 A1.6: the frozen output directory -- there is no other."""
    return pathlib.Path(R4.EVIDENCE_ROOT) / OUT_NAME


def _child_argv() -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed"]


def run() -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals, THE CLAIM, then the child in its own
    session under the hard watchdog; `watchdog.json` on every exit. No seam is accepted."""
    _gate()
    out = out_dir()
    if os.path.lexists(out):
        raise FactorRefused(f"{out} is occupied: the study is one-shot")
    D.positions(D._frozen())
    D._identity(R4.frozen_argmax_config())
    claim, nonce = _claim(out)
    watch = CC._watch(_child_argv(),
                      limit_s=HARD_LIMIT_S - (time.monotonic() - claim["claimed_monotonic"]),
                      env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                      cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                      stderr_path=out / CHILD_STREAMS["stderr"])
    summary = _settle(out, watch)
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def run_claimed() -> Dict[str, Any]:
    """The CHILD, started only by `run` with the claim's nonce: the gate and the pure
    checks again, the claim, then the model load and the frozen plan."""
    _gate()
    out = out_dir()
    fixtures = D.positions(D._frozen())
    config = R4.frozen_argmax_config()
    identity = D._identity(config)
    claim = _accept_claim(out)
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(fixtures, config, identity, lambda: R4.load_production_evaluator(identity), out,
                wall_s=HARD_LIMIT_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_incumbent_defence_factors",
                                 description="H5 incumbent defence factor study -- "
                                             "EXPLORATORY. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("run", "run-claimed"))
    a = ap.parse_args(argv)
    if not AUTH.H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED:
        print("the H5 incumbent defence factor study is UNAUTHORIZED. Nothing was opened, "
              "read, built, loaded, decided or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "run":
            res = run()
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed()["status"]}))
    except (FactorRefused, FactorUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
