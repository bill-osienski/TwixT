"""THE FRESH TACTICAL QUALIFICATION CORPUS: generator and freeze.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §4, §14 A4, §15 A5.

Rules-engine-only random legal playouts on the official 24x24 board in two frozen streams
(uniform; pegs-biased). From games that end in a win, every position at plies
max(10, end-20) .. min(end-1, 270) is classified under the protocol (`classify`, imported
unchanged). A complete prior-position inventory is excluded by digest. One seeded representative
per (class, game), at most 20 games per class; shortages recorded; repeats decided once. Every
selected position is cross-checked by the exhaustive engine-only scan: a disagreement STOPS.

PUBLICATION (§15 A5.2). The child writes `corpus.provisional.json` only. The PARENT publishes
`corpus.json` -- the provisional bytes, create-only -- after a normal exit, a COMPLETE record and its
own validation; a completed payload followed by termination is never published. One-shot
destination, a 3 h limit enforced by the parent, the child's alarm 60 s earlier, every completed
game durable before the next, record.json on every exit, no retry.
Gate: `tactical_candidate_authorization`, created closed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import platform
import random
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from . import candidate_match_runner as CM
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_tactics as T
from . import tactical_candidate as TC
from . import tactical_candidate_authorization as AUTH

MODULE = "scripts.GPU.alphazero.tactical_corpus"
CARD = "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"
LABEL = "TACTICAL CANDIDATE QUALIFICATION CORPUS"
FROZEN_SPEC = {                            # §14 A4.1-A4.5, §15 A5 -- never a parameter of a public entry
    "streams": ["tq24-uniform", "tq24-pegs"], "seeds": [0, 4000], "p_near": 0.7,
    "knight": [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]],
    "window": 20, "min_ply": 10, "max_ply": 270, "quota": 20, "select": "tq24-select",
    "cap": 280, "board": 24, "python": "3.14.7"}
H5_CORPUS_SHA256 = "214b958c016e90a0bfc051b505eb2885f13e10efd7970e8b3fcb20ae3d892b2a"
H5_RECORD_SHA256 = "189e6add3beab8a571ec52cfc5a62956711a8132c93f9ecd2f945353d7553140"
PRIOR_FILE = pathlib.Path(__file__).with_name("tactical_corpus_prior.json")
PRIOR_SHA256 = "dc344b079823bf200d77970ccc1e980e561f7cf7af618edf3b28c1bc12bf8bd4"
HARD_LIMIT_S, MARGIN_S = 3 * 3600.0, 60.0
OUT_NAME = "tactical-candidate-corpus"
PROVISIONAL, CORPUS, INVENTORY, CANDIDATES = ("corpus.provisional.json", "corpus.json",
                                              "inventory.json", "candidates.jsonl")
CLAIM_FILE, WATCHDOG_FILE, CLAIM_ENV = "claim.json", "watchdog.json", "TACTICAL_CORPUS_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class CorpusRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class CorpusUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, generated or written."""


class CorpusStop(Exception):
    """A check failed during the freeze: recorded, never retried."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def _write_file(path: pathlib.Path, raw: bytes) -> str:
    """Create-only; every byte or a STOP (CM._write_all); fsynced, directory too."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        CM._write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    CM._fsync_dir(path.parent)
    return hashlib.sha256(raw).hexdigest()


# ─────────────────────────────── generation (§14 A4.1-A4.3) ───────────────────────────────

def _move(stream: str, rng: random.Random, state, spec: Mapping[str, Any]) -> Tuple[int, int]:
    """§14 A4.2, exact draw order: uniform -> choice(legal); pegs -> a u is drawn ONLY when some
    legal move is a knight's step from one of the mover's pegs."""
    legal = sorted(state.legal_moves())
    if stream == "tq24-uniform":
        return rng.choice(legal)
    if stream != "tq24-pegs":
        raise CorpusRefused(f"no move policy for the stream {stream!r}")
    own = {(r + a, c + b) for (r, c), col in state.pegs.items() if col == state.to_move
           for a, b in spec["knight"]}
    near = [m for m in legal if m in own]
    if near:
        return rng.choice(near) if rng.random() < spec["p_near"] else rng.choice(legal)
    return rng.choice(legal)


def play(stream: str, seed: int, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """One game from the empty official board until the first winner, the cap, or no legal move."""
    if stream not in spec["streams"]:
        raise CorpusRefused(f"unknown stream {stream!r}")
    rng = random.Random(f"{stream}:{seed}")
    state, moves = T.replay([]), []
    while state.winner() is None and state.ply < spec["cap"] and state.legal_moves():
        m = _move(stream, rng, state, spec)
        moves.append(list(m))
        state = state.apply_move(m)
    reason = ("win" if state.winner() is not None else "cap" if state.ply >= spec["cap"]
              else "no_legal_move")
    return {"stream": stream, "seed": seed, "moves": moves, "end": state.ply, "reason": reason}


def window(game: Mapping[str, Any], spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """§14 A4.3: only a WON game; plies max(10, end-20) .. min(end-1, 270), protocol classes."""
    if game["reason"] != "win":
        return []
    lo = max(spec["min_ply"], game["end"] - spec["window"])
    hi = min(game["end"] - 1, spec["max_ply"])
    out, s = [], T.replay([])
    for i, m in enumerate(game["moves"]):
        if i > hi:
            break
        if i >= lo:
            c = T.classify(s, semantics="protocol")
            if c is not None:
                out.append({"ply": i, "class": c["class"], "digest": T.state_digest(s)})
        s = s.apply_move(_t(m))
    return out


# ─────────────────────────────── the prior-position inventory (§14 A4.4, §15 A5.1) ───────────

def _prefixes(moves: Sequence) -> List[str]:
    s, out = T.replay([]), []
    out.append(T.state_digest(s))
    for m in moves:
        s = s.apply_move(_t(m))
        out.append(T.state_digest(s))
    return out


def neighbourhood(base: Sequence, spec: Mapping[str, Any]) -> Set[bytes]:
    """§15 A5.1 (f): every one- and two-ply extension of `base` whose ply lies in the eligible
    window. No reply follows a win."""
    s, out = T.replay(base), set()
    if s.winner() is not None:
        return out
    for m in s.legal_moves():
        c = s.apply_move(m)
        if spec["min_ply"] <= c.ply <= spec["max_ply"]:
            out.add(bytes.fromhex(T.state_digest(c)))
        if c.winner() is None and c.ply + 1 <= spec["max_ply"]:
            for r in c.legal_moves():
                g = c.apply_move(r)
                if g.ply >= spec["min_ply"]:
                    out.add(bytes.fromhex(T.state_digest(g)))
    return out


def build_inventory(sources: Mapping[str, Any], spec: Mapping[str, Any]
                    ) -> Tuple[Set[bytes], Dict[str, Any]]:
    """Sources (a)-(e) digest by digest with their sources; (f) by base, rule, count and hash."""
    listed: Dict[str, Set[str]] = collections.defaultdict(set)
    counts = collections.Counter()

    def add(d: str, src: str) -> None:
        listed[d].add(src)
        counts[src[0]] += 1
    for name, d in sources["fixtures"].items():
        add(d, f"a:{name}")
    for key in ("selected", "fixture_equivalent"):
        for e in sources["h5_corpus"][key]:
            add(e["state_digest"], f"b:{e['key']}")
    for r in sources["h5_record"]["rows"]:
        add(T.state_digest(T.replay(r["moves"])), f"c:{r['id']}")
    prior = sources["prior"]
    for tag, group in (("d", prior["pinned"]), ("e", prior["standin"])):
        for name, moves in sorted(group.items()):
            for i, d in enumerate(_prefixes(moves)):
                add(d, f"{tag}:{name}[:{i}]")
    near: Set[bytes] = set()
    per_base = {}
    for b in prior["bases"]:
        x = neighbourhood(prior["pinned"][b["pinned"]][:b["plies"]], spec)
        per_base[f"{b['pinned']}[:{b['plies']}]"] = len(x)
        near |= x
    digests = {bytes.fromhex(d) for d in listed} | near
    summary = {"listed": {d: sorted(s) for d, s in sorted(listed.items())},
               "counts": {k: counts[k] for k in "abcde"},
               "neighbourhood": {"rule": "every one- and two-ply extension of each base, kept where "
                                         "its ply lies in the eligible window; no reply after a win",
                                 "bases": per_base, "count": len(near),
                                 "sorted_sha256": hashlib.sha256(b"".join(sorted(near))).hexdigest()},
               "total_distinct": len(digests)}
    return digests, summary


def _sources() -> Dict[str, Any]:
    """The production sources, each bound to its pinned hash -- refused otherwise."""
    def pinned(path: pathlib.Path, sha: str, what: str) -> Any:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise CorpusRefused(f"{what} is not the pinned file (sha256 differs)")
        return json.loads(raw)
    return {"fixtures": T.fixture_digests(D._frozen()),
            "h5_corpus": pinned(T.corpus_path(), H5_CORPUS_SHA256, "the H5 tactics corpus"),
            "h5_record": pinned(T.out_dir() / "record.json", H5_RECORD_SHA256, "the H5 tactics record"),
            "prior": pinned(PRIOR_FILE, PRIOR_SHA256, "the prior-position file")}


# ─────────────────────────────── selection (§14 A4.5) ───────────────────────────────

def select(games: Sequence[Mapping[str, Any]], inventory: Set[bytes], spec: Mapping[str, Any]
           ) -> Dict[str, Any]:
    """Prior positions out first; then one seeded representative per (class, game) and a seeded
    sample of at most `quota` games per class; repeats decided once."""
    excluded = collections.Counter()
    by_class: Dict[str, Dict[Tuple[str, int], List[Mapping[str, Any]]]] = \
        {c: collections.defaultdict(list) for c in T.CLASSES}
    for g in games:
        for p in g["positions"]:
            if bytes.fromhex(p["digest"]) in inventory:
                excluded[p["class"]] += 1
            else:
                by_class[p["class"]][(g["stream"], g["seed"])].append(p)
    selected, shortfall, universe = [], {}, {}
    for cls in T.CLASSES:
        found = by_class[cls]
        universe[cls] = {"positions": sum(len(v) for v in found.values()), "games": len(found)}
        reps = {}
        for (stream, seed), ps in found.items():
            ply = random.Random(f"{spec['select']}:{cls}:{stream}:{seed}").choice(
                sorted(p["ply"] for p in ps))
            reps[(stream, seed)] = next(p for p in ps if p["ply"] == ply)
        keys = sorted(found)
        chosen = random.Random(f"{spec['select']}:{cls}").sample(keys, min(spec["quota"], len(keys)))
        shortfall[cls] = max(0, spec["quota"] - len(keys))
        for k in sorted(chosen):
            selected.append({"class": cls, "stream": k[0], "seed": k[1], "ply": reps[k]["ply"],
                             "state_digest": reps[k]["digest"]})
    first: Dict[str, str] = {}
    repeats: Dict[str, List[str]] = collections.defaultdict(list)
    for e in selected:
        key = f"{e['stream']}:{e['seed']}@{e['ply']}"
        e["repeat_of"] = first.get(e["state_digest"])
        if e["repeat_of"] is None:
            first[e["state_digest"]] = key
        else:
            repeats[e["state_digest"]].append(key)
    return {"selected": selected, "quota_shortfall": shortfall, "universe": universe,
            "excluded_prior": {c: excluded[c] for c in T.CLASSES},
            "repeats": {d: [first[d]] + ks for d, ks in sorted(repeats.items())}}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def _code_shas() -> Dict[str, str]:
    return {pathlib.Path(m.__file__).name: hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()
            for m in (sys.modules[__name__], T, TC)}


def _run(spec: Mapping[str, Any], sources: Mapping[str, Any], d: pathlib.Path, *,
         clock: Callable[[], float] = time.monotonic,
         wall_s: float = HARD_LIMIT_S - MARGIN_S) -> Dict[str, Any]:
    """The freeze. `spec` and `sources` are TEST seams; the public entry passes the frozen ones."""
    if spec["python"] != platform.python_version():
        raise CorpusRefused(f"the interpreter is {platform.python_version()}, not {spec['python']}")
    if not _unconsumed(d):
        raise CorpusRefused(f"{d} does not hold exactly an unconsumed claim")
    t0 = clock()
    left = lambda: wall_s - (clock() - t0)    # noqa: E731
    expected = len(spec["streams"]) * (spec["seeds"][1] - spec["seeds"][0])
    log, partial, step = None, None, "claim"
    status, failure, interrupt = "COMPLETE", None, None
    games: List[Dict[str, Any]] = []
    inv_sha, prov_sha, counts = None, None, None
    try:
        log = CC._Log(d / "log.jsonl")
        with CM._alarm(left(), "the corpus freeze"):
            step = "inventory"
            inventory, summary = build_inventory(sources, spec)
            inv_sha = _write_file(d / INVENTORY, _canonical(summary))
            counts = {**summary["counts"], "f": summary["neighbourhood"]["count"],
                      "total_distinct": summary["total_distinct"]}
            log.done("inventory", counts)
            partial = CM._Partial(d / CANDIDATES)
            for stream in spec["streams"]:
                for seed in range(*spec["seeds"]):
                    step = f"generate:{stream}:{seed}"
                    if left() <= 0:
                        raise CorpusStop(f"the {wall_s:.0f} s deadline passed before {step}")
                    g = play(stream, seed, spec)
                    row = {"stream": stream, "seed": seed, "end": g["end"], "reason": g["reason"],
                           "positions": window(g, spec)}
                    partial(row)                 # durable before the next game
                    games.append(row)
                    log.done("game", {"stream": stream, "seed": seed, "window": len(row["positions"])})
            partial.close()
            partial = None
            step = "completion_check"
            if len(games) != expected:
                raise CorpusStop(f"{len(games)} games generated, {expected} planned")
            step = "select"
            sel = select(games, inventory, spec)
            step = "replay"
            for e in sel["selected"]:
                moves = play(e["stream"], e["seed"], spec)["moves"][:e["ply"]]
                s = T.replay(moves)
                if T.state_digest(s) != e["state_digest"]:
                    raise CorpusStop(f"{e['stream']}:{e['seed']}@{e['ply']} does not replay to its "
                                     "digest: the generator is not reproducible")
                e["moves"], e["class_dict"] = moves, T.classify(s, semantics="protocol")
            step = "cross_check"
            checked: Dict[str, Dict[str, Any]] = {}
            for e in sel["selected"]:
                if e["state_digest"] not in checked:
                    r = TC.cross_check(T.replay(e["moves"]))
                    if not r["agree"]:
                        raise CorpusStop(f"{e['stream']}:{e['seed']}@{e['ply']}: the optimized "
                                         "detector and the exhaustive scan DISAGREE")
                    checked[e["state_digest"]] = {"agree": True, "replies_examined": r["replies_examined"]}
                e["cross_check"] = checked[e["state_digest"]]
            log.done("cross_check", {"positions": len(checked), "agree": True})
            step = "provisional"
            payload = {"label": LABEL, "card": CARD, "spec": dict(spec),
                       "python": platform.python_version(), "code": _code_shas(),
                       "inventory_sha256": inv_sha, "inventory_counts": counts,
                       "games_generated": len(games),
                       "games_won": sum(g["reason"] == "win" for g in games), **sel}
            prov_sha = _write_file(d / PROVISIONAL, _canonical(payload))
            log.done("provisional", {"sha256": prov_sha})
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    finally:
        if partial is not None:
            partial.close()
        if log is not None:
            log.close()
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "spec_sha256": _sha(dict(spec)), "python": platform.python_version(),
              "games_processed": len(games), "games_planned": expected,
              "inventory_sha256": inv_sha, "provisional_sha256": prov_sha,
              "deadline_s": wall_s, "elapsed_s": clock() - t0,
              "note": "the child never writes corpus.json; only the parent publishes it (§15 A5.2)"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── the parent: validation and publication (§15 A5.2) ───────────

def validate_provisional(d: pathlib.Path, rec: Mapping[str, Any], spec: Mapping[str, Any]) -> bytes:
    """The parent's own checks of the child's payload; returns its bytes, or raises CorpusStop."""
    p = d / PROVISIONAL
    if not p.exists():
        raise CorpusStop("no provisional payload")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != rec.get("provisional_sha256"):
        raise CorpusStop("the provisional payload is not the one the record hashed")
    c = json.loads(raw)
    if c.get("spec") != dict(spec) or c.get("python") != spec["python"]:
        raise CorpusStop("the payload is not of the frozen specification and interpreter")
    if hashlib.sha256((d / INVENTORY).read_bytes()).hexdigest() != c.get("inventory_sha256"):
        raise CorpusStop("inventory.json is not the inventory the payload recorded")
    per = collections.Counter(e["class"] for e in c["selected"])
    if any(per[k] > spec["quota"] for k in per) or len(c["selected"]) > spec["quota"] * len(T.CLASSES):
        raise CorpusStop(f"the quotas are exceeded: {dict(per)}")
    for e in c["selected"]:
        s = T.replay(e["moves"])
        if (T.state_digest(s), len(e["moves"])) != (e["state_digest"], e["ply"]) or \
                not spec["min_ply"] <= e["ply"] <= spec["max_ply"]:
            raise CorpusStop(f"{e['stream']}:{e['seed']}@{e['ply']} does not replay to its digest "
                             "at an eligible ply")
        cd = T.classify(s, semantics="protocol")
        if cd != e["class_dict"] or cd["class"] != e["class"]:
            raise CorpusStop(f"{e['stream']}:{e['seed']}@{e['ply']} is not of its recorded class")
        if (e.get("cross_check") or {}).get("agree") is not True:
            raise CorpusStop(f"{e['stream']}:{e['seed']}@{e['ply']} has no agreeing cross-check")
    return raw


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
        raise CorpusRefused(f"{d} is occupied: the freeze is one-shot") from None
    nonce = secrets.token_hex(16)
    claim = {"claimed_monotonic": time.monotonic(),
             "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_pid": os.getpid(), "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
             **extra}
    with open(d / CLAIM_FILE, "x", encoding="utf-8") as fh:
        json.dump(claim, fh, indent=1, sort_keys=True)
    return claim, nonce


def _accept_claim(d: pathlib.Path) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise CorpusRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise CorpusRefused("the claim's nonce does not match: this process was not started by "
                            "the parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any], *,
            spec: Mapping[str, Any] = FROZEN_SPEC) -> Dict[str, Any]:
    """watchdog.json -- and the ONLY publication of corpus.json, after a normal exit, a COMPLETE
    record and `validate_provisional`."""
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
    published, why = None, None
    if status == "COMPLETE" and watch["rc"] == 0:
        try:
            raw = validate_provisional(d, rec, spec)
            published = {"file": CORPUS, "corpus_sha256": _write_file(d / CORPUS, raw)}
        except (CorpusStop, CM.MatchStop, OSError, ValueError, KeyError, TypeError) as e:
            status, why = "NOT_PUBLISHED", f"{type(e).__name__}: {e}"
    else:
        why = f"the child's run is {status}"
    rc, streams = watch["rc"], {}
    for s, name in sorted(CHILD_STREAMS.items()):
        raw_s = (d / name).read_bytes() if (d / name).exists() else None
        streams[s] = None if raw_s is None else {
            "file": name, "bytes": len(raw_s), "sha256": hashlib.sha256(raw_s).hexdigest()}
    summary = {"label": LABEL, "card": CARD, "status": status, "cause": watch["cause"],
               "limit_s": watch["limit_s"], "elapsed_s": watch["elapsed_s"],
               "child_pid": watch["child_pid"], "child_rc": rc,
               "child_exit": "never started" if watch["child_pid"] is None else
               f"killed by signal {CC._signal_name(-rc)}" if rc < 0 else f"exit status {rc}",
               "child_streams": streams, "signal": watch["signal"],
               "survivors_killed": watch["survivors_killed"], "journal": CC._journal(d),
               "record_present": present, "record_readable": rec is not None,
               "record_status": None if rec is None else rec.get("status"),
               "record_sha256": CM._file_sha(str(d / "record.json")) if present else None,
               "provisional_sha256": CM._file_sha(str(d / PROVISIONAL)) if (d / PROVISIONAL).exists()
               else None, "published": published, "not_published": why}
    with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


def committed_corpus() -> Tuple[Dict[str, Any], str]:
    """§15 A5.2, for the later qualification: the PUBLISHED corpus, committed, bound to a COMPLETE
    parent summary and a COMPLETE child record by matching hashes."""
    d = out_dir()
    try:
        F.check_durable([str(d / x) for x in (CORPUS, PROVISIONAL, "record.json", WATCHDOG_FILE)],
                        _fixture=False)
    except F.H4AnalysisRefused as e:
        raise CorpusRefused(f"the corpus is not committed: {e}") from None
    raw = (d / CORPUS).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    w = json.loads((d / WATCHDOG_FILE).read_text(encoding="utf-8"))
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    if w.get("status") != "COMPLETE" or (w.get("published") or {}).get("corpus_sha256") != sha:
        raise CorpusRefused("the parent did not publish THIS corpus as COMPLETE")
    if rec.get("status") != "COMPLETE" or rec.get("provisional_sha256") != sha or \
            CM._file_sha(str(d / PROVISIONAL)) != sha:
        raise CorpusRefused("the child's COMPLETE record does not hash to the published corpus")
    return json.loads(raw), sha


# ─────────────────────────────── the public entries ───────────────────────────────

def _gate() -> None:
    if not AUTH.TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED:
        raise CorpusUnauthorized("the corpus freeze is UNAUTHORIZED. Nothing was opened, read, "
                                 "generated or written.")


def out_dir() -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / OUT_NAME


def _child_argv() -> List[str]:
    return [sys.executable, "-m", MODULE, "freeze-claimed"]


def freeze() -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals (interpreter, pinned sources), THE CLAIM, the
    child under the parent's hard limit, then validation and publication."""
    _gate()
    out = out_dir()
    if os.path.lexists(out):
        raise CorpusRefused(f"{out} is occupied: the freeze is one-shot")
    if platform.python_version() != FROZEN_SPEC["python"]:
        raise CorpusRefused(f"the interpreter is {platform.python_version()}, not "
                            f"{FROZEN_SPEC['python']}")
    _sources()
    claim, nonce = _claim(out, {"spec_sha256": _sha(FROZEN_SPEC)})
    watch = CC._watch(_child_argv(),
                      limit_s=HARD_LIMIT_S - (time.monotonic() - claim["claimed_monotonic"]),
                      env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                      cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                      stderr_path=out / CHILD_STREAMS["stderr"])
    summary = _settle(out, watch)
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def freeze_claimed() -> Dict[str, Any]:
    """The CHILD: the gate, the claim (nonce, the frozen spec), the pinned sources; then the run."""
    _gate()
    out = out_dir()
    claim = _accept_claim(out)
    if claim.get("spec_sha256") != _sha(FROZEN_SPEC):
        raise CorpusRefused("the claim is for another specification")
    sources = _sources()
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(FROZEN_SPEC, sources, out, wall_s=HARD_LIMIT_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="tactical_corpus",
                                 description="Tactical candidate corpus freeze. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("freeze", "freeze-claimed"))
    a = ap.parse_args(argv)
    if not AUTH.TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED:
        print("the corpus freeze is UNAUTHORIZED. Nothing was opened, read, generated or written.",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "freeze":
            res = freeze()
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": freeze_claimed()["status"]}))
    except (CorpusRefused, CorpusUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
