"""H5 POST-HOC WEAKNESS REVIEW -- EXPLORATORY. Read-only.

Frozen by `docs/superpowers/2026-10-01-t1j-h5-weakness-review-card.md` (with §7,
clarifications C1-C3). 🔴 THE CARD IS THE AUTHORITY. Nothing here is a confirmatory
claim, amends H5's result, pools with H4, or attributes anything to policy, value or
search. No model, JVM, T1j query, game or seed; no gate (none is needed).

Inputs: the four committed H5 segment results, the study manifest and the H5
estimate -- each hash-pinned and checked durable -- re-derived by H4's reader
(`h4_runner.load_games`), bound to the manifest's schedules, and cross-checked against
the committed estimate's aggregates. Any failure is a `ReviewRefused`: nothing is
analysed.

Tactical labels (card §7 C1), per pre-move position P, mover X, opponent Y:
  W = X's immediate wins; T = Y's immediate wins were Y to move (BOARD threats).
  MISSED_WIN      W non-empty, played move not in W
  MISSED_DEFENCE  W empty, T non-empty, the move leaves Y a win, and some move leaves none
  UNDEFENDABLE    W empty, T non-empty, and no move leaves Y without a win (not an error)
  DEFENDED        W empty, T non-empty, the move leaves Y none (context only)
An immediate win counts as saving; nothing after a win is examined. Exact reduction:
Y's wins after any X move are a subset of T (verified in tests, asserted per move).
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import d0_postmortem as D0
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_runner as H5
from .game.twixt_state import TwixtState

LABEL = "EXPLORATORY"
ROOT = R4.REPO_ROOT
MANIFEST = "docs/superpowers/evidence/t1j-h5-manifests/study_manifest.json"
ESTIMATE = "docs/superpowers/evidence/t1j-h5-confirmatory/estimate.json"
#: Card §0: every input hash-pinned, checked BEFORE it is parsed.
PINS = {
    MANIFEST: "e08c5f352eaabc9fac2fa73c48a432abbbdc23caf8ec5c4c52735de8bb619d32",
    ESTIMATE: "73600cf26875abb39b223093642a841c2e90b1c4ce5733470127db346e66edb4",
    f"{H5.EVIDENCE_DIRS[0]}/results.jsonl":
        "e9c54d5f430c4a01bfd181ba53c3818dce8d79a5b05700e8d8a3f8121de74cef",
    f"{H5.EVIDENCE_DIRS[1]}/results.jsonl":
        "7e1ca0b9f8a84b3749ecfd365016ec7c27298633dff6d8fb657b6f79ed6d768a",
    f"{H5.EVIDENCE_DIRS[2]}/results.jsonl":
        "d13a3bd8ec700def08e7cfef948cb3b89d0e6655ca2f1de27a606e8485929ad7",
    f"{H5.EVIDENCE_DIRS[3]}/results.jsonl":
        "4ff643415e997e8eaddc31766742739af81563468159b8aedb0ade8bef04a030",
}
PREFIX_DEPTHS = (2, 4, 6, 8, 10, 12)
LABELS = ("MISSED_WIN", "MISSED_DEFENCE", "UNDEFENDABLE")
ENGINES = ("incumbent", "t1j")
ARMS = ("A", "B")
STOP_S = 7200.0


class ReviewRefused(Exception):
    """An input failed integrity: nothing was analysed."""


# ─────────────────────────────── 1. inputs, integrity ───────────────────────────────

def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_of(winner: Optional[str], terminal_reason: str, incumbent_colour: str) -> str:
    """The INCUMBENT's outcome: a cap is UNRESOLVED -- never a loss, never a win."""
    if terminal_reason == "cap":
        return "cap"
    if terminal_reason != "win" or winner not in ("red", "black"):
        raise ReviewRefused(f"unknown terminal {terminal_reason!r} / winner {winner!r}")
    return "win" if winner == incumbent_colour else "loss"


def load_and_verify(root: pathlib.Path = ROOT) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """(games, input sha256s). Pins first, then durability, then H4's reader, the
    manifest binding, and the cross-check against the committed estimate."""
    for rel, want in PINS.items():
        if _sha(root / rel) != want:
            raise ReviewRefused(f"{rel} does not hash to its pin {want[:12]}...")
    try:
        F.check_durable([str(root / rel) for rel in PINS], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise ReviewRefused(f"not durable: {e}") from None
    manifest = json.loads((root / MANIFEST).read_text())
    games = []
    for k in H5.SEGMENTS:
        path = root / H5.EVIDENCE_DIRS[k] / "results.jsonl"
        try:
            loaded = R4.load_games(str(path))
            F.bind_games(loaded, manifest["segments"][k]["schedule"])
        except (R4.H4RunError, F.H4AnalysisRefused) as e:
            raise ReviewRefused(f"segment {k}: {e}") from None
        for g in loaded:
            s, res = g["start"], g["result"]
            games.append({
                "segment": k, "task_id": g["task_id"], "pair_id": s["pair_id"],
                "arm": s["arm"], "seed": s["seed"], "incumbent_colour": s["incumbent_colour"],
                "moves": [tuple(p["move"]) for p in g["plies"]],
                "actors": [p["actor"] for p in g["plies"]],
                "movers": [p["mover"] for p in g["plies"]],
                "winner": res["winner"], "terminal_reason": res["terminal_reason"],
                "outcome": outcome_of(res["winner"], res["terminal_reason"],
                                      s["incumbent_colour"]),
                "replay": f"{H5.EVIDENCE_DIRS[k]}/replays/{g['task_id']}.json"})
    cross_check(games, json.loads((root / ESTIMATE).read_text()))
    return games, {rel: PINS[rel] for rel in PINS}


def cross_check(games: Sequence[Dict[str, Any]], estimate: Dict[str, Any]) -> Dict[str, Any]:
    """Card §4 Step 1: the per-arm totals must reproduce the committed estimate EXACTLY."""
    patterns = estimate["concentration"]["outcome_patterns"]
    want = {"A": collections.Counter(), "B": collections.Counter()}
    for pat, n in patterns.items():
        a, b = pat.split("|")
        want["A"][a] += n
        want["B"][b] += n
    got = {"A": collections.Counter(), "B": collections.Counter()}
    for g in games:
        got[g["arm"]]["cap" if g["terminal_reason"] == "cap" else g["winner"]] += 1
    by_pair = collections.defaultdict(list)
    for g in games:
        by_pair[g["pair_id"]].append(g)
    cap_pairs = sum(any(x["terminal_reason"] == "cap" for x in v) for v in by_pair.values())
    unique = {arm: len({tuple(g["moves"]) for g in games if g["arm"] == arm}) for arm in ARMS}
    checks = {
        "per_arm_outcomes": {a: dict(got[a]) for a in ARMS} == {a: dict(want[a]) for a in ARMS},
        "cap_games": sum(got[a]["cap"] for a in ARMS) == estimate["caps"]["cap_games"],
        "cap_affected_pairs": cap_pairs == estimate["caps"]["cap_affected_pairs"],
        "unique_games_by_arm": unique == {a: estimate["concentration"]["by_arm"][a]["unique_games"]
                                          for a in ARMS},
        "games": len(games) == 592,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ReviewRefused(f"the games do not reproduce the committed estimate: {bad}")
    return checks


# ─────────────────────────────── 2. trajectories, inventory ───────────────────────────────

def trajectory_digest(arm: str, moves: Sequence[Sequence[int]]) -> str:
    """Card §7 C3: sha256 of the canonical JSON {"arm", "moves"}."""
    body = json.dumps({"arm": arm, "moves": [list(m) for m in moves]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def distinct_trajectories(games: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keyed by (arm, moves); every game kept as a member, so frequency is exact."""
    out: Dict[str, Dict[str, Any]] = {}
    for g in games:
        d = trajectory_digest(g["arm"], g["moves"])
        t = out.setdefault(d, {"digest": d, "arm": g["arm"], "moves": list(g["moves"]),
                               "actors": g["actors"], "movers": g["movers"],
                               "outcome": g["outcome"], "winner": g["winner"],
                               "terminal_reason": g["terminal_reason"], "members": []})
        if (t["outcome"], t["actors"], t["movers"]) != (g["outcome"], g["actors"], g["movers"]):
            raise ReviewRefused(f"{g['task_id']}: the same moves with a different outcome or "
                                f"actors")
        t["members"].append({"task_id": g["task_id"], "replay": g["replay"]})
    for t in out.values():
        t["frequency"] = len(t["members"])
    return out


def inventory(trajs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    inv = {"label": LABEL}
    for arm in ARMS:
        ts = [t for t in trajs.values() if t["arm"] == arm]
        lengths = sorted(len(t["moves"]) for t in ts for _ in range(t["frequency"]))
        inv[arm] = {
            "games": sum(t["frequency"] for t in ts), "distinct_trajectories": len(ts),
            "outcomes_occurrence_weighted": dict(collections.Counter(
                t["outcome"] for t in ts for _ in range(t["frequency"]))),
            "outcomes_distinct": dict(collections.Counter(t["outcome"] for t in ts)),
            "terminal_reasons": dict(collections.Counter(
                t["terminal_reason"] for t in ts for _ in range(t["frequency"]))),
            "length_occurrence_weighted": {"min": lengths[0], "median": statistics.median(lengths),
                                           "max": lengths[-1]},
            "frequencies": sorted((t["frequency"] for t in ts), reverse=True),
        }
    return inv


# ─────────────────────────────── 3. Q1, prefix tables ───────────────────────────────

def prefix_tables(trajs: Dict[str, Dict[str, Any]],
                  depths: Sequence[int] = PREFIX_DEPTHS) -> Dict[str, Any]:
    """Per arm and depth: every prefix with (games, wins, losses, caps), occurrence-
    weighted AND distinct; a prefix RECURS IN LOSSES at >= 2 distinct loss trajectories."""
    out: Dict[str, Any] = {"label": LABEL}
    for arm in ARMS:
        out[arm] = {}
        for d in depths:
            rows: Dict[str, Dict[str, Any]] = {}
            for t in trajs.values():
                if t["arm"] != arm or len(t["moves"]) < d:
                    continue
                key = " ".join(f"{r},{c}" for r, c in t["moves"][:d])
                row = rows.setdefault(key, {
                    "prefix": key, "by": [t["actors"][i] for i in range(d)],
                    "occurrence": {"games": 0, "win": 0, "loss": 0, "cap": 0},
                    "distinct": {"games": 0, "win": 0, "loss": 0, "cap": 0}})
                for view, n in (("occurrence", t["frequency"]), ("distinct", 1)):
                    row[view]["games"] += n
                    row[view][t["outcome"]] += n
            for row in rows.values():
                row["recurs_in_losses"] = row["distinct"]["loss"] >= 2
            out[arm][str(d)] = sorted(rows.values(), key=lambda r: (-r["occurrence"]["loss"],
                                                                   r["prefix"]))
    return out


# ─────────────────────────────── 4. Q2, the tactical labels ───────────────────────────────

def label_ply(W: Sequence, T: Sequence, played, threats_left: Optional[Sequence],
              saving_moves: Callable[[], List]) -> Tuple[Optional[str], Optional[List]]:
    """Card §7 C1, exclusive in precedence. `saving_moves` is called ONLY when the
    mover has no immediate win and the played move left the opponent a win."""
    if W:
        return (None, None) if tuple(played) in {tuple(w) for w in W} else ("MISSED_WIN", None)
    if not T:
        return None, None
    if not threats_left:
        return "DEFENDED", None
    saving = saving_moves()
    return ("MISSED_DEFENCE", saving) if saving else ("UNDEFENDABLE", [])


def saving_moves_of(state: TwixtState, T: Sequence) -> List[Tuple[int, int]]:
    """Every legal mover move after which the opponent has NO immediate win. EXACT
    REDUCTION (card §7 C1): the opponent's wins after any mover move are a subset of T,
    so only T is tested. Called only when the mover has no immediate win."""
    opp = "black" if state.to_move == "red" else "red"
    out = []
    for m in state.legal_moves():
        after = state.apply_move(m)
        if not any(after.is_valid_placement(*t) and after.apply_move(t).winner() == opp
                   for t in T):
            out.append(tuple(m))
    return sorted(out)


def scan_trajectory(moves: Sequence, actors: Sequence[str], terminal_reason: str, *,
                    active_size: int = 24) -> List[Dict[str, Any]]:
    """Every pre-move position of one trajectory: W and T by FULL scan (D0's
    `one_ply_threats`), the move's left-over threats = the next ply's full W scan (or a
    full scan after a capped final move), the subset assertion, and the label."""
    states = [TwixtState(active_size=active_size, to_move="red")]
    for m in moves:
        states.append(states[-1].apply_move(tuple(m)))
    n = len(moves)
    Ws = [D0.one_ply_threats(states[i]) for i in range(n)]
    out = []
    for i in range(n):
        P, m = states[i], tuple(moves[i])
        W, T = Ws[i], D0.one_ply_threats(D0._flip(P))
        won = m in {tuple(w) for w in W}
        left: Optional[List] = None
        reply_available = None
        if not won:
            if i + 1 < n:
                left, reply_available = Ws[i + 1], True
            else:                          # the final move did not win: the game capped
                if terminal_reason != "cap":
                    raise ReviewRefused(f"ply {i + 1}: a non-winning final move of a "
                                        f"{terminal_reason!r} game")
                left, reply_available = D0.one_ply_threats(states[i + 1]), False
            if not {tuple(x) for x in left} <= {tuple(t) for t in T}:
                raise ReviewRefused(f"ply {i + 1}: the move left opponent wins outside T -- "
                                    f"the exact reduction does not hold")
        label, saving = label_ply(W, T, m, left, lambda: saving_moves_of(P, T))
        if label in ("MISSED_DEFENCE", "UNDEFENDABLE") and (m in set(saving or [])):
            raise ReviewRefused(f"ply {i + 1}: the played move is both saving and not")
        out.append({
            "ply": i + 1, "mover": P.to_move, "engine": actors[i], "move": list(m),
            "W": [list(w) for w in W], "T": [list(t) for t in T], "label": label,
            "threats_left": None if left is None else [list(x) for x in left],
            "reply_available": reply_available,
            "opponent_won_next_ply": bool(i + 1 < n and tuple(moves[i + 1]) in
                                          {tuple(w) for w in Ws[i + 1]}),
            "saving_moves": None if saving is None else [list(s) for s in saving],
            "distance_to_terminal": n - (i + 1)})
    return out


def label_rates(trajs: Dict[str, Dict[str, Any]],
                scans: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Card §7 C2: hits / inspected pre-move positions, by engine and arm, occurrence-
    weighted and distinct, numerator and denominator shown."""
    out: Dict[str, Any] = {"label": LABEL}
    for arm in ARMS:
        for eng in ENGINES:
            cell = {}
            for view in ("occurrence", "distinct"):
                den, nums = 0, collections.Counter()
                for d, rows in scans.items():
                    t = trajs[d]
                    if t["arm"] != arm:
                        continue
                    w = t["frequency"] if view == "occurrence" else 1
                    for r in rows:
                        if r["engine"] == eng:
                            den += w
                            if r["label"]:
                                nums[r["label"]] += w
                cell[view] = {lab: {"numerator": nums[lab], "denominator": den,
                                    "rate": nums[lab] / den if den else None}
                              for lab in (*LABELS, "DEFENDED")}
            out[f"{arm}/{eng}"] = cell
    return out


def fixed_distance_flags(scans: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """D0's fifth condition, prospectively: a label whose every hit sits at ONE distance
    from the terminal ply restates the ending."""
    out = {}
    for lab in LABELS:
        dists = sorted({r["distance_to_terminal"] for rows in scans.values() for r in rows
                        if r["label"] == lab})
        out[lab] = {"hit_distances": dists, "restates_the_ending": len(dists) == 1}
    return out


# ─────────────────────────────── 5. Q3, caps ───────────────────────────────

def final_features(moves: Sequence, active_size: int = 24) -> Dict[str, Any]:
    state = TwixtState(active_size=active_size, to_move="red")
    for m in moves:
        state = state.apply_move(tuple(m))
    f: Dict[str, Any] = {}
    for colour in ("red", "black"):
        comps = D0.component_boundary_distances(state, colour)
        f[colour] = {"pegs": sum(1 for o in state.pegs.values() if o == colour),
                     "bridges": sum(1 for (a, _b) in state.bridges if state.pegs[a] == colour),
                     "components": len(comps),
                     "largest_component": max((c["size"] for c in comps), default=0),
                     "min_goal1_distance": min((c["to_goal1"] for c in comps), default=None),
                     "min_goal2_distance": min((c["to_goal2"] for c in comps), default=None)}
    return f


def cap_review(trajs: Dict[str, Dict[str, Any]], tables: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": LABEL, "note": "capped games are UNRESOLVED, never losses"}
    for arm in ARMS:
        caps = [t for t in trajs.values() if t["arm"] == arm and t["outcome"] == "cap"]
        rows = []
        for t in sorted(caps, key=lambda t: (-t["frequency"], t["digest"])):
            rows.append({"digest": t["digest"], "frequency": t["frequency"],
                         "replays": [m["replay"] for m in t["members"]],
                         "final": final_features(t["moves"])})
        profiles = collections.Counter(json.dumps(r["final"], sort_keys=True) for r in rows)
        shared_prefixes = {d: [r["prefix"] for r in tables[arm][d] if r["distinct"]["cap"] >= 2]
                           for d in tables[arm]}
        out[arm] = {"capped_games": sum(t["frequency"] for t in caps),
                    "distinct_capped_trajectories": len(caps), "trajectories": rows,
                    "recurring_by_prefix": shared_prefixes,
                    "recurring_by_final_profile": [json.loads(p) for p, n in profiles.items()
                                                   if n >= 2]}
    return out


# ─────────────────────────────── 6. the run ───────────────────────────────

def timing_sample(trajs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Card §7 C3: shortest, median-length, longest by (length, digest)."""
    order = sorted(trajs.values(), key=lambda t: (len(t["moves"]), t["digest"]))
    return [order[0], order[(len(order) - 1) // 2], order[-1]]


def projection(sample_times: Sequence[float], sample_plies: Sequence[int], N: int) -> float:
    """Card §7 C3, frozen."""
    return sum(sample_times) / sum(sample_plies) * N


def render(moves: Sequence, upto: Optional[int] = None, size: int = 24) -> str:
    """Final (or ply-`upto`) board: R/B pegs, the last move lower-case."""
    moves = list(moves)[:upto] if upto is not None else list(moves)
    grid = [["." for _ in range(size)] for _ in range(size)]
    for i, (r, c) in enumerate(moves):
        ch = "R" if i % 2 == 0 else "B"
        grid[r][c] = ch.lower() if i == len(moves) - 1 else ch
    head = "    " + "".join(f"{c % 10}" for c in range(size))
    return "\n".join([head] + [f"{r:3d} " + "".join(row) for r, row in enumerate(grid)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="h5_weakness_review",
                                 description="H5 weakness review -- EXPLORATORY, read-only.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if "EXPLORATORY" not in out.name:
        print("REFUSED: the evidence directory name must say EXPLORATORY", file=sys.stderr)
        return 4
    out.mkdir(parents=False)                       # create-only

    def write(name, obj):
        with (out / name).open("x", encoding="utf-8") as fh:       # create-only
            fh.write(obj if isinstance(obj, str) else json.dumps(obj, indent=1, sort_keys=True))
    try:
        games, inputs = load_and_verify()
    except ReviewRefused as e:
        write("00_REFUSED.txt", f"{LABEL}: integrity refused, nothing analysed: {e}\n")
        print(f"REFUSED: {e}", file=sys.stderr)
        return 4
    write("00_inputs.json", {"label": LABEL, "sha256": inputs,
                             "cross_check": "per-arm outcomes, cap games, cap-affected pairs, "
                                            "unique games by arm, 592 games: all EQUAL"})
    trajs = distinct_trajectories(games)
    write("01_inventory.json", inventory(trajs))
    tables = prefix_tables(trajs)
    write("02_prefix_tables.json", tables)
    sample = timing_sample(trajs)
    times = []
    try:
        for t in sample:
            t0 = time.monotonic()
            scan_trajectory(t["moves"], t["actors"], t["terminal_reason"])
            times.append(time.monotonic() - t0)
    except ReviewRefused as e:
        write("03_STOPPED.txt", f"{LABEL}: the timing sample refused: {e}\n")
        return 4
    N = sum(len(t["moves"]) for t in trajs.values())
    proj = projection(times, [len(t["moves"]) for t in sample], N)
    timing = {"label": LABEL, "sample": [{"digest": t["digest"], "plies": len(t["moves"]),
                                          "seconds": s} for t, s in zip(sample, times)],
              "N": N, "projected_s": proj, "stop_s": STOP_S, "started": proj <= STOP_S}
    write("03a_timing.json", timing)
    print(f"timing: projected {proj:.0f} s for N={N} positions (stop {STOP_S:.0f} s)", flush=True)
    if proj > STOP_S:
        write("03_STOPPED.txt", f"{LABEL}: projection {proj:.0f} s > {STOP_S:.0f} s; "
                                f"the full run did not start\n")
        return 1
    scans, t0, done = {}, time.monotonic(), 0
    for d in sorted(trajs, key=lambda d: (len(trajs[d]["moves"]), d)):
        if time.monotonic() - t0 > STOP_S:
            write("03_STOPPED.txt", f"{LABEL}: stopped at {STOP_S:.0f} s after {done} of "
                                    f"{len(trajs)} trajectories\n")
            break
        t = trajs[d]
        try:
            scans[d] = scan_trajectory(t["moves"], t["actors"], t["terminal_reason"])
        except ReviewRefused as e:
            write("03_STOPPED.txt", f"{LABEL}: trajectory {d[:12]} refused after {done} of "
                                    f"{len(trajs)}: {e}\n")
            break
        done += 1
    hits = [{"digest": d, "arm": trajs[d]["arm"], "frequency": trajs[d]["frequency"],
             "outcome": trajs[d]["outcome"], "replays": [m["replay"] for m in trajs[d]["members"]],
             **r} for d, rows in scans.items() for r in rows if r["label"]]
    write("03_mechanical_checks.json", {
        "label": LABEL, "trajectories_scanned": done, "trajectories": len(trajs),
        "elapsed_s": time.monotonic() - t0, "rates": label_rates(trajs, scans),
        "fixed_distance": fixed_distance_flags(scans), "hits": hits})
    write("04_caps.json", cap_review(trajs, tables))
    notes = [f"# {LABEL} -- per-trajectory inspection notes (distinct losses and caps)\n"]
    for d in sorted(trajs, key=lambda d: (trajs[d]["outcome"], -trajs[d]["frequency"], d)):
        t = trajs[d]
        if t["outcome"] not in ("loss", "cap") or d not in scans:
            continue
        rows = scans[d]
        first = next((r["ply"] for r in rows if r["mover"] == t["winner"] and r["W"]), None) \
            if t["outcome"] == "loss" else None
        labs = [f"ply {r['ply']} {r['engine']} {r['label']}" for r in rows if r["label"]]
        notes.append(f"\n## {t['arm']} {t['outcome'].upper()} x{t['frequency']}  {d[:12]}  "
                     f"{len(t['moves'])} plies\n\nreplays: {', '.join(m['task_id'] for m in t['members'])}"
                     f"\n\nlabels: {labs or 'none'}\n\nwinner's first immediate win available "
                     f"at ply: {first}\n\n```\n{render(t['moves'])}\n```\n")
    write("05_inspection_notes.md", "".join(notes))
    print(f"done: {done}/{len(trajs)} trajectories in {time.monotonic() - t0:.0f} s; "
          f"{len(hits)} labelled positions", flush=True)
    return 0 if done == len(trajs) else 1


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
