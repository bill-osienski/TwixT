"""D0 -- zero-inference postmortem of the canonical 64-game L0 match.

NO EXECUTION. This module loads no model, starts no JVM, queries no T1j, runs
no inference, plays no game, draws no seed and trains nothing. It reads the
published record and recomputes deterministic board facts with our own rules
engine.

TWO RULES SHAPE THIS MODULE.

1. DIGEST BINDING IS A PRECONDITION, NOT A PEER. `bind_record` is the only
   producer of a `Bound`, and every function that can expose a move requires
   one. So there is no path from an unverified record to a move.

2. THE DISCOVERY/CONFIRMATION SPLIT IS TWO FUNCTIONS, NOT A FLAG. `inventory`
   answers integrity and outcome questions over all 64 games; `game_features`
   computes diagnostics and refuses anything outside repetitions 0 and 1. A
   flag would have a default, and a default is a switch-off.

This module deliberately does NOT import `eval_loss_replay_analysis`. That
module's vocabulary is `root_value`, `selected_visit_rank`, `root_top1_share` --
observables this record does not contain and which D0 is forbidden to claim.
Importing the vocabulary is how the forbidden claim gets made by accident.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from . import l0_match_plan as PLAN
from . import l0_match_rules as RULES

Pos = Tuple[int, int]

#: Repetitions whose diagnostics D0 may open, and those it may not.
DISCOVERY_REPS = (0, 1)
CONFIRMATION_REPS = (2, 3)


class D0Error(Exception):
    """Base for every D0 refusal."""


class D0BindingError(D0Error):
    """Identity, digest or record-shape failure. Never recoverable in-run."""


class D0ScopeError(D0Error):
    """An attempt to compute diagnostics outside the discovery half."""


@dataclass(frozen=True)
class Bound:
    """A record whose identity has been verified. The only key to the moves."""
    record_sha256: str
    plan_sha256: str
    task_digest: str
    header: Dict[str, Any]
    tasks: Tuple[Dict[str, Any], ...]
    openings: Dict[str, List[Pos]]
    starts: Dict[str, Dict[str, Any]]
    results: Dict[str, Dict[str, Any]]
    plies: Dict[str, List[Pos]]
    opening_bound: Dict[str, int]


def bind_record(jsonl_path: str, plan_path: str) -> Bound:
    """Verify identity, THEN expose the record. Refuses on any mismatch.

    Both digests are checked against the frozen definitions that produced them
    -- `l0_match_plan.L0_PLAN_SHA256` and `l0_match_rules.l0_task_digest` -- and
    never against a second definition invented here.
    """
    try:
        raw = open(jsonl_path, "rb").read()
    except OSError as e:
        raise D0BindingError(f"cannot read the record: {e}") from None
    record_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    except (ValueError, UnicodeDecodeError) as e:
        raise D0BindingError(f"record is not valid JSONL: {e}") from None

    headers = [r for r in rows if r.get("record_type") == "run_header"]
    if len(headers) != 1:
        raise D0BindingError(f"expected exactly 1 run_header, found {len(headers)}")
    header = headers[0]

    # --- the plan the record names must be the pinned plan, on disk and in the header
    if header.get("plan_sha256") != PLAN.L0_PLAN_SHA256:
        raise D0BindingError(
            f"header plan sha256 {header.get('plan_sha256')} != pinned {PLAN.L0_PLAN_SHA256}")
    try:
        plan_raw = open(plan_path, "rb").read()
    except OSError as e:
        raise D0BindingError(f"cannot read the plan: {e}") from None
    got_plan = hashlib.sha256(plan_raw).hexdigest()
    if got_plan != PLAN.L0_PLAN_SHA256:
        raise D0BindingError(f"plan sha256 {got_plan} != pinned {PLAN.L0_PLAN_SHA256}")

    # --- the embedded schedule must RECOMPUTE to the pinned digest
    try:
        embedded = header["identity"]["plan"]["plan"]
        tasks = embedded["tasks"]
        openings = {k: [tuple(m) for m in v] for k, v in embedded["openings"].items()}
    except (KeyError, TypeError) as e:
        raise D0BindingError(f"header does not embed a readable plan: {e}") from None
    try:
        recomputed = RULES.l0_task_digest(tasks)
    except ValueError as e:
        raise D0BindingError(f"embedded schedule is malformed: {e}") from None
    if recomputed != header.get("task_digest"):
        raise D0BindingError(
            f"recomputed task digest {recomputed} != header {header.get('task_digest')}")
    if recomputed != RULES.L0_TASK_DIGEST:
        raise D0BindingError(f"task digest {recomputed} != pinned {RULES.L0_TASK_DIGEST}")

    starts, results, ob = {}, {}, {}
    plies: Dict[str, List[Tuple[int, Pos]]] = {}
    for r in rows:
        t = r.get("record_type")
        if t == "task_start":
            starts[r["task_id"]] = r
        elif t == "task_result":
            results[r["task_id"]] = r
        elif t == "opening_bound":
            ob[r["task_id"]] = r["ply"]
        elif t == "ply":
            plies.setdefault(r["task_id"], []).append((r["ply"], tuple(r["move"])))

    ordered = {k: [m for _, m in sorted(v)] for k, v in plies.items()}
    return Bound(record_sha256=record_sha256, plan_sha256=header["plan_sha256"],
                 task_digest=recomputed, header=header, tasks=tuple(tasks),
                 openings=openings, starts=starts, results=results,
                 plies=ordered, opening_bound=ob)


def _require_bound(bound: Any) -> Bound:
    if not isinstance(bound, Bound):
        raise D0BindingError("not a bound record: call bind_record first")
    return bound


def game_moves(bound: Any, task_id: str) -> List[Pos]:
    """The full ordered move list: OPENING PREFIX + recorded plies.

    [A1]-3: the `ply` stream starts at ply 7 in every task; the six opening
    plies are not in it. They come from the embedded frozen plan, which is only
    trustworthy because `bind_record` checked its digest first.
    """
    b = _require_bound(bound)
    start = b.starts.get(task_id)
    if start is None:
        raise D0BindingError(f"no task_start for {task_id}")
    prefix = b.openings[start["opening"]]
    if len(prefix) != b.opening_bound[task_id]:
        raise D0BindingError(
            f"{task_id}: opening prefix is {len(prefix)} moves but opening_bound "
            f"says {b.opening_bound[task_id]}")
    return list(prefix) + list(b.plies.get(task_id, []))


def replay(bound: Any, task_id: str):
    """Reconstruct the final state of one game. Integrity, not diagnostics."""
    from .game.twixt_state import TwixtState
    st = TwixtState()
    for mv in game_moves(bound, task_id):
        st = st.apply_move(mv)
    return st


def discovery_task_ids(bound: Any) -> List[str]:
    """The 32 games whose diagnostics D0 may open. Sorted for determinism."""
    b = _require_bound(bound)
    return sorted(t for t, s in b.starts.items() if s["rep"] in DISCOVERY_REPS)


def inventory(bound: Any) -> Dict[str, Any]:
    """Integrity and OUTCOME COUNTS over both halves -- 4.2 permits exactly this.

    It deliberately emits no board diagnostic. Opening the confirmation half to
    outcome counts is what preregistration allows; opening it to features is
    what would burn the holdout.
    """
    b = _require_bound(bound)
    out: Dict[str, Any] = {"n_games": len(b.results), "reconstructed_ok": 0,
                           "record_sha256": b.record_sha256,
                           "plan_sha256": b.plan_sha256, "task_digest": b.task_digest}
    halves = {"discovery": DISCOVERY_REPS, "confirmation": CONFIRMATION_REPS}
    for name, reps in halves.items():
        ids = sorted(t for t, s in b.starts.items() if s["rep"] in reps)
        winners: Dict[str, int] = {}
        reasons: Dict[str, int] = {}
        pts = 0.0
        for t in ids:
            r = b.results[t]
            winners[str(r["winner"])] = winners.get(str(r["winner"]), 0) + 1
            reasons[r["terminal_reason"]] = reasons.get(r["terminal_reason"], 0) + 1
            pts += float(r["t1j_points"])
        out[name] = {"n_games": len(ids), "winners": winners,
                     "terminal_reasons": reasons, "t1j_points": pts,
                     "plies_total": sum(b.results[t]["plies"] for t in ids)}
    for t in b.results:
        st = replay(b, t)
        if st.ply == b.results[t]["plies"] and st.winner() == b.results[t]["winner"]:
            out["reconstructed_ok"] += 1
    return out


def _require_discovery(bound: Bound, task_id: str) -> Dict[str, Any]:
    """THE SPLIT. No flag, no override, no caller-supplied cohort."""
    start = bound.starts.get(task_id)
    if start is None:
        raise D0BindingError(f"no task_start for {task_id}")
    rep = start["rep"]
    if rep in CONFIRMATION_REPS:
        raise D0ScopeError(
            f"{task_id} is repetition {rep}: a confirmation game. Its diagnostics stay "
            f"closed until D2 has frozen a hypothesis, metric, effect and power.")
    if rep not in DISCOVERY_REPS:
        raise D0ScopeError(f"{task_id} has repetition {rep}, which is in neither half")
    return start


def game_features(bound: Any, task_id: str) -> List[Dict[str, Any]]:
    """Per-ply deterministic facts for ONE DISCOVERY game."""
    from .game.twixt_state import TwixtState
    b = _require_bound(bound)
    start = _require_discovery(b, task_id)
    result = b.results[task_id]
    moves = game_moves(b, task_id)
    rows = []
    st = TwixtState()
    n = len(moves)
    for i, mv in enumerate(moves):
        row = {"task_id": task_id, "opening": start["opening"],
               "colour_arm": start["colour_arm"], "rep": start["rep"],
               "ply": i, "mover": st.to_move, "move": list(mv),
               "winner": result["winner"], "phase": phase_of(i, n),
               "plies_from_terminal": n - i}
        row.update(ply_features(st, mv))
        rows.append(row)
        st = st.apply_move(mv)
    return rows


#: Which board coordinate each player must span. Red joins row 0 to row
#: active-1; black joins col 0 to col active-1. Taken from TwixtState._check_win.
_AXIS = {"red": 0, "black": 1}


def _components(state, player: str) -> List[List[Pos]]:
    """Every bridge-connected component of `player`, via the engine's own graph."""
    seen: set = set()
    out = []
    for p, owner in sorted(state.pegs.items()):
        if owner != player or p in seen:
            continue
        comp = state._get_connected_component(p, player)
        seen |= comp
        out.append(sorted(comp))
    return out


def component_boundary_distances(state, player: str) -> List[Dict[str, int]]:
    """[A1]-4. THE METRIC IS GEOMETRIC, not graph distance.

    For each component, the fewest rows (red) or columns (black) between its
    nearest peg and each of that player's two goal edges. Geometric is chosen
    because it needs no model of how an intervening hole could be filled; a
    graph distance would smuggle in a playability judgement D0 may not make.
    A component touching a goal scores 0 there.
    """
    if player not in _AXIS:
        raise D0Error(f"unknown player {player!r}")
    axis = _AXIS[player]
    far = state.active_size - 1
    return [{"size": len(comp),
             "to_goal1": min(p[axis] for p in comp),
             "to_goal2": min(far - p[axis] for p in comp)}
            for comp in _components(state, player)]


def _flip(state):
    """The same position with the other side to move. Deterministic, no search."""
    other = state.copy()
    other.to_move = "black" if state.to_move == "red" else "red"
    return other


def one_ply_threats(state) -> List[Pos]:
    """Legal moves that win for `state.to_move` immediately. Sorted.

    Measured before use: the naive scan costs ~2.7 ms/ply over the discovery
    half (~4 s total), so it is computed exactly as written, with no pruning to
    go wrong.
    """
    mover = state.to_move
    return [m for m in state.legal_moves() if state.apply_move(m).winner() == mover]


def ply_features(state, move: Pos) -> Dict[str, Any]:
    """Deterministic facts about ONE ply: the position faced, and the transition.

    Position counts describe the state BEFORE the move -- what the mover saw.
    Threat fields describe the transition. Nothing here is a move-quality label:
    "ignored" means a winning reply remained available, not that the move was bad.
    """
    from .game.twixt_state import KNIGHT_MOVES
    mover = state.to_move
    before_threats = one_ply_threats(_flip(state))     # what the OPPONENT could win with
    after = state.apply_move(move)
    opp_threats_after = one_ply_threats(after)         # `after` is the opponent's turn
    under = len(before_threats) > 0
    won = after.winner() == mover

    r, c = move
    own_knight_neighbours = sum(
        1 for dr, dc in KNIGHT_MOVES if state.pegs.get((r + dr, c + dc)) == mover)
    new_bridges = len(after.bridges) - len(state.bridges)

    row: Dict[str, Any] = {
        "empty_holes": state.active_size ** 2 - len(state.pegs),
        "legal_move_count": len(state.legal_moves()),
        "new_bridges": new_bridges,
        "blocked_bridge_opportunities": own_knight_neighbours - new_bridges,
        "immediate_win": won,
        "under_threat": under,
        "answered_threat": under and not opp_threats_after,
        "ignored_threat": under and bool(opp_threats_after),
        "created_threat": bool(one_ply_threats(_flip(after))),
        "opponent_threat_count": len(before_threats),
    }
    for colour in ("red", "black"):
        comps = component_boundary_distances(state, colour)
        row[f"pegs_{colour}"] = sum(1 for o in state.pegs.values() if o == colour)
        row[f"bridges_{colour}"] = sum(
            1 for (a, _b) in state.bridges if state.pegs[a] == colour)
        row[f"components_{colour}"] = len(comps)
        row[f"largest_component_{colour}"] = max((c["size"] for c in comps), default=0)
        row[f"min_goal1_distance_{colour}"] = min(
            (c["to_goal1"] for c in comps), default=None)
        row[f"min_goal2_distance_{colour}"] = min(
            (c["to_goal2"] for c in comps), default=None)
    opponent = "black" if mover == "red" else "red"
    row["mover_more_fragmented"] = (
        row[f"components_{mover}"] > row[f"components_{opponent}"])
    return row


#: Coarse game phase. Derived HERE, deliberately: the D1 analyzer has a phase
#: helper, but importing that module drags its telemetry vocabulary with it.
PHASES = ("opening", "early", "middle", "late")
OPENING_PLIES = 6


def phase_of(ply: int, n_plies: int) -> str:
    """Scripted opening, then three equal bands of the remaining game."""
    if ply < OPENING_PLIES:
        return "opening"
    span = max(1, n_plies - OPENING_PLIES)
    return PHASES[1 + min(2, int((ply - OPENING_PLIES) / span * 3))]


#: Dimensions every aggregate is cut by. Each cell keeps its own denominator.
AGGREGATE_DIMENSIONS = ("opening", "colour_arm", "winner", "phase")

#: Numeric columns worth a mean. Booleans are counted as rates by the same code.
_SUMMARY_SKIP = {"task_id", "move", "opening", "colour_arm", "winner", "phase", "mover"}


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Group per-ply rows by each dimension. EVERY cell carries `n`.

    Descriptive only. A per-opening or per-colour difference here is a count,
    never a finding: nothing in D0 is a preregistered test.
    """
    out: Dict[str, Any] = {}
    for dim in AGGREGATE_DIMENSIONS:
        groups: Dict[str, Any] = {}
        for r in rows:
            key = str(r.get(dim))
            cell = groups.setdefault(key, {"n": 0, "sums": {}, "counts": {}})
            cell["n"] += 1
            for k, v in r.items():
                if k in _SUMMARY_SKIP or not isinstance(v, (int, float, bool)):
                    continue
                cell["sums"][k] = cell["sums"].get(k, 0) + float(v)
                cell["counts"][k] = cell["counts"].get(k, 0) + 1
        for cell in groups.values():
            # EACH COLUMN CARRIES ITS OWN DENOMINATOR. A column that is None on
            # some plies -- min_goal*_distance_* before a colour has a peg --
            # must not be averaged over rows where it never existed.
            sums = cell.pop("sums")
            cell["means"] = {k: v / cell["counts"][k] for k, v in sums.items()}
        out[dim] = groups
    return out


#: Columns a held-out game could recompute identically -- i.e. those ply_features
#: itself produces. Condition 3 of the 4.5 gate is exactly this membership test.
def recomputable_columns() -> set:
    from .game.twixt_state import TwixtState
    st = TwixtState(active_size=5).apply_move((0, 1))
    return set(ply_features(st, (1, 3)))


def evaluate_gate(signatures: Sequence[Dict[str, Any]],
                  rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The 4.5 gate. GO only if some signature meets ALL FOUR conditions.

    NO_GO is a successful outcome: it says the recorded match yields a score but
    no repeated, measurable weakness. It is the default, and no signature list
    can reach GO by being empty.
    """
    recomputable = recomputable_columns()
    verdicts = []
    for sig in signatures:
        col = sig.get("column")
        hits = [r for r in rows if r.get(col)]
        openings = {r.get("opening") for r in hits}
        arms = {r.get("colour_arm") for r in hits}
        scope = sig.get("colour_scope")
        failed = []
        if len(openings) <= 1:
            failed.append(f"appears in {len(openings)} opening(s), needs more than one opening")
        if scope == "both":
            if len(arms) < 2:
                failed.append(f"claims both colour arms but appears in {sorted(arms)}")
        elif scope in arms and len(arms) == 1:
            pass                                  # explicitly scoped, and honoured
        else:
            failed.append(f"colour arm scope {scope!r} does not match observed {sorted(arms)}")
        if col not in recomputable:
            failed.append(f"column {col!r} is not produced by ply_features, so the "
                          f"held-out half could not recompute it identically")
        if not str(sig.get("d1_observable", "")).strip():
            failed.append("no named D1 observable measurable from both systems")
        offsets = {r.get("plies_from_terminal") for r in hits}
        offsets.discard(None)
        if hits and not offsets:
            failed.append("rows carry no plies_from_terminal, so the restates-the-ending "
                          "check cannot be evaluated; refusing rather than assuming")
        elif hits and len(offsets) == 1:
            failed.append(
                f"every hit sits at the same distance {sorted(offsets)} from the terminal ply, "
                f"so the signature restates the ending rather than a structural pattern")
        verdicts.append({"name": sig.get("name"), "column": col, "n_hits": len(hits),
                         "openings": sorted(o for o in openings if o is not None),
                         "colour_arms": sorted(a for a in arms if a is not None),
                         "failed": failed, "passes": not failed})
    return {"verdict": "GO" if any(v["passes"] for v in verdicts) else "NO_GO",
            "signatures": verdicts, "n_candidates": len(verdicts)}


def identity(bound: Any, repo_root: str = ".") -> Dict[str, Any]:
    """Section 3: what this analysis is bound to. Recorded, never inferred."""
    import subprocess
    b = _require_bound(bound)
    ident = b.header.get("identity", {})
    try:
        commit = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"repo_commit": commit,
            "record_sha256": b.record_sha256,
            "plan_sha256": b.plan_sha256,
            "task_digest": b.task_digest,
            "checkpoint": ident.get("checkpoint"),
            "jar": ident.get("jar"),
            "jdk": ident.get("jdk"),
            "task_ids": sorted(t["task_id"] for t in b.tasks)}


def run_d0(jsonl_path: str, plan_path: str, repo_root: str = ".") -> Dict[str, Any]:
    """The whole read-only postmortem: bind, inventory both halves, diagnose the
    discovery half, aggregate. The gate is evaluated separately, against
    signatures a human proposes after reading these aggregates.
    """
    b = bind_record(jsonl_path, plan_path)
    ids = discovery_task_ids(b)
    rows: List[Dict[str, Any]] = []
    for t in ids:
        rows.extend(game_features(b, t))
    return {"identity": identity(b, repo_root),
            "inventory": inventory(b),
            "n_discovery_games": len(ids),
            "n_discovery_plies": len(rows),
            "aggregates": aggregate(rows),
            "rows": rows}


#: Candidate structural signatures, evaluated by the 4.5 gate. Each names the
#: observable D1 would measure from BOTH systems; none is a move-quality label.
CANDIDATE_SIGNATURES = (
    {"name": "unanswered_immediate_threat", "column": "ignored_threat",
     "colour_scope": "both",
     "d1_observable": "root visit share and raw-policy mass on the threat-answering move"},
    {"name": "mover_fragmentation", "column": "mover_more_fragmented",
     "colour_scope": "both",
     "d1_observable": "root visit share and raw-policy mass on connection-joining moves"},
    {"name": "created_threat", "column": "created_threat", "colour_scope": "both",
     "d1_observable": "root visit share on the threat-creating move"},
)


def moved_by(colour_arm: str, mover: str) -> str:
    """Which engine moved: ``"ours"`` or ``"t1j"``. THE ONE DEFINITION.

    The arm names the ANCHOR's colour -- ``t1j_red`` / ``t1j_black`` here, and
    ``anchor_red`` / ``anchor_black`` in the E4 schedules -- so the mover is the
    anchor exactly when its colour is the arm's suffix, and ours otherwise.

    Extracted from `by_system`, where it was inline, because D1's selection rule
    needs the same cut to pick plies where OUR INCUMBENT is to move. A second
    statement of it could drift, and this cut decides what D1 interrogates at
    all. The expression is preserved exactly; a regression test pins the
    classification against evidence written before the extraction.
    """
    return "t1j" if mover == colour_arm.split("_")[1] else "ours"


def by_system(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Split each boolean signature by WHICH ENGINE moved.

    The 4.5 gate does not make this cut, and it is the cut that decides whether
    a recurring signature says anything about `calib020_0001` in particular. A
    signature present equally in both engines is a fact about losing at TwixT.
    Reported at TWO denominators: plies, and games -- plies within one game are
    not independent, so the ply rate overstates its own precision.
    """
    out: Dict[str, Any] = {}
    cols = [c for c in ("ignored_threat", "created_threat", "mover_more_fragmented",
                        "immediate_win") if rows and c in rows[0]]
    for col in cols:
        hits = {"ours": 0, "t1j": 0}
        plies = {"ours": 0, "t1j": 0}
        per_game: Dict[str, Dict[str, int]] = {}
        for r in rows:
            who = moved_by(r["colour_arm"], r["mover"])
            plies[who] += 1
            g = per_game.setdefault(r["task_id"], {"ours": 0, "t1j": 0})
            if r[col]:
                hits[who] += 1
                g[who] += 1
        out[col] = {
            "ply_rate": {w: {"hits": hits[w], "n_plies": plies[w],
                             "rate": hits[w] / plies[w] if plies[w] else None}
                         for w in ("ours", "t1j")},
            "n_games": len(per_game),
            "games_where_ours_exceeds_t1j": sum(
                1 for g in per_game.values() if g["ours"] > g["t1j"]),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Write the D0 evidence package. Create-only: never overwrites a record."""
    import argparse
    import os
    ap = argparse.ArgumentParser(description="D0 zero-inference postmortem")
    ap.add_argument("--record", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    r = run_d0(a.record, a.plan)
    artifacts = {
        "01_identity.json": r["identity"],
        "02_inventory.json": r["inventory"],
        "03_aggregates.json": {"n_discovery_games": r["n_discovery_games"],
                               "n_discovery_plies": r["n_discovery_plies"],
                               "aggregates": r["aggregates"]},
        "04_gate.json": evaluate_gate(CANDIDATE_SIGNATURES, r["rows"]),
        "05_by_system.json": by_system(r["rows"]),
    }
    os.makedirs(a.out, exist_ok=True)
    for name in artifacts:
        if os.path.exists(os.path.join(a.out, name)):
            raise D0Error(f"{name} already exists in {a.out}: evidence is create-only")
    for name, payload in artifacts.items():
        fd = os.open(os.path.join(a.out, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
