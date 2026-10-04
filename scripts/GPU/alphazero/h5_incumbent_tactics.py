"""H5 INCUMBENT TACTICS -- EXACT THREAT HANDLING (M1) vs PROVEN RESULTS (M2) -- EXPLORATORY.

Card: `docs/superpowers/2026-10-03-t1j-h5-incumbent-tactics-card.md`, with §10 A1.
🔴 THE CARD IS THE AUTHORITY. Tactical correctness only: no candidate, no strength claim.

Both mechanisms are READOUT-LEVEL functions of the UNCHANGED baseline tree, computed
separately and never composed:
  M1  an exact one-ply scan: OWN_WIN > DEFEND (every defence answers every threat) >
      explicit UNDEFENDABLE / QUIET (no change); the choice is the most-visited admissible
      move, flagged when it has no search evidence.
  M2  WIN / LOSS / UNKNOWN over the final tree (unvisited = UNKNOWN, never LOSS); every
      proven status persisted with a CERTIFICATE that the engine-only verifier replays.
Cap semantics (§10 A1.2): the search stays board-only (uncapped states, as in play and in
B-repro); M1 and M2 are evaluated under the capped H5 PROTOCOL (280; the winner is checked
before the cap) AND under board-only continuation, labelled apart. A board-only proof never
justifies a protocol exclusion or choice.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import eval_readout as RO
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_defence_factors as X
from . import h5_incumbent_tactics_authorization as AUTH
from . import h5_weakness_review as WR
from .game.twixt_state import TwixtState
from .mcts import decode_move, encode_move

LABEL = "EXPLORATORY"
CARD = "docs/superpowers/2026-10-03-t1j-h5-incumbent-tactics-card.md"
MODULE = "scripts.GPU.alphazero.h5_incumbent_tactics"
PLY_CAP = 280                       # the H5 protocol cap (§10 A1.2); a test pins it to R4's
SEMANTICS = ("protocol", "board")
CLASSES = ("OWN_WIN", "DEFEND_SINGLE", "DEFEND_MULTI", "UNDEFENDABLE", "QUIET")
QUOTA, CORPUS_SEED = 20, 20261003
CORPUS_NAME, CORPUS_FILE = "t1j-h5-incumbent-tactics-corpus-EXPLORATORY", "corpus.json"
OUT_NAME = "t1j-h5-incumbent-tactics-EXPLORATORY"
#: §6: every game that sources a fixture or a control
EXCLUDED_GAMES = tuple(sorted({g for ep in CC.EPISODES.values() for g in ep["members"]} |
                              {ct["game"] for ct in CC.CONTROLS.values()}))
#: §4, frozen: derived from the committed B-repro trees (protocol semantics)
FIXTURE_M1 = {"E1": (16, 9), "E2": (12, 5), "E3": (16, 6), "CD1": (16, 9), "CD2": (12, 5)}
FIXTURE_M1_NO_EVIDENCE = {"E3"}
FIXTURE_M2 = {"E1": (1, 2), "E2": (1, 4), "E3": (11, 10), "CD1": (16, 9), "CD2": (12, 5)}
HARD_LIMIT_S, MARGIN_S = 1800.0, 60.0
CLAIM_FILE, WATCHDOG_FILE = "claim.json", "watchdog.json"
CLAIM_ENV = "H5_INCUMBENT_TACTICS_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class TacticsRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class TacticsUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class TacticsStop(Exception):
    """A recorded STOP: a check failed, a certificate did not verify, or a rule was broken."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


# ─────────────────────────────── replay and digests ───────────────────────────────

class _Bad(Exception):
    pass


def replay(moves: Sequence, *, size: int = 24, semantics: str = "board",
           ply_cap: int = PLY_CAP) -> TwixtState:
    """Moves from the empty board, refusing a move after a winner and -- under the
    protocol -- a move at or past the cap. `apply_move` itself checks only the cell."""
    s = TwixtState(active_size=size, to_move="red")
    for m in moves:
        if s.winner() is not None:
            raise _Bad(f"a move {list(m)} after the game was won at ply {s.ply}")
        if semantics == "protocol" and s.ply >= ply_cap:
            raise _Bad(f"a move {list(m)} at ply {s.ply}, past the cap {ply_cap}")
        if not s.is_valid_placement(*_t(m)):
            raise _Bad(f"the illegal move {list(m)} at ply {s.ply}")
        s = s.apply_move(_t(m))
    return s


def state_digest(state: TwixtState) -> str:
    """§10 A1.1: sorted pegs [r, c, colour], sorted bridges, side to move."""
    body = {"pegs": sorted([r, c, col] for (r, c), col in state.pegs.items()),
            "bridges": sorted(sorted([list(a), list(b)]) for a, b in state.bridges),
            "to_move": state.to_move}
    return hashlib.sha256(json.dumps(body, separators=(",", ":"), sort_keys=True)
                          .encode()).hexdigest()


# ─────────────────────────────── M1: the exact one-ply scan (§3.1, §10 A1.2) ─────────────────

def classify(state: TwixtState, *, semantics: str, ply_cap: int = PLY_CAP
             ) -> Optional[Dict[str, Any]]:
    """None when the position is not a decision position under `semantics`: a winner
    exists, no legal move remains, or -- protocol -- ply >= cap. Otherwise the class with
    W (own immediate wins), T (the opponent's), D (moves leaving it none)."""
    if semantics not in SEMANTICS:
        raise ValueError(semantics)
    if state.winner() is not None or not state.legal_moves():
        return None
    if semantics == "protocol" and state.ply >= ply_cap:
        return None
    W = CC.immediate_wins(state)
    if W:                                        # terminal precedence: T is irrelevant
        return {"class": "OWN_WIN", "W": [list(w) for w in W], "T": None, "D": None,
                "T_void": None}
    T = CC.threats_against(state)
    void = None
    if semantics == "protocol" and state.ply + 1 >= ply_cap and T:
        #: the game ends as a draw after the mover's (non-winning) move: no reply exists
        void, T = [list(t) for t in T], []
    if not T:
        return {"class": "QUIET", "W": [], "T": [], "D": [], "T_void": void}
    Dset = sorted(_t(m) for m in WR.saving_moves_of(state, T))
    cls = ("UNDEFENDABLE" if not Dset else "DEFEND_SINGLE" if len(T) == 1 else "DEFEND_MULTI")
    return {"class": cls, "W": [], "T": [list(t) for t in T], "D": [list(m) for m in Dset],
            "T_void": None}


def admissible(c: Mapping[str, Any], legal: Sequence) -> List[Tuple[int, int]]:
    if c["class"] == "OWN_WIN":
        return [_t(m) for m in c["W"]]
    if c["class"] in ("DEFEND_SINGLE", "DEFEND_MULTI"):
        return [_t(m) for m in c["D"]]
    return [_t(m) for m in legal]                # UNDEFENDABLE, QUIET: no change


def _most_visited(moves: Sequence, counts: Mapping) -> Tuple[int, int]:
    return min(moves, key=lambda m: (-counts.get(_t(m), 0), _t(m)))


def m1_choice(c: Mapping[str, Any], counts: Mapping) -> Dict[str, Any]:
    baseline = RO._argmax(counts)
    A = admissible(c, list(counts))
    choice = _most_visited(A, counts)
    return {"baseline": list(baseline), "choice": list(choice), "changed": choice != baseline,
            "baseline_admissible": baseline in set(A),
            "no_search_evidence": counts.get(choice, 0) == 0, "choice_visits": counts.get(choice, 0)}


# ─────────────────────────────── M2: proven results (§3.2, §10 A1.2, A1.4) ──────────────────

def _past_cap(state: TwixtState, semantics: str, ply_cap: int) -> bool:
    return semantics == "protocol" and state.ply >= ply_cap


def proof_statuses(root, *, semantics: str, ply_cap: int = PLY_CAP) -> Dict[int, str]:
    """Bottom-up over the final tree, read-only. LOSS: a winner exists (the player to move
    lost), or every legal move has a created child that is WIN. WIN: some child is LOSS.
    UNKNOWN otherwise -- including an absent child, a protocol draw at the cap, and any
    value of +-1 that is not a terminal. Nothing below a terminal or the cap is read."""
    status: Dict[int, str] = {}
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        s = node.state
        if not done:
            stack.append((node, True))
            if s.winner() is None and not _past_cap(s, semantics, ply_cap):
                stack.extend((c, False) for c in node.children.values())
            continue
        if s.winner() is not None:
            status[id(node)] = "LOSS"
        elif _past_cap(s, semantics, ply_cap) or not node.is_expanded or not node.children:
            status[id(node)] = "UNKNOWN"
        elif any(status[id(c)] == "LOSS" for c in node.children.values()):
            status[id(node)] = "WIN"
        elif all(status[id(c)] == "WIN" for c in node.children.values()) and \
                {encode_move(*m) for m in s.legal_moves()} <= set(node.children):
            status[id(node)] = "LOSS"            # every legal move covered, every one WIN
        else:
            status[id(node)] = "UNKNOWN"
    return status


def certificate(node, status: Mapping[int, str]) -> Dict[str, Any]:
    """The witness for a proven node, reconstructible from the prefix by the engine alone."""
    st, s = status[id(node)], node.state
    if st == "LOSS" and s.winner() is not None:
        return {"kind": "TERMINAL_LOSS", "winner": s.winner()}
    if st == "WIN":
        mid = min((m for m, c in node.children.items() if status[id(c)] == "LOSS"),
                  key=decode_move)
        return {"kind": "WIN", "move": list(decode_move(mid)),
                "child": certificate(node.children[mid], status)}
    if st == "LOSS":
        legal = sorted(s.legal_moves())
        return {"kind": "ALL_REPLIES_WIN", "legal": [list(m) for m in legal],
                "children": {f"{r},{c}": certificate(node.children[encode_move(r, c)], status)
                             for r, c in legal}}
    raise ValueError(f"no certificate for an {st} node")


def verify_certificate(prefix: Sequence, path: Sequence, claim: str, cert: Mapping[str, Any],
                       *, semantics: str, ply_cap: int = PLY_CAP, size: int = 24
                       ) -> Tuple[bool, str]:
    """§10 A1.4: PURE and engine-only. Replays prefix + path (no move after a win; under the
    protocol none at or past the cap) and checks the claim recursively."""
    try:
        s = replay(list(prefix) + list(path), size=size, semantics=semantics, ply_cap=ply_cap)
        kind = cert.get("kind")
        if kind == "TERMINAL_LOSS":
            if claim != "LOSS":
                raise _Bad(f"a TERMINAL_LOSS certificate for a {claim} claim")
            if s.winner() is None or s.winner() == s.to_move or cert.get("winner") != s.winner():
                raise _Bad(f"not a terminal win for the previous mover (winner {s.winner()}, "
                           f"certificate {cert.get('winner')})")
            return True, "verified"
        if s.is_terminal() or _past_cap(s, semantics, ply_cap):
            #: a winner, the protocol cap, or the engine's DRAW (no legal move): no proof rule
            #: applies -- an empty ALL_REPLIES_WIN would otherwise "cover" a full board vacuously
            why = ("a winner" if s.winner() is not None else
                   f"the protocol cap {ply_cap}" if _past_cap(s, semantics, ply_cap) else
                   f"a draw: no winner, {len(s.legal_moves())} legal moves")
            raise _Bad(f"a {kind} certificate at a terminal position (ply {s.ply}: {why})")
        if kind == "WIN":
            if claim != "WIN":
                raise _Bad(f"a WIN certificate for a {claim} claim")
            move = _t(cert["move"])
            if not s.is_valid_placement(*move):
                raise _Bad(f"the winning move {list(move)} is illegal")
            ok, why = verify_certificate(prefix, list(path) + [move], "LOSS", cert["child"],
                                         semantics=semantics, ply_cap=ply_cap, size=size)
            if not ok:
                raise _Bad(why)
            return True, "verified"
        if kind == "ALL_REPLIES_WIN":
            if claim != "LOSS":
                raise _Bad(f"an ALL_REPLIES_WIN certificate for a {claim} claim")
            legal = sorted(s.legal_moves())
            if [_t(m) for m in cert["legal"]] != legal:
                raise _Bad("the certificate's legal list is not the engine's")
            if sorted(cert["children"]) != sorted(f"{r},{c}" for r, c in legal):
                raise _Bad("the certificate's children do not cover exactly the legal moves")
            for r, c in legal:
                ok, why = verify_certificate(prefix, list(path) + [(r, c)], "WIN",
                                             cert["children"][f"{r},{c}"], semantics=semantics,
                                             ply_cap=ply_cap, size=size)
                if not ok:
                    raise _Bad(why)
            return True, "verified"
        raise _Bad(f"unknown certificate kind {kind!r}")
    except (_Bad, KeyError, TypeError, ValueError) as e:
        return False, str(e)


def m2_choice(root, counts: Mapping, *, semantics: str, ply_cap: int = PLY_CAP
              ) -> Dict[str, Any]:
    """§3.2: a proven win is taken; otherwise proven-losing root moves are excluded and the
    most-visited of the rest chosen; if every legal move is proven losing, the unchanged
    readout stands (ROOT_PROVEN_LOSS). Every proven root child carries its certificate."""
    status = proof_statuses(root, semantics=semantics, ply_cap=ply_cap)
    baseline = RO._argmax(counts)
    proven = []
    for mid, child in sorted(root.children.items(), key=lambda kv: decode_move(kv[0])):
        st = status.get(id(child), "UNKNOWN")
        if st != "UNKNOWN":
            proven.append({"move": list(decode_move(mid)), "status": st,
                           "certificate": certificate(child, status)})
    wins = [_t(p["move"]) for p in proven if p["status"] == "LOSS"]   # the opponent is lost
    excluded = {_t(p["move"]) for p in proven if p["status"] == "WIN"}
    if wins:
        choice, rule = _most_visited(wins, counts), "PROVEN_WIN"
    else:
        rest = [m for m in counts if m not in excluded]
        if not rest:
            choice, rule = baseline, "ROOT_PROVEN_LOSS"
        else:
            choice = _most_visited(rest, counts)
            rule = "EXCLUSION" if excluded else "UNCHANGED"
    return {"baseline": list(baseline), "choice": list(choice), "changed": choice != baseline,
            "rule": rule, "root_status": status.get(id(root), "UNKNOWN"),
            "excluded": sorted(list(m) for m in excluded), "proven_root_children": proven,
            "no_search_evidence": counts.get(choice, 0) == 0, "choice_visits": counts.get(choice, 0)}


VALUE_NOTE = ("q_root_mover: the child's mean value for the ROOT's mover (+1 = a win for the player "
              "to move at the root); q_node: the same mean for the player to move AT the child; "
              "both null when never visited")


def tactics(state: TwixtState, counts: Mapping, root, *, ply_cap: int = PLY_CAP
            ) -> Dict[str, Any]:
    """Both mechanisms, under both semantics, labelled apart. Reads only. §7: the visits and
    the root mover's value of the baseline and of each mechanism's choice, per semantics."""
    out: Dict[str, Any] = {}
    for sem in SEMANTICS:
        c = classify(state, semantics=sem, ply_cap=ply_cap)
        m1 = None if c is None else m1_choice(c, counts)
        m2 = m2_choice(root, counts, semantics=sem, ply_cap=ply_cap)
        stat = lambda m: D.node_stat(root, encode_move(*_t(m)), 1)   # noqa: E731
        out[sem] = {"class": c, "M1": m1, "M2": m2,
                    "values": {"note": VALUE_NOTE, "baseline": stat(m2["baseline"]),
                               "M1": None if m1 is None else stat(m1["choice"]),
                               "M2": stat(m2["choice"])}}
    p, b = out["protocol"], out["board"]
    out["cap_sensitive"] = (p["class"] != b["class"] or p["M1"] != b["M1"] or
                            p["M2"]["choice"] != b["M2"]["choice"] or
                            p["M2"]["proven_root_children"] != b["M2"]["proven_root_children"])
    return out


# ─────────────────────────────── the corpus (§6, §10 A1.1) ───────────────────────────────

def fixture_digests(frozen: Mapping[str, Any]) -> Dict[str, str]:
    """The state digests of the five fixtures and the seven controls."""
    out = {}
    for e, ep in frozen["episodes"].items():
        out[e] = state_digest(CC.state_of(ep["incumbent_position"]["moves"]))
    for c, ct in frozen["controls"].items():
        out[c] = state_digest(CC.state_of(ct["position"]["moves"]))
    return out


def corpus_universe(games: Sequence[Mapping[str, Any]], *, excluded: Sequence[str],
                    fixtures: Mapping[str, str], ply_cap: int = PLY_CAP
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every incumbent decision position under the protocol, outside the excluded games,
    classified both ways; fixture-equivalent states set aside."""
    entries, fixture_equivalent = [], []
    rev = {v: k for k, v in fixtures.items()}
    for g in sorted(games, key=lambda g: g["task_id"]):
        if g["task_id"] in set(excluded):
            continue
        s = TwixtState(active_size=24, to_move="red")
        for i, mv in enumerate(g["moves"]):
            if g["actors"][i] == "incumbent":
                cp = classify(s, semantics="protocol", ply_cap=ply_cap)
                if cp is not None:
                    cb = cp if s.ply + 1 < ply_cap else classify(s, semantics="board",
                                                                 ply_cap=ply_cap)
                    moves = [list(_t(m)) for m in g["moves"][:i]]
                    e = {"key": f"{g['task_id']}@{i}", "game": g["task_id"], "ply": i,
                         "moves": moves, "prefix_digest": CC.prefix_digest(moves),
                         "state_digest": state_digest(s), "class_protocol": cp,
                         "class_board": cb}
                    if e["state_digest"] in rev:
                        fixture_equivalent.append({**e, "equivalent_to": rev[e["state_digest"]]})
                    else:
                        entries.append(e)
            s = s.apply_move(_t(mv))
    return entries, fixture_equivalent


def select_corpus(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§10 A1.1: per class, ONE representative per game, then at most QUOTA games; a
    shortage takes every game and is recorded; repeated states decided once."""
    selected, shortfall, universe = [], {}, {}
    for s in CLASSES:
        by_game = collections.defaultdict(list)
        for e in entries:
            if e["class_protocol"]["class"] == s:
                by_game[e["game"]].append(e)
        universe[s] = {"positions": sum(len(v) for v in by_game.values()), "games": len(by_game)}
        reps = {}
        for g, es in by_game.items():
            plies = sorted(e["ply"] for e in es)
            ply = random.Random(f"{CORPUS_SEED}:{s}:{g}").choice(plies)
            reps[g] = next(e for e in es if e["ply"] == ply)
        games = sorted(reps)
        chosen = random.Random(f"{CORPUS_SEED}:{s}").sample(games, min(QUOTA, len(games)))
        shortfall[s] = max(0, QUOTA - len(games))
        selected += [{**reps[g], "class": s} for g in sorted(chosen, key=lambda g: (g, reps[g]["ply"]))]
    first, repeats = {}, collections.defaultdict(list)
    for e in selected:                           # class order, then (task_id, ply)
        if e["state_digest"] in first:
            e["repeat_of"] = first[e["state_digest"]]
            repeats[e["state_digest"]].append(e["key"])
        else:
            e["repeat_of"] = None
            first[e["state_digest"]] = e["key"]
    return {"selected": selected, "quota_shortfall": shortfall, "universe": universe,
            "repeats": {d: [first[d]] + ks for d, ks in sorted(repeats.items())}}


def build_manifest(games, frozen) -> Dict[str, Any]:
    fx = fixture_digests(frozen)
    entries, fixture_equivalent = corpus_universe(games, excluded=EXCLUDED_GAMES, fixtures=fx)
    sel = select_corpus(entries)
    return {"label": LABEL, "card": CARD, "seed": CORPUS_SEED, "quota": QUOTA, "ply_cap": PLY_CAP,
            "excluded_games": list(EXCLUDED_GAMES), "fixture_digests": fx,
            "fixture_equivalent": [{k: e[k] for k in ("key", "state_digest", "equivalent_to")}
                                   for e in fixture_equivalent],
            **sel}


def manifest_bytes(m: Mapping[str, Any]) -> bytes:
    return (json.dumps(m, indent=1, sort_keys=True) + "\n").encode()


# ─────────────────────────────── one position, observed (§10 A1.3) ───────────────────────────

def observe(kind: str, build: Callable, config, position: Mapping[str, Any]) -> Dict[str, Any]:
    """kind 'B-repro' (fixture: historical move + pinned fingerprint) or 'corpus' (no
    in-game comparison). Off/on equivalence, the frozen config, seed and budget; then M1
    and M2 from both trees, shown to leave trees and streams untouched."""
    state = CC.state_of(position["moves"])
    off = D.decide(build, state, capture=False)
    on = D.decide(build, state, capture=True)
    passed: List[str] = []

    def need(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise TacticsStop(f"{kind} {position['id']}: check '{name}' failed. {detail}")
        passed.append(name)

    for tag, r in (("off", off), ("on", on)):
        a = r["agent"]
        diff = X.config_diff(a, config)
        need(f"configuration ({tag})", diff == set(), f"differs in {sorted(diff)}")
        need(f"seed and colour ({tag})", a.seed == D.SEED and a.colour == state.to_move)
        need(f"budget ({tag})", r["root"].visit_count == config.mcts_sims == 400,
             f"root visits {r['root'].visit_count}")
        need(f"readout is the visit argmax ({tag})", RO._argmax(r["counts"]) == r["move"])
    need("move off == on", off["move"] == on["move"])
    need("search stream", off["search_rng"] == on["search_rng"])
    need("readout stream", off["readout_rng"] == on["readout_rng"])
    need("counts", off["counts"] == on["counts"])
    need("tie set", D.tie_set(off["counts"]) == D.tie_set(on["counts"]))
    fp = {tag: D.fingerprint(r["root"], r["counts"], r["root_value"])
          for tag, r in (("off", off), ("on", on))}
    need("tree fingerprint off == on", fp["off"] == fp["on"])
    if kind == "B-repro":
        need("B-repro move", off["move"] == _t(position["recorded"]))
        need("B-repro fingerprint", fp["off"] == X.PINNED_FINGERPRINTS[position["id"]])
    results = {}
    for tag, r in (("off", off), ("on", on)):
        before = (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate())
        results[tag] = tactics(state, r["counts"], r["root"])
        need(f"tactics left the tree unchanged ({tag})",
             D.fingerprint(r["root"], r["counts"], r["root_value"]) == fp[tag])
        need(f"tactics drew from no stream ({tag})",
             (r["agent"].mcts.rng.getstate(), r["agent"].readout_rng.getstate()) == before)
    need("tactics agree off/on", results["off"] == results["on"])
    return {"kind": kind, "id": position["id"], "moves": [list(_t(m)) for m in position["moves"]],
            "ply": state.ply, "move": list(off["move"]), "fingerprint": fp["off"],
            "checks": passed, "tactics": results["off"]}


# ─────────────────────────────── the conclusion: verification + structural rules ─────────────

def verify_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    """Every persisted certificate, replayed by the engine; any failure STOPs."""
    n = 0
    for r in rows:
        for sem in SEMANTICS:
            for p in r["tactics"][sem]["M2"]["proven_root_children"]:
                ok, why = verify_certificate(r["moves"], [p["move"]], p["status"],
                                             p["certificate"], semantics=sem)
                if not ok:
                    raise TacticsStop(f"{r['id']} {sem}: the certificate for {p['move']} "
                                      f"does not verify: {why}")
                n += 1
    return n


def _both_visited(rs: Sequence[Mapping[str, Any]], mech: str) -> List[Dict[str, Any]]:
    """§7, descriptive: where the baseline and the mechanism's choice were both visited, the
    root mover's value of each (protocol)."""
    out = []
    for r in rs:
        v = r["tactics"]["protocol"]["values"]
        if v[mech] is not None and v["baseline"]["visits"] > 0 and v[mech]["visits"] > 0:
            out.append({"id": r["id"], "changed": r["tactics"]["protocol"][mech]["changed"],
                        "baseline_q_root_mover": v["baseline"]["q_root_mover"],
                        "choice_q_root_mover": v[mech]["q_root_mover"]})
    return out


def interpret(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§7: the structural rules (a violation STOPs) and the descriptive tallies."""
    bad = []
    for r in rows:
        for sem in SEMANTICS:
            t = r["tactics"][sem]
            c, m1, m2 = t["class"], t["M1"], t["M2"]
            if c is not None:
                if c["class"] in ("QUIET", "UNDEFENDABLE"):
                    if m1["changed"]:
                        bad.append(f"{r['id']} {sem}: M1 changed a {c['class']} move")
                else:                                # OWN_WIN / DEFEND: A is W or D
                    if _t(m1["choice"]) not in set(admissible(c, [])):
                        bad.append(f"{r['id']} {sem}: M1 chose a non-admissible move")
                    if m1["baseline_admissible"] and m1["changed"]:
                        bad.append(f"{r['id']} {sem}: M1 replaced an admissible move")
            proven = {_t(p["move"]): p["status"] for p in m2["proven_root_children"]}
            if any(proven.get(_t(m)) != "WIN" for m in m2["excluded"]):
                bad.append(f"{r['id']} {sem}: M2 excluded a move not proven losing")
            if m2["changed"] and not (m2["rule"] == "PROVEN_WIN" or
                                      proven.get(_t(m2["baseline"])) == "WIN"):
                bad.append(f"{r['id']} {sem}: M2 changed a move without a proof")
        if r["kind"] == "B-repro":
            p = r["tactics"]["protocol"]
            if _t(p["M1"]["choice"]) != FIXTURE_M1[r["id"]] or \
                    p["M1"]["no_search_evidence"] != (r["id"] in FIXTURE_M1_NO_EVIDENCE):
                bad.append(f"{r['id']}: M1 is not the frozen prediction {FIXTURE_M1[r['id']]}")
            if _t(p["M2"]["choice"]) != FIXTURE_M2[r["id"]]:
                bad.append(f"{r['id']}: M2 is not the frozen prediction {FIXTURE_M2[r['id']]}")
    if bad:
        raise TacticsStop("structural rules violated: " + "; ".join(bad[:12]))
    tally = {}
    for s in CLASSES:
        rs = [r for r in rows if r["kind"] == "corpus" and r.get("repeat_of") is None and
              r["tactics"]["protocol"]["class"] and r["tactics"]["protocol"]["class"]["class"] == s]
        tally[s] = {"distinct_positions": len(rs),
                    "M1_changed": sum(r["tactics"]["protocol"]["M1"]["changed"] for r in rs),
                    "M1_no_search_evidence": sum(r["tactics"]["protocol"]["M1"]["changed"] and
                                                 r["tactics"]["protocol"]["M1"]["no_search_evidence"]
                                                 for r in rs),
                    "M2_changed": sum(r["tactics"]["protocol"]["M2"]["changed"] for r in rs),
                    "M2_root_proven_loss": sum(r["tactics"]["protocol"]["M2"]["rule"] ==
                                               "ROOT_PROVEN_LOSS" for r in rs),
                    "values_where_both_visited": {m: _both_visited(rs, m) for m in ("M1", "M2")}}
    return {"verdict": "the structural rules hold", "protocol_tally": tally,
            "cap_sensitive": [r["id"] for r in rows if r["tactics"]["cap_sensitive"]],
            "note": "tactical behaviour of two readout rules on the unchanged tree; no candidate, "
                    "no strength claim; the protocol results are primary, board-only apart"}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def plan_of(fixtures: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
            ) -> List[Tuple[str, Dict[str, Any]]]:
    """B-repro first (fixtures), then each DISTINCT corpus state once."""
    out = [("B-repro", dict(p)) for p in fixtures]
    out += [("corpus", {"id": e["key"], "moves": e["moves"], "repeat_of": None})
            for e in manifest["selected"] if e["repeat_of"] is None]
    return out


def _run(plan: Sequence[Tuple[str, Mapping[str, Any]]], config, identity: Mapping[str, Any],
         loader: Callable, d: pathlib.Path, *, clock: Callable[[], float] = time.monotonic,
         wall_s: float = HARD_LIMIT_S - MARGIN_S) -> Dict[str, Any]:
    from . import d1_probe as D1
    if not _unconsumed(d):
        raise TacticsRefused(f"{d} does not hold exactly an unconsumed claim")
    deadline = D1.Deadline(limit_s=wall_s, clock=clock).start()
    log, step, rows = None, "claim", []
    status, failure, interrupt = "COMPLETE", None, None

    def due(next_step: str) -> None:
        if deadline.elapsed() > wall_s:
            raise TacticsStop(f"the {wall_s:.0f} s deadline passed before {next_step}")

    try:
        log = CC._Log(d / "log.jsonl")
        step = "model_load"
        with D1._supervisor(deadline):
            build = D.study_builder(config, loader(), D.tasks_for(identity))
            log.done("model_load", {"incumbent_identity": identity})
            for kind, p in plan:
                step = f"observe:{kind}:{p['id']}"
                due(step)
                row = observe(kind, build, config, p)
                rows.append(row)
                log.done("observation", row)
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    finally:
        if log is not None:
            log.close()
    #: the completion check, the verification and the interpretation sit inside their OWN
    #: recorded boundary (the controlled study's repair): a failure here still leaves the record
    interpretation, verified = None, None
    if status == "COMPLETE":
        step = "completion_check"
        try:
            want = collections.Counter((k, p["id"]) for k, p in plan)
            got = collections.Counter((r["kind"], r["id"]) for r in rows)
            if got != want:
                status, failure = "INCOMPLETE", {
                    "failing_step": "completion_check", "exception_type": None,
                    "reason": "the observations are not the plan",
                    "missing": sorted(list(k) for k in want - got),
                    "duplicated": sorted(list(k) for k, n in got.items() if n > 1),
                    "extra": sorted(list(k) for k in set(got) - set(want))}
            else:
                step = "verification"
                verified = verify_rows(rows)
                step = "interpretation"
                interpretation = interpret(rows)
        except (KeyboardInterrupt, SystemExit) as e:
            status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
            interpretation = None
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            status, failure, interpretation = "STOP", _failure(d, step, e), None
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "interpretable": status == "COMPLETE", "interpretation": interpretation,
              "interpretation_note": None if status == "COMPLETE" else
              f"not interpreted: the run is {status}",
              "certificates_verified": verified, "incumbent_identity": dict(identity),
              "ply_cap": PLY_CAP, "rows": rows, "deadline_s": wall_s,
              "elapsed_s": deadline.elapsed(),
              "note": "protocol results primary, board-only apart; no candidate, no strength claim"}
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
        raise TacticsRefused(f"{d} is occupied: the study is one-shot") from None
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
        raise TacticsRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise TacticsRefused("the claim's nonce does not match: this process was not "
                             "started by the parent that claimed")
    return claim


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
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
    if not AUTH.H5_INCUMBENT_TACTICS_AUTHORIZED:
        raise TacticsUnauthorized("the H5 incumbent tactics study is UNAUTHORIZED. Nothing "
                                  "was opened, read, built, loaded, decided or written.")


def corpus_path() -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / CORPUS_NAME / CORPUS_FILE


def out_dir() -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / OUT_NAME


def _derive_manifest() -> Dict[str, Any]:
    try:
        games, _ = WR.load_and_verify()
    except WR.ReviewRefused as e:
        raise TacticsRefused(f"the committed H5 records: {e}") from None
    return build_manifest(games, D._frozen())


def _committed_manifest(*, rederive: bool) -> Tuple[Dict[str, Any], str]:
    path = corpus_path()
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise TacticsRefused(f"the corpus must be committed before any model load: {e}") from None
    raw = path.read_bytes()
    if rederive and raw != manifest_bytes(_derive_manifest()):
        raise TacticsRefused("the committed corpus is not the one re-derived from the records")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def freeze_corpus() -> str:
    """PUBLIC ENTRY. Writes the corpus manifest once -- to be COMMITTED before any model load."""
    _gate()
    path = corpus_path()
    if os.path.lexists(path.parent):
        raise TacticsRefused(f"{path.parent} is occupied: the corpus is frozen once")
    manifest = _derive_manifest()
    os.makedirs(path.parent)
    with open(path, "xb") as fh:
        fh.write(manifest_bytes(manifest))
    return str(path)


def _child_argv() -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed"]


def run() -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals (the committed corpus RE-DERIVED), THE
    CLAIM, the child under the hard watchdog; `watchdog.json` on every exit."""
    _gate()
    out = out_dir()
    if os.path.lexists(out):
        raise TacticsRefused(f"{out} is occupied: the study is one-shot")
    _manifest, sha = _committed_manifest(rederive=True)
    D.positions(D._frozen())
    D._identity(R4.frozen_argmax_config())
    claim, nonce = _claim(out, {"corpus_sha256": sha})
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
    """The CHILD: the gate, the committed corpus (durable, the claim's sha), the fixtures,
    the identity, the claim; then the model load and the plan."""
    _gate()
    out = out_dir()
    manifest, sha = _committed_manifest(rederive=False)
    fixtures = D.positions(D._frozen())
    config = R4.frozen_argmax_config()
    identity = D._identity(config)
    claim = _accept_claim(out)
    if claim.get("corpus_sha256") != sha:
        raise TacticsRefused("the corpus is not the one the parent re-derived")
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(plan_of(fixtures, manifest), config, identity,
                lambda: R4.load_production_evaluator(identity), out,
                wall_s=HARD_LIMIT_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_incumbent_tactics",
                                 description="H5 incumbent tactics study -- EXPLORATORY. "
                                             "NOT AUTHORIZED.")
    ap.add_argument("step", choices=("freeze-corpus", "run", "run-claimed"))
    a = ap.parse_args(argv)
    if not AUTH.H5_INCUMBENT_TACTICS_AUTHORIZED:
        print("the H5 incumbent tactics study is UNAUTHORIZED. Nothing was opened, read, "
              "built, loaded, decided or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "freeze-corpus":
            print(freeze_corpus())
        elif a.step == "run":
            res = run()
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed()["status"]}))
    except (TacticsRefused, TacticsUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
