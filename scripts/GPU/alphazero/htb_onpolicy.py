"""H-OP -- distillation on fresh self-play from the current `hard` (card revision 3).

Card: docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md, pinned below by sha256. This module is the card's
model-free build: nothing runs on import, and importing it loads no MLX, TensorFlow or model. Every runtime entry is
behind its own closed gate, read at call time:

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_onpolicy run gen-smoke     # HTB_ONPOLICY_GEN_SMOKE_AUTHORIZED
    python -m scripts.GPU.alphazero.htb_onpolicy yield                              # pure: the smoke's yield verdict
    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_onpolicy run generate      # HTB_ONPOLICY_GENERATE_AUTHORIZED
    <teacher python> -m scripts.GPU.alphazero.htb_onpolicy run label --clone-root D  # HTB_ONPOLICY_LABEL_AUTHORIZED
    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_onpolicy run fit           # HTB_ONPOLICY_FIT_AUTHORIZED
    python -m scripts.GPU.alphazero.htb_onpolicy openings                           # pure: the frozen opening lists

Reused, never modified: `htb_pilot` (the key, the teacher, label storage, examples, scoring, the terminal-record
boundary), `htb_bench_teacher.bench_divergences` (the benchmark's lockstep check), the research protocol pieces G1
ported to Node (`RootSequentialHalvingMCTS`, `MirrorAveragingEvaluator`, `eval_readout.select`).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from . import htb_pilot as P
from .game.twixt_state import TwixtState

ROOT = P.ROOT
CARD = "docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md"
CARD_SHA256 = "a2a5ba2f7fad3018c16df2abc98c9c5d3ab8e9095415b3c06f41cd35a5aef5f1"   # revision 3
NS = "htb-onpolicy/20261114"
LAST = 23

# ── card §4.2: openings (the benchmark's rule), seeds, the run's split ──
CENTRAL_MIN, CENTRAL_MAX, OPENING_PLIES = 4, 19, 4
BENCH_NS = "htb-teacher-bench/20261114"
BENCH_OPENINGS = {"opening": 200, "smoke opening": 1}          # its match and smoke opening lists
SIMS = 200

# ── card §4.3-4.4: counts, stopping rule, smoke ──
N_TRAIN, N_VAL = 120_000, 12_000
STOP = {"train": 138_000, "val": 13_800}                        # 1.15 x N
G_MAX = 4_000
SMOKE_GAMES = 40
GEN_CAP_S, SMOKE_CAP_S = 144_000, 3_600
TRUNC, Z_ONE_SIDED, F_MIN, MIN_SMOKE_PLIES = 150, 1.645, 0.90, 800

# ── card §5 / §7 / §8 ──
SELECT_TAG, FIT_TAG = "htb-onpolicy-select/1", "htb-onpolicy-fit/1"
RECIPE = P.Recipe()                                             # the first fit's recipe, unchanged
UPDATES = RECIPE.epochs * N_TRAIN // RECIPE.rows_per_batch      # 7,500
I1_FACTOR = 0.90
START_ROWS, START_KL_FLOOR = P.START_ROWS, P.START_KL_FLOOR

STUDENT = {"path": P.CACHE / "student.safetensors",
           "sha256": "0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5",
           "state_digest": "586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b"}
CACHE = pathlib.Path.home() / "twixt-cache" / "htb-onpolicy"   # outside the repository

# evidence (create-only, one run each)
EV = {"openings": "docs/superpowers/evidence/htb-onpolicy-openings",
      "gen-smoke": "docs/superpowers/evidence/htb-onpolicy-gen-smoke",
      "generate": "docs/superpowers/evidence/htb-onpolicy-gen",
      "label": "docs/superpowers/evidence/htb-onpolicy-label",
      "fit": "docs/superpowers/evidence/htb-onpolicy-fit",
      "export": "docs/superpowers/evidence/htb-onpolicy-export"}
OUTER = {"gen-smoke": "docs/superpowers/evidence/htb-onpolicy-gen-smoke-run",
         "generate": "docs/superpowers/evidence/htb-onpolicy-gen-run",
         "label": "docs/superpowers/evidence/htb-onpolicy-label-run",
         "fit": "docs/superpowers/evidence/htb-onpolicy-fit-run",
         "export": "docs/superpowers/evidence/htb-onpolicy-export-run"}
OPENINGS_REL = f"{EV['openings']}/openings.json"
YIELD_REL = f"{EV['gen-smoke']}/yield.json"

# exclusion sources (card §5.3), all committed
BENCH_GAMES = ("docs/superpowers/evidence/htb-teacher-bench-match/games",
               "docs/superpowers/evidence/htb-teacher-bench-smoke/games")
G3_GAMES = ("docs/superpowers/evidence/htb-g3-match/games", "docs/superpowers/evidence/htb-g3-smoke/games")
DIAG_O = "docs/superpowers/evidence/htb-diag-o/teacher_in.json"
FIRST_SELECTION = "docs/superpowers/evidence/htb-label/selection.json"
FIRST_LABEL_RECORD = "docs/superpowers/evidence/htb-label/record.json"

Stop, Refused = P.Stop, P.Refused
_sha, _jsha = P._sha, P._jsha
state_key, reflect_rc = P.state_key, P.reflect_rc


# ─────────────────────────────── openings (card §4.2, §10; the benchmark's rule) ───────────────────────────────

def opening_at(ns: str, kind: str, index: int) -> List[Tuple[int, int]]:
    """`teacher_bench.mjs` openingAt, in Python: ply k takes index int(sha256(f"{ns}/{kind} {index}/ply {k}")[:12], 16)
    mod n over our legal moves in rows and columns 4-19, in sorted (row, col) order."""
    s, moves = TwixtState(), []
    for k in range(OPENING_PLIES):
        cands = sorted((int(r), int(c)) for r, c in s.legal_moves()
                       if CENTRAL_MIN <= r <= CENTRAL_MAX and CENTRAL_MIN <= c <= CENTRAL_MAX)
        h = _sha(f"{ns}/{kind} {index}/ply {k}".encode())
        mv = cands[int(h[:12], 16) % len(cands)]
        moves.append(mv)
        s = s.apply_move(mv)
    return moves


def _seq(moves: Sequence[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in moves)


def _reflected(moves: Sequence[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(reflect_rc(m) for m in moves)


def opening_list(ns: str, kind: str, n: int, exclude: Iterable[Sequence[Sequence[int]]] = ()) -> List[Dict[str, Any]]:
    """The first `n` openings of `kind` distinct from each other and from `exclude`, directly or reflected (the
    benchmark's `openingList`, plus an exclusion set). Returns [{"index", "moves"}]."""
    seen = {_seq(m) for m in exclude}
    out, i = [], 0
    while len(out) < n:
        moves = opening_at(ns, kind, i)
        if _seq(moves) not in seen and _reflected(moves) not in seen:
            seen.add(_seq(moves))
            out.append({"index": i, "moves": [list(m) for m in moves]})
        i += 1
    return out


def _game_files(root: pathlib.Path, dirs: Sequence[str]) -> List[pathlib.Path]:
    files = []
    for d in dirs:
        got = sorted((root / d).glob("*.json"))
        if not got:
            raise Refused(f"no committed games under {d}")
        files += got
    return files


def bench_openings() -> List[List[List[int]]]:
    """The benchmark's openings, re-derived by its own rule: its 200 match openings and its smoke opening."""
    return [o["moves"] for kind, n in BENCH_OPENINGS.items() for o in opening_list(BENCH_NS, kind, n)]


def g3_openings(root: pathlib.Path = ROOT) -> List[List[List[int]]]:
    """The first four plies of every G3 game (its 220 match games and its smoke pair): G3 started from the empty board,
    so these are recorded moves, not supplied openings (card §4.2)."""
    return [json.loads(p.read_text())["moves"][:OPENING_PLIES] for p in _game_files(root, G3_GAMES)]


def build_openings(root: pathlib.Path = ROOT, n_gen: int = G_MAX, n_smoke: int = SMOKE_GAMES, n_match: int = 272,
                   n_match_smoke: int = 2) -> Dict[str, Any]:
    """Every opening list H-OP will use, frozen at build time (card §4.2, §4.4, §10). Exclusions, cumulative:
    generation <- benchmark + G3; generation smoke <- + generation; match <- + generation smoke; match smoke <- + match."""
    bench, g3 = bench_openings(), g3_openings(root)
    excl = [*bench, *g3]
    gen = opening_list(NS, "gen opening", n_gen, excl)
    excl += [o["moves"] for o in gen]
    smoke = opening_list(NS, "gen smoke opening", n_smoke, excl)
    excl += [o["moves"] for o in smoke]
    match = opening_list(NS, "match opening", n_match, excl)
    excl += [o["moves"] for o in match]
    match_smoke = opening_list(NS, "match smoke opening", n_match_smoke, excl)
    return {"kind": "htb-onpolicy-openings/1", "card_sha256": CARD_SHA256, "namespace": NS,
            "rule": "teacher_bench openingAt: rows and columns 4-19, sorted legal moves, sha256[:12] mod n; distinct "
                    "directly or reflected from every earlier list and from the exclusions",
            "excluded": {"benchmark_openings": len(bench), "g3_openings": len(g3),
                         "benchmark_sha256": _jsha(bench), "g3_sha256": _jsha(g3)},
            "gen": gen, "gen_smoke": smoke, "match": match, "match_smoke": match_smoke}


# ─────────────────────────────── seeds and arms (card §4.2, §4.4) ───────────────────────────────

def move_seeds(path: str, game: int, ply: int) -> Dict[str, int]:
    """Two 48-bit seeds, hex digits 0-11 (search) and 12-23 (readout), as in G3 and the benchmark.
    `path` is "gen" or "gen smoke"."""
    h = _sha(f"{NS}/{path}/game {game}/ply {ply}".encode())
    return {"search": int(h[:12], 16), "readout": int(h[12:24], 16)}


def run_arm(i: int) -> str:
    """The full run's split, fixed before play: game i is validation iff i mod 10 = 9 (card §4.2)."""
    return "val" if i % 10 == 9 else "train"


def smoke_arm(j: int) -> str:
    """Revision 3: the smoke's 40 games alternate, j even training-like and j odd validation-like (20/20)."""
    return "val" if j % 2 == 1 else "train"


def n_tr(g: int) -> int:
    return g - g // 10


def n_val(g: int) -> int:
    return g // 10


# ─────────────────────────────── keys and exclusion sets (card §5) ───────────────────────────────

def replay(moves: Sequence[Sequence[int]]) -> List[TwixtState]:
    """States at plies 0..n through our rules engine; an illegal move STOPs."""
    s, out = TwixtState(), []
    out.append(s)
    for k, (r, c) in enumerate(moves):
        if (int(r), int(c)) not in s.legal_moves():
            raise Stop("replay", f"ply {k}: {(r, c)} is illegal in our engine")
        s = s.apply_move((int(r), int(c)))
        out.append(s)
    return out


def searched_keys(moves: Sequence[Sequence[int]]) -> List[str]:
    """The keys of the positions before plies 4..n-1: the positions `hard` searched, and the only pool candidates."""
    states = replay(moves)
    return [state_key(states[k]) for k in range(OPENING_PLIES, len(moves))]


@dataclasses.dataclass
class Exclusions:
    """E1-E4 of card §5.3. E1-E3 apply to both pools; E4 to validation only."""
    e1: Set[str]
    e2: Set[str]
    e3: Set[str]
    e4: Set[str]
    sources: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def train_excluded(self, k: str) -> bool:
        return k in self.e1 or k in self.e2 or k in self.e3

    def val_excluded(self, k: str) -> bool:
        return self.train_excluded(k) or k in self.e4

    @classmethod
    def empty(cls) -> "Exclusions":
        return cls(set(), set(), set(), set())


def _all_keys(moves) -> Set[str]:
    return {state_key(s) for s in replay(moves)}


def build_exclusions(root: pathlib.Path = ROOT) -> Exclusions:
    """From committed evidence only. E1: every position (plies 0..n) of the benchmark's 620 match and 4 smoke games;
    E2: every position of G3's 220 match games and its smoke pair; E3: Diagnostic O's 1,100 positions; E4: the first
    fit's 264,000 selected rows."""
    def from_games(dirs):
        keys, files = set(), _game_files(root, dirs)
        for p in files:
            keys |= _all_keys(json.loads(p.read_text())["moves"])
        return keys, len(files)

    e1, n1 = from_games(BENCH_GAMES)
    e2, n2 = from_games(G3_GAMES)
    prefixes = json.loads((root / DIAG_O).read_text())
    e3 = {state_key(replay(p["prefix"])[-1]) for p in prefixes}
    sel = json.loads((root / FIRST_SELECTION).read_text())
    e4 = {r["key"] for arm in ("train", "val") for r in sel["rows"][arm]}
    if (n1, n2, len(prefixes), len(e4)) != (624, 222, 1100, 264_000):
        raise Refused(f"exclusion sources are not the card's: games {n1}/{n2}, positions {len(prefixes)}, rows {len(e4)}")
    return Exclusions(e1, e2, e3, e4, {"games": {"E1": n1, "E2": n2}, "E3_positions": len(prefixes), "E4_rows": len(e4),
                                       "keys": {"E1": len(e1), "E2": len(e2), "E3": len(e3), "E4": len(e4)}})


class PoolCounter:
    """The run's stopping counter (card §4.3), counting as final selection counts (§5.2): training keys net of E1-E3;
    validation keys net of E1-E4 and of every key found in ANY training game generated so far."""

    def __init__(self, ex: Exclusions):
        self.ex = ex
        self.train_raw: Set[str] = set()       # every training-game position, excluded or not
        self.train: Set[str] = set()           # counted training keys
        self.val_all: Set[str] = set()         # validation keys net of E1-E4 (not yet net of training)
        self.val = 0                           # |val_all - train_raw|

    def add(self, arm: str, keys: Sequence[str]) -> Dict[str, int]:
        if arm == "train":
            for k in keys:
                if k in self.train_raw:
                    continue
                self.train_raw.add(k)
                if k in self.val_all:
                    self.val -= 1
                if not self.ex.train_excluded(k):
                    self.train.add(k)
        elif arm == "val":
            for k in keys:
                if k in self.val_all or self.ex.val_excluded(k):
                    continue
                self.val_all.add(k)
                if k not in self.train_raw:
                    self.val += 1
        else:
            raise Stop("arm", f"unknown arm {arm!r}")
        return self.counts()

    def counts(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": self.val}

    def reached(self) -> bool:
        c = self.counts()
        return c["train"] >= STOP["train"] and c["val"] >= STOP["val"]


# ─────────────────────────────── the smoke's yield verdict (card §4.4, frozen) ───────────────────────────────

def smoke_rows(games: Sequence[Mapping[str, Any]], ex: Exclusions) -> List[Dict[str, Any]]:
    """Per smoke game: x (searched plies) and y (usable rows), counted as selection counts (card §4.4).
    `games` = [{"index", "arm", "keys"}] where keys are the searched positions' keys in ply order."""
    train_raw = {k for g in games if g["arm"] == "train" for k in g["keys"]}
    seen = {"train": set(), "val": set()}
    out = []
    for g in sorted(games, key=lambda g: g["index"]):
        arm, y = g["arm"], 0
        for k in g["keys"]:
            if k in seen[arm]:
                continue
            if arm == "train":
                if ex.train_excluded(k):
                    continue
            elif ex.val_excluded(k) or k in train_raw:
                continue
            seen[arm].add(k)
            y += 1
        out.append({"index": g["index"], "arm": arm, "x": len(g["keys"]), "y": y})
    return out


def games_for(pool: str, target: float, u: float) -> float:
    """The smallest G whose first G games, split 9:1 as the run splits them, hold `target` rows of `pool` at `u` rows
    per game. math.inf when u <= 0."""
    if not u > 0:
        return math.inf
    k = math.ceil(target / u)
    if pool == "val":
        return 10 * k
    g = k + (k - 1) // 9                       # smallest G with G - floor(G / 10) >= k
    assert n_tr(g) >= k and (g == 0 or n_tr(g - 1) < k)
    return g


def yield_verdict(rows: Sequence[Mapping[str, Any]], t0: float, w: float, complete: bool,
                  ) -> Dict[str, Any]:
    """Y1-Y5 of card §4.4 from per-game rows (any pool sizes: the 36/4 formula test and the 20/20 smoke both use it).
    `complete`: every smoke game finished with one record per searched ply."""
    pools = {a: [r for r in rows if r["arm"] == a] for a in ("train", "val")}
    if not pools["train"] or not pools["val"]:
        raise Stop("yield", "a pool has no smoke games")
    trunc = [min(r["y"], TRUNC) for r in rows]
    sigma = statistics.stdev(trunc) if len(trunc) > 1 else 0.0
    est: Dict[str, Dict[str, float]] = {}
    for a, rs in pools.items():
        n = len(rs)
        u = sum(r["y"] for r in rs) / n
        lb = sum(min(r["y"], TRUNC) for r in rs) / n - Z_ONE_SIDED * sigma / math.sqrt(n)
        x = sum(r["x"] for r in rs)
        est[a] = {"games": n, "u": u, "u_lb": lb, "f": (sum(r["y"] for r in rs) / x) if x else 0.0,
                  "searched_plies": x}
    stop = {a: games_for(a, STOP[a], est[a]["u"]) for a in est}
    exact = {a: games_for(a, {"train": N_TRAIN, "val": N_VAL}[a], est[a]["u_lb"]) for a in est}
    g_stop, g_n = max(stop.values()), max(exact.values())
    by = lambda d: max(d, key=lambda a: (d[a], a == "val"))     # noqa: E731 - the pool that sets the count
    T = t0 + g_stop * w
    total_x = sum(r["x"] for r in rows)
    y = {"Y1": bool(complete) and total_x >= MIN_SMOKE_PLIES,
         "Y2": est["train"]["f"] >= F_MIN and est["val"]["f"] >= F_MIN,
         "Y3": g_n <= G_MAX,
         "Y4": g_stop <= G_MAX,
         "Y5": 1.5 * T <= GEN_CAP_S}
    failed = [k for k, v in y.items() if not v]
    return {"kind": "htb-onpolicy-yield/1", "card_sha256": CARD_SHA256,
            "pools": est, "sigma": sigma, "truncate_at": TRUNC, "z": Z_ONE_SIDED,
            "games_to_stop_targets": {"by_pool": stop, "G_stop": g_stop, "set_by": by(stop)},
            "games_to_exact_counts": {"by_pool": exact, "G_N": g_n, "set_by": by(exact)},
            "t0": t0, "w": w, "T": T, "launch_estimate": 1.5 * T, "cap_s": GEN_CAP_S, "g_max": G_MAX,
            "smoke_searched_plies": total_x, "complete": bool(complete),
            "conditions": y, "failed": failed, "verdict": "LAUNCH_OK" if not failed else "REVIEW"}


def _json_inf(o):
    return "inf" if isinstance(o, float) and math.isinf(o) else o


def dump_yield(v: Mapping[str, Any]) -> bytes:
    def fix(x):
        if isinstance(x, dict):
            return {k: fix(val) for k, val in x.items()}
        if isinstance(x, list):
            return [fix(val) for val in x]
        return _json_inf(x)
    return (json.dumps(fix(v), sort_keys=True, indent=1) + "\n").encode()


# ─────────────────────────────── selection (card §5.2) ───────────────────────────────

def game_id(i: int) -> str:
    return f"gen_{i:04d}"


def _order_key(arm: str, key: str) -> str:
    return _sha(f"{SELECT_TAG}|{arm}|{key}".encode())


def select(games: Sequence[Mapping[str, Any]], ex: Exclusions, n_train: int = N_TRAIN, n_val: int = N_VAL
           ) -> Dict[str, Any]:
    """Card §5.2 over the generation's games [{"index", "arm", "moves"}]: each game's arm must be the run's rule.
    Pools by first occurrence in (game, ply) order; validation keys net of every training-game position."""
    first: Dict[str, Dict[str, Tuple[str, int]]] = {"train": {}, "val": {}}
    train_raw: Set[str] = set()
    hits = {"train": {"E1-E3": 0}, "val": {"E1-E3": 0, "E4": 0, "in_training_games": 0}}
    order = sorted(games, key=lambda g: g["index"])
    if [g["index"] for g in order] != list(range(len(order))):
        raise Stop("selection", "the generation's game indices are not 0..n-1")
    keyed = []
    for g in order:
        if g["arm"] != run_arm(g["index"]):
            raise Stop("selection", f"game {g['index']} is recorded as {g['arm']}, the rule says {run_arm(g['index'])}")
        keys = searched_keys(g["moves"])
        keyed.append((g, keys))
        if g["arm"] == "train":
            train_raw.update(keys)
    for g, keys in keyed:
        arm = g["arm"]
        for j, k in enumerate(keys):
            if k in first[arm]:
                continue
            if ex.train_excluded(k):
                hits[arm]["E1-E3"] += 1
                continue
            if arm == "val":
                if k in ex.e4:
                    hits["val"]["E4"] += 1
                    continue
                if k in train_raw:
                    hits["val"]["in_training_games"] += 1
                    continue
            first[arm][k] = (game_id(g["index"]), OPENING_PLIES + j)
    out: Dict[str, Any] = {"kind": "htb-onpolicy-selection/1", "card_sha256": CARD_SHA256,
                           "games": {a: sum(g["arm"] == a for g in order) for a in ("train", "val")},
                           "pool": {a: len(first[a]) for a in first}, "exclusion_hits": hits, "rows": {}}
    for arm, n in (("train", n_train), ("val", n_val)):
        if len(first[arm]) < n:
            raise Refused(f"the {arm} pool holds {len(first[arm])} distinct positions, fewer than {n}; no fallback")
        chosen = sorted(first[arm], key=lambda k: _order_key(arm, k))[:n]
        out["rows"][arm] = [{"game_id": first[arm][k][0], "ply": first[arm][k][1], "key": k} for k in chosen]
    return out


# ─────────────────────────────── gates, preconditions, admission ───────────────────────────────

GATES = {"gen-smoke": ("htb_onpolicy_gen_smoke_authorization", "HTB_ONPOLICY_GEN_SMOKE_AUTHORIZED"),
         "generate": ("htb_onpolicy_generate_authorization", "HTB_ONPOLICY_GENERATE_AUTHORIZED"),
         "label": ("htb_onpolicy_label_authorization", "HTB_ONPOLICY_LABEL_AUTHORIZED"),
         "fit": ("htb_onpolicy_fit_authorization", "HTB_ONPOLICY_FIT_AUTHORIZED"),
         "export": ("htb_onpolicy_export_authorization", "HTB_ONPOLICY_EXPORT_AUTHORIZED")}
VERSIONS = {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.10"}       # the research runtime (card §3)
GIB = P.GIB
FIT_MEMORY_REQUIRED = P.FIT_MEMORY_REQUIRED                             # 12 GiB, the first fit's (card §12)
LABEL_MEMORY_REQUIRED = P.LABEL_MEMORY_REQUIRED * (N_TRAIN + N_VAL) // (P.N_TRAIN + P.N_VAL)   # scaled: 5.5 GiB
OUTER_OK = (("inner_exit", 0), ("outer_exit", 0), ("outcome", "COMPLETED"), ("gate_restored", True),
            ("survivors", ""), ("record_by", "script"))


def gate_open(stage: str) -> bool:
    import importlib

    mod_name, name = GATES[stage]
    mod = importlib.import_module(f"{__package__}.{mod_name}")
    return getattr(mod, name) is True


def committed(root: pathlib.Path, rel: str) -> bool:
    return P._committed_git(root, rel)


def head_of(root: pathlib.Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


@dataclasses.dataclass
class Config:
    """Everything an entry touches. The defaults ARE the card; tests substitute fakes and smaller counts."""
    root: pathlib.Path = ROOT
    data_root: pathlib.Path = dataclasses.field(default_factory=P.main_checkout)   # the first fit's corpus
    cache: pathlib.Path = CACHE
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    recipe: P.Recipe = RECIPE
    start_rows: int = START_ROWS
    shard_rows: int = P.SHARD_ROWS
    committed: Callable[[pathlib.Path, str], bool] = committed
    head: Callable[[pathlib.Path], str] = head_of
    gate_open: Callable[[str], bool] = gate_open
    exclusions: Callable[[pathlib.Path], Exclusions] = build_exclusions
    tf32_required: bool = True
    memory_required_bytes: Optional[int] = FIT_MEMORY_REQUIRED
    label_memory_required_bytes: Optional[int] = LABEL_MEMORY_REQUIRED
    available_memory: Callable[[], Dict[str, int]] = lambda: P.available_memory()

    def ev(self, stage: str) -> pathlib.Path:
        return self.root / EV[stage]

    def labels(self, arm: str) -> pathlib.Path:
        return self.cache / f"labels-{arm}"

    @property
    def challenger(self) -> pathlib.Path:
        return self.cache / "challenger.safetensors"


def preconditions(stage: str, cfg: Config) -> None:
    """Refuse before anything heavy: the gate, then the card's pin and commit, then the environment."""
    if cfg.gate_open(stage) is not True:
        raise Refused(f"{GATES[stage][1]} is closed")
    if _sha((cfg.root / CARD).read_bytes()) != CARD_SHA256:
        raise Refused("the card is not the pinned one")
    if not cfg.committed(cfg.root, CARD):
        raise Refused("the card is not committed")
    if cfg.tf32_required and os.environ.get("MLX_ENABLE_TF32") != "0":
        raise Refused("MLX_ENABLE_TF32 must be 0 in this process")


def read_committed_json(cfg: Config, rel: str) -> Dict[str, Any]:
    if not (cfg.root / rel).is_file() or not cfg.committed(cfg.root, rel):
        raise Refused(f"{rel} is absent, uncommitted or changed")
    return json.loads((cfg.root / rel).read_text())


def stage_admission(cfg: Config, stage: str) -> Dict[str, Any]:
    """A completed earlier stage: its record COMPLETED and committed, its wrapper's outer.json clean and committed,
    and the two bound to one run (the same launch commit; pgid.txt is the recorded pid)."""
    rec = read_committed_json(cfg, f"{EV[stage]}/record.json")
    outer = read_committed_json(cfg, f"{OUTER[stage]}/outer.json")
    bad = [k for k, v in OUTER_OK if outer.get(k) != v]
    if rec.get("status") != "COMPLETED" or rec.get("card_sha256") != CARD_SHA256:
        raise Refused(f"{EV[stage]}/record.json is not a COMPLETED record of this card")
    if bad:
        raise Refused(f"{OUTER[stage]}/outer.json is not clean: {bad}")
    pgid_rel = f"{OUTER[stage]}/pgid.txt"
    if not cfg.committed(cfg.root, pgid_rel) or (cfg.root / pgid_rel).read_text().strip() != str(rec.get("pid")):
        raise Refused(f"{pgid_rel} is not the recorded pid {rec.get('pid')}")
    if outer.get("head") != rec.get("head"):
        raise Refused(f"the {stage} record and its outer.json name different launch commits")
    return {"record_sha256": _sha((cfg.root / EV[stage] / "record.json").read_bytes()), "head": rec.get("head")}


def _record(stage: str, cfg: Config, **extra) -> Dict[str, Any]:
    return {"kind": f"htb-onpolicy-{stage}-record/1", "card_sha256": CARD_SHA256, "stage": stage,
            "head": cfg.head(cfg.root), "pid": os.getpid(),
            "module_sha256": _sha(pathlib.Path(__file__).read_bytes()), **extra}


# ─────────────────────────────── generation (card §4) ───────────────────────────────

def research_move(evaluator, state: TwixtState, seeds: Mapping[str, int], n_sims: int = SIMS) -> Dict[str, Any]:
    """One move of `hard`'s protocol in the research implementation (the pieces G1 ported to Node, built exactly as
    G1's fixtures build them): RootSequentialHalvingMCTS at 200 simulations, plain search through ply 19 with the
    T 1.0 readout on the readout seed, the halving choice from ply 20."""
    from . import eval_readout
    from . import root_sequential_halving as RSH
    from .eval_runner import EvalConfig, cfg_from, readout_from_eval_config
    from .external_learning_stage3 import PROTOCOL

    config = EvalConfig(**PROTOCOL)
    m = RSH.RootSequentialHalvingMCTS(evaluator, cfg_from(config, n_sims), random.Random(seeds["search"]))
    counts, _root_value, _root = m.search_with_root(state, add_noise=False)
    if m.last_choice is not None:
        move, halving = m.last_choice, True
    else:
        move, _ = eval_readout.select(counts, state.ply, readout_from_eval_config(config), random.Random(seeds["readout"]))
        halving = False
    return {"move": (int(move[0]), int(move[1])), "visits": int(sum(counts.values())), "n_sims": n_sims,
            "halving": halving}


@dataclasses.dataclass
class GenSeams:
    """What generation reads from outside this module. TEST seams; the entries pass `_real_gen_seams` only."""
    versions: Callable[[], Dict[str, str]]
    student_sha256: Callable[[], str]
    load: Callable[[], Any]                              # the student as a base evaluator (has .network)
    digest: Callable[[Any], str]                         # a network -> its state digest
    require_eval: Callable[[Any, str], Any]
    background: Callable[[], Dict[str, Any]]
    wrap: Callable[[Any], Any]                           # base -> MirrorAveragingEvaluator(base)
    move: Callable[[Any, TwixtState, Mapping[str, int]], Dict[str, Any]] = research_move
    clock: Callable[[], float] = time.monotonic


def _real_gen_seams() -> GenSeams:   # pragma: no cover - loads the student on the GPU; never under the build approval
    from . import background_compute as BG
    from . import combination_policy_pilot as CPP
    from . import eval_runner as ER
    from . import search_distillation_d2_measure as M
    from .mirror_averaging import MirrorAveragingEvaluator

    path = pathlib.Path(STUDENT["path"])
    return GenSeams(versions=CPP._versions, student_sha256=lambda: _sha(path.read_bytes()),
                    load=lambda: ER._default_evaluator_factory(str(path)), digest=P._digest,
                    require_eval=lambda ev, when: M._require_eval(ev.network, when), background=BG.observe,
                    wrap=MirrorAveragingEvaluator)


def natural_terminal(s: TwixtState) -> bool:
    """The rules alone (the product's end rule): a winner, or no legal move. No ply cap."""
    return s.winner() is not None or not s.legal_moves()


def _watch(seams: GenSeams, rec: Dict[str, Any], log, when: str) -> None:
    obs = seams.background()
    rec["background_checks"] = rec.get("background_checks", 0) + 1
    log("bg_check", when=when, ok=obs.get("ok") is True, hits=obs.get("hits"), error=obs.get("error"))
    if obs.get("ok") is not True:
        raise Stop("background_compute", f"{when}: " + (f"the check failed: {obs.get('error')}" if obs.get("error")
                                                        else f"BOINC detected: {obs.get('hits')}"))


def play_game(seams: GenSeams, ev, path: str, i: int, opening: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """One self-play game from its supplied opening to the product's end, both sides the student under `hard`'s
    protocol. Returns the game with its searched positions' keys; every search spends exactly 200 visits."""
    states = replay(opening)
    s, moves, keys, rows = states[-1], [list(m) for m in opening], [], []
    if natural_terminal(s):
        raise Stop("opening", f"game {i}: the opening ends the game")
    while not natural_terminal(s):
        seeds = move_seeds(path, i, s.ply)
        t = seams.clock()
        r = seams.move(ev, s, seeds)
        dt = seams.clock() - t
        if r.get("n_sims") != SIMS or r.get("visits") != SIMS:
            raise Stop("budget", f"game {i} ply {s.ply}: {r.get('n_sims')} simulations budgeted, {r.get('visits')} "
                                 f"visits spent; both must be {SIMS}")
        mv = (int(r["move"][0]), int(r["move"][1]))
        if mv not in s.legal_moves():
            raise Stop("illegal", f"game {i} ply {s.ply}: {mv} is illegal")
        keys.append(state_key(s))
        rows.append({"ply": s.ply, "seconds": round(dt, 4), "visits": r["visits"], "halving": bool(r["halving"])})
        s = s.apply_move(mv)
        moves.append(list(mv))
    return {"moves": moves, "keys": keys, "per_move": rows, "winner": s.winner(),
            "termination": "connection" if s.winner() is not None else "board_full"}


def generate(stage: str, seams: GenSeams, out_dir: pathlib.Path, rec: Dict[str, Any], log, ex: Exclusions,
             openings: Sequence[Mapping[str, Any]], t_start: float) -> None:
    """Card §4: the generation loop -- play, then count keys, then write the game's records -- for the smoke (40
    games, 20/20 alternating) and for the run (9:1, stopping on position counts at 1.15 x N, or at G_MAX)."""
    smoke = stage == "gen-smoke"
    path, arm_of = ("gen smoke", smoke_arm) if smoke else ("gen", run_arm)
    limit = SMOKE_GAMES if smoke else G_MAX
    if len(openings) < limit:
        raise Stop("openings", f"{len(openings)} openings for {limit} games")
    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Stop("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    sha = seams.student_sha256()
    if sha != STUDENT["sha256"]:
        raise Stop("student", f"the student file's sha256 {sha} is not the pinned one")
    _watch(seams, rec, log, "before load")
    base = seams.load()
    for when in ("after load",):
        d = seams.digest(base.network)
        if d != STUDENT["state_digest"]:
            raise Stop("student", f"{when}: state digest {d} is not the student's")
        seams.require_eval(base, when)
    ev = seams.wrap(base)
    counter = PoolCounter(ex)
    rec["student"] = {"path": str(STUDENT["path"]), "sha256": sha, "state_digest": STUDENT["state_digest"]}
    rec["stopped_by"] = None
    t_loop = seams.clock()
    rec["startup_s"] = round(t_loop - t_start, 3)
    n = 0
    with open(out_dir / "games.jsonl", "x") as fh:
        for i in range(limit):
            log("game_start", game=i)
            g = play_game(seams, ev, path, i, openings[i]["moves"])
            arm = arm_of(i)
            counts = counter.add(arm, g.pop("keys"))
            row = {"index": i, "arm": arm, "opening_index": openings[i]["index"], "plies": len(g["moves"]),
                   "searched": len(g["per_move"]), "counts_after": counts, **g}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n += 1
            log("game_end", game=i, plies=row["plies"], counts=counts)
            _watch(seams, rec, log, f"after game {i}")
            if not smoke and counter.reached():
                rec["stopped_by"] = "targets"
                break
    rec["loop_seconds"] = round(seams.clock() - t_loop, 3)
    rec["games"] = n
    rec["w"] = rec["loop_seconds"] / n
    rec["stopped_by"] = rec["stopped_by"] or ("smoke" if smoke else "g_max")
    rec["counts"] = counter.counts()
    d = seams.digest(base.network)
    if d != STUDENT["state_digest"]:
        raise Stop("student", f"at the end: state digest {d} is not the student's")
    seams.require_eval(base, "at the end")


def read_games(path: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_games(games: Sequence[Mapping[str, Any]], openings: Sequence[Mapping[str, Any]], arm_of) -> List[str]:
    """Replay-validate recorded games: index order, the arm rule, the supplied opening, legal moves, one per-move row
    per searched ply with 200 visits, and the product's end. Returns the faults (empty when every game is valid)."""
    faults = []
    for j, g in enumerate(games):
        if g.get("index") != j or g.get("arm") != arm_of(j):
            faults.append(f"game {j}: index or arm is not the rule's")
            continue
        if g["moves"][:OPENING_PLIES] != openings[j]["moves"]:
            faults.append(f"game {j}: not its frozen opening")
            continue
        try:
            states = replay(g["moves"])
        except Stop as e:
            faults.append(f"game {j}: {e.detail}")
            continue
        searched = len(g["moves"]) - OPENING_PLIES
        if [r.get("ply") for r in g["per_move"]] != list(range(OPENING_PLIES, len(g["moves"]))) or \
                any(r.get("visits") != SIMS for r in g["per_move"]) or g.get("searched") != searched:
            faults.append(f"game {j}: per-move rows are not one per searched ply with {SIMS} visits")
        if not natural_terminal(states[-1]) or any(natural_terminal(s) for s in states[:-1]):
            faults.append(f"game {j}: does not end exactly at the product's end")
    return faults


def derive_yield(cfg: Config, ex: Optional[Exclusions] = None) -> Tuple[Dict[str, Any], bytes]:
    """The smoke's yield verdict (card §4.4) from its committed evidence: the record, the games and the frozen
    openings. Outcomes are never read."""
    rec = read_committed_json(cfg, f"{EV['gen-smoke']}/record.json")
    games_rel = f"{EV['gen-smoke']}/games.jsonl"
    if not cfg.committed(cfg.root, games_rel):
        raise Refused(f"{games_rel} is absent, uncommitted or changed")
    openings = load_openings(cfg)["gen_smoke"]
    games = read_games(cfg.root / games_rel)
    faults = check_games(games, openings, smoke_arm)
    complete = rec.get("status") == "COMPLETED" and len(games) == SMOKE_GAMES and not faults
    ex = ex if ex is not None else cfg.exclusions(cfg.root)
    rows = smoke_rows([{"index": g["index"], "arm": g["arm"], "keys": searched_keys(g["moves"])} for g in games], ex)
    v = yield_verdict(rows, float(rec.get("startup_s", math.inf)), float(rec.get("w", math.inf)), complete)
    v["faults"] = faults
    v["smoke_record_sha256"] = _sha((cfg.root / EV["gen-smoke"] / "record.json").read_bytes())
    v["exclusions"] = ex.sources
    return v, dump_yield(v)


def load_openings(cfg: Config) -> Dict[str, Any]:
    raw = (cfg.root / OPENINGS_REL).read_bytes()
    if _sha(raw) != OPENINGS_SHA256 or not cfg.committed(cfg.root, OPENINGS_REL):
        raise Refused(f"{OPENINGS_REL} is not the committed, pinned opening list")
    return json.loads(raw)


OPENINGS_SHA256 = "ca4e3bba05bfc6abddf782b1a1f68fc474bc3600410c7cbd4308073d306d3915"


def generation_admission(cfg: Config, ex: Exclusions) -> Dict[str, Any]:
    """The run admits only a committed, clean smoke whose yield verdict re-derives byte-equal and is LAUNCH_OK."""
    adm = stage_admission(cfg, "gen-smoke")
    committed_yield = read_committed_json(cfg, YIELD_REL)
    _v, raw = derive_yield(cfg, ex)
    if raw != (cfg.root / YIELD_REL).read_bytes():
        raise Refused(f"{YIELD_REL} does not re-derive byte-equal from the committed smoke")
    if committed_yield.get("verdict") != "LAUNCH_OK":
        raise Refused(f"the smoke's yield verdict is {committed_yield.get('verdict')}, failed "
                      f"{committed_yield.get('failed')}: no launch (card §4.4)")
    return {**adm, "yield_sha256": _sha(raw)}


def run_gen(stage: str, cfg: Optional[Config] = None, seams: Optional[GenSeams] = None,
            t_start: Optional[float] = None) -> int:
    cfg = cfg or Config()
    t_start = time.monotonic() if t_start is None else t_start
    preconditions(stage, cfg)
    out_dir = cfg.ev(stage)
    P._absent(out_dir)
    openings = load_openings(cfg)["gen_smoke" if stage == "gen-smoke" else "gen"]
    ex = cfg.exclusions(cfg.root)
    admission = generation_admission(cfg, ex) if stage == "generate" else {}
    rec = _record(stage, cfg, admission=admission, openings_sha256=OPENINGS_SHA256, exclusions=ex.sources,
                  caps={"g_max": G_MAX, "smoke_games": SMOKE_GAMES,
                        "kill_s": GEN_CAP_S if stage == "generate" else SMOKE_CAP_S})
    log_path = out_dir / "progress.jsonl"

    def log(event, **kw):
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "event": event, **kw}) + "\n")

    def body(rec):
        generate(stage, seams or _real_gen_seams(), out_dir, rec, log, ex, openings, t_start)

    return P._terminal(out_dir, rec, body)


# ─────────────────────────────── labels (card §6) ───────────────────────────────

class OnpolicyTeacher(P.TwixtbotTeacher):
    """The first fit's teacher, unchanged, with the benchmark's lockstep check: every field compared, the adapter's
    cap-free `terminal` line left out, rules-level terminal status compared directly; legal sets equal."""

    def divergences(self, our: TwixtState, tb) -> List[str]:
        from . import htb_bench_teacher as B
        from . import twixtbot_adapter as A

        out = B.bench_divergences(our, tb, self.rt["twixt"])
        if A.tb_legal_moves(tb) != set(our.legal_moves()):
            out.append("legal sets differ")
        return out


def _load_teacher_real(clone_root: str):   # pragma: no cover - needs TensorFlow; never under the build approval
    from . import external_learning as EL
    from . import htb_bench_teacher as B

    bad = B.env_mismatches(B.runtime_versions())
    if bad:
        raise Refused(f"the teacher runtime is not the card's: {bad}")
    rt, identity = EL.load_runtime(clone_root)
    return OnpolicyTeacher(rt), identity


def gen_games(cfg: Config) -> List[Dict[str, Any]]:
    rel = f"{EV['generate']}/games.jsonl"
    if not cfg.committed(cfg.root, rel):
        raise Refused(f"{rel} is absent, uncommitted or changed")
    return read_games(cfg.root / rel)


def run_label(clone_root: str, cfg: Optional[Config] = None, load_teacher: Callable = _load_teacher_real) -> int:
    cfg = cfg or Config()
    preconditions("label", cfg)
    P._absent(cfg.ev("label"), *(cfg.labels(a) for a in ("train", "val")))
    admission = stage_admission(cfg, "generate")
    memory = P._check_available(cfg, cfg.label_memory_required_bytes, "label run")
    rec = _record("label", cfg, admission=admission, teacher=P.TEACHER, counts={"train": cfg.n_train, "val": cfg.n_val},
                  memory_at_start=memory, label_memory_required_bytes=cfg.label_memory_required_bytes)

    def body(rec):
        games = gen_games(cfg)
        faults = check_games(games, load_openings(cfg)["gen"], run_arm)
        if faults:
            raise Stop("games", f"{len(faults)} invalid generation games, e.g. {faults[:3]}")
        sel = select(games, cfg.exclusions(cfg.root), n_train=cfg.n_train, n_val=cfg.n_val)
        rec["selection_sha256"] = P._write_create_only(cfg.ev("label") / "selection.json", sel)
        teacher, identity = load_teacher(clone_root)
        rec["runtime"] = identity
        moves = {game_id(g["index"]): g["moves"] for g in games}
        rec["labels"] = []
        for arm in ("train", "val"):
            by_game: Dict[str, List[Mapping[str, Any]]] = {}
            for r in sel["rows"][arm]:
                by_game.setdefault(r["game_id"], []).append(r)
            arrays = P.LabelArrays(sel["rows"][arm])
            for gid in sorted(by_game):
                for row in P.label_game(gid, moves[gid], by_game[gid], teacher):
                    arrays.put(row)
            manifest = P.write_label_arrays(arrays, cfg.labels(arm), cfg.shard_rows)
            del arrays
            rec["labels"].append({"arm": arm, "n": manifest["n"],
                                  "manifest_sha256": _sha((cfg.labels(arm) / "manifest.json").read_bytes())})
        rec["teacher_calls"] = getattr(teacher, "calls", None)

    return P._terminal(cfg.ev("label"), rec, body)


# ─────────────────────────────── the fit, the gate, retention (card §7, §8) ───────────────────────────────

def epoch_order(keys: Sequence[str], epoch: int, tag: str = FIT_TAG) -> List[int]:
    return sorted(range(len(keys)), key=lambda i: _sha(f"{tag}|{epoch}|{keys[i]}".encode()))


def start_check(reference, examples, n: int = START_ROWS, tag: str = FIT_TAG) -> Dict[str, float]:
    src = P._source(examples)
    first = P._Subset(src, epoch_order(src.keys, 0, tag)[:n])
    s = P.score(reference, first)
    if not s["kl"] >= START_KL_FLOOR:
        raise Stop("start", f"KL(T || reference_MA) {s['kl']} on the first {len(first)} rows is below {START_KL_FLOOR}")
    return s


def fit(student, reference, examples, recipe: P.Recipe = RECIPE, tag: str = FIT_TAG,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """`htb_pilot.fit`, line for line, with the epoch namespace as a parameter (card §7: `htb-onpolicy-fit/1`).
    htb_pilot.py is pinned by the H-TB screen, so it is not edited; a test checks this copy reproduces P.fit bitwise
    when given P's namespace."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from .trainer import clip_grad_norm, compute_masked_log_probs

    if student is reference:
        raise Stop("fit", "the student and the reference are the same object")
    for net, who in ((student, "the student"), (reference, "the reference")):
        P._require_eval(net, who)
    ref_digest = P._digest(reference)
    if P._digest(student) != ref_digest:
        raise Stop("fit", "the student does not start as an exact copy of the reference")
    src = P._source(examples)
    n = len(src)
    if n < recipe.rows_per_batch or n % recipe.rows_per_batch:
        raise Stop("fit", f"{n} rows do not divide into batches of {recipe.rows_per_batch}")
    names = P.surface_names(student)
    before = P._snapshot(student)
    opt = optim.Adam(learning_rate=recipe.learning_rate)
    resolved = {"betas": [float(b) for b in opt.betas], "eps": float(opt.eps), "bias_correction": bool(opt.bias_correction)}
    if resolved != P.ADAM_DEFAULTS:
        raise Stop("optimizer", f"MLX's Adam resolves to {resolved}, not {P.ADAM_DEFAULTS}")
    keys = src.keys
    if len(set(keys)) != n:
        raise Stop("fit", "duplicate keys among the training rows")

    def loss_fn(model, boards, rows, cols, mask, target, tv):
        logits, values, _ = model.forward_padded(boards, rows, cols, mask, active_size=P.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        value = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(model.trainable_parameters()))
        return policy + recipe.value_weight * value + recipe.l2_weight * l2, policy, value

    grad_fn = nn.value_and_grad(student, loss_fn)
    log, step = [], 0
    for epoch in range(recipe.epochs):
        order = epoch_order(keys, epoch, tag)
        for b in range(0, n, recipe.rows_per_batch):
            idx = order[b:b + recipe.rows_per_batch]
            recs, tv = [], []
            for e in src.take(idx):
                recs += [e["record"], e["mirror"]]
                tv += [e["value"]] * 2
            boards, rows, cols, mask, target = P._batch(recs)
            (loss, policy, value), grads = grad_fn(student, boards, rows, cols, mask, target,
                                                   mx.array(np.asarray(tv, dtype=np.float32)))
            grads, norm = clip_grad_norm(grads, recipe.clip_norm)
            opt.update(student, grads)
            mx.eval(student.parameters(), opt.state, loss, policy, value, norm)
            del recs, boards, rows, cols, mask, target, grads
            vals = (float(loss), float(policy), float(value), float(norm))
            if not all(np.isfinite(vals)):
                raise Stop("fit", f"update {step}: non-finite loss or gradient norm {vals}")
            if step % P.STEP_LOG_EVERY == 0:
                log.append({"update": step, "epoch": epoch, "loss": vals[0], "policy": vals[1], "value": vals[2],
                            "grad_norm": vals[3]})
                if on_step is not None:
                    on_step(log[-1])
            step += 1
    P._require_eval(student, "the student after the fit")
    if P._digest(reference) != ref_digest:
        raise Stop("fit", "the reference changed during the fit")
    after = P._snapshot(student)
    stats = sorted(k for k in before if k.endswith(("running_mean", "running_var")))
    moved_stats = [k for k in stats if not np.array_equal(before[k], after[k])]
    if moved_stats:
        raise Stop("batchnorm", f"running statistics changed: {moved_stats[:5]}")
    moved = [k for k in names if not np.array_equal(before[k], after[k])]
    groups = {g: any(k.startswith(g) for k in moved) for g in ("encoder.", "policy_head.", "value_head.")}
    if not all(groups.values()):
        raise Stop("surface", f"a part of the network never moved: {groups}")
    return {"recipe": dataclasses.asdict(recipe), "optimizer": {"name": "Adam", **resolved}, "updates": step,
            "epoch_tag": tag, "log": log, "surface_tensors": len(names), "surface_tensors_changed": len(moved),
            "running_statistics_unchanged": len(stats), "reference_digest": ref_digest,
            "student_digest": P._digest(student)}


GATE_MISSED = "STOP: the challenger missed the imitation gate"


def gate_verdict(challenger: Mapping[str, float], student: Mapping[str, float]) -> Dict[str, Any]:
    """I1-I3 of card §8 and every metric's movement against the current student (reported either way)."""
    i = {"I1": challenger["kl"] <= I1_FACTOR * student["kl"],
         "I2": challenger["top1"] > student["top1"],
         "I3": challenger["value_mse"] <= student["value_mse"]}
    lower_is_toward = {"kl": True, "top1": False, "value_mse": True}
    movement = {}
    for m, lower in lower_is_toward.items():
        s, c = float(student[m]), float(challenger[m])
        d = c - s
        direction = "unchanged" if d == 0 else ("toward the teacher" if (d < 0) == lower else "away from the teacher")
        movement[m] = {"S": s, "C": c, "C_minus_S": d, "C_over_S": (c / s) if s else None, "direction": direction}
    held = [k for k, v in i.items() if v]
    return {**i, "held": held, "failed": [k for k in i if k not in held], "movement": movement,
            "i1_factor": I1_FACTOR, "verdict": "ELIGIBLE" if len(held) == 3 else GATE_MISSED}


def check_label_binding(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The committed label evidence: COMPLETED, this card, this teacher, its own selection, the exact counts."""
    rels = [f"{EV['label']}/record.json", f"{EV['label']}/selection.json"]
    for rel in rels:
        if not (cfg.root / rel).is_file() or not cfg.committed(cfg.root, rel):
            raise Refused(f"{rel} is absent, uncommitted or changed")
    raw_sel = (cfg.root / rels[1]).read_bytes()
    label, sel = json.loads((cfg.root / rels[0]).read_text()), json.loads(raw_sel)
    want = {"kind": "htb-onpolicy-label-record/1", "status": "COMPLETED", "card_sha256": CARD_SHA256,
            "teacher": P.TEACHER, "counts": {"train": cfg.n_train, "val": cfg.n_val},
            "selection_sha256": _sha(raw_sel)}
    bad = [k for k, v in want.items() if label.get(k) != v]
    if bad:
        raise Refused(f"the label record does not bind: {bad}")
    if sel.get("kind") != "htb-onpolicy-selection/1" or sel.get("card_sha256") != CARD_SHA256:
        raise Refused("the selection is not this card's")
    n = {"train": cfg.n_train, "val": cfg.n_val}
    if sorted(sel.get("rows", {})) != ["train", "val"] or any(len(sel["rows"][a]) != n[a] for a in n):
        raise Refused("the selection does not hold the card's exact counts")
    keys = [r["key"] for a in n for r in sel["rows"][a]]
    if len(set(keys)) != len(keys):
        raise Refused("the selection repeats a key within or across arms")
    if [e.get("arm") for e in label.get("labels", [])] != ["train", "val"] or \
            any(e["n"] != n[e["arm"]] for e in label["labels"]):
        raise Refused("the label record's arms or counts are not the card's")
    return label, sel


def load_bound_labels(cfg: Config, label: Mapping[str, Any], sel: Mapping[str, Any]) -> Dict[str, Tuple]:
    out = {}
    for e in label["labels"]:
        rows = P.load_label_compact(cfg.labels(e["arm"]), e["manifest_sha256"])
        got = [(r["game_id"], int(r["ply"]), r["key"]) for r in rows[0]]
        want = [(r["game_id"], int(r["ply"]), r["key"]) for r in sel["rows"][e["arm"]]]
        if got != want:
            raise Refused(f"the {e['arm']} labels are not the selection's rows, in order")
        out[e["arm"]] = rows
    return out


def _load_student_real():   # pragma: no cover - loads the student; never under the build approval
    from .probe_eval import load_network_for_scoring

    path = pathlib.Path(STUDENT["path"])
    if _sha(path.read_bytes()) != STUDENT["sha256"]:
        raise Stop("student", "the student file is not the pinned one")
    net = load_network_for_scoring(str(path), verbose=False)[0]
    net.eval()
    if P._digest(net) != STUDENT["state_digest"]:
        raise Stop("student", "the loaded student's state digest is not the pinned one")
    return net


def _retention_real(cfg: Config):   # pragma: no cover - reads the first fit's labels and the corpus
    """The first fit's 24,000 validation rows and the move lists of their games, for the descriptive check."""
    first = json.loads((cfg.root / FIRST_LABEL_RECORD).read_text())
    msha = next(e["manifest_sha256"] for e in first["labels"] if e["arm"] == "val")
    meta, target, value = P.load_label_compact(P.CACHE / "labels-val", msha)
    games = {g["id"]: g["moves"] for g in P.eligible_games(cfg.data_root / P.CORPUS, arms=("val",))}
    return P.ExampleStore(meta, target, value, games)


def run_fit(cfg: Optional[Config] = None, load_student: Callable[[], Any] = _load_student_real,
            retention: Callable[[Config], Any] = _retention_real,
            save: Callable[[Any, pathlib.Path], Dict[str, str]] = P._save_student_real) -> int:
    cfg = cfg or Config()
    preconditions("fit", cfg)
    P._absent(cfg.ev("fit"), cfg.challenger)
    label, sel = check_label_binding(cfg)
    memory = P._check_available(cfg, cfg.memory_required_bytes, "fit")
    rows = load_bound_labels(cfg, label, sel)
    rec = _record("fit", cfg, label_record_sha256=_sha((cfg.root / EV["label"] / "record.json").read_bytes()),
                  selection_sha256=label["selection_sha256"], recipe=dataclasses.asdict(cfg.recipe),
                  epoch_tag=FIT_TAG, memory_at_start=memory, memory_required_bytes=cfg.memory_required_bytes)

    def body(rec):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        moves = {game_id(g["index"]): g["moves"] for g in gen_games(cfg)}
        arms = {a: P.ExampleStore(*rows[a], moves) for a in ("train", "val")}
        reference, student = load_student(), load_student()
        rec["start"] = start_check(reference, arms["train"], n=cfg.start_rows)
        rec["student_val"] = P.score(reference, arms["val"])            # S: the first measurement on these positions
        rec["fit"] = fit(student, reference, arms["train"], cfg.recipe)
        rec["challenger_val"] = P.score(student, arms["val"])           # C
        rec["gate"] = gate_verdict(rec["challenger_val"], rec["student_val"])
        rec["verdict"] = rec["gate"]["verdict"]
        old = retention(cfg)                                             # descriptive only, never gated
        rec["retention"] = {"note": "descriptive only (card §8): the first fit's 24,000 validation rows",
                            "student": P.score(reference, old), "challenger": P.score(student, old)}
        rec["challenger"] = save(student, cfg.challenger)

    return P._terminal(cfg.ev("fit"), rec, body)


# ─────────────────────────────── CLI ───────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    t_start = time.monotonic()
    ap = argparse.ArgumentParser(prog="htb_onpolicy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("what", choices=("gen-smoke", "generate", "label", "fit"))
    run.add_argument("--clone-root")
    sub.add_parser("yield")
    sub.add_parser("openings")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "openings":                       # pure: re-derive and compare with the committed list
            raw = (json.dumps(build_openings(), sort_keys=True, indent=1) + "\n").encode()
            same = raw == (ROOT / OPENINGS_REL).read_bytes()
            print(f"{OPENINGS_REL}: {'re-derives byte-equal' if same else 'DIFFERS'} ({_sha(raw)})")
            return 0 if same else 3
        if a.cmd == "yield":                          # pure: write the verdict once, or confirm it re-derives
            cfg = Config()
            v, raw = derive_yield(cfg)
            path = ROOT / YIELD_REL
            if path.exists():
                same = path.read_bytes() == raw
                print(f"{YIELD_REL}: {'re-derives byte-equal' if same else 'DIFFERS'}; verdict {v['verdict']}")
                return 0 if same else 3
            with open(path, "xb") as fh:
                fh.write(raw)
            print(f"wrote {YIELD_REL}: {v['verdict']} (failed {v['failed']})")
            return 0
        if a.what in ("gen-smoke", "generate"):
            return run_gen(a.what, t_start=t_start)
        if a.what == "label":
            if not a.clone_root:
                raise Refused("run label needs --clone-root")
            return run_label(a.clone_root)
        return run_fit()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return P.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
