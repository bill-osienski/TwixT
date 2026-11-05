"""H-TB PILOT -- whole-network distillation of twixtbot's raw network into calib020_0001's architecture.

Card: docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md (pinned below by sha256). This module is the
card's minimal CPU build. Nothing runs on import. Both runtime entries are behind their own closed gates:

    python -m scripts.GPU.alphazero.htb_pilot run label --clone-root <pinned twixtbot-ui checkout>   # HTB_LABEL_AUTHORIZED
    python -m scripts.GPU.alphazero.htb_pilot run fit                                                # HTB_FIT_AUTHORIZED

Each refuses unless its gate is open and the card is committed and unchanged. Neither has run. The pure pieces below
(selection, keys, reflection, the teacher's mapping, mirror averaging, label storage, the fit and the scoring) are what
the focused CPU tests exercise, with fake teachers and tiny networks: no checkpoint, corpus or TensorFlow.

Definitions (card §5): R is the left-right reflection (r, c) -> (r, 23 - c). Every target is mirror-averaged,
T(s)[m] = (p_T(s)[m] + p_T(Rs)[Rm]) / 2, and so is every scored prediction, Q_MA. Training applies the loss to the
student's raw outputs on both orientations of each row.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import capacity_c0_inventory as C0
from . import htb_fit_authorization as FIT_GATE
from . import htb_label_authorization as LABEL_GATE
from . import twixtbot_adapter as A
from .game.twixt_state import TwixtState

ROOT = pathlib.Path(__file__).resolve().parents[3]
CARD = "docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md"
CARD_SHA256 = "ccb3bc67e32e1e8ac0f60858fcac57e99e25f1e451df8ba9c45688e19e7ab423"   # §16: compact labels
BOARD = 24
PADDED = BOARD * BOARD                       # neutral cell index r*24+c; never truncates a legal list
LAST = BOARD - 1

# card §4
CORPUS = "scripts/GPU/logs/games"
C0_INVENTORY = "docs/superpowers/evidence/2026-08-24-capacity-c0/inventory_games.json"
ELIGIBLE_DIGEST = "dc7713c601ae6fc861b47b21fbfa7939ecdb6a361e05eef2f8ff09671ec22832"
ARMS = ("train", "val")                      # the test arm is never replayed, keyed, labelled or scored
START = "red"                                # card §14: red-first games only (TwixtState() and the adapter)
N_TRAIN, N_VAL = 240_000, 24_000
SELECT_TAG, FIT_TAG = "htb-select/1", "htb-fit/1"

# card §3: the teacher (the G3 pins, reused)
TEACHER = {"repo": "github.com/stevens68/twixtbot-ui", "commit": "83749f230a0bae1766b46a05bfde0ed87f0a9a0a",
           "model_dir": "model/pb",
           "weights_sha256": "1958f8476e9d56cbb87fa570db88f9cc9d389b30e20571fd66e8e779a4cefbab",
           "saved_model_sha256": "e0c2b882bc97c4661ac92af47ae5ef78443e57b42d0cb7c9c3a8cb7e8d01993b",
           "variables_index_sha256": "1d4073c4da30e515a984faef2c3cc9c8b8ba402353a01933c8bb80e1f279f8cd"}

# card §6
REFERENCE_PATH = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
REFERENCE_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
ADAM_DEFAULTS = {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}
START_ROWS, START_KL_FLOOR = 2048, 0.01
STEP_LOG_EVERY = 25

# card §7
F1_FACTOR, F3_FACTOR = 0.5, 0.5

CACHE = pathlib.Path.home() / "twixt-cache" / "htb-pilot"      # outside the repository
EVIDENCE_LABEL = "docs/superpowers/evidence/htb-label"
EVIDENCE_FIT = "docs/superpowers/evidence/htb-fit"
SHARD_ROWS = 8192
GIB = 1 << 30
FIT_MEMORY_REQUIRED = 12 * GIB      # 12,884,901,888 B: the user's approval (2026-11-05)
LABEL_MEMORY_REQUIRED = 11 * GIB    # 11,811,160,064 B: PROPOSED (htb_budget.label_budget rounded up), for review


class Stop(RuntimeError):
    """A card check failed: the run stops with its record. Never retried automatically."""

    def __init__(self, step: str, detail: str):
        super().__init__(f"STOP at {step}: {detail}")
        self.step, self.detail = step, detail


class Refused(RuntimeError):
    """A precondition for even starting is not met (closed gate, changed card, wrong environment)."""


@dataclasses.dataclass(frozen=True)
class Recipe:
    """Card §6. The frozen values are FROZEN_RECIPE; tests use smaller ones of the same shape."""
    epochs: int = 2
    rows_per_batch: int = 32                 # x 2 orientations = 64 tensors per update
    learning_rate: float = 1e-4
    value_weight: float = 1.0
    l2_weight: float = 1e-4
    clip_norm: float = 1.0


FROZEN_RECIPE = Recipe()
FROZEN_UPDATES = FROZEN_RECIPE.epochs * N_TRAIN // FROZEN_RECIPE.rows_per_batch      # 15,000


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _jsha(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


# ─────────────────────────────── reflection and keys (card §4.4, §5) ───────────────────────────────

def reflect_rc(rc: Sequence[int]) -> Tuple[int, int]:
    r, c = rc
    return int(r), LAST - int(c)


def _canon_bridge(a, b) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    a, b = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    return (a, b) if a <= b else (b, a)


def state_serial(state: TwixtState, reflect: bool = False) -> bytes:
    """(sorted red pegs, sorted black pegs, sorted canonical bridges, side to move), optionally under R."""
    f = reflect_rc if reflect else (lambda rc: (int(rc[0]), int(rc[1])))
    red = sorted(f(p) for p, col in state.pegs.items() if col == A.RED)
    black = sorted(f(p) for p, col in state.pegs.items() if col == A.BLACK)
    bridges = sorted(_canon_bridge(f(a), f(b)) for a, b in state.bridges)
    return json.dumps([red, black, bridges, state.to_move], separators=(",", ":")).encode()


def state_key(state: TwixtState) -> str:
    """sha256 of the smaller of the state's serialization and its reflection's: equal for s and R s."""
    return _sha(min(state_serial(state), state_serial(state, reflect=True)))


def board_hwc(state: TwixtState) -> np.ndarray:
    """Our network input as self-play stores it: to_tensor() is (C, H, W); records are (H, W, C)."""
    return np.transpose(state.to_tensor(), (1, 2, 0))


def reflect_state_matches(state: TwixtState, reflected: TwixtState) -> bool:
    """Whether `reflected` (replayed from the reflected moves) IS R applied to `state`."""
    return state_serial(state, reflect=True) == state_serial(reflected)


def positions(moves: Sequence[Sequence[int]], players: Optional[Sequence[str]] = None
              ) -> Iterable[Tuple[int, TwixtState]]:
    """(ply, state before move ply) for ply 0..n-1, replayed through our rules engine; refuses an illegal move
    or (when `players` is given) a side to move that disagrees with the record."""
    s = TwixtState()
    for k, (r, c) in enumerate(moves):
        if players is not None and players[k] != s.to_move:
            raise Stop("corpus", f"ply {k}: recorded player {players[k]} but {s.to_move} to move")
        yield k, s
        if (r, c) not in s.legal_moves():
            raise Stop("corpus", f"ply {k}: {(r, c)} illegal in our engine")
        s = s.apply_move((r, c))


# ─────────────────────────────── selection (card §4) ───────────────────────────────

def _order_key(arm: str, key: str) -> str:
    return _sha(f"{SELECT_TAG}|{arm}|{key}".encode())


def select(games: Sequence[Mapping[str, Any]], n_train: int = N_TRAIN, n_val: int = N_VAL) -> Dict[str, Any]:
    """Card §4.5-4.8 over `games` = [{"id", "moves", "players", "starting_player"}], C0-eligible red-first games. Deterministic; no RNG.

    The arm is C0's split_of(id). A test-arm game is refused outright: it must never reach this function."""
    require_red_first(games)
    by_arm: Dict[str, List[Mapping[str, Any]]] = {a: [] for a in ARMS}
    for g in games:
        arm = C0.split_of(g["id"])
        if arm not in by_arm:
            raise Refused(f"{g['id']} is in the {arm} arm; only {ARMS} may be selected from")
        by_arm[arm].append(g)
    first: Dict[str, Dict[str, Tuple[str, int]]] = {a: {} for a in ARMS}
    occurrences = {a: 0 for a in ARMS}
    for arm in ARMS:
        for g in sorted(by_arm[arm], key=lambda g: g["id"]):
            for ply, s in positions(g["moves"], g["players"]):
                occurrences[arm] += 1
                first[arm].setdefault(state_key(s), (g["id"], ply))
    train_keys = set(first["train"])
    leaked = sorted(k for k in first["val"] if k in train_keys)
    pools = {"train": dict(first["train"]),
             "val": {k: v for k, v in first["val"].items() if k not in train_keys}}
    out: Dict[str, Any] = {"kind": "htb-selection/1", "card_sha256": CARD_SHA256,
                           "games": {a: len(by_arm[a]) for a in ARMS},
                           "positions": occurrences,
                           "distinct_keys": {a: len(first[a]) for a in ARMS},
                           "val_keys_excluded_as_in_train": len(leaked),
                           "pool": {a: len(pools[a]) for a in ARMS}, "rows": {}}
    for arm, n in (("train", n_train), ("val", n_val)):
        if len(pools[arm]) < n:
            raise Refused(f"the {arm} pool holds {len(pools[arm])} distinct positions, fewer than {n}; no fallback")
        chosen = sorted(pools[arm], key=lambda k: _order_key(arm, k))[:n]
        out["rows"][arm] = [{"game_id": pools[arm][k][0], "ply": pools[arm][k][1], "key": k} for k in chosen]
    return out


def recorded_start(g: Mapping[str, Any]) -> str:
    """Card §14.2: the recorded `starting_player`, checked against the moves' `player` sequence. Never defaulted."""
    sp = g.get("starting_player")
    if sp not in ("red", "black"):
        raise Stop("corpus", f"{g.get('id')}: starting_player is {sp!r}, not 'red' or 'black'")
    other = {"red": "black", "black": "red"}[sp]
    players = [m.get("player") for m in g.get("moves") or []]
    if players != [(sp, other)[k % 2] for k in range(len(players))]:
        raise Stop("corpus", f"{g.get('id')}: the moves' players do not alternate from starting_player {sp}")
    return sp


def require_red_first(games: Iterable[Mapping[str, Any]]) -> None:
    """Card §14.4: every corpus consumer takes only checked red-first games from the filtered pool."""
    bad = [g.get("id") for g in games if g.get("starting_player") != START
           or not g.get("players") or g["players"][0] != START]
    if bad:
        raise Stop("pool", f"{len(bad)} games are not checked red-first games, e.g. {bad[:3]}")


def pool_from(elig: Sequence[Mapping[str, Any]], arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """Card §14 over authenticated C0-eligible raw games: check every train/val game's starting player, exclude
    black-first games, count both per arm (move-list lengths only: nothing is replayed or keyed), and return the
    red-first games of `arms`. Test-arm games are neither checked nor counted nor returned."""
    counts = {a: {"games": 0, "positions": 0, "black_first_games_excluded": 0, "black_first_positions_excluded": 0}
              for a in ARMS}
    out = []
    for g in elig:
        arm = C0.split_of(g["id"])
        if arm not in ARMS:
            continue
        sp, n = recorded_start(g), len(g["moves"])
        if sp != START:
            counts[arm]["black_first_games_excluded"] += 1
            counts[arm]["black_first_positions_excluded"] += n
            continue
        counts[arm]["games"] += 1
        counts[arm]["positions"] += n
        if arm in arms:
            out.append({"id": g["id"], "starting_player": sp, "moves": [(m["row"], m["col"]) for m in g["moves"]],
                        "players": [m["player"] for m in g["moves"]]})
    return {"games": out, "counts": counts}


def eligible_pool(root: pathlib.Path, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """The C0-eligible games authenticated by recomputing C0's eligible digest over the WHOLE eligible set, then
    filtered to red-first games (card §14) by `pool_from`.

    Isolation, stated exactly: EVERY file -- test arm included -- is read, hashed and JSON-parsed, because the
    eligibility predicate needs each game's metadata (reason, board size, iteration, timestamp) and the digest needs
    its bytes. Test-arm move lists are parsed with the rest but are never returned, so no test-arm position is
    replayed, keyed, selected, labelled or scored."""
    committed = json.loads((ROOT / C0_INVENTORY).read_text(encoding="utf-8"))
    if committed["eligible"]["digest"] != ELIGIBLE_DIGEST:
        raise Refused("the committed C0 inventory does not carry the pinned eligible digest")
    files = sorted(p for p in root.glob("*.json") if not p.name.endswith("_stats.json"))
    raw_games, windows = [], {}
    for p in files:
        raw = p.read_bytes()
        g = json.loads(raw)
        meta = g.get("meta") or {}
        it, ts = meta.get("iteration"), g.get("timestamp") or ""
        lo, hi = windows.get(it, (ts, ts))
        windows[it] = (min(lo, ts), max(hi, ts))
        raw_games.append((p.name, _sha(raw), g, meta))
    suffix = C0.intact_suffix(windows)
    elig = [(name, sha, g) for name, sha, g, meta in raw_games
            if meta.get("reason") == C0.ELIGIBLE_REASON and meta.get("board_size") == C0.ELIGIBLE_BOARD_SIZE
            and suffix is not None and suffix[0] <= meta.get("iteration") <= suffix[1]]
    if C0._digest_of((name, sha) for name, sha, _ in elig) != ELIGIBLE_DIGEST:
        raise Refused("the live corpus does not reproduce C0's eligible digest")
    return pool_from([g for _, _, g in elig], arms)


def eligible_games(root: pathlib.Path, arms: Sequence[str] = ARMS) -> List[Dict[str, Any]]:
    """The red-first games of `arms` from `eligible_pool` (card §14)."""
    return eligible_pool(root, arms)["games"]


# ─────────────────────────────── the teacher (card §3) ───────────────────────────────

def legal_softmax(logits528: Sequence[float], colour: str, legal: Sequence[Tuple[int, int]]) -> np.ndarray:
    """The teacher's prior on OUR legal moves, in our order: softmax over the legal slots only."""
    z = np.asarray(logits528, dtype=np.float64).reshape(-1)
    if z.size != A.POLICY_N:
        raise Stop("teacher", f"{z.size} move logits, expected {A.POLICY_N}")
    if not np.all(np.isfinite(z)):
        raise Stop("teacher", "non-finite move logits")
    sel = z[[A.policy_index(colour, r, c) for r, c in legal]]
    e = np.exp(sel - sel.max())
    return e / e.sum()


class TwixtbotTeacher:
    """The pinned twixtbot-ui raw network through its own evaluator, one position per call, rotation off.
    Built only inside `run label`, from `external_learning.load_runtime` (G3's verifier, CPU-only TensorFlow)."""

    def __init__(self, rt: Mapping[str, Any]):
        self.rt = rt
        self.naf = rt["nnmplayer"].naf
        self.ev = rt["nneval"].NNEvaluater(TEACHER["model_dir"])
        self.calls = 0

    def new_game(self):
        return self.rt["twixt"].Game(allow_scl=False)

    def play(self, tb, rc) -> None:
        tb.play(self.rt["Point"](*A.rc_to_xy(*rc)))

    def divergences(self, our: TwixtState, tb) -> List[str]:
        out = list(A.state_divergences(our, tb, self.rt["twixt"], ply_cap=None))
        if A.tb_legal_moves(tb) != set(our.legal_moves()):
            out.append("legal sets differ")
        return out

    def evaluate(self, tb, colour: str, legal: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        pw, ml = self.ev.eval_one(self.naf.NetInputs(tb))
        self.calls += 1
        value = A.pwin_to_value(pw, self.naf)
        if not np.isfinite(value):
            raise Stop("teacher", "non-finite value")
        return legal_softmax(ml, colour, legal), value


def _replay_and_evaluate(moves, plies, teacher) -> Dict[int, Tuple[TwixtState, List[Tuple[int, int]], np.ndarray, float]]:
    our, tb, out = TwixtState(), teacher.new_game(), {}
    last = max(plies)
    for k in range(last + 1):
        if k in plies:
            d = teacher.divergences(our, tb)
            if d:
                raise Stop("lockstep", f"ply {k}: {d[:3]}")
            legal = list(our.legal_moves())
            p, v = teacher.evaluate(tb, our.to_move, legal)
            out[k] = (our, legal, p, v)
        rc = tuple(moves[k])
        our = our.apply_move(rc)
        teacher.play(tb, rc)
    return out


def label_game(game_id: str, moves: Sequence[Sequence[int]], rows: Sequence[Mapping[str, Any]], teacher
               ) -> List[Dict[str, Any]]:
    """Card §5: each selected ply evaluated on the game and on its reflection (replayed in lockstep), then averaged."""
    plies = {int(r["ply"]) for r in rows}
    ident = _replay_and_evaluate(moves, plies, teacher)
    refl = _replay_and_evaluate([reflect_rc(m) for m in moves], plies, teacher)
    out = []
    for r in sorted(rows, key=lambda r: r["ply"]):
        k = int(r["ply"])
        s, legal, p, v = ident[k]
        rs, rlegal, rp, rv = refl[k]
        if not reflect_state_matches(s, rs):
            raise Stop("reflection", f"{game_id} ply {k}: the reflected replay is not R of the state")
        if state_key(s) != r["key"]:
            raise Stop("selection", f"{game_id} ply {k}: key differs from the selection record")
        at = {m: i for i, m in enumerate(rlegal)}
        rp_back = np.array([rp[at[reflect_rc(m)]] for m in legal])
        target = 0.5 * (p + rp_back)
        out.append({"game_id": game_id, "ply": k, "key": r["key"], "to_move": s.to_move,
                    "legal": [list(m) for m in legal], "target": target,
                    "value": 0.5 * (v + rv), "value_id": v, "value_ref": rv})
    return out


# ─────────────────────────────── label storage ───────────────────────────────

def write_label_shards(rows: Sequence[Mapping[str, Any]], out_dir: pathlib.Path, shard_rows: int = SHARD_ROWS
                       ) -> Dict[str, Any]:
    """Create-only shards: targets on the neutral 576-cell index (float32), values float64; returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=False)
    shards = []
    for i in range(0, len(rows), shard_rows):
        part = rows[i:i + shard_rows]
        tgt = np.zeros((len(part), PADDED), dtype=np.float32)
        for j, r in enumerate(part):
            for (rr, cc), t in zip(r["legal"], r["target"]):
                tgt[j, A.padded_index(rr, cc)] = t
        name = f"labels-{i // shard_rows:05d}.npz"
        path = out_dir / name
        with open(path, "xb") as fh:
            np.savez(fh, target=tgt,
                     value=np.array([r["value"] for r in part], dtype=np.float64),
                     value_id=np.array([r["value_id"] for r in part], dtype=np.float64),
                     value_ref=np.array([r["value_ref"] for r in part], dtype=np.float64),
                     ply=np.array([r["ply"] for r in part], dtype=np.int32))
        meta = [{"game_id": r["game_id"], "ply": int(r["ply"]), "key": r["key"]} for r in part]
        (out_dir / (name + ".rows.json")).open("x").write(json.dumps(meta))
        shards.append({"file": name, "sha256": _sha(path.read_bytes()), "rows_sha256": _jsha(meta), "n": len(part)})
    manifest = {"kind": "htb-labels/1", "n": len(rows), "shards": shards}
    (out_dir / "manifest.json").open("x").write(json.dumps(manifest, sort_keys=True, indent=1))
    return manifest


class LabelArrays:
    """Card §16.1: one arm's labels held compactly AS THEY ARE PRODUCED, in selection order: float32 targets on the
    576-cell index, float64 values, int32 plies, and the selection's own row metadata. A label row's legal-move list
    is consumed by `put` and not kept. Every selected row must be filled exactly once."""

    def __init__(self, selected: Sequence[Mapping[str, Any]]):
        self.meta = [{"game_id": r["game_id"], "ply": int(r["ply"]), "key": r["key"]} for r in selected]
        self.at = {m["key"]: i for i, m in enumerate(self.meta)}
        if len(self.at) != len(self.meta):
            raise Stop("labels", "the selection repeats a key within an arm")
        n = len(self.meta)
        self.target = np.zeros((n, PADDED), dtype=np.float32)
        self.value, self.value_id, self.value_ref = (np.zeros(n, dtype=np.float64) for _ in range(3))
        self.ply = np.array([m["ply"] for m in self.meta], dtype=np.int32)
        self.filled = np.zeros(n, dtype=bool)

    def put(self, row: Mapping[str, Any]) -> None:
        i = self.at.get(row["key"])
        if i is None or self.meta[i]["game_id"] != row["game_id"] or self.meta[i]["ply"] != int(row["ply"]):
            raise Stop("labels", f"{row['game_id']} ply {row['ply']}: a labelled row the selection does not hold")
        if self.filled[i]:
            raise Stop("labels", f"{row['game_id']} ply {row['ply']}: labelled twice")
        for (rr, cc), t in zip(row["legal"], row["target"]):              # as write_label_shards always stored it
            self.target[i, A.padded_index(rr, cc)] = t
        self.value[i], self.value_id[i], self.value_ref[i] = row["value"], row["value_id"], row["value_ref"]
        self.filled[i] = True

    def complete(self) -> None:
        if not self.filled.all():
            raise Stop("labels", f"{int((~self.filled).sum())} selected rows were never labelled")


def write_label_arrays(arrays: LabelArrays, out_dir: pathlib.Path, shard_rows: int = SHARD_ROWS) -> Dict[str, Any]:
    """`write_label_shards`' exact format (shards, row lists, manifest), written from compact arrays."""
    arrays.complete()
    out_dir.mkdir(parents=True, exist_ok=False)
    shards = []
    for i in range(0, len(arrays.meta), shard_rows):
        j = min(i + shard_rows, len(arrays.meta))
        name = f"labels-{i // shard_rows:05d}.npz"
        path = out_dir / name
        with open(path, "xb") as fh:
            np.savez(fh, target=arrays.target[i:j], value=arrays.value[i:j], value_id=arrays.value_id[i:j],
                     value_ref=arrays.value_ref[i:j], ply=arrays.ply[i:j])
        meta = arrays.meta[i:j]
        (out_dir / (name + ".rows.json")).open("x").write(json.dumps(meta))
        shards.append({"file": name, "sha256": _sha(path.read_bytes()), "rows_sha256": _jsha(meta), "n": j - i})
    manifest = {"kind": "htb-labels/1", "n": len(arrays.meta), "shards": shards}
    (out_dir / "manifest.json").open("x").write(json.dumps(manifest, sort_keys=True, indent=1))
    return manifest


def load_label_compact(out_dir: pathlib.Path, manifest_sha256: str
                       ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Card §15.1: reload, verifying the manifest and every shard, into compact form -- the rows' (game_id, ply, key),
    one preallocated float32 (n, 576) target array and a float64 value array. Each shard's arrays are read once."""
    raw = (out_dir / "manifest.json").read_bytes()
    if _sha(raw) != manifest_sha256:
        raise Refused("the label manifest is not the pinned one")
    manifest = json.loads(raw)
    n = manifest["n"]
    if sum(sh["n"] for sh in manifest["shards"]) != n:
        raise Refused("the shards do not hold the manifest's row count")
    meta: List[Dict[str, Any]] = []
    target = np.empty((n, PADDED), dtype=np.float32)
    value = np.empty(n, dtype=np.float64)
    for sh in manifest["shards"]:
        data = (out_dir / sh["file"]).read_bytes()
        if _sha(data) != sh["sha256"]:
            raise Refused(f"{sh['file']} changed since labelling")
        rows = json.loads((out_dir / (sh["file"] + ".rows.json")).read_text())
        if _jsha(rows) != sh["rows_sha256"] or len(rows) != sh["n"]:
            raise Refused(f"{sh['file']} row list changed since labelling")
        with np.load(out_dir / sh["file"]) as z:            # each array materialized ONCE per shard: indexing
            t, v = z["target"], z["value"]                   # z["..."] inside a row loop re-reads the whole array
        if t.shape != (len(rows), PADDED) or v.shape != (len(rows),):
            raise Refused(f"{sh['file']} arrays do not match its row list")
        i = len(meta)
        target[i:i + len(rows)] = t
        value[i:i + len(rows)] = v
        meta += rows
    return meta, target, value


def load_label_shards(out_dir: pathlib.Path, manifest_sha256: str) -> List[Dict[str, Any]]:
    """Reload as one dict per row (targets as float64 on the 576-cell index): `load_label_compact`, expanded."""
    meta, target, value = load_label_compact(out_dir, manifest_sha256)
    return [{**m, "target576": target[i].astype(np.float64), "value": float(value[i])} for i, m in enumerate(meta)]


# ─────────────────────────────── records for the network ───────────────────────────────

def build_examples(labelled: Sequence[Mapping[str, Any]], moves_by_game: Mapping[str, Sequence]) -> List[Dict[str, Any]]:
    """Each labelled row as (record, reflected record, teacher value), boards rebuilt by replay.

    The stored float32 target is renormalized over the legal moves (rounding only). The reflected record comes from
    self_play._mirror_position_lr, whose moves keep the original order, so the same target list is T on R s."""
    from .self_play import PositionRecord, _mirror_position_lr

    want: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for r in labelled:
        want.setdefault(r["game_id"], {})[int(r["ply"])] = r
    out = {}
    for gid, by_ply in want.items():
        for ply, s in positions(moves_by_game[gid]):
            if ply > max(by_ply):
                break
            if ply not in by_ply:
                continue
            row = by_ply[ply]
            if state_key(s) != row["key"]:
                raise Stop("examples", f"{gid} ply {ply}: key differs from the labels")
            legal = list(s.legal_moves())
            t = np.array([row["target576"][A.padded_index(r, c)] for r, c in legal], dtype=np.float64)
            if not t.sum() > 0:
                raise Stop("examples", f"{gid} ply {ply}: target has no mass on the legal moves")
            t = t / t.sum()
            board = board_hwc(s)
            rec = PositionRecord(board_tensor=board, to_move=s.to_move, legal_moves=legal,
                                 visit_counts=t.tolist(), outcome=0.0, active_size=BOARD, ply=ply)
            mb, mm, mt = _mirror_position_lr(board, legal, t.tolist(), BOARD)
            mrec = PositionRecord(board_tensor=mb, to_move=s.to_move, legal_moves=list(mm),
                                  visit_counts=list(mt), outcome=0.0, active_size=BOARD, ply=ply)
            out[row["key"]] = {"key": row["key"], "record": rec, "mirror": mrec, "value": float(row["value"]),
                               "target": t}
    return [out[r["key"]] for r in labelled]


SCORE_CHUNK = 64                               # scoring's chunk, unchanged (mirror_averaged's default)


class ExampleStore:
    """Card §15: compact labelled rows plus the move lists of their games. COMPLETE examples -- both orientations,
    made by `build_examples` itself -- exist only for the indices passed to `take`, and are released by the caller.
    `keys`, `values` and `len` read the compact rows and never build an example."""

    def __init__(self, meta: Sequence[Mapping[str, Any]], target: np.ndarray, value: np.ndarray,
                 moves_by_game: Mapping[str, Sequence]):
        n = len(meta)
        if target.shape != (n, PADDED) or target.dtype != np.float32 or value.shape != (n,):
            raise Stop("examples", "the compact labels do not match their row list")
        missing = sorted({m["game_id"] for m in meta} - set(moves_by_game))
        if missing:
            raise Stop("examples", f"{len(missing)} labelled games are not in the pool, e.g. {missing[:3]}")
        self.meta, self.target, self.value = list(meta), target, value
        self.moves = {gid: moves_by_game[gid] for gid in {m["game_id"] for m in meta}}
        self.keys = [m["key"] for m in self.meta]

    def __len__(self) -> int:
        return len(self.meta)

    def row(self, i: int) -> Dict[str, Any]:
        """The row exactly as `load_label_shards` gives it."""
        return {**self.meta[i], "target576": self.target[i].astype(np.float64), "value": float(self.value[i])}

    def take(self, idx: Iterable[int]) -> List[Dict[str, Any]]:
        return build_examples([self.row(i) for i in idx], self.moves)


class _Built:
    """A list of examples built up front, behind the same interface (the timing and the tests use this form)."""

    def __init__(self, examples: Sequence[Mapping[str, Any]]):
        self.examples = list(examples)
        self.keys = [e["key"] for e in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def take(self, idx: Iterable[int]) -> List[Mapping[str, Any]]:
        return [self.examples[i] for i in idx]


class _Subset:
    def __init__(self, src, idx: Sequence[int]):
        self.src, self.idx = src, list(idx)
        self.keys = [src.keys[i] for i in self.idx]

    def __len__(self) -> int:
        return len(self.idx)

    def take(self, idx: Iterable[int]) -> List[Mapping[str, Any]]:
        return self.src.take([self.idx[i] for i in idx])


def _source(examples):
    return examples if isinstance(examples, (ExampleStore, _Built, _Subset)) else _Built(examples)


# ─────────────────────────────── network side: forward, MA, scoring (card §5, §7) ───────────────────────────────

def _batch(records):
    from .trainer import make_padded_batch

    boards, rows, cols, mask, target, _ = make_padded_batch(list(records), max_moves_cap=PADDED)
    if mask.shape[1] < max(len(r.legal_moves) for r in records):
        raise Stop("batch", "the padded batch truncated a legal-move list")
    return boards, rows, cols, mask, target


def _require_eval(network, who: str) -> None:
    training = [n or "<root>" for n, m in network.named_modules() if m.training]
    if training:
        raise Stop("mode", f"{who} has modules in training mode: {training[:5]}")


def mirror_averaged(network, examples: Sequence[Mapping[str, Any]], chunk: int = 64
                    ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Q_MA: per example, the mean of the two orientations' legal-move probabilities (index-aligned) and values."""
    import mlx.core as mx
    from .trainer import compute_masked_log_probs

    _require_eval(network, "the scored network")
    probs, values = [], []
    for i in range(0, len(examples), chunk):
        part = examples[i:i + chunk]
        got = []
        for which in ("record", "mirror"):
            boards, rows, cols, mask, _ = _batch([e[which] for e in part])
            logits, v, _ = network.forward_padded(boards, rows, cols, mask, active_size=BOARD)
            lp = compute_masked_log_probs(logits, mask)
            mx.eval(lp, v)
            got.append((np.array(lp, dtype=np.float64), np.array(v, dtype=np.float64)))
        (lp1, v1), (lp2, v2) = got
        for j, e in enumerate(part):
            n = len(e["record"].legal_moves)
            probs.append(0.5 * (np.exp(lp1[j, :n]) + np.exp(lp2[j, :n])))
        values.append(0.5 * (v1 + v2))
    return probs, np.concatenate(values)


def top1(x: np.ndarray) -> int:
    """argmax with ties to the first move in our legal order."""
    return int(np.flatnonzero(x == x.max())[0])


def score(network, examples) -> Dict[str, float]:
    """KL_Q, TOP_Q and MSE_Q of card §7 for the network's mirror-averaged predictions. Examples are built one
    64-row chunk at a time (card §15), the same chunks `mirror_averaged` always used."""
    src = _source(examples)
    kls, tops, vqs, vts = [], [], [], []
    for i in range(0, len(src), SCORE_CHUNK):
        part = src.take(range(i, min(i + SCORE_CHUNK, len(src))))
        q, vq = mirror_averaged(network, part, chunk=SCORE_CHUNK)
        for e, qi in zip(part, q):
            t = e["target"]
            pos = t > 0
            if np.any(qi[pos] <= 0):
                raise Stop("score", "a mirror-averaged probability underflowed to zero where the target has mass")
            kls.append(float(np.sum(t[pos] * (np.log(t[pos]) - np.log(qi[pos])))))
            tops.append(top1(qi) == top1(t))
        vqs.append(vq)
        vts += [e["value"] for e in part]
        del part, q
    vq, vt = np.concatenate(vqs), np.array(vts, dtype=np.float64)
    out = {"n": len(src), "kl": float(np.mean(kls)), "top1": float(np.mean(tops)),
           "value_mse": float(np.mean((vq - vt) ** 2))}
    if not all(np.isfinite(v) for v in out.values()):
        raise Stop("score", f"non-finite metric {out}")
    return out


def verdict(student: Mapping[str, float], reference: Mapping[str, float]) -> Dict[str, Any]:
    f = {"F1": student["kl"] <= F1_FACTOR * reference["kl"],
         "F2": student["top1"] > reference["top1"],
         "F3": student["value_mse"] <= F3_FACTOR * reference["value_mse"]}
    return {**f, "verdict": "ELIGIBLE" if all(f.values()) else "STOP"}


# ─────────────────────────────── the fit (card §6) ───────────────────────────────

def epoch_order(keys: Sequence[str], epoch: int) -> List[int]:
    return sorted(range(len(keys)), key=lambda i: _sha(f"{FIT_TAG}|{epoch}|{keys[i]}".encode()))


def start_check(reference, examples, n: int = START_ROWS) -> Dict[str, float]:
    src = _source(examples)
    first = _Subset(src, epoch_order(src.keys, 0)[:n])
    s = score(reference, first)
    if not s["kl"] >= START_KL_FLOOR:
        raise Stop("start", f"KL(T || reference_MA) {s['kl']} on the first {len(first)} rows is below {START_KL_FLOOR}")
    return s


def _snapshot(network) -> Dict[str, np.ndarray]:
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(network.parameters())}


def _digest(network) -> str:
    from . import training_replay_capture as TRC

    return TRC.model_state_digest(_snapshot(network))


def surface_names(network) -> List[str]:
    from mlx.utils import tree_flatten

    names = sorted(k for k, _ in tree_flatten(network.trainable_parameters()))
    if any(n.endswith(("running_mean", "running_var")) for n in names):
        raise Stop("surface", "a BatchNorm running statistic is trainable")
    return names


def fit(student, reference, examples, recipe: Recipe = FROZEN_RECIPE,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        update_times: Optional[List[float]] = None) -> Dict[str, Any]:
    """Train `student` in place, whole network, evaluation-mode BatchNorm, on both orientations of every row.
    `examples` is an ExampleStore (each batch's examples built when it is used, card §15) or a built list."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from .trainer import clip_grad_norm, compute_masked_log_probs

    if student is reference:
        raise Stop("fit", "the student and the reference are the same object")
    for net, who in ((student, "the student"), (reference, "the reference")):
        _require_eval(net, who)
    ref_digest = _digest(reference)
    if _digest(student) != ref_digest:
        raise Stop("fit", "the student does not start as an exact copy of the reference")
    src = _source(examples)
    n = len(src)
    if n < recipe.rows_per_batch or n % recipe.rows_per_batch:
        raise Stop("fit", f"{n} rows do not divide into batches of {recipe.rows_per_batch}")
    names = surface_names(student)
    before = _snapshot(student)
    opt = optim.Adam(learning_rate=recipe.learning_rate)
    resolved = {"betas": [float(b) for b in opt.betas], "eps": float(opt.eps), "bias_correction": bool(opt.bias_correction)}
    if resolved != ADAM_DEFAULTS:
        raise Stop("optimizer", f"MLX's Adam resolves to {resolved}, not {ADAM_DEFAULTS}")
    keys = src.keys
    if len(set(keys)) != n:
        raise Stop("fit", "duplicate keys among the training rows")

    def loss_fn(model, boards, rows, cols, mask, target, tv):
        logits, values, _ = model.forward_padded(boards, rows, cols, mask, active_size=BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        value = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(model.trainable_parameters()))
        return policy + recipe.value_weight * value + recipe.l2_weight * l2, policy, value

    grad_fn = nn.value_and_grad(student, loss_fn)
    log, step = [], 0
    for epoch in range(recipe.epochs):
        order = epoch_order(keys, epoch)
        for b in range(0, n, recipe.rows_per_batch):
            idx = order[b:b + recipe.rows_per_batch]
            recs, tv = [], []
            for e in src.take(idx):                          # this batch's complete examples only
                recs += [e["record"], e["mirror"]]
                tv += [e["value"]] * 2
            boards, rows, cols, mask, target = _batch(recs)
            (loss, policy, value), grads = grad_fn(student, boards, rows, cols, mask, target,
                                                   mx.array(np.asarray(tv, dtype=np.float32)))
            grads, norm = clip_grad_norm(grads, recipe.clip_norm)
            opt.update(student, grads)
            mx.eval(student.parameters(), opt.state, loss, policy, value, norm)
            del recs, boards, rows, cols, mask, target, grads
            if update_times is not None:                     # timing only: a perf_counter stamp per finished update
                update_times.append(time.perf_counter())
            vals = (float(loss), float(policy), float(value), float(norm))
            if not all(np.isfinite(vals)):
                raise Stop("fit", f"update {step}: non-finite loss or gradient norm {vals}")
            if step % STEP_LOG_EVERY == 0:
                log.append({"update": step, "epoch": epoch, "loss": vals[0], "policy": vals[1], "value": vals[2],
                            "grad_norm": vals[3]})
                if on_step is not None:
                    on_step(log[-1])
            step += 1
    _require_eval(student, "the student after the fit")
    if _digest(reference) != ref_digest:
        raise Stop("fit", "the reference changed during the fit")
    after = _snapshot(student)
    stats = sorted(k for k in before if k.endswith(("running_mean", "running_var")))
    moved_stats = [k for k in stats if not np.array_equal(before[k], after[k])]
    if moved_stats:
        raise Stop("batchnorm", f"running statistics changed: {moved_stats[:5]}")
    moved = [k for k in names if not np.array_equal(before[k], after[k])]
    groups = {g: any(k.startswith(g) for k in moved) for g in ("encoder.", "policy_head.", "value_head.")}
    if not all(groups.values()):
        raise Stop("surface", f"a part of the network never moved: {groups}")
    return {"recipe": dataclasses.asdict(recipe), "optimizer": {"name": "Adam", **resolved}, "updates": step,
            "log": log, "surface_tensors": len(names), "surface_tensors_changed": len(moved),
            "running_statistics_unchanged": len(stats), "reference_digest": ref_digest,
            "student_digest": _digest(student)}


# ─────────────────────────────── gated entries ───────────────────────────────
#
# Order in both entries (card §13): every REFUSAL comes first and writes nothing -- the gate, the card, the
# destinations and (fit) the label binding -- so a refused run loads, reads and trains nothing. Only then is the
# evidence directory created, with `started.json`; everything after runs inside `_terminal`, which always leaves a
# create-only `record.json` whose status is COMPLETED, STOP, FAILED or INTERRUPTED (SIGTERM / SIGINT).

EXIT = {"COMPLETED": 0, "REFUSED": 2, "STOP": 3, "FAILED": 4, "INTERRUPTED": 5}


class Interrupted(BaseException):
    """SIGTERM or SIGINT arrived inside the terminal-record boundary."""


def _committed_git(root: pathlib.Path, rel: str) -> bool:
    """Tracked, and unchanged against HEAD (worktree and index)."""
    r = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", rel], capture_output=True)
    d = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", rel])
    c = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet", "HEAD", "--", rel])
    return r.returncode == 0 and d.returncode == 0 and c.returncode == 0


def main_checkout(root: pathlib.Path = ROOT) -> pathlib.Path:
    """The main checkout (the parent of the shared git directory): it holds the untracked corpus and checkpoint that a
    clean linked worktree lacks. Evidence still goes to the tree the run executes in."""
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return pathlib.Path(out).parent


def _load_reference_real(root: pathlib.Path):   # pragma: no cover - loads the checkpoint; never under this approval
    from .probe_eval import load_network_for_scoring

    path = root / REFERENCE_PATH
    if hashlib.sha1(path.read_bytes()).hexdigest() != REFERENCE_SHA1:
        raise Stop("reference", "the reference checkpoint is not calib020_0001")
    net = load_network_for_scoring(str(path), verbose=False)[0]
    net.eval()
    return net


def _load_teacher_real(clone_root: str):   # pragma: no cover - needs TensorFlow; never under this approval
    from . import external_learning as EL

    rt, identity = EL.load_runtime(clone_root)
    return TwixtbotTeacher(rt), identity


def _save_student_real(network, path: pathlib.Path) -> Dict[str, str]:
    from . import search_distillation_d2 as D

    return D.save_student(network, str(path))


@dataclasses.dataclass
class Config:
    """Everything an entry touches. The defaults ARE the card; tests substitute fakes and smaller counts."""
    root: pathlib.Path = ROOT
    data_root: pathlib.Path = dataclasses.field(default_factory=main_checkout)   # corpus + checkpoint
    cache: pathlib.Path = CACHE
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    recipe: Recipe = FROZEN_RECIPE
    committed: Callable[[pathlib.Path, str], bool] = _committed_git
    games: Callable[[pathlib.Path], List[Dict[str, Any]]] = lambda root: eligible_games(root / CORPUS)
    load_teacher: Callable[[str], Tuple[Any, Dict[str, Any]]] = _load_teacher_real
    load_reference: Callable[[pathlib.Path], Any] = _load_reference_real
    save_student: Callable[[Any, pathlib.Path], Dict[str, str]] = _save_student_real
    shard_rows: int = SHARD_ROWS
    start_rows: int = START_ROWS
    #: bytes the full fit needs at launch: 12 GiB, approved by the user (2026-11-05) from the card §15 budget (12.55 GB,
    #: an estimate; peak memory not measured). None REFUSES.
    memory_required_bytes: Optional[int] = FIT_MEMORY_REQUIRED
    #: bytes `run label` needs at launch (card §16.3): PROPOSED from the compact label budget, for the launch review
    label_memory_required_bytes: Optional[int] = LABEL_MEMORY_REQUIRED
    available_memory: Callable[[], Dict[str, int]] = lambda: available_memory()

    def labels(self, arm: str) -> pathlib.Path:
        return self.cache / f"labels-{arm}"

    @property
    def student(self) -> pathlib.Path:
        return self.cache / "student.safetensors"

    @property
    def evidence_label(self) -> pathlib.Path:
        return self.root / EVIDENCE_LABEL

    @property
    def evidence_fit(self) -> pathlib.Path:
        return self.root / EVIDENCE_FIT


def available_memory() -> Dict[str, int]:
    """Physical memory and what macOS reports as reclaimable now: (free + inactive + speculative) pages, from
    `sysctl hw.memsize` and `vm_stat`. A conservative reading: purgeable and compressible memory are not counted."""
    phys = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines()
    page = int(out[0].split("page size of")[1].split()[0])
    pages = {}
    for line in out[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[k.strip()] = int(v)
    avail = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative")) * page
    return {"physical_bytes": phys, "available_bytes": avail, "page_size": page}


def preconditions(gate_module, gate_name: str, cfg: Config) -> None:
    """Refuse before anything heavy: the gate, then the card's pin and commit, then the environment."""
    if getattr(gate_module, gate_name) is not True:
        raise Refused(f"{gate_name} is closed")
    if _sha((cfg.root / CARD).read_bytes()) != CARD_SHA256:
        raise Refused("the card is not the pinned one")
    if not cfg.committed(cfg.root, CARD):
        raise Refused("the card is not committed")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise Refused("MLX_ENABLE_TF32 must be 0 in this process")


def _absent(*paths: pathlib.Path) -> None:
    there = [str(p) for p in paths if os.path.lexists(p)]
    if there:
        raise Refused(f"destination already exists (one run only): {there}")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def _write_create_only(path: pathlib.Path, obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, indent=1, default=_jsonable).encode()
    with open(path, "xb") as fh:
        fh.write(raw)
    return _sha(raw)


def _terminal(ev_dir: pathlib.Path, rec: Dict[str, Any], body: Callable[[Dict[str, Any]], None],
              guard: Optional[Callable[[Dict[str, Any]], None]] = None, keep: Sequence[str] = ()) -> int:
    """Create ev_dir with started.json, run body(rec) and ALWAYS write record.json with the terminal status.

    `guard`, if given, runs on EVERY terminal record before it is written; if it raises Stop, only the `keep` fields
    survive and the record becomes a STOP at the guard's step."""
    import signal
    import traceback

    ev_dir.mkdir(parents=True, exist_ok=False)
    _write_create_only(ev_dir / "started.json", {k: rec[k] for k in ("kind", "card_sha256") if k in rec})

    def _raise(signum, _frame):
        raise Interrupted(signal.Signals(signum).name)

    old = {s: signal.signal(s, _raise) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        body(rec)
        rec["status"] = "COMPLETED"
    except Stop as e:
        rec["status"], rec["stop"] = "STOP", {"step": e.step, "detail": e.detail}
    except Refused as e:                               # a binding check that could only be made inside the body
        rec["status"], rec["stop"] = "STOP", {"step": "refused", "detail": str(e)}
    except Interrupted as e:
        rec["status"], rec["interrupted_by"] = "INTERRUPTED", str(e)
    except BaseException as e:                         # noqa: BLE001 - the record must exist whatever happened
        rec["status"], rec["error"] = "FAILED", {"type": type(e).__name__, "detail": str(e),
                                                  "traceback": traceback.format_exc()}
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        if guard is not None:
            try:
                guard(rec)
            except Stop as e:
                for k in [k for k in rec if k not in keep]:
                    del rec[k]
                rec["status"], rec["stop"] = "STOP", {"step": e.step, "detail": e.detail}
        _write_create_only(ev_dir / "record.json", rec)
    return EXIT[rec["status"]]


def run_label(clone_root: str, cfg: Optional[Config] = None) -> int:
    cfg = cfg or Config()
    preconditions(LABEL_GATE, "HTB_LABEL_AUTHORIZED", cfg)
    _absent(cfg.evidence_label, *(cfg.labels(a) for a in ARMS))
    memory = check_label_memory(cfg)                   # card §16.3: before the teacher or the corpus is loaded
    rec: Dict[str, Any] = {"kind": "htb-label-record/1", "card_sha256": CARD_SHA256, "teacher": TEACHER,
                           "counts": {"train": cfg.n_train, "val": cfg.n_val},
                           "memory_at_start": memory, "label_memory_required_bytes": cfg.label_memory_required_bytes}

    def body(rec):
        teacher, identity = cfg.load_teacher(clone_root)
        rec["runtime"] = identity
        games = {g["id"]: g for g in cfg.games(cfg.data_root)}
        sel = select(list(games.values()), n_train=cfg.n_train, n_val=cfg.n_val)
        rec["selection_sha256"] = _write_create_only(cfg.evidence_label / "selection.json", sel)
        rec["labels"] = []
        for arm in ARMS:
            by_game: Dict[str, List[Mapping[str, Any]]] = {}
            for r in sel["rows"][arm]:
                by_game.setdefault(r["game_id"], []).append(r)
            arrays = LabelArrays(sel["rows"][arm])                       # card §16.1: compact as produced
            for gid in sorted(by_game):
                for row in label_game(gid, games[gid]["moves"], by_game[gid], teacher):
                    arrays.put(row)
            manifest = write_label_arrays(arrays, cfg.labels(arm), cfg.shard_rows)
            del arrays
            rec["labels"].append({"arm": arm, "n": manifest["n"],
                                  "manifest_sha256": _sha((cfg.labels(arm) / "manifest.json").read_bytes())})
        rec["teacher_calls"] = getattr(teacher, "calls", None)

    return _terminal(cfg.evidence_label, rec, body)


def check_label_binding(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Refuse unless the label evidence is committed, unchanged, COMPLETED and bound to this card, this teacher, its
    own selection and the card's exact counts. Reads only the two small evidence files."""
    rels = [f"{EVIDENCE_LABEL}/record.json", f"{EVIDENCE_LABEL}/selection.json"]
    for rel in rels:
        if not (cfg.root / rel).is_file() or not cfg.committed(cfg.root, rel):
            raise Refused(f"{rel} is absent, uncommitted or changed")
    raw_sel = (cfg.root / rels[1]).read_bytes()
    label, sel = json.loads((cfg.root / rels[0]).read_text()), json.loads(raw_sel)
    want = {"kind": "htb-label-record/1", "status": "COMPLETED", "card_sha256": CARD_SHA256, "teacher": TEACHER,
            "counts": {"train": cfg.n_train, "val": cfg.n_val}, "selection_sha256": _sha(raw_sel)}
    bad = [k for k, v in want.items() if label.get(k) != v]
    if bad:
        raise Refused(f"the label record does not bind: {bad}")
    if sel.get("kind") != "htb-selection/1" or sel.get("card_sha256") != CARD_SHA256:
        raise Refused("the selection is not this card's")
    n = {"train": cfg.n_train, "val": cfg.n_val}
    if sorted(sel.get("rows", {})) != sorted(ARMS) or any(len(sel["rows"][a]) != n[a] for a in ARMS):
        raise Refused("the selection does not hold the card's exact arm counts")
    keys = [r["key"] for a in ARMS for r in sel["rows"][a]]
    if len(set(keys)) != len(keys):
        raise Refused("the selection repeats a key within or across arms")
    if [e.get("arm") for e in label.get("labels", [])] != list(ARMS) or \
            any(e["n"] != n[e["arm"]] for e in label["labels"]):
        raise Refused("the label record's arms or counts are not the card's")
    return label, sel


def load_bound_labels(cfg: Config, label: Mapping[str, Any], sel: Mapping[str, Any]
                      ) -> Dict[str, Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]]:
    """Each arm's shards in compact form (card §15.1), verified against the record's manifest hash, with row
    identities equal to the selection's, in order."""
    out = {}
    for e in label["labels"]:
        rows = load_label_compact(cfg.labels(e["arm"]), e["manifest_sha256"])
        got = [(r["game_id"], int(r["ply"]), r["key"]) for r in rows[0]]
        want = [(r["game_id"], int(r["ply"]), r["key"]) for r in sel["rows"][e["arm"]]]
        if got != want:
            raise Refused(f"the {e['arm']} labels are not the selection's rows, in order")
        out[e["arm"]] = rows
    return out


def _check_available(cfg: Config, required: Optional[int], who: str) -> Dict[str, int]:
    if required is None:
        raise Refused(f"the {who}'s memory requirement is not set (only a reviewed edit sets it)")
    mem = cfg.available_memory()
    if mem["available_bytes"] < required:
        raise Refused(f"available memory {mem['available_bytes']} B is below the {who}'s requirement {required} B")
    return mem


def check_memory(cfg: Config) -> Dict[str, int]:
    """Refuse unless a reviewed memory requirement is set and the machine has at least that much available now."""
    return _check_available(cfg, cfg.memory_required_bytes, "fit")


def check_label_memory(cfg: Config) -> Dict[str, int]:
    """Card §16.3: the same check for `run label`, before the teacher or the corpus is loaded."""
    return _check_available(cfg, cfg.label_memory_required_bytes, "label run")


def run_fit(cfg: Optional[Config] = None) -> int:
    cfg = cfg or Config()
    preconditions(FIT_GATE, "HTB_FIT_AUTHORIZED", cfg)
    _absent(cfg.evidence_fit, cfg.student)
    label, sel = check_label_binding(cfg)
    memory = check_memory(cfg)                         # before the shards (and everything heavier) are loaded
    rows = load_bound_labels(cfg, label, sel)          # before any corpus replay, checkpoint or update
    rec: Dict[str, Any] = {"kind": "htb-fit-record/1", "card_sha256": CARD_SHA256,
                           "label_record_sha256": _sha((cfg.root / EVIDENCE_LABEL / "record.json").read_bytes()),
                           "selection_sha256": label["selection_sha256"], "recipe": dataclasses.asdict(cfg.recipe),
                           "memory_at_start": memory, "memory_required_bytes": cfg.memory_required_bytes}

    def body(rec):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        pool = cfg.games(cfg.data_root)
        require_red_first(pool)                                          # card §14.4
        games = {g["id"]: g["moves"] for g in pool}
        del pool
        arms = {a: ExampleStore(*rows[a], games) for a in ARMS}          # card §15: built per batch / chunk
        rec["storage"] = {"examples": "built per training batch and scoring chunk (card §15)",
                          "rows_per_batch": cfg.recipe.rows_per_batch, "score_chunk": SCORE_CHUNK}
        reference, student = cfg.load_reference(cfg.data_root), cfg.load_reference(cfg.data_root)
        rec["start"] = start_check(reference, arms["train"], n=cfg.start_rows)
        rec["reference_val"] = score(reference, arms["val"])
        rec["fit"] = fit(student, reference, arms["train"], cfg.recipe)
        rec["student_val"] = score(student, arms["val"])
        rec["verdict"] = verdict(rec["student_val"], rec["reference_val"])
        rec["student"] = cfg.save_student(student, cfg.student)

    return _terminal(cfg.evidence_fit, rec, body)


def main(argv: Optional[Sequence[str]] = None, cfg: Optional[Config] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("what", choices=("label", "fit"))
    run.add_argument("--clone-root")
    a = ap.parse_args(argv)
    try:
        if a.what == "label":
            if not a.clone_root:
                raise Refused("run label needs --clone-root")
            return run_label(a.clone_root, cfg)
        return run_fit(cfg)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
