"""Durable training replay capture (twixt-training-capture/1) -- the MINIMAL SERIAL PATH.

Pure and CPU-only: no MLX, search, model, worker or trainer import. Holds each searched decision's complete legal
order, integer visit counts before/after search and the counts passed to selection, with the generating model's
identity (a named-tensors/1 digest bound in the context); publishes a phase create-only (context, one envelope per
scheduled game, receipts, then the completion marker LAST) and reopens it strictly through the rules engine.
The serial driver is training_capture_serial.py. Deferred: workers/IPC, snapshot files, mirror/position-cap
transforms, the closeout tie-break, forced visits and the wider fault matrix (spec sections 9-10).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import struct
import time
import uuid
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .game.twixt_state import TwixtState

SCHEMA = "twixt-training-capture/1"
MODEL_DIGEST_VERSION = "named-tensors/1"
BOARD, CAP_PLIES = 24, 280
MAX_ROWS = CAP_PLIES                     # one row per played move; the empty start makes rows == final ply
CONTEXT_MAX_BYTES = 1 << 20              # spec section 6
ENVELOPE_MAX_BYTES = 8 << 20             # spec section 9: the per-game ceiling
SOURCE_KINDS = ("native_self_play",)
ADJUSTMENT_KINDS = ("closeout_tiebreak",)
COLOURS = ("red", "black")
PHASE_FILES = ("context.json", "receipts.jsonl", "finalized.json")


class CaptureError(RuntimeError):
    """The capture cannot be completed faithfully; no completion marker is written."""


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_digest(state) -> str:
    body = {"board_size": state.board_size, "active_size": state.active_size, "to_move": state.to_move,
            "ply": state.ply, "max_plies_limit": state.max_plies_limit,
            "pegs": sorted([r, c, col] for (r, c), col in state.pegs.items()),
            "bridges": sorted(sorted([list(a), list(b)]) for a, b in state.bridges)}
    return sha256_hex(canonical_bytes(body))


def model_state_digest(tensors: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(MODEL_DIGEST_VERSION.encode("utf-8") + b"\x00")
    for name in sorted(tensors):
        a = tensors[name]
        if not isinstance(a, np.ndarray) or a.dtype.fields is not None or a.dtype.kind not in "biuf":
            raise CaptureError(f"{name}: unsupported dtype")
        dt = a.dtype.newbyteorder("<") if a.dtype.itemsize > 1 else a.dtype
        data = np.ascontiguousarray(a, dtype=dt).tobytes(order="C")
        header = canonical_bytes({"name": name, "shape": list(a.shape), "dtype": dt.str})
        h.update(struct.pack("<Q", len(header)) + header + struct.pack("<Q", len(data)) + data)
    return h.hexdigest()


def encode_rng_state(rng) -> dict:
    version, internal, gauss_next = rng.getstate()              # observation: draws nothing
    return {"algorithm": "python-random", "python_version": platform.python_version(),
            "state_version": version, "internal_state": list(internal), "gauss_next": gauss_next}


def policy_target(row) -> List[Fraction]:
    total = sum(row["child_visits_after"])
    return [Fraction(v, total) for v in row["child_visits_after"]]


def _counts(mapping, legal, what) -> List[int]:
    extra = set(mapping) - set(legal)
    if extra:
        raise CaptureError(f"{what}: moves outside the legal set")
    out = [mapping.get(m, 0) for m in legal]
    if any(type(v) is not int or v < 0 for v in out):
        raise CaptureError(f"{what}: counts must be non-negative ints")
    return out


def _check_adjustment(after, sel, adj, legal) -> None:
    """The adjustment must EXPLAIN the change: only its target differs, raised to max(search) + 1 (the
    closeout tie-break's exact transformation, mcts.apply_closeout_selection_tiebreak)."""
    if adj is None:
        if sel != after:
            raise CaptureError("selection differs from search with no recorded adjustment")
        return
    if adj.get("kind") not in ADJUSTMENT_KINDS or adj.get("overrode_to") not in ("endpoint", "reducer"):
        raise CaptureError("adjustment: unsupported kind")
    if adj.get("target_move") not in legal:
        raise CaptureError("adjustment: target is not a legal move")
    expected = list(after)
    expected[legal.index(adj["target_move"])] = max(after) + 1
    if sel != expected:
        raise CaptureError("adjustment does not explain the selection counts")


def _hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _pos_int(v) -> bool:
    return type(v) is int and v > 0


def validate_context(ctx) -> None:
    """The fields the serial path relies on. A data field is not an authorization check."""
    if not isinstance(ctx, dict) or ctx.get("schema") != SCHEMA or ctx.get("source_kind") not in SOURCE_KINDS:
        raise CaptureError("context: schema or source kind")
    ids = ctx.get("scheduled_game_ids")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) \
            or len(set(ids)) != len(ids):
        raise CaptureError("context: scheduled_game_ids must be distinct non-negative ints")
    models = ctx.get("models")
    if not isinstance(models, dict) or not models or any(
            not isinstance(m, dict) or m.get("digest_version") != MODEL_DIGEST_VERSION
            or not _hex64(m.get("state_digest")) for m in models.values()):
        raise CaptureError("context: every model needs a named-tensors/1 state digest")
    games = ctx.get("games")
    if not isinstance(games, dict) or set(games) != {str(i) for i in ids}:
        raise CaptureError("context: games must describe exactly the scheduled ids")
    for gid, g in games.items():
        mbc = g.get("models_by_colour")
        if g.get("start_player") not in COLOURS or g.get("learner_player") not in (None, *COLOURS) \
                or not isinstance(mbc, dict) or set(mbc) != set(COLOURS) or set(mbc.values()) - set(models):
            raise CaptureError(f"context: game {gid} start, learner or model binding")
    if not _pos_int((ctx.get("configuration") or {}).get("n_simulations")):
        raise CaptureError("context: configuration.n_simulations")
    lim = ctx.get("limits") or {}
    if not (_pos_int(lim.get("game_bytes")) and lim["game_bytes"] <= ENVELOPE_MAX_BYTES
            and _pos_int(lim.get("phase_bytes"))):
        raise CaptureError(f"context: limits.game_bytes (<= {ENVELOPE_MAX_BYTES}) and limits.phase_bytes")


def context_model_digests(ctx) -> Dict[str, str]:
    return {mid: m["state_digest"] for mid, m in ctx["models"].items()}


def _reason(final) -> str:
    if final.winner() is not None:
        return "win"
    if final.ply >= CAP_PLIES:
        return "state_cap"
    if not final.legal_moves():
        return "board_full"
    raise CaptureError("the history is not a completed game")


class DecisionCapture:
    def __init__(self, *, context, context_sha256, game_id, game_rng):
        if game_id not in context["scheduled_game_ids"]:
            raise CaptureError(f"game {game_id} is not scheduled")
        g = context["games"][str(game_id)]
        self.context_sha256, self.game_id = context_sha256, game_id
        self.n_simulations = context["configuration"]["n_simulations"]
        self.models_by_colour, self.learner_player = dict(g["models_by_colour"]), g["learner_player"]
        self.rng_at_entry = encode_rng_state(game_rng)
        self.rows: List[dict] = []
        self._open: Optional[dict] = None
        self._legal: List[tuple] = []
        self._refs: Dict[int, tuple] = {}
        self.envelope: Optional[dict] = None

    def _need(self) -> dict:
        if self._open is None:
            raise CaptureError("no decision is open")
        return self._open

    def begin_decision(self, *, state, child_visits_before, root_visits_before):
        if self._open is not None or self.envelope is not None:
            raise CaptureError("a decision is open, or the game is finished")
        if state.ply != len(self.rows) or len(self.rows) >= MAX_ROWS:
            raise CaptureError("row count does not match the ply, or exceeds the checkpoint bound")
        self._legal = [tuple(m) for m in state.legal_moves()]
        mover = state.to_move
        self._open = {"row_id": f"{self.game_id}:{state.ply}", "ply": state.ply, "to_move": mover,
                      "model_id": self.models_by_colour[mover],
                      "learner_side": self.learner_player is None or mover == self.learner_player,
                      "state_sha256": state_digest(state), "legal_moves": [list(m) for m in self._legal],
                      "child_visits_before": _counts(child_visits_before, self._legal, "before"),
                      "root_visits_before": int(root_visits_before)}

    def after_search(self, *, visit_counts, root_visits_after, root_value):
        row = self._need()
        if set(visit_counts) != set(self._legal):
            raise CaptureError("search counts do not cover exactly the legal set")
        after = _counts(visit_counts, self._legal, "after")
        if sum(after) == 0 or any(a < b for a, b in zip(after, row["child_visits_before"])):
            raise CaptureError("zero after-mass, or a child count decreased")
        if int(root_visits_after) - row["root_visits_before"] != self.n_simulations:
            raise CaptureError(f"the root gained {int(root_visits_after) - row['root_visits_before']} visits, "
                               f"not the configured {self.n_simulations}")
        rv = float(root_value)
        if not math.isfinite(rv):
            raise CaptureError("non-finite root value")
        row.update(child_visits_after=after, root_visits_after=int(root_visits_after), root_value=rv)

    def note_training_row(self, obj, transform="identity"):
        row = self._need()
        if transform != "identity" or not row["learner_side"] or id(obj) in self._refs:
            raise CaptureError("bad, repeated or opponent-side training-row reference")
        self._refs[id(obj)] = (obj, row["row_id"], transform)

    def before_selection(self, *, selection_counts, adjustment):
        row = self._need()
        if "child_visits_after" not in row or set(selection_counts) != set(self._legal):
            raise CaptureError("selection before search, or not over the legal set")
        sel = _counts(selection_counts, self._legal, "selection")
        adj = None if adjustment is None else json.loads(canonical_bytes(adjustment))
        _check_adjustment(row["child_visits_after"], sel, adj, row["legal_moves"])
        row.update(selection_counts=sel, selection_adjustment=adj)

    def after_selection(self, *, move):
        row = self._need()
        if "selection_counts" not in row or tuple(move) not in self._legal:
            raise CaptureError("selection not observed, or an illegal move")
        row["selected_move"] = [int(move[0]), int(move[1])]
        self.rows.append(row)
        self._open = None

    def finish(self, *, move_history, start_player, final_state, kept_positions) -> dict:
        if self._open is not None or self.envelope is not None:
            raise CaptureError("finish: a decision is open or the game was already finished")
        winner, reason = final_state.winner(), _reason(final_state)
        if len(move_history) != len(self.rows) or final_state.ply != len(self.rows):
            raise CaptureError("finish: rows, moves and final ply disagree")
        for row in self.rows:
            row["z"] = 0 if winner is None else (1 if row["to_move"] == winner else -1)
        members = []
        for obj in kept_positions:
            ref = self._refs.get(id(obj))
            if ref is None or ref[0] is not obj:
                raise CaptureError("a retained training row has no captured source")
            members.append({"source_row_id": ref[1], "transform": ref[2]})
        self.envelope = {"schema": SCHEMA, "context_sha256": self.context_sha256, "game_id": self.game_id,
                         "start_player": start_player, "models_by_colour": self.models_by_colour,
                         "learner_player": self.learner_player, "rng_at_entry": self.rng_at_entry,
                         "move_history": [[int(m[0]), int(m[1])] for m in move_history], "rows": self.rows,
                         "outcome": {"winner": winner, "reason": reason, "final_ply": final_state.ply,
                                     "final_state_sha256": state_digest(final_state)},
                         "buffer_membership": members}
        return self.envelope


def _no_constants(name):
    raise CaptureError(f"non-finite constant {name}")


def validate_envelope(data: bytes, context, context_sha256: str) -> dict:
    if len(data) > context["limits"]["game_bytes"] or sha256_hex(canonical_bytes(context)) != context_sha256:
        raise CaptureError("envelope too large, or context hash mismatch")
    env = json.loads(data.decode("utf-8"), parse_constant=_no_constants)
    if env.get("schema") != SCHEMA or env.get("context_sha256") != context_sha256:
        raise CaptureError("envelope schema or context binding is wrong")
    gid = env.get("game_id")
    if gid not in context["scheduled_game_ids"]:
        raise CaptureError("unscheduled game")
    g = context["games"][str(gid)]
    for key in ("start_player", "learner_player", "models_by_colour"):
        if env.get(key) != g[key]:
            raise CaptureError(f"{key} differs from the context")
    if set(env["models_by_colour"].values()) - set(context["models"]):
        raise CaptureError("a model not bound in the context")
    hist, rows = [tuple(m) for m in env["move_history"]], env["rows"]
    if not 0 < len(rows) <= MAX_ROWS or len(rows) != len(hist):
        raise CaptureError("rows are missing or extra")
    s = TwixtState(active_size=BOARD, to_move=env["start_player"], max_plies_limit=CAP_PLIES)
    for i, (row, mv) in enumerate(zip(rows, hist)):
        legal = [list(m) for m in s.legal_moves()]
        if row.get("row_id") != f"{gid}:{i}" or row.get("ply") != i or row.get("legal_moves") != legal:
            raise CaptureError(f"row {i}: identity, ply or legal order differs from the replay")
        for k in ("child_visits_before", "child_visits_after", "selection_counts"):
            v = row.get(k)
            if not isinstance(v, list) or len(v) != len(legal) or any(type(x) is not int or x < 0 for x in v):
                raise CaptureError(f"row {i}: {k} is not {len(legal)} non-negative ints")
        if any(a < b for a, b in zip(row["child_visits_after"], row["child_visits_before"])) \
                or sum(row["child_visits_after"]) == 0:
            raise CaptureError(f"row {i}: a decreasing count or zero after-mass")
        for k in ("root_visits_before", "root_visits_after"):
            if type(row.get(k)) is not int or row[k] < 0:
                raise CaptureError(f"row {i}: {k} is not a non-negative int")
        if row["root_visits_after"] - row["root_visits_before"] != context["configuration"]["n_simulations"]:
            raise CaptureError(f"row {i}: the root's visit gain is not the configured simulation count")
        if type(row.get("root_value")) is not float or not math.isfinite(row["root_value"]):
            raise CaptureError(f"row {i}: root value")
        if row.get("state_sha256") != state_digest(s) or row.get("to_move") != s.to_move:
            raise CaptureError(f"row {i}: state or mover differs from the replay")
        if row.get("model_id") != g["models_by_colour"][s.to_move] or row.get("learner_side") is not (
                g["learner_player"] is None or s.to_move == g["learner_player"]):
            raise CaptureError(f"row {i}: model attribution or learner side differs")
        _check_adjustment(row["child_visits_after"], row["selection_counts"], row.get("selection_adjustment"), legal)
        if row.get("selected_move") != list(mv) or list(mv) not in legal:
            raise CaptureError(f"row {i}: selected move is not the played, legal move")
        s = s.apply_move(mv)
    out, winner = env["outcome"], s.winner()
    if out != {"winner": winner, "reason": _reason(s), "final_ply": s.ply, "final_state_sha256": state_digest(s)}:
        raise CaptureError("outcome differs from the replay")
    for r in rows:
        if type(r.get("z")) is not int or r["z"] != (0 if winner is None else (1 if r["to_move"] == winner else -1)):
            raise CaptureError(f"row {r['row_id']}: z differs from the completed game")
    by_id, seen = {r["row_id"]: r for r in rows}, set()
    for m in env["buffer_membership"]:
        src = by_id.get(m.get("source_row_id"))
        if src is None or not src["learner_side"] or m.get("transform") != "identity" or m["source_row_id"] in seen:
            raise CaptureError("membership names a missing, opponent, repeated or transformed row")
        seen.add(m["source_row_id"])
    return env


def _publish(root: str, rel: str, data: bytes) -> str:
    final = os.path.join(root, rel)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    staging = os.path.join(os.path.dirname(final), f".staging-{uuid.uuid4().hex}")
    try:
        fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(staging, final)                               # no-replace publication
        except FileExistsError:
            raise CaptureError(f"{rel}: the final path already exists") from None
    finally:
        if os.path.exists(staging):
            os.unlink(staging)
    return sha256_hex(data)


def _read(root: str, rel: str, limit: int) -> bytes:
    with open(os.path.join(root, rel), "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise CaptureError(f"{rel}: exceeds {limit} bytes")
    return data


def _game_path(game_id: int) -> str:
    return f"games/game_{game_id}.json"


class PhaseWriter:
    """One capture phase, written serially: context first, each game as it finishes, the marker LAST.
    Nothing is overwritten; a phase that cannot finalize stays visibly incomplete (no finalized.json)."""

    def __init__(self, capture_dir: str, context):
        validate_context(context)
        ctx_bytes = canonical_bytes(context)
        if len(ctx_bytes) > CONTEXT_MAX_BYTES:
            raise CaptureError("context exceeds its ceiling")
        if os.path.lexists(capture_dir):
            raise CaptureError("capture directory already exists")
        os.makedirs(capture_dir)
        self.dir, self.context = capture_dir, json.loads(ctx_bytes)
        self.context_sha256 = _publish(capture_dir, "context.json", ctx_bytes)
        self.bytes_used, self.receipts, self.finalized = len(ctx_bytes), {}, False
        self.timing = {"write_s": 0.0, "reopen_validate_s": 0.0}

    def publish_game(self, envelope) -> dict:
        if self.finalized:
            raise CaptureError("the phase is finalized")
        gid = envelope.get("game_id") if isinstance(envelope, dict) else None
        if gid not in self.context["scheduled_game_ids"] or gid in self.receipts:
            raise CaptureError(f"game {gid!r} is not scheduled or was already published")
        data = canonical_bytes(envelope)
        if self.bytes_used + len(data) > self.context["limits"]["phase_bytes"]:
            raise CaptureError("the phase byte limit would be exceeded")
        t0 = time.perf_counter()
        validate_envelope(data, self.context, self.context_sha256)            # before anything is published
        rel = _game_path(gid)
        sha = _publish(self.dir, rel, data)
        t1 = time.perf_counter()
        reread = _read(self.dir, rel, self.context["limits"]["game_bytes"])
        if reread != data:
            raise CaptureError(f"{rel}: the published bytes differ")
        env = validate_envelope(reread, self.context, self.context_sha256)
        t2 = time.perf_counter()
        self.timing["write_s"] += t1 - t0
        self.timing["reopen_validate_s"] += t2 - t1
        self.bytes_used += len(data)
        self.receipts[gid] = {"game_id": gid, "path": rel, "sha256": sha, "bytes": len(data), "rows": len(env["rows"])}
        return self.receipts[gid]

    def finalize(self, live_model_digests: Mapping[str, str]) -> dict:
        """Only when every scheduled game has exactly one receipt and the live models still hash to the digests
        bound in the context: receipts.jsonl, then finalized.json."""
        if self.finalized:
            raise CaptureError("the phase is already finalized")
        missing = [g for g in self.context["scheduled_game_ids"] if g not in self.receipts]
        if missing:
            raise CaptureError(f"scheduled games without a published envelope: {missing}")
        if dict(live_model_digests) != context_model_digests(self.context):
            raise CaptureError("a live model no longer matches the digest bound in the context")
        receipts = b"".join(canonical_bytes(self.receipts[g]) + b"\n" for g in sorted(self.receipts))
        marker = canonical_bytes({"schema": SCHEMA, "context_sha256": self.context_sha256,
                                  "receipts_sha256": sha256_hex(receipts), "games": len(self.receipts),
                                  "rows": sum(r["rows"] for r in self.receipts.values())})
        if self.bytes_used + len(receipts) + len(marker) > self.context["limits"]["phase_bytes"]:
            raise CaptureError("the phase byte limit would be exceeded")
        _publish(self.dir, "receipts.jsonl", receipts)
        _publish(self.dir, "finalized.json", marker)
        self.bytes_used += len(receipts) + len(marker)
        self.finalized = True
        return {"bytes": self.bytes_used, "games": len(self.receipts), **self.timing}


def load_phase(capture_dir: str):
    """THE STRICT READER: a finalized phase only. Returns (context, envelopes in scheduled order)."""
    top = set(os.listdir(capture_dir))
    if top != {*PHASE_FILES, "games"}:
        raise CaptureError(f"unexpected or missing files: {sorted(top ^ {*PHASE_FILES, 'games'})}")
    marker = json.loads(_read(capture_dir, "finalized.json", CONTEXT_MAX_BYTES), parse_constant=_no_constants)
    ctx_bytes = _read(capture_dir, "context.json", CONTEXT_MAX_BYTES)
    if marker.get("schema") != SCHEMA or sha256_hex(ctx_bytes) != marker.get("context_sha256"):
        raise CaptureError("the context differs from the completion marker")
    ctx = json.loads(ctx_bytes, parse_constant=_no_constants)
    validate_context(ctx)
    ids = ctx["scheduled_game_ids"]
    rec_bytes = _read(capture_dir, "receipts.jsonl", len(ids) * 1024)
    if sha256_hex(rec_bytes) != marker.get("receipts_sha256"):
        raise CaptureError("the receipts differ from the completion marker")
    receipts = [json.loads(line) for line in rec_bytes.decode("utf-8").splitlines()]
    if [r.get("game_id") for r in receipts] != sorted(ids) or marker.get("games") != len(ids):
        raise CaptureError("receipts are not exactly one per scheduled game")
    if set(os.listdir(os.path.join(capture_dir, "games"))) != {os.path.basename(_game_path(g)) for g in ids}:
        raise CaptureError("games/ holds unlisted or missing files")
    total, rows, envs = len(ctx_bytes) + len(rec_bytes) + len(canonical_bytes(marker)), 0, {}
    for r in receipts:
        if r.get("path") != _game_path(r["game_id"]):
            raise CaptureError(f"receipt {r['game_id']}: unexpected path")
        data = _read(capture_dir, r["path"], ctx["limits"]["game_bytes"])
        if len(data) != r.get("bytes") or sha256_hex(data) != r.get("sha256"):
            raise CaptureError(f"{r['path']}: bytes differ from the receipt")
        env = validate_envelope(data, ctx, marker["context_sha256"])
        if env["game_id"] != r["game_id"] or len(env["rows"]) != r.get("rows"):
            raise CaptureError(f"{r['path']}: identity or rows differ from the receipt")
        total, rows, envs[r["game_id"]] = total + len(data), rows + len(env["rows"]), env
    if total > ctx["limits"]["phase_bytes"] or rows != marker.get("rows"):
        raise CaptureError("phase bytes or row total")
    return ctx, [envs[g] for g in ids]
