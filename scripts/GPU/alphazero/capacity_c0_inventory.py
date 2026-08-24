"""C0 read-only inventory of a self-play game-replay directory.

Capacity-headroom probe, stage C0. This module NEVER loads a model, runs
inference, computes a loss, or writes into a corpus. It reads game JSON and
emits a structural inventory; `verify` recomputes that inventory from the live
corpus and compares it byte-exactly to a stored one.

Why it exists: the replay directories are untracked (`.gitignore`), carry no
field naming the network that produced them, and share one flat
`iter_NNNN_game_NNN.json` namespace across runs -- so a later run silently
overwrites the start of an earlier one. That has already happened. Nothing can
be pinned to "the corpus" without a content digest computed outside the corpus.

The single code path is `inventory()`. `emit` writes its output; `verify`
recomputes it and compares. There is no separate verification logic that could
drift from the thing it verifies.

Exit codes (fail closed):
    0  ok
    2  usage / precondition (missing root, refusing to overwrite, unreadable)
    3  MISMATCH between the stored inventory and the live corpus

stdlib only: no MLX, no numpy, no game engine.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1

# Frozen eligibility predicate (see the C0 card). Games outside it are counted
# but excluded from the eligible digest and from the split/leakage figures.
# The third clause -- provenance -- is not a hardcoded iteration number; it is
# derived from the data by `intact_suffix()` below.
ELIGIBLE_REASON = "win"
ELIGIBLE_BOARD_SIZE = 24

# Frozen outcome-blind whole-game split. The key is the game id, which is
# derived from (iteration, game_idx) alone -- it cannot encode the winner.
SPLIT_BOUNDS = (("train", 80), ("val", 90), ("test", 100))

# Leakage is reported per ply up to this depth and aggregated beyond it.
LEAK_PLY_DETAIL = 16


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_of(pairs) -> str:
    """Order-independent digest of (name, sha256) pairs."""
    h = hashlib.sha256()
    for name, sha in sorted(pairs):
        h.update(f"{name} {sha}\n".encode())
    return h.hexdigest()


def intact_suffix(windows):
    """Longest run of iterations, ending at the highest one, that was written
    once in increasing iteration order.

    `windows` maps iteration -> (min_timestamp, max_timestamp) over its games.
    A directory holding one training run has every iteration finishing before
    the next one starts. `iter_NNNN_game_NNN.json` is a flat namespace shared by
    every run that ever wrote here, so a later, shorter run silently replaces
    the low iterations of an earlier one -- and the replacement announces itself
    as a timestamp that runs backwards against its neighbour. Taking the
    increasing suffix keeps the part of the original run that survived and drops
    everything a later run landed on, without naming a cutoff by hand.

    Returns (first_iteration, last_iteration) or None if there are no games.
    """
    its = sorted(windows)
    if not its:
        return None
    k = its[-1]
    for i in range(len(its) - 1, 0, -1):
        if windows[its[i - 1]][1] > windows[its[i]][0]:
            break
        k = its[i - 1]
    return (k, its[-1])


def split_of(game_id: str) -> str:
    """Outcome-blind whole-game split. Deterministic in the game id only."""
    b = int(hashlib.sha1(game_id.encode()).hexdigest()[:8], 16) % 100
    for name, upper in SPLIT_BOUNDS:
        if b < upper:
            return name
    raise AssertionError("split bounds must cover 0..99")


def _prefix_keys(moves):
    """Per-ply key for the position BEFORE each move, exact and order-sensitive.

    Two positions reached by different move orders get different keys, so a
    match is a genuine repeat and the resulting overlap is a LOWER bound on
    true state-level overlap.
    """
    h = hashlib.sha1()
    out = []
    for (r, c) in moves:
        out.append(h.hexdigest())
        h.update(f"{r},{c};".encode())
    return out


def _pegset_keys(moves_with_player):
    """Per-ply key ignoring move order: placed pegs by colour plus side to move.

    Bridges are created greedily in knight-offset order and blocked by crossing
    (`scripts/GPU/game/bridge.py`), so equal peg sets can carry different link
    sets. Treating them as identical OVER-counts, giving an UPPER bound.
    """
    red, blk, out = [], [], []
    for (pl, r, c) in moves_with_player:
        out.append(hashlib.sha1(
            repr((tuple(sorted(red)), tuple(sorted(blk)), pl)).encode()
        ).hexdigest())
        (red if pl == "red" else blk).append((r, c))
    return out


def _leakage(elig, keyfn):
    train, tot, seen = set(), 0, 0
    by_tot = collections.Counter()
    by_seen = collections.Counter()
    for g in elig:
        if g["split"] == "train":
            train.update(keyfn(g))
    for g in elig:
        if g["split"] != "test":
            continue
        for i, k in enumerate(keyfn(g)):
            tot += 1
            by_tot[i] += 1
            if k in train:
                seen += 1
                by_seen[i] += 1
    detail = {
        str(i): [by_seen[i], by_tot[i]]
        for i in range(LEAK_PLY_DETAIL) if by_tot[i]
    }
    deep_seen = sum(v for i, v in by_seen.items() if i >= LEAK_PLY_DETAIL)
    deep_tot = sum(v for i, v in by_tot.items() if i >= LEAK_PLY_DETAIL)
    first_clean = next(
        (i for i in sorted(by_tot) if by_seen[i] == 0), None
    )
    return {
        "test_positions": tot,
        "matched_in_train": seen,
        "by_ply": detail,
        f"ply_ge_{LEAK_PLY_DETAIL}": [deep_seen, deep_tot],
        "first_ply_with_zero_overlap": first_clean,
    }


def inventory(root: str) -> dict:
    """Recompute the full structural inventory of one replay directory.

    The result is already canonical (see `_canonical`), so `emit` writes it
    unchanged and `verify` can compare it directly to what it reads back.
    """
    return _canonical(_inventory(root))


def _inventory(root: str) -> dict:
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        raise FileNotFoundError(f"no *.json under {root}")

    file_pairs = [(os.path.basename(p), _sha256_file(p)) for p in files]
    counters = {k: collections.Counter() for k in (
        "reason", "winner", "starting_player", "board_size", "simulations",
        "mode", "config_hash", "depth", "move_key_signature",
        "top_key_signature", "timestamp_date",
    )}
    iterations = collections.Counter()
    ids = collections.Counter()
    seq_hashes = collections.Counter()
    stats_files = 0
    games = 0
    positions = 0
    n_moves_mismatch = 0
    turn_sequence_anomaly = 0
    files_with_visit_counts = 0
    all_games = []
    windows = {}

    for p in files:
        base = os.path.basename(p)
        if base.endswith("_stats.json"):
            stats_files += 1
            continue
        with open(p, "rb") as fh:
            raw = fh.read()
        g = json.loads(raw)
        meta = g.get("meta") or {}
        moves = g.get("moves") or []
        games += 1
        positions += len(moves)
        ids[g.get("id")] += 1
        iterations[meta.get("iteration")] += 1
        counters["reason"][meta.get("reason")] += 1
        counters["winner"][g.get("winner")] += 1
        counters["starting_player"][g.get("starting_player")] += 1
        counters["board_size"][meta.get("board_size")] += 1
        counters["simulations"][meta.get("simulations")] += 1
        counters["mode"][meta.get("mode")] += 1
        counters["config_hash"][g.get("config_hash")] += 1
        counters["depth"][g.get("depth")] += 1
        counters["timestamp_date"][(g.get("timestamp") or "")[:10]] += 1
        counters["top_key_signature"]["|".join(sorted(g.keys()))] += 1
        if moves:
            counters["move_key_signature"]["|".join(sorted(moves[0].keys()))] += 1
        if meta.get("n_moves") != len(moves):
            n_moves_mismatch += 1
        if [m.get("turn") for m in moves] != list(range(1, len(moves) + 1)):
            turn_sequence_anomaly += 1
        # The training policy target is a per-move visit distribution, and its
        # absence is the load-bearing C0 fact -- so this counts it over every
        # file's own bytes rather than sampling, and rather than reasoning about
        # what the writer is supposed to emit. The match is a bare substring, so
        # it OVER-detects (any key merely containing "policy" counts): a total of
        # zero is therefore conclusive, a non-zero total needs opening.
        if b'"visit_counts"' in raw or b'"visits"' in raw or b'"policy"' in raw:
            files_with_visit_counts += 1
        seq_hashes[hashlib.sha1(
            (str(g.get("starting_player")) + "|" +
             ";".join(f"{m.get('row')},{m.get('col')}" for m in moves)).encode()
        ).hexdigest()] += 1
        it = meta.get("iteration")
        ts = g.get("timestamp") or ""
        lo, hi = windows.get(it, (ts, ts))
        windows[it] = (min(lo, ts), max(hi, ts))
        all_games.append({
            "id": g["id"],
            "file": base,
            "iteration": it,
            "reason": meta.get("reason"),
            "board_size": meta.get("board_size"),
            "winner": g.get("winner"),
            "split": split_of(g["id"]),
            "moves": [(m["row"], m["col"]) for m in moves],
            "moves_pl": [(m["player"], m["row"], m["col"]) for m in moves],
        })

    suffix = intact_suffix(windows)
    elig = [
        g for g in all_games
        if g["reason"] == ELIGIBLE_REASON
        and g["board_size"] == ELIGIBLE_BOARD_SIZE
        and suffix is not None
        and suffix[0] <= g["iteration"] <= suffix[1]
    ]
    excluded_by_provenance = sorted(
        it for it in windows if suffix is None or not (suffix[0] <= it <= suffix[1])
    )

    by_name = dict(file_pairs)
    elig_digest = _digest_of((g["file"], by_name[g["file"]]) for g in elig)

    split_stats = {}
    for name, _ in SPLIT_BOUNDS:
        sub = [g for g in elig if g["split"] == name]
        w = collections.Counter(g["winner"] for g in sub)
        split_stats[name] = {
            "games": len(sub),
            "positions": sum(len(g["moves"]) for g in sub),
            "winner": dict(sorted(w.items())),
            "iterations_present": len(set(g["iteration"] for g in sub)),
            "min_games_per_iteration": (
                min(collections.Counter(g["iteration"] for g in sub).values()) if sub else 0
            ),
        }

    plies = sorted(len(g["moves"]) for g in elig)

    def pct(q):
        return plies[min(len(plies) - 1, int(q * len(plies)))] if plies else None

    return {
        "schema_version": SCHEMA_VERSION,
        "root": os.path.normpath(root),
        "files": {
            "n_json": len(files),
            "n_stats": stats_files,
            "n_games": games,
            "total_bytes": sum(os.path.getsize(p) for p in files),
            "directory_digest": _digest_of(file_pairs),
        },
        "distributions": {
            k: dict(sorted(((str(a), b) for a, b in v.items()), key=lambda x: x[0]))
            for k, v in counters.items()
        },
        "iterations": {
            "count": len(iterations),
            "min": min(iterations) if iterations else None,
            "max": max(iterations) if iterations else None,
            "games_per_iteration": dict(sorted(
                collections.Counter(iterations.values()).items())),
        },
        "structural": {
            "positions_total": positions,
            "duplicate_ids_extra": sum(v - 1 for v in ids.values() if v > 1),
            "duplicate_move_sequences_extra": sum(v - 1 for v in seq_hashes.values() if v > 1),
            "distinct_move_sequences": len(seq_hashes),
            "n_moves_field_mismatch": n_moves_mismatch,
            "turn_sequence_anomaly": turn_sequence_anomaly,
            "files_with_policy_targets": files_with_visit_counts,
        },
        "provenance": {
            "rule": "longest increasing-timestamp suffix of iterations",
            "intact_iterations": list(suffix) if suffix else None,
            "excluded_iterations": excluded_by_provenance,
            "excluded_games": sum(
                1 for g in all_games if g["iteration"] in set(excluded_by_provenance)),
            "iteration_timestamp_windows": {
                str(it): list(windows[it]) for it in sorted(windows)
            },
        },
        "eligible": {
            "predicate": {
                "reason": ELIGIBLE_REASON,
                "board_size": ELIGIBLE_BOARD_SIZE,
                "iteration_in_intact_suffix": True,
            },
            "games": len(elig),
            "positions": sum(len(g["moves"]) for g in elig),
            "iterations_present": len(set(g["iteration"] for g in elig)),
            "iteration_min": min((g["iteration"] for g in elig), default=None),
            "iteration_max": max((g["iteration"] for g in elig), default=None),
            "winner": dict(sorted(collections.Counter(g["winner"] for g in elig).items())),
            "plies_per_game": {
                "min": plies[0] if plies else None, "p05": pct(0.05),
                "median": pct(0.50), "p95": pct(0.95),
                "max": plies[-1] if plies else None,
            },
            "digest": elig_digest,
        },
        "split": {"rule": "sha1(game_id) % 100", "bounds": dict(SPLIT_BOUNDS), "arms": split_stats},
        "leakage_lower_bound_exact_prefix": _leakage(
            elig, lambda g: _prefix_keys(g["moves"])),
        "leakage_upper_bound_pegset": _leakage(
            elig, lambda g: _pegset_keys(g["moves_pl"])),
    }


def _canonical(obj):
    """Put a value into the form it will have after a JSON round trip.

    `emit` writes JSON and `verify` compares against what was read back, so the
    live inventory has to be normalised the same way or the comparison trips on
    representation rather than content: JSON has no integer keys and no tuples.
    Doing it once here covers every field, including ones added later.
    """
    return json.loads(json.dumps(obj, sort_keys=True))


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: absent in stored, live={b[k]!r}")
            elif k not in b:
                out.append(f"{path}/{k}: stored={a[k]!r}, absent live")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
    elif a != b:
        out.append(f"{path}: stored={a!r} live={b!r}")
    return out


def _selftest() -> int:
    """Assertions for the two pieces of non-obvious logic: the provenance rule
    and the mismatch comparison. Run as a subprocess, not imported."""
    W = lambda pairs: {i: (a, b) for i, (a, b) in pairs}

    assert intact_suffix({}) is None
    assert intact_suffix(W([(7, ("a", "b"))])) == (7, 7)
    # clean single run -> everything is intact
    assert intact_suffix(W([(0, ("a", "b")), (1, ("c", "d")), (2, ("e", "f"))])) == (0, 2)
    # iteration 0 rewritten by a later run -> dropped, 1..2 kept
    assert intact_suffix(W([(0, ("y", "z")), (1, ("c", "d")), (2, ("e", "f"))])) == (1, 2)
    # break in the middle -> only the part after it survives
    assert intact_suffix(W([(0, ("a", "b")), (1, ("y", "z")), (2, ("e", "f"))])) == (2, 2)
    # touching windows (one ends exactly when the next starts) are still intact
    assert intact_suffix(W([(0, ("a", "c")), (1, ("c", "d"))])) == (0, 1)
    # the shape actually on disk: a short recent run landed on the low iterations
    real = W([(i, (f"2026-06-2{i}", f"2026-06-2{i}")) for i in range(3)] +
             [(i, (f"2026-04-{i:02d}", f"2026-04-{i:02d}")) for i in range(3, 10)])
    assert intact_suffix(real) == (3, 9), intact_suffix(real)

    # split is a total function of the id alone, and covers all three arms
    arms = {split_of(f"iter_{i:04d}_game_{j:03d}") for i in range(40) for j in range(40)}
    assert arms == {"train", "val", "test"}, arms
    assert split_of("iter_0200_game_000") == split_of("iter_0200_game_000")

    # Regression: `verify` compares a live dict to one read back from JSON, and
    # JSON has no integer keys or tuples. Before `_canonical` this raised
    # TypeError instead of reporting a difference -- a gate that crashes is not
    # a gate. Canonicalising must make a round trip a no-op.
    raw = {"n": {1: 2}, "t": (3, 4), "s": {"k": "v"}}
    canon = _canonical(raw)
    assert canon == {"n": {"1": 2}, "t": [3, 4], "s": {"k": "v"}}, canon
    assert _diff(canon, _canonical(canon)) == []
    assert _diff(json.loads(json.dumps(canon)), canon) == []

    # the comparison used by `verify` must catch a changed value AND a dropped key
    assert _diff({"a": {"b": 1}}, {"a": {"b": 1}}) == []
    assert _diff({"a": {"b": 1}}, {"a": {"b": 2}})
    assert _diff({"a": {"b": 1}}, {"a": {}})
    assert _diff({"a": {}}, {"a": {"b": 1}})

    # prefix keys are order-sensitive, peg-set keys are not: that is exactly
    # what makes one bound a floor and the other a ceiling
    assert _prefix_keys([(0, 0), (1, 1)]) != _prefix_keys([(1, 1), (0, 0)])
    assert (_pegset_keys([("red", 0, 0), ("black", 1, 1), ("red", 2, 2)])[2]
            == _pegset_keys([("red", 0, 0), ("black", 1, 1), ("red", 3, 3)])[2])
    print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="write a new inventory (refuses to overwrite)")
    e.add_argument("--root", required=True)
    e.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="recompute and compare to a stored inventory")
    v.add_argument("--root", required=True)
    v.add_argument("--inventory", required=True)
    sub.add_parser("selftest", help="assertions for the provenance and diff logic")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "selftest":
            return _selftest()
        if args.cmd == "emit":
            if os.path.exists(args.out):
                print(f"REFUSING: {args.out} exists; evidence is create-only", file=sys.stderr)
                return 2
            inv = inventory(args.root)
            with open(args.out, "w") as fh:
                json.dump(inv, fh, indent=1, sort_keys=True)
                fh.write("\n")
            print(f"wrote {args.out}")
            print(f"  directory_digest {inv['files']['directory_digest']}")
            print(f"  eligible.digest  {inv['eligible']['digest']}")
            return 0

        with open(args.inventory) as fh:
            stored = json.load(fh)
        live = inventory(args.root)
        # `root` is part of the compared dict, so pointing verify at a different
        # directory is a mismatch rather than a silent pass. Run from the repo
        # root so the recorded relative path reproduces.
        diffs = _diff(stored, live)
        if diffs:
            print(f"MISMATCH: {len(diffs)} difference(s) vs {args.inventory}", file=sys.stderr)
            for d in diffs[:20]:
                print(f"  {d}", file=sys.stderr)
            if len(diffs) > 20:
                print(f"  ... {len(diffs) - 20} more", file=sys.stderr)
            return 3
        print(f"OK: {args.root} matches {args.inventory}")
        print(f"  directory_digest {live['files']['directory_digest']}")
        return 0
    except FileNotFoundError as exc:
        print(f"PRECONDITION: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        print(f"PRECONDITION: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
