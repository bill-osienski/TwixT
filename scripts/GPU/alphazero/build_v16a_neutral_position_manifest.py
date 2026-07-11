"""Build a deterministic, GAME-HELD-OUT, NON-A-SELECTED neutral position manifest
for the v16a FPU collateral-damage screen.

Samples ordinary positions across mixed games/plies/sides/outcomes, EXCLUDING
entire games named in the A discovery manifest, so discovery and validation share
no games. Output conforms to the canonical position_probe_cases schema and flows
through the SAME trusted path (load_csv_manifest -> search_for_row ->
position_state), unmodified. Stratified (opening/early-mid/midgame/late),
game-first round-robin, per-game capped, min-ply-gap separated, ~50/50 side
balanced per bucket. Winner-null games are kept (game_result="unknown"): in the
default corpus all such games are state_cap 280-ply marathons -- the most
search-stressed, highest-value late samples.

READ-ONLY on replays; writes one CSV + sidecar meta JSON. No MCTS/network/train.
Building/running THIS builder is in scope for v16a; running the FPU sweep is NOT.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .goal_line_trigger_probe_cases import position_state

BUCKETS = (("opening", 1, 15), ("early_mid", 16, 40),
           ("midgame", 41, 90), ("late", 91, None))
BUCKET_ORDER = [name for name, _, _ in BUCKETS]
DEFAULT_QUOTAS = {"opening": 40, "early_mid": 100, "midgame": 100, "late": 100}
DEFAULT_PER_GAME_CAPS = {"opening": 1, "early_mid": 2, "midgame": 2, "late": 2}
DEFAULT_MIN_PLY_GAP = 8

NEUTRAL_FIELDNAMES = [
    "case_id", "game_idx", "replay_path", "position_ply", "side_to_move",
    "ply_bucket", "game_result", "total_game_plies", "source_replay", "sample_seed",
]


def side_to_move_for_ply(ply: int) -> str:
    """Ply 0 is red's opener; TwixtState alternates each ply. position_state
    asserts exactly this parity -> a mismatch fails loud at validation."""
    return "red" if ply % 2 == 0 else "black"


def bucket_for_ply(ply: int, buckets=BUCKETS):
    for name, lo, hi in buckets:
        if ply >= lo and (hi is None or ply <= hi):
            return name
    return None


def candidate_positions(game_records, buckets=BUCKETS):
    """{bucket -> {game_idx -> [valid plies asc]}}; valid iff in bucket AND
    0 <= ply < n_moves (position_state range)."""
    out = {name: {} for name, _, _ in buckets}
    for r in sorted(game_records, key=lambda x: x["game_idx"]):
        n = r["n_moves"]
        for name, lo, hi in buckets:
            hi_eff = (n - 1) if hi is None else min(hi, n - 1)
            plies = list(range(lo, hi_eff + 1))
            if plies:
                out[name][r["game_idx"]] = plies
    return out


def _pick_ply(cand, per_game_selected, min_gap, side_count, side_of):
    ok = [p for p in cand if all(abs(p - q) >= min_gap for q in per_game_selected)]
    if not ok:
        return None
    behind = min(side_count, key=lambda s: (side_count[s], s))
    for p in ok:
        if side_of(p) == behind:
            return p
    return ok[0]


def sample_bucket(pool_by_game, *, quota, cap, min_gap, seed,
                  side_of=side_to_move_for_ply, state_key_fn=None):
    """Game-first round-robin: pass 1 takes <=1 ply/game (covers every game),
    later passes add up to `cap`/game, each >= min_gap from that game's picks;
    side balanced toward 50/50; optional state_key_fn de-dupes across games.
    Deterministic. Returns (selected [(game,ply)], side_count)."""
    rng = random.Random(seed)
    games = sorted(pool_by_game)
    rng.shuffle(games)
    plies = {g: pool_by_game[g][:] for g in games}
    for g in games:
        rng.shuffle(plies[g])
    picked, selected, seen = {}, [], set()
    side_count = {"red": 0, "black": 0}
    progress = True
    while len(selected) < quota and progress:
        progress = False
        for g in games:
            if len(selected) >= quota:
                break
            if len(picked.get(g, [])) >= cap:
                continue
            cand = plies[g]
            if state_key_fn is not None:
                cand = [p for p in cand if state_key_fn(g, p) not in seen]
            chosen = _pick_ply(cand, picked.get(g, []), min_gap, side_count, side_of)
            if chosen is None:
                continue
            picked.setdefault(g, []).append(chosen)
            plies[g].remove(chosen)
            if state_key_fn is not None:
                seen.add(state_key_fn(g, chosen))
            selected.append((g, chosen))
            side_count[side_of(chosen)] += 1
            progress = True
    return selected, side_count


def sample_neutral_rows(game_records, *, base_seed, source_replay,
                        buckets=BUCKETS, quotas=DEFAULT_QUOTAS,
                        per_game_caps=DEFAULT_PER_GAME_CAPS,
                        min_ply_gap=DEFAULT_MIN_PLY_GAP,
                        state_key_fn_by_bucket=None):
    recs = {r["game_idx"]: r for r in game_records}
    pools = candidate_positions(game_records, buckets)
    key_by_bucket = state_key_fn_by_bucket or {}
    rows, stats = [], {}
    for offset, (name, _lo, _hi) in enumerate(buckets):
        sel, side_count = sample_bucket(
            pools[name], quota=quotas[name], cap=per_game_caps[name],
            min_gap=min_ply_gap, seed=base_seed + offset,
            state_key_fn=key_by_bucket.get(name))
        for game_idx, ply in sel:
            rec = recs[game_idx]
            rows.append({
                "case_id": f"neutral_game_{game_idx:06d}_ply_{ply:03d}",
                "game_idx": game_idx, "replay_path": rec["replay_path"],
                "position_ply": ply, "side_to_move": side_to_move_for_ply(ply),
                "ply_bucket": name, "game_result": rec["winner"],
                "total_game_plies": rec["n_moves"], "source_replay": source_replay,
                "sample_seed": base_seed,
            })
        stats[name] = {"requested": quotas[name], "achieved": len(sel),
                       "games_used": len({g for g, _ in sel}),
                       "eligible_games": len(pools[name]), "side_balance": side_count}
    return rows, stats


DEFAULT_SOURCE_JSONL = ("logs/eval/"
                        "calib020_0001_vs_0379_800g_w4_seed20115_replay_games.jsonl")
DEFAULT_OUT = "logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv"
DEFAULT_SEED = 20260710


def load_game_index(jsonl_path, *, require_winner=False):
    """Read the replay-eval JSONL INDEX (per line: game_idx, n_moves, winner,
    replay_path -- not the moves). Winner-null games are KEPT with
    winner='unknown' (require_winner=False); they reconstruct like any other and
    are the most search-stressed samples. Returns (records sorted, dropped)."""
    recs, dropped = [], 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            w = g.get("winner")
            if w not in ("red", "black"):
                if require_winner:
                    dropped += 1
                    continue
                w = "unknown"
            recs.append({"game_idx": int(g["game_idx"]), "n_moves": int(g["n_moves"]),
                         "winner": w, "replay_path": g["replay_path"]})
    recs.sort(key=lambda r: r["game_idx"])
    return recs, dropped


def load_excluded_game_ids(paths):
    out = set()
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out.add(int(r["game_idx"]))
    return out


def opening_prefix_key(moves, ply):
    """Ordered move-prefix key. TwixT has no captures, so identical prefixes reach
    identical states; transpositions (same pegs, different order) are NOT merged
    -- this is opening-PREFIX dedup, not full board-state dedup."""
    return tuple((m["row"], m["col"]) for m in moves[:ply])


def make_opening_key_fn(records_by_idx):
    cache = {}

    def keyfn(game_idx, ply):
        moves = cache.get(game_idx)
        if moves is None:
            moves = json.loads(Path(records_by_idx[game_idx]["replay_path"]).read_text())["moves"]
            cache[game_idx] = moves
        return opening_prefix_key(moves, ply)

    return keyfn


def validate_row(row):
    replay = json.loads(Path(row["replay_path"]).read_text())
    position_state(replay, int(row["position_ply"]), row["side_to_move"])


def write_manifest(rows, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NEUTRAL_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def write_meta(out_csv, meta):
    Path(str(out_csv) + ".meta.json").write_text(json.dumps(meta, indent=2))


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Build a deterministic, game-held-out, non-A-selected neutral "
                    "position manifest for the v16a FPU collateral screen. "
                    "READ-ONLY; writes one CSV + meta JSON. Does NOT run the sweep.")
    ap.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--exclude-manifest", action="append", default=None,
                    help="CSV(s) whose game_idx values are held out (repeatable). "
                         "The A probe manifest is added unless --no-default-exclude.")
    ap.add_argument("--no-default-exclude", action="store_true")
    ap.add_argument("--exclude-winnerless", action="store_true",
                    help="drop winner-null games (default: keep as game_result=unknown).")
    ap.add_argument("--min-ply-gap", type=int, default=DEFAULT_MIN_PLY_GAP)
    ap.add_argument("--quota-opening", type=int, default=DEFAULT_QUOTAS["opening"])
    ap.add_argument("--quota-early-mid", type=int, default=DEFAULT_QUOTAS["early_mid"])
    ap.add_argument("--quota-midgame", type=int, default=DEFAULT_QUOTAS["midgame"])
    ap.add_argument("--quota-late", type=int, default=DEFAULT_QUOTAS["late"])
    ap.add_argument("--no-validate", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    # DEFAULT_A_MANIFEST is the SINGLE source of truth for the discovery set;
    # import (deferred: diagnose_fpu_sweep pulls mcts) so the holdout cannot drift.
    from .diagnose_fpu_sweep import DEFAULT_A_MANIFEST
    args = _parse_args(argv)
    quotas = {"opening": args.quota_opening, "early_mid": args.quota_early_mid,
              "midgame": args.quota_midgame, "late": args.quota_late}

    records, dropped = load_game_index(args.source_jsonl,
                                       require_winner=args.exclude_winnerless)
    excludes = list(args.exclude_manifest or [])
    if not args.no_default_exclude:
        excludes.append(DEFAULT_A_MANIFEST)
    requested_ids = load_excluded_game_ids(excludes) if excludes else set()
    corpus_ids = {r["game_idx"] for r in records}
    matched_ids = requested_ids & corpus_ids
    held = [r for r in records if r["game_idx"] not in matched_ids]
    print(f"[v16a] {len(records)} games (dropped winnerless {dropped}); excluded "
          f"{len(matched_ids)}/{len(requested_ids)} requested A-games -> {len(held)} held-out")

    records_by_idx = {r["game_idx"]: r for r in held}
    rows, stats = sample_neutral_rows(
        held, base_seed=args.seed, source_replay=args.source_jsonl,
        quotas=quotas, min_ply_gap=args.min_ply_gap,
        state_key_fn_by_bucket={"opening": make_opening_key_fn(records_by_idx)})

    for name in BUCKET_ORDER:
        st = stats[name]
        flag = ("  <-- SHORTFALL (data-limited)" if st["achieved"] < st["requested"] else "")
        print(f"[v16a] {name:10s} {st['achieved']:3d}/{st['requested']:<3d} across "
              f"{st['games_used']} games  sides={st['side_balance']}{flag}")

    validate_dropped = 0
    if not args.no_validate:
        kept = []
        for r in rows:
            try:
                validate_row(r)
                kept.append(r)
            except Exception:
                if r["game_result"] == "unknown":     # tolerate odd winner-null games
                    validate_dropped += 1
                    continue
                raise                                  # winner-having failure is a real bug
        rows = kept
        print(f"[v16a] validated {len(rows)} rows (dropped {validate_dropped} "
              f"unreconstructable winner-null rows)")

    write_manifest(rows, args.out)
    write_meta(args.out, {
        "source_jsonl": args.source_jsonl, "base_seed": args.seed,
        "buckets": {n: [lo, hi] for n, lo, hi in BUCKETS}, "quotas": quotas,
        "per_game_caps": DEFAULT_PER_GAME_CAPS, "min_ply_gap": args.min_ply_gap,
        "excluded_manifests": excludes,
        "requested_excluded_game_count": len(requested_ids),
        "matched_excluded_game_count": len(matched_ids),
        "matched_excluded_game_ids": sorted(matched_ids),
        "winnerless_dropped": dropped, "validate_dropped": validate_dropped,
        "num_rows": len(rows), "per_bucket_stats": stats,
        "fieldnames": NEUTRAL_FIELDNAMES,
        "sample_kind": "stratified_game_held_out_non_selected",
    })
    print(f"[v16a] wrote {len(rows)} rows -> {args.out}  (+ .meta.json)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
