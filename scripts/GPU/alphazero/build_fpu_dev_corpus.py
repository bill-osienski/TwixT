"""FPU (policy-mass) development-corpus builder.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Tasks 5-6.

=============================================================================
PURE SECTION (Task 5) -- constants + pure functions ONLY.
=============================================================================
Everything in this file is pure: plain-stdlib classification and sampling over
plain-dict "rows". NO MCTS / evaluator / GPU / MLX / heavy-numpy imports, no
I/O, no argument parsing. The operator shell (two-stage scan, per-ply n_legal,
raw-policy forward pass, 400-sim anchor confirm, canonical hashing/disjointness,
manifest writing, `main()`) is added by Task 6 BELOW this section and imports
these pure functions -- so keep this section cleanly separated and importable.

What this section does
----------------------
Sample a 240-row development corpus (180 target + 60 matched controls) from
anchor-CONFIRMED candidate positions and split it into 160 tuning / 80
frozen_check rows BY WHOLE GAME (a game's positions never straddle splits).
Membership/role was decided UPSTREAM by the fpu-off anchor + raw policy (never
by any candidate-FPU result); this section only classifies, allocates, and
samples.

A "row" is a dict carrying at least:
    game_idx, role ("target"|"control"), band, side ("red"|"black"),
    ply, ply_bucket, canonical_sha1
`sample_dev_rows` returns rows each additionally stamped with `split`.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Tuple

# ---------------------------------------------------------------------------
# Frozen constants (verbatim from the design + Task-5 brief)
# ---------------------------------------------------------------------------

BANDS: Tuple[str, str, str] = ("b200_299", "b300_399", "b400_plus")

# Ply buckets (reused from v16a): opening 1-15, early_mid 16-40,
# midgame 41-90, late 91+.
PLY_BUCKETS: Tuple[str, str, str, str] = ("opening", "early_mid", "midgame", "late")

TARGET_PER_BAND = 60
CONTROL_PER_BAND = 20
MIN_PLY_GAP = 12
MAX_PER_GAME = 2

# Per-split side balance tolerance: |red - black| <= SIDE_TOL within each split.
# Every game supplies one red + one black position, so whole-game (2-per-game)
# picks are side-neutral; only the odd-quota control cells (13 / 7) leave a
# single leftover, and the sampler steers each leftover toward the deficit side.
# With two odd cells per split that can cancel, observed imbalance is 0; 2 is a
# safe cap that also tolerates any ordering/reserve variation.
SIDE_TOL = 2

SPLITS: Tuple[str, str] = ("tuning", "frozen_check")

# Frozen whole-game split allocation (Task-5 brief fix 7).
#   target cells: 40 tuning / 20 frozen each band  -> 120 / 60
#   control cells: {b200: 13/7, b300: 13/7, b400: 14/6} -> 40 / 20
#   totals: tuning 160, frozen_check 80; target 180, control 60; grand 240.
SPLIT_ALLOC: Dict[Tuple[str, str], Dict[str, int]] = {
    ("target", "b200_299"): {"tuning": 40, "frozen_check": 20},
    ("target", "b300_399"): {"tuning": 40, "frozen_check": 20},
    ("target", "b400_plus"): {"tuning": 40, "frozen_check": 20},
    ("control", "b200_299"): {"tuning": 13, "frozen_check": 7},
    ("control", "b300_399"): {"tuning": 13, "frozen_check": 7},
    ("control", "b400_plus"): {"tuning": 14, "frozen_check": 6},
}

# Deterministic (role, band) cell order = SPLIT_ALLOC insertion order.
CELL_ORDER: List[Tuple[str, str]] = list(SPLIT_ALLOC.keys())

# Total manifest size implied by the frozen allocation (240).
CORPUS_SIZE = sum(a["tuning"] + a["frozen_check"] for a in SPLIT_ALLOC.values())


# ---------------------------------------------------------------------------
# Two-stage classifiers (pure)
# ---------------------------------------------------------------------------

def band_of(n_legal: int):
    """Branching band for a legal-move count, or None below the target floor.

    b200_299: 200-299, b300_399: 300-399, b400_plus: 400+. Below 200 is not an
    eligible dev-corpus position (returns None).
    """
    if n_legal >= 400:
        return "b400_plus"
    if n_legal >= 300:
        return "b300_399"
    if n_legal >= 200:
        return "b200_299"
    return None


def ply_bucket_of(ply: int) -> str:
    """Coarse game-phase bucket for the <=50% cap. opening 1-15, early_mid
    16-40, midgame 41-90, late 91+."""
    if ply <= 15:
        return "opening"
    if ply <= 40:
        return "early_mid"
    if ply <= 90:
        return "midgame"
    return "late"


def raw_policy_role(normalized_entropy: float, top1_prior: float):
    """Stage-2 raw-policy role from geometry, or None for the grey zone.

    target  iff normalized_entropy >= 0.90 AND top1_prior <= 0.025 (flat, diffuse)
    control iff normalized_entropy <  0.85 OR  top1_prior >= 0.05  (concentrated)
    otherwise None (the grey band between, e.g. 0.88 / 0.03).

    The two positive conditions are mutually exclusive (a target requires
    entropy >= 0.90 and top1 <= 0.025, both of which fail the control test), so
    evaluation order is immaterial.
    """
    if normalized_entropy >= 0.90 and top1_prior <= 0.025:
        return "target"
    if normalized_entropy < 0.85 or top1_prior >= 0.05:
        return "control"
    return None


def anchor_eligible(root_value_stm: float) -> bool:
    """Near-even fpu-off anchor gate: |root_value_stm| <= 0.25 (inclusive)."""
    return abs(root_value_stm) <= 0.25


# ---------------------------------------------------------------------------
# Whole-game split assignment (contribution-aware, deterministic)
# ---------------------------------------------------------------------------

def _greedy_assign(games_profile, seed, attempt):
    """One deterministic greedy pass. Returns {game_idx: split} if it satisfies
    every per-(role, band, split) quota (capacity), else None.

    Each WHOLE game is placed in the split whose still-unmet quotas it fills
    most; ties break toward the split with the larger total remaining need, then
    toward tuning. Games are visited in a seed-shuffled order (attempt 0) or its
    deterministic reverse (attempt 1, the secondary-ordering retry).
    """
    rng = random.Random(seed * 1_000_003 + attempt)
    order = sorted(games_profile)
    rng.shuffle(order)
    if attempt == 1:
        order = order[::-1]

    need = {cell: dict(alloc) for cell, alloc in SPLIT_ALLOC.items()}
    assign: Dict[Any, str] = {}
    for gi in order:
        prof = games_profile[gi]
        cells = [c for c in prof if c in need]

        def useful(split, _cells=cells, _prof=prof):
            return sum(min(_prof[c], need[c][split]) for c in _cells)

        u_t, u_f = useful("tuning"), useful("frozen_check")
        if u_t > u_f:
            split = "tuning"
        elif u_f > u_t:
            split = "frozen_check"
        else:
            tot_t = sum(need[c]["tuning"] for c in cells)
            tot_f = sum(need[c]["frozen_check"] for c in cells)
            split = "tuning" if tot_t >= tot_f else "frozen_check"

        assign[gi] = split
        for c in cells:
            need[c][split] = max(0, need[c][split] - prof[c])

    if all(v == 0 for cell in need for v in need[cell].values()):
        return assign
    return None


def assign_split(games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
                 seed: int) -> Dict[Any, str]:
    """Assign each WHOLE game to "tuning" or "frozen_check" so every
    per-(role, band, split) SPLIT_ALLOC quota is satisfiable (enough capacity).

    `games_profile`: {game_idx: {(role, band): n_available_positions}}.

    Raises ValueError if any cell's total capacity is below its combined demand,
    or if neither the primary nor the deterministic secondary ordering yields a
    quota-satisfying assignment (a shortfall surfaces here).
    """
    capacity: Counter = Counter()
    for prof in games_profile.values():
        for cell, n in prof.items():
            capacity[cell] += n
    for cell, alloc in SPLIT_ALLOC.items():
        demand = alloc["tuning"] + alloc["frozen_check"]
        have = capacity.get(cell, 0)
        if have < demand:
            raise ValueError(
                f"assign_split: cell {cell} capacity {have} < demand {demand}")

    for attempt in range(2):
        result = _greedy_assign(games_profile, seed, attempt)
        if result is not None:
            return result
    raise ValueError(
        "assign_split: no deterministic ordering satisfied the split quotas")


# ---------------------------------------------------------------------------
# 240-row sampler (round-robin, exact composition or raise)
# ---------------------------------------------------------------------------

def _choose_positions(positions, take_n, side_count, gap):
    """Pick up to `take_n` positions from one game's `positions` (sorted by ply).

    take_n == 1 -> steer toward the side that reduces the split's current
    |red - black| (ties break by lower ply then side, for determinism).
    take_n >= 2 -> greedily take the earliest positions that stay >= `gap` plies
    apart (a game's red+black pair is designed to clear the gap, keeping the pair
    side-neutral).
    """
    if take_n <= 0 or not positions:
        return []
    if take_n == 1:
        red, black = side_count["red"], side_count["black"]

        def imbalance_if(r):
            if r["side"] == "red":
                return abs((red + 1) - black)
            return abs(red - (black + 1))

        return [min(positions, key=lambda r: (imbalance_if(r), r["ply"], r["side"]))]

    chosen: List[dict] = []
    last_ply = None
    for r in positions:
        if last_ply is None or (r["ply"] - last_ply) >= gap:
            chosen.append(r)
            last_ply = r["ply"]
            if len(chosen) == take_n:
                break
    return chosen


def sample_dev_rows(confirmed: List[dict], *, seed: int) -> Tuple[List[dict], dict]:
    """Sample the 240-row dev corpus from anchor-CONFIRMED candidate rows.

    Steps: (1) build each game's (role, band) contribution profile; (2)
    `assign_split` places whole games into tuning / frozen_check; (3) round-robin
    within each assigned split fills every SPLIT_ALLOC cell EXACTLY, subject to
    MAX_PER_GAME (<=2/game), MIN_PLY_GAP (>=12-ply separation within a game), a
    per-split side balance |red-black| <= SIDE_TOL, a global ply-bucket <=50%
    cap, and no duplicate canonical_sha1.

    Every cell must reach its quota exactly; a final-manifest shortfall is an
    ERROR (raises ValueError), never a silent truncation. Deterministic under
    `seed`. Returns (rows, stats); each row is stamped with `split`.
    """
    games: Dict[Any, List[dict]] = defaultdict(list)
    for r in confirmed:
        games[r["game_idx"]].append(r)

    profile = {gi: Counter((r["role"], r["band"]) for r in rows_)
               for gi, rows_ in games.items()}

    split_of = assign_split(profile, seed)   # may raise ValueError (infeasible)

    bucket_cap = 0.5 * CORPUS_SIZE
    used_sha1: set = set()
    bucket_count: Counter = Counter()
    side_count = {s: {"red": 0, "black": 0} for s in SPLITS}
    selected: List[dict] = []

    for split in SPLITS:
        for cell in CELL_ORDER:
            quota = SPLIT_ALLOC[cell][split]
            picked = 0
            cand_games = sorted(
                gi for gi in games
                if split_of.get(gi) == split and cell in profile[gi])
            for gi in cand_games:
                if picked >= quota:
                    break
                positions = sorted(
                    (r for r in games[gi] if (r["role"], r["band"]) == cell),
                    key=lambda r: r["ply"])
                positions = [
                    r for r in positions
                    if r["canonical_sha1"] not in used_sha1
                    and bucket_count[r["ply_bucket"]] < bucket_cap]
                take_n = min(MAX_PER_GAME, quota - picked, len(positions))
                for r in _choose_positions(positions, take_n,
                                           side_count[split], MIN_PLY_GAP):
                    out = dict(r)
                    out["split"] = split
                    selected.append(out)
                    used_sha1.add(r["canonical_sha1"])
                    bucket_count[r["ply_bucket"]] += 1
                    side_count[split][r["side"]] += 1
                    picked += 1
            if picked != quota:
                raise ValueError(
                    f"final-manifest shortfall: cell {(cell[0], cell[1], split)} "
                    f"filled {picked} of required {quota}")

    stats = {
        "n_rows": len(selected),
        "seed": seed,
        "cell_counts": {
            f"{role}|{band}|{split}": SPLIT_ALLOC[(role, band)][split]
            for (role, band) in SPLIT_ALLOC for split in SPLITS},
        "side_count": {s: dict(side_count[s]) for s in SPLITS},
        "bucket_count": dict(bucket_count),
        "n_games_per_split": {
            s: sum(1 for gi in split_of if split_of[gi] == s) for s in SPLITS},
        "n_games_total": len(split_of),
    }
    return selected, stats


# =============================================================================
# OPERATOR SHELL (Task 6) -- appended below by a later task.
# per-ply n_legal, candidate enumeration, raw-policy forward pass, 400-sim
# anchor confirm, canonical hashing/disjointness, manifest writing, main().
# Nothing above this line imports MCTS/GPU/MLX or performs I/O.
# =============================================================================
