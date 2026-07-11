"""Tests for the FPU dev-corpus PURE sampler (Task 5).

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Task 5: two-stage classify + contribution-aware whole-game split +
EXACT frozen composition.

This exercises only the PURE section of build_fpu_dev_corpus.py -- classifiers,
the whole-game split assigner, and the 240-row sampler operating on plain dict
"rows" (no MCTS/GPU/MLX; canonical_sha1 values are synthetic strings, NOT real
hashes -- the real hashing/scan is Task 6).

The crux is test_exact_split_composition_and_totals: every (role, band, split)
cell must equal its SPLIT_ALLOC quota EXACTLY and simultaneously satisfy no-dup
hash, ply-bucket <=50% cap, and per-split side balance. `_abundant_pool()` is
co-designed with the sampler to make all of those hold at once.
"""
from collections import Counter, defaultdict

import pytest

from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    SPLIT_ALLOC,
    SIDE_TOL,
    MIN_PLY_GAP,
    MAX_PER_GAME,
    TARGET_PER_BAND,
    CONTROL_PER_BAND,
    BANDS,
    band_of,
    ply_bucket_of,
    raw_policy_role,
    anchor_eligible,
    assign_split,
    sample_dev_rows,
)

# ---------------------------------------------------------------------------
# Synthetic pool construction (co-designed with the sampler).
#
# Each game contributes exactly two eligible positions to a SINGLE (role, band)
# cell: one red + one black, >= MIN_PLY_GAP plies apart, both in one ply bucket.
# Buckets cycle over all four buckets by global game index, so no bucket can
# approach the 50% cap and every cell's rows span buckets evenly. Every position
# gets a globally-unique synthetic canonical_sha1. Providing many games per cell
# (well above the whole-game quota minimum) leaves assign_split slack to place
# whole games into tuning vs frozen_check and hit each quota, plus reserve games
# for round-robin/side-balance headroom.
# ---------------------------------------------------------------------------

_BUCKET_CYCLE = ("opening", "early_mid", "midgame", "late")
# (low_ply, high_ply): high - low >= 12 and both plies land in the named bucket.
_BUCKET_PLIES = {
    "opening": (1, 13),      # 1..15
    "early_mid": (16, 28),   # 16..40
    "midgame": (41, 60),     # 41..90
    "late": (91, 110),       # 91+
}


def _game_rows(game_idx, role, band):
    """Two rows (red low-ply, black high-ply) for one game in one (role, band)."""
    bucket = _BUCKET_CYCLE[game_idx % len(_BUCKET_CYCLE)]
    p_red, p_black = _BUCKET_PLIES[bucket]
    return [
        {
            "game_idx": game_idx, "role": role, "band": band, "side": "red",
            "ply": p_red, "ply_bucket": bucket,
            "canonical_sha1": f"sha-{game_idx:04d}-{p_red}-red",
        },
        {
            "game_idx": game_idx, "role": role, "band": band, "side": "black",
            "ply": p_black, "ply_bucket": bucket,
            "canonical_sha1": f"sha-{game_idx:04d}-{p_black}-black",
        },
    ]


# games per cell -- comfortably above the whole-game minimum for every cell
# (target needs 30 games = ceil(40/2)+ceil(20/2); control needs <=11).
_GAMES_PER_CELL = {"target": 50, "control": 30}
_CELLS = list(SPLIT_ALLOC.keys())  # frozen (role, band) order


def _abundant_pool():
    rows = []
    gi = 0
    for role, band in _CELLS:
        for _ in range(_GAMES_PER_CELL[role]):
            rows.extend(_game_rows(gi, role, band))
            gi += 1
    return rows


def _insufficient_pool():
    """Abundant everywhere except one target cell starved below its quota, so
    that cell can never be filled -> sample_dev_rows must raise ValueError."""
    rows = []
    gi = 0
    for role, band in _CELLS:
        n = 3 if (role, band) == ("target", "b200_299") else _GAMES_PER_CELL[role]
        for _ in range(n):
            rows.extend(_game_rows(gi, role, band))
            gi += 1
    return rows


# ---------------------------------------------------------------------------
# Two-stage classifiers
# ---------------------------------------------------------------------------

def test_raw_policy_role_required_points():
    assert raw_policy_role(0.95, 0.01) == "target"
    assert raw_policy_role(0.80, 0.20) == "control"
    assert raw_policy_role(0.88, 0.03) is None


def test_raw_policy_role_boundaries():
    assert raw_policy_role(0.90, 0.025) == "target"    # both thresholds inclusive
    assert raw_policy_role(0.899, 0.025) is None        # entropy just under target
    assert raw_policy_role(0.90, 0.0251) is None        # top1 just over target
    assert raw_policy_role(0.84, 0.01) == "control"     # entropy < 0.85
    assert raw_policy_role(0.99, 0.05) == "control"     # top1 >= 0.05
    assert raw_policy_role(0.86, 0.04) is None          # grey zone


def test_anchor_eligible_boundaries():
    assert anchor_eligible(0.20) is True
    assert anchor_eligible(0.30) is False
    assert anchor_eligible(0.25) is True                # inclusive
    assert anchor_eligible(-0.25) is True
    assert anchor_eligible(-0.2501) is False


def test_band_of_boundaries():
    assert band_of(200) == "b200_299"
    assert band_of(299) == "b200_299"
    assert band_of(300) == "b300_399"
    assert band_of(399) == "b300_399"
    assert band_of(400) == "b400_plus"
    assert band_of(1000) == "b400_plus"
    assert band_of(199) is None                          # below the target floor
    assert BANDS == ("b200_299", "b300_399", "b400_plus")


def test_ply_bucket_boundaries():
    assert ply_bucket_of(1) == "opening"
    assert ply_bucket_of(15) == "opening"
    assert ply_bucket_of(16) == "early_mid"
    assert ply_bucket_of(40) == "early_mid"
    assert ply_bucket_of(41) == "midgame"
    assert ply_bucket_of(90) == "midgame"
    assert ply_bucket_of(91) == "late"
    assert ply_bucket_of(300) == "late"


# ---------------------------------------------------------------------------
# EXACT composition (fix 7) -- the crux -- copied verbatim from the brief
# ---------------------------------------------------------------------------

def test_exact_split_composition_and_totals():
    rows, stats = sample_dev_rows(_abundant_pool(), seed=1)
    cell = Counter((r["role"], r["band"], r["split"]) for r in rows)
    for (role, band), alloc in SPLIT_ALLOC.items():
        for split, n in alloc.items():
            assert cell[(role, band, split)] == n            # every cell EXACTLY full
    assert sum(1 for r in rows if r["split"] == "tuning") == 160
    assert sum(1 for r in rows if r["split"] == "frozen_check") == 80
    assert sum(1 for r in rows if r["role"] == "target") == 180
    assert sum(1 for r in rows if r["role"] == "control") == 60
    assert len({r["canonical_sha1"] for r in rows}) == len(rows)   # no dup hash
    assert max(Counter(r["ply_bucket"] for r in rows).values()) <= 0.5 * len(rows)
    for split in ("tuning", "frozen_check"):
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL


def test_shortfall_on_final_manifest_is_an_error():
    with pytest.raises(ValueError):
        sample_dev_rows(_insufficient_pool(), seed=1)          # cannot fill a cell -> raise


# ---------------------------------------------------------------------------
# Sampling invariants
# ---------------------------------------------------------------------------

def test_total_rows_is_240():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    assert len(rows) == 240


def test_determinism_same_seed():
    a, sa = sample_dev_rows(_abundant_pool(), seed=1)
    b, sb = sample_dev_rows(_abundant_pool(), seed=1)
    assert a == b
    assert sa == sb


def test_at_most_two_per_game():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    counts = Counter(r["game_idx"] for r in rows)
    assert counts and max(counts.values()) <= MAX_PER_GAME


def test_min_ply_gap_within_game():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game_idx"]].append(r["ply"])
    for plies in by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP


def test_whole_game_split_isolation():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    game_splits = defaultdict(set)
    for r in rows:
        game_splits[r["game_idx"]].add(r["split"])
    assert game_splits
    assert all(len(splits) == 1 for splits in game_splits.values())


def test_no_duplicate_hash():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))


def test_every_row_carries_a_valid_split():
    rows, _ = sample_dev_rows(_abundant_pool(), seed=1)
    assert all(r["split"] in ("tuning", "frozen_check") for r in rows)


def test_assign_split_is_whole_game_and_deterministic():
    pool = _abundant_pool()
    profile = {}
    for r in pool:
        profile.setdefault(r["game_idx"], Counter())[(r["role"], r["band"])] += 1
    a = assign_split(profile, seed=1)
    b = assign_split(profile, seed=1)
    assert a == b                                       # deterministic under seed
    assert set(a) == set(profile)                       # every game assigned
    assert all(v in ("tuning", "frozen_check") for v in a.values())


def test_constants_match_frozen_spec():
    assert TARGET_PER_BAND == 60
    assert CONTROL_PER_BAND == 20
    assert MIN_PLY_GAP == 12
    assert MAX_PER_GAME == 2
    # SPLIT_ALLOC frozen totals
    tuning = sum(a["tuning"] for a in SPLIT_ALLOC.values())
    frozen = sum(a["frozen_check"] for a in SPLIT_ALLOC.values())
    assert (tuning, frozen) == (160, 80)
    target = sum(sum(a.values()) for c, a in SPLIT_ALLOC.items() if c[0] == "target")
    control = sum(sum(a.values()) for c, a in SPLIT_ALLOC.items() if c[0] == "control")
    assert (target, control) == (180, 60)
