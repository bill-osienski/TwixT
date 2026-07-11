"""Tests for the FPU dev-corpus PURE sampler (Task 5) + PURE scan helpers
(Task 6).

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Task 5: two-stage classify + contribution-aware whole-game split +
EXACT frozen composition. Plan Task 6: two-stage scan pure helpers
(per-ply n_legal + fallback, candidate-ply enumeration, raw-policy geometry
features, forbidden-hash union + disjointness, the RESERVE constant).

This exercises only the PURE helpers of build_fpu_dev_corpus.py -- classifiers,
the whole-game split assigner, the 240-row sampler operating on plain dict
"rows", and (Task 6) the per-ply/candidate/policy-feature/hash-union/
disjointness helpers. Still NO MCTS/GPU/MLX/checkpoint/evaluator anywhere, and
NO reads of the real seed20116 replay corpus: in the Task-5 tests
canonical_sha1 values are synthetic strings, NOT real hashes; the Task-6
fallback/reconstruction tests below build small SYNTHETIC real replays/CSVs on
disk (via plain TwixtState.apply_move / tmp_path), never the real corpus. The
real hashing/scan is `main()` (operator-run, never invoked here).

The crux is test_exact_split_composition_and_totals: every (role, band, split)
cell must equal its SPLIT_ALLOC quota EXACTLY and simultaneously satisfy no-dup
hash, ply-bucket <=50% cap, and per-split side balance. `_abundant_pool()` is
co-designed with the sampler to make all of those hold at once.
"""
import json
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
    RESERVE,
    band_of,
    ply_bucket_of,
    raw_policy_role,
    anchor_eligible,
    assign_split,
    sample_dev_rows,
    per_ply_n_legal,
    enumerate_candidate_plies,
    _policy_features_from_priors,
    load_forbidden_hashes,
    assert_disjoint,
)

# Task 6 fallback/reconstruction tests build small REAL replays/positions via
# TwixtState directly (still no MCTS/GPU/MLX/checkpoint -- TwixtState is pure
# numpy) rather than reading the real seed20116 corpus.
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1

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


def _pos(game_idx, role, band, side, ply, bucket):
    """One synthetic row (distinct 'pos-' hash) for a hand-built game -- used by
    the multi-position / multi-cell / gap-starved fixtures below."""
    return {
        "game_idx": game_idx, "role": role, "band": band, "side": side,
        "ply": ply, "ply_bucket": bucket,
        "canonical_sha1": f"pos-{game_idx:05d}-{ply}-{side}",
    }


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


# --- discriminating fixtures (review-fix pass) -----------------------------

def _pool_with_multiposition_game():
    """Abundant pool, but game 0 holds FOUR positions in ONE (role, band) cell
    with two of them <MIN_PLY_GAP apart. A correct sampler must BOTH cap the
    game at MAX_PER_GAME and skip the sub-gap position; the single-cell
    _abundant_pool never exercises either (its games have exactly two positions,
    already >=MIN_PLY_GAP apart). Plies 41,45,57,70 in one bucket: cap+gap pick
    {41,57}; dropping the cap would add 70, dropping the gap-skip would take 45.
    """
    rows = [r for r in _abundant_pool() if r["game_idx"] != 0]
    role, band = _CELLS[0]
    for side, ply in (("red", 41), ("red", 45), ("black", 57), ("black", 70)):
        rows.append(_pos(0, role, band, side, ply, "midgame"))
    return rows


def _pool_with_multicell_games():
    """Abundant single-cell pool PLUS two genuine MULTI-CELL games: games 0 and 1
    each contribute one position to (target,b200_299) AND one to
    (target,b300_399) -- a game whose positions straddle two bands as n_legal
    fell. This drives _greedy_assign's multi-cell branch, which every single-cell
    fixture leaves dead. The pair is side-mirrored so EXACT composition (side
    balance included) still holds."""
    rows = [
        _pos(0, "target", "b200_299", "red",   41, "midgame"),
        _pos(0, "target", "b300_399", "black", 57, "midgame"),
        _pos(1, "target", "b200_299", "black", 41, "midgame"),
        _pos(1, "target", "b300_399", "red",   57, "midgame"),
    ]
    gi = 2
    for role, band in _CELLS:
        for _ in range(_GAMES_PER_CELL[role]):
            rows.extend(_game_rows(gi, role, band))
            gi += 1
    return rows


_DUP_SHA1 = "dup-shared-red-41"


def _pool_with_duplicate_hash():
    """Games 0 and 1 (both (target,b200_299)) each carry THREE positions sharing
    ONE canonical_sha1 on a gap-valid red@41. Whichever game the round-robin
    draws first claims the shared hash; the other's red@41 is otherwise gap-valid
    and would be picked, so it MUST be dropped by the used_sha1 filter. Each game
    still yields MAX_PER_GAME from its two unique positions, so the dedup fires
    without starving the cell (both games stay live in the output)."""
    rows = [r for r in _abundant_pool() if r["game_idx"] not in (0, 1)]
    role, band = _CELLS[0]
    for gi in (0, 1):
        rows.append({"game_idx": gi, "role": role, "band": band, "side": "red",
                     "ply": 41, "ply_bucket": "midgame",
                     "canonical_sha1": _DUP_SHA1})
        rows.append(_pos(gi, role, band, "black", 57, "midgame"))
        rows.append(_pos(gi, role, band, "red", 73, "midgame"))
    return rows


def _pool_gap_starved_cell():
    """Abundant for five cells, but (control,b400_plus) is supplied only by games
    whose two positions sit <MIN_PLY_GAP apart, so each yields just ONE pickable
    row via the gap-skip. assign_split's capacity precheck PASSES (12 games * 2
    positions exceed the demand of 20) yet the round-robin cannot reach the
    cell's quota -> the final-manifest `picked != quota` raise, DISTINCT from the
    capacity precheck that _insufficient_pool trips."""
    rows = []
    gi = 0
    for role, band in _CELLS:
        if (role, band) == ("control", "b400_plus"):
            continue
        for _ in range(_GAMES_PER_CELL[role]):
            rows.extend(_game_rows(gi, role, band))
            gi += 1
    for _ in range(12):
        rows.append(_pos(gi, "control", "b400_plus", "red",   91, "late"))
        rows.append(_pos(gi, "control", "b400_plus", "black", 96, "late"))
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
    # Discriminating: game 0 has FOUR positions in one cell, so the MAX_PER_GAME
    # cap actually binds (a single-cell 2-position fixture would pass even with
    # the cap deleted). Mutation-verified in the review-fix report.
    rows, _ = sample_dev_rows(_pool_with_multiposition_game(), seed=1)
    counts = Counter(r["game_idx"] for r in rows)
    assert counts and max(counts.values()) <= MAX_PER_GAME
    assert counts[0] == MAX_PER_GAME          # the 4-position game 0, capped at 2


def test_min_ply_gap_within_game():
    # Discriminating: game 0's four positions include a pair 4 plies apart
    # (41 & 45), so the gap-skip actually binds. Mutation-verified in the report.
    rows, _ = sample_dev_rows(_pool_with_multiposition_game(), seed=1)
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game_idx"]].append(r["ply"])
    for plies in by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP
    # game 0 skips the sub-gap 45 and stops before 70: exactly {41, 57}.
    assert sorted(by_game[0]) == [41, 57]


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


# ---------------------------------------------------------------------------
# Multi-cell split assignment (review-fix pass) -- exercises _greedy_assign's
# multi-cell branch, its reversed-order retry, and its "no ordering" raise, none
# of which any single-cell fixture reaches.
# ---------------------------------------------------------------------------

def test_multicell_game_split_exact_composition():
    pool = _pool_with_multicell_games()
    # a genuinely multi-cell game must be present, else the branch stays dead
    prof = defaultdict(Counter)
    for r in pool:
        prof[r["game_idx"]][(r["role"], r["band"])] += 1
    assert any(len(cells) > 1 for cells in prof.values())

    rows, stats = sample_dev_rows(pool, seed=1)
    cell = Counter((r["role"], r["band"], r["split"]) for r in rows)
    for (role, band), alloc in SPLIT_ALLOC.items():
        for split, n in alloc.items():
            assert cell[(role, band, split)] == n        # exact composition holds
    assert len(rows) == 240
    for split in ("tuning", "frozen_check"):
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
    # the multi-cell game is placed WHOLE -- its two cells share one split
    game_splits = defaultdict(set)
    for r in rows:
        game_splits[r["game_idx"]].add(r["split"])
    assert game_splits[0] and len(game_splits[0]) == 1


def test_assign_split_raises_on_infeasible_multicell_partition():
    # Every cell clears the capacity precheck, but (control,b200_299)'s realizable
    # capacity is exactly its demand (20) supplied as ten MULTI-CELL games each
    # realizing <= MAX_PER_GAME. No whole-game tuning/frozen partition can give
    # tuning >= 13 AND frozen >= 7 for it (that needs >= 11 games), so BOTH greedy
    # orderings fail -- the reversed-order retry runs -- and assign_split raises
    # the "no ordering" error, distinct from the capacity precheck error.
    profile = {}
    for gi in range(10):
        profile[gi] = Counter({("control", "b200_299"): 2,
                               ("control", "b300_399"): 1})
    gi = 10
    for (role, band), alloc in SPLIT_ALLOC.items():
        if (role, band) == ("control", "b200_299"):
            continue
        for _ in range(alloc["tuning"] + alloc["frozen_check"]):
            profile[gi] = Counter({(role, band): 2})
            gi += 1
    with pytest.raises(ValueError, match="no deterministic ordering"):
        assign_split(profile, seed=1)


# ---------------------------------------------------------------------------
# Per-pick filters: dedup exclusion + a shortfall distinct from the precheck.
# ---------------------------------------------------------------------------

def test_duplicate_hash_is_excluded():
    rows, _ = sample_dev_rows(_pool_with_duplicate_hash(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))            # used_sha1 filter left no dup
    assert shas.count(_DUP_SHA1) == 1             # the collided hash survives once
    # both colliding games stay live in the output, so the exclusion was FORCED
    # (the loser lost its red@41 to the filter, not by going unselected).
    assert {0, 1} <= {r["game_idx"] for r in rows}


def test_final_manifest_shortfall_from_pick_filters():
    # Distinct from _insufficient_pool (which trips assign_split's capacity
    # precheck): here every cell has enough capacity, but the gap-skip per-pick
    # filter starves (control,b400_plus) so the round-robin hits its
    # picked != quota raise.
    with pytest.raises(ValueError, match="final-manifest shortfall"):
        sample_dev_rows(_pool_gap_starved_cell(), seed=1)


def test_stats_cell_counts_are_real_counts():
    # cell_counts must be an INDEPENDENT witness counted from the selected rows,
    # not a re-emission of the SPLIT_ALLOC quotas: it must equal the actual
    # per-cell tally (and, since exact-or-raise held, the frozen quotas), and sum
    # to the 240 selected rows.
    rows, stats = sample_dev_rows(_abundant_pool(), seed=1)
    actual = Counter((r["role"], r["band"], r["split"]) for r in rows)
    for (role, band), alloc in SPLIT_ALLOC.items():
        for split, quota in alloc.items():
            key = f"{role}|{band}|{split}"
            assert stats["cell_counts"][key] == actual[(role, band, split)]
            assert stats["cell_counts"][key] == quota
    assert sum(stats["cell_counts"].values()) == len(rows) == 240


# ---------------------------------------------------------------------------
# Task 6 -- per-ply n_legal (primary + fallback)
# ---------------------------------------------------------------------------

def test_per_ply_n_legal_primary_reads_stored_values():
    moves = [{"n_legal": 528}, {"n_legal": 527}, {"n_legal": 200}, {"n_legal": 199}]
    replay = {"moves": moves}
    assert per_ply_n_legal(replay) == [528, 527, 200, 199]


def test_per_ply_n_legal_fallback_reconstructs_every_fourth_ply():
    # No "n_legal" on any move -> fallback path. Build a REAL small replay via
    # repeated TwixtState.apply_move (still no GPU/MLX -- TwixtState is pure
    # numpy), independently tracking the legal count at each of the
    # reconstructed (every-4th) plies as we go, so the assertion is a genuine
    # cross-check against the function's own reconstruction, not a restatement
    # of it.
    state = TwixtState(active_size=24, to_move="red", max_plies_limit=50)
    moves = []
    expected_at_stride = {}
    for ply in range(10):
        legal = state.legal_moves()
        if ply % 4 == 0:
            expected_at_stride[ply] = len(legal)
        mv = legal[0]
        moves.append({"row": mv[0], "col": mv[1]})   # deliberately no n_legal/player
        state = state.apply_move(mv)
    replay = {"board_size": 24, "n_moves": len(moves), "moves": moves}

    result = per_ply_n_legal(replay)

    assert len(result) == 10
    for ply in range(10):
        if ply % 4 == 0:
            assert result[ply] == expected_at_stride[ply]
        else:
            assert result[ply] is None


def test_per_ply_n_legal_partial_n_legal_still_uses_fallback():
    # Only ONE move carries "n_legal" -- the primary-path guard must require
    # ALL moves to carry it (`all(...)`), not just one (`any(...)`): an
    # `any(...)` bug would wrongly take the primary list comprehension, which
    # KeyErrors on every OTHER move lacking "n_legal". Reuses the real small
    # replay construction from the fallback test above, with one deliberately
    # WRONG stored value injected to also prove it gets ignored.
    state = TwixtState(active_size=24, to_move="red", max_plies_limit=50)
    moves = []
    expected_at_stride = {}
    for ply in range(8):
        legal = state.legal_moves()
        if ply % 4 == 0:
            expected_at_stride[ply] = len(legal)
        mv = legal[0]
        move = {"row": mv[0], "col": mv[1]}
        if ply == 1:
            move["n_legal"] = 999999          # bogus -- must be ignored (fallback, not primary)
        moves.append(move)
        state = state.apply_move(mv)
    replay = {"board_size": 24, "n_moves": len(moves), "moves": moves}

    result = per_ply_n_legal(replay)

    assert len(result) == 8
    for ply in range(8):
        if ply % 4 == 0:
            assert result[ply] == expected_at_stride[ply]
        else:
            assert result[ply] is None
    assert result[1] is None                  # bogus stored value at a non-stride ply: ignored


# ---------------------------------------------------------------------------
# Task 6 -- candidate-ply enumeration (stride/cap over the qualifying subseq)
# ---------------------------------------------------------------------------

def test_enumerate_candidate_plies_takes_first_fifth_ninth_qualifying():
    # All 20 plies qualify (n_legal=250 >= 200) -> the qualifying subsequence
    # IS the ply index itself, so "1st/5th/9th... qualifying" is exactly
    # ply 0/4/8/....
    moves = [{"n_legal": 250} for _ in range(20)]
    replay = {"moves": moves}
    result = enumerate_candidate_plies(replay, stride=4, cap=6)
    assert result == [0, 4, 8, 12, 16]        # 5 qualifying picks (20/4); cap not reached


def test_enumerate_candidate_plies_irregular_qualifying_stride_and_cap():
    # 30 hand-picked, strictly ascending "qualifying" plies interleaved with
    # non-qualifying gaps of irregular width, so stride selection cannot be
    # confused with a raw-ply-index stride, and the cap genuinely binds
    # (30 qualifying -> 8 strided -> capped to 6).
    qualifying_plies = [1, 3, 4, 7, 10, 11, 15, 20, 21, 22, 30, 40, 41, 50, 60,
                        61, 70, 80, 90, 99, 101, 103, 105, 110, 120, 121, 130,
                        140, 150, 160]
    assert len(qualifying_plies) == 30
    qual_set = set(qualifying_plies)
    n_plies = qualifying_plies[-1] + 1
    moves = [{"n_legal": 250 if p in qual_set else 50} for p in range(n_plies)]
    replay = {"moves": moves}

    result = enumerate_candidate_plies(replay, stride=4, cap=6)

    expected = [qualifying_plies[i] for i in (0, 4, 8, 12, 16, 20)]
    assert result == expected == [1, 10, 21, 41, 70, 101]
    assert len(result) == 6


def test_enumerate_candidate_plies_empty_when_none_qualify():
    moves = [{"n_legal": 50} for _ in range(20)]
    replay = {"moves": moves}
    assert enumerate_candidate_plies(replay, stride=4, cap=6) == []


# ---------------------------------------------------------------------------
# Task 6 -- raw-policy geometry features
# ---------------------------------------------------------------------------

def test_policy_features_flat_prior_high_entropy_low_top1():
    n = 250
    priors = [1.0 / n] * n
    feats = _policy_features_from_priors(priors)
    assert feats["normalized_entropy"] == pytest.approx(1.0, abs=1e-6)
    assert feats["top1_prior"] == pytest.approx(1.0 / n, abs=1e-9)
    assert feats["top1_prior"] < 0.01
    assert feats["top4_mass"] >= feats["top1_prior"]
    assert feats["top8_mass"] >= feats["top4_mass"]


def test_policy_features_peaked_prior_top1_about_0_9():
    # The peak sits at index 100 (mid-list), NOT index 0: a buggy
    # `top1_prior = priors[0]` (or an unsorted `priors[:4]`/`priors[:8]` for
    # the mass features) would coincidentally pass if the peak were first --
    # placing it mid-list forces a genuine max/sort over the whole list.
    n = 250
    rest = (1.0 - 0.9) / (n - 1)
    priors = [rest] * 100 + [0.9] + [rest] * (n - 101)
    assert len(priors) == n
    feats = _policy_features_from_priors(priors)
    assert feats["top1_prior"] == pytest.approx(0.9, abs=1e-9)
    assert feats["normalized_entropy"] < 0.3      # concentrated -> low normalized entropy
    assert feats["top4_mass"] >= feats["top1_prior"]
    assert feats["top8_mass"] >= feats["top4_mass"]


# ---------------------------------------------------------------------------
# Task 6 -- forbidden-hash union (primary hash column + fallback reconstruction)
# ---------------------------------------------------------------------------

def test_load_forbidden_hashes_union_from_hash_column(tmp_path):
    p1 = tmp_path / "a.csv"
    p1.write_text("canonical_position_sha1,note\nh1,x\nh2,y\n")
    p2 = tmp_path / "b.csv"
    p2.write_text("canonical_position_sha1,note\nh2,z\nh3,w\n")
    result = load_forbidden_hashes([str(p1), str(p2)])
    assert result == {"h1", "h2", "h3"}


def test_load_forbidden_hashes_fallback_reconstructs_from_replay(tmp_path):
    # selected-A / v16a manifests carry replay_path+position_ply+side_to_move
    # but NO canonical_position_sha1 column (both predate Task 4's hash) --
    # this is the path load_forbidden_hashes must take against them. Build a
    # tiny SYNTHETIC replay on disk (not the real seed20116 corpus) + a CSV
    # row pointing at it, and independently compute the expected hash via the
    # already-tested Task-4/goal_line_trigger_probe_cases primitives.
    state = TwixtState(active_size=8, to_move="red", max_plies_limit=20)
    moves = []
    for _ in range(3):
        mv = state.legal_moves()[0]
        moves.append({"row": mv[0], "col": mv[1]})
        state = state.apply_move(mv)
    replay = {"board_size": 8, "n_moves": len(moves), "moves": moves}
    replay_path = tmp_path / "game_000000.json"
    replay_path.write_text(json.dumps(replay))

    expected_state = position_state(replay, 2, "red")     # ply0 red, ply1 black, ply2 red
    expected_hash = canonical_state_sha1(expected_state)

    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "game_idx,replay_path,position_ply,side_to_move\n"
        f"0,{replay_path},2,red\n"
    )

    result = load_forbidden_hashes([str(csv_path)])
    assert result == {expected_hash}


# ---------------------------------------------------------------------------
# Task 6 -- disjointness assertion
# ---------------------------------------------------------------------------

def test_assert_disjoint_raises_on_forbidden_collision():
    with pytest.raises((ValueError, AssertionError)):
        assert_disjoint(["h1", "h2"], forbidden={"h2", "h9"})


def test_assert_disjoint_raises_on_internal_duplicate():
    with pytest.raises((ValueError, AssertionError)):
        assert_disjoint(["h1", "h1", "h2"], forbidden=set())


def test_assert_disjoint_passes_when_clean():
    assert assert_disjoint(["h1", "h2", "h3"], forbidden={"h9", "h10"}) is None


# ---------------------------------------------------------------------------
# Task 6 -- RESERVE constant
# ---------------------------------------------------------------------------

def test_reserve_is_2x_per_band_quota():
    assert RESERVE == {"target": 2 * TARGET_PER_BAND, "control": 2 * CONTROL_PER_BAND}
    assert RESERVE == {"target": 120, "control": 40}
