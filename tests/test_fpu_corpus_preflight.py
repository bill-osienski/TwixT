"""Tests for the PURE replay-geometry feasibility preflight (design §11.4).

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
AMENDMENT (2026-07-11) §11 -- seed20116 retired; a pure preflight must prove,
from replay geometry alone (n_legal / ply / side; NO NN, NO MCTS, NO
raw-policy), that a (source corpus, enumeration) pair can JOINTLY satisfy the
four structural sampler constraints, or STOP before the evaluator loads.

Everything here is SYNTHETIC geometry: plain `candidates_by_game` dicts (the
builder's output shape) and, for the enumeration-reuse test, a tiny in-memory
`replay` dict of stored n_legal moves. NO real seed20116 files, NO MCTS, NO
evaluator. The crux is:
  * the three walls §11.2 lists (band capacity / side aliasing / ply-bucket)
    must each return feasible=False with the binding constraint named, and
  * `test_soundness_joint_infeasible_but_all_necessary_pass` -- a corpus that
    clears EVERY per-band necessary check yet is JOINTLY infeasible must still
    return feasible=False, proving the constructive WITNESS (not the necessary
    checks) governs feasible=True.
"""
from collections import Counter, defaultdict

import pytest

from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    BANDS,
    MAX_PER_GAME,
    MIN_PLY_GAP,
    SIDE_TOL,
    CORPUS_SIZE,
    QUOTA_PER_BAND,
    band_of,
    ply_bucket_of,
    side_to_move_for_ply,
    enumerate_candidate_plies,
    build_candidates_by_game,
    geometry_feasibility,
    preflight_source,
    PreflightReport,
)

# ---------------------------------------------------------------------------
# Synthetic geometry helpers.
#
# A "candidate" is the builder's tuple-as-dict: game_idx, ply, band, ply_bucket,
# side. geometry_feasibility reads ONLY these fields (band/bucket/side are
# carried explicitly, so a synthetic test can decouple them from ply). All
# ply/side values below are nonetheless parity-consistent (red<->even ply) so
# the fixtures also read as plausible real geometry.
# ---------------------------------------------------------------------------

_BUCKETS = ("opening", "early_mid", "midgame", "late")
# (red_even_ply, black_odd_ply): >= MIN_PLY_GAP apart, both inside the bucket.
_BUCKET_PAIR = {
    "opening": (2, 15),      # 1..15
    "early_mid": (16, 29),   # 16..40
    "midgame": (42, 55),     # 41..90
    "late": (92, 105),       # 91+
}
# (red_even, red_even): both red, >= MIN_PLY_GAP apart, inside the bucket.
_BUCKET_REDPAIR = {
    "opening": (2, 14),
    "early_mid": (16, 28),
    "midgame": (42, 54),
    "late": (92, 104),
}


def _c(game_idx, ply, band, side, bucket):
    return {"game_idx": game_idx, "ply": ply, "band": band,
            "ply_bucket": bucket, "side": side}


def _pair_game(gi, band, bucket):
    """One game contributing a red+black PAIR (>=MIN_PLY_GAP apart) to `band`."""
    rp, bp = _BUCKET_PAIR[bucket]
    return [_c(gi, rp, band, "red", bucket), _c(gi, bp, band, "black", bucket)]


def _by_game_with_counts(counts):
    """{band: n_pair_games} -> candidates_by_game, buckets cycled by game so no
    bucket approaches the 0.5*CORPUS_SIZE cap and every band spans buckets."""
    by_game = {}
    gi = 0
    for band in BANDS:
        for _ in range(counts[band]):
            by_game[gi] = _pair_game(gi, band, _BUCKETS[gi % 4])
            gi += 1
    return by_game


def _feasible_by_game():
    """40 distinct pair-games per band = 120 games, 240 positions, buckets
    spread (~60/bucket) -- a genuinely feasible corpus."""
    return _by_game_with_counts({b: 40 for b in BANDS})


def _wall1_band_capacity():
    """b300_399 supplied by only 20 pair-games -> 40 realizable < 80 demand."""
    return _by_game_with_counts({"b200_299": 40, "b300_399": 20, "b400_plus": 40})


def _wall2_side_aliasing():
    """b200_299 has ample realizable positions (>=80) but they are ALL RED
    (stride aliased onto one ply-parity), so no red+black pair -> side balance
    is impossible even though the band-capacity check passes."""
    by_game = {}
    gi = 0
    for _ in range(60):                    # 60 all-red games -> 120 realizable
        bucket = _BUCKETS[gi % 4]
        r1, r2 = _BUCKET_REDPAIR[bucket]
        by_game[gi] = [_c(gi, r1, "b200_299", "red", bucket),
                       _c(gi, r2, "b200_299", "red", bucket)]
        gi += 1
    for band in ("b300_399", "b400_plus"):
        for _ in range(40):
            by_game[gi] = _pair_game(gi, band, _BUCKETS[gi % 4])
            gi += 1
    return by_game


def _wall3_ply_bucket():
    """b200_299 AND b300_399 candidates are ALL in the opening bucket, so the
    forced opening minimum is 160 > the 120 (=0.5*240) cap -- even though each
    band alone clears capacity and side."""
    by_game = {}
    gi = 0
    for band in ("b200_299", "b300_399"):
        for _ in range(40):
            by_game[gi] = [_c(gi, 2, band, "red", "opening"),
                           _c(gi, 15, band, "black", "opening")]
            gi += 1
    for _ in range(40):
        by_game[gi] = _pair_game(gi, "b400_plus", _BUCKETS[gi % 4])
        gi += 1
    return by_game


def _soundness_joint_infeasible():
    """60 games, each able to form a red+black PAIR in ALL THREE bands (buckets
    rotated so no band is bucket-forced). Every per-band necessary check passes
    (realizable 120>=80, pairs 60>=40, forced bucket 0<=120), but a game capped
    at MAX_PER_GAME can realize only ONE pair -> at most 60 distinct pair-games
    for 120 needed (40/band) -> JOINTLY infeasible. Only the constructive
    witness (not the per-band checks) can catch this."""
    by_game = {}
    for gi in range(60):
        cands = []
        for j, band in enumerate(BANDS):
            bucket = _BUCKETS[(gi + j) % 4]          # 3 distinct buckets/game
            rp, bp = _BUCKET_PAIR[bucket]
            cands.append(_c(gi, rp, band, "red", bucket))
            cands.append(_c(gi, bp, band, "black", bucket))
        by_game[gi] = cands
    return by_game


# ---------------------------------------------------------------------------
# Feasible case -- with a REAL constructive witness
# ---------------------------------------------------------------------------

def test_feasible_corpus_returns_true_with_witness():
    report = geometry_feasibility(_feasible_by_game())
    assert report.feasible is True
    assert report.binding_constraint is None

    w = report.witness
    assert w is not None
    assert len(w) == CORPUS_SIZE == 240

    # (1) band quota: exactly QUOTA_PER_BAND selected per band
    by_band = Counter(c["band"] for c in w)
    for band in BANDS:
        assert by_band[band] == QUOTA_PER_BAND == 80

    # (2) <=MAX_PER_GAME per game and >=MIN_PLY_GAP within a game
    plies_by_game = defaultdict(list)
    for c in w:
        plies_by_game[c["game_idx"]].append(c["ply"])
    assert max(len(v) for v in plies_by_game.values()) <= MAX_PER_GAME
    for plies in plies_by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP

    # (3) per-split side balance |red - black| <= SIDE_TOL
    for split in ("tuning", "frozen_check"):
        sc = Counter(c["side"] for c in w if c["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
    assert sum(1 for c in w if c["split"] == "tuning") == 160
    assert sum(1 for c in w if c["split"] == "frozen_check") == 80

    # (4) ply-bucket <= 0.5 * CORPUS_SIZE
    bc = Counter(c["ply_bucket"] for c in w)
    assert max(bc.values()) <= 0.5 * CORPUS_SIZE


# ---------------------------------------------------------------------------
# The three walls §11.2 -- each must return feasible=False, binding named
# ---------------------------------------------------------------------------

def test_wall1_band_capacity_infeasible():
    report = geometry_feasibility(_wall1_band_capacity())
    assert report.feasible is False
    assert report.witness is None
    assert "band-capacity" in report.binding_constraint
    assert "b300_399" in report.binding_constraint
    # the diagnostic exposes the realized shortfall (like §11.2.1's 52/42)
    assert report.realizable_by_band["b300_399"] < QUOTA_PER_BAND


def test_wall2_side_aliasing_infeasible():
    report = geometry_feasibility(_wall2_side_aliasing())
    assert report.feasible is False
    assert report.witness is None
    assert "side" in report.binding_constraint
    assert "b200_299" in report.binding_constraint
    # band capacity itself is fine -- the wall is purely side (no both-side pair)
    assert report.realizable_by_band["b200_299"] >= QUOTA_PER_BAND
    assert report.pairs_by_band["b200_299"] == 0


def test_wall3_ply_bucket_infeasible():
    report = geometry_feasibility(_wall3_ply_bucket())
    assert report.feasible is False
    assert report.witness is None
    assert "ply-bucket" in report.binding_constraint
    assert "opening" in report.binding_constraint
    assert report.forced_bucket_min["opening"] > 0.5 * CORPUS_SIZE


# ---------------------------------------------------------------------------
# Soundness -- all per-band necessary checks pass, yet JOINTLY infeasible
# ---------------------------------------------------------------------------

def test_soundness_joint_infeasible_but_all_necessary_pass():
    report = geometry_feasibility(_soundness_joint_infeasible())

    # Every per-band NECESSARY check passes individually...
    for band in BANDS:
        assert report.realizable_by_band[band] >= QUOTA_PER_BAND        # capacity ok
        assert report.pairs_by_band[band] * 2 >= QUOTA_PER_BAND         # side ok
    for bucket, forced in report.forced_bucket_min.items():
        assert forced <= 0.5 * CORPUS_SIZE                              # bucket ok

    # ...but the JOINT problem is infeasible, and only the witness catches it.
    assert report.feasible is False
    assert report.witness is None
    assert "joint" in report.binding_constraint


# ---------------------------------------------------------------------------
# Enumeration reuse -- the builder cannot drift from the real scan
# ---------------------------------------------------------------------------

def test_builder_ply_selection_matches_enumerate_candidate_plies():
    moves = [{"n_legal": 250} for _ in range(20)]
    replay = {"moves": moves}
    by_game = build_candidates_by_game({7: replay}, stride=4, cap=6)

    plies = [c["ply"] for c in by_game[7]]
    assert plies == enumerate_candidate_plies(replay, stride=4, cap=6)
    for c in by_game[7]:
        assert c["game_idx"] == 7
        assert c["band"] == band_of(250) == "b200_299"
        assert c["side"] == side_to_move_for_ply(c["ply"])
        assert c["ply_bucket"] == ply_bucket_of(c["ply"])


def test_builder_reuses_per_ply_n_legal_bands_and_irregular_stride():
    # Qualifying plies interleaved with non-qualifying gaps: the builder must
    # stride over the QUALIFYING subsequence (not raw ply index) and read each
    # selected ply's band from its own n_legal -- proving it delegates to the
    # real enumerate/band functions rather than re-deriving them.
    n_legal_by_ply = {0: 250, 1: 50, 2: 350, 3: 50, 4: 450, 5: 50, 6: 260}
    moves = [{"n_legal": n_legal_by_ply[p]} for p in range(7)]
    replay = {"moves": moves}
    by_game = build_candidates_by_game({0: replay}, stride=1, cap=6)

    got = [(c["ply"], c["band"]) for c in by_game[0]]
    # qualifying plies (n_legal>=200) are 0,2,4,6 -> stride-1 keeps all four
    assert [p for p, _ in got] == [0, 2, 4, 6]
    assert got == [(0, "b200_299"), (2, "b300_399"),
                   (4, "b400_plus"), (6, "b200_299")]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_input_same_report():
    d = _feasible_by_game()
    assert geometry_feasibility(d) == geometry_feasibility(d)

    d2 = _wall1_band_capacity()
    assert geometry_feasibility(d2) == geometry_feasibility(d2)


def test_report_is_frozen_dataclass_with_readable_str():
    report = geometry_feasibility(_feasible_by_game())
    assert isinstance(report, PreflightReport)
    with pytest.raises(dataclasses_FrozenInstanceError()):
        report.feasible = False            # frozen: cannot mutate
    s = str(report)
    assert "feasible" in s.lower()
    for band in BANDS:
        assert band in s                    # per-band diagnostics rendered


def dataclasses_FrozenInstanceError():
    import dataclasses
    return dataclasses.FrozenInstanceError


# ---------------------------------------------------------------------------
# I/O wrapper -- thin file->geometry composition (tiny synthetic replays)
# ---------------------------------------------------------------------------

def test_preflight_source_composes_builder_and_core(tmp_path):
    import json
    # Two tiny stored-n_legal replays on disk (NOT the real corpus); the wrapper
    # must read them, build candidates, and return a PreflightReport. A 2-game
    # toy corpus cannot satisfy the 240-row demand, so feasible is False -- what
    # matters here is that the wrapper composes read->build->geometry and yields
    # a real report with per-band diagnostics.
    records = []
    for gi in range(2):
        replay = {"moves": [{"n_legal": 250} for _ in range(12)]}
        p = tmp_path / f"game_{gi:06d}.json"
        p.write_text(json.dumps(replay))
        records.append({"game_idx": gi, "replay_path": str(p)})

    report = preflight_source(records, stride=4, cap=6)
    assert isinstance(report, PreflightReport)
    assert isinstance(report.feasible, bool)
    assert report.feasible is False                       # 2 games << 240 demand
    assert set(report.realizable_by_band) == set(BANDS)
