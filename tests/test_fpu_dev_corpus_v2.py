"""Regression test proving the v1 branching-band + <=50%-ply-bucket-cap
corpus design is mathematically IMPOSSIBLE on the project's fixed 24x24
board (Task 0), plus tests for the v2 phase-primary constants + the
`proposal_cell_of` classifier in `fpu_dev_corpus_v2.py` (Task 1).

Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
  section 0 ("Why v2 (the impossibility that retired v1)"), section 1.2
  (SPLIT_ALLOC_V2), section 1.3 (late coverage floors), section 1.5
  (proposal enumerator / proposal cells).
v2 plan Task 0 ("Impossibility regression test (the WHY, and a guard)") and
Task 1 ("v2 constants + classifiers").

This is the WHY behind retiring `build_fpu_dev_corpus.py`'s v1 branching-band
(n_legal 200-299 / 300-399 / 400+) stratification in favor of the v2
phase-primary design, encoded as a permanent regression test so nobody
resurrects the v1 bands.

The argument (already verified against 800 real games -- asserted here as
pure arithmetic/geometry, NOT re-derived from replay data):
  1. On a 24x24 board each side's legal region is exactly 528 cells on the
     empty board, and each placed peg removes AT MOST one legal cell (for
     either side), so for every reachable position: n_legal >= 528 - ply.
  2. Therefore n_legal <= 399 implies ply >= 528 - 399 = 129.
  3. 129 >= 91, so every position in the two low bands (b200_299, b300_399)
     falls in the "late" ply_bucket_of bucket (ply_bucket_of returns "late"
     for ply >= 91) -- checked here across a range of plies, not just the
     boundary value.
  4. v1's frozen composition needs QUOTA_PER_BAND (80) rows in EACH of the
     two low bands = 160 "late" rows. The <=50% ply-bucket cap on the
     CORPUS_SIZE-row (240) corpus permits at most int(0.5 * 240) = 120.
     160 > 120 => impossible, independent of seed/length/enumeration/FPU.

Pure arithmetic/geometry: only stdlib + the pure v1 helpers/constants +
TwixtState. No evaluator, no MCTS, no GPU/MLX, no checkpoint, no I/O.

Task 1's tests below exercise `fpu_dev_corpus_v2.py`'s frozen phase-primary
constants (PHASES, SPLIT_ALLOC_V2, LATE_TARGET_FLOORS, PROPOSAL_CELLS, the
v2-specific MAX_PER_CELL_PER_GAME, the derived CORPUS_SIZE, and the shared
v1 MIN_PLY_GAP/MAX_PER_GAME/SIDE_TOL re-exports) and its one classifier,
`proposal_cell_of` -- still pure stdlib, still no evaluator/MCTS/GPU/MLX.

Task 3's tests (bottom of the file) exercise the phase-stratified sampler
`sample_v2_rows` + `assign_split_v2` over FABRICATED `kept` screen rows
(plain dicts; synthetic canonical_sha1 strings, never real hashes) -- the
same pure, evaluator-free shape as v1's tests/test_fpu_dev_corpus.py.
"""
import inspect
from collections import Counter, defaultdict

import pytest

from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    CORPUS_SIZE,
    QUOTA_PER_BAND,
    SPLIT_ALLOC,
    band_of,
    ply_bucket_of,
)
from scripts.GPU.alphazero.fpu_dev_corpus_v2 import (
    ASSIGN_ATTEMPTS,
    CELL_ORDER_V2,
    CORPUS_SIZE as CORPUS_SIZE_V2,
    LATE_TARGET_CELL,
    LATE_TARGET_FLOORS,
    MAX_PER_CELL_PER_GAME,
    MAX_PER_GAME,
    MIN_PLY_GAP,
    PHASES,
    PROPOSAL_CELLS,
    SIDE_TOL,
    SPLIT_ALLOC_V2,
    SPLIT_TOTALS,
    SPLITS,
    assign_split_v2,
    enumerate_v2_proposals,
    proposal_cell_of,
    sample_v2_rows,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState


def test_empty_board_legal_moves_is_528_both_sides():
    """Anchor fact: the empty-board legal region is exactly 528 cells for
    BOTH sides on the project's fixed 24x24 board -- the base case of the
    `n_legal >= 528 - ply` invariant (ply == 0)."""
    red = TwixtState(board_size=24, active_size=24, to_move="red")
    black = TwixtState(board_size=24, active_size=24, to_move="black")
    assert len(red.legal_moves()) == 528
    assert len(black.legal_moves()) == 528
    assert red.ply == 0 and black.ply == 0


def test_low_n_legal_forces_late_ply_bucket():
    """n_legal <= 399 implies ply >= 129 (from the `n_legal >= 528 - ply`
    invariant), and every ply from that floor across a wide sampled range
    lands in the REAL ply_bucket_of's "late" bucket -- not merely the single
    boundary value."""
    ply_floor = 528 - 399
    assert ply_floor == 129
    assert ply_floor >= 91   # ply_bucket_of's "late" threshold

    for ply in range(ply_floor, ply_floor + 100):
        assert ply_bucket_of(ply) == "late"

    # Sanity: the bucket boundary itself is exactly where "late" begins, so
    # the low bands' floor (129) is comfortably inside it, not adjacent.
    assert ply_bucket_of(91) == "late"
    assert ply_bucket_of(90) != "late"


def test_v1_bands_impossible_on_24board():
    """The headline regression: v1's two low branching bands (b200_299,
    b300_399) together demand 160 "late" rows, but the <=50% ply-bucket cap
    on the real 240-row corpus permits at most 120 -- so the v1
    branching-band + 50%-cap design is IMPOSSIBLE on this board, regardless
    of source corpus, replay length, enumeration stride, or FPU config.

    Derived from the real v1 constants (SPLIT_ALLOC / QUOTA_PER_BAND /
    CORPUS_SIZE) rather than hard-coded, so this stays honest if those
    constants ever move -- then cross-checked against the brief's explicit
    pinned values for a readable, unambiguous assertion.
    """
    # --- Step 1: the ply-floor half of the argument (pinned form). ---
    assert 528 - 399 == 129 >= 91

    # --- Step 2: the two low bands' combined row requirement, derived
    # directly from the real frozen SPLIT_ALLOC (sum tuning + frozen_check
    # across BOTH roles for exactly the b200_299 / b300_399 cells). ---
    low_bands = {"b200_299", "b300_399"}
    low_band_rows = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, band), alloc in SPLIT_ALLOC.items()
        if band in low_bands
    )
    # QUOTA_PER_BAND is itself derived (uniform per-band total) from
    # SPLIT_ALLOC in build_fpu_dev_corpus.py, so this is an independent
    # cross-check, not a tautology.
    assert low_band_rows == QUOTA_PER_BAND * 2
    assert low_band_rows == 160

    # --- Step 3: the <=50% ply-bucket cap on the real corpus size. ---
    bucket_cap = int(0.5 * CORPUS_SIZE)
    assert bucket_cap == 120
    assert bucket_cap == int(0.5 * 240)

    # --- Step 4: the pinned form from the brief (explicit + readable). ---
    assert 80 + 80 > int(0.5 * 240)

    # --- The impossibility itself. ---
    assert low_band_rows > bucket_cap


# ---------------------------------------------------------------------------
# Task 1 -- v2 phase-primary constants + proposal_cell_of classifier
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.2 (phase-stratified SPLIT_ALLOC_V2), Sec 1.3 (late coverage
#   floors), Sec 1.5 (proposal enumerator / proposal cells).
# v2 plan Task 1 ("v2 constants + classifiers"). Exercises
# scripts/GPU/alphazero/fpu_dev_corpus_v2.py's pure constants + its one
# classifier -- still stdlib-only, no MCTS/GPU/MLX/evaluator/I-O.
# ---------------------------------------------------------------------------

def test_phases_matches_real_ply_bucket_of_vocabulary():
    """PHASES must be the v1 ply_bucket_of bucket vocabulary, in the same
    opening/early_mid/midgame/late order -- checked against the REAL
    imported ply_bucket_of (not a hand-typed duplicate that could silently
    drift from v1's naming) at every v1 bucket boundary (mirrors
    test_ply_bucket_boundaries in tests/test_fpu_dev_corpus.py)."""
    assert PHASES == ("opening", "early_mid", "midgame", "late")
    assert ply_bucket_of(1) == PHASES[0]
    assert ply_bucket_of(15) == PHASES[0]
    assert ply_bucket_of(16) == PHASES[1]
    assert ply_bucket_of(40) == PHASES[1]
    assert ply_bucket_of(41) == PHASES[2]
    assert ply_bucket_of(90) == PHASES[2]
    assert ply_bucket_of(91) == PHASES[3]
    assert ply_bucket_of(300) == PHASES[3]


def test_shared_v1_constants_pinned():
    """MIN_PLY_GAP / MAX_PER_GAME / SIDE_TOL are the IDENTICAL v1 frozen
    constants (build_fpu_dev_corpus.py) -- imported+re-exported by
    fpu_dev_corpus_v2 rather than restated, so pin their frozen values here
    too: a v1 drift then fails loudly in the v2 module as well."""
    assert MIN_PLY_GAP == 12
    assert MAX_PER_GAME == 2
    assert SIDE_TOL == 2


def test_max_per_cell_per_game_is_a_distinct_v2_constant():
    """MAX_PER_CELL_PER_GAME (Task-2 enumerator: caps PROPOSALS per game per
    proposal-cell) and MAX_PER_GAME (Task-3 sampler: caps SELECTED rows per
    game GLOBALLY across all cells) are separate frozen constants that
    happen to share the value 2 while gating different pipeline stages --
    both must exist independently under their own names."""
    assert MAX_PER_CELL_PER_GAME == 2
    assert MAX_PER_GAME == 2


def test_split_alloc_v2_totals_derived():
    """SPLIT_ALLOC_V2 sums to the frozen composition -- each total is
    RE-DERIVED directly from the dict (not trusting the module's own
    arithmetic), then cross-checked against the brief's pinned literals
    (240 grand / 180 target-60 control / 160 tuning-80 frozen_check)."""
    target_total = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, _phase), alloc in SPLIT_ALLOC_V2.items()
        if role == "target"
    )
    control_total = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, _phase), alloc in SPLIT_ALLOC_V2.items()
        if role == "control"
    )
    tuning_total = sum(alloc["tuning"] for alloc in SPLIT_ALLOC_V2.values())
    frozen_total = sum(alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values())
    grand_total = sum(
        alloc["tuning"] + alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values()
    )

    assert target_total == 180
    assert control_total == 60
    assert tuning_total == 160
    assert frozen_total == 80
    assert grand_total == 240

    # Every phase contributes IDENTICALLY (45 target / 15 control) -- 45/15
    # both divide evenly, unlike v1's odd-quota control bands (13/7 vs
    # 14/6), so there is no per-phase asymmetry to special-case.
    assert set(SPLIT_ALLOC_V2.keys()) == {
        (role, phase) for role in ("target", "control") for phase in PHASES
    }
    for phase in PHASES:
        assert SPLIT_ALLOC_V2[("target", phase)] == {"tuning": 30, "frozen_check": 15}
        assert SPLIT_ALLOC_V2[("control", phase)] == {"tuning": 10, "frozen_check": 5}


def test_corpus_size_v2_is_derived_not_hardcoded():
    """CORPUS_SIZE (v2) must be DERIVED from SPLIT_ALLOC_V2 -- mirroring how
    build_fpu_dev_corpus.py derives its own CORPUS_SIZE
    (build_fpu_dev_corpus.py:115) rather than hard-coding 240. Recomputed
    independently here (not just re-reading the module's own arithmetic)
    and cross-checked against the pinned literal."""
    recomputed = sum(
        alloc["tuning"] + alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values()
    )
    assert CORPUS_SIZE_V2 == recomputed
    assert CORPUS_SIZE_V2 == 240


def test_late_target_floors_values():
    assert LATE_TARGET_FLOORS == {"b300_399": 12, "b200_299": 12}


def test_proposal_cells_exact_six_in_load_bearing_order():
    """PROPOSAL_CELLS order is deterministic and load-bearing for later
    tasks (Task 2's enumerator iterates cells in this order): the three
    non-late phases (band None) in PHASES order, then the three late bands
    in DESCENDING branching order (b400_plus, b300_399, b200_299) -- per
    the v2 controller resolution, NOT v1's BANDS ascending order."""
    assert PROPOSAL_CELLS == [
        ("opening", None),
        ("early_mid", None),
        ("midgame", None),
        ("late", "b400_plus"),
        ("late", "b300_399"),
        ("late", "b200_299"),
    ]
    assert len(PROPOSAL_CELLS) == 6
    assert len(set(PROPOSAL_CELLS)) == 6   # no duplicate cells


def test_proposal_cell_of_non_late_ignores_n_legal():
    """Non-late phases collapse to (phase, None) regardless of n_legal --
    the brief's exact example plus extra phases/extreme n_legal values."""
    assert proposal_cell_of("opening", 520) == ("opening", None)
    assert proposal_cell_of("early_mid", 10) == ("early_mid", None)
    assert proposal_cell_of("midgame", 528) == ("midgame", None)


def test_proposal_cell_of_late_splits_by_band():
    """The brief's exact three examples, plus the sub-200 ineligibility
    sentinel (controller resolution #1)."""
    assert proposal_cell_of("late", 520) == ("late", "b400_plus")
    assert proposal_cell_of("late", 350) == ("late", "b300_399")
    assert proposal_cell_of("late", 250) == ("late", "b200_299")
    # Sub-200 late position: intentionally NOT a PROPOSAL_CELLS member --
    # this is how sub-200 positions become ineligible (Task 2's enumerator
    # additionally enforces n_legal >= 200 for every cell; this sentinel is
    # belt-and-suspenders, not the only guard).
    assert proposal_cell_of("late", 150) == ("late", None)
    assert ("late", None) not in PROPOSAL_CELLS


def test_proposal_cell_of_late_matches_real_band_of():
    """Reuse check: proposal_cell_of's band element must equal the REAL
    imported band_of(n_legal) -- not a reimplemented/hand-typed copy of its
    boundaries -- checked across every band_of boundary, including the
    sub-200 None case."""
    for n_legal in (1000, 400, 399, 300, 299, 200, 199, 0):
        assert proposal_cell_of("late", n_legal) == ("late", band_of(n_legal))


def test_proposal_cell_of_outputs_land_in_proposal_cells():
    """Every phase/n_legal combination that SHOULD be eligible maps into an
    actual PROPOSAL_CELLS member -- ties the classifier and the cell list
    together so they can't silently drift apart."""
    for phase in ("opening", "early_mid", "midgame"):
        assert proposal_cell_of(phase, 999) in PROPOSAL_CELLS
        assert proposal_cell_of(phase, 1) in PROPOSAL_CELLS
    for n_legal, band in ((520, "b400_plus"), (350, "b300_399"), (250, "b200_299")):
        cell = proposal_cell_of("late", n_legal)
        assert cell in PROPOSAL_CELLS
        assert cell == ("late", band)


# ---------------------------------------------------------------------------
# Task 2 -- enumerate_v2_proposals (phase-aware side-opposed proposal
# enumerator, no global stride)
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.5 (proposal enumerator / proposal cells).
# v2 plan Task 2. Exercises scripts/GPU/alphazero/fpu_dev_corpus_v2.py's
# `enumerate_v2_proposals` -- still stdlib-only synthetic replays (a dict
# with `game_idx` + `moves`, each move carrying `n_legal`), no
# evaluator/MCTS/GPU/MLX/I-O; `per_ply_n_legal`'s reconstruction fallback
# (real TwixtState replay) is v1's own concern and already covered by v1's
# tests, not re-exercised here.
#
# Fixtures use the board's own physical lower bound, `n_legal >= 528 - ply`
# (Task 0's regression), so every fixture's n_legal values are REACHABLE at
# the plies they claim -- most via `_honest_replay`'s default TIGHT schedule
# `n_legal(ply) = 528 - ply`, which conveniently makes each phase/band's
# eligible-ply range exactly the contiguous span the design's own geometry
# predicts (matching the canonical example pairs already used elsewhere in
# this suite, e.g. tests/test_fpu_corpus_preflight.py's `_BUCKET_PAIR`).
# One test (`test_pair_output_order_is_ascending_ply_not_side_order`)
# overrides specific plies to a HIGHER (still honest -- the invariant is a
# LOWER bound only) band on purpose, to construct a cell whose selected pair
# is black-before-red; another (sub-200 guard) is explicitly and only
# unrealistic on purpose -- see its own docstring.
# ---------------------------------------------------------------------------

def _honest_replay(game_idx, n_moves, overrides=None):
    """Synthetic replay `{"game_idx": ..., "moves": [{"n_legal": ...}, ...]}`
    covering plies `0..n_moves-1`. Defaults every ply to the TIGHT physical
    floor `528 - ply` (Task 0's `n_legal >= 528 - ply` invariant, held as an
    equality) -- monotonically decreasing, so every phase/band region is
    reachable exactly where the design geometry predicts. `overrides` (ply ->
    n_legal) replaces individual plies; every caller-supplied override in
    this file still respects the >= 528 - ply LOWER bound (never below it),
    so all fixtures stay physically honest even when overridden.
    """
    overrides = overrides or {}
    moves = [{"n_legal": overrides.get(ply, 528 - ply)} for ply in range(n_moves)]
    return {"game_idx": game_idx, "moves": moves}


def test_cell_yields_side_opposed_pair_at_least_min_gap_apart():
    """Brief case 1: a cell yields a red+black pair >=12 apart -- the two
    sides differ and |ply gap| >= MIN_PLY_GAP. Opening (ply 0-15, all
    n_legal>=513 hence eligible): the earliest-satisfying pair is (0, 13)."""
    replay = _honest_replay(game_idx=1, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    opening_rows = [p for p in proposals if p["proposal_cell"] == ("opening", None)]
    assert len(opening_rows) == 2
    sides = {row["side"] for row in opening_rows}
    assert sides == {"red", "black"}
    plies = [row["ply"] for row in opening_rows]
    assert abs(plies[0] - plies[1]) >= MIN_PLY_GAP
    assert sorted(plies) == [0, 13]


def test_cell_caps_at_max_per_cell_per_game_despite_abundant_candidates():
    """Brief case 2: a cell with MANY eligible plies still yields <=2.
    Midgame (ply 41-90, 50 plies, all honestly eligible) has far more than 2
    valid opposed plies on each side, yet the cell still yields exactly
    MAX_PER_CELL_PER_GAME (2) proposals -- one pair, not every possible
    pair."""
    replay = _honest_replay(game_idx=2, n_moves=91)
    proposals = enumerate_v2_proposals(replay)
    midgame_rows = [p for p in proposals if p["proposal_cell"] == ("midgame", None)]
    assert len(midgame_rows) == MAX_PER_CELL_PER_GAME
    assert len(midgame_rows) <= 2


def test_cell_with_no_valid_gap_pair_yields_zero():
    """Brief case 3: a cell with no valid opposed-gap pair yields 0 for that
    cell -- never a lone unpaired proposal. A 9-ply game (ply 0-8, all
    "opening") has BOTH reds ([0,2,4,6,8]) and blacks ([1,3,5,7]) present,
    but the widest possible span is only 8 (< MIN_PLY_GAP), so no pair
    qualifies and the whole game yields nothing (every other cell has zero
    candidates at all, trivially)."""
    replay = _honest_replay(game_idx=20, n_moves=9)
    assert enumerate_v2_proposals(replay) == []


def test_late_cell_only_selects_matching_band():
    """Brief case 4: late cells only select plies whose band_of(n_legal)
    matches the cell's band. late x b300_399 is reachable only at ply>=129
    (Task 0's floor); under the tight honest schedule this cell's pair is
    (130, 143), both genuinely in the 300-399 range -- never a b400_plus or
    b200_299 ply leaking in from the same "late" phase."""
    replay = _honest_replay(game_idx=3, n_moves=230)
    proposals = enumerate_v2_proposals(replay)
    cell_rows = [p for p in proposals if p["proposal_cell"] == ("late", "b300_399")]
    assert len(cell_rows) == 2
    for row in cell_rows:
        assert row["ply"] >= 129
        assert band_of(row["n_legal"]) == "b300_399"
    assert {row["side"] for row in cell_rows} == {"red", "black"}
    plies = [row["ply"] for row in cell_rows]
    assert abs(plies[0] - plies[1]) >= MIN_PLY_GAP
    assert sorted(plies) == [130, 143]


def test_determinism_same_replay_same_proposals():
    """Brief case 5: same replay -> same proposals, called twice."""
    replay = _honest_replay(game_idx=42, n_moves=330)
    first = enumerate_v2_proposals(replay)
    second = enumerate_v2_proposals(replay)
    assert first == second


def test_full_replay_yields_up_to_twelve_proposals_in_cell_then_ply_order():
    """A single game can yield up to 12 proposals (2 x 6 cells) -- NOT
    capped by the global <=2/game rule, which is a Task-3 SAMPLER concern
    over SELECTED rows, never enforced by the enumerator (brief note). A
    330-ply honest game reaches every one of the 6 PROPOSAL_CELLS (opening,
    early_mid, midgame, late x b400_plus, late x b300_399, late x b200_299).
    Output order is deterministic: cells in PROPOSAL_CELLS order, each cell's
    pair in ascending ply."""
    replay = _honest_replay(game_idx=42, n_moves=330)
    proposals = enumerate_v2_proposals(replay)

    assert len(proposals) == 12
    got_cells_in_order = [proposals[i]["proposal_cell"] for i in range(0, 12, 2)]
    assert got_cells_in_order == PROPOSAL_CELLS
    for i in range(0, 12, 2):
        pair = proposals[i:i + 2]
        assert pair[0]["proposal_cell"] == pair[1]["proposal_cell"]
        assert pair[0]["ply"] < pair[1]["ply"]   # ascending ply within the cell

    expected_plies = [(0, 13), (16, 29), (42, 55), (92, 105), (130, 143), (230, 243)]
    got_plies = [(proposals[i]["ply"], proposals[i + 1]["ply"]) for i in range(0, 12, 2)]
    assert got_plies == expected_plies

    for row in proposals:
        assert row["game_idx"] == 42
        assert row["phase"] == row["proposal_cell"][0]
        assert row["phase"] == ply_bucket_of(row["ply"])
        assert row["side"] in ("red", "black")


def test_proposal_dict_has_exact_schema():
    """Interface: each proposal dict carries EXACTLY game_idx, ply, side,
    phase, n_legal, band, proposal_cell -- no more, no fewer."""
    replay = _honest_replay(game_idx=5, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    expected_keys = {"game_idx", "ply", "side", "phase", "n_legal", "band", "proposal_cell"}
    assert proposals   # sanity: this fixture is non-empty
    for row in proposals:
        assert set(row.keys()) == expected_keys
        assert row["game_idx"] == 5


def test_band_field_is_band_of_n_legal_not_cell_band_component():
    """Interface note: `band` is ALWAYS the real `band_of(n_legal)`, never
    the cell's own band component -- an `("opening", None)` cell's proposals
    still record their real band (b400_plus: opening n_legal>=513 on this
    board), not None."""
    replay = _honest_replay(game_idx=6, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    opening_rows = [p for p in proposals if p["proposal_cell"] == ("opening", None)]
    assert len(opening_rows) == 2
    for row in opening_rows:
        assert row["band"] == band_of(row["n_legal"])
        assert row["band"] == "b400_plus"
        assert row["proposal_cell"][1] is None   # the cell's OWN band component


def test_low_n_legal_excludes_ply_even_when_phase_matches():
    """n_legal >= 200 is an independent eligibility guard, not solely reliant
    on proposal_cell_of's own ("late", None) sub-200 sentinel
    ("belt-and-suspenders", per that function's docstring) -- checked
    directly with an otherwise phase-matching low value. NOTE: unlike this
    file's other fixtures, this one is deliberately NOT physically honest (an
    opening ply can never really have n_legal < 200 on this board -- Task
    0's floor already forces >=513 that early); it exists solely to prove
    the enumerator's OWN n_legal>=200 guard fires independently of
    proposal_cell_of, not to claim a reachable position."""
    moves = [{"n_legal": 150} for _ in range(16)]
    replay = {"game_idx": 10, "moves": moves}
    assert enumerate_v2_proposals(replay) == []


def test_pair_output_order_is_ascending_ply_not_side_order():
    """Interface: within a cell the pair is output in ASCENDING PLY, not
    "red always first". Constructed (still honestly -- the >=528-ply
    invariant is a LOWER bound, so pushing a ply's n_legal UP into a higher
    band is always honest) so late x b300_399's only eligible black is ply
    129 and its only eligible red is ply 142 (plies 130-141 pushed into
    b400_plus, hence ineligible for THIS cell) -- the pair is (black@129,
    red@142), so a naive "always emit red then black" implementation would
    emit them in the wrong order."""
    overrides = {129: 399, 142: 390}
    for ply in range(130, 142):
        overrides[ply] = 450   # honest (>= the ply's floor); b400_plus, not b300_399
    replay = _honest_replay(game_idx=7, n_moves=143, overrides=overrides)
    proposals = enumerate_v2_proposals(replay)
    cell_rows = [p for p in proposals if p["proposal_cell"] == ("late", "b300_399")]
    assert [p["ply"] for p in cell_rows] == [129, 142]
    assert [p["side"] for p in cell_rows] == ["black", "red"]


# ---------------------------------------------------------------------------
# Task 3 -- phase-stratified sampler (`sample_v2_rows`) + hard late floors
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.2 (SPLIT_ALLOC_V2), Sec 1.3 (late coverage floors).
# v2 plan Task 3. Operates on FABRICATED `kept` screen rows (the Task-5 screen
# stage produces the real ones): plain dicts carrying game_idx, role, phase,
# band, side, ply, canonical_sha1. Synthetic sha1 strings, no evaluator/MCTS/
# GPU/MLX/I-O -- exactly v1's tests/test_fpu_dev_corpus.py shape, re-cut for
# v2's (role, phase) cells, its GLOBAL <=MAX_PER_GAME rule, and the floors.
#
# Fixture geometry is PHYSICALLY HONEST on the 24x24 board (Task 0's
# `n_legal >= 528 - ply` floor + `side_to_move_for_ply`'s red-on-even parity):
#   * red rows sit on EVEN plies, black rows on ODD plies;
#   * b300_399 exists only at ply >= 528-399 = 129 and b200_299 only at
#     ply >= 528-299 = 229 -- BOTH "late", which is exactly why the floors live
#     in the late phase;
#   * every non-late position is necessarily b400_plus (ply <= 90 =>
#     n_legal >= 438), so all opening/early_mid/midgame rows carry that band as
#     their recorded covariate.
# ---------------------------------------------------------------------------

# (red_ply, black_ply) per phase: >= MIN_PLY_GAP apart, both inside the phase's
# real ply_bucket_of range, correct side parity.
_V2_PHASE_PLIES = {
    "opening": (2, 15),       # ply_bucket_of: 1-15
    "early_mid": (16, 29),    # 16-40
    "midgame": (42, 55),      # 41-90
}
# late-only, per BAND -- the floors' geometry (b400_plus anywhere late;
# b300_399 only at ply >= 129; b200_299 only at ply >= 229).
_V2_LATE_BAND_PLIES = {
    "b400_plus": (92, 105),
    "b300_399": (130, 143),
    "b200_299": (230, 243),
}


def _v2_row(game_idx, role, phase, band, side, ply):
    """One fabricated KEPT screen row (globally-unique synthetic sha1)."""
    return {
        "game_idx": game_idx, "role": role, "phase": phase, "band": band,
        "side": side, "ply": ply,
        "canonical_sha1": f"v2-{game_idx:05d}-{ply:04d}-{side}",
    }


def _v2_game(game_idx, role, phase, band):
    """One game = ONE side-opposed pair (red on an even ply + black on an odd
    ply, >= MIN_PLY_GAP apart) in ONE (role, phase) cell -- v1 `_game_rows`'
    shape, re-cut for v2's phase cells and honest band geometry."""
    p_red, p_black = (_V2_LATE_BAND_PLIES[band] if phase == "late"
                      else _V2_PHASE_PLIES[phase])
    return [
        _v2_row(game_idx, role, phase, band, "red", p_red),
        _v2_row(game_idx, role, phase, band, "black", p_black),
    ]


# Per-cell game supply. 240 rows at <=2 rows/game needs >=120 games; the
# whole-game split assignment additionally needs ~23 games per TARGET cell (15
# to fill tuning's 30 + 8 to fill frozen_check's 15) and ~8 per CONTROL cell --
# so every cell here carries a real surplus over its own minimum.
#
# The (target, late) cell's band mix is the FLOOR DISCRIMINATOR: its 40
# b400_plus games (80 rows -- more than the whole 45-row late-target quota) come
# FIRST, i.e. at the LOWEST game_idx, and the round-robin visits candidate games
# in ascending game_idx. So an earliest-game fill with NO floor-satisfaction pass
# takes all 45 late-target rows from b400_plus and misses BOTH floors, even
# though a floor-satisfying selection plainly exists. 20 games (40 rows) per
# floor band is a surplus that survives the split assignment sending up to 8
# late-target games to frozen_check (>=12 floor-band games always remain in
# tuning), so the floors are reachable under ANY seed.
#
# The (control, late) cell deliberately carries floor-BAND rows too, floor-band
# games FIRST so they are actually SELECTED: the floors count ONLY late TARGET
# rows, so a sampler that credited a late CONTROL row to a floor counter would
# under-fill the real floor and the final floor verification would fire.
_V2_POOL_SPEC = {
    ("target", "opening"): [("b400_plus", 50)],
    ("control", "opening"): [("b400_plus", 30)],
    ("target", "early_mid"): [("b400_plus", 50)],
    ("control", "early_mid"): [("b400_plus", 30)],
    ("target", "midgame"): [("b400_plus", 50)],
    ("control", "midgame"): [("b400_plus", 30)],
    ("target", "late"): [("b400_plus", 40), ("b300_399", 20), ("b200_299", 20)],
    ("control", "late"): [("b300_399", 10), ("b200_299", 10), ("b400_plus", 10)],
}


def _v2_singleton_game(game_idx, role, phase, band, side):
    """One game contributing a SINGLE row to one cell -- because the screen
    classified its side-opposed partner into the OTHER role, or grey-dropped it
    (`raw_policy_role` reads each row independently). Routine on a real screen, and
    the ONLY way a game can be forced to shift a split's side balance."""
    p_red, p_black = (_V2_LATE_BAND_PLIES[band] if phase == "late"
                      else _V2_PHASE_PLIES[phase])
    return [_v2_row(game_idx, role, phase, band, side,
                    p_red if side == "red" else p_black)]


def _v2_pool(overrides=None, start_gi=0):
    """Build a fabricated `kept` pool from a {(role, phase): [(band, n_games)]}
    spec (defaulting to `_V2_POOL_SPEC`), game_idx ascending from `start_gi` in
    CELL_ORDER_V2 order. (`start_gi` lets a fixture reserve the LOW game_idx range
    for its own probe games -- the sampler's deterministic tie-breaks favour the
    lowest game_idx, so a probe game must be able to sit below the bulk supply.)"""
    spec = dict(_V2_POOL_SPEC)
    spec.update(overrides or {})
    rows, gi = [], start_gi
    for cell in CELL_ORDER_V2:
        for band, n_games in spec[cell]:
            for _ in range(n_games):
                rows.extend(_v2_game(gi, cell[0], cell[1], band))
                gi += 1
    return rows


def _abundant_pool_v2():
    return _v2_pool()


def _insufficient_pool_v2():
    """Abundant everywhere except (target, midgame), starved to 3 games (6 rows)
    against its 45-row demand -> assign_split_v2's per-cell capacity PRECHECK
    fires before any selection."""
    return _v2_pool({("target", "midgame"): [("b400_plus", 3)]})


def _pool_late_floor_unmeetable():
    """Every phase quota is comfortably fillable -- the (target, late) cell holds
    85 games (170 rows) for its 45-row quota -- but the pool contains only FIVE
    b300_399 target games (10 rows), below the 12-row floor, so NO selection can
    satisfy LATE_TARGET_FLOORS. The composition round-robin therefore SUCCEEDS
    and only the hard floor verification can catch it."""
    return _v2_pool({("target", "late"): [("b400_plus", 60), ("b300_399", 5),
                                          ("b200_299", 20)]})


def _pool_gap_starved_cell_v2():
    """(control, opening) is supplied ONLY by games whose two rows sit
    < MIN_PLY_GAP apart (red@2 + black@9), so each yields exactly ONE pickable
    row. Both capacity PRECHECKS pass (8 games x min(2, MAX_PER_GAME) = 16 >= the
    15-row demand), yet the round-robin cannot reach the cell's quota -> the
    exact-or-raise `final-manifest shortfall`, a DIFFERENT failure from
    _insufficient_pool_v2's capacity precheck. Proves a shortfall is never
    silently truncated."""
    rows = _v2_pool({("control", "opening"): []})
    gi = 10_000
    for _ in range(8):
        rows.append(_v2_row(gi, "control", "opening", "b400_plus", "red", 2))
        rows.append(_v2_row(gi, "control", "opening", "b400_plus", "black", 9))
        gi += 1
    return rows


def _pool_global_two_per_game_starved():
    """100 games, each offering a side-opposed pair in ALL EIGHT cells (16 rows
    per game). Every PER-CELL capacity is ample (100 games x min(2, MAX_PER_GAME)
    = 200 >= every cell's demand), so a v1-style per-cell accounting sees a
    healthy pool -- but under v2's GLOBAL <=MAX_PER_GAME rule those 100 games can
    yield at most 200 < CORPUS_SIZE (240) rows, so the corpus is impossible. Only
    an accounting that caps a game's TOTAL contribution can see it."""
    rows = []
    for gi in range(100):
        for cell in CELL_ORDER_V2:
            band = "b400_plus" if cell[1] != "late" else "b400_plus"
            rows.extend(_v2_game(gi, cell[0], cell[1], band))
    return rows


def _pool_with_multicell_game_v2():
    """Abundant pool, but game 0 spans TWO cells with FOUR rows:
    (target, opening) red@2 + black@15 and (target, early_mid) black@27 + red@40.
    EVERY pair of those four plies is >= MIN_PLY_GAP apart and no hash repeats,
    so nothing but the GLOBAL <=MAX_PER_GAME rule can hold game 0 to 2 rows -- a
    v1-style PER-CELL cap would take all four (2 from each cell), because the
    round-robin reaches game 0 first (lowest game_idx) in BOTH of its cells,
    which share one split (whole-game isolation)."""
    rows = [r for r in _abundant_pool_v2() if r["game_idx"] != 0]
    rows += [
        _v2_row(0, "target", "opening", "b400_plus", "red", 2),
        _v2_row(0, "target", "opening", "b400_plus", "black", 15),
        _v2_row(0, "target", "early_mid", "b400_plus", "black", 27),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 40),
    ]
    return rows


def _pool_v2_gap_probe():
    """Two gap discriminators the abundant pool cannot reach.

    game 0 -- a CROSS-CELL gap: ONE row in (target, opening) (red@8) and TWO in
    (target, early_mid) (red@16, red@28). Once red@8 is taken, red@16 is only 8
    plies away: a sampler enforcing the gap only WITHIN a cell (v1's
    `sample_dev_rows` does exactly that -- its `positions` are pre-filtered to
    one cell) would take it. Both early_mid rows are RED, so `_choose_positions`'
    take_n==1 side-steering is a TIE and breaks to the LOWER ply -- such a mutant
    deterministically picks 16 regardless of the running side balance, while the
    correct GLOBAL gap filter leaves only 28.

    game 1 -- v1's within-cell probe, re-cut: FOUR (target, midgame) rows red@42,
    red@46, black@55, black@71. cap+gap pick exactly {42, 55}: dropping the
    gap-skip would take 46 (4 plies away), dropping the <=2 cap would add 71.

    A game can only ever contribute to TWO cells by giving ONE row to each -- and a
    one-row take is, by definition, not a side-opposed pair, so under the sampler's
    side-aware fill it is a PASS-2 (steered) draw. Pass 1 would otherwise fill both
    of game 0's cells from side-opposed-pair games alone and game 0 would simply
    never be reached, making the cross-cell probe vacuous. So this pool supplies
    (target, opening) and (target, early_mid) with SINGLETON games only (70 each,
    alternating red-only / black-only, so Pass 2's steering can still balance
    them): both cells are then genuinely filled by Pass 2, and game 0 -- holding
    the LOWEST game_idx, which is Pass 2's final tie-break -- is reliably drawn in
    each. The bulk supply is pushed to game_idx >= 100 to keep that range free.
    """
    rows = _v2_pool({("target", "opening"): [], ("target", "early_mid"): []},
                    start_gi=100)
    gi = 40_000
    for cell in (("target", "opening"), ("target", "early_mid")):
        for k in range(70):
            rows.extend(_v2_singleton_game(gi, cell[0], cell[1], "b400_plus",
                                           "red" if k % 2 == 0 else "black"))
            gi += 1
    rows += [
        _v2_row(0, "target", "opening", "b400_plus", "red", 8),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 16),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 28),
        _v2_row(1, "target", "midgame", "b400_plus", "red", 42),
        _v2_row(1, "target", "midgame", "b400_plus", "red", 46),
        _v2_row(1, "target", "midgame", "b400_plus", "black", 55),
        _v2_row(1, "target", "midgame", "b400_plus", "black", 71),
    ]
    return rows


# --- the SAME-SIDE fixtures (the side-balance regression) --------------------
# `_abundant_pool_v2` and every fixture above it is built from `_v2_game`: one
# game = ONE side-opposed pair in ONE cell. Such a pool is STRUCTURALLY INCAPABLE
# of offering a cell two same-side rows, so it can never express this regression
# -- which is exactly how v1's "a whole-game 2-take is side-neutral for free"
# premise (build_fpu_dev_corpus.SIDE_TOL's own note) survived into v2 untested.
#
# It is FALSE in v2 for two independent reasons, and each gets a fixture below:
#   * v2's (role, "late") SAMPLER cell aggregates THREE proposal cells
#     (late/b400_plus, late/b300_399, late/b200_299), so ONE game can offer that
#     single cell up to 3 reds + 3 blacks; and
#   * the screen's `raw_policy_role` classifies each row INDEPENDENTLY, so a
#     proposal pair's red can land in `target` while its black lands in `control`.
# Both stay PHYSICALLY HONEST (n_legal >= 528 - ply; red on even plies): the
# same-side rows below are all REDS on EVEN plies, b300_399 only at ply >= 129 and
# b200_299 only at ply >= 229.

# One game's (target, late) contribution = TWO REDS from two DIFFERENT floor
# bands -- red@130 (b300_399, ply >= 129 OK) and red@230 (b200_299, ply >= 229
# OK), 100 plies apart so NO gap filter binds -- while the opposed blacks of those
# very proposal pairs (143 / 243, odd plies) are classified `control`.
_V2_SAME_SIDE_LATE_PLIES = {"b300_399": (130, 143), "b200_299": (230, 243)}


def _pool_same_side_late_floor():
    """(target, late)'s ONLY floor-band supply is 20 SAME-SIDE (red) multi-band
    games, so the LATE_TARGET_FLOORS themselves FORCE the sampler to draw
    same-side rows: its 40 b400_plus late-target games (opposed pairs) cannot
    reach either floor, and every b300_399 / b200_299 target row in the pool is
    red. The floor pass then draws each floor band from a game via a SEPARATE
    band-restricted take_n == 1 call, so no 2-take rule can pair them -- 12 + 12
    = 24 RED late-target rows, and tuning ends up ~24 reds heavy.

    A side-balanced selection does exist in principle (the sampler could steer the
    other cells' leftovers black), but the greedy round-robin fills every other
    cell with side-opposed PAIRS and has no such slack -- so the correct behaviour
    is to RAISE, never to emit the skewed manifest. This is the reviewer's
    falsifying pool.
    """
    rows = _v2_pool({("target", "late"): [("b400_plus", 40)]})
    gi = 20_000
    for _ in range(20):
        for band, (p_red, p_black) in _V2_SAME_SIDE_LATE_PLIES.items():
            rows.append(_v2_row(gi, "target", "late", band, "red", p_red))
            rows.append(_v2_row(gi, "control", "late", band, "black", p_black))
        gi += 1
    return rows


# --- the WHOLE-GAME SPLIT ASSIGNMENT fixtures --------------------------------
# Sub-MIN_PLY_GAP side-opposed pairs: the two rows sit only 7 plies apart, so the
# game's PROFILE reports 2 rows for the cell but the fill can only ever pick ONE
# (`_choose_positions` cannot take a second row inside the gap). That gap between
# the assignment greedy's optimistic accounting and what the fill can realize is
# exactly what turns a starved `frozen_check` candidate pool into a shortfall.
_V2_TIGHT_PHASE_PLIES = {          # red on EVEN plies, black on ODD, 7 apart
    "opening": (2, 9),             # ply_bucket_of: 1-15
    "early_mid": (16, 23),         # 16-40
    "midgame": (42, 49),           # 41-90
    "late": (92, 99),              # 91+  (n_legal >= 528-99 = 429 => b400_plus)
}
# The four phases a single replay really does span, with the honest per-phase plies
# `_V2_PHASE_PLIES` / `_V2_LATE_BAND_PLIES` already pin.
_V2_ALL_PHASE_CELLS = ("opening", "early_mid", "midgame", "late")


def _v2_tight_game(game_idx, role, phase):
    """A GAP-CRIPPLED game: a side-opposed pair < MIN_PLY_GAP apart, so its profile
    claims 2 rows for the cell but it can only ever yield 1."""
    p_red, p_black = _V2_TIGHT_PHASE_PLIES[phase]
    return [_v2_row(game_idx, role, phase, "b400_plus", "red", p_red),
            _v2_row(game_idx, role, phase, "b400_plus", "black", p_black)]


def _v2_multicell_game(game_idx, role):
    """One game spanning ALL FOUR phases -- what a real replay does. It holds 8 rows
    but the GLOBAL <=MAX_PER_GAME budget still lets it give only 2, so its cells
    CONTEND for it. A large surplus of these is what made the old raw-count greedy
    dump nearly every game into `tuning`."""
    rows = []
    for phase in _V2_ALL_PHASE_CELLS:
        rows.extend(_v2_game(game_idx, role, phase, "b400_plus"))
    return rows


def _pool_frozen_starving_multicell():
    """The SPLIT-ASSIGNMENT regression pool -- a deterministic, fast (~400-game)
    distillation of what realistic screens actually do to the old greedy.

    Two ingredients, both real:
      * 60 multi-cell TARGET games (each spanning all four phases, 8 rows, but a
        <=2-row global budget) -- so the cells contend for every game; and
      * 240 GAP-CRIPPLED games (30 per role per phase) whose profile claims 2 rows
        but which can only ever yield 1 -- the optimistic-accounting trap.

    The OLD greedy compared RAW realizable row counts and broke ties toward "the
    split with the larger total remaining need". Because a multi-cell game keeps
    `realizable("tuning") == 2` for as long as ANY of its cells still wants a tuning
    row, and tuning's need (160) dwarfs frozen_check's (80), essentially every game
    went to tuning: measured on this pool, `frozen_check` is pinned at ~42 games no
    matter how large the pool grows (tuning 275, frozen 137 here vs 42 before). 42
    games x 2 rows = 84 for an 80-row demand -- and once a few of them are
    gap-crippled, `frozen_check` cannot reach its quota and the sampler raised a
    `final-manifest shortfall`. A valid assignment plainly exists (the pool has a
    large healthy surplus), so that was a FALSE infeasibility.

    Scoring by FILL FRACTION instead makes a game that closes 2 of frozen's
    remaining 80 outrank one that closes 2 of tuning's remaining 160, so the two
    candidate pools grow in the splits' own 160:80 ratio and frozen keeps real slack.
    """
    rows, gi = [], 0
    for _ in range(60):                                   # multi-cell TARGET games
        rows.extend(_v2_multicell_game(gi, "target"))
        gi += 1
    for phase in _V2_ALL_PHASE_CELLS:                     # the gap-crippled trap
        for role in ("target", "control"):
            for _ in range(30):
                rows.extend(_v2_tight_game(gi, role, phase))
                gi += 1
    for band in LATE_TARGET_FLOORS:                       # the late floors' supply
        for _ in range(20):
            rows.extend(_v2_game(gi, "target", "late", band))
            gi += 1
    for phase in ("opening", "early_mid", "midgame"):     # healthy CONTROL supply
        for _ in range(12):
            rows.extend(_v2_game(gi, "control", phase, "b400_plus"))
            gi += 1
    for band in ("b400_plus", "b300_399", "b200_299"):
        for _ in range(12):
            rows.extend(_v2_game(gi, "control", "late", band))
            gi += 1
    return rows


# Two RED plies per cell -- even (red) and >= MIN_PLY_GAP apart, each inside its
# phase's real ply_bucket_of range and its band's honest n_legal >= 528 - ply floor
# (b300_399 needs ply >= 129, b200_299 needs ply >= 229).
_V2_ALL_RED_PLIES = {
    ("opening", "b400_plus"): (2, 14),        # ply_bucket_of: 1-15
    ("early_mid", "b400_plus"): (16, 28),     # 16-40
    ("midgame", "b400_plus"): (42, 56),       # 41-90
    ("late", "b400_plus"): (92, 104),         # 91+
    ("late", "b300_399"): (130, 142),         # ...and ply >= 129
    ("late", "b200_299"): (230, 242),         # ...and ply >= 229
}


def _pool_all_red():
    """A pool holding NO black row anywhere: every game contributes TWO RED rows
    (even plies, >= MIN_PLY_GAP apart) to one cell. Composition is comfortably
    satisfiable and both late floors are reachable, so the sampler fills all 240
    rows and clears every floor -- and then MUST refuse, because every split is 100%
    red. GENUINELY side-infeasible: no split assignment, no draw order and no number
    of retries can conjure a black row that does not exist, so the raise is CORRECT.
    """
    rows, gi = [], 0
    for cell in CELL_ORDER_V2:
        for band, n_games in _V2_POOL_SPEC[cell]:
            for _ in range(n_games):
                p1, p2 = _V2_ALL_RED_PLIES[(cell[1], band)]
                rows.append(_v2_row(gi, cell[0], cell[1], band, "red", p1))
                rows.append(_v2_row(gi, cell[0], cell[1], band, "red", p2))
                gi += 1
    return rows


def _pool_side_sorted_by_game_idx():
    """The GAME-ORDER regression pool. (target, midgame) is supplied by THREE
    blocks, deliberately ordered so that ascending game_idx is the WORST possible
    draw order:

      gi   0- 29  30 RED-only games   (red@42 + red@56)      <- naive order hits
      gi  30- 59  30 BLACK-only games (black@43 + black@57)      these first
      gi  60-139  80 side-OPPOSED-pair games (red@42 + black@55)

    The first two blocks hold NO gap-valid side-opposed pair, so
    `_choose_positions_v2` cannot rescue them: every take from one MUST move the
    balance by +-2. The third block is side-neutral and, on its own, can fill the
    cell's whole 45-row demand in EITHER split.

    So a side-balanced 240-row manifest plainly exists -- just draw the opposed
    block. But the naive `sorted(cand_games)` walk never looks at side: it drains
    the RED block first, filling tuning's 30-row quota with 30 REDS
    (|red - black| = 30), and the exact-or-raise side check fires. A FALSE
    INFEASIBILITY, purely from the draw order.

    The opposed block is what makes this test isolate the ORDERING and nothing
    else: because each of its games is side-neutral by itself, ANY subset of them
    that `assign_split_v2` hands a split can still fill that split balanced. (That
    matters -- `assign_split_v2` is side-BLIND, so a cell supplied ONLY by
    same-side games can be dealt a side-degenerate candidate set for one split,
    which no cell-fill order can repair. See the report's residual analysis.)

    Physically honest: midgame is ply 41-90, red on EVEN plies (42, 56), black on
    ODD (43, 55, 57); every pair is >= MIN_PLY_GAP apart; all are necessarily
    b400_plus at ply <= 90.
    """
    rows = _v2_pool({("target", "midgame"): []}, start_gi=200)
    gi = 0
    for _ in range(30):                       # RED-only games FIRST (low game_idx)
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 42))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 56))
        gi += 1
    for _ in range(30):                       # BLACK-only games NEXT
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 43))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 57))
        gi += 1
    for _ in range(80):                       # SIDE-OPPOSED games LAST (high gi)
        rows.extend(_v2_game(gi, "target", "midgame", "b400_plus"))
        gi += 1
    return rows


def _pool_same_side_earliest_chain():
    """(target, midgame) is supplied by 50 games whose THREE rows are red@42,
    red@56, black@71 -- the EARLIEST >= MIN_PLY_GAP-apart chain (42 -> 56) is
    SAME-SIDE, while a gap-valid side-OPPOSED pair (42 + 71, 29 apart) also
    exists. v1's `_choose_positions` take_n >= 2 path walks that earliest chain
    with no side steering, so it takes two REDS from every such game; preferring
    the opposed pair takes the SAME TWO rows' worth (no shortfall) side-neutrally.

    Unlike `_pool_same_side_late_floor`, this pool is comfortably satisfiable, so
    the fixed sampler must EMIT a balanced 240-row manifest here -- not raise.
    (Physically honest: midgame is ply 41-90, so red@42/red@56 are even, black@71
    is odd, and all three are necessarily b400_plus at ply <= 90.)
    """
    rows = _v2_pool({("target", "midgame"): []})
    gi = 30_000
    for _ in range(50):
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 42))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 56))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 71))
        gi += 1
    return rows


_V2_DUP_SHA1 = "v2-dup-shared-red-42"


def _pool_with_duplicate_hash_v2():
    """Games 0 and 1 (both (target, midgame)) each carry THREE rows sharing ONE
    canonical_sha1 on a gap-valid red@42. Whichever game the round-robin draws
    first claims the shared hash; the other's red@42 is otherwise gap-valid AND
    earliest, so it MUST be dropped by the used_sha1 filter. Each game still
    yields MAX_PER_GAME from its two unique rows, so the dedup fires without
    starving the cell (both games stay live in the output)."""
    rows = [r for r in _abundant_pool_v2() if r["game_idx"] not in (0, 1)]
    for gi in (0, 1):
        rows.append({"game_idx": gi, "role": "target", "phase": "midgame",
                     "band": "b400_plus", "side": "red", "ply": 42,
                     "canonical_sha1": _V2_DUP_SHA1})
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 55))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 68))
    return rows


# --- frozen v2 sampler surface ---------------------------------------------

def test_v2_cell_order_and_late_target_cell():
    """CELL_ORDER_V2 is the frozen (role, phase) cell order (SPLIT_ALLOC_V2
    insertion order, mirroring v1's CELL_ORDER), and LATE_TARGET_CELL is the ONE
    allocation cell the floors constrain -- pinned so a PHASES/role rename cannot
    silently orphan the floors."""
    assert CELL_ORDER_V2 == list(SPLIT_ALLOC_V2.keys())
    assert len(CELL_ORDER_V2) == 8
    assert set(CELL_ORDER_V2) == {(role, phase)
                                  for role in ("target", "control")
                                  for phase in PHASES}
    assert LATE_TARGET_CELL == ("target", "late")
    assert LATE_TARGET_CELL in SPLIT_ALLOC_V2
    assert SPLITS == ("tuning", "frozen_check")
    # The floors are a COMBINED requirement over that cell's 45 rows (30 tuning
    # + 15 frozen_check), and 12 + 12 = 24 <= 45 -- satisfiable in principle.
    assert sum(SPLIT_ALLOC_V2[LATE_TARGET_CELL].values()) == 45
    assert sum(LATE_TARGET_FLOORS.values()) <= 45


def test_sample_v2_rows_has_no_bucket_cap_parameter():
    """v2 DROPS v1's <=50% ply-bucket cap (design Sec 1.2: subsumed -- each phase
    is exactly 60/240 = 25%). It must not survive as a vestigial knob."""
    sig = inspect.signature(sample_v2_rows)
    assert list(sig.parameters) == ["kept", "seed"]
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any("bucket" in name for name in sig.parameters)


# --- EXACT composition (the crux) ------------------------------------------

def test_v2_exact_split_composition_and_totals():
    rows, _stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n        # every cell EXACTLY full
    assert len(rows) == CORPUS_SIZE_V2 == 240
    assert sum(1 for r in rows if r["split"] == "tuning") == 160
    assert sum(1 for r in rows if r["split"] == "frozen_check") == 80
    assert sum(1 for r in rows if r["role"] == "target") == 180
    assert sum(1 for r in rows if r["role"] == "control") == 60
    for phase in PHASES:
        assert sum(1 for r in rows if r["phase"] == phase) == 60
    assert len({r["canonical_sha1"] for r in rows}) == len(rows)     # no dup hash
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL


def test_v2_every_row_carries_a_valid_split():
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert all(r["split"] in SPLITS for r in rows)


def test_v2_phase_quota_subsumes_the_v1_bucket_cap():
    """Each phase is exactly 60/240 = 25% of the corpus, so v1's <=50%
    ply-bucket cap is structurally unreachable -- verified against the REAL
    ply_bucket_of of each selected row's PLY (the fixtures' plies are physically
    honest), which simultaneously pins phase == ply_bucket_of(ply)."""
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    buckets = Counter(ply_bucket_of(r["ply"]) for r in rows)
    assert set(buckets) == set(PHASES)
    for phase in PHASES:
        assert buckets[phase] == 60 == 0.25 * CORPUS_SIZE_V2
    assert max(buckets.values()) <= 0.5 * len(rows)        # v1's cap, now free
    for r in rows:
        assert r["phase"] == ply_bucket_of(r["ply"])


# --- the late coverage floors (hard) ---------------------------------------

def test_v2_late_target_floors_are_met():
    """Among the 45 late TARGET rows -- the 30 tuning + 15 frozen_check rows
    COMBINED, not per split (design Sec 1.3) -- >= 12 b300_399 AND >= 12
    b200_299.

    Discriminating by construction: the pool's 80 b400_plus late-target rows all
    sit at LOWER game_idx than any floor-band game and alone exceed the whole
    45-row quota, so an earliest-game fill WITHOUT the floor-satisfaction pass
    takes 45 b400_plus rows and misses both floors.
    """
    pool = _abundant_pool_v2()
    late_target_pool = [r for r in pool if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    b400_pool = [r for r in late_target_pool if r["band"] == "b400_plus"]
    floor_pool = [r for r in late_target_pool if r["band"] in LATE_TARGET_FLOORS]
    assert len(b400_pool) == 80 > 45          # b400_plus alone could fill the cell
    assert (max(r["game_idx"] for r in b400_pool)
            < min(r["game_idx"] for r in floor_pool))     # ...and it is visited FIRST

    rows, stats = sample_v2_rows(pool, seed=1)
    late_target = [r for r in rows if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    assert len(late_target) == 45
    band_counts = Counter(r["band"] for r in late_target)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor                  # >= 12 / >= 12
    assert stats["late_target_band_count"] == dict(sorted(band_counts.items()))

    # The floors are a COMBINED requirement: frozen_check holds only 15 late
    # target rows, so a per-SPLIT reading of ">= 12 and >= 12" (24 rows) would be
    # arithmetically impossible -- the combined reading is the only coherent one.
    frozen_late = [r for r in late_target if r["split"] == "frozen_check"]
    assert len(frozen_late) == 15 < sum(LATE_TARGET_FLOORS.values())

    # Late CONTROL rows in the floor bands ARE selected -- yet they must NOT be
    # credited to the floors (which count late TARGET rows only).
    control_floor_rows = [r for r in rows
                          if (r["role"], r["phase"]) == ("control", "late")
                          and r["band"] in LATE_TARGET_FLOORS]
    assert control_floor_rows


def test_v2_late_floors_hold_across_seeds():
    """The floors (and the exact composition) are a property of the sampler, not
    of one lucky seed / one lucky whole-game split assignment."""
    for seed in (1, 2, 3, 7, 20260712):
        rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=seed)
        assert len(rows) == 240
        cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
        for (role, phase), alloc in SPLIT_ALLOC_V2.items():
            for split, n in alloc.items():
                assert cell[(role, phase, split)] == n
        band_counts = Counter(r["band"] for r in rows
                              if (r["role"], r["phase"]) == LATE_TARGET_CELL)
        for band, floor in LATE_TARGET_FLOORS.items():
            assert band_counts[band] >= floor


def test_v2_unmeetable_late_floor_raises():
    """A pool that MEETS every phase quota but cannot reach a late floor (only 10
    b300_399 target rows exist, floor 12) must RAISE -- the floor is a hard
    requirement, never a best-effort. The failure must be the FLOOR one, not a
    composition shortfall."""
    pool = _pool_late_floor_unmeetable()
    supply = Counter(r["band"] for r in pool
                     if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    assert supply["b300_399"] == 10 < LATE_TARGET_FLOORS["b300_399"]   # unmeetable
    assert sum(supply.values()) == 170 > 45          # ...yet the cell is fillable
    with pytest.raises(ValueError, match="late-target coverage floor unmet"):
        sample_v2_rows(pool, seed=1)


# --- per-game / per-split invariants ---------------------------------------

def test_v2_whole_game_split_isolation():
    rows, _ = sample_v2_rows(_pool_with_multicell_game_v2(), seed=1)
    game_splits = defaultdict(set)
    for r in rows:
        game_splits[r["game_idx"]].add(r["split"])
    assert game_splits
    assert all(len(splits) == 1 for splits in game_splits.values())


def test_v2_at_most_two_rows_per_game_globally():
    """v2's <=MAX_PER_GAME rule is GLOBAL -- across ALL cells and both splits --
    where v1 applied it PER CELL. Game 0 offers 2 mutually gap-valid rows in EACH
    of two cells, so a per-cell cap would take FOUR."""
    pool = _pool_with_multicell_game_v2()
    g0 = sorted((r for r in pool if r["game_idx"] == 0), key=lambda r: r["ply"])
    assert len(g0) == 4                                         # fixture honesty:
    assert len({(r["role"], r["phase"]) for r in g0}) == 2      # two cells,
    plies = [r["ply"] for r in g0]
    assert all(hi - lo >= MIN_PLY_GAP for lo, hi in zip(plies, plies[1:]))
    assert len({r["canonical_sha1"] for r in g0}) == 4          # no filter can bind

    rows, _ = sample_v2_rows(pool, seed=1)
    assert len(rows) == 240                       # no shortfall from the cap
    counts = Counter(r["game_idx"] for r in rows)
    assert counts and max(counts.values()) <= MAX_PER_GAME
    assert counts[0] == MAX_PER_GAME              # game 0 IS selected -- capped at 2
    assert {(r["role"], r["phase"]) for r in rows if r["game_idx"] == 0} == {
        ("target", "opening")}                    # ...all from its first cell


def test_v2_global_two_per_game_capacity_is_enforced():
    """100 games x 16 rows: every PER-CELL capacity is ample, but under the GLOBAL
    <=2/game rule the pool can yield at most 200 < 240 rows. A per-cell-only
    accounting would happily proceed; the global one must refuse."""
    pool = _pool_global_two_per_game_starved()
    per_cell = Counter((r["role"], r["phase"]) for r in pool)
    for cell, alloc in SPLIT_ALLOC_V2.items():
        assert per_cell[cell] == 200 >= alloc["tuning"] + alloc["frozen_check"]
    with pytest.raises(ValueError, match="global capacity"):
        sample_v2_rows(pool, seed=1)


def test_v2_min_ply_gap_within_game_holds_across_cells():
    """>= MIN_PLY_GAP between ANY two rows selected from one game -- including
    rows drawn from DIFFERENT cells (v1 only enforced it within a cell)."""
    rows, _ = sample_v2_rows(_pool_v2_gap_probe(), seed=1)
    assert len(rows) == 240
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game_idx"]].append(r["ply"])
    for plies in by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP
    # game 0's two rows come from two DIFFERENT cells -- so the cross-cell gap
    # filter was genuinely exercised -- and red@16 (8 plies from red@8) is skipped.
    assert len({(r["role"], r["phase"]) for r in rows if r["game_idx"] == 0}) == 2
    assert sorted(by_game[0]) == [8, 28]
    # game 1: the sub-gap 46 is skipped and the <=2 cap stops before 71.
    assert sorted(by_game[1]) == [42, 55]


def test_v2_per_split_side_balance():
    """The HAPPY path only. `_abundant_pool_v2` gives every cell exactly one
    side-opposed pair per game, so it is STRUCTURALLY INCAPABLE of producing a
    same-side 2-take -- it cannot express the real side-balance regression at all.
    The two tests below carry that load."""
    rows, stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
        assert stats["side_count"][split] == dict(sc)      # stats is a real witness


def test_v2_same_side_two_take_prefers_a_side_opposed_pair():
    """A 2-take whose EARLIEST >=gap chain is SAME-SIDE must instead take the
    gap-valid side-OPPOSED pair -- the same 2 rows' worth, so no shortfall, but
    side-neutral.

    v1's `_choose_positions` take_n >= 2 path walks the earliest chain with NO
    side steering (only its take_n == 1 branch steers), which was safe in v1 only
    because a v1 (role, band) cell could receive at most ONE side-opposed pair
    from a game. v2 breaks that premise, so the unfixed sampler takes red@42 +
    red@56 from all 50 of this pool's (target, midgame) games and emits a
    240-row manifest with tuning at red 95 / black 65 -- |diff| 30, a silent
    SIDE_TOL (2) violation.
    """
    pool = _pool_same_side_earliest_chain()
    g = sorted((r for r in pool if r["game_idx"] == 30_000), key=lambda r: r["ply"])
    assert [(r["ply"], r["side"]) for r in g] == [        # fixture honesty:
        (42, "red"), (56, "red"), (71, "black")]
    assert 56 - 42 >= MIN_PLY_GAP                         # the earliest chain is
    assert g[0]["side"] == g[1]["side"] == "red"          # ...gap-valid AND same-side
    assert 71 - 42 >= MIN_PLY_GAP                         # ...yet an OPPOSED pair
    assert len({r["canonical_sha1"] for r in g}) == 3     # ...is also gap-valid, and
    assert len({(r["role"], r["phase"]) for r in g}) == 1  # nothing else can bind

    rows, stats = sample_v2_rows(pool, seed=1)

    # No shortfall: preferring the opposed pair still yields 2 rows per game.
    assert len(rows) == CORPUS_SIZE_V2 == 240
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n
    for split in SPLITS:                                   # ...and it is BALANCED
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
        assert stats["side_count"][split] == dict(sc)

    # The discriminator: every same-side game that gave 2 rows gave one red + one
    # black -- the opposed pair (42, 71) -- and red@56, the unfixed sampler's
    # second pick, is NEVER selected.
    same_side_rows = [r for r in rows if r["game_idx"] >= 30_000]
    assert same_side_rows
    assert 56 not in {r["ply"] for r in same_side_rows}
    by_game = defaultdict(list)
    for r in same_side_rows:
        by_game[r["game_idx"]].append(r)
    assert any(len(v) == MAX_PER_GAME for v in by_game.values())   # 2-takes happened
    for picked in by_game.values():
        assert len(picked) <= MAX_PER_GAME
        if len(picked) == MAX_PER_GAME:
            assert {r["side"] for r in picked} == {"red", "black"}
            assert sorted(r["ply"] for r in picked) == [42, 71]


def test_v2_cell_fill_game_order_is_side_aware():
    """The cell fill must choose WHICH GAME to draw from by side, not just by
    game_idx. Where every candidate game forces a same-side 2-take, the naive
    `sorted(cand_games)` walk drains the RED block first and skews tuning by 30 --
    a FALSE INFEASIBILITY, since alternating red/black games fills the identical
    quota from the identical pool, side-balanced. Steering at the game-selection
    level finds that selection; the row-level steering inside a single game cannot.
    """
    pool = _pool_side_sorted_by_game_idx()
    mid = [r for r in pool if (r["role"], r["phase"]) == ("target", "midgame")]
    by_game = defaultdict(list)
    for r in mid:
        by_game[r["game_idx"]].append(r)
    sides = {g: {r["side"] for r in v} for g, v in by_game.items()}
    reds = sorted(g for g, s in sides.items() if s == {"red"})
    blacks = sorted(g for g, s in sides.items() if s == {"black"})
    opposed = sorted(g for g, s in sides.items() if s == {"red", "black"})
    assert len(reds) == len(blacks) == 30       # fixture honesty: 30 + 30 same-side
    assert len(opposed) == 80                   # ...and an ample side-NEUTRAL block
    assert max(reds) < min(blacks) < min(opposed)      # but ascending game_idx hits
    for v in by_game.values():                         # every RED one FIRST.
        assert len(v) == MAX_PER_GAME                  # Each game is a real 2-take
        plies = sorted(r["ply"] for r in v)            # (both rows gap-valid), so a
        assert plies[1] - plies[0] >= MIN_PLY_GAP      # same-side game moves the
                                                       # balance by a full +-2.
    rows, stats = sample_v2_rows(pool, seed=1)

    assert len(rows) == CORPUS_SIZE_V2 == 240              # no shortfall
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n         # exact composition
    band_counts = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor                  # floors still met
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL    # ...and BALANCED
        assert stats["side_count"][split] == dict(sc)

    # The discriminator: the fill reached PAST the two same-side blocks sitting at
    # the LOWEST game_idx and drew the side-neutral block instead -- the one thing a
    # `sorted(cand_games)` walk can never do, because it never looks at side.
    # Pre-fix it takes 15 RED-only games for tuning: 30 RED rows out of the red
    # block, ZERO opposed, |red - black| = 30.
    mid_sel = [r for r in rows if (r["role"], r["phase"]) == ("target", "midgame")]
    assert len(mid_sel) == 45
    from_reds = [r for r in mid_sel if r["game_idx"] in set(reds)]
    from_opposed = [r for r in mid_sel if r["game_idx"] in set(opposed)]
    assert len(from_opposed) >= 35        # the side-NEUTRAL block did the work (0!)
    assert len(from_reds) <= 10           # the low-game_idx RED trap: avoided (30!)
    msc = Counter(r["side"] for r in mid_sel)      # and the trap cell itself came
    assert abs(msc["red"] - msc["black"]) <= 5     # out near-balanced (30/0 pre-fix)


def test_v2_side_aware_order_is_deterministic():
    """Same input + same seed => byte-identical rows AND stats. The new draw order
    must be a TOTAL, reproducible sort (no set iteration, no dict-order reliance)."""
    for pool_fn in (_pool_side_sorted_by_game_idx, _pool_same_side_earliest_chain,
                    _abundant_pool_v2):
        a, sa = sample_v2_rows(pool_fn(), seed=3)
        b, sb = sample_v2_rows(pool_fn(), seed=3)
        assert a == b
        assert sa == sb


def test_v2_same_side_only_supply_raises_rather_than_skewing():
    """When a cell can ONLY be filled with same-side rows, the sampler must RAISE
    -- never silently emit a side-skewed manifest. Side balance is a hard
    constraint under the SAME exact-or-raise contract as the floors.

    Here the LATE_TARGET_FLOORS themselves force the skew: every b300_399 /
    b200_299 TARGET row in the pool is RED (their opposed blacks were classified
    `control` -- `raw_policy_role` classifies each row independently), and one
    game offers the single (target, late) cell both of them, in two different
    bands. The floor pass draws each band via its own band-restricted take_n == 1
    call, so NO 2-take rule can pair them: the 24 floor rows are all red.

    The unfixed sampler SUCCEEDS here -- 240 rows, exact composition, floors met
    -- and hands back tuning at red 92 / black 68 (|diff| 24 vs SIDE_TOL 2). The
    raise below is therefore reached only AFTER the composition round-robin and
    the floor verification have both PASSED, which is what makes it specifically a
    side-balance failure and not a shortfall in disguise. (Task 6's
    `post_screen_qualification` is the pre-registered place to reject such a screen
    earlier, before the sampler is ever reached.)
    """
    pool = _pool_same_side_late_floor()
    late_target = [r for r in pool if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    floor_rows = [r for r in late_target if r["band"] in LATE_TARGET_FLOORS]
    assert {r["side"] for r in floor_rows} == {"red"}     # fixture honesty: the ONLY
    assert len(floor_rows) == 40                          # floor-band supply is RED,
    for band, floor in LATE_TARGET_FLOORS.items():        # ...ample for the floors,
        assert sum(1 for r in floor_rows if r["band"] == band) == 20 > floor
    assert {r["side"] for r in late_target                # ...while b400_plus (which
            if r["band"] == "b400_plus"} == {"red", "black"}   # cannot reach a floor)
    by_game = defaultdict(list)                                # is side-opposed.
    for r in floor_rows:
        by_game[r["game_idx"]].append(r)
    two_band_games = [g for g, v in by_game.items()
                      if {r["band"] for r in v} == set(LATE_TARGET_FLOORS)]
    assert len(two_band_games) == 20      # ONE game, ONE cell, TWO same-side bands
    probe = sorted(by_game[two_band_games[0]], key=lambda r: r["ply"])
    assert [(r["ply"], r["side"], r["band"]) for r in probe] == [
        (130, "red", "b300_399"), (230, "red", "b200_299")]
    assert 230 - 130 >= MIN_PLY_GAP       # no gap filter binds -- only the side rule

    with pytest.raises(ValueError, match="per-split side balance violated"):
        sample_v2_rows(pool, seed=1)


def test_v2_same_side_late_floor_pool_is_never_skewed_across_seeds():
    """The CONTRACT, on the reviewer's original falsifying pool: under every seed
    the sampler either RAISES or returns a manifest that is exactly composed,
    floor-satisfying AND side-balanced. It never emits a skewed one.

    Both branches are legitimate and both occur here, which is the point. This pool
    is genuinely HARD -- its whole floor-band target supply is red -- but it is not
    infeasible: the retry over whole-game split assignments plus the fill's
    deficit-side correction does find a perfectly balanced selection under some
    seeds (e.g. seed 3, on the 3rd ordering: tuning 80/80, frozen_check 40/40).
    Under others, 8 orderings are not enough and it conservatively refuses. Pinning
    "always raises" would therefore be pinning the GREEDY's luck, not the contract.

    The unconditional exact-or-raise pin lives in
    `test_v2_all_red_pool_always_raises_on_side_balance`, on a pool that is
    GENUINELY side-infeasible -- no retry can ever rescue it.
    """
    for seed in (1, 2, 3, 7, 20260712):
        try:
            rows, stats = sample_v2_rows(_pool_same_side_late_floor(), seed=seed)
        except ValueError as exc:                       # conservative refusal: fine
            assert "side balance violated" in str(exc)
            continue
        # ...but if it DID return a manifest, that manifest must be fully valid.
        assert len(rows) == CORPUS_SIZE_V2 == 240
        cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
        for (role, phase), alloc in SPLIT_ALLOC_V2.items():
            for split, n in alloc.items():
                assert cell[(role, phase, split)] == n
        band_counts = Counter(r["band"] for r in rows
                              if (r["role"], r["phase"]) == LATE_TARGET_CELL)
        for band, floor in LATE_TARGET_FLOORS.items():
            assert band_counts[band] >= floor
        for split in SPLITS:
            sc = Counter(r["side"] for r in rows if r["split"] == split)
            assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
            assert stats["side_count"][split] == dict(sc)


def test_v2_assignment_feeds_the_scarce_split_and_avoids_a_false_shortfall():
    """The whole-game split assignment must be CAPACITY-AWARE. `frozen_check` is the
    scarce split -- 80 rows, and every game it uses costs a whole <=2-row budget --
    so a greedy that compares RAW remaining need over-feeds `tuning` (160) and pins
    frozen at a starved sliver of games. With multi-cell games contending for the
    global budget and some games gap-crippled, that sliver cannot reach frozen's
    quota and the sampler raised a `final-manifest shortfall` on a pool that plainly
    has a valid assignment -- a FALSE infeasibility.
    """
    pool = _pool_frozen_starving_multicell()
    by_game = defaultdict(list)
    for r in pool:
        by_game[r["game_idx"]].append(r)

    # Fixture honesty: multi-cell games really do span 4 cells with 8 rows...
    multi = [g for g, v in by_game.items()
             if len({(r["role"], r["phase"]) for r in v}) == 4]
    assert len(multi) == 60
    assert all(len(by_game[g]) == 8 for g in multi)
    # ...and the gap-crippled games really are crippled: their profile claims 2 rows
    # for the cell, but the two are < MIN_PLY_GAP apart, so the fill can pick ONE.
    tight = [g for g, v in by_game.items()
             if len(v) == 2 and abs(v[0]["ply"] - v[1]["ply"]) < MIN_PLY_GAP]
    assert len(tight) == 240
    for g in tight:
        v = by_game[g]
        assert {r["side"] for r in v} == {"red", "black"}      # a real opposed pair
        assert len({(r["role"], r["phase"]) for r in v}) == 1  # ...in ONE cell

    rows, stats = sample_v2_rows(pool, seed=1)

    # No false shortfall: a complete, exactly-composed, floor-satisfying, balanced
    # manifest (pre-fix: `final-manifest shortfall` in a frozen_check cell).
    assert len(rows) == CORPUS_SIZE_V2 == 240
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n
    band_counts = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL

    # The MECHANISM: the candidate pools now grow in the splits' own 160:80 ratio,
    # so frozen_check keeps real slack. Pre-fix it was pinned near 42 games however
    # big the pool grew -- 84 rows for an 80-row demand, and any gap-crippled game
    # in that sliver broke it.
    per_split = stats["n_games_per_split"]
    assert sum(per_split.values()) == stats["n_games_total"] == len(by_game)
    ratio = per_split["tuning"] / per_split["frozen_check"]
    want = SPLIT_TOTALS["tuning"] / SPLIT_TOTALS["frozen_check"]      # 160/80 == 2
    assert want == 2
    assert abs(ratio - want) < 0.5                    # ~2:1, NOT a starved sliver
    assert per_split["frozen_check"] > 100            # pre-fix: ~42


def test_v2_all_red_pool_always_raises_on_side_balance():
    """The UNCONDITIONAL exact-or-raise pin. `_pool_all_red` holds no BLACK row at
    all, so every split is 100% red and NO assignment, ordering or fill can ever
    balance it -- the raise is CORRECT, and no number of retries may paper over it.

    It must fail on SIDE BALANCE specifically, not on composition or a floor: the
    pool fills every cell exactly and clears both late floors, so reaching the side
    check proves the manifest was otherwise perfectly valid and was refused solely
    for being skewed. That is the guarantee the whole check exists for.
    """
    pool = _pool_all_red()
    assert {r["side"] for r in pool} == {"red"}          # fixture honesty: NO black
    for seed in (1, 2, 3, 7, 20260712):
        with pytest.raises(ValueError, match="per-split side balance violated"):
            sample_v2_rows(pool, seed=seed)


def test_v2_no_duplicate_hash():
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))


def test_v2_duplicate_hash_is_excluded():
    rows, _ = sample_v2_rows(_pool_with_duplicate_hash_v2(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))            # used_sha1 filter left no dup
    assert shas.count(_V2_DUP_SHA1) == 1          # the collided hash survives once
    # both colliding games stay live, so the exclusion was FORCED (the loser lost
    # its red@42 to the filter, not by going unselected).
    assert {0, 1} <= {r["game_idx"] for r in rows}


def test_v2_duplicate_hash_within_one_game_is_excluded():
    """The dedup must hold for ANY input, not only for a (hash-deduped) real
    screen: game 0 carries the SAME canonical_sha1 on TWO gap-valid rows of its
    OWN, which `_choose_positions` would otherwise return in a single batch --
    the batch is screened against `used_sha1` BEFORE any of it is claimed.

    Deliberately NOT physically honest (two positions 26 plies apart in one game
    have different peg counts, so they can never share a canonical hash, and the
    screen stage's collision filter would drop such a row anyway) -- it exists
    solely to prove the per-row re-check fires, mirroring this file's other
    intentionally-unrealistic guard fixture."""
    rows_pool = [r for r in _abundant_pool_v2() if r["game_idx"] != 0]
    same = "v2-same-hash-in-one-game"
    for ply, side in ((42, "red"), (68, "red")):        # 26 plies apart: gap-valid
        rows_pool.append({"game_idx": 0, "role": "target", "phase": "midgame",
                          "band": "b400_plus", "side": side, "ply": ply,
                          "canonical_sha1": same})
    rows_pool.append(_v2_row(0, "target", "midgame", "b400_plus", "black", 55))

    rows, _ = sample_v2_rows(rows_pool, seed=1)
    assert len(rows) == 240
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))            # no dup ANYWHERE in the output
    assert shas.count(same) == 1                  # claimed exactly once
    g0 = [r["ply"] for r in rows if r["game_idx"] == 0]
    assert len(g0) <= MAX_PER_GAME
    assert 42 in g0 and 68 not in g0              # the 2nd copy of the hash dropped


# --- determinism ------------------------------------------------------------

def test_v2_determinism_same_seed():
    a, sa = sample_v2_rows(_abundant_pool_v2(), seed=1)
    b, sb = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert a == b
    assert sa == sb


def test_v2_assign_split_v2_is_whole_game_and_deterministic():
    pool = _abundant_pool_v2()
    profile = {}
    for r in pool:
        profile.setdefault(r["game_idx"], Counter())[(r["role"], r["phase"])] += 1
    a = assign_split_v2(profile, seed=1)
    b = assign_split_v2(profile, seed=1)
    assert a == b                                  # deterministic under seed
    assert set(a) == set(profile)                  # every game assigned
    assert all(v in SPLITS for v in a.values())


# --- shortfalls are ERRORS, never silent truncation -------------------------

def test_v2_insufficient_pool_raises():
    with pytest.raises(ValueError, match="capacity"):
        sample_v2_rows(_insufficient_pool_v2(), seed=1)


def test_v2_final_manifest_shortfall_from_pick_filters_raises():
    """Distinct from the capacity precheck: every capacity check passes, but the
    gap-skip per-pick filter starves (control, opening), so the round-robin hits
    its exact-or-raise guard instead of returning a short manifest."""
    with pytest.raises(ValueError, match="final-manifest shortfall"):
        sample_v2_rows(_pool_gap_starved_cell_v2(), seed=1)


# --- stats are an INDEPENDENT witness --------------------------------------

def test_v2_stats_shape_and_independent_counts():
    """cell_counts / late_target_band_count must be counted FROM THE SELECTED
    ROWS (an independent composition + floor witness), not re-emitted from the
    frozen quotas. v1's `bucket_count` is GONE (no bucket cap in v2)."""
    rows, stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert set(stats) == {"n_rows", "seed", "assignment_attempt", "cell_counts",
                          "side_count", "late_target_band_count",
                          "n_games_per_split", "n_games_total"}
    # PROVENANCE: which of the ASSIGN_ATTEMPTS whole-game split orderings won.
    assert stats["assignment_attempt"] in range(ASSIGN_ATTEMPTS)
    assert "bucket_count" not in stats
    assert stats["n_rows"] == len(rows) == 240
    assert stats["seed"] == 1

    actual = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, quota in alloc.items():
            key = f"{role}|{phase}|{split}"
            assert stats["cell_counts"][key] == actual[(role, phase, split)]
            assert stats["cell_counts"][key] == quota
    assert sum(stats["cell_counts"].values()) == len(rows) == 240

    late_actual = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    assert stats["late_target_band_count"] == dict(sorted(late_actual.items()))
    assert sum(stats["late_target_band_count"].values()) == 45
    for band, floor in LATE_TARGET_FLOORS.items():
        assert stats["late_target_band_count"][band] >= floor

    assert (stats["n_games_per_split"]["tuning"]
            + stats["n_games_per_split"]["frozen_check"]) == stats["n_games_total"]
    assert stats["n_games_total"] == len({r["game_idx"] for r in _abundant_pool_v2()})
