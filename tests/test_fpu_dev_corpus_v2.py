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
"""
from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    CORPUS_SIZE,
    QUOTA_PER_BAND,
    SPLIT_ALLOC,
    band_of,
    ply_bucket_of,
)
from scripts.GPU.alphazero.fpu_dev_corpus_v2 import (
    CORPUS_SIZE as CORPUS_SIZE_V2,
    LATE_TARGET_FLOORS,
    MAX_PER_CELL_PER_GAME,
    MAX_PER_GAME,
    MIN_PLY_GAP,
    PHASES,
    PROPOSAL_CELLS,
    SIDE_TOL,
    SPLIT_ALLOC_V2,
    enumerate_v2_proposals,
    proposal_cell_of,
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
