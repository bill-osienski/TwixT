"""H3 PILOT — pair-preserving handling, timing, and complete/partial reporting.

Binds the card's §4.4, §5, §5.1 and §6 exactly. The rule that decides every
partial case: **a monotone COUNT may fire; a RATIO or a QUANTILE may not.**

Nothing here plays a game. Every record below is constructed.
"""
import pytest

from scripts.GPU.alphazero import h3_pilot_analysis as A
from scripts.GPU.alphazero import h3_pilot_rules as R


def _g(pair, colour, *, digest, capped=False, points=1.0, elapsed=30.0):
    """One game record, as the runner writes it."""
    return {"task_id": f"h3pilot-x-p{pair:02d}-inc_{colour}", "pair_id": pair,
            "incumbent_colour": colour, "transcript_digest": digest,
            "terminal_reason": "cap" if capped else "win",
            "winner": None if capped else "red",
            "t1j_points": 0.5 if capped else points, "elapsed_s": elapsed,
            "plies": 280 if capped else 50}


def _pair(pair, dred, dblack, **kw):
    return [_g(pair, "red", digest=dred, **kw), _g(pair, "black", digest=dblack, **kw)]


def _n_pairs(n, **kw):
    out = []
    for i in range(n):
        out += _pair(i, f"r{i}", f"b{i}", **kw)
    return out


# ═════════════════ the transcript digest is structural, and carries movers ═══

def test_the_digest_covers_MOVES_AND_MOVERS_not_the_whole_record():
    """H2's §2.2 lesson: hashing the whole record makes `seed`/`task_id` give a
    distinct digest for a cell that played one game many times."""
    moves = [("red", 3, 4), ("black", 5, 6)]
    a = A.transcript_digest(moves)
    assert a == A.transcript_digest(list(moves)), "must not depend on identity"
    assert a != A.transcript_digest([("red", 3, 4), ("black", 5, 7)]), "moves matter"
    assert a != A.transcript_digest([("black", 3, 4), ("red", 5, 6)]), "MOVERS matter"


def test_a_red_slot_game_can_NEVER_share_a_digest_with_a_black_slot_one():
    """The card's §5.1 derivation depends on this: because movers are in the
    digest, the red slot can only collide with a red slot."""
    same_squares = [(3, 4), (5, 6), (7, 8)]
    as_red = A.transcript_digest([("red" if i % 2 == 0 else "black", r, c)
                                  for i, (r, c) in enumerate(same_squares)])
    as_black = A.transcript_digest([("black" if i % 2 == 0 else "red", r, c)
                                    for i, (r, c) in enumerate(same_squares)])
    assert as_red != as_black


# ═════════════════════ the exclusion pipeline, in order ══════════════════════

def test_a_clean_run_keeps_every_pair():
    rep = A.summarise(_n_pairs(20), total_elapsed_s=1200.0)
    assert rep["complete"] is True
    assert (rep["pairs_nominal"], rep["pairs_scored"], rep["pairs_informative"],
            rep["pairs_distinct"]) == (20, 20, 20, 20)
    assert rep["duplicate_pairs"] == rep["within_pair_identical"] == 0


def test_AN_INCOMPLETE_PAIR_IS_EXCLUDED_WHOLE_and_no_game_is_dropped_alone():
    """🔑 The rule the design turns on. A half-pair has no paired score."""
    games = _n_pairs(19) + [_g(19, "red", digest="r19")]      # the black game missing
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["pairs_scored"] == 19, "the half pair is not scored"
    assert rep["incomplete_pairs"] == 1
    assert rep["games_completed"] == 39
    assert 19 not in {p["pair_id"] for p in rep["pairs"]}, "kept as half an observation"


def test_a_WITHIN_PAIR_IDENTICAL_pair_is_excluded_whole_but_does_not_close():
    """S3 fires only above its threshold; one such pair is excluded and counted."""
    games = _n_pairs(19) + _pair(19, "same", "same")
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["within_pair_identical"] == 1
    assert rep["pairs_scored"] == 20 and rep["pairs_informative"] == 19
    assert rep["stop_rules"]["S3"]["status"] == "CLEAR", "1 is not > 2"


def test_DUPLICATE_PAIRS_collapse_to_one_and_keep_BOTH_their_games():
    games = _n_pairs(18) + _pair(18, "dup_r", "dup_b") + _pair(19, "dup_r", "dup_b")
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["duplicate_pairs"] == 1
    assert rep["pairs_informative"] == 20 and rep["pairs_distinct"] == 19
    survivor = [p for p in rep["pairs"] if p["red_digest"] == "dup_r"]
    assert len(survivor) == 1 and survivor[0]["black_digest"] == "dup_b"
    assert survivor[0]["n_games"] == 2, "the survivor kept BOTH games"


def test_the_pipeline_runs_in_the_CARDS_ORDER():
    """nominal -> drop incomplete -> drop within-pair-identical -> collapse dups."""
    games = (_n_pairs(17)
             + _pair(17, "same", "same")                    # within-pair identical
             + _pair(18, "d_r", "d_b") + _pair(19, "d_r", "d_b"))   # duplicates
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert (rep["pairs_scored"], rep["within_pair_identical"],
            rep["pairs_informative"], rep["duplicate_pairs"],
            rep["pairs_distinct"]) == (20, 1, 19, 1, 18)


# ═══════════════════ partial overlap: two counts that differ ═════════════════

def test_partial_overlap_counts_OBSERVATIONS_and_RELATIONSHIPS_separately():
    """The card's own example: A shares a game with B, and A a different one with
    C -> THREE pairs involved, TWO relationships."""
    games = (_n_pairs(17)
             + _pair(17, "shared_r", "b17")       # A
             + _pair(18, "shared_r", "b18")       # B  (shares A's red game)
             + [_g(19, "red", digest="r19"), _g(19, "black", digest="b17")])  # C
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["partial_overlap_pairs"] == 3, "A, B and C are all involved"
    assert rep["partial_overlap_relations"] == 2, "A~B and A~C"


def test_overlap_is_computed_AFTER_duplicates_collapse():
    """Otherwise a collapsed duplicate masquerades as an overlap."""
    games = _n_pairs(18) + _pair(18, "d_r", "d_b") + _pair(19, "d_r", "d_b")
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["duplicate_pairs"] == 1
    assert rep["partial_overlap_pairs"] == 0 and rep["partial_overlap_relations"] == 0


# ═══════════════════════════ timing ═════════════════════════════════════════

def test_timing_statistics_come_from_the_recorded_per_game_elapsed():
    games = _n_pairs(20)
    for i, g in enumerate(games):
        g["elapsed_s"] = float(i + 1)
    rep = A.summarise(games, total_elapsed_s=820.0)
    t = rep["timing"]
    assert (t["min"], t["max"], t["n"]) == (1.0, 40.0, 40)
    assert t["median"] == pytest.approx(20.5)
    assert t["p90"] == pytest.approx(36.0, abs=1.0)
    assert t["total_elapsed_s"] == 820.0


def test_a_result_WITHOUT_a_recorded_duration_is_REFUSED():
    """FAIL CLOSED. Attempt 3's records carried no timing at all, which is why
    §4.1 was unanswerable after the fact."""
    games = _n_pairs(20)
    del games[0]["elapsed_s"]
    with pytest.raises(A.H3AnalysisError, match="elapsed"):
        A.summarise(games, total_elapsed_s=1200.0)


def test_a_NEGATIVE_duration_is_REFUSED():
    """A monotonic clock cannot go backwards; a record that says so is not one."""
    games = _n_pairs(20)
    games[0]["elapsed_s"] = -1.0
    with pytest.raises(A.H3AnalysisError, match="elapsed"):
        A.summarise(games, total_elapsed_s=1200.0)


# ═══════════════════════ stop rules, complete run ═══════════════════════════

@pytest.mark.parametrize("rule", ["S1", "S2", "S3", "S4a", "S4b"])
def test_every_rule_is_CLEAR_on_a_clean_complete_run(rule):
    rep = A.summarise(_n_pairs(20), total_elapsed_s=1200.0)
    assert rep["stop_rules"][rule]["status"] == "CLEAR"
    assert rep["any_fired"] is False


def test_S1_fires_above_the_duplicate_threshold():
    games = _n_pairs(16)
    for i in (16, 17, 18, 19):                      # four pairs, all the same
        games += _pair(i, "same_r", "same_b")
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["duplicate_pairs"] == 3 > R.MAX_DUPLICATE_PAIRS
    assert rep["stop_rules"]["S1"]["status"] == "FIRED"
    assert rep["any_fired"] is True


def test_S2_fires_above_the_cap_threshold():
    games = _n_pairs(15) + _n_pairs(5)[:0]
    for i in range(15, 20):                          # 5 pairs = 10 capped games
        games += _pair(i, f"r{i}", f"b{i}", capped=True)
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["capped_games"] == 10 > R.MAX_CAPPED_GAMES
    assert rep["stop_rules"]["S2"]["status"] == "FIRED"


def test_S3_fires_above_the_within_pair_threshold():
    games = _n_pairs(17)
    for i in (17, 18, 19):
        games += _pair(i, f"s{i}", f"s{i}")
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["within_pair_identical"] == 3 > R.MAX_WITHIN_PAIR_IDENTICAL
    assert rep["stop_rules"]["S3"]["status"] == "FIRED"


def test_S4a_fires_on_elapsed_time():
    rep = A.summarise(_n_pairs(20), total_elapsed_s=R.MAX_TOTAL_ELAPSED_S + 1)
    assert rep["stop_rules"]["S4a"]["status"] == "FIRED"


def test_S4b_fires_on_a_HEAVY_TAIL():
    games = _n_pairs(20)
    for g in games:
        g["elapsed_s"] = 10.0
    for g in games[-4:]:
        g["elapsed_s"] = 500.0                      # p90 >> median
    rep = A.summarise(games, total_elapsed_s=1200.0)
    assert rep["stop_rules"]["S4b"]["status"] == "FIRED"


# ═══════════════════ PARTIAL RUNS: counts may fire, ratios may not ══════════

def test_on_a_partial_run_NOTHING_is_ever_declared_CLEAR():
    rep = A.summarise(_n_pairs(12), total_elapsed_s=1000.0)
    assert rep["complete"] is False
    for rule in ("S1", "S2", "S3", "S4a"):
        assert rep["stop_rules"][rule]["status"] == "NOT FIRED -- UNDETERMINED"


def test_S4b_is_UNRESOLVED_on_a_partial_run_even_with_a_heavy_tail():
    """🔑 A quantile is not monotone: the unplayed games could pull p90 down."""
    games = _n_pairs(12)
    for g in games:
        g["elapsed_s"] = 10.0
    for g in games[-4:]:
        g["elapsed_s"] = 500.0
    rep = A.summarise(games, total_elapsed_s=1000.0)
    assert rep["stop_rules"]["S4b"]["status"] == "UNRESOLVED -- SCHEDULE INCOMPLETE"


@pytest.mark.parametrize("rule,builder", [
    ("S1", lambda: _n_pairs(9) + _pair(9, "s", "t") + _pair(10, "s", "t")
                 + _pair(11, "s", "t") + _pair(12, "s", "t")),
    ("S2", lambda: _n_pairs(7) + [g for i in (7, 8, 9, 10, 11)
                                  for g in _pair(i, f"r{i}", f"b{i}", capped=True)]),
    ("S3", lambda: _n_pairs(9) + _pair(9, "a", "a") + _pair(10, "b", "b")
                 + _pair(11, "c", "c")),
])
def test_a_MONOTONE_COUNT_fires_conclusively_on_a_PARTIAL_run(rule, builder):
    """Later games cannot un-observe an event already recorded."""
    rep = A.summarise(builder(), total_elapsed_s=1000.0)
    assert rep["complete"] is False
    assert rep["stop_rules"][rule]["status"] == "FIRED"
    assert rep["any_fired"] is True


def test_S4a_fires_on_a_partial_run_from_elapsed_time_alone():
    rep = A.summarise(_n_pairs(5), total_elapsed_s=R.RUN_DEADLINE_S)
    assert rep["stop_rules"]["S4a"]["status"] == "FIRED"


# ══════════════════════ the floor suppresses INTERPRETATION ═════════════════

def test_below_the_floor_INTERPRETATION_is_withheld():
    rep = A.summarise(_n_pairs(4), total_elapsed_s=1000.0)
    assert rep["interpretation_withheld"] is True
    assert rep["outcome_distribution"] is None, "no reading of outcomes below the floor"


def test_below_the_floor_MONOTONE_RULES_STILL_FIRE():
    """🔴 The correction: the floor previously suppressed conclusive findings.
    Nine completed pairs can already hold three duplicate pairs, and later games
    cannot undo them."""
    games = _n_pairs(5) + _pair(5, "s", "t") + _pair(6, "s", "t") \
        + _pair(7, "s", "t") + _pair(8, "s", "t")
    rep = A.summarise(games, total_elapsed_s=1000.0)
    assert rep["pairs_scored"] == 9 < R.REPORT_FLOOR_PAIRS
    assert rep["interpretation_withheld"] is True
    assert rep["duplicate_pairs"] == 3
    assert rep["stop_rules"]["S1"]["status"] == "FIRED", "a crossed count still fires"


def test_above_the_floor_interpretation_is_allowed():
    rep = A.summarise(_n_pairs(12), total_elapsed_s=1000.0)
    assert rep["interpretation_withheld"] is False
    assert rep["outcome_distribution"] is not None


# ══════════════════════════ no strength verdict, ever ═══════════════════════

def test_THE_REPORT_CARRIES_NO_RATE_NO_INTERVAL_NO_VERDICT():
    """The pilot cannot supply a strength verdict, so the words must not be there
    to be quoted out of it."""
    rep = A.summarise(_n_pairs(20), total_elapsed_s=1200.0)
    # `is_strength_verdict` is the explicit DENIAL and must be present.
    assert rep["is_strength_verdict"] is False

    # 🔑 EXACT names, not substrings. My first version scanned for "score" and
    # tripped on `pairs_scored`, which is how many pairs were SCOREABLE -- a
    # structural count, not a strength quantity. A crude scan that flags honest
    # fields gets relaxed until it flags nothing.
    FORBIDDEN = {"win_rate", "rate", "interval", "hoeffding", "elo", "verdict",
                 "conclusion", "t1j_points", "points", "stronger", "parity",
                 "confidence", "ci", "p_value"}

    def all_keys(o):
        if isinstance(o, dict):
            return set(o) | {k for v in o.values() for k in all_keys(v)}
        if isinstance(o, (list, tuple)):
            return {k for v in o for k in all_keys(v)}
        return set()

    found = {k for k in all_keys(rep) if k != "is_strength_verdict"} & FORBIDDEN
    assert not found, found

    # THE SUBSTANTIVE ONE: the per-game records carry `t1j_points`, and the report
    # must not aggregate them anywhere. A summed points column IS a strength
    # quantity whatever it is called.
    assert "t1j_points" not in repr(rep), "the report aggregates the points"
