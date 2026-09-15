"""H3 FULL STUDY — the paired score, the Hoeffding interval, the sensitivities.

Binds card §2, §3 and §4 as amended twice. Nothing here plays anything.
"""
import math

import pytest

from scripts.GPU.alphazero import h3_study_analysis as A
from scripts.GPU.alphazero import h3_study_rules as R

HEX = "a" * 64


def _game(pair, colour, *, winner=None, reason="win", digest=None, stratum=None,
          seed=None, elapsed=40.0, plies=51, opening=None):
    """One record in the shape §6.2 requires — seed, stratum and opening_digest
    included, because the pair identity must be computable from a single row."""
    return {
        "task_id": f"h3study-p{pair:03d}-{colour}",
        "pair_id": pair,
        "segment": R.segment_of(pair),
        "stratum": stratum or R.STRATUM_UNIFORM,
        "incumbent_colour": colour,
        "seed": 777000000 + pair * 2 + (colour == "black"),
        "opening_digest": opening or f"{pair:064d}",
        "transcript_digest": digest or f"{pair:063d}{0 if colour == 'red' else 1}",
        "terminal_reason": reason,
        "winner": winner,
        "plies": plies,
        "elapsed_s": elapsed,
    }


def _pair(pair, red_wins, black_wins, **kw):
    """`red_wins` / `black_wins`: did the INCUMBENT win that arm?"""
    return [
        _game(pair, "red", winner="red" if red_wins else "black", **kw),
        _game(pair, "black", winner="black" if black_wins else "red", **kw),
    ]


def _run(pairs):
    games = []
    for g in pairs:
        games.extend(g)
    return A.summarise(games, total_elapsed_s=float(len(games)) * 40.0)


# ───────────────────────── the paired score ────────────────────────────────

def test_the_pair_score_is_incumbent_points_over_two():
    assert A.pair_score(2.0) == 1.0
    assert A.pair_score(1.0) == 0.5
    assert A.pair_score(0.0) == 0.0
    for bad in (-0.5, 2.5):
        with pytest.raises(A.H3StudyAnalysisError, match="points"):
            A.pair_score(bad)


def test_a_WIN_WIN_pair_scores_1_and_a_LOSS_LOSS_scores_0():
    rep = _run([_pair(0, True, True), _pair(1, False, False)])
    scores = {p["pair_id"]: p["score"] for p in rep["pairs"]}
    assert scores == {0: 1.0, 1: 0.0}
    assert rep["primary"]["mean"] == 0.5


def test_a_SPLIT_pair_scores_one_half():
    rep = _run([_pair(0, True, False)])
    assert rep["pairs"][0]["score"] == 0.5


def test_a_CAP_scores_one_half_FOR_THAT_GAME_and_stays_in():
    """Excluding caps would select on play. §4.5."""
    games = _pair(0, True, True)
    games[1] = _game(0, "black", winner=None, reason="cap")
    rep = A.summarise(games, total_elapsed_s=80.0)
    assert rep["pairs"][0]["score"] == 0.75, "1 + 0.5, over two"
    assert rep["capped_games"] == 1 and rep["pairs_with_a_cap"] == 1
    assert rep["primary"]["n"] == 1, "the pair is IN the primary"


# ───────────────────────── the interval ────────────────────────────────────

def test_the_interval_is_HOEFFDING_at_the_realised_n():
    rep = _run([_pair(i, True, True) for i in range(10)])
    h = R.half_width(10)
    assert rep["primary"]["half_width"] == pytest.approx(h)
    assert rep["primary"]["interval"] == [pytest.approx(1.0 - h),
                                          pytest.approx(1.0 + h)]
    assert rep["primary"]["n"] == 10


def test_the_interval_is_declared_NOMINAL_under_a_MODEL():
    """🔴 Amendment 1: distinct seeds and openings do not PROVE independence."""
    rep = _run([_pair(i, True, True) for i in range(4)])
    note = rep["primary"]["interval_note"].lower()
    assert "nominal" in note and "model" in note
    assert "proof" not in note and "proves" not in note


@pytest.mark.parametrize("wins,expected", [
    (10, "incumbent"),      # every pair to the incumbent -> decisive above
    (0, "t1j"),             # every pair to T1j           -> decisive below
])
def test_a_DECISIVE_interval_names_a_direction(wins, expected):
    pairs = [_pair(i, i < wins, i < wins) for i in range(10)]
    rep = _run(pairs)
    assert rep["primary"]["decisive"] is True
    assert rep["primary"]["favours"] == expected


def test_a_STRADDLING_interval_is_INCONCLUSIVE_and_NOT_no_difference():
    # at the floor, so the FLOOR does not pre-empt the interval's own verdict
    n = R.REPORT_FLOOR_PAIRS
    rep = _run([_pair(i, i % 2 == 0, i % 2 == 1) for i in range(n)])
    assert rep["primary"]["mean"] == 0.5
    assert rep["primary"]["decisive"] is False
    assert rep["primary"]["favours"] is None
    assert rep["verdict"] == A.INCONCLUSIVE
    assert "no difference" not in rep["verdict_note"].lower()


def test_the_null_is_PARITY_at_one_half():
    assert A.PARITY == 0.5


# ───────────────────────── identity: the opening is IN it ──────────────────

def test_GAME_AND_PAIR_IDENTITY_INCLUDE_THE_OPENING():
    """🔴 Amendment 1 §4.1. H2's transcript EXCLUDES the opening because H2
    compared repetitions within ONE fixed opening. Here the opening is the
    VARIABLE."""
    a = _game(0, "red", winner="red", digest=HEX, opening="1" * 64)
    b = _game(1, "red", winner="red", digest=HEX, opening="2" * 64)
    assert A.game_identity(a) != A.game_identity(b), (
        "same continuation, different openings -- different games")
    assert A.game_identity(a)[0] == "1" * 64


def test_two_pairs_from_DIFFERENT_openings_are_NEVER_duplicates():
    """Same continuations, different openings: distinct observations, and the
    counts must say so."""
    pairs = [_pair(0, True, True, digest=HEX, opening="1" * 64),
             _pair(1, True, True, digest=HEX, opening="2" * 64)]
    rep = _run(pairs)
    assert rep["duplicate_pairs"] == 0
    assert rep["primary"]["n"] == 2, "neither pair was collapsed"
    assert rep["shared_continuation_pairs"] == 2
    assert rep["shared_continuation_relations"] >= 1


def test_SHARED_CONTINUATIONS_GATE_NOTHING():
    """🔴 Amendment 2: the `> 7` ceiling is REMOVED. Nothing preregistered says
    identical continuations invalidate evidence."""
    assert not hasattr(R, "MAX_SHARED_CONTINUATION_PAIRS")
    pairs = [_pair(i, True, True, digest=HEX, opening=f"{i:064d}")
             for i in range(20)]
    rep = _run(pairs)
    assert rep["shared_continuation_pairs"] == 20
    assert all(g["rule"] != "shared_continuations" for g in rep["gates"])
    assert not any("continuation" in g["what"] for g in rep["gates"])


def test_a_DUPLICATE_PAIR_is_a_HARNESS_FAULT_at_threshold_zero():
    """With `opening_digest` in the identity the openings are distinct by
    construction, so this cannot happen without a fault."""
    g = _pair(0, True, True)
    rep = A.summarise(g + [dict(x, task_id=x["task_id"] + "-again") for x in g],
                      total_elapsed_s=160.0)
    assert rep["duplicate_pairs"] == 1
    gate = next(x for x in rep["gates"] if x["rule"] == "duplicate_pairs")
    assert gate["ceiling"] == 0 and gate["status"] == A.FIRED
    assert "harness" in gate["what"].lower()
    assert rep["interpretation_withheld"] is True


# ───────────────────────── the sensitivities ───────────────────────────────

def test_THREE_SENSITIVITIES_ARE_ALWAYS_REPORTED():
    rep = _run([_pair(i, True, True) for i in range(R.REPORT_FLOOR_PAIRS)])
    assert set(rep["sensitivities"]) == {"cap_free", "identity_free", "both"}
    for s in rep["sensitivities"].values():
        assert {"n", "mean", "half_width", "interval", "decisive",
                "favours", "computable"} <= set(s)


def test_a_verdict_needs_the_primary_AND_every_sensitivity_to_agree():
    rep = _run([_pair(i, True, True) for i in range(R.REPORT_FLOOR_PAIRS)])
    assert rep["primary"]["decisive"] is True
    assert rep["verdict"] == A.INCUMBENT_STRONGER
    assert rep["is_strength_verdict"] is True


def test_a_sensitivity_DECISIVE_THE_OTHER_WAY_suppresses_the_verdict():
    """The most dangerous case: each interval looks conclusive on its own.

    🔑 TESTED AT THE RULE, not end to end, and the reason is worth recording: to
    make `cap_free` decisive the OTHER way, enough pairs must contain a cap that
    the `capped_games` gate (> 118) fires first and withholds interpretation
    anyway. The design defends this twice over; the rule still has to be right on
    its own, because the gate's ceiling could move and this must not depend on it.
    """
    out = A.evaluate_verdict(
        primary={"decisive": True, "favours": "incumbent", "mean": 0.62,
                 "computable": True},
        sensitivities={"cap_free": {"computable": True, "mean": 0.30,
                                    "favours": "t1j", "decisive": True}})
    assert out["verdict"] == A.NO_VERDICT
    assert "opposite" in out["why"].lower()


def test_a_sensitivity_whose_ESTIMATE_merely_crosses_parity_also_suppresses():
    """It need not be DECISIVE the other way; landing on the far side of parity
    is already a disagreement about direction."""
    out = A.evaluate_verdict(
        primary={"decisive": True, "favours": "incumbent", "mean": 0.62,
                 "computable": True},
        sensitivities={"identity_free": {"computable": True, "mean": 0.49,
                                         "favours": None, "decisive": False}})
    assert out["verdict"] == A.NO_VERDICT
    assert "other side" in out["why"].lower()


def test_the_cap_gate_fires_BEFORE_a_cap_driven_disagreement_can_form():
    """The end-to-end half of the test above: a cap rate able to flip the
    sensitivity trips the gate, so the study withholds either way."""
    pairs = []
    for i in range(236):
        g = _pair(i, True, True)
        g[0] = dict(g[0], terminal_reason="cap", winner=None)
        pairs.append(g)
    pairs += [_pair(236 + i, False, False) for i in range(60)]
    rep = _run(pairs)
    assert rep["capped_games"] == 236 > R.MAX_CAPPED_GAMES
    assert rep["interpretation_withheld"] is True
    assert rep["verdict"] == A.NO_VERDICT
    assert rep["is_strength_verdict"] is False


def test_a_sensitivity_estimate_of_EXACTLY_ONE_HALF_suppresses_the_verdict():
    """🔴 Amendment 2. It does not CROSS 0.5, so a rule phrased as crossing would
    have let it through — and an estimate sitting on the null supports neither
    direction."""
    pairs = [_pair(i, True, True) for i in range(R.REPORT_FLOOR_PAIRS)]
    ident = []
    for i in range(R.REPORT_FLOOR_PAIRS, R.REPORT_FLOOR_PAIRS + 4):
        d = f"{i:064d}"
        ident.append([_game(i, "red", winner="red", digest=d),
                      _game(i, "black", winner="red", digest=d)])  # identical pair
    rep = _run(pairs + ident)
    assert rep["primary"]["decisive"] is True
    forced = A.evaluate_verdict(
        primary={"decisive": True, "favours": "incumbent", "mean": 0.9,
                 "computable": True},
        sensitivities={"cap_free": {"computable": True, "mean": 0.5,
                                    "favours": None, "decisive": False}})
    assert forced["verdict"] == A.NO_VERDICT
    assert "0.5" in forced["why"] or "parity" in forced["why"].lower()


def test_an_UNCOMPUTABLE_sensitivity_suppresses_the_verdict():
    """🔴 Amendment 2: an absent check is not a passed check."""
    out = A.evaluate_verdict(
        primary={"decisive": True, "favours": "incumbent", "mean": 0.9,
                 "computable": True},
        sensitivities={"cap_free": {"computable": False, "mean": None,
                                    "favours": None, "decisive": False}})
    assert out["verdict"] == A.NO_VERDICT
    assert "comput" in out["why"].lower()


def test_a_sensitivity_that_merely_WIDENS_does_not_suppress():
    """Losing n widens the interval; that is not disagreement, and penalising it
    would fail the study for arithmetic rather than for evidence."""
    out = A.evaluate_verdict(
        primary={"decisive": True, "favours": "incumbent", "mean": 0.9,
                 "computable": True},
        sensitivities={"cap_free": {"computable": True, "mean": 0.9,
                                    "favours": None, "decisive": False}})
    assert out["verdict"] == A.INCUMBENT_STRONGER


def test_an_UNCOMPUTABLE_PRIMARY_yields_NO_VERDICT():
    out = A.evaluate_verdict(
        primary={"decisive": False, "favours": None, "mean": None,
                 "computable": False},
        sensitivities={})
    assert out["verdict"] == A.NO_VERDICT


# ───────────────────────── gates and the floor ─────────────────────────────

def test_the_floor_withholds_INTERPRETATION_but_never_the_counts():
    rep = _run([_pair(i, True, True) for i in range(5)])
    assert rep["primary"]["n"] == 5
    assert rep["interpretation_withheld"] is True
    assert rep["verdict"] == A.NO_VERDICT
    assert rep["games_scored"] == 10, "the counts are still reported"


def test_the_gates_are_the_cards_ceilings():
    rep = _run([_pair(i, True, True) for i in range(R.REPORT_FLOOR_PAIRS)])
    by = {g["rule"]: g for g in rep["gates"]}
    assert by["duplicate_pairs"]["ceiling"] == 0
    assert by["within_pair_identical"]["ceiling"] == R.MAX_WITHIN_PAIR_IDENTICAL == 7
    assert by["capped_games"]["ceiling"] == R.MAX_CAPPED_GAMES == 118
    assert rep["interpretation_withheld"] is False


def test_a_fired_gate_withholds_interpretation_and_keeps_the_counts():
    pairs = [_pair(i, True, True) for i in range(R.REPORT_FLOOR_PAIRS)]
    for i in range(8):
        d = f"{i:064d}"
        pairs[i] = [_game(i, "red", winner="red", digest=d),
                    _game(i, "black", winner="red", digest=d)]
    rep = _run(pairs)
    assert rep["within_pair_identical"] == 8 > R.MAX_WITHIN_PAIR_IDENTICAL
    assert rep["interpretation_withheld"] is True
    assert rep["verdict"] == A.NO_VERDICT
    assert rep["primary"]["n"] == R.REPORT_FLOOR_PAIRS


# ───────────────────────── refusals ────────────────────────────────────────

def test_a_duplicated_task_id_is_REFUSED_as_a_harness_fault():
    g = _pair(0, True, True)
    with pytest.raises(A.H3StudyAnalysisError, match="duplicate task_id"):
        A.summarise(g + [g[0]], total_elapsed_s=120.0)


def test_a_record_WITHOUT_A_SEED_is_REFUSED():
    """🔴 §6.2 exists so accounting is READ, not derived. A record without its
    seed puts the study back where the pilot was."""
    g = _pair(0, True, True)
    g[0] = {k: v for k, v in g[0].items() if k != "seed"}
    with pytest.raises(A.H3StudyAnalysisError, match="seed"):
        A.summarise(g, total_elapsed_s=80.0)


def test_a_record_without_STRATUM_or_OPENING_DIGEST_is_REFUSED():
    for field in ("stratum", "opening_digest"):
        g = _pair(0, True, True)
        g[0] = {k: v for k, v in g[0].items() if k != field}
        with pytest.raises(A.H3StudyAnalysisError, match=field):
            A.summarise(g, total_elapsed_s=80.0)


def test_a_NaN_or_INF_duration_is_REFUSED():
    for bad in (float("nan"), float("inf")):
        g = _pair(0, True, True)
        g[0] = dict(g[0], elapsed_s=bad)
        with pytest.raises(A.H3StudyAnalysisError, match="elapsed"):
            A.summarise(g, total_elapsed_s=80.0)


def test_an_UNPAIRED_game_excludes_its_pair_WHOLE_and_counts_it():
    rep = A.summarise(_pair(0, True, True) + [_game(1, "red", winner="red")],
                      total_elapsed_s=120.0)
    assert rep["primary"]["n"] == 1
    assert rep["pairs_excluded_incomplete"] == 1


def test_a_VOID_game_excludes_its_pair_WHOLE():
    g = _pair(0, True, True)
    g[1] = _game(0, "black", winner=None, reason="void")
    rep = A.summarise(g, total_elapsed_s=80.0)
    assert rep["primary"]["n"] == 0
    assert rep["pairs_excluded_void"] == 1


def test_the_stratum_split_is_DESCRIPTIVE_and_never_a_verdict():
    pairs = [_pair(i, True, True, stratum=R.STRATUM_UNIFORM) for i in range(6)]
    pairs += [_pair(6 + i, False, False, stratum=R.STRATUM_CO_PRODUCED)
              for i in range(6)]
    rep = _run(pairs)
    assert set(rep["by_stratum"]) == set(R.STRATA)
    assert rep["by_stratum"][R.STRATUM_UNIFORM]["mean"] == 1.0
    assert rep["by_stratum"][R.STRATUM_CO_PRODUCED]["mean"] == 0.0
    for s in rep["by_stratum"].values():
        assert "interval" not in s and "decisive" not in s, (
            "a stratum figure is descriptive; an interval would invite a verdict")
