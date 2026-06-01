from scripts.GPU.alphazero.eval_runner import EvalGameResult
from scripts.GPU.alphazero.eval_summary import summarize_match, summarize_tournament


def _res(game_idx, red, black, winner, reason, n=50):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return EvalGameResult(game_idx, "A_vs_B", game_idx, red, black,
                          winner, wc, reason, n, rs, bs)


def _match_results():
    # 4 games, balanced colors. A wins both as red; split as black.
    return [
        _res(0, "A", "B", "red", "win"),     # A(red) win
        _res(1, "B", "A", "black", "win"),   # A(black) win
        _res(2, "A", "B", "black", "win"),   # B(black) win -> A loss
        _res(3, "B", "A", None, "state_cap"),  # draw
    ]


def test_summarize_match_counts_and_score():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    assert s["games"] == 4
    assert s["a_wins"] == 2 and s["b_wins"] == 1 and s["state_caps"] == 1
    assert s["a_score"] == 2.5
    assert abs(s["a_score_rate"] - 0.625) < 1e-9
    assert s["verdict"] == "stronger"


def test_summarize_match_by_color_blocks():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    assert s["a_as_red"]["games"] == 2 and s["a_as_red"]["wins"] == 1
    assert s["a_as_black"]["games"] == 2 and s["a_as_black"]["wins"] == 1


def test_summarize_match_color_bias_and_avg_plies():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    # decisive winners by color: red wins in g0; black wins in g1,g2 -> red 1/3
    assert abs(s["color_bias"]["red_win_rate_decisive"] - (1 / 3)) < 1e-9
    assert s["avg_plies"] == 50.0
    assert s["draw_score_policy"] == "state_cap_and_board_full_score_0.5"


def test_summarize_tournament_groups_by_pairing():
    r = _match_results()
    out = summarize_tournament(r, [("A", "B")], config={})
    assert len(out["pairings"]) == 1
    assert out["pairings"][0]["pairing_id"] == "A_vs_B"
    assert len(out["table"]) == 1
    assert out["table"][0]["verdict"] == "stronger"


def _self_match_results():
    # Same checkpoint "X" on both sides; balanced colors by game_idx parity.
    return [
        _res(0, "X", "X", "red", "win"),
        _res(1, "X", "X", "black", "win"),
        _res(2, "X", "X", "red", "win"),
        _res(3, "X", "X", None, "state_cap"),
    ]


def test_summarize_match_self_match_nulls_comparison():
    s = summarize_match(_self_match_results(), "X", "X", "X_vs_X", config={})
    assert s["self_match"] is True
    # per-checkpoint comparison undefined
    for k in ("a_wins", "b_wins", "a_score", "a_score_rate", "elo_estimate",
              "elo_ci95", "score_rate_ci95", "verdict", "a_as_red", "a_as_black"):
        assert s[k] is None, f"{k} should be None for self-match"
    # color balance IS meaningful: red wins g0,g2 / black wins g1 -> 2/3 of decisive
    assert abs(s["color_bias"]["red_win_rate_decisive"] - (2 / 3)) < 1e-9
    assert s["games"] == 4 and s["state_caps"] == 1


def test_summarize_match_normal_sets_self_match_false():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    assert s["self_match"] is False
    assert s["a_score_rate"] is not None and s["verdict"] == "stronger"
