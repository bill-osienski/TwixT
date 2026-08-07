"""Agent-mode aggregation, and explicit rejection by the legacy path."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, build_agent_pairing_tasks, build_pairing_tasks, make_result,
)
from scripts.GPU.alphazero.eval_summary import (
    summarize_agent_match, summarize_match,
)

CKPT = "checkpoints/x/model_iter_0001.safetensors"
CONTROL = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
CANDIDATE = AgentSpec("candidate", CKPT,
                      R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB))


def _results(winners):
    """Build agent results. build_agent_pairing_tasks puts CANDIDATE red on
    even game_idx, so a caller controls colour by index."""
    tasks = build_agent_pairing_tasks("p", CANDIDATE, CONTROL, len(winners), 100)
    return [make_result(t, w, "win" if w else "state_cap", 50)
            for t, w in zip(tasks, winners)]


def test_agent_summary_scores_by_agent_not_checkpoint():
    # 4 games, candidate wins all 4 regardless of colour.
    # even game_idx: red=candidate -> "red"; odd: red=control -> "black"
    res = _results(["red", "black", "red", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_wins"] == 4
    assert s["b_wins"] == 0
    assert s["a_score_rate"] == pytest.approx(1.0)


def test_agent_summary_emits_real_confidence_intervals():
    res = _results(["red", "red", "black", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["score_rate_ci95"] is not None
    assert s["elo_ci95"] is not None
    assert s["comparison_unit"] == "agent"
    assert s["same_checkpoint"] is True


def test_agent_summary_reports_per_colour_stats_for_agent_a():
    res = _results(["red", "black", "red", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_as_red"]["games"] == 2
    assert s["a_as_black"]["games"] == 2
    assert s["a_as_red"]["wins"] == 2
    assert s["a_as_black"]["wins"] == 2


def test_per_colour_intervals_exist_because_the_colour_rule_needs_them():
    """Spec 8.1 rejects only when a colour's own 95% UPPER bound is below
    50%. Without a per-colour interval that rule cannot be applied at all."""
    res = _results(["red", "black", None, "red"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    for key in ("a_as_red", "a_as_black"):
        lo, hi = s[key]["score_rate_ci95"]
        assert 0.0 <= lo <= hi <= 1.0


def test_decisive_only_rates_are_reported_as_secondary():
    res = _results(["red", "black", None, None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["decisive_games"] == 2
    assert s["a_decisive_score_rate"] == pytest.approx(1.0)
    # Primary stays draw-inclusive: 2 wins + 2 draws over 4 games.
    assert s["a_score_rate"] == pytest.approx(0.75)


def test_decisive_rate_is_none_not_zero_when_every_game_drew():
    res = _results([None, None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["decisive_games"] == 0
    assert s["a_decisive_score_rate"] is None
    assert s["a_as_red"]["decisive_score_rate"] is None


def test_termination_distribution_is_reported():
    res = _results(["red", None, "black", None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["termination_distribution"] == {"win": 2, "state_cap": 2}
    assert s["state_cap_rate"] == pytest.approx(0.5)


def test_agent_summary_treats_draws_as_half():
    res = _results(["red", None, "red", None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_score"] == pytest.approx(3.0)   # 2 wins + 2 draws * 0.5


def test_agent_summary_records_the_complete_readouts():
    res = _results(["red", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["readout_a"]["mode"] == R.MODE_HOEFFDING_LCB
    assert s["readout_b"]["mode"] == R.MODE_ARGMAX


def test_agent_summary_rejects_an_unknown_agent_id():
    res = _results(["red", "black"])
    with pytest.raises(ValueError, match="does not appear"):
        summarize_agent_match(res, "candidate", "ghost", "p", {})


def test_agent_summary_rejects_legacy_checkpoint_results():
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    res = [make_result(t, "red", "win", 50) for t in tasks]
    with pytest.raises(ValueError, match="not an agent comparison"):
        summarize_agent_match(res, "candidate", "control", "p", {})


def test_agent_summary_rejects_empty_results():
    with pytest.raises(ValueError, match="no results"):
        summarize_agent_match([], "candidate", "control", "p", {})


def test_legacy_summary_explicitly_rejects_agent_results():
    # CONSTRUCTED negative case: checkpoint keying is meaningless here, and
    # silently returning nulls would look like a valid self-match.
    res = _results(["red", "black"])
    with pytest.raises(ValueError, match="comparison_unit"):
        summarize_match(res, CKPT, CKPT, "p", {})


def test_legacy_summary_still_works_for_checkpoint_results():
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    res = [make_result(t, "red", "win", 50) for t in tasks]
    s = summarize_match(res, "a.safetensors", "b.safetensors", "p", {})
    assert s["a_wins"] == 1 and s["b_wins"] == 1


def test_legacy_self_match_still_returns_nulls_for_checkpoint_results():
    # The legacy self-match behaviour must be untouched for legacy artifacts.
    tasks = build_pairing_tasks("p", "a.safetensors", "a.safetensors", 2, 100, 0)
    res = [make_result(t, "red", "win", 50) for t in tasks]
    s = summarize_match(res, "a.safetensors", "a.safetensors", "p", {})
    assert s["self_match"] is True
    assert s["a_score_rate"] is None
    assert s["elo_estimate"] is None
