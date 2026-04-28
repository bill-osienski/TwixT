"""Tests for the strong_advantage probe tier: structural features,
admission filter, ID determinism, category assignment, and the promotion
workflow.

Labeling is mocked: tests inject a stub labeler. The opt-in live smoke
test lives separately in tests/test_strong_advantage_smoke_live.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_state(moves, starting_player="red"):
    """Build a TwixtState by applying the given (row, col) moves in order."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=24, to_move=starting_player)
    for r, c in moves:
        s = s.apply_move((r, c))
    return s


def test_phase1_features_red_chain_top_to_mid_board():
    """Red builds a knight-connected chain from row 0 down through the
    middle. cc_size, cc_axis_span, cc_touches_own_goal must reflect the
    chain.
    """
    from scripts.GPU.alphazero.probe_eval import compute_phase1_features

    # Red knight chain: (0,12) -> (2,11) -> (4,12) -> (6,11) -> (8,12)
    # Black filler so plies alternate; black pegs placed away from red's chain.
    moves = [
        (0, 12), (1, 0),
        (2, 11), (1, 1),
        (4, 12), (1, 2),
        (6, 11), (1, 3),
        (8, 12), (1, 4),
    ]
    state = _make_state(moves)
    feats = compute_phase1_features(state, winner="red")
    assert feats["cc_size"] >= 5
    assert feats["cc_axis_span"] >= 0.30  # spans rows 0..8 of 23
    assert feats["cc_touches_own_goal"] is True  # (0, 12) touches row 0
    assert feats["forced_within_2"] is False
    # axis_span_margin = winner_span - loser_span; loser is black with no chain
    assert feats["axis_span_margin"] >= 0.20
    # centroid around row 4, col ~12; center is (11.5, 11.5) so Chebyshev
    # distance is ~7-8 (row 4 is 7.5 from center)
    assert feats["centroid_chebyshev_from_center"] <= 9


def _make_decisive_game_dict(winner_color, terminal_ply, moves):
    """Build the minimal game-record dict that probe_eval ingests."""
    return {
        "meta": {"iteration": 70},
        "winner": winner_color,
        "winner_reason": "win",
        "moves": [{"row": r, "col": c} for r, c in moves],
        "starting_player": "red",
    }


def test_extract_strong_advantage_candidates_drops_midband():
    """Mid-band centroid (Chebyshev 7-8) candidates are excluded with the
    category_midband audit reason; central and edge candidates survive.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    games = [
        _make_decisive_game_dict("red", 30, _central_red_chain()),
        _make_decisive_game_dict("red", 30, _edge_red_chain()),
        _make_decisive_game_dict("red", 30, _midband_red_chain()),
    ]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    cats = sorted(c["category"] for c in candidates)
    assert "chain_advantage_central_red" in cats
    assert "chain_advantage_edge_red" in cats
    assert all("midband" not in c["category"] for c in candidates)
    midband_drops = [a for a in audit if a["reason"] == "category_midband"]
    assert len(midband_drops) >= 1


def test_extract_strong_advantage_candidates_drops_low_axis_span_margin():
    """A candidate where the loser's chain is as long as the winner's is
    rejected via axis_span_margin < 0.10.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    games = [_make_decisive_game_dict("red", 20, _both_strong_chain())]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    assert candidates == []
    assert any(a["reason"] == "phase1_axis_span_margin" for a in audit)


def _central_red_chain():
    # Red knight chain alternating cols 11/12 across rows 0..22.
    # Centroid ≈ (11.0, 11.5), Chebyshev from (11.5, 11.5) = 0.5 → 0
    # → central (≤6).
    base = [(0, 12), (2, 11), (4, 12), (6, 11), (8, 12), (10, 11), (12, 12),
            (14, 11), (16, 12), (18, 11), (20, 12), (22, 11)]
    return _interleave_with_filler(base, filler_col=22)


def _edge_red_chain():
    # Red knight chain alternating cols 1/2 across rows 0..22 (col 0 is
    # black's goal edge — red can't play there). Centroid ≈ (11.0, 1.5),
    # Chebyshev from (11.5, 11.5) = 10 → edge (≥9).
    base = [(0, 1), (2, 2), (4, 1), (6, 2), (8, 1), (10, 2), (12, 1),
            (14, 2), (16, 1), (18, 2), (20, 1), (22, 2)]
    return _interleave_with_filler(base, filler_col=15)


def _midband_red_chain():
    # Red knight chain alternating cols 3/4 across rows 0..22.
    # Centroid ≈ (11.0, 3.5), Chebyshev from (11.5, 11.5) = 8 → midband
    # (7-8 → dropped at category assignment, audit reason category_midband).
    # Spans rows 0..22 (axis_span ≈ 0.96), cc_size = 12 — passes Phase-1
    # gates so the midband-drop path is what actually rejects it.
    base = [(0, 3), (2, 4), (4, 3), (6, 4), (8, 3), (10, 4), (12, 3),
            (14, 4), (16, 3), (18, 4), (20, 3), (22, 4)]
    return _interleave_with_filler(base, filler_col=18)


def _both_strong_chain():
    # Red knight chain cols 20/21 rows 0..22 (12 pegs, span ≈ 0.96).
    # Black knight chain rows 12..21, cols 1..19 (10 pegs, col span ≈ 0.78).
    # Chains are placed in non-overlapping column regions so no bridge
    # crossing occurs. At the k=3 sample point red has 10 pegs (cc_size ≥ 10,
    # span ≥ 0.55) but margin ≈ 0.087 < 0.10 → fails phase1_axis_span_margin.
    # Remaining sample points fail phase1_cc_size, so candidates == [].
    red = [(0, 20), (2, 21), (4, 20), (6, 21), (8, 20), (10, 21),
           (12, 20), (14, 21), (16, 20), (18, 21), (20, 20), (22, 21)]
    black = [(12, 1), (13, 3), (14, 5), (15, 7), (16, 9), (17, 11),
             (18, 13), (19, 15), (20, 17), (21, 19)]
    out = []
    for i in range(max(len(red), len(black))):
        if i < len(red):
            out.append(red[i])
        if i < len(black):
            out.append(black[i])
    return out


def _interleave_with_filler(red_moves, filler_col):
    out = []
    for i, rm in enumerate(red_moves):
        out.append(rm)
        out.append((1 + (i % 22), filler_col))  # black filler in safe column
    return out


def test_extract_strong_advantage_candidates_reads_canonical_schema():
    """Regression: real on-disk game records use meta.reason (not top-level
    winner_reason) and have an `id` field. Function must extract candidates
    from records with that schema, not just from synthetic test fixtures.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    # Mirror the schema scripts/GPU/alphazero/game_saver.py emits:
    # - top-level `id`, `winner`, `moves`, `starting_player`
    # - `meta.reason`, `meta.iteration`, `meta.game_idx`, `meta.board_size`
    # NO top-level `winner_reason`.
    canonical_game = {
        "id": "iter_0070_game_042",
        "winner": "red",
        "starting_player": "red",
        "moves": [{"row": r, "col": c} for r, c in _central_red_chain()],
        "meta": {
            "iteration": 70,
            "game_idx": 42,
            "reason": "win",
            "board_size": 24,
        },
    }
    candidates, audit = extract_strong_advantage_candidates(
        [canonical_game], k_plies_range=(3, 8), category_min_count=0
    )
    assert len(candidates) >= 1, (
        f"Expected at least one candidate from canonical-schema game; got {len(candidates)}. "
        f"Audit: {[(a.get('source_ply'), a.get('reason')) for a in audit][:10]}"
    )
    # source_game must be the explicit `id`, not a fallback-derived placeholder
    assert candidates[0]["source_game"] == "iter_0070_game_042"


def test_extract_strong_advantage_candidates_skips_non_decisive_canonical():
    """Regression: a canonical-schema game with meta.reason != 'win' must
    skip cleanly (zero candidates), not crash.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    draw_game = {
        "id": "iter_0070_game_099",
        "winner": None,
        "starting_player": "red",
        "moves": [{"row": r, "col": c} for r, c in _central_red_chain()],
        "meta": {
            "iteration": 70,
            "game_idx": 99,
            "reason": "draw",
            "board_size": 24,
        },
    }
    candidates, audit = extract_strong_advantage_candidates(
        [draw_game], k_plies_range=(3, 8), category_min_count=0
    )
    assert candidates == []


def test_label_candidate_with_mcts_uses_injected_labeler():
    """The labeler signature must be (state, sims, seed) -> (root_value,
    top1_share). Test that a stub labeler produces the expected aggregate
    (mean_root_value, value_per_run, value_stability, min_top1_share).
    """
    from scripts.GPU.alphazero.probe_eval import label_candidate_with_mcts

    state = _make_state([(0, 12), (1, 0), (2, 11)])

    canned = [(0.6, 0.30), (0.7, 0.25), (0.5, 0.40)]
    calls = []

    def stub_labeler(state, sims, seed):
        calls.append((sims, seed))
        return canned[len(calls) - 1]

    label = label_candidate_with_mcts(
        state, sims=10000, repeats=3,
        rng_seed_base=12345, labeler=stub_labeler,
    )
    assert calls == [(10000, 12345 ^ 0), (10000, 12345 ^ 1), (10000, 12345 ^ 2)]
    assert label["mean_root_value"] == pytest.approx(0.6)
    assert label["value_per_run"] == [0.6, 0.7, 0.5]
    assert label["value_stability"] == pytest.approx(0.2)
    assert label["min_top1_share"] == pytest.approx(0.25)


@pytest.fixture
def passing_candidate():
    """A candidate whose default Phase-2 evaluation passes every clause."""
    return {
        "winner": "red",
        "phase1_features": {
            "cc_size": 14,
            "cc_axis_span": 0.74,
            "cc_touches_own_goal": True,
            "axis_span_margin": 0.20,
            "centroid_chebyshev_from_center": 4,
            "forced_within_2": False,
        },
        "phase2_label": {
            "mean_root_value": 0.62,
            "value_per_run": [0.60, 0.65, 0.61],
            "value_stability": 0.05,
            "min_top1_share": 0.25,
            "label_mcts_sims": 10000,
            "label_mcts_repeats": 3,
            "rng_seed_base": 1,
        },
    }


def test_admission_passes_when_all_clauses_satisfied(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    admitted, reason = apply_admission_filter(
        passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert admitted is True
    assert reason == "admitted"


def test_admission_rejects_sign_mismatch(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = -0.62
    passing_candidate["phase2_label"]["value_per_run"] = [-0.60, -0.65, -0.61]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "sign_mismatch"


def test_admission_rejects_low_magnitude(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = 0.30
    passing_candidate["phase2_label"]["value_per_run"] = [0.28, 0.32, 0.30]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "magnitude_below_threshold"


def test_admission_rejects_low_top1_share(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["min_top1_share"] = 0.10
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "low_top1_share"


def test_admission_rejects_unstable_value(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["value_stability"] = 0.30
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "unstable_value"


def test_admission_rejects_already_forced(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase1_features"]["forced_within_2"] = True
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "position_already_forced"


def test_admission_perspective_contract_black_winner_passes_negative_value(passing_candidate):
    """Documents the red-perspective contract: for a black winner, a
    correctly red-normalized mean_root_value must be negative (red losing
    = black winning) to be admitted.

    If a future caller forgets to negate the labeler's STM-perspective
    output for black-to-move candidates, this contract is violated and
    the filter would silently flag valid candidates as sign_mismatch.
    Catching that drift is the whole point of this test.
    """
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter

    # Reshape the fixture for a black-winner scenario:
    # red-perspective values are negative (red is losing → black wins).
    passing_candidate["winner"] = "black"
    passing_candidate["phase2_label"]["mean_root_value"] = -0.62
    passing_candidate["phase2_label"]["value_per_run"] = [-0.60, -0.65, -0.61]

    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is True, f"Expected admission, got reason={reason!r}"
    assert reason == "admitted"


def test_admission_perspective_contract_black_winner_rejects_positive_value(passing_candidate):
    """Inverse of the above: for a black winner, a positive red-perspective
    mean_root_value (red is winning) must be sign_mismatch — this is the
    cross-check against the source-game outcome.
    """
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter

    passing_candidate["winner"] = "black"
    passing_candidate["phase2_label"]["mean_root_value"] = 0.62  # POS = red winning
    passing_candidate["phase2_label"]["value_per_run"] = [0.60, 0.65, 0.61]

    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "sign_mismatch"


def test_run_strong_advantage_writes_draft_with_admitted_candidates(tmp_path, monkeypatch):
    """End-to-end on the generation path: mock candidate-extraction +
    labeler + checkpoint loader so the test runs without disk I/O. Two
    synthetic candidates feed in: one is admitted (high magnitude), the
    other is rejected (low magnitude). Assert the draft lists only the
    admitted probe.
    """
    import json
    import unittest.mock as _mock
    import scripts.build_probe_suite as bps

    sample_central = {
        "move_history": [(0, 12), (1, 0), (2, 11), (1, 1)],
        "ply": 4, "winner": "red",
        "category": "chain_advantage_central_red",
        "phase1_features": {
            "cc_size": 12, "cc_axis_span": 0.65, "cc_touches_own_goal": True,
            "axis_span_margin": 0.20, "centroid_chebyshev_from_center": 4,
            "forced_within_2": False,
        },
        "source_game": "iter_0070_game_001", "source_ply": 4,
        "starting_player": "red",
    }
    sample_edge = {
        "move_history": [(0, 1), (1, 22), (2, 0), (1, 21)],
        "ply": 4, "winner": "red",
        "category": "chain_advantage_edge_red",
        "phase1_features": {
            "cc_size": 11, "cc_axis_span": 0.60, "cc_touches_own_goal": True,
            "axis_span_margin": 0.15, "centroid_chebyshev_from_center": 10,
            "forced_within_2": False,
        },
        "source_game": "iter_0070_game_002", "source_ply": 4,
        "starting_player": "red",
    }

    def fake_extract(games, **kw):
        return [sample_central, sample_edge], []

    def fake_labeler(state, sims, seed):
        # TwixtState has no peg_at() — use pegs.get((r,c)).
        if state.pegs.get((0, 12)) == "red":
            return (0.65, 0.30)
        return (0.20, 0.30)  # below magnitude_threshold

    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.extract_strong_advantage_candidates",
        fake_extract,
    )
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval._default_mcts_labeler",
        fake_labeler,
    )
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.load_network_for_scoring",
        lambda *_a, **_kw: (_mock.MagicMock(), 30, 128, 6),
    )
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval._set_default_labeler_network",
        lambda _net: None,
    )
    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
    ])
    assert rc == 0
    draft = out_path.with_suffix(".draft.json")
    assert draft.exists(), f"expected draft at {draft}; out_path was {out_path}"

    payload = json.loads(draft.read_text())
    assert payload["meta"]["tier"] == "strong_advantage"
    assert len(payload["probes"]) == 1
    assert payload["probes"][0]["category"] == "chain_advantage_central_red"


def test_promote_errors_with_no_draft(tmp_path):
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "tester",
        "--out", str(out),
    ])
    assert rc != 0


def test_promote_writes_committed_with_reviewer_and_timestamp(tmp_path):
    import json, datetime
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    draft = out.with_suffix(".draft.json")
    draft.write_text(json.dumps({
        "meta": {"tier": "strong_advantage", "review_mode": "draft"},
        "probes": [],
    }))
    rc = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice",
        "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["meta"]["review_mode"] == "light_review"
    assert payload["meta"]["reviewer"] == "alice"
    ts = payload["meta"]["reviewed_at_utc"]
    # Round-trips as ISO 8601 UTC.
    assert ts.endswith("Z")
    datetime.datetime.fromisoformat(ts[:-1])


def test_promote_refuses_overwrite_without_force(tmp_path):
    import json
    import scripts.build_probe_suite as bps
    out = tmp_path / "x.json"
    draft = out.with_suffix(".draft.json")
    draft.write_text(json.dumps({
        "meta": {"tier": "strong_advantage", "review_mode": "draft"},
        "probes": [],
    }))
    out.write_text("{}")  # pre-existing committed file
    rc1 = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice", "--out", str(out),
    ])
    assert rc1 != 0  # refused
    # With --force it succeeds
    rc2 = bps.main_with_args([
        "--tier", "strong_advantage", "--promote",
        "--reviewer", "alice", "--force", "--out", str(out),
    ])
    assert rc2 == 0


COMMITTED_STRONG_SUITE = PROJECT_ROOT / "tests" / "probes" / "strong_advantage_probes.json"


def _load_committed_suite():
    if not COMMITTED_STRONG_SUITE.exists():
        pytest.skip("committed strong_advantage_probes.json not present yet")
    import json
    return json.loads(COMMITTED_STRONG_SUITE.read_text())


def test_committed_meta_block_well_formed():
    suite = _load_committed_suite()
    meta = suite["meta"]
    assert meta["tier"] == "strong_advantage"
    assert meta["not_gate_suite"] is True
    assert meta["review_mode"] == "light_review"
    assert isinstance(meta["reviewer"], str) and meta["reviewer"]
    assert meta["reviewed_at_utc"].endswith("Z")
    sha = meta["selection_rules"]["label_checkpoint_sha256"]
    assert isinstance(sha, str) and len(sha) == 64 and all(
        c in "0123456789abcdef" for c in sha
    )


def test_committed_probes_have_required_fields():
    suite = _load_committed_suite()
    valid_categories = {
        "chain_advantage_central_red", "chain_advantage_central_black",
        "chain_advantage_edge_red", "chain_advantage_edge_black",
    }
    for p in suite["probes"]:
        assert p["confidence"] == "strong_advantage"
        assert p["category"] in valid_categories
        assert p["side_to_move"] in ("red", "black")
        assert p["expected_value_sign"] in (-1, 1)
        assert isinstance(p["move_history"], list)
        # phase1_features: 5 keys
        feats = p["phase1_features"]
        assert set(feats.keys()) == {
            "cc_size", "cc_axis_span", "cc_touches_own_goal",
            "axis_span_margin", "centroid_chebyshev_from_center",
            "forced_within_2",
        }
        # phase2_label: must include the 8 specified keys (additional keys
        # like label_checkpoint added downstream are tolerated).
        label = p["phase2_label"]
        assert set(label.keys()) >= {
            "mean_root_value", "value_per_run", "value_stability",
            "min_top1_share", "label_checkpoint", "label_mcts_sims",
            "label_mcts_repeats", "rng_seed_base",
        }
        assert isinstance(label["rng_seed_base"], int)


def test_extract_strong_advantage_writes_no_admitted_audit_rows_in_phase1():
    """Phase 1 must NOT write admitted audit rows. The Phase-2 audit row
    (written by build_probe_suite.py's _run_strong_advantage loop, then
    superseded by the diversity selector) is the single canonical
    post-labeling record. See spec §7.1.

    This test loads a real committed game file and runs the extractor on
    it. The strong assertion: no audit row carries reason="admitted",
    regardless of how many Phase-1 candidates the game produces. This
    test passes trivially if the game produces zero candidates, but the
    end-to-end integration test in tests/test_strong_advantage_diversity_selector.py
    (added in Task 8) provides the load-bearing check that admitted rows
    appear exactly once per kept probe in the final selector output.
    """
    import json
    from pathlib import Path

    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    project_root = Path(__file__).resolve().parent.parent
    games_dir = project_root / "scripts" / "GPU" / "logs" / "games"

    # Pick any one decisive game with iteration in the committed range.
    games = []
    for fp in sorted(games_dir.glob("iter_0057_game_*.json"))[:5]:
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        meta = g.get("meta") or {}
        if (meta.get("reason") or g.get("winner_reason")) == "win":
            g["source_game"] = fp.stem
            games.append(g)
        if len(games) >= 1:
            break

    assert games, "no decisive game files found in iter_0057 range — fixture missing"

    candidates, audit = extract_strong_advantage_candidates(games)

    assert candidates, (
        "iter_0057 game produced no Phase-1 candidates — "
        "the admitted-row assertion would pass trivially; update the fixture"
    )

    admitted_audit_rows = [r for r in audit if r["reason"] == "admitted"]
    assert admitted_audit_rows == [], (
        f"Phase 1 wrote {len(admitted_audit_rows)} admitted audit row(s); "
        f"after the cleanup, Phase 1 should write only rejection rows. "
        f"Sample: {admitted_audit_rows[:2]}"
    )
