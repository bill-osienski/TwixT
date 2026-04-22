"""Schema + basic-flow tests for probe suite tooling."""
import json
import subprocess
import tempfile
import os

import pytest


def test_sampler_cli_help():
    """Sampler CLI should respond to --help without error."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_probe_candidates.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--min-source-iter" in result.stdout


@pytest.mark.slow
def test_sampler_produces_candidates_json(tmp_path):
    """Sampler against the current logs/games should produce non-empty candidates.json
    with required fields per candidate."""
    out = tmp_path / "candidates.json"
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_probe_candidates.py",
         "--input", "scripts/GPU/logs/games",
         "--out", str(out),
         "--min-source-iter", "995",
         "--per-category-target", "10"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    for cand in data["candidates"][:5]:
        assert "id" in cand
        assert "category" in cand
        assert "side_to_move" in cand
        assert "move_history" in cand
        assert "source_game" in cand
        assert "source_ply" in cand
        assert "active_size" in cand
        assert cand["active_size"] == 24  # default filter


# --- Schema validation tests (fire only when tests/probes/twixt_probes.json exists
#     AND it's flagged as the formal spec §7 curated gate suite; the bootstrap
#     rule-selected suite has different size / starting-player constraints and
#     is distinguished by meta.not_gate_suite == True. See tests/probes/README.md
#     for the distinction.) ---

TWIXT_PROBES_PATH = "tests/probes/twixt_probes.json"

REQUIRED_FIELDS = {
    "id", "category", "confidence", "side_to_move",
    "expected_value_sign", "active_size", "move_history",
    "source_game", "source_ply",
}
VALID_CATEGORIES = {
    "near_win_red", "near_win_black",
    "blocked_or_trap", "false_positive_connectivity",
    "dense_but_disconnected",
    "central_win", "edge_corner_legitimate", "symmetric_sanity",
}
VALID_CONFIDENCE = {"forced", "strong_advantage"}  # unclear_do_not_use discarded
VALID_SIDES = {"red", "black"}


def _load_probes_or_skip():
    """Load the committed probes file; skip if missing."""
    import pytest
    if not os.path.exists(TWIXT_PROBES_PATH):
        pytest.skip(f"{TWIXT_PROBES_PATH} not yet committed (Phase 0 pending)")
    return json.loads(open(TWIXT_PROBES_PATH).read())


def _load_gate_suite_or_skip():
    """Load the probe file and skip if it's the bootstrap (not the §7 gate suite).

    Used only for tests whose assertions are gate-suite-specific (e.g., probe
    count 50..120). Schema correctness + reconstructability apply to any
    shape-valid probe file, so those tests use _load_probes_or_skip instead.
    """
    import pytest
    data = _load_probes_or_skip()
    meta = data.get("meta") or {}
    if meta.get("not_gate_suite") is True:
        pytest.skip(
            f"{TWIXT_PROBES_PATH} is flagged not_gate_suite=True "
            f"(type={meta.get('type')!r}); this test asserts gate-suite-specific "
            f"constraints only. See tests/probes/README.md."
        )
    return data


def test_probe_suite_file_well_formed():
    """If the formal gate suite is committed, it must parse and have a list of probes."""
    data = _load_gate_suite_or_skip()
    assert isinstance(data, dict)
    assert "probes" in data
    assert isinstance(data["probes"], list)
    assert len(data["probes"]) >= 50  # minimum curated size
    assert len(data["probes"]) <= 120  # sanity upper bound


def test_probe_suite_schema_valid():
    """Every probe has required fields + valid enum values (applies to any
    committed probe file, bootstrap or curated gate suite)."""
    data = _load_probes_or_skip()
    for p in data["probes"]:
        missing = REQUIRED_FIELDS - set(p.keys())
        assert not missing, f"probe {p.get('id')} missing: {missing}"
        assert p["category"] in VALID_CATEGORIES, f"bad category: {p['category']}"
        assert p["confidence"] in VALID_CONFIDENCE, f"bad confidence: {p['confidence']}"
        assert p["side_to_move"] in VALID_SIDES
        assert p["expected_value_sign"] in (-1, 0, 1)
        assert 8 <= p["active_size"] <= 24
        assert isinstance(p["move_history"], list)
        if "mirror_of" in p and p["mirror_of"] is not None:
            assert isinstance(p["mirror_of"], str)  # must be a probe id


def test_probe_suite_reconstruction():
    """Every probe's move_history replays to a valid state matching auxiliary
    metadata. Applies to any committed probe file. Probes may include a
    `starting_player` field to signal black-starts; older probes without it
    default to red-starts.
    """
    data = _load_probes_or_skip()
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    for p in data["probes"]:
        starting_player = p.get("starting_player", "red")
        state = TwixtState(active_size=p["active_size"], to_move=starting_player)
        for move in p["move_history"]:
            r, c = int(move[0]), int(move[1])
            state = state.apply_move((r, c))
        # ply should match len(move_history)
        if "ply" in p:
            assert len(p["move_history"]) == p["ply"], \
                f"probe {p['id']} ply={p['ply']} but move_history has {len(p['move_history'])} moves"
        # side_to_move should match state.to_move
        assert state.to_move == p["side_to_move"], \
            f"probe {p['id']} replay to_move={state.to_move} != declared {p['side_to_move']}"


def test_probe_eval_help():
    """probe_eval CLI responds to --help."""
    result = subprocess.run(
        [".venv/bin/python", "-m", "scripts.GPU.alphazero.probe_eval", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--weights" in result.stdout
    assert "--probes" in result.stdout
    assert "--sims" in result.stdout
    assert "--out" in result.stdout


def test_probe_eval_rejects_missing_weights():
    """Formal runs require --weights; without it, eval exits non-zero."""
    result = subprocess.run(
        [".venv/bin/python", "-m", "scripts.GPU.alphazero.probe_eval",
         "--probes", "tests/probes/twixt_probes.json",
         "--sims", "10",
         "--out", "/tmp/_probe_test.csv"],
        capture_output=True, text=True,
    )
    # Should fail because --weights is required
    assert result.returncode != 0, "probe_eval should reject formal run without --weights"
