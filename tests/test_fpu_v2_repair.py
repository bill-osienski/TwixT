"""Task 0 of the fpu-v2 role-feasibility repair plan: schema-1 GOLDEN PINS,
captured on the tree BEFORE any repair-plan code change (main @ fca9c0d).

Frozen design ref: docs/superpowers/plans/2026-07-18-fpu-v2-role-feasibility-repair.md
  Task 0 ("schema-1 golden capture FIRST on the unmodified tree").

Later tasks rework `fpu_dev_corpus_v2.py`'s sampler and
`diagnose_fpu_policy_mass.py`'s dev-safety verdict while promising schema-1
(`alloc=None` / no new config authority) stays byte-identical to today's
behaviour. Comparing a NEW `alloc=None` code path against a NEW
`AllocationProfile.legacy()` code path only proves the two new paths agree
with each other -- it says nothing about whether either matches what
actually shipped. These two tests are the independent, pre-repair authority:
their goldens were captured by a throwaway script run against this exact
pre-repair tree and must never be regenerated from a post-repair run.

`_golden_pool()` reuses the existing sampler suite's feasible-pool builder
(`tests/test_fpu_dev_corpus_v2.py::_abundant_pool_v2`) verbatim -- it is
already a deterministic, pure-stdlib fixture (synthetic canonical_sha1
strings, no RNG/time/IO), so the golden test re-derives the identical `kept`
input forever. `_golden_verdict_rows()` / `_golden_ref()` / `_golden_cand()`
copy the `_safe_target()` / `_safe_control()` row shapes
`test_fpu_diagnostic_modes.py`'s `dev_safety_verdict` tests use, combining a
target block and a control block so `verdict.metrics` populates BOTH the
target-side and control-side keys (not just an all-clean/empty subset).
"""
import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.GPU.alphazero import build_teacher_calibration_manifest as btcm
from scripts.GPU.alphazero import diagnose_fpu_policy_mass as diag
from scripts.GPU.alphazero import fpu_dev_corpus_v2 as v2
from scripts.GPU.alphazero import fpu_dev_reservoir_protocol as proto
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.test_fpu_dev_corpus_v2 import _abundant_pool_v2
from tests.test_fpu_dev_reservoir_protocol import (
    _conformant_reservoir, _fake_preflight, _protocol_params,
    _write_precheckable_config, _write_qualifiable_reservoir)

GOLDEN_DIR = Path(__file__).parent / "goldens"


def _golden_pool():
    """Deterministic feasible `kept` pool -- the existing sampler suite's
    builder (700 fabricated screen rows, synthetic hashes), reused verbatim
    so this file never drifts from the suite it was captured against."""
    return _abundant_pool_v2()


def _safe_target(**over):
    """Copied from test_fpu_diagnostic_modes.py's `_safe_target` -- a
    minimal, all-clean `dev_safety_verdict` target row."""
    row = dict(role="target", band="b200_299", new_collapse=False, lock_in=False,
               mover_delta=0.0, eff_children_reduction=0.0, top_share_inc=0.0)
    row.update(over)
    return row


def _safe_control(**over):
    """Copied from test_fpu_diagnostic_modes.py's `_safe_control` -- a
    minimal, all-clean `dev_safety_verdict` control row."""
    row = dict(role="control", mover_delta=0.0, control_flip_to_lower_prior=False)
    row.update(over)
    return row


def _golden_verdict_rows():
    """100 target rows (5% new_collapse, 2 lock-ins, varying mover_delta) +
    30 control rows (10% flip, varying mover_delta) -- exercises every
    `verdict.metrics` key on both the target and control side at once."""
    target = [_safe_target(band="solo", new_collapse=(i < 5), lock_in=(i < 2),
                            mover_delta=0.01 * i, eff_children_reduction=0.2,
                            top_share_inc=0.05) for i in range(100)]
    control = [_safe_control(mover_delta=0.01 * i, control_flip_to_lower_prior=(i < 3))
               for i in range(30)]
    return target + control


def _golden_ref():
    return diag.R0


def _golden_cand():
    """Shared lock-in baseline. `dev_safety_verdict` takes separate
    `r0_lockin`/`absoff_lockin` args (no default for either) -- the existing
    suite's fixtures universally set them equal (`_LOCKIN_BASE` for both), so
    this single helper is passed for both positions."""
    return 5


def test_schema1_sampler_output_matches_pre_repair_golden():
    golden = json.loads(
        (GOLDEN_DIR / "fpu_v2_schema1_sampler_golden.json").read_text())
    rows, stats = v2.sample_v2_rows(_golden_pool(), seed=3)
    assert json.loads(json.dumps(
        {"rows": rows, "stats": stats}, sort_keys=True)) == golden


def test_schema1_verdict_metrics_match_pre_repair_golden():
    golden = json.loads(
        (GOLDEN_DIR / "fpu_v2_schema1_verdict_golden.json").read_text())
    verdict = diag.dev_safety_verdict(_golden_verdict_rows(), _golden_ref(),
                                       _golden_cand(), _golden_cand())
    assert json.loads(json.dumps(verdict.metrics, sort_keys=True)) == golden


# ---------------------------------------------------------------------------
# Task 1: AllocationProfile -- the one validated, schema-2 config-authoritative
# allocation object.
# ---------------------------------------------------------------------------

# The spec's production profile, as the schema-2 JSON fields carry it.
PRODUCTION_PROFILE_RAW = {
    "config_schema_version": 2,
    "run_kind": "production",
    "phase_allocation": {
        "target|late":       {"tuning": 40, "frozen_check": 20},
        "control|opening":   {"tuning": 10, "frozen_check": 5},
        "control|early_mid": {"tuning": 10, "frozen_check": 5},
        "control|midgame":   {"tuning": 10, "frozen_check": 5},
        "control|late":      {"tuning": 10, "frozen_check": 5},
    },
    "late_floors": {"b400_plus": 8, "b300_399": 12, "b200_299": 12},
    "late_target_band_minima": {
        "tuning":       {"b400_plus": 4, "b300_399": 8, "b200_299": 8},
        "frozen_check": {"b400_plus": 4, "b300_399": 5, "b200_299": 5},
    },
    "max_per_game": 2,
    "min_ply_gap": 12,
    "side_tol": 2,
    "corpus_size": 120,
}


def test_parse_production_profile_totals():
    p = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    assert p.corpus_size == 120
    assert p.split_totals == {"tuning": 80, "frozen_check": 40}
    assert p.quota_by_phase == {
        "opening": 15, "early_mid": 15, "midgame": 15, "late": 75}
    assert p.allocation[("target", "late")] == {"tuning": 40, "frozen_check": 20}
    assert p.run_kind == "production"


def test_legacy_profile_mirrors_module_constants():
    p = v2.AllocationProfile.legacy()
    assert p.schema_version == 1
    assert p.allocation == {c: dict(a) for c, a in v2.SPLIT_ALLOC_V2.items()}
    assert p.corpus_size == v2.CORPUS_SIZE == 240
    assert p.band_minima_total == dict(v2.LATE_TARGET_FLOORS)
    assert p.band_minima_per_split == {}
    assert (p.max_per_game, p.min_ply_gap, p.side_tol) == (
        v2.MAX_PER_GAME, v2.MIN_PLY_GAP, v2.SIDE_TOL)


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.__setitem__("config_schema_version", 3), "config_schema_version"),
    (lambda r: r.__setitem__("run_kind", "experiment"), "run_kind"),
    (lambda r: r["phase_allocation"].__setitem__(
        "targetlate", {"tuning": 1, "frozen_check": 1}), "role|phase"),
    (lambda r: r["phase_allocation"].__setitem__(
        "hero|late", {"tuning": 1, "frozen_check": 1}), "role"),
    (lambda r: r["phase_allocation"].__setitem__(
        "target|endgame", {"tuning": 1, "frozen_check": 1}), "phase"),
    (lambda r: r["phase_allocation"]["target|late"].__setitem__("tuning", -1),
     "negative"),
    (lambda r: r["phase_allocation"]["target|late"].__setitem__("tuning", 40.5),
     "integer"),
    (lambda r: r.__setitem__("corpus_size", 121), "corpus_size"),
    (lambda r: r["late_target_band_minima"]["tuning"].__setitem__(
        "b400_plus", 99), "minima"),
    (lambda r: r["late_floors"].__setitem__("b100", 1), "band"),
    # Review correction 5: an incomplete per-split map (e.g. frozen_check
    # silently omitted) must be rejected, not silently accepted.
    (lambda r: r["late_target_band_minima"].pop("frozen_check"), "every split"),
])
def test_parse_rejects_malformed_profiles(mutate, needle):
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    mutate(raw)
    with pytest.raises(ValueError, match="(?i)" + needle):
        v2.parse_allocation_profile(raw, source="test")


@pytest.mark.parametrize("key", [
    "phase_allocation", "late_floors", "late_target_band_minima",
    "max_per_game", "min_ply_gap", "side_tol", "corpus_size",
])
def test_parse_rejects_missing_profile_keys(key):
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    del raw[key]
    with pytest.raises(ValueError, match="missing required"):
        v2.parse_allocation_profile(raw, source="test")


def test_per_split_minima_must_cover_totals():
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    raw["late_target_band_minima"]["tuning"]["b300_399"] = 3   # 3+5=8 < total 12
    with pytest.raises(ValueError, match="b300_399"):
        v2.parse_allocation_profile(raw, source="test")


def test_fingerprint_covers_the_complete_effective_profile():
    p = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    fp = p.fingerprint()
    assert fp["run_kind"] == "production"
    assert fp["allocation"]["target|late"] == {"tuning": 40, "frozen_check": 20}
    assert fp["band_minima_per_split"]["frozen_check"]["b300_399"] == 5
    assert fp["corpus_size"] == 120
    assert fp["max_per_game"] == 2 and fp["min_ply_gap"] == 12 and fp["side_tol"] == 2
    json.dumps(fp, sort_keys=True)   # must be JSON-serializable as-is


# ---------------------------------------------------------------------------
# Task 2: schema-2 config loading + profile_for(config).
# ---------------------------------------------------------------------------

SCHEMA1_CONFIG_KEYS = {
    "source_index_path": "idx.jsonl", "seed_range": [0, 4800],
    "selection_seed": 7, "phase_allocation": {}, "late_floors": {},
    "enumerator_params": {}, "new_collapse_stratum": "ply_bucket",
    "checkpoint": "ck.safetensors", "forbidden_manifests": [],
    "screen_out": "s.csv", "select_out": "m.csv",
    "expected_fingerprints": {}, "config_schema_version": 1,
    "protocol_path": "p.json", "match_summary_path": "ms.json",
    "replay_dir": "replays", "report_out": "r.json",
}


def _write_config(tmp_path, extra):
    cfg = dict(SCHEMA1_CONFIG_KEYS)
    cfg.update(extra)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_schema1_config_loads_and_yields_legacy_profile(tmp_path):
    cfg = v2.load_v2_config(_write_config(tmp_path, {}))
    assert cfg.run_kind is None            # schema-1 carries none of the new keys
    p = v2.profile_for(cfg)
    assert p.schema_version == 1
    assert p.corpus_size == 240


def test_schema2_config_missing_new_keys_is_rejected(tmp_path):
    path = _write_config(tmp_path, {"config_schema_version": 2})
    with pytest.raises(ValueError, match="run_kind"):
        v2.load_v2_config(path)


def test_schema2_config_round_trips_into_a_profile(tmp_path):
    extra = dict(PRODUCTION_PROFILE_RAW)
    extra["post_screen_report_out"] = "psq.json"
    cfg = v2.load_v2_config(_write_config(tmp_path, extra))
    p = v2.profile_for(cfg)
    assert p.schema_version == 2 and p.corpus_size == 120
    assert cfg.post_screen_report_out == "psq.json"


# ---------------------------------------------------------------------------
# Task 3: pure qualification report + the real-failure regression fixture.
# ---------------------------------------------------------------------------

def _kept_row(game_idx, ply, side, phase, band, role, tag=""):
    return {
        "game_idx": game_idx, "ply": ply, "side": side, "phase": phase,
        "band": band, "role": role, "n_legal": 528 - ply,
        "canonical_sha1": f"sha-{game_idx}-{ply}-{side}{tag}",
        "root_value_stm": 0.0, "normalized_entropy": 0.5, "top1_prior": 0.1,
        "top4_mass": 0.4, "top8_mass": 0.6,
    }


def make_gate_fail_fixture():
    """COUNT- AND GEOMETRY-FAITHFUL analogue of the OBSERVED reservoir_v1
    kept pool (review correction 5): 155 late-target rows in 86 games
    (36x1-row + 31x2-row + 19x3-row), realizable exactly 136 under <=2/game;
    bands 12 b400_plus (12 games, ONE row each, 7 black / 5 red) /
    52 b300_399 / 91 b200_299; every ply lies in its band's geometric range
    on a 24 board (n_legal = 528 - ply: b400_plus needs ply <= 128, b300_399
    ply 129-228, b200_299 ply 229-328; late phase needs ply >= 91). Targets
    0/0/0 outside late; controls ample in all four phases."""
    rows, gi = [], 0
    # The real b400 scarcity: 12 single-row games, sides 7 black / 5 red.
    for i in range(12):
        rows.append(_kept_row(gi, 91 + i, "black" if i < 7 else "red",
                              "late", "b400_plus", "target"))
        gi += 1
    # Deal the remaining 143 rows (52 b300 + 91 b200) into 24 one-row +
    # 31 two-row + 19 three-row games (per-game overlap represented).
    bands = ["b300_399"] * 52 + ["b200_299"] * 91
    BAND_BASE_PLY = {"b300_399": 150, "b200_299": 240}
    it = iter(bands)
    for size in [1] * 24 + [2] * 31 + [3] * 19:
        for k in range(size):
            band = next(it)
            rows.append(_kept_row(
                gi, BAND_BASE_PLY[band] + 13 * k,      # >=12-ply spacing
                "red" if (gi + k) % 2 == 0 else "black",
                "late", band, "target"))
        gi += 1
    # Ample controls: 40 games per phase, one red+one black control each
    # (ply in the opening/early_mid/midgame/late ranges; n_legal consistent).
    for phase, base_ply in (("opening", 2), ("early_mid", 20),
                            ("midgame", 50), ("late", 95)):
        for _ in range(40):
            rows.append(_kept_row(gi, base_ply, "red", phase, "b400_plus",
                                  "control"))
            rows.append(_kept_row(gi, base_ply + 12, "black", phase,
                                  "b400_plus", "control"))
            gi += 1
    return rows


def test_gate_fail_fixture_is_faithful_to_the_measured_screen():
    late_targets = [r for r in make_gate_fail_fixture()
                    if (r["role"], r["phase"]) == ("target", "late")]
    assert len(late_targets) == 155
    assert len({r["game_idx"] for r in late_targets}) == 86
    per_game = Counter(r["game_idx"] for r in late_targets)
    assert sum(min(2, n) for n in per_game.values()) == 136
    b400 = [r for r in late_targets if r["band"] == "b400_plus"]
    assert len(b400) == 12 == len({r["game_idx"] for r in b400})
    assert Counter(r["side"] for r in b400) == {"black": 7, "red": 5}
    assert Counter(r["band"] for r in late_targets) == {
        "b400_plus": 12, "b300_399": 52, "b200_299": 91}
    for r in late_targets:                     # geometry: band matches n_legal
        n = 528 - r["ply"]
        assert r["band"] == ("b400_plus" if n >= 400
                             else "b300_399" if n >= 300 else "b200_299")


def test_old_allocation_gate_fails_naming_target_opening():
    report = v2.post_screen_qualification_report(
        make_gate_fail_fixture(), v2.AllocationProfile.legacy())
    assert report["status"] == "GATE_FAIL"
    assert "target|opening" in report["binding_constraint"]
    assert "capacity 0" in report["binding_constraint"]
    assert "demand 45" in report["binding_constraint"]
    assert report["cells"]["target|opening"]["capacity"] == 0
    assert report["cells"]["target|late"]["capacity"] == 136


def test_new_production_profile_passes_capacity_on_the_fixture():
    alloc = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    report = v2.post_screen_qualification_report(
        make_gate_fail_fixture(), alloc)
    assert report["status"] == "PASS"
    assert report["binding_constraint"] is None
    assert report["late_target_bands"]["b400_plus"]["capacity"] == 12
    assert report["late_target_bands"]["b400_plus"]["n_games"] == 12
    assert report["profile"]["run_kind"] == "production"


def test_band_capacity_below_total_minimum_gate_fails():
    rows = [r for r in make_gate_fail_fixture()
            if not (r["role"] == "target" and r["band"] == "b400_plus")]
    alloc = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    report = v2.post_screen_qualification_report(rows, alloc)
    assert report["status"] == "GATE_FAIL"
    assert any("b400_plus" in f for f in report["failures"])


def test_global_capacity_message_names_the_profile_field_for_schema2():
    """Review fix: the global-capacity failure string must name whichever
    field actually holds the bound. For a schema-2 profile that's the
    `max_per_game` instance attribute (can legally differ from the module
    constant), not the module constant's own name `MAX_PER_GAME`."""
    alloc = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    games_profile = {0: {("target", "late"): 1}}  # 1 game -> far under corpus_size=120
    failures = v2._capacity_shortfalls(games_profile, alloc)
    message = " ".join(failures)
    assert "<=max_per_game (" in message
    assert "MAX_PER_GAME" not in message

    legacy_failures = v2._capacity_shortfalls(games_profile, v2.AllocationProfile.legacy())
    legacy_message = " ".join(legacy_failures)
    assert "<=MAX_PER_GAME (" in legacy_message


# ---------------------------------------------------------------------------
# Task 4: alloc threaded through greedy/sampler/qualification/select.
# ---------------------------------------------------------------------------

def make_feasible_120_pool():
    """A pool the PRODUCTION profile can exactly fill: 70 late-target pair
    games (bands 14/28/28) + 40 control pair-games per phase. Pairs are
    side-opposed, >=12 plies apart, and BAND-CONSISTENT with their own plies
    (n_legal = 528 - ply stays inside the pair's band). Deliberately more
    generous than the real screen (2 b400 rows/game) -- real-scarcity
    tightness is exercised by the faithful gate-fail fixture and the Task 14
    run against the actual screen."""
    BAND_PAIR_PLIES = {"b400_plus": (100, 114), "b300_399": (150, 164),
                       "b200_299": (240, 254)}
    rows = []
    gi = 0
    for band in (["b400_plus"] * 14 + ["b300_399"] * 28 + ["b200_299"] * 28):
        p0, p1 = BAND_PAIR_PLIES[band]
        rows.append(_kept_row(gi, p0, "red", "late", band, "target"))
        rows.append(_kept_row(gi, p1, "black", "late", band, "target"))
        gi += 1
    for phase, base_ply in (("opening", 2), ("early_mid", 20),
                            ("midgame", 50), ("late", 95)):
        for _ in range(40):
            rows.append(_kept_row(gi, base_ply, "red", phase, "b400_plus",
                                  "control"))
            rows.append(_kept_row(gi, base_ply + 12, "black", phase,
                                  "b400_plus", "control"))
            gi += 1
    return rows


def _production_alloc():
    return v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")


def test_sampler_fills_the_production_profile_exactly():
    rows, stats = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                                    alloc=_production_alloc())
    assert stats["n_rows"] == 120
    assert stats["cell_counts"]["target|late|tuning"] == 40
    assert stats["cell_counts"]["target|late|frozen_check"] == 20
    assert stats["cell_counts"]["control|opening|tuning"] == 10
    splits = Counter(r["split"] for r in rows)
    assert splits == {"tuning": 80, "frozen_check": 40}
    roles = Counter(r["role"] for r in rows)
    assert roles == {"target": 60, "control": 60}
    # Whole-game split isolation, global <=2/game, >=12-ply gap, no dup hashes.
    split_by_game, plies_by_game = {}, {}
    for r in rows:
        assert split_by_game.setdefault(r["game_idx"], r["split"]) == r["split"]
        plies_by_game.setdefault(r["game_idx"], []).append(r["ply"])
    for plies in plies_by_game.values():
        assert len(plies) <= 2
        if len(plies) == 2:
            assert abs(plies[0] - plies[1]) >= 12
    hashes = [r["canonical_sha1"] for r in rows]
    assert len(hashes) == len(set(hashes))


def test_sampler_is_deterministic_for_a_profile():
    a = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                          alloc=_production_alloc())
    b = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                          alloc=_production_alloc())
    assert a == b


def test_mutating_module_constants_cannot_change_a_schema2_result(monkeypatch):
    alloc = _production_alloc()
    before = v2.sample_v2_rows(make_feasible_120_pool(), seed=11, alloc=alloc)
    monkeypatch.setattr(v2, "CORPUS_SIZE", 999)
    monkeypatch.setattr(v2, "MAX_PER_GAME", 1)
    monkeypatch.setattr(v2, "MIN_PLY_GAP", 100)
    monkeypatch.setattr(v2, "SIDE_TOL", 0)
    monkeypatch.setattr(v2, "SPLIT_ALLOC_V2",
                        {("target", "opening"): {"tuning": 1,
                                                 "frozen_check": 1}})
    after = v2.sample_v2_rows(make_feasible_120_pool(), seed=11, alloc=alloc)
    assert after == before


def test_qualification_raise_matches_report_verdict():
    fixture = make_gate_fail_fixture()
    with pytest.raises(ValueError, match="target.*opening"):
        v2.post_screen_qualification(fixture)          # legacy default
    v2.post_screen_qualification(fixture, alloc=_production_alloc())  # no raise


# ---------------------------------------------------------------------------
# Task 5: per-split late-target band minima in the sampler.
# ---------------------------------------------------------------------------

def test_per_split_band_minima_hold_on_selected_rows():
    rows, stats = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                                    alloc=_production_alloc())
    by_split = {s: Counter() for s in ("tuning", "frozen_check")}
    for r in rows:
        if (r["role"], r["phase"]) == ("target", "late"):
            by_split[r["split"]][r["band"]] += 1
    assert by_split["tuning"]["b400_plus"] >= 4
    assert by_split["tuning"]["b300_399"] >= 8
    assert by_split["tuning"]["b200_299"] >= 8
    assert by_split["frozen_check"]["b400_plus"] >= 4
    assert by_split["frozen_check"]["b300_399"] >= 5
    assert by_split["frozen_check"]["b200_299"] >= 5
    assert stats["late_target_band_count_by_split"]["tuning"]["b400_plus"] >= 4


def test_schema2_profile_with_empty_per_split_minima_still_gets_the_key():
    # build_selector_witness reads stats["late_target_band_count_by_split"]
    # unconditionally for every schema-2 run; the key must be present even
    # when a legal schema-2 profile carries NO per-split minima (only totals).
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    raw["late_target_band_minima"] = {}
    alloc = v2.parse_allocation_profile(raw, source="test")
    assert alloc.schema_version == 2
    assert alloc.band_minima_per_split == {}
    rows, stats = v2.sample_v2_rows(make_feasible_120_pool(), seed=11, alloc=alloc)
    assert "late_target_band_count_by_split" in stats
    assert sum(stats["late_target_band_count_by_split"]["tuning"].values()) > 0


@pytest.mark.parametrize("starve_band", ["b400_plus", "b300_399", "b200_299"])
def test_insufficient_band_capacity_fails_by_name(starve_band):
    pool = [r for r in make_feasible_120_pool()
            if not (r["role"] == "target" and r["band"] == starve_band)]
    with pytest.raises(ValueError, match=starve_band):
        v2.sample_v2_rows(pool, seed=11, alloc=_production_alloc())


def test_legacy_profile_selection_is_unchanged_by_the_split_minima_code():
    # Golden guard: the legacy path must not change. `_golden_pool()` wraps the
    # existing sampler suite's feasible-pool builder (_abundant_pool_v2).
    pool = _golden_pool()
    assert (v2.sample_v2_rows(pool, seed=3) ==
            v2.sample_v2_rows(pool, seed=3,
                              alloc=v2.AllocationProfile.legacy()))


# ---------------------------------------------------------------------------
# Task 6: per-phase geometric preflight quotas (non-uniform, odd-safe).
# ---------------------------------------------------------------------------

def _pair_proposals(gi, phase, ply, band):
    cell = (phase, None) if phase != "late" else ("late", band)
    return [
        {"game_idx": gi, "ply": ply, "side": "red", "phase": phase,
         "n_legal": 528 - ply, "band": band, "proposal_cell": cell},
        {"game_idx": gi, "ply": ply + 12, "side": "black", "phase": phase,
         "n_legal": 528 - ply - 12, "band": band, "proposal_cell": cell},
    ]


def make_production_geometry():
    """Whole-pair candidate geometry ample for quotas 15/15/15/75 + the
    candidate floors 8/12/12."""
    by_game, gi = {}, 0
    for phase, ply in (("opening", 2), ("early_mid", 20), ("midgame", 50)):
        for _ in range(12):                       # 12 pairs = 24 >= 15
            by_game[gi] = _pair_proposals(gi, phase, ply, "b400_plus"); gi += 1
    for band, n in (("b400_plus", 10), ("b300_399", 20), ("b200_299", 20)):
        for _ in range(n):                        # 50 pairs = 100 >= 75
            by_game[gi] = _pair_proposals(gi, "late", 100, band); gi += 1
    return by_game


def test_preflight_accepts_per_phase_quota_mapping():
    alloc = _production_alloc()
    report = v2.v2_geometry_feasibility(
        make_production_geometry(),
        quota_per_phase=alloc.quota_by_phase,
        late_candidate_floors=alloc.band_minima_total,
        max_per_game=alloc.max_per_game, min_gap=alloc.min_ply_gap,
        side_tol=alloc.side_tol,
        split_totals=alloc.split_totals)
    assert report.feasible, report.binding_constraint
    assert report.quota_per_phase == alloc.quota_by_phase


def test_preflight_names_the_starved_phase_under_a_mapping():
    geometry = {gi: rows for gi, rows in make_production_geometry().items()
                if rows[0]["phase"] != "opening"}
    alloc = _production_alloc()
    report = v2.v2_geometry_feasibility(
        geometry, quota_per_phase=alloc.quota_by_phase,
        late_candidate_floors=alloc.band_minima_total,
        split_totals=alloc.split_totals)
    assert not report.feasible
    assert "opening" in report.binding_constraint


def test_scalar_quota_legacy_path_unchanged():
    # The existing suite's own preflight fixtures keep passing untouched --
    # this is just the direct scalar-call sanity check on the new signature.
    geometry = make_production_geometry()
    report = v2.v2_geometry_feasibility(geometry, quota_per_phase=24)
    assert report.quota_per_phase == {p: 24 for p in v2.PHASES}


# ---------------------------------------------------------------------------
# Task 7: post-screen-qualify stage + select requires a PASS report.
# ---------------------------------------------------------------------------

def test_post_screen_report_document_binds_screen_and_profile(tmp_path):
    alloc = v2.AllocationProfile.legacy()
    kept = make_gate_fail_fixture()
    report = v2.post_screen_qualification_report(kept, alloc)
    doc = v2.build_post_screen_report_document(
        report, selector_witness=None, selector_error=None,
        screen_csv_sha1="abc123", config=None, alloc=alloc)
    assert doc["status"] == "GATE_FAIL"
    assert doc["screen_csv_sha1"] == "abc123"
    assert doc["no_manifest_written"] is True
    assert doc["profile"] == alloc.fingerprint()


def test_pass_requires_a_complete_selector_witness():
    # Review correction 1: capacity PASS + no witness must NOT yield PASS.
    alloc = _production_alloc()
    kept = make_feasible_120_pool()
    report = v2.post_screen_qualification_report(kept, alloc)
    assert report["status"] == "PASS"                  # necessary bounds hold
    no_witness = v2.build_post_screen_report_document(
        report, selector_witness=None,
        selector_error="synthetic: selector refused", screen_csv_sha1="s",
        config=None, alloc=alloc)
    assert no_witness["status"] == "GATE_FAIL"
    assert "selector" in no_witness["binding_constraint"]
    rows, stats = v2.sample_v2_rows(kept, seed=20260718, alloc=alloc)
    witness = v2.build_selector_witness(rows, stats)
    with_witness = v2.build_post_screen_report_document(
        report, selector_witness=witness, selector_error=None,
        screen_csv_sha1="s", config=None, alloc=alloc)
    assert with_witness["status"] == "PASS"
    assert with_witness["selector_witness"]["n_rows"] == 120
    assert with_witness["selector_witness"]["cell_counts"][
        "target|late|frozen_check"] == 20


def test_require_pass_report_rejects_missing_failed_stale_mismatched(tmp_path):
    alloc = _production_alloc()
    report_path = tmp_path / "psq.json"

    class Cfg:   # duck-typed: only the fields the gatekeeper reads
        post_screen_report_out = str(report_path)
        config_path = "cfg.json"
        selection_seed = 20260718
        expected_fingerprints = {"protocol_sha1": "proto1"}

    kw = dict(screen_csv_sha1="s1", config_sha1="cfg1")
    with pytest.raises(FileNotFoundError):
        v2.require_pass_report(Cfg(), "screen.csv", alloc, **kw)
    # Final review edit 1: the PASS report binds the COMPLETE config --
    # protocol_sha1, config_sha1, selection_seed (the witness depends on it),
    # and run_kind, alongside screen bytes + profile.
    ok = {"status": "PASS", "screen_csv_sha1": "s1",
          "profile": alloc.fingerprint(), "protocol_sha1": "proto1",
          "config_sha1": "cfg1", "selection_seed": 20260718,
          "run_kind": "production"}
    for bad, needle in [
            (dict(ok, status="GATE_FAIL"), "GATE_FAIL"),
            (dict(ok, screen_csv_sha1="other"), "stale"),
            (dict(ok, profile=v2.AllocationProfile.legacy().fingerprint()),
             "profile"),
            (dict(ok, protocol_sha1="other"), "protocol_sha1"),
            (dict(ok, config_sha1="other"), "config_sha1"),
            (dict(ok, selection_seed=1), "selection_seed"),
            (dict(ok, run_kind="tooling_smoke"), "run_kind"),
    ]:
        report_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=needle):
            v2.require_pass_report(Cfg(), "screen.csv", alloc, **kw)
    report_path.write_text(json.dumps(ok))
    assert v2.require_pass_report(
        Cfg(), "screen.csv", alloc, **kw)["status"] == "PASS"


def test_cli_post_screen_qualify_requires_screen_and_schema2(tmp_path):
    cfg_path = _write_config(tmp_path, {})       # schema 1
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", str(tmp_path / "nope.csv")]) == 2


# ---------------------------------------------------------------------------
# Task 8: protocol v2 (run_kind + allocation authority) carried and
# fingerprinted through build_protocol / derive_config / qualify preflight.
# Real fixtures: `_protocol_params()` (v1 params builder) and
# `_conformant_reservoir()` (returns a (protocol, measurements) pair) from
# tests/test_fpu_dev_reservoir_protocol.py.
# ---------------------------------------------------------------------------

def _v2_protocol_params():
    params = _protocol_params()
    params.update({
        "protocol_version": 2, "config_schema_version": 2,
        "run_kind": "production",
        "phase_allocation": PRODUCTION_PROFILE_RAW["phase_allocation"],
        "late_floors": PRODUCTION_PROFILE_RAW["late_floors"],
        "late_target_band_minima":
            PRODUCTION_PROFILE_RAW["late_target_band_minima"],
        "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
        "corpus_size": 120, "post_screen_report_out": "psq.json",
    })
    return params


def test_build_protocol_v2_requires_and_validates_run_kind():
    params = _v2_protocol_params()
    p = proto.build_protocol(params)
    assert p["run_kind"] == "production"
    params["run_kind"] = "experiment"
    with pytest.raises(ValueError, match="run_kind"):
        proto.build_protocol(params)
    del params["run_kind"]
    with pytest.raises(ValueError, match="run_kind"):
        proto.build_protocol(params)


def test_v1_protocol_schema_is_untouched():
    p = proto.build_protocol(_protocol_params())
    assert "run_kind" not in p


def test_build_protocol_v2_validates_allocation_via_parser():
    params = _v2_protocol_params()
    params["corpus_size"] = 121   # inconsistent with the allocation total
    with pytest.raises(ValueError, match="corpus_size"):
        proto.build_protocol(params)


def test_derive_config_v2_carries_run_kind_and_profile_fields():
    _, measurements = _conformant_reservoir()
    protocol = proto.build_protocol(_v2_protocol_params())
    cfg = proto.derive_config(protocol, measurements, protocol_path="p.json")
    for key in ("run_kind", "late_target_band_minima", "max_per_game",
                "min_ply_gap", "side_tol", "corpus_size",
                "post_screen_report_out"):
        assert key in cfg, key
    assert cfg["run_kind"] == "production"
    assert cfg["config_schema_version"] == 2


def test_derive_config_v1_carries_no_v2_fields():
    protocol, measurements = _conformant_reservoir()   # v1
    cfg = proto.derive_config(protocol, measurements, protocol_path="p.json")
    assert "run_kind" not in cfg
    assert "post_screen_report_out" not in cfg


# ---------------------------------------------------------------------------
# Task 9: diagnostic honesty -- smoke rejection, run_kind fingerprint, honest
# inactive gates. `_safe_target`/`_golden_ref`/`_golden_cand` are this file's
# own Task 0 golden fixtures (same row shape `dev_safety_verdict` tests use
# throughout the suite); `PRODUCTION_PROFILE_RAW`/`_write_config` are Task 1/2's
# schema-2 config fixtures already defined above.
# ---------------------------------------------------------------------------

def test_dev_safety_verdict_names_inactive_strata():
    # 25 band-A target rows (active, n>=DEV_BAND_MIN_N) + 5 band-B rows
    # (inactive, n<DEV_BAND_MIN_N): the gate for B must be reported as NOT
    # having run, with its sample size -- both keyed under the DEFAULT
    # stratum_key ("band"), since no --dev-corpus-config is in play here.
    rows = ([_safe_target(band="b300_399") for _ in range(25)]
            + [_safe_target(band="b200_299") for _ in range(5)])
    verdict = diag.dev_safety_verdict(rows, _golden_ref(), _golden_cand(),
                                      _golden_cand(), include_stratum_census=True)
    assert verdict.metrics["band_stratum_sizes"] == {
        "b300_399": 25, "b200_299": 5}
    assert verdict.metrics["band_inactive_strata"] == ["b200_299"]
    # Default OFF: v1 gate-JSON metrics unchanged (Task 0 golden also pins this).
    plain = diag.dev_safety_verdict(rows, _golden_ref(), _golden_cand(),
                                    _golden_cand())
    assert "band_stratum_sizes" not in plain.metrics
    assert "band_inactive_strata" not in plain.metrics


def test_dev_safety_verdict_census_covers_gated_and_band_summary():
    # stratum_key="ply_bucket" (v2): census must appear under BOTH the gated
    # ply_bucket_ prefix AND the always-reported band_ summary branch.
    rows = [_safe_target(band="b300_399", ply_bucket="late") for _ in range(25)] + [
        _safe_target(band="b200_299", ply_bucket="mid") for _ in range(3)]
    verdict = diag.dev_safety_verdict(rows, _golden_ref(), _golden_cand(),
                                      _golden_cand(), stratum_key="ply_bucket",
                                      include_stratum_census=True)
    assert verdict.metrics["ply_bucket_stratum_sizes"] == {"late": 25, "mid": 3}
    assert verdict.metrics["ply_bucket_inactive_strata"] == ["mid"]
    assert verdict.metrics["band_stratum_sizes"] == {"b300_399": 25, "b200_299": 3}
    assert verdict.metrics["band_inactive_strata"] == ["b200_299"]


def test_production_diagnostic_rejects_tooling_smoke(tmp_path):
    extra = dict(PRODUCTION_PROFILE_RAW)
    extra["run_kind"] = "tooling_smoke"
    extra["post_screen_report_out"] = "psq.json"
    cfg = v2.load_v2_config(_write_config(tmp_path, extra))
    with pytest.raises(SystemExit, match="tooling_smoke"):
        diag.require_production_run_kind(cfg)


def test_production_diagnostic_accepts_production_run_kind(tmp_path):
    extra = dict(PRODUCTION_PROFILE_RAW)
    extra["post_screen_report_out"] = "psq.json"
    cfg = v2.load_v2_config(_write_config(tmp_path, extra))
    diag.require_production_run_kind(cfg)   # must not raise


@dataclass
class _FpBaseCfg:
    """Adapted from tests/test_fpu_evidence_chain.py's `_FakeCfg` -- the
    minimal `base_cfg` shape `build_run_fingerprint` feeds to
    `dataclasses.asdict`."""
    c_puct: float = 1.5
    fpu_policy_mass_reduction: object = None
    eval_batch_size: int = 14
    stall_flush_sims: int = 48
    n_simulations: int = 400


def test_run_fingerprint_records_run_kind_only_when_given(tmp_path):
    ckpt = tmp_path / "ck.npz"; ckpt.write_bytes(b"CKPT-BYTES")
    manifest = tmp_path / "m.csv"; manifest.write_text("split\ntuning\n")
    src = tmp_path / "src.jsonl"; src.write_text('{"game_idx": 0}\n')
    rp = tmp_path / "r0.json"; rp.write_text('{"moves": []}')
    common = dict(
        dev_manifest=str(manifest), checkpoint=str(ckpt), base_cfg=_FpBaseCfg(),
        source_jsonl=str(src), replay_paths=[str(rp)],
        seeds={"seed_base": 1, "eval_batch_size": 14, "stall_flush_sims": 48},
        selected_a_manifest=None, mode="tuning", stage="controls")

    fp = diag.build_run_fingerprint(**common, run_kind="production")
    assert fp["run_kind"] == "production"

    legacy = diag.build_run_fingerprint(**common)
    assert "run_kind" not in legacy      # review correction 4: v1 bytes intact


def test_analyze_core_pass_with_witness():
    doc = v2._analyze_screen_kept(make_feasible_120_pool(),
                                  _production_alloc(), 20260718)
    assert doc["status"] == "PASS" and doc["discovery_only"] is True
    assert doc["selector_witness"]["n_rows"] == 120
    assert doc["qualification"]["status"] == "PASS"


def test_analyze_core_old_allocation_fails():
    legacy_raw = {
        "config_schema_version": 2, "run_kind": "production",
        "phase_allocation": {f"{r}|{p}": dict(a) for (r, p), a
                             in v2.SPLIT_ALLOC_V2.items()},
        "late_floors": dict(v2.LATE_TARGET_FLOORS),
        "late_target_band_minima": {},
        "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
        "corpus_size": 240,
    }
    alloc = v2.parse_allocation_profile(legacy_raw, source="test")
    doc = v2._analyze_screen_kept(make_gate_fail_fixture(), alloc, 20260718)
    assert doc["status"] == "GATE_FAIL"
    assert "target|opening" in doc["qualification"]["binding_constraint"]


def test_analyze_core_is_deterministic():
    a = v2._analyze_screen_kept(make_feasible_120_pool(),
                                _production_alloc(), 20260718)
    b = v2._analyze_screen_kept(make_feasible_120_pool(),
                                _production_alloc(), 20260718)
    assert a == b


def test_load_analysis_profile_seed_happy_path_and_missing_key(tmp_path):
    """Review fix: `_load_analysis_profile` is the guard-reachable seam
    `analyze_screen_feasibility` extracted its profile-load+parse+seed-read
    into. Happy path returns (alloc, seed); a profile JSON missing
    `selection_seed` (NOT a `parse_allocation_profile` schema key -- the
    protocol/qualify path legitimately parses profiles without one) must
    raise a plain `ValueError`, not the `KeyError` that used to escape
    `analyze_screen_feasibility` uncaught."""
    with_seed = dict(PRODUCTION_PROFILE_RAW, selection_seed=20260718)
    ok_path = tmp_path / "profile.json"
    ok_path.write_text(json.dumps(with_seed))
    alloc, seed = v2._load_analysis_profile(str(ok_path))
    assert seed == 20260718
    assert isinstance(alloc, v2.AllocationProfile)

    no_seed = copy.deepcopy(PRODUCTION_PROFILE_RAW)   # never had one
    bad_path = tmp_path / "no_seed_profile.json"
    bad_path.write_text(json.dumps(no_seed))
    with pytest.raises(ValueError, match="selection_seed"):
        v2._load_analysis_profile(str(bad_path))


def test_binomial_lower_bound_pins_the_299_rule():
    assert v2._binomial_lower_bound(299, 299, 0.05) >= 0.99
    assert v2._binomial_lower_bound(298, 298, 0.05) < 0.99
    assert v2._binomial_lower_bound(0, 100, 0.05) == 0.0
    # One failure in 299 must drop the bound below the criterion.
    assert v2._binomial_lower_bound(298, 299, 0.05) < 0.99


def test_precheck_step5_uses_the_configs_own_profile_not_legacy(monkeypatch, tmp_path):
    """Task 12 fix: `precheck_before_screen`'s defensive step-5 preflight must
    use the config's OWN allocation (via `_precheck_preflight_alloc` ->
    `profile_for`), not the legacy 240-row module quotas that `alloc=None`
    imposes -- so a reservoir `run_qualify` accepts under the config's profile
    is not then REJECTED at screen precheck under a stricter, wrong bar.

    Value: schema-2 -> the config's own profile (NOT legacy/None); schema-1 ->
    None (byte-identical to the prior single-arg call). Wiring: step 5 actually
    consults the seam (spied to raise, proving precheck reaches it before the
    evaluator loads)."""
    cfg2 = v2.load_v2_config(_write_config(
        tmp_path, dict(PRODUCTION_PROFILE_RAW, post_screen_report_out="psq.json")))
    alloc = proto._precheck_preflight_alloc(cfg2)
    assert alloc is not None
    assert alloc.fingerprint() == v2.profile_for(cfg2).fingerprint()
    assert alloc.fingerprint() != v2.AllocationProfile.legacy().fingerprint()

    cfg1 = v2.load_v2_config(_write_config(tmp_path, {}))
    assert proto._precheck_preflight_alloc(cfg1) is None

    # Wiring: precheck's step 5 consults the seam (default preflight, gate True).
    config, *_ = _write_precheckable_config(tmp_path)
    marker = RuntimeError("step-5 seam reached")

    def _spy(_config):
        raise marker

    monkeypatch.setattr(proto, "_precheck_preflight_alloc", _spy)
    with pytest.raises(RuntimeError, match="step-5 seam reached"):
        proto.precheck_before_screen(config)


def test_sizing_core_universe_includes_zero_yield_games():
    kept = make_feasible_120_pool()                    # games 0..229
    universe = list(range(250))                        # + 20 zero-yield games
    alloc = _production_alloc()
    kw = dict(game_counts=[100, 250], trials=8, seed=5)
    a = v2.sizing_analysis_core(kept, universe, alloc, 20260718, **kw)
    b = v2.sizing_analysis_core(kept, universe, alloc, 20260718, **kw)
    assert a == b                                      # deterministic
    assert a["n_games_available"] == 250
    assert a["n_zero_yield_games"] == 20
    full = a["by_game_count"]["250"]
    assert full["degenerate_full_reservoir"] is True
    assert full["n_trials"] == 1                       # not 8 identical draws
    small = a["by_game_count"]["100"]
    assert small["n_trials"] == 8
    assert small["n_successes"] + sum(
        small["failure_reasons"].values()) == 8
    assert 0.0 <= small["lower_bound_95"] <= small["success_rate"] or (
        small["success_rate"] == 0.0)
    assert a["cannot_certify_beyond"] == 250
    assert a["method"] == "finite-reservoir whole-game subsampling"


# ---------------------------------------------------------------------------
# Task 12 fix: the Sec 5 re-derive + byte-compare tamper check
# (`assert_config_byte_equals_rederivation`) must reconstruct the SAME key set
# `derive_config` emits, including the seven schema-2-only fields -- otherwise
# EVERY schema-2 config fails precheck and the whole v2 screen/select path is
# unreachable. Fixture pattern from Task 8's own tests (`_v2_protocol_params` +
# `_conformant_reservoir` -> real `derive_config`).
# ---------------------------------------------------------------------------

def test_bytecompare_passes_on_honest_schema2_config():
    """An honest schema-2 config (real `derive_config` on a v2 protocol)
    byte-equals its own re-derivation -- the 7 schema-2 fields are compared,
    not dropped. RED before the fix (they were absent from the reconstruction
    and flagged as differing)."""
    from types import SimpleNamespace
    _, measurements = _conformant_reservoir()
    protocol = proto.build_protocol(_v2_protocol_params())
    recomputed = proto.derive_config(protocol, measurements, protocol_path="p.json")
    config = SimpleNamespace(**recomputed)   # duck-typed supplied config
    proto.assert_config_byte_equals_rederivation(config, recomputed)   # no raise


def test_bytecompare_flags_tampered_schema2_only_field():
    """Mutating a schema-2-ONLY field (`corpus_size`) on the supplied side is
    caught, naming the key -- proves the fix did not open a hole for the very
    fields it started comparing."""
    from types import SimpleNamespace
    _, measurements = _conformant_reservoir()
    protocol = proto.build_protocol(_v2_protocol_params())
    recomputed = proto.derive_config(protocol, measurements, protocol_path="p.json")
    tampered = dict(recomputed)
    tampered["corpus_size"] = recomputed["corpus_size"] + 1
    config = SimpleNamespace(**tampered)
    with pytest.raises(ValueError, match="corpus_size"):
        proto.assert_config_byte_equals_rederivation(config, recomputed)


# ---------------------------------------------------------------------------
# Task 12: zero-GPU schema-2 CLI integration on FABRICATED artifacts. The first
# full traversal of the real screen -> post-screen-qualify -> select evidence
# path; ONLY the two evaluator seams (`_build_v2_anchor_search_fn`,
# `build_teacher_calibration_manifest._teacher_infer`) are faked. Every
# artifact reader/writer/hash, config re-derivation, CLI route, precheck, the
# geometric preflight, and the real sampler/selector all execute for real.
#
# Allocation = the Task-15 tooling_smoke SHAPE (corpus 18): target|late 4+2 and
# control|{4 phases} 2+1 each, empty band minima. Chosen because it is
# comfortably fillable by a small fabricated pool: 12 games x the 6 v2 proposal
# cells is geometry-feasible (>=9 games suffice) AND the role-aware select gate
# is satisfiable with sides balanced within every split (see `_fake_role_*`).
# ---------------------------------------------------------------------------

_SMOKE_PROFILE = {
    "config_schema_version": 2,
    "run_kind": "tooling_smoke",
    "phase_allocation": {
        "target|late":       {"tuning": 4, "frozen_check": 2},
        "control|opening":   {"tuning": 2, "frozen_check": 1},
        "control|early_mid": {"tuning": 2, "frozen_check": 1},
        "control|midgame":   {"tuning": 2, "frozen_check": 1},
        "control|late":      {"tuning": 2, "frozen_check": 1},
    },
    "late_floors": {},
    "late_target_band_minima": {},
    "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2, "corpus_size": 18,
}

_SMOKE_GAMES = 12   # >=9 for feasibility; 12 gives slack after opening collisions


def _legal_game_moves(n_plies, game_idx, *, board_size=24):
    """A reconstruction-valid legal TwixtState game prefix (like
    tests/goal_line_probe_fixtures.legal_replay, which position_state needs),
    but DIVERGENT per game_idx (pick `legal[game_idx % len(legal)]` each ply)
    so canonical_state_sha1 differs across games -- an identical game every
    time would collide on every shared position and get screened out. Yields
    exactly `n_plies` well-formed ply records (row/col legal + honest
    n_legal), red first, alternating -- the schema B4 move-parity + the
    enumerator both require."""
    st = TwixtState(active_size=board_size, to_move="red",
                    max_plies_limit=board_size * board_size)
    moves = []
    for ply in range(n_plies):
        legal = st.legal_moves()
        assert st.winner() is None and legal, (game_idx, ply)   # must reach n_plies
        r, c = legal[game_idx % len(legal)]
        moves.append({
            "ply": ply, "player": st.to_move, "row": r, "col": c,
            "root_value": 0.0, "root_top1_share": 0.5,
            "selected_visit_rank": 1, "selected_visit_count": 100,
            "root_total_visits": 100, "n_legal": len(legal),
        })
        st = st.apply_move((r, c))
    return moves


def _fabricate_qualified_v2_reservoir(tmp_path):
    """Fabricate an on-disk reservoir + a v2 (tooling_smoke) protocol, run the
    REAL run_qualify, and return the derived schema-2 config path.

    Reuses the suite's `_write_qualifiable_reservoir` for the checkpoints /
    index / match-summary / protocol-freeze / B4+B5 conformance, then
    OVERWRITES each replay's `moves` (its (1,1) fillers cannot replay through
    position_state, and a 330-ply game is needed to reach all six proposal
    cells) with a divergent legal game -- BEFORE qualify, so measure_reservoir
    hashes the final legal bytes into replay_data_sha1."""
    overrides = dict(_SMOKE_PROFILE)
    overrides.update({
        "protocol_version": 2,
        "screen_out": str(tmp_path / "screen.csv"),
        "select_out": str(tmp_path / "manifest_v2.csv"),
        "post_screen_report_out": str(tmp_path / "post_screen_report.json"),
    })
    protocol, protocol_path, info = _write_qualifiable_reservoir(
        tmp_path, games=_SMOKE_GAMES, n_moves=330, protocol_overrides=overrides)

    for row in info["rows"]:
        replay_path = Path(row["replay_path"])
        sidecar = json.loads(replay_path.read_text())
        sidecar["moves"] = _legal_game_moves(sidecar["n_moves"], row["game_idx"])
        replay_path.write_text(json.dumps(sidecar))

    # The fixture's forbidden manifest uses a `state_sha1` header run_screen's
    # `load_forbidden_hashes` does not accept; replace it (BEFORE qualify, so
    # its hash is captured) with an empty, correctly-headed manifest -- an
    # empty forbidden set is what we want (no proposal is pre-excluded).
    Path(info["manifest_path"]).write_text("canonical_position_sha1\n")

    assert proto.run_qualify(
        str(protocol_path), preflight=_fake_preflight(True)) == proto.EXIT_OK
    return protocol["config_out"]


# --- the two evaluator seams, faked deterministically ----------------------

def _fake_teacher_infer(state, evaluator):
    """`_teacher_infer(state, evaluator) -> (legal, priors, raw_value)`; only
    `priors` is read (-> `_policy_features_from_priors` -> `raw_policy_role`).
    Role is derived FROM the state so it agrees with the proposal's own cell:
    late b400_plus/b300_399 -> target (flat priors: entropy 1.0, top1 0.005);
    everything else -> control (peaked: top1 0.9). This keeps BOTH roles on
    BOTH sides within every cell, so each split stays side-balanced
    (|red-black| <= side_tol) -- a role-by-side scheme cannot."""
    phase = v2.ply_bucket_of(state.ply)
    band = v2.band_of(len(state.legal_moves()))
    is_target = phase == "late" and band in ("b400_plus", "b300_399")
    priors = [1.0 / 200] * 200 if is_target else [0.9] + [0.1 / 199] * 199
    return [], priors, 0.0


@dataclass(frozen=True)
class _FakeAnchorCfg:
    """Whatever run_screen records via `dataclasses.asdict(anchor_cfg)` into
    the screen meta -- only needs to BE a dataclass."""
    fpu_policy_mass_reduction: object = None
    n_simulations: int = v2.ANCHOR_SIMS_V2


def _fake_build_v2_anchor_search_fn(checkpoint, eval_batch_size, stall_flush_sims):
    """Replaces the whole seam (its real body loads a checkpoint + MLX MCTS).
    `search_fn(state, seed) -> (counts, root_value_stm, root)`: root_value_stm
    0.0 is anchor-eligible (|v|<=0.25) so survivors are kept; root.visit_count
    must equal ANCHOR_SIMS_V2 or run_screen raises."""
    def search_fn(state, seed):
        return {}, 0.0, SimpleNamespace(visit_count=v2.ANCHOR_SIMS_V2)
    return object(), search_fn, _FakeAnchorCfg()


def test_schema2_cli_end_to_end_on_fabricated_artifacts(tmp_path, monkeypatch):
    # (1) Fabricate + qualify -> derived schema-2 config on disk.
    cfg_path = _fabricate_qualified_v2_reservoir(tmp_path)

    # (2) Fake ONLY the two evaluator seams (both resolved at call time).
    monkeypatch.setattr(v2, "_build_v2_anchor_search_fn",
                        _fake_build_v2_anchor_search_fn)
    monkeypatch.setattr(btcm, "_teacher_infer", _fake_teacher_infer)

    # (3) Drive the REAL CLI through the whole evidence path.
    assert v2.main(["--mode", "screen", "--config", cfg_path]) == 0
    cfg = v2.load_v2_config(cfg_path)
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    report_bytes = Path(cfg.post_screen_report_out).read_bytes()
    assert json.loads(report_bytes)["status"] == "PASS"
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    manifest_bytes = Path(cfg.select_out).read_bytes()

    # (4) Idempotency: re-running is a clean accept, byte-identical artifacts.
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    assert Path(cfg.post_screen_report_out).read_bytes() == report_bytes
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    assert Path(cfg.select_out).read_bytes() == manifest_bytes

    # (5) Report tampering: missing -> 2, edited (status flipped) -> 3.
    saved = Path(cfg.post_screen_report_out)
    saved.unlink()
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 2
    doc = json.loads(report_bytes)
    doc["status"] = "GATE_FAIL"
    saved.write_text(json.dumps(doc))
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 3
    saved.write_bytes(report_bytes)

    # (6) Smoke isolation: the production diagnostic rejects this config.
    with pytest.raises(SystemExit, match="tooling_smoke"):
        diag.require_production_run_kind(cfg)


# ---------------------------------------------------------------------------
# Task 14a: discovery-only historical-screen identity policy
# (`historical_screen_discovery_v1`). The immutable pre-repair v1 screen was
# PRODUCED by an earlier revision of the two v2 tooling modules; today's
# repaired analyzer differs. The READ-ONLY discovery stages
# (analyze-screen-feasibility / sizing-analysis) RECORD that producer-vs-
# analyzer split for the two allowlisted files (A==B still required, C may
# diverge) while every strict stage (screen/post-screen-qualify/select) and
# every other identity/field stays byte-strict.
#
# Faithful simulation of "the analyzer .py changed since this screen was
# produced": leave every recorded artifact (config, screen, meta) UNTOUCHED --
# so config_sha1 / screen_csv_sha1 / A==B all stay honest -- and drift only the
# FRESH source-hash recompute (C), which both the identity check and the
# rederive take through `fpu_provenance.source_file_sha1s`.
# ---------------------------------------------------------------------------

def _screened_historical_reservoir(tmp_path, monkeypatch):
    """Fabricate + qualify a real reservoir, fake the two evaluator seams, and
    run the REAL `screen` -- yielding an immutable historical screen whose
    recorded provenance (A==B) was taken with TODAY's source bytes. Returns
    (cfg_path, cfg, profile_path)."""
    cfg_path = _fabricate_qualified_v2_reservoir(tmp_path)
    monkeypatch.setattr(v2, "_build_v2_anchor_search_fn",
                        _fake_build_v2_anchor_search_fn)
    monkeypatch.setattr(btcm, "_teacher_infer", _fake_teacher_infer)
    assert v2.main(["--mode", "screen", "--config", cfg_path]) == 0
    cfg = v2.load_v2_config(cfg_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(
        dict(_SMOKE_PROFILE, selection_seed=cfg.selection_seed)))
    return cfg_path, cfg, str(profile_path)


def _drift_recompute(monkeypatch, *basenames):
    """Make the FRESH source-hash recompute (C) diverge from the on-disk
    recorded A/B for `basenames` -- without touching any recorded artifact."""
    real = v2.fpu_provenance.source_file_sha1s

    def drifted(paths):
        d = real(paths)
        for name in basenames:
            if name in d:
                d[name] = "drift" + "0" * 36
        return d

    monkeypatch.setattr(v2.fpu_provenance, "source_file_sha1s", drifted)


_ALLOWLIST = v2.HISTORICAL_SCREEN_DISCOVERY_V1_EXPECTED_SOURCE_DRIFT


# Group 1: the known two-file historical divergence PASSES discovery. -----

def test_discovery_accepts_the_two_file_historical_source_drift(tmp_path, monkeypatch):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    out = tmp_path / "analyze.json"
    rc = v2.main(["--mode", "analyze-screen-feasibility", "--config", cfg_path,
                  "--screen", cfg.screen_out, "--profile-json", profile,
                  "--out", str(out)])
    assert rc == v2.EXIT_OK
    doc = json.loads(out.read_bytes())
    prov = doc["historical_screen_discovery"]
    assert set(prov) == {
        "policy", "expected_source_drift",
        "historical_ab_source_file_sha1s", "current_analysis_source_file_sha1s",
        "source_file_divergence", "git_commit", "analysis_source_files_clean"}
    assert prov["policy"] == "historical_screen_discovery_v1"
    assert sorted(prov["source_file_divergence"]) == sorted(_ALLOWLIST)
    assert prov["historical_ab_source_file_sha1s"] != prov[
        "current_analysis_source_file_sha1s"]


def test_sizing_analysis_accepts_the_two_file_historical_source_drift(
        tmp_path, monkeypatch):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    out = tmp_path / "sizing.json"
    rc = v2.main(["--mode", "sizing-analysis", "--config", cfg_path,
                  "--screen", cfg.screen_out, "--profile-json", profile,
                  "--out", str(out), "--game-counts", "9", "--trials", "2",
                  "--seed", "20260719"])
    assert rc == v2.EXIT_OK
    prov = json.loads(out.read_bytes())["historical_screen_discovery"]
    assert prov["policy"] == "historical_screen_discovery_v1"
    assert sorted(prov["source_file_divergence"]) == sorted(_ALLOWLIST)


# Group 2: any NON-source identity mismatch still exits 3 under discovery. -

def test_discovery_still_rejects_a_non_source_identity_mismatch(
        tmp_path, monkeypatch):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    # Corrupt the screen ARTIFACT's bytes -> screen_csv_sha1 (a NON-source
    # identity) mismatches; discovery must still exit MISMATCH (3).
    with open(cfg.screen_out, "ab") as f:
        f.write(b"\n# tamper\n")
    out = tmp_path / "analyze.json"
    rc = v2.main(["--mode", "analyze-screen-feasibility", "--config", cfg_path,
                  "--screen", cfg.screen_out, "--profile-json", profile,
                  "--out", str(out)])
    assert rc == v2.EXIT_MISMATCH
    assert not out.exists()


# Group 3: an UNEXPECTED third source-file divergence is rejected. --------

def test_discovery_rejects_an_unexpected_third_source_file_divergence(
        tmp_path, monkeypatch, capsys):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    # `fpu_state_hash.py` is a v2 corpus source but NOT on the allowlist.
    _drift_recompute(monkeypatch, *_ALLOWLIST, "fpu_state_hash.py")
    out = tmp_path / "analyze.json"
    rc = v2.main(["--mode", "analyze-screen-feasibility", "--config", cfg_path,
                  "--screen", cfg.screen_out, "--profile-json", profile,
                  "--out", str(out)])
    assert rc == v2.EXIT_MISMATCH
    assert "fpu_state_hash.py" in capsys.readouterr().out
    assert not out.exists()


# Group 4: strict select / post-screen-qualify still reject the SAME drift. -

def test_strict_stages_still_reject_historical_source_drift(tmp_path, monkeypatch):
    cfg_path, cfg, _profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    # A clean post-screen report first, so `select` reaches its identity check.
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    # (a) strict post-screen-qualify MISMATCHes on the SAME drift discovery accepts.
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == v2.EXIT_MISMATCH
    # (b) strict select's identity hard-match raises RAW (evidence-chain failure,
    #     never a terse exit code -- see select's own main clause).
    with pytest.raises(ValueError, match="source_file_sha1s"):
        v2.main(["--mode", "select", "--config", cfg_path,
                 "--screen", cfg.screen_out])


# Group 5: a NON-source config-field tamper still fails discovery rederive. -

def test_discovery_rederive_still_catches_a_non_source_config_tamper(
        tmp_path, monkeypatch):
    import dataclasses
    _cfg_path, cfg, _profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    # The honest config (only the allowlisted source drift) PASSES the
    # discovery rederive -- the normalization pins the historical A/B block.
    v2._rederive_config_unchanged_discovery(cfg)   # no raise
    # A tampered NON-source field still fails, naming it -- the normalization
    # does not weaken strictness for any other field.
    tampered = dataclasses.replace(cfg, selection_seed=cfg.selection_seed + 1)
    with pytest.raises(ValueError, match="selection_seed"):
        v2._rederive_config_unchanged_discovery(tampered)


# Group 6: repeated discovery runs are byte-identical (write_atomic UNCHANGED).

def test_repeated_discovery_runs_are_byte_identical(tmp_path, monkeypatch):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    out = tmp_path / "analyze.json"
    args = ["--mode", "analyze-screen-feasibility", "--config", cfg_path,
            "--screen", cfg.screen_out, "--profile-json", profile, "--out", str(out)]
    assert v2.main(args) == v2.EXIT_OK
    first = out.read_bytes()
    # Re-run on the SAME tree: write_atomic sees identical bytes -> UNCHANGED,
    # idempotently accepted; the report is byte-for-byte the same.
    assert v2.main(args) == v2.EXIT_OK
    assert out.read_bytes() == first


def test_discovery_refuses_to_clobber_a_byte_different_report(tmp_path, monkeypatch):
    cfg_path, cfg, profile = _screened_historical_reservoir(tmp_path, monkeypatch)
    _drift_recompute(monkeypatch, *_ALLOWLIST)
    out = tmp_path / "analyze.json"
    out.write_bytes(b'{"stale":"report"}')   # a DIFFERENT existing report
    rc = v2.main(["--mode", "analyze-screen-feasibility", "--config", cfg_path,
                  "--screen", cfg.screen_out, "--profile-json", profile,
                  "--out", str(out)])
    assert rc == v2.EXIT_USAGE                # immutable: refused, not clobbered
    assert out.read_bytes() == b'{"stale":"report"}'
