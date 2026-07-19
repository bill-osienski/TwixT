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
from pathlib import Path

import pytest

from scripts.GPU.alphazero import diagnose_fpu_policy_mass as diag
from scripts.GPU.alphazero import fpu_dev_corpus_v2 as v2
from tests.test_fpu_dev_corpus_v2 import _abundant_pool_v2

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
