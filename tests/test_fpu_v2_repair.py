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
