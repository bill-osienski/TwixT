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
import json
from pathlib import Path

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
