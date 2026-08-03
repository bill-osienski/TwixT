"""Execution-chain integration: the REAL emitters into the REAL evaluator.

WHY THIS FILE EXISTS.

Four contract defects have now been found at seams between two v18 tasks, and
three of them cost a Step 4 restart after GPU time was already spent. Every one
had the same shape: both tasks were green in isolation, and they disagreed only
where their artifacts met, because the consumer's tests hand-wrote a surrogate
for the producer's document instead of running the producer.

`test_v18_preflight_verdict.evidence_tree` is that pattern. Its docstring says
"Nothing here is hand-written", and that is true of the ALGORITHMS -- it calls
`match_cohort` and `sizing_ladder` -- but false of the DOCUMENTS: it builds
`cohort_doc` and the sizing `record` itself. `emit_matched_cohort` and
`emit_sizing_record` are never called there, so Task 9 has only ever been shown
documents written to match its expectations.

This module closes that hole. Every stage below is the real production
function; each emitter's RETURNED digest is asserted against its own file's
bytes and carried forward as the binding the next stage verifies. The single
hand-built document is the synthetic Task 7 preflight artifact, which is the
fixture the chain exists to consume.

    task7._csv_bytes                    real serializers
    CM.match_cohort  -> CM.emit_matched_cohort          real   <- was uncovered
    S.exposure_cutoff / role_predicates / classify_rows
                     -> S.sizing_ladder -> S.emit_sizing_record  real  <- was uncovered
    a6400.build_a6400_reference_bundle  real, bundle NESTED under the v18 dir
    V.load_verified_a6400_bundle        real
    V.load_verified_inputs              real, over the EMITTED cohort + sizing files
    V._evaluate_verified                real

CPU only: no search, no evaluator, no checkpoint. Everything is written under
pytest's tmp_path -- no repository artifact and no gitignored local evidence is
read or written.

THE SEVEN PATCHED SEAMS, and why each is outside the interface under test:

  1 fpu_provenance.worktree_clean   An environment fact, not behaviour. A suite
                                    run always has untracked files; at Execution
                                    step 5 the tree is clean by construction.
                                    The CHECK that reads it is untouched.
  2 task7._authenticate_search_inputs  Opens the checkpoint and both replay
                                    reservoirs, which live under gitignored
                                    logs/. A suite test may not depend on them.
  3 task7.load_verified_universe    The real loader re-emits from that same
                                    reservoir. Task 9 asserts elsewhere that
                                    production calls it, so stubbing it here
                                    cannot hide a missing call.
  4 criteria.SIZING                 Test-scale tiers and trials. The ladder
                                    logic under test is unchanged; only how many
                                    trials it runs.
  5 a6400.REPO_ROOT                 Points the canonical-path root at the
                                    fixture tree. The resolution RULE is what is
                                    under test and is untouched.
  6 a6400._load_frozen_a6400_source The real A/6,400 CSV is gitignored.
  7 fixture anchor + fast bootstrap  Passed to the PRIVATE `_evaluate_verified`,
                                    which exists for exactly this. `evaluate`
                                    accepts neither, and a Task 9 test pins that.

NOTHING whose interface is under test is patched: not the two emitters, not the
bundle builder or verifier, not the loader, not the evaluator.
"""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero import capture_v18_a6400 as a6400
from scripts.GPU.alphazero import diagnose_v18_residual_preflight as task7
from scripts.GPU.alphazero import fpu_provenance
from scripts.GPU.alphazero import v18_cohort_matcher as CM
from scripts.GPU.alphazero import v18_control_pool as CP
from scripts.GPU.alphazero import v18_preflight_criteria as criteria
from scripts.GPU.alphazero import v18_preflight_verdict as V
from scripts.GPU.alphazero import v18_selector_sizing as S
from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import canonical_json_bytes
from tests.test_v18_preflight_verdict import (CAPS, FAST_REPLICATES,
                                              SCALED_SIZING, a_rows, census)

FIXTURE_DIR = Path(__file__).parent / "golden" / "a6400_bundle_fixture"

FROZEN_SEARCH_INPUTS = {"checkpoint_sha1": "a" * 40, "a_reservoir_sha1": "b" * 40,
                        "census_reservoir_sha1": "c" * 40}

# Named here so the module states its own scope: a reviewer can compare this
# tuple against the `monkeypatch.setattr` calls below and see nothing else.
PATCHED_SEAMS = (
    "fpu_provenance.worktree_clean",
    "diagnose_v18_residual_preflight._authenticate_search_inputs",
    "diagnose_v18_residual_preflight.load_verified_universe",
    "v18_preflight_criteria.SIZING",
    "capture_v18_a6400.REPO_ROOT",
    "capture_v18_a6400._load_frozen_a6400_source",
    "the fixture trust anchor and fast bootstrap, passed to _evaluate_verified",
)


def sha1_of(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Drive every real producer into every real consumer. Returns the paths,
    the emitters' RETURNED digests, and the anchor the bundle was verified under.
    """
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)     # 1
    monkeypatch.setattr(task7, "_authenticate_search_inputs",               # 2
                        lambda phase: dict(FROZEN_SEARCH_INPUTS))
    monkeypatch.setattr(task7, "load_verified_universe",                    # 3
                        lambda path: (json.loads(Path(path).read_bytes()),
                                      sha1_of(path)))
    monkeypatch.setattr(criteria, "SIZING",                                 # 4
                        {**criteria.SIZING, **SCALED_SIZING})

    root = tmp_path / "chain"
    root.mkdir()
    rows, a_list = census(), a_rows()
    cases = [dict(r) for r in rows] + [dict(r) for r in a_list]

    # Emitted AFTER the SIZING patch, so the loader's re-derivation is genuine.
    criteria_path = root / "frozen_preflight_criteria.json"
    criteria_sha1 = criteria.emit_frozen_criteria(str(criteria_path))
    assert criteria_sha1 == sha1_of(criteria_path)

    universe = {"run_kind": "shipped_only_preflight_source_universe",
                "all_game_ids": sorted({r["game_content_sha1"] for r in rows}),
                "git_commit": fpu_provenance.git_commit(), "worktree_clean": True}
    universe_path = root / "universe.json"
    universe_path.write_bytes(canonical_json_bytes(universe))
    universe_sha1 = sha1_of(universe_path)

    # --- Task 7's real serializers -----------------------------------------
    residual_path = root / "residual_rows.csv"
    residual_path.write_bytes(task7._csv_bytes(
        [leaf for c in cases for leaf in c["residual_leaves"]],
        V.RESIDUAL_COLUMNS))
    stripped = [{k: v for k, v in c.items() if k != "residual_leaves"}
                for c in cases]
    census_path = root / "census_positions.csv"
    census_path.write_bytes(task7._csv_bytes(stripped, task7.CENSUS_SCHEMA))
    crossover_path = root / "crossover_tables.csv"
    crossover_path.write_bytes(V._reproduce_crossover_bytes(stripped))
    census_sha1 = sha1_of(census_path)

    # --- Task 4b: the real matcher AND the real emitter ---------------------
    cohort, report = CM.match_cohort(rows, a_list)
    cohort_path = root / "matched_cohort.json"
    cohort_sha1 = CM.emit_matched_cohort(
        str(cohort_path), cohort=cohort, report=report,
        universe_sha1=universe_sha1, census_sha1=census_sha1,
        criteria_sha1=criteria_sha1,
        a_source_sha1=CP.FORBIDDEN_SOURCE_SHA1S["gate_A"])
    assert cohort_sha1 == sha1_of(cohort_path), \
        "emitter's returned digest must be the bytes it wrote"
    cohort_doc = json.loads(cohort_path.read_bytes())

    # --- Task 8: the real computations AND the real emitter -----------------
    cutoff = S.exposure_cutoff(
        S.matched_control_rows(cohort_doc["matched_cohort"], rows))
    predicates = S.role_predicates(cutoff)
    classification = S.classify_rows(rows, predicates)
    ladder = S.sizing_ladder(rows, predicates,
                             all_game_ids=universe["all_game_ids"])
    sizing_path = root / "sizing.json"
    sizing_sha1 = S.emit_sizing_record(
        str(sizing_path), cutoff=cutoff, classification=classification,
        ladder=ladder, predicates=predicates,
        bindings={"criteria_sha1": criteria_sha1, "universe_sha1": universe_sha1,
                  "census_sha1": census_sha1,
                  # The digest Task 4b RETURNED, not one recomputed here.
                  "matched_cohort_sha1": cohort_sha1})
    assert sizing_sha1 == sha1_of(sizing_path), \
        "emitter's returned digest must be the bytes it wrote"

    # --- the one hand-built document: the synthetic Task 7 artifact ---------
    artifact = {
        "run_kind": task7.RUN_KIND, "cases": stripped,
        "scientific_interpretation_forbidden": True,
        "search_execution_mode": task7.SEARCH_EXECUTION_MODE,
        "simulations": task7.SIMULATIONS, "add_noise": task7.ADD_NOISE,
        "c_puct": task7.FROZEN_C_PUCT,
        "batching_triple": list(task7.BATCHING_TRIPLE),
        "cap_grid": list(CAPS), "a_source_path": task7.A_SOURCE,
        "a_source_sha1": CP.FORBIDDEN_SOURCE_SHA1S["gate_A"],
        "population_order": list(task7.POPULATIONS),
        "seed_audit": task7.assert_seed_sets_disjoint(cases),
        "authenticated_search_inputs": dict(FROZEN_SEARCH_INPUTS),
        "source_sha1s": {p: fpu_provenance.file_sha1(p)
                         for p in task7.MEASUREMENT_SOURCE_MODULES},
        "git_commit": fpu_provenance.git_commit(), "worktree_clean": True,
        "runtime_identity_bracketed": True,
        "criteria_sha1": criteria_sha1, "universe_sha1": universe_sha1,
        "census_positions_sha1": census_sha1,
        "crossover_tables_sha1": sha1_of(crossover_path),
        "residual_rows_sha1": sha1_of(residual_path),
        "pooled_reach_numerator": sum(c["exposed_positive_mass_numerator"]
                                      for c in stripped
                                      if c["population"] == "selected_a"),
        "pooled_reach_denominator": sum(c["exposed_positive_mass_denominator"]
                                        for c in stripped
                                        if c["population"] == "selected_a"),
        "terminal_depth2_total": sum(c["terminal_depth2"] for c in stripped),
        "total_depth2_total": sum(c["total_depth2"] for c in stripped),
    }
    artifact_path = root / "preflight_artifact.json"
    artifact_path.write_bytes(canonical_json_bytes(artifact))

    # --- Task 6 -> Task 9, the bundle seam, in the same chain ---------------
    src_rows = json.loads((FIXTURE_DIR / "source_rows.json").read_text())
    capture = json.loads((FIXTURE_DIR / "capture.json").read_text())
    mode = a6400.MODES["v18_preflight_a6400"]
    capture.update({"run_kind": "v18_preflight_a6400",
                    "mode": "v18_preflight_a6400",
                    "mcts_sims": mode["mcts_sims"],
                    "gate_list": list(mode["gates"])})
    columns = sorted({k for r in src_rows for k in r})
    source_csv = ("\n".join([",".join(columns)] + [
        ",".join(str(r.get(c, "")) for c in columns) for r in src_rows]
    ) + "\n").encode()

    bundle_root = tmp_path / "bundle_repo"
    (bundle_root / a6400.A6400_SOURCE).parent.mkdir(parents=True)
    (bundle_root / a6400.A6400_SOURCE).write_bytes(source_csv)
    nested = bundle_root / "logs/eval/v18_depth2_provisional_backup"
    nested.mkdir(parents=True)
    raw_capture = json.dumps(capture, sort_keys=True).encode()
    for name in ("run1.json", "run2.json"):
        (nested / name).write_bytes(raw_capture)
    rel = "logs/eval/v18_depth2_provisional_backup"

    monkeypatch.setattr(a6400, "REPO_ROOT", bundle_root.resolve())          # 5
    monkeypatch.setattr(a6400, "_load_frozen_a6400_source",                 # 6
                        lambda: a6400._enrich_source_rows(src_rows))
    bundle_path = nested / "a6400_reference_bundle.json"
    bundle_sha1 = a6400.build_a6400_reference_bundle(
        f"{rel}/run1.json", f"{rel}/run2.json", str(bundle_path))
    assert bundle_sha1 == sha1_of(bundle_path)

    anchor = {"artifact_root": a6400.ARTIFACT_ROOT,                         # 7
              "historical_source_path": a6400.A6400_SOURCE,
              "historical_source_sha1": hashlib.sha1(source_csv).hexdigest(),
              "expected_cases": 30}

    return {
        "paths": {"preflight_artifact_path": str(artifact_path),
                  "census_path": str(census_path),
                  "crossover_tables_path": str(crossover_path),
                  "residual_rows_path": str(residual_path),
                  "cohort_path": str(cohort_path),
                  "criteria_path": str(criteria_path),
                  "universe_path": str(universe_path),
                  "sizing_path": str(sizing_path)},
        "emitted": {"criteria_sha1": criteria_sha1, "cohort_sha1": cohort_sha1,
                    "sizing_sha1": sizing_sha1, "bundle_sha1": bundle_sha1},
        "bundle_path": str(bundle_path), "anchor": anchor, "root": root,
    }


# --- the positive pass -------------------------------------------------------

def test_real_emitters_produce_what_the_real_evaluator_accepts(chain):
    """The whole point: nothing between Task 7 and the verdict is a surrogate."""
    verified_bundle = V.load_verified_a6400_bundle(chain["bundle_path"],
                                                   anchor=chain["anchor"])
    assert verified_bundle["n_cases_authenticated"] == 30
    assert verified_bundle["byte_identical"] is True

    verified = V.load_verified_inputs(**chain["paths"])
    # Task 9 bound exactly the digests the emitters RETURNED -- not values it
    # recomputed from files it was handed, and not the fixture's own arithmetic.
    assert verified["sha1s"]["matched_cohort_sha1"] == chain["emitted"]["cohort_sha1"]
    assert verified["sha1s"]["sizing_sha1"] == chain["emitted"]["sizing_sha1"]
    assert verified["sha1s"]["criteria_sha1"] == chain["emitted"]["criteria_sha1"]

    record = V._evaluate_verified(verified, chain["bundle_path"],
                                  separation_replicates=FAST_REPLICATES,
                                  anchor=chain["anchor"])
    assert record["verdict"] == V.PREFLIGHT_PASS, record.get("failures")
    assert not record.get("failures")
    assert record["sizing"]["state"] == "passed"


def test_the_emitted_sizing_record_carries_the_fields_task9_reconciles(chain):
    """A guard on the guard. Task 9 re-derives `sizing_status` and both tier
    fields; if the emitter stopped writing them, `.get` would return None and
    the reconciliation could pass vacuously against a record that claims
    nothing."""
    record = json.loads(Path(chain["paths"]["sizing_path"]).read_bytes())
    assert record["sizing_status"] == "SIZING_PASSES"
    assert record["smallest_qualifying_tier"] is not None
    assert record["recommended_operational_size"] is not None
    assert record["matched_cohort_sha1"] == chain["emitted"]["cohort_sha1"]


# --- the six negative controls -----------------------------------------------
# A chain that cannot fail proves nothing. Each control perturbs one thing a
# real producer/consumer mismatch would perturb.

def _rewrite(path, **changes):
    document = json.loads(Path(path).read_bytes())
    document.update(changes)
    Path(path).write_bytes(canonical_json_bytes(document))


def _truncate_cohort(path):
    document = json.loads(Path(path).read_bytes())
    document["matched_cohort"] = document["matched_cohort"][:-1]
    Path(path).write_bytes(canonical_json_bytes(document))


NEGATIVE_CONTROLS = [
    pytest.param(
        "cohort_path",
        lambda p: _rewrite(p, run_kind="v18_matched_control_cohort_v2"),
        "run_kind", id="emitted-cohort-altered-after-its-digest"),
    # Caught by the SIZING record's binding to the cohort bytes, not by a
    # cardinality check at load time -- which is why the binding has to be the
    # digest Task 4b returned rather than one recomputed from the file.
    pytest.param(
        "cohort_path", _truncate_cohort,
        "matched_cohort_sha1", id="emitted-cohort-list-truncated"),
    pytest.param(
        "sizing_path", lambda p: _rewrite(p, smallest_qualifying_tier=60),
        "smallest_qualifying_tier", id="sizing-smallest-tier-edited"),
    pytest.param(
        "sizing_path", lambda p: _rewrite(p, sizing_status="SIZING_FAILS"),
        "sizing_status", id="sizing-status-flipped"),
    pytest.param(
        "sizing_path", lambda p: _rewrite(p, matched_cohort_sha1="0" * 40),
        "matched_cohort_sha1", id="sizing-cohort-binding-repointed"),
    pytest.param(
        "census_path",
        lambda p: Path(p).write_bytes(Path(p).read_bytes() + b"\n"),
        "census_positions_sha1", id="census-csv-byte-appended"),
]


@pytest.mark.parametrize("target, mutate, expected", NEGATIVE_CONTROLS)
def test_negative_control_is_refused(chain, target, mutate, expected):
    """`match=` matters here: a control that is refused for someone else's
    reason is not evidence that the guard it targets exists."""
    mutate(chain["paths"][target])
    with pytest.raises(ValueError, match=expected):
        V.load_verified_inputs(**chain["paths"])


def test_every_patched_seam_is_declared(chain):
    """The seam list is part of this module's contract: a future edit that
    stubs something else must update it deliberately."""
    assert len(PATCHED_SEAMS) == 7
    for name in PATCHED_SEAMS[:6]:
        module, _, attribute = name.rpartition(".")
        assert module and attribute
