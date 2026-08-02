"""v18 preflight verdict -- Task 9.

A PURE evaluator: authenticated artifacts plus the frozen criteria in, a verdict
out. It measures nothing, searches nothing, and writes no artifact of its own.

Two things it does not do, deliberately:

  * it does not restate a threshold. Every number and every formula comes from
    `v18_preflight_criteria`, and the estimators are that module's functions,
    called rather than copied;
  * it does not TRUST the A/6,400 reference bundle. Every claim the bundle makes
    is recomputed from the files it names -- a stored `byte_identical: true` over
    two differing captures is exactly the forgery the verifier exists to catch.

**Verdict vocabulary.** Missing sizing and FAILING sizing are different states.
Mapping both to provisional would mean an infeasible selector could never
formally reject v18; it would sit at "provisional" forever. A completed sizing
run that cannot fill the four-role geometry is a real negative result.

    PREFLIGHT_FAIL                          a mechanism criterion failed, OR
                                            sizing ran and could not satisfy the
                                            frozen geometry
    MECHANISM_PREFLIGHT_PROVISIONAL_PASS    every mechanism criterion passed and
                                            sizing is absent / not yet run.
                                            NOT a preflight pass. Authorizes
                                            nothing.
    PREFLIGHT_PASS                          both passed. Only this may be
                                            presented as satisfying spec Sec 2.3.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import capture_v18_a6400 as a6400
from . import v18_preflight_criteria as criteria
from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .v18_selector_sizing import matched_control_rows

import re

RUN_KIND = "v18_preflight_verdict"

# Canonical lowercase hex. Length alone authenticates nothing.
_SHA1_RE = re.compile(r"\A[0-9a-f]{40}\Z")

PREFLIGHT_FAIL = "PREFLIGHT_FAIL"
PROVISIONAL_PASS = "MECHANISM_PREFLIGHT_PROVISIONAL_PASS"
PREFLIGHT_PASS = "PREFLIGHT_PASS"

# Every v18 artifact lives under this root. A bundle that names a capture
# outside it is naming something the frozen tree does not contain.
ARTIFACT_ROOT = "logs/eval"

# The frozen expectations the bundle verifier checks AGAINST. Injectable only so
# a suite test can stand up a fixture tree -- production never passes one, and a
# test pins that `evaluate` does not. The historical source hash is imported
# from Task 6, never restated.
TRUST_ANCHOR = {
    "artifact_root": ARTIFACT_ROOT,
    "historical_source_path": a6400.A6400_SOURCE,
    "historical_source_sha1": a6400.A6400_SOURCE_SHA1,
    "expected_cases": a6400.A6400_EXPECTED_CASES,
}


# ---------------------------------------------------------------------------
# The A/6,400 reference bundle. RECOMPUTED, never read.
# ---------------------------------------------------------------------------

def _resolve_within_root(raw_path: str, *, bundle_path: str, anchor: Dict,
                         label: str) -> Path:
    """Resolve a path the bundle names, and refuse anything outside the root.

    Relative paths resolve against the BUNDLE's own directory -- the bundle is
    written beside its captures -- so a bundle cannot be moved somewhere else
    and keep pointing at the same files by accident. `..` segments are resolved
    BEFORE the containment test, so traversal cannot escape.
    """
    root = Path(anchor["artifact_root"]).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(bundle_path).resolve().parent / candidate
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(
            f"bundle {label} resolves to {resolved}, outside the frozen "
            f"artifact root {root}")
    return resolved


def verify_a6400_bundle_bytes(raw: bytes, *, bundle_path: str,
                              anchor: Dict = None) -> Dict[str, Any]:
    """Verify the bundle BYTES, recomputing every claim they make.

    `raw` is what gets verified; `bundle_path` resolves the capture paths the
    bundle references and restricts them to the frozen artifact root. Nothing
    the document asserts about itself is believed: the two capture digests are
    recomputed from the live files, the byte-identity of the captures is
    established HERE rather than read from `byte_identical`, the historical
    source is reopened and hashed against Task 6's pin, and all thirty per-case
    authentications are recomputed from the reopened files.
    """
    anchor = TRUST_ANCHOR if anchor is None else anchor
    try:
        document = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        raise ValueError(f"bundle is a {type(document).__name__}, not an object")

    # (5) EXACT key set -- missing and extra both refused.
    expected_keys = set(a6400.A6400_BUNDLE_KEYS)
    missing = sorted(expected_keys - set(document))
    extra = sorted(set(document) - expected_keys)
    if missing or extra:
        raise ValueError(
            f"bundle key set is wrong: missing {missing}, unexpected {extra}")

    for field, value in (("artifact_kind", "v18_a6400_reference_bundle"),
                         ("schema_version", 1),
                         ("run_kind", "v18_a6400_reference_bundle")):
        if document[field] != value:
            raise ValueError(
                f"bundle {field} is {document[field]!r}, expected {value!r}")
    if document["scientific_interpretation_forbidden"] is not True:
        raise ValueError("bundle does not forbid scientific interpretation")

    run1 = _resolve_within_root(document["capture_run_1_path"],
                                bundle_path=bundle_path, anchor=anchor,
                                label="capture_run_1_path")
    run2 = _resolve_within_root(document["capture_run_2_path"],
                                bundle_path=bundle_path, anchor=anchor,
                                label="capture_run_2_path")
    source = _resolve_within_root(document["historical_source_path"],
                                  bundle_path=bundle_path, anchor=anchor,
                                  label="historical_source_path")
    # The frozen PATH, not merely a file that hashes correctly. Without this an
    # identical copy parked anywhere inside the root would pass, and the claim
    # that path substitution is rejected would be false.
    frozen_source = Path(anchor["historical_source_path"]).resolve()
    if source != frozen_source:
        raise ValueError(
            f"bundle names historical source {source}, the frozen source is "
            f"{frozen_source}")

    # (1) LIVE digests of both captures, against what the bundle recorded.
    raw1, raw2 = _read(run1, "capture_run_1"), _read(run2, "capture_run_2")
    for label, blob, claimed in (("capture_run_1", raw1, document["capture_run_1_sha1"]),
                                 ("capture_run_2", raw2, document["capture_run_2_sha1"])):
        live = hashlib.sha1(blob).hexdigest()
        if live != claimed:
            raise ValueError(
                f"{label} hashes {live}, the bundle records {claimed}: the file "
                f"changed after the bundle was written")

    # (2) Byte identity established HERE. `byte_identical` is not consulted --
    # reading it would let a forged `true` stand over two differing captures.
    # RAW bytes, exactly as the builder compares them: canonical-JSON equality
    # is weaker, and would accept two files the builder itself would refuse.
    if raw1 != raw2:
        raise ValueError(
            "the two captures are NOT byte-identical; a reference bundle over "
            "differing runs has no defined authentication block")
    doc1 = _parse_capture(raw1, "capture_run_1")
    _assert_capture_metadata(doc1)
    if document["byte_identical"] is not True:
        raise ValueError(
            f"bundle records byte_identical={document['byte_identical']!r}; the "
            f"builder emits a bundle only over identical runs")

    # (3) The historical source, reopened and hashed against Task 6's pin.
    raw_source = _read(source, "historical_source")
    live_source = hashlib.sha1(raw_source).hexdigest()
    if live_source != anchor["historical_source_sha1"]:
        raise ValueError(
            f"historical source hashes {live_source}, the frozen pin is "
            f"{anchor['historical_source_sha1']}: this is not the frozen A/6,400 "
            f"source, whatever its path claims")

    # (4) All thirty per-case authentications, RECOMPUTED from the reopened
    # files by Task 6's own reporter -- not read from the document.
    source_rows = a6400._enrich_source_rows(a6400._parse_source_rows(raw_source))
    if len(source_rows) != anchor["expected_cases"]:
        raise ValueError(
            f"historical source has {len(source_rows)} rows, expected "
            f"{anchor['expected_cases']}")
    recomputed = a6400.authentication_report(source_rows, doc1["cases"])
    if canonical_json_bytes(recomputed) != canonical_json_bytes(
            document["authentication"]):
        raise ValueError(
            "the bundle's authentication block is not what recomputing it from "
            "the captures produces: it was written, not derived")
    failed = [entry["case_id"] for entry in recomputed if not entry["ok"]]
    if failed:
        raise ValueError(
            f"{len(failed)} case(s) fail authentication against the frozen "
            f"source, e.g. {failed[:3]}")

    return {
        "artifact_kind": document["artifact_kind"],
        "schema_version": document["schema_version"],
        # RECOMPUTED digests, so a caller recording these records what was
        # verified rather than what was claimed.
        "capture_run_1_sha1": hashlib.sha1(raw1).hexdigest(),
        "capture_run_2_sha1": hashlib.sha1(raw2).hexdigest(),
        "capture_run_1_path": str(run1),
        "capture_run_2_path": str(run2),
        "byte_identical": True,
        "historical_source_sha1": live_source,
        "n_cases_authenticated": len(recomputed),
    }


def load_verified_a6400_bundle(path: str, *, anchor: Dict = None) -> Dict[str, Any]:
    """Convenience: read ONCE and delegate to the byte verifier."""
    return verify_a6400_bundle_bytes(Path(path).read_bytes(),
                                     bundle_path=path, anchor=anchor)


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"bundle {label} is unreadable at {path}: {exc}") from None


def _assert_capture_metadata(document: Dict) -> None:
    """The captures must be the RIGHT captures.

    Digests prove a file did not change; they say nothing about what it is. A
    bundle naming two identical captures taken at 400 simulations, or over the
    wrong gate, would otherwise authenticate perfectly.
    """
    mode = a6400.MODES["v18_preflight_a6400"]
    expected = {
        "run_kind": "v18_preflight_a6400",
        "mode": "v18_preflight_a6400",
        "mcts_sims": mode["mcts_sims"],
        "gate_list": list(mode["gates"]),
        "scientific_interpretation_forbidden": True,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"capture {field} is {document.get(field)!r}, the frozen A/6,400 "
                f"mode requires {value!r}")
    if len(document["cases"]) != a6400.A6400_EXPECTED_CASES:
        raise ValueError(
            f"capture carries {len(document['cases'])} cases, expected "
            f"{a6400.A6400_EXPECTED_CASES}")


def _parse_capture(raw: bytes, label: str) -> Dict:
    try:
        document = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from None
    if "cases" not in document:
        raise ValueError(f"{label} carries no `cases` block")
    return document


# ---------------------------------------------------------------------------
# Populations. Task 5 splits them deliberately; getting this wrong is the
# difference between a threshold fitted to its own sample and one that is not.
# ---------------------------------------------------------------------------

def prospective_target_subset(census_rows: Sequence[Dict],
                              cutoff: float) -> List[Dict]:
    """Broad non-A census rows at or above the cutoff.

    R_min and the revisit form read THIS, not the 30-row matched cohort, whose
    top exposure decile is about three rows.
    """
    for row in census_rows:
        if row.get("population") == "selected_a":
            raise ValueError(
                "an A row reached the prospective target subset: A rows "
                "establish reach and never a derived threshold")
    return [row for row in census_rows
            if criteria.meets_exposure_cutoff(
                row[criteria.PRIMARY_EXPOSURE_COLUMN], cutoff)]


def _cohort_rows(cohort_artifact: Dict, census_rows: Sequence[Dict]) -> List[Dict]:
    """The matched cohort resolved to census rows, with its cardinality checked.

    The cohort artifact drops every exposure column by design, so the join is
    mandatory -- and it is Task 8's, not a second one.
    """
    cohort = cohort_artifact.get("matched_cohort")
    if not isinstance(cohort, list):
        raise ValueError("cohort artifact carries no matched_cohort list")
    required = criteria.MATCHING["cardinality"]["n_a"]
    if len(cohort) != required:
        raise ValueError(
            f"matched cohort has {len(cohort)} rows, the frozen cardinality is "
            f"{required}: a smaller cohort is a PREFLIGHT_FAIL, never a cohort")
    return matched_control_rows(cohort, census_rows)


# ---------------------------------------------------------------------------
# The measured criteria.
# ---------------------------------------------------------------------------

def separation(cohort_artifact: Dict, a_rows: Sequence[Dict],
               census_rows: Sequence[Dict], *, replicates: int = None,
               criteria_module=criteria) -> Dict[str, Any]:
    """A versus the MATCHED cohort, on the single frozen primary statistic.

    The negative class is the matched cohort -- matching is what makes the two
    classes comparable row for row. `census_rows` is required because the cohort
    artifact deliberately carries no exposure column.
    """
    spec = criteria_module.SEPARATION
    controls = _cohort_rows(cohort_artifact, census_rows)
    if len(a_rows) != spec["n_a"]:
        raise ValueError(
            f"separation needs exactly {spec['n_a']} A rows, got {len(a_rows)}")
    statistic = spec["statistic"]
    positive = [row[statistic] for row in a_rows]
    negative = [row[statistic] for row in controls]

    auc = criteria_module.mann_whitney_auc(positive, negative)
    replicates = spec["bootstrap"]["replicates"] if replicates is None else replicates
    lower = criteria_module.auc_lower_bound(
        _bootstrap_auc_distribution(positive, negative, replicates,
                                    spec["bootstrap"]["seed"], criteria_module))
    return {
        "statistic": statistic,
        "auc": auc,
        "auc_lower_bound": lower,
        "min_auc": spec["min_auc"],
        "min_lower_bound": spec["min_lower_bound"],
        "n_a": len(positive),
        "n_c": len(negative),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": spec["bootstrap"]["seed"],
        # BOTH gate. A point estimate over the bar with a lower bound under it
        # has not established selectivity.
        "passes": auc >= spec["min_auc"] and lower >= spec["min_lower_bound"],
        "on_failure_means": spec["on_failure_means"],
        "on_failure_does_not_mean": spec["on_failure_does_not_mean"],
    }


def _bootstrap_auc_distribution(positive, negative, replicates, seed,
                                criteria_module) -> List[float]:
    """Stratified bootstrap: each class resampled to its OWN size.

    The per-replicate estimator is `criteria.mann_whitney_auc`, so the AUC has
    exactly one implementation. Only the resampling loop is here, and it is
    seeded from `SEPARATION["bootstrap"]["seed"]` -- deliberately NOT
    `criteria.bootstrap_aucs`, whose stream comes from the operating-
    characteristics model's own seed and would conflate a simulation with a
    measurement.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(replicates):
        a = [rng.choice(positive) for _ in positive]
        c = [rng.choice(negative) for _ in negative]
        out.append(criteria_module.mann_whitney_auc(a, c))
    return out


def decide_revisit_form(census_rows: Sequence[Dict], cutoff: float, *,
                        criteria_module=criteria) -> str:
    """`paired` or `candidate_only_floor`, over the prospective target subset."""
    spec = criteria_module.REVISIT_FORM_CRITERION
    subset = prospective_target_subset(census_rows, cutoff)
    counts = [row[f"would_clip_{spec['cap']}"] for row in subset]
    return criteria_module.revisit_form(counts)


def _pooled_by_cap(rows: Sequence[Dict], criteria_module) -> Dict[float, float]:
    """pooled(c) over the prospective target subset, at every grid cap.

    POOLED, never a mean of per-row ratios: `criteria.pooled_ratio` owns that
    distinction and is called rather than re-derived.
    """
    out = {}
    for cap in criteria_module.CAP_GRID:
        pairs = []
        for row in rows:
            table = row["crossover"][str(cap)]
            shipped = table["predicted_shipped_replies"]
            capped = table["predicted_capped_replies"]
            pairs.append((shipped - capped, shipped))
        out[cap] = criteria_module.pooled_ratio(pairs)
    return out


def derive_thresholds(census_rows: Sequence[Dict], cohort_artifact: Dict, *,
                      criteria_module=criteria) -> Dict[str, Any]:
    """EXPOSURE_CUTOFF from the matched cohort; R_min from the broad census.

    Raises when the band is empty: `R_min >= R_max` is unsatisfiable, and an
    unsatisfiable band is a failure, never a pass.
    """
    from .v18_selector_sizing import exposure_cutoff

    controls = _cohort_rows(cohort_artifact, census_rows)
    cutoff = exposure_cutoff(controls)
    subset = prospective_target_subset(census_rows, cutoff)

    rule = criteria_module.R_MIN_RULE
    if len(subset) < rule["subset_floor"]:
        raise ValueError(
            f"{rule['subset_below_floor_reason']}: the prospective target "
            f"subset holds {len(subset)} rows against a floor of "
            f"{rule['subset_floor']}")

    pooled = _pooled_by_cap(subset, criteria_module)
    # Raises the frozen reason when NO cap predicts positive conversion. The
    # floor never rescues a nonpositive prediction: `weakest_positive_cap`
    # raises first, so no such cap can ever define R_min.
    r_min = criteria_module.r_min_from_pooled(pooled)
    r_max = criteria_module.R_MAX_RULE["value"]
    if not criteria_module.r_band_is_satisfiable(r_min, r_max):
        raise ValueError(
            f"empty R band: R_min {r_min} >= R_max {r_max}. An empty band is "
            f"unsatisfiable, not a pass")
    return {
        "exposure_cutoff": cutoff,
        "prospective_target_subset_size": len(subset),
        "pooled_by_cap": {str(cap): value for cap, value in sorted(pooled.items())},
        "weakest_positive_cap": criteria_module.weakest_positive_cap(pooled),
        "r_min": r_min,
        "r_max": r_max,
        "r_min_floor": rule["floor"],
    }


# ---------------------------------------------------------------------------
# The verdict.
# ---------------------------------------------------------------------------

_REQUIRED_INPUT_SHA1S = ("criteria_sha1", "universe_sha1", "census_sha1",
                         "crossover_tables_sha1", "residual_rows_sha1",
                         "matched_cohort_sha1", "preflight_artifact_sha1")


def _read_and_hash(path: str, label: str):
    """ONE read. The bytes hashed and the bytes parsed are the same object.

    Hashing a path and then parsing a path is two reads, and a file changed
    between them would authenticate one byte sequence and evaluate another.
    """
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} at {path} is not valid JSON: {exc}") from None
    return payload, hashlib.sha1(raw).hexdigest()


def _agree(label: str, field: str, claimed, actual: str) -> None:
    if claimed != actual:
        raise ValueError(
            f"{label}.{field} is {claimed}, but the file it names hashes "
            f"{actual}: these documents do not describe one measurement")


CROSSOVER_COLUMNS = ("population", "case_id", "cap",
                     "predicted_shipped_replies", "predicted_capped_replies",
                     "predicted_reply_delta", "predicted_reply_reduction",
                     "excluded_terminal", "excluded_synthetic")


def _reproduce_census_bytes(cases: Sequence[Dict]) -> bytes:
    """Task 7's own serializer over the artifact's own cases.

    Comparing `artifact.census_positions_sha1` with the CSV proves the artifact
    contains the CSV's digest. It does NOT prove the `cases` the verdict
    evaluates came from those bytes -- exposure and crossover fields inside
    `cases` can be edited with the stored digest untouched. Rebuilding the CSV
    from the cases is what closes that.
    """
    from .diagnose_v18_residual_preflight import _csv_bytes
    return _csv_bytes(cases, criteria.CENSUS_SCHEMA)


def _reproduce_crossover_bytes(cases: Sequence[Dict]) -> bytes:
    """The nested per-case crossover tables, in Task 7's emission order.

    `R_min` is derived from these tables, so they are evidence and must be
    authenticated like any other.
    """
    from .diagnose_v18_residual_preflight import _csv_bytes
    rows = []
    for case in cases:
        # CAP_GRID order, not dict order. Task 7 builds the nested table as
        # `{str(cap): ... for cap in caps}` and emits in that order, but this
        # module reads the artifact back from canonical JSON, where sort_keys
        # has already reordered the keys to 0.5, 0.75, 1.0, 1.25. Iterating the
        # reloaded dict would reproduce a CSV Task 7 never wrote.
        for cap in criteria.CAP_GRID:
            table = case["crossover"][str(cap)]
            rows.append({
                "population": case["population"], "case_id": case["case_id"],
                "cap": cap,
                "predicted_shipped_replies": table["predicted_shipped_replies"],
                "predicted_capped_replies": table["predicted_capped_replies"],
                "predicted_reply_delta": table["predicted_reply_delta"],
                "predicted_reply_reduction": table["predicted_reply_reduction"],
                "excluded_terminal": sum(n["excluded_terminal"]
                                         for n in table["per_node"]),
                "excluded_synthetic": sum(n["excluded_synthetic"]
                                          for n in table["per_node"]),
            })
    return _csv_bytes(rows, CROSSOVER_COLUMNS)


RESIDUAL_COLUMNS = (
    ["population", "case_id", "game_idx", "position_ply", "side_to_move",
     "canonical_state_sha1", "raw_parent", "raw_leaf", "residual",
     "leaf_visit_count", "leaf_terminating_backups", "leaf_has_depth3_child"]
    + [f"{prefix}_{cap}" for cap in criteria.CAP_GRID
       for prefix in ("would_clip", "clipped_amount")])


def _reach_masses_from_residual_rows(rows: Sequence[Dict]) -> Dict[str, tuple]:
    """Per case, `(numerator, denominator)` recomputed from the LEAF evidence.

    `v18_tree_walk.exposed_positive_backup_mass`, expressed over the published
    rows: the weight is `terminating_backups * max(0, raw_leaf)` over every
    eligible leaf, and the numerator restricts it to leaves the STRONGEST cap
    pulled DOWN -- `clip_direction == +1`, which over these columns is
    `would_clip` at that cap together with a positive residual.

    This is the only reproduction that binds reach. The two mass fields appear
    in neither `CENSUS_SCHEMA` nor the crossover tables, so summing them out of
    `cases` and comparing with the artifact's own totals proves only that the
    artifact agrees with itself.
    """
    strongest = str(criteria.STRONGEST_CAP)
    out: Dict[str, list] = {}
    for row in rows:
        weight = (float(row["leaf_terminating_backups"])
                  * max(0.0, float(row["raw_leaf"])))
        entry = out.setdefault(row["case_id"], [0.0, 0.0])
        entry[1] += weight
        if int(row[f"would_clip_{strongest}"]) and float(row["residual"]) > 0:
            entry[0] += weight
    return {case_id: tuple(value) for case_id, value in out.items()}


def _reproduce_reach_masses(cases: Sequence[Dict],
                            residual_rows: Sequence[Dict]) -> None:
    """Every case's reach masses, against the residual rows that determine them."""
    recomputed = _reach_masses_from_residual_rows(residual_rows)
    for case in cases:
        numerator, denominator = recomputed.get(case["case_id"], (0.0, 0.0))
        for field, value in (("exposed_positive_mass_numerator", numerator),
                             ("exposed_positive_mass_denominator", denominator)):
            if not math.isclose(case[field], value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"case {case['case_id']} records {field}={case[field]}, "
                    f"recomputing it from residual_rows.csv gives {value}: the "
                    f"reach gate would run on unpublished leaf evidence")


def _assert_task7_envelope(artifact: Dict) -> None:
    """The artifact must describe the FROZEN measurement, not merely be
    well-formed.

    `run_kind` alone admits evidence produced at the wrong simulation budget,
    with noise on, through the batched route, against another checkpoint, or
    from a dirty tree -- every one of which changes the numbers the gates read.
    """
    from . import diagnose_v18_residual_preflight as task7

    expected = {
        "run_kind": task7.RUN_KIND,
        "scientific_interpretation_forbidden": True,
        "search_execution_mode": task7.SEARCH_EXECUTION_MODE,
        "simulations": task7.SIMULATIONS,
        "add_noise": task7.ADD_NOISE,
        "c_puct": task7.FROZEN_C_PUCT,
        "batching_triple": list(task7.BATCHING_TRIPLE),
        "cap_grid": list(criteria.CAP_GRID),
        "a_source_path": task7.A_SOURCE,
        "population_order": list(task7.POPULATIONS),
        # The bracket is what makes the runtime identity mean anything: HEAD and
        # the source tree are captured before the first search and re-checked
        # after the last.
        "runtime_identity_bracketed": True,
        "worktree_clean": True,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise ValueError(
                f"preflight artifact {field} is {artifact.get(field)!r}, the "
                f"frozen Task 7 envelope requires {value!r}")

    for field in ("a_source_sha1", "git_commit"):
        if not _SHA1_RE.match(str(artifact.get(field, ""))):
            raise ValueError(
                f"preflight artifact {field} is {artifact.get(field)!r}, not a "
                f"canonical sha1")
    for field in ("source_sha1s", "authenticated_search_inputs", "seed_audit"):
        if not artifact.get(field):
            raise ValueError(
                f"preflight artifact carries no {field}: the measurement's "
                f"authenticated state is unrecorded")


def _assert_provenance(artifact: Dict, *, criteria_payload: Dict,
                       universe_payload: Dict) -> None:
    """Every provenance value against its AUTHORITY, not its shape.

    Canonical-looking is not authentic. A wholly fabricated artifact -- an
    invented commit, an A-source hash that is not the frozen one, every module
    hash set to zeros, a made-up checkpoint identity and a self-declared seed
    audit -- satisfies a shape check completely.
    """
    from . import diagnose_v18_residual_preflight as task7
    from . import fpu_provenance
    from . import v18_control_pool as control_pool

    pinned_a = control_pool.FORBIDDEN_SOURCE_SHA1S["gate_A"]
    if artifact["a_source_sha1"] != pinned_a:
        raise ValueError(
            f"preflight artifact's a_source_sha1 is "
            f"{artifact['a_source_sha1']}, the frozen gate-A source is "
            f"{pinned_a}: these reach rows are not the frozen A rows")

    # HEAD. The artifact, the criteria record and the universe record must all
    # describe THIS commit, and each must have been emitted from a clean tree --
    # Task 7's own rule, called rather than restated.
    task7.assert_runtime_matches_records(
        ("preflight artifact", artifact), ("criteria", criteria_payload),
        ("universe", universe_payload),
        expected_commit=fpu_provenance.git_commit())

    # Exact keys AND exact values: a stale module hash names code that no longer
    # exists, which is the same problem as omitting it.
    live = {path: fpu_provenance.file_sha1(path)
            for path in task7.MEASUREMENT_SOURCE_MODULES}
    recorded = artifact["source_sha1s"]
    differing = sorted(set(live) | set(recorded)
                       if set(live) != set(recorded)
                       else (p for p in live if live[p] != recorded[p]))
    if differing:
        raise ValueError(
            f"preflight artifact's source_sha1s do not match the live "
            f"measurement modules; differing: {differing[:4]}")

    # Checkpoint and both reservoirs, RE-HASHED against their frozen pins.
    expected_inputs = task7._authenticate_search_inputs("opening")
    if canonical_json_bytes(artifact["authenticated_search_inputs"]) != \
            canonical_json_bytes(expected_inputs):
        raise ValueError(
            "preflight artifact's authenticated_search_inputs do not reproduce: "
            "the checkpoint or a replay reservoir is not the pinned one")

    # The seed audit is a CONCLUSION about the cases, so it is recomputed from
    # them rather than believed.
    expected_audit = task7.assert_seed_sets_disjoint(artifact["cases"])
    if canonical_json_bytes(artifact["seed_audit"]) != \
            canonical_json_bytes(expected_audit):
        raise ValueError(
            "preflight artifact's seed_audit does not reproduce from its own "
            "cases: the recorded disjointness is a claim, not a measurement")


def _read_csv_rows(raw: bytes) -> List[Dict]:
    """Parse published CSV bytes back into rows."""
    import csv
    import io
    return list(csv.DictReader(io.StringIO(raw.decode())))


def _recompute_aggregates(artifact: Dict, cases: Sequence[Dict]) -> None:
    """Pooled reach and the terminal totals are determined by the bound cases.

    Reading them from the artifact would let an edited total drive the verdict
    while every digest still matched.
    """
    expected = {
        "pooled_reach_numerator": sum(
            case["exposed_positive_mass_numerator"] for case in cases
            if case["population"] == "selected_a"),
        "pooled_reach_denominator": sum(
            case["exposed_positive_mass_denominator"] for case in cases
            if case["population"] == "selected_a"),
        "terminal_depth2_total": sum(case["terminal_depth2"] for case in cases),
        "total_depth2_total": sum(case["total_depth2"] for case in cases),
    }
    for field, value in expected.items():
        if not math.isclose(artifact[field], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"artifact records {field}={artifact[field]}, recomputing it "
                f"from its own cases gives {value}")


def _reproduce_cohort(cohort_document: Dict, census_rows: Sequence[Dict],
                      a_rows: Sequence[Dict]) -> None:
    """Re-run Task 4b's matcher and compare.

    Cross-binding proves the cohort is the one the other documents name; it does
    not prove it is the one the frozen matcher PRODUCES. A friendlier cohort
    with a regenerated report and updated digests satisfies every cross-check.
    """
    from .v18_cohort_matcher import _project, match_cohort

    cohort, report = match_cohort(census_rows, a_rows)
    if canonical_json_bytes([_project(row) for row in cohort]) != \
            canonical_json_bytes(cohort_document["matched_cohort"]):
        raise ValueError(
            "the matched cohort does not reproduce: re-running the frozen "
            "matcher over the verified census and A rows selects different "
            "rows, so this cohort was chosen rather than derived")
    if canonical_json_bytes(report) != \
            canonical_json_bytes(cohort_document["matching_report"]):
        raise ValueError(
            "the cohort's matching report does not reproduce from the frozen "
            "matcher")


def _reproduce_sizing(sizing: Dict, census_rows: Sequence[Dict],
                      cohort_document: Dict, universe: Dict) -> None:
    """Re-run Task 8's ladder and compare it with the record.

    Arithmetic reconciliation catches an incoherent ladder. It cannot catch a
    fabricated one whose derived fields are internally correct -- only
    re-running the selector over the verified evidence can.
    """
    from .v18_selector_sizing import (exposure_cutoff, matched_control_rows,
                                      recommended_operational_size,
                                      role_predicates, sizing_ladder,
                                      smallest_qualifying_tier)

    cutoff = exposure_cutoff(
        matched_control_rows(cohort_document["matched_cohort"], census_rows))
    ladder = sizing_ladder(census_rows, role_predicates(cutoff),
                           all_game_ids=universe["all_game_ids"])
    if canonical_json_bytes(ladder) != canonical_json_bytes(sizing["ladder"]):
        raise ValueError(
            "the sizing ladder does not reproduce: re-running Task 8 over the "
            "verified census, cutoff and universe yields a different ladder, so "
            "this record was written rather than measured")
    smallest = smallest_qualifying_tier(ladder)
    for field, value in (
            ("smallest_qualifying_tier", smallest),
            ("recommended_operational_size", recommended_operational_size(ladder)),
            ("exposure_cutoff", cutoff),
            # The status is re-derived HERE too, not only in `_sizing_state`:
            # the boundary is where a record stops being believed.
            ("sizing_status",
             "SIZING_PASSES" if smallest is not None else "SIZING_FAILS")):
        if field in sizing and sizing[field] != value:
            raise ValueError(
                f"sizing record's {field} is {sizing[field]!r}, re-running "
                f"Task 8 gives {value!r}")


def load_verified_inputs(*, preflight_artifact_path: str, census_path: str,
                         crossover_tables_path: str, residual_rows_path: str,
                         cohort_path: str, criteria_path: str,
                         universe_path: str,
                         sizing_path: str = None) -> Dict[str, Any]:
    """Read every input from disk, hash THOSE bytes, and cross-bind them.

    A digest handed in beside an object authenticates nothing -- whoever
    supplied the object could supply the digest. Every document here already
    records what it was built from, so the check that matters is that they all
    describe the SAME evidence:

        census file   == artifact.census_positions_sha1 == cohort.census_sha1
        criteria file == artifact.criteria_sha1         == cohort.criteria_sha1
        universe file == artifact.universe_sha1         == cohort.universe_sha1
        cohort file   == sizing.matched_cohort_sha1

    The criteria file is additionally RE-DERIVED from the committed module by
    Task 7's own loader rather than trusted.
    """
    from .diagnose_v18_residual_preflight import (load_verified_criteria,
                                                  load_verified_universe)

    artifact, artifact_sha1 = _read_and_hash(preflight_artifact_path, "preflight artifact")
    cohort, cohort_sha1 = _read_and_hash(cohort_path, "matched cohort")
    criteria_payload, criteria_sha1 = load_verified_criteria(criteria_path)
    # AUTHENTICATED, not merely hashed: this loader re-emits the record from the
    # authenticated source, because a coherently substituted universe reconciles
    # with itself perfectly.
    universe, universe_sha1 = load_verified_universe(universe_path)
    census_bytes = Path(census_path).read_bytes()
    census_sha1 = hashlib.sha1(census_bytes).hexdigest()
    crossover_bytes = Path(crossover_tables_path).read_bytes()
    crossover_sha1 = hashlib.sha1(crossover_bytes).hexdigest()
    residual_bytes = Path(residual_rows_path).read_bytes()
    residual_sha1 = hashlib.sha1(residual_bytes).hexdigest()

    _assert_task7_envelope(artifact)
    _assert_provenance(artifact, criteria_payload=criteria_payload,
                       universe_payload=universe)
    if cohort.get("run_kind") != "v18_matched_control_cohort":
        raise ValueError(
            f"cohort artifact has run_kind {cohort.get('run_kind')!r}")

    _agree("artifact", "census_positions_sha1",
           artifact.get("census_positions_sha1"), census_sha1)
    _agree("artifact", "crossover_tables_sha1",
           artifact.get("crossover_tables_sha1"), crossover_sha1)
    _agree("artifact", "residual_rows_sha1",
           artifact.get("residual_rows_sha1"), residual_sha1)
    _agree("artifact", "criteria_sha1", artifact.get("criteria_sha1"), criteria_sha1)
    _agree("artifact", "universe_sha1", artifact.get("universe_sha1"), universe_sha1)
    _agree("cohort", "census_sha1", cohort.get("census_sha1"), census_sha1)
    _agree("cohort", "criteria_sha1", cohort.get("criteria_sha1"), criteria_sha1)
    _agree("cohort", "universe_sha1", cohort.get("universe_sha1"), universe_sha1)

    sizing = sizing_sha1 = None
    if sizing_path is not None:
        sizing, sizing_sha1 = _read_and_hash(sizing_path, "sizing record")
        if sizing.get("run_kind") != "v18_preflight_sizing":
            raise ValueError(
                f"sizing record has run_kind {sizing.get('run_kind')!r}")
        _agree("sizing", "census_sha1", sizing.get("census_sha1"), census_sha1)
        _agree("sizing", "criteria_sha1", sizing.get("criteria_sha1"), criteria_sha1)
        _agree("sizing", "universe_sha1", sizing.get("universe_sha1"), universe_sha1)
        _agree("sizing", "matched_cohort_sha1",
               sizing.get("matched_cohort_sha1"), cohort_sha1)

    # DERIVED from the verified artifact, never supplied. A caller who could
    # hand in row lists could hand in any rows at all.
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise ValueError("preflight artifact carries no `cases` list")
    a_rows = [row for row in cases if row.get("population") == "selected_a"]
    census_rows = [row for row in cases if row.get("population") == "census"]
    if not a_rows or not census_rows:
        raise ValueError(
            f"artifact yields {len(a_rows)} A rows and {len(census_rows)} census "
            f"rows; both populations are required")

    # REPRODUCTION, not agreement. Everything above establishes that these
    # documents name one another consistently; everything below establishes
    # that they are what the committed code produces from the evidence.
    if _reproduce_census_bytes(cases) != census_bytes:
        raise ValueError(
            "the census CSV does not reproduce from the artifact's own cases: "
            "the rows this verdict would evaluate are not the rows that were "
            "published")
    if _reproduce_crossover_bytes(cases) != crossover_bytes:
        raise ValueError(
            "the crossover tables do not reproduce from the artifact's own "
            "cases: R_min would be derived from unpublished predictions")
    _reproduce_reach_masses(cases, _read_csv_rows(residual_bytes))
    _recompute_aggregates(artifact, cases)
    _reproduce_cohort(cohort, census_rows, a_rows)
    if sizing is not None:
        _reproduce_sizing(sizing, census_rows, cohort, universe)

    return {
        "artifact": artifact, "cohort": cohort, "sizing": sizing,
        "universe": universe, "a_rows": a_rows, "census_rows": census_rows,
        "sha1s": {
            "preflight_artifact_sha1": artifact_sha1,
            "matched_cohort_sha1": cohort_sha1,
            "criteria_sha1": criteria_sha1,
            "universe_sha1": universe_sha1,
            "census_sha1": census_sha1,
            "crossover_tables_sha1": crossover_sha1,
            "residual_rows_sha1": residual_sha1,
            **({"sizing_sha1": sizing_sha1} if sizing_sha1 else {}),
        },
    }


def evaluate(*, preflight_artifact_path: str, census_path: str,
             crossover_tables_path: str, residual_rows_path: str,
             cohort_path: str, criteria_path: str, universe_path: str,
             a_reference_bundle_path: str,
             sizing_path: str = None) -> Dict[str, Any]:
    """PRODUCTION entry point. Paths in, one verdict out. Writes nothing.

    Deliberately parameterless beyond its inputs: the committed criteria, the
    frozen 10,000 bootstrap replicates and the frozen A-reference trust anchor
    are not negotiable by a caller. A production API that let one of them be
    passed would let a caller obtain PREFLIGHT_PASS by lowering the bar.
    """
    verified = load_verified_inputs(
        preflight_artifact_path=preflight_artifact_path, census_path=census_path,
        crossover_tables_path=crossover_tables_path,
        residual_rows_path=residual_rows_path, cohort_path=cohort_path,
        criteria_path=criteria_path, universe_path=universe_path,
        sizing_path=sizing_path)
    return _evaluate_verified(verified, a_reference_bundle_path)


def _evaluate_verified(verified: Dict[str, Any], a_reference_bundle_path: str, *,
                       separation_replicates: int = None,
                       criteria_module=criteria,
                       anchor: Dict = None) -> Dict[str, Any]:
    """The pure evaluator over ALREADY-VERIFIED inputs.

    Private, and the overrides are why: they exist so fixture tests can run a
    short bootstrap and a fixture trust anchor. `evaluate` passes none of them.
    """
    artifact = verified["artifact"]
    cohort_artifact = verified["cohort"]
    census_rows = verified["census_rows"]
    a_rows = verified["a_rows"]
    input_sha1s = verified["sha1s"]

    missing = sorted(set(_REQUIRED_INPUT_SHA1S) - set(input_sha1s))
    if missing:
        raise ValueError(f"missing input sha1(s): {missing}")

    raw = Path(a_reference_bundle_path).read_bytes()          # 1. the ONLY read
    bundle_sha1 = hashlib.sha1(raw).hexdigest()               # 2. those bytes
    verified_bundle = verify_a6400_bundle_bytes(              # 3. the SAME bytes
        raw, bundle_path=a_reference_bundle_path, anchor=anchor)

    failures: List[str] = []
    thresholds = None
    try:
        thresholds = derive_thresholds(census_rows, cohort_artifact,
                                       criteria_module=criteria_module)
    except ValueError as exc:
        failures.append(str(exc).splitlines()[0])

    sep = separation(cohort_artifact, a_rows, census_rows,
                     replicates=separation_replicates,
                     criteria_module=criteria_module)
    if not sep["passes"]:
        failures.append(f"separation: {sep['on_failure_means']}")

    reach = _reach(artifact, criteria_module)
    if not reach["passes"]:
        failures.append(f"reach: pooled {reach['value']} below "
                        f"{reach['min']} (A rows establish reach only)")

    terminal = _terminal_fraction(artifact, criteria_module)
    if not terminal["passes"]:
        failures.append(f"terminal_fraction: {terminal['value']} above "
                        f"{terminal['max']}")

    dominance = _sign_dominance(a_rows, criteria_module)
    if not dominance["passes"]:
        failures.append(f"sign_dominance: {dominance['value']} below "
                        f"{dominance['min']}")

    form = None
    if thresholds is not None:
        form = decide_revisit_form(census_rows, thresholds["exposure_cutoff"],
                                   criteria_module=criteria_module)

    sizing = _sizing_state(verified["sizing"], input_sha1s)
    if failures:
        verdict = PREFLIGHT_FAIL
    elif sizing["state"] == "absent":
        verdict = PROVISIONAL_PASS
    elif sizing["state"] == "failed":
        verdict = PREFLIGHT_FAIL
        failures.append("sizing: the frozen four-role geometry was not "
                        "satisfiable at any tier")
    else:
        verdict = PREFLIGHT_PASS

    return {
        "run_kind": RUN_KIND,
        "scientific_interpretation_forbidden": True,
        "scope_boundary": criteria_module.SCOPE_BOUNDARY,
        "verdict": verdict,
        "failures": failures,
        "thresholds": thresholds,
        "separation": sep,
        "reach": reach,
        "terminal_fraction": terminal,
        "sign_dominance": dominance,
        "revisit_form": form,
        # Execution Step 6 reports the efficiency floor alongside R_min, R_max,
        # the cutoff and the revisit form. These are NORMATIVE requirements on
        # the future candidate runs -- they do not gate preflight data, and the
        # block says so rather than leaving a reader to assume it.
        "derived_guards": {
            "conversion_efficiency_min": criteria_module.CONVERSION_EFFICIENCY_MIN["value"],
            "conversion_efficiency_basis":
                criteria_module.CONVERSION_EFFICIENCY_MIN["basis"],
            "min_lost_replies": dict(criteria_module.MIN_LOST_REPLIES),
            "stable_leader_min_fraction":
                criteria_module.STABLE_LEADER_MIN_FRACTION,
            "applies_to": "the future candidate runs, not this preflight",
        },
        "sizing": sizing,
        "a6400_reference_bundle_sha1": bundle_sha1,
        "a6400_capture_run_1_sha1": verified_bundle["capture_run_1_sha1"],
        "a6400_capture_run_2_sha1": verified_bundle["capture_run_2_sha1"],
        "a6400_historical_source_sha1": verified_bundle["historical_source_sha1"],
        "a6400_cases_authenticated": verified_bundle["n_cases_authenticated"],
        **{name: input_sha1s[name] for name in _REQUIRED_INPUT_SHA1S},
        **({"sizing_sha1": input_sha1s["sizing_sha1"]}
           if "sizing_sha1" in input_sha1s else {}),
    }


def _reach(artifact: Dict, criteria_module) -> Dict[str, Any]:
    """Pooled over A ROWS ONLY -- reach is what A establishes, and the only
    thing it establishes."""
    spec = criteria_module.REACH
    numerator = artifact["pooled_reach_numerator"]
    denominator = artifact["pooled_reach_denominator"]
    if denominator == 0:
        return {"value": None, "min": spec["min"], "passes": False,
                "population": spec["population"],
                "note": spec["on_zero_denominator"]}
    value = numerator / denominator
    return {"value": value, "min": spec["min"], "passes": value >= spec["min"],
            "population": spec["population"], "aggregation": spec["aggregation"]}


def _terminal_fraction(artifact: Dict, criteria_module) -> Dict[str, Any]:
    spec = criteria_module.TERMINAL_FRACTION
    total = artifact["total_depth2_total"]
    if total == 0:
        return {"value": None, "max": spec["max"], "passes": False}
    value = artifact["terminal_depth2_total"] / total
    return {"value": value, "max": spec["max"], "passes": value <= spec["max"]}


def _sign_dominance(a_rows: Sequence[Dict], criteria_module) -> Dict[str, Any]:
    """Pooled, using the frozen formula's own zero-denominator rule."""
    spec = criteria_module.SIGN_DOMINANCE
    positive = sum(row["positive_mass"] for row in a_rows)
    negative = sum(row["negative_mass"] for row in a_rows)
    if positive + negative == 0:
        value = spec["on_zero_denominator"]
    else:
        value = positive / (positive + negative)
    return {"value": value, "min": spec["min"], "passes": value >= spec["min"]}


def _sizing_state(sizing_record, input_sha1s: Dict[str, str]) -> Dict[str, Any]:
    """Absent and FAILED are different states, and conflating them would mean an
    infeasible selector could never formally reject v18.

    The record is RECONCILED, not read: `sizing_status` is a claim, and a
    one-field edit from SIZING_FAILS to SIZING_PASSES would otherwise flip the
    final verdict. Task 8's own ladder reconciliation is re-run over it, and the
    smallest qualifying tier, the recommended size and the status are all
    re-derived here and compared with what the document says.
    """
    from .v18_selector_sizing import (_reconcile_ladder,
                                      recommended_operational_size,
                                      smallest_qualifying_tier)

    if sizing_record is None:
        return {"state": "absent",
                "note": "sizing has not run; this authorizes nothing"}
    if "sizing_sha1" not in input_sha1s:
        raise ValueError(
            "a sizing record was supplied without a digest: it was not read "
            "through the verified-input boundary")

    ladder = sizing_record.get("ladder")
    if not isinstance(ladder, list):
        raise ValueError("sizing record carries no ladder")
    skipped = [tier["n_games"] for tier in ladder if "skipped" in tier]
    if skipped:
        # A tier larger than the universe. `UNIVERSE["on_insufficient_games"]`
        # is STOP, so this cannot happen on a legitimate 800-game run -- and
        # letting it reach `_reconcile_ladder` produces a bare KeyError instead
        # of saying what is wrong.
        raise ValueError(
            f"sizing ladder skipped tier(s) {skipped}: the universe was too "
            f"small for the frozen ladder, which is a STOP condition")
    _reconcile_ladder(ladder)                     # Task 8's own arithmetic

    smallest = smallest_qualifying_tier(ladder)
    recommended = recommended_operational_size(ladder)
    expected_status = "SIZING_PASSES" if smallest is not None else "SIZING_FAILS"
    claimed = sizing_record.get("sizing_status")
    if claimed != expected_status:
        raise ValueError(
            f"sizing record claims {claimed!r}, but re-deriving it from its own "
            f"ladder gives {expected_status!r}")
    for field, value in (("smallest_qualifying_tier", smallest),
                         ("recommended_operational_size", recommended)):
        if sizing_record.get(field) != value:
            raise ValueError(
                f"sizing record's {field} is {sizing_record.get(field)!r}, "
                f"re-deriving it from the ladder gives {value!r}")

    if smallest is None:
        return {"state": "failed", "smallest_qualifying_tier": None,
                "sizing_sha1": input_sha1s["sizing_sha1"]}
    return {"state": "passed", "smallest_qualifying_tier": smallest,
            "recommended_operational_size": recommended,
            "sizing_sha1": input_sha1s["sizing_sha1"]}
