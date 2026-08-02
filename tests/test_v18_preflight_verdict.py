"""Task 9 -- the preflight verdict evaluator and the A/6,400 bundle verifier.

The bundle tests are ATTACKS. Each one writes a bundle that is internally
coherent and would pass any verifier that reads its fields, and each must be
caught by RECOMPUTATION from the files the bundle names.

Populations are the other axis under test: separation and the exposure cutoff
read the 30-row matched cohort; R_min and the revisit form read the broad non-A
census. Swapping them is not a style difference -- a threshold derived from the
cohort's three-row top decile is fitted to its own sample.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from scripts.GPU.alphazero import capture_v18_a6400 as a6400
from scripts.GPU.alphazero import v18_preflight_criteria as criteria
from scripts.GPU.alphazero import v18_preflight_verdict as M

FIXTURE_DIR = Path(__file__).parent / "golden" / "a6400_bundle_fixture"
CAPS = criteria.CAP_GRID
EXPOSURE = criteria.PRIMARY_EXPOSURE_COLUMN

# Small enough to keep the suite fast; one test pins that the frozen default is
# the criteria module's 10,000.
FAST_REPLICATES = 200

# The bootstrap lower bound this fixture yields under SEPARATION's frozen seed.
# Pinned so the seed is load-bearing: any other seed moves this number.
BOOTSTRAP_LOWER_BOUND_AT_FROZEN_SEED = 0.8177500000000001


# --- a census that REPRODUCES ------------------------------------------------
#
# Revision 38 requires the census CSV, the crossover tables, the cohort and the
# sizing ladder to be REPRODUCED from committed code, so the fixture cannot be
# hand-written any more: it is generated, and every derived document is built by
# the real producer.

PHASES = ("opening", "early_mid", "midgame", "late")
PHASE_BASE = {"opening": 10, "early_mid": 40, "midgame": 70, "late": 100}

# (abs root value, would_clip at every cap, exposure). The cohort-derived cutoff
# must land BETWEEN two exposure bands: the bait spreads 1..64 so its 0.90
# quantile is 53, the target cell sits at 500 and everything that must NOT be a
# target sits at 5. Getting this backwards makes every row a target and starves
# the representative quota -- which is how the fixture was first wrong.
ROLE_PROFILE = {"bait": (0.1, 0, None), "target": (0.1, 0, 500.0),
                "identity": (0.5, 0, 5.0), "flip": (0.5, 6, 5.0),
                "representative": (0.1, 0, 5.0)}

CENSUS_PLAN = ([("bait", "late", 32), ("target", "late", 18),
                ("identity", "late", 12), ("flip", "late", 12)]
               + [("representative", phase, 12) for phase in PHASES])

# Scaled ladder for the sizing-REPRODUCTION tests. The frozen 200..800 x 299
# would need an 800-game fixture and thousands of selector runs per test; the
# code path is identical and `test_the_production_sizing_parameters_are_frozen`
# pins the real values. 9 all-success gives 0.7169 and 8 of 9 gives 0.5709, so a
# floor of 0.70 keeps "every trial succeeded" necessary -- which is what makes
# trial 0 a witness.
SCALED_SIZING = {"probabilistic_tiers": (60, 70), "degenerate_tier": 122,
                 "trials_per_probabilistic_tier": 9,
                 "minimum_lower_bound": 0.70}


def case(idx, *, population, phase, ply, exposure, would_clip, stm, game,
         stm_abs=0.1, shipped=100.0, capped=80.0, positive=9.0, negative=1.0,
         leaves=None, terminal=1, total=40,
         capped_by_cap=None):
    """One artifact case carrying the FULL census schema.

    Every `CENSUS_SCHEMA` column must be present, because the boundary rebuilds
    the census CSV from these rows and byte-compares it.
    """
    row = {
        "population": population, "case_id": f"{population}:{idx}",
        "source_universe_ordinal": idx,
        "game_content_sha1": f"{game:040x}", "game_idx": game,
        "position_ply": ply, "side_to_move": stm,
        "canonical_state_sha1": f"{idx:040x}", "phase": phase,
        "root_value_stm": stm_abs, "n_legal": 400,
        "eligible_depth2_leaves": 60, "replies": 40, "explored_replies": 30,
        "depth_ge3_backups": 10, "depth_ge3_fraction": 0.25,
        "follow_up_visits_per_reply": 1.5, "positive_mass": positive,
        "negative_mass": negative, "sign_dominance": 0.9,
        "terminal_depth2": terminal, "total_depth2": total,
        EXPOSURE: exposure, "exposure_descriptive_count": would_clip,
        "exposure_descriptive_clipped_mass": 0.7,
        # Placeholders; overwritten below from the leaves themselves.
        "exposed_positive_mass_numerator": 0.0,
        "exposed_positive_mass_denominator": 0.0,
        "seed": 1000 + idx,
    }
    for cap in CAPS:
        row[f"would_clip_{cap}"] = would_clip
        row[f"clipped_amount_{cap}"] = 0.7
        row[f"revisit_to_depth3_rate_{cap}"] = 0.5
    # The LEAF evidence reach is derived from. Two leaves: one the strongest cap
    # pulls down (positive residual above 0.50) and one it does not, so the
    # numerator is a strict subset of the denominator.
    row["residual_leaves"] = [
        {"population": population, "case_id": f"{population}:{idx}",
         "game_idx": game, "position_ply": ply, "side_to_move": stm,
         "canonical_state_sha1": f"{idx:040x}",
         "raw_parent": 0.1, "raw_leaf": leaf_value, "residual": residual,
         "leaf_visit_count": 5, "leaf_terminating_backups": backups,
         "leaf_has_depth3_child": False}
        for leaf_value, residual, backups in (leaves or _RESIDUAL_LEAVES)]
    for leaf in row["residual_leaves"]:
        for cap in CAPS:
            binds = abs(leaf["residual"]) > cap
            leaf[f"would_clip_{cap}"] = int(binds)
            leaf[f"clipped_amount_{cap}"] = (abs(leaf["residual"]) - cap
                                             if binds else 0.0)
    row["crossover"] = {
        str(cap): {"predicted_shipped_replies": shipped,
                   "predicted_capped_replies": (capped_by_cap or {}).get(cap, capped),
                   "predicted_reply_delta":
                       shipped - (capped_by_cap or {}).get(cap, capped),
                   "predicted_reply_reduction":
                       (shipped - (capped_by_cap or {}).get(cap, capped)) / shipped,
                   "per_node": [{"excluded_terminal": 0, "excluded_synthetic": 0}]}
        for cap in CAPS}
    strongest = str(criteria.STRONGEST_CAP)
    weights = [(leaf[f"would_clip_{strongest}"] and leaf["residual"] > 0,
                leaf["leaf_terminating_backups"] * max(0.0, leaf["raw_leaf"]))
               for leaf in row["residual_leaves"]]
    row["exposed_positive_mass_numerator"] = sum(w for hit, w in weights if hit)
    row["exposed_positive_mass_denominator"] = sum(w for _hit, w in weights)
    return row


# (raw_leaf, residual, terminating_backups). The first clips at 0.50, the
# second does not, so numerator < denominator and reach is a real ratio: 9/14.
_RESIDUAL_LEAVES = ((0.9, 0.8, 10), (0.5, 0.2, 10))

# No leaf clips at the strongest cap, so the numerator is zero and reach fails.
# Reach can only be failed by changing the LEAF EVIDENCE now -- which is exactly
# what binding residual_rows.csv buys.
_UNREACHED_LEAVES = ((0.9, 0.2, 10), (0.5, 0.1, 10))


def census(would_clip=None, **over):
    """Bait games FIRST, so their low canonical hashes win the matcher's
    zero-cost tie-break and the cohort lands on the low-exposure band."""
    rows, idx, game = [], 0, 0
    for role, phase, n_games in CENSUS_PLAN:
        stm_abs, clip, exposure = ROLE_PROFILE[role]
        for _ in range(n_games):
            game += 1
            for offset, stm in ((0, "red"), (13, "black")):
                idx += 1
                rows.append(case(idx, population="census", phase=phase,
                                 ply=PHASE_BASE[phase] + offset,
                                 exposure=float(idx) if exposure is None
                                 else exposure,
                                 would_clip=(clip if would_clip is None
                                             else would_clip),
                                 stm=stm, game=game, stm_abs=stm_abs, **over))
    return rows


def separation_classes(rows=None, a_list=None):
    """The two classes separation actually compares: A rows against the MATCHED
    COHORT -- not the first 30 census rows, which are disjoint from A and would
    make the bootstrap degenerate."""
    from scripts.GPU.alphazero import v18_selector_sizing as S
    rows = census() if rows is None else rows
    a_list = a_rows() if a_list is None else a_list
    controls = S.matched_control_rows(
        cohort_for(rows, a_list)["matched_cohort"], rows)
    return ([r[EXPOSURE] for r in a_list], [r[EXPOSURE] for r in controls])


def cohort_for(rows=None, a_list=None):
    """The matched-cohort DOCUMENT, produced by Task 4b's own matcher."""
    from scripts.GPU.alphazero import v18_cohort_matcher as CM
    rows = census() if rows is None else rows
    a_list = a_rows() if a_list is None else a_list
    cohort, report = CM.match_cohort(rows, a_list)
    return {"run_kind": "v18_matched_control_cohort",
            "matched_cohort": [CM._project(r) for r in cohort],
            "matching_report": report}


def aggregates(cases):
    """The four artifact aggregates, recomputed from cases exactly as the
    boundary recomputes them."""
    a_only = [c for c in cases if c["population"] == "selected_a"]
    return {"pooled_reach_numerator": sum(c["exposed_positive_mass_numerator"]
                                          for c in a_only),
            "pooled_reach_denominator": sum(c["exposed_positive_mass_denominator"]
                                            for c in a_only),
            "terminal_depth2_total": sum(c["terminal_depth2"] for c in cases),
            "total_depth2_total": sum(c["total_depth2"] for c in cases)}


def starved_census():
    """A census with no flip-control cell. The ladder still reproduces -- it
    reproduces as a FAILURE, which is what a real infeasible geometry looks
    like."""
    rows = []
    for row in census():
        if row["root_value_stm"] == 0.5 and row["would_clip_1.25"] == 6:
            row = dict(row)
            for cap in CAPS:                       # no longer flip-eligible
                row[f"would_clip_{cap}"] = 0
        rows.append(row)
    return rows


def a_rows(n=30, exposure=40.0, **over):
    """Late and side-alternating, so all 30 match distinct bait games.

    The exposure band OVERLAPS the matched cohort's 1..60: disjoint classes give
    AUC exactly 1.0 and a zero-spread bootstrap, which would make the
    lower-bound gate vacuous. At base 40 the AUC is 0.8772 and the bootstrap
    lower bound 0.8178 -- both above their floors, both genuinely computed.
    Matching never reads exposure (it is a FORBIDDEN field), so this costs the
    cohort nothing.
    """
    return [case(9000 + i, population="selected_a", phase="late",
                 ply=_A_PLIES[i], exposure=exposure + i, would_clip=0,
                 stm=("red" if i % 2 == 0 else "black"), game=9000 + i, **over)
            for i in range(n)]


def _a_plies():
    """Plies giving the FROZEN selected-A seed shape: 30 rows, 27 unique seeds,
    3 duplicate groups.

    The A seed rule is the historical XOR `base ^ game_idx ^ position_ply`, and
    `SEED_POLICY` records those exact counts as accepted provenance. Task 9
    recomputes the seed audit from the cases, so a fixture whose seeds have any
    other shape is refused -- correctly.
    """
    plies = {i: PHASE_BASE["late"] for i in range(30)}
    for first, second in ((0, 1), (2, 3), (4, 5)):
        plies[second] = (9000 + second) ^ ((9000 + first) ^ plies[first])
    return plies


_A_PLIES = _a_plies()


# --- the A/6,400 bundle fixture tree -----------------------------------------

@pytest.fixture
def bundle_tree(tmp_path):
    """A complete, VALID bundle plus its captures and historical source.

    The source CSV is written from the tracked 30-case fixture, so the whole
    chain is reproducible in a clean checkout -- the real A/6,400 source lives
    under gitignored logs/ and a suite test may not depend on it.
    """
    rows = json.loads((FIXTURE_DIR / "source_rows.json").read_text())
    capture = json.loads((FIXTURE_DIR / "capture.json").read_text())
    assert len(rows) == 30 and len(capture["cases"]) == 30
    # Task 6's tracked fixture is REDUCED -- it carries the cases and little
    # else. Task 9 revalidates the capture's own metadata, which a real
    # `capture()` document carries, so the frozen values are restored here from
    # Task 6's constants rather than invented.
    mode = a6400.MODES["v18_preflight_a6400"]
    capture.update({"run_kind": "v18_preflight_a6400",
                    "mode": "v18_preflight_a6400",
                    "mcts_sims": mode["mcts_sims"],
                    "gate_list": list(mode["gates"])})
    assert capture["mcts_sims"] == 6400 and capture["gate_list"] == ["A"]

    columns = sorted({key for row in rows for key in row})
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    source_csv = ("\n".join(lines) + "\n").encode()

    root = tmp_path / "eval"
    root.mkdir()
    source_path = root / "position_probe_cases.csv"
    source_path.write_bytes(source_csv)

    raw_capture = json.dumps(capture, sort_keys=True).encode()
    (root / "run1.json").write_bytes(raw_capture)
    (root / "run2.json").write_bytes(raw_capture)

    anchor = {"artifact_root": str(root),
              "historical_source_path": str(source_path),
              "historical_source_sha1": hashlib.sha1(source_csv).hexdigest(),
              "expected_cases": 30}

    report = a6400.authentication_report(
        a6400._enrich_source_rows(a6400._parse_source_rows(source_csv)),
        capture["cases"])
    assert all(entry["ok"] for entry in report), "fixture must authenticate"

    document = {
        "artifact_kind": "v18_a6400_reference_bundle",
        "schema_version": 1,
        "capture_run_1_path": "run1.json",
        "capture_run_1_sha1": hashlib.sha1(raw_capture).hexdigest(),
        "capture_run_2_path": "run2.json",
        "capture_run_2_sha1": hashlib.sha1(raw_capture).hexdigest(),
        "byte_identical": True,
        "historical_source_path": "position_probe_cases.csv",
        "historical_source_sha1": anchor["historical_source_sha1"],
        "authentication": report,
        "run_kind": "v18_a6400_reference_bundle",
        "scientific_interpretation_forbidden": True,
    }
    path = root / "a6400_reference_bundle.json"
    path.write_bytes(json.dumps(document, sort_keys=True).encode())
    return {"root": root, "path": str(path), "document": document,
            "anchor": anchor, "capture": capture}


def rewrite(tree, **changes):
    """Rewrite the bundle with `changes` applied, and return its path."""
    document = dict(tree["document"])
    document.update(changes)
    Path(tree["path"]).write_bytes(json.dumps(document, sort_keys=True).encode())
    return tree["path"]


def test_the_bundle_fixture_is_accepted_as_written(bundle_tree):
    """Guard the guard: every attack below must fail for its OWN reason."""
    verified = M.load_verified_a6400_bundle(bundle_tree["path"],
                                            anchor=bundle_tree["anchor"])
    assert verified["n_cases_authenticated"] == 30
    assert verified["byte_identical"] is True


# --- the five tamper attacks -------------------------------------------------

def test_attack_forged_byte_identical_true_over_differing_captures(bundle_tree):
    """The captures genuinely differ and BOTH recorded digests are correct, so
    only recomputing the comparison catches it."""
    root = bundle_tree["root"]
    other = json.loads(json.dumps(bundle_tree["capture"]))
    other["cases"][0]["top_share_repr"] = "0.99"
    raw = json.dumps(other, sort_keys=True).encode()
    (root / "run2.json").write_bytes(raw)
    path = rewrite(bundle_tree, capture_run_2_sha1=hashlib.sha1(raw).hexdigest(),
                   byte_identical=True)
    with pytest.raises(ValueError, match="NOT byte-identical"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_attack_capture_altered_after_hashing(bundle_tree):
    """The bundle's stored digest no longer matches the live file."""
    altered = json.loads(json.dumps(bundle_tree["capture"]))
    altered["cases"][0]["recomputed_black_value_repr"] = "0.5"
    for name in ("run1.json", "run2.json"):
        (bundle_tree["root"] / name).write_bytes(
            json.dumps(altered, sort_keys=True).encode())
    with pytest.raises(ValueError, match="changed after the bundle was written"):
        M.load_verified_a6400_bundle(bundle_tree["path"],
                                     anchor=bundle_tree["anchor"])


def test_attack_fabricated_per_case_authentication_block(bundle_tree):
    """Every case marked `ok`, but recomputation disagrees."""
    forged = [dict(entry, ok=True, abs_value_delta_repr="0.0")
              for entry in bundle_tree["document"]["authentication"]]
    forged[0]["captured_black_value_repr"] = "0.123456"
    path = rewrite(bundle_tree, authentication=forged)
    with pytest.raises(ValueError, match="written, not derived"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_attack_substituted_historical_source_path(bundle_tree):
    """A look-alike whose hash is not the frozen pin."""
    frozen = Path(bundle_tree["anchor"]["historical_source_path"])
    original = frozen.read_bytes()

    # (a) a look-alike at ANOTHER path -- refused on the path, before content.
    lookalike = bundle_tree["root"] / "position_probe_cases_v2.csv"
    lookalike.write_bytes(original)
    path = rewrite(bundle_tree, historical_source_path="position_probe_cases_v2.csv")
    with pytest.raises(ValueError, match="the frozen source is"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])

    # (b) the frozen path, substituted CONTENT -- refused on the hash.
    rewrite(bundle_tree)
    frozen.write_bytes(original + b"# trailing\n")
    try:
        with pytest.raises(ValueError, match="not the frozen A/6,400 source"):
            M.load_verified_a6400_bundle(bundle_tree["path"],
                                         anchor=bundle_tree["anchor"])
    finally:
        frozen.write_bytes(original)


@pytest.mark.parametrize("mutate, err", [
    (lambda d: d.pop("byte_identical"), "missing"),
    (lambda d: d.pop("authentication"), "missing"),
    (lambda d: d.update({"extra_field": 1}), "unexpected"),
    (lambda d: d.update({"note": "helpful"}), "unexpected"),
])
def test_attack_missing_or_extra_keys_rejected(bundle_tree, mutate, err):
    document = dict(bundle_tree["document"])
    mutate(document)
    Path(bundle_tree["path"]).write_bytes(
        json.dumps(document, sort_keys=True).encode())
    with pytest.raises(ValueError, match=f"key set is wrong.*{err}"):
        M.load_verified_a6400_bundle(bundle_tree["path"],
                                     anchor=bundle_tree["anchor"])


# --- other bundle refusals ---------------------------------------------------

def test_a_capture_outside_the_artifact_root_is_refused(bundle_tree, tmp_path):
    outside = tmp_path / "elsewhere.json"
    outside.write_bytes(Path(bundle_tree["root"] / "run1.json").read_bytes())
    path = rewrite(bundle_tree, capture_run_1_path=str(outside))
    with pytest.raises(ValueError, match="outside the frozen artifact root"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_path_traversal_out_of_the_root_is_refused(bundle_tree):
    path = rewrite(bundle_tree, capture_run_1_path="../../etc/passwd")
    with pytest.raises(ValueError, match="outside the frozen artifact root"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_a_relabelled_bundle_is_refused(bundle_tree):
    path = rewrite(bundle_tree, scientific_interpretation_forbidden=False)
    with pytest.raises(ValueError, match="does not forbid"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_an_honestly_recorded_FAILING_authentication_is_still_refused(bundle_tree):
    """The block matches recomputation exactly -- and records a case that does
    not authenticate. Comparing the block to its recomputation cannot catch
    this; only reading the verdicts inside it can."""
    capture = json.loads(json.dumps(bundle_tree["capture"]))
    capture["cases"][0]["recomputed_black_value_repr"] = "0.123456789"
    raw = json.dumps(capture, sort_keys=True).encode()
    for name in ("run1.json", "run2.json"):
        (bundle_tree["root"] / name).write_bytes(raw)

    source_csv = Path(bundle_tree["anchor"]["historical_source_path"]).read_bytes()
    honest = a6400.authentication_report(
        a6400._enrich_source_rows(a6400._parse_source_rows(source_csv)),
        capture["cases"])
    assert not honest[0]["ok"], "the fixture must actually fail one case"

    digest = hashlib.sha1(raw).hexdigest()
    path = rewrite(bundle_tree, capture_run_1_sha1=digest,
                   capture_run_2_sha1=digest, authentication=honest)
    with pytest.raises(ValueError, match="fail authentication"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_the_verdict_takes_the_capture_digests_from_the_VERIFIED_result(bundle_tree, monkeypatch, tmp_path):
    """Not from the document. A verdict that re-read the bundle's own fields
    would record what was claimed rather than what was checked."""
    real = M.verify_a6400_bundle_bytes

    def sentinel(raw, **kwargs):
        verified = dict(real(raw, **kwargs))
        verified["capture_run_1_sha1"] = "1" * 40
        verified["capture_run_2_sha1"] = "2" * 40
        return verified

    monkeypatch.setattr(M, "verify_a6400_bundle_bytes", sentinel)
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert record["a6400_capture_run_1_sha1"] == "1" * 40
    assert record["a6400_capture_run_2_sha1"] == "2" * 40
    assert record["a6400_capture_run_1_sha1"] != \
        bundle_tree["document"]["capture_run_1_sha1"]


def test_the_production_trust_anchor_is_the_frozen_one():
    """The default anchor is Task 6's pins, imported rather than restated."""
    assert M.TRUST_ANCHOR["historical_source_sha1"] == a6400.A6400_SOURCE_SHA1
    assert M.TRUST_ANCHOR["historical_source_sha1"] == \
        "a17d4737c747e2799253bebbc3d0261e0e697114"
    assert M.TRUST_ANCHOR["historical_source_path"] == a6400.A6400_SOURCE
    assert M.TRUST_ANCHOR["expected_cases"] == 30
    assert M.TRUST_ANCHOR["artifact_root"] == "logs/eval"


def test_load_verified_delegates_to_the_byte_verifier(bundle_tree, monkeypatch):
    """`load_verified_a6400_bundle` reads ONCE and delegates -- it does not
    carry a second copy of the verification."""
    seen = []
    real = M.verify_a6400_bundle_bytes
    monkeypatch.setattr(M, "verify_a6400_bundle_bytes",
                        lambda raw, **kw: (seen.append((raw, kw)), real(raw, **kw))[1])
    M.load_verified_a6400_bundle(bundle_tree["path"], anchor=bundle_tree["anchor"])
    assert len(seen) == 1
    raw, kwargs = seen[0]
    assert raw == Path(bundle_tree["path"]).read_bytes()
    assert kwargs["bundle_path"] == bundle_tree["path"]


def test_verdict_hashes_and_verifies_the_same_bytes(bundle_tree, monkeypatch, tmp_path):
    """No re-read between hashing and verification: the bytes recorded in
    `a6400_reference_bundle_sha1` ARE the bytes that were verified."""
    seen = {}
    real = M.verify_a6400_bundle_bytes

    def spy(raw, **kwargs):
        seen["raw"] = raw
        return real(raw, **kwargs)

    monkeypatch.setattr(M, "verify_a6400_bundle_bytes", spy)
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert hashlib.sha1(seen["raw"]).hexdigest() == \
        record["a6400_reference_bundle_sha1"]


# --- populations -------------------------------------------------------------

def test_thresholds_derive_only_from_non_a_rows(bundle_tree):
    """The winner's-curse guard, and it MUST be able to fail.

    The A rows carry exposure an order of magnitude above every census row, so a
    cutoff contaminated by them would sit far above the one the cohort yields.
    """
    rows = census()
    clean = M.derive_thresholds(rows, cohort_for(rows))
    assert clean["exposure_cutoff"] == 53.0

    # The guard MUST be able to fail: these A rows sit an order of magnitude
    # above every census row, so a population that admitted them would derive a
    # DIFFERENT cutoff -- which is exactly the winner's curse.
    hot = a_rows(exposure=1000.0)
    contaminated = criteria.nearest_rank_quantile(
        [r[EXPOSURE] for r in rows[:30] + hot],
        criteria.EXPOSURE_CUTOFF_RULE["quantile"])
    assert contaminated != clean["exposure_cutoff"]
    assert contaminated > max(r[EXPOSURE] for r in rows[:30])

    with pytest.raises(ValueError, match="A row reached"):
        M.derive_thresholds(rows + hot, cohort_for(rows))


def test_reach_is_measured_on_a_rows_only():
    assert criteria.REACH["population"] == "a_rows"
    assert criteria.REACH["establishes"] == "reach_only"
    high = M._reach(aggregates(a_rows()), criteria)
    low = M._reach(aggregates(a_rows(leaves=_UNREACHED_LEAVES)), criteria)
    assert high["passes"] and not low["passes"]
    assert high["value"] == pytest.approx(9 / 14)
    assert low["value"] == 0.0
    # Census rows never enter reach, whatever their leaves say.
    assert M._reach(aggregates(a_rows() + census()), criteria)["value"] == \
        pytest.approx(9 / 14)


def test_separation_reads_the_matched_cohort_not_the_broad_census():
    """The negative class is the 30 matched rows. Reading the broad census
    instead would change both n_c and the AUC."""
    rows = census()
    result = M.separation(cohort_for(rows), a_rows(), rows,
                          replicates=FAST_REPLICATES)
    assert result["n_c"] == 30 and result["n_a"] == 30
    assert result["n_c"] != len(rows)


def test_r_min_reads_the_broad_census_not_the_matched_cohort():
    """The cohort's top decile is about three rows; the frozen floor is 16."""
    rows = census()
    derived = M.derive_thresholds(rows, cohort_for(rows))
    assert derived["prospective_target_subset_size"] >= \
        criteria.R_MIN_RULE["subset_floor"]
    assert derived["prospective_target_subset_size"] > 3


def test_verdict_refuses_a_cohort_artifact_that_is_not_exactly_30_rows():
    rows = census()
    for size in (29, 31):
        document = dict(cohort_for(rows))
        entries = document["matched_cohort"]
        document["matched_cohort"] = (entries[:size] if size < 30
                                      else entries + [entries[0]])
        with pytest.raises(ValueError, match="frozen cardinality"):
            M.derive_thresholds(rows, document)


def test_verdict_authenticates_the_matched_cohort_artifact_sha1(bundle_tree, tmp_path, monkeypatch):
    """The digest is COMPUTED from the cohort file's bytes, and every other
    document must already agree with it."""
    paths = evidence_tree(tmp_path, monkeypatch)
    verified = M.load_verified_inputs(**paths)
    live = hashlib.sha1(Path(paths["cohort_path"]).read_bytes()).hexdigest()
    assert verified["sha1s"]["matched_cohort_sha1"] == live
    record = M._evaluate_verified(verified, bundle_tree["path"],
                                  separation_replicates=FAST_REPLICATES,
                                  anchor=bundle_tree["anchor"])
    assert record["matched_cohort_sha1"] == live


def test_undersized_prospective_target_subset_is_preflight_fail(bundle_tree, tmp_path, monkeypatch):
    """Floor 16, reason `prospective_target_subset_below_floor`."""
    rows = census()[:64]
    # A cohort of the HIGH-exposure rows pushes the cutoff to the top, leaving
    # almost nothing above it.
    high = {"matched_cohort": [{"game_content_sha1": r["game_content_sha1"],
                                "position_ply": r["position_ply"],
                                "canonical_state_sha1": r["canonical_state_sha1"]}
                               for r in rows[-30:]]}
    with pytest.raises(ValueError,
                       match=criteria.R_MIN_RULE["subset_below_floor_reason"]):
        M.derive_thresholds(rows, high)
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch, census_rows=rows,
                           sizing=False)
    assert record["verdict"] == M.PREFLIGHT_FAIL
    assert any(criteria.R_MIN_RULE["subset_below_floor_reason"] in reason
               for reason in record["failures"])


# --- separation --------------------------------------------------------------

def test_separation_uses_the_single_frozen_primary_formula():
    rows = census()
    result = M.separation(cohort_for(rows), a_rows(), rows,
                          replicates=FAST_REPLICATES)
    assert result["statistic"] == criteria.PRIMARY_EXPOSURE_COLUMN
    assert result["statistic"] == "exposure_primary_0.50"


def test_descriptive_formulas_cannot_rescue_a_failed_primary():
    """A rows dominant on a DESCRIPTIVE column and beaten on the primary one
    must still fail: only the primary statistic gates."""
    rows = census()
    weak = [dict(row, **{EXPOSURE: 0.0, "exposure_descriptive_count": 9999})
            for row in a_rows()]
    result = M.separation(cohort_for(rows), weak, rows,
                          replicates=FAST_REPLICATES)
    assert result["passes"] is False
    assert result["auc"] < criteria.SEPARATION["min_auc"]


def test_separation_requires_both_point_estimate_and_bootstrap_lower_bound():
    rows = census()
    result = M.separation(cohort_for(rows), a_rows(), rows,
                          replicates=FAST_REPLICATES)
    assert result["auc"] >= criteria.SEPARATION["min_auc"]
    assert result["auc_lower_bound"] >= criteria.SEPARATION["min_lower_bound"]
    assert result["passes"] is True
    # Neither alone is sufficient: the gate is the conjunction.
    assert result["passes"] == (
        result["auc"] >= result["min_auc"]
        and result["auc_lower_bound"] >= result["min_lower_bound"])
    # The lower bound is strictly below the point estimate, so it is a real
    # bootstrap quantile and not the point estimate wearing another name.
    assert result["auc_lower_bound"] < result["auc"]


def test_a_lower_bound_under_its_floor_fails_a_passing_point_estimate():
    """The conjunction, exercised where the two gates actually disagree.

    The real fixture cannot separate them -- at n = 30/30 a point estimate of
    0.80 carries a lower bound of 0.72 -- so the FLOOR is raised instead, via an
    injected criteria view. The code path is the production one.
    """
    import copy
    import types

    stub = types.SimpleNamespace(**{name: getattr(criteria, name)
                                    for name in dir(criteria)
                                    if not name.startswith("__")})
    stub.SEPARATION = copy.deepcopy(criteria.SEPARATION)
    stub.SEPARATION["min_lower_bound"] = 0.90        # above the real 0.7188
    rows = census()
    result = M.separation(cohort_for(rows), a_rows(), rows,
                          replicates=FAST_REPLICATES, criteria_module=stub)
    assert result["auc"] >= result["min_auc"]        # the point estimate passes
    assert result["auc_lower_bound"] < result["min_lower_bound"]
    assert result["passes"] is False                 # and the verdict is still no


def test_the_bootstrap_is_pinned_to_the_frozen_seed():
    """A different seed gives a different distribution; pinning the value is
    what makes `SEPARATION["bootstrap"]["seed"]` load-bearing rather than
    decorative."""
    rows = census()
    positive, negative = separation_classes(rows)
    frozen = criteria.auc_lower_bound(M._bootstrap_auc_distribution(
        positive, negative, FAST_REPLICATES,
        criteria.SEPARATION["bootstrap"]["seed"], criteria))
    other = criteria.auc_lower_bound(M._bootstrap_auc_distribution(
        positive, negative, FAST_REPLICATES, 0, criteria))
    assert frozen != other
    assert frozen == pytest.approx(BOOTSTRAP_LOWER_BOUND_AT_FROZEN_SEED, abs=1e-12)
    assert M.separation(cohort_for(rows), a_rows(), rows,
                        replicates=FAST_REPLICATES)["auc_lower_bound"] == \
        pytest.approx(BOOTSTRAP_LOWER_BOUND_AT_FROZEN_SEED, abs=1e-12)


def test_the_bootstrap_distribution_is_not_degenerate():
    """A constant-returning bootstrap would satisfy every gate above."""
    rows = census()
    positive, negative = separation_classes(rows)
    dist = M._bootstrap_auc_distribution(positive, negative, FAST_REPLICATES,
                                         criteria.SEPARATION["bootstrap"]["seed"],
                                         criteria)
    assert len(dist) == FAST_REPLICATES
    assert len(set(dist)) > FAST_REPLICATES // 4
    assert max(dist) - min(dist) > 0.1
    assert criteria.auc_lower_bound(dist) < criteria.mann_whitney_auc(
        positive, negative)


def test_the_bootstrap_defaults_to_the_frozen_replicate_count_and_seed():
    assert criteria.SEPARATION["bootstrap"]["replicates"] == 10000
    assert criteria.SEPARATION["bootstrap"]["seed"] == 20260729
    assert criteria.SEPARATION["lower_bound_quantile"] == 0.05


def test_separation_is_deterministic_under_the_frozen_seed():
    rows = census()
    first = M.separation(cohort_for(rows), a_rows(), rows,
                         replicates=FAST_REPLICATES)
    second = M.separation(cohort_for(rows), a_rows(), rows,
                          replicates=FAST_REPLICATES)
    assert first == second


def test_a_separation_failure_is_not_a_refutation():
    """Frozen before the number was known, so it cannot be renegotiated after."""
    rows = census()
    result = M.separation(cohort_for(rows),
                          [dict(row, **{EXPOSURE: 0.0}) for row in a_rows()],
                          rows, replicates=FAST_REPLICATES)
    assert "no effect" not in result["on_failure_means"]
    assert result["on_failure_does_not_mean"] == "no effect exists"


# --- R_min / R_max -----------------------------------------------------------

def test_r_min_failure_reason_matches_the_frozen_string():
    """Asserted against the Task 5 constant, never a string literal, so the two
    cannot drift apart again."""
    rows = census(shipped=100.0, capped=100.0)      # pooled(c) == 0 at every cap
    with pytest.raises(ValueError) as excinfo:
        M.derive_thresholds(rows, cohort_for(rows))
    assert criteria.R_MIN_RULE["fail_reason"] in str(excinfo.value)
    assert criteria.R_MIN_RULE["fail_reason"] == \
        "mechanism_not_predicted_to_act_at_any_cap"


def test_r_min_fails_on_a_nonpositive_pooled_prediction():
    rows = census(shipped=100.0, capped=120.0)      # pooled(c) < 0 everywhere
    with pytest.raises(ValueError,
                       match=criteria.R_MIN_RULE["fail_reason"]):
        M.derive_thresholds(rows, cohort_for(rows))


def test_r_min_is_not_floored_up_from_a_negative_prediction():
    """The floor raises a small POSITIVE prediction and never lifts a
    nonpositive one -- the result is PREFLIGHT_FAIL, never R_min == 0.01."""
    rows = census(shipped=100.0, capped=120.0)
    with pytest.raises(ValueError) as excinfo:
        M.derive_thresholds(rows, cohort_for(rows))
    assert criteria.R_MIN_RULE["fail_reason"] in str(excinfo.value)
    assert str(criteria.R_MIN_RULE["floor"]) not in str(excinfo.value)


def test_r_min_uses_the_WEAKEST_positive_cap_not_the_largest_pooled():
    """`c*` is the weakest cap with pooled(c) > 0 -- not the cap with the
    biggest prediction. With a uniform grid the two coincide, which is exactly
    why the fixture varies pooled across caps."""
    # 1.25 predicts nothing; 1.0 predicts a little; the stronger caps a lot.
    per_cap = {1.25: 110.0, 1.0: 96.0, 0.75: 40.0, 0.5: 20.0}
    rows = [case(i, population="census", phase="late",
                 ply=PHASE_BASE["late"] + (i % 2) * 13, exposure=float(i),
                 would_clip=0, stm=("red" if i % 2 == 0 else "black"), game=i,
                 capped_by_cap=per_cap)
            for i in range(1, 61)]
    derived = M.derive_thresholds(rows, cohort_for(rows))
    assert derived["weakest_positive_cap"] == 1.0
    assert derived["pooled_by_cap"]["1.25"] < 0
    assert derived["pooled_by_cap"]["0.5"] == pytest.approx(0.8)
    # 0.5 * pooled(1.0) = 0.02, above the 0.01 floor. Taking the LARGEST pooled
    # instead would give 0.4 and an empty band.
    assert derived["r_min"] == pytest.approx(0.02)
    assert derived["r_min"] < derived["r_max"]


def test_the_floor_does_raise_a_small_positive_prediction():
    rows = census(shipped=1000.0, capped=999.0)     # pooled == 0.001
    derived = M.derive_thresholds(rows, cohort_for(rows))
    assert derived["r_min"] == criteria.R_MIN_RULE["floor"]


def test_empty_r_band_is_a_failure():
    """R_min >= R_max is unsatisfiable, not a pass."""
    rows = census(shipped=100.0, capped=0.0)        # pooled == 1.0, R_min = 0.5
    with pytest.raises(ValueError, match="empty R band"):
        M.derive_thresholds(rows, cohort_for(rows))
    assert criteria.R_MAX_RULE["on_r_min_ge_r_max"] == "PREFLIGHT_FAIL"


# --- revisit form ------------------------------------------------------------

def test_revisit_form_paired_when_prospective_targets_are_dense():
    rows = census(would_clip=criteria.REVISIT_FORM_CRITERION[
        "min_would_clip_leaves"])
    cutoff = M.derive_thresholds(rows, cohort_for(rows))["exposure_cutoff"]
    assert M.decide_revisit_form(rows, cutoff) == "paired"


def test_revisit_form_candidate_only_when_sparse():
    rows = census(would_clip=1)
    cutoff = M.derive_thresholds(rows, cohort_for(rows))["exposure_cutoff"]
    assert M.decide_revisit_form(rows, cutoff) == "candidate_only_floor"


# --- the other mechanism criteria --------------------------------------------

def test_sign_dominance_failure_rejects(bundle_tree, tmp_path, monkeypatch):
    weak = a_rows(positive=1.0, negative=9.0)
    assert M._sign_dominance(weak, criteria)["passes"] is False
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch, a=weak)
    assert record["verdict"] == M.PREFLIGHT_FAIL
    assert any("sign_dominance" in reason for reason in record["failures"])


def test_terminal_fraction_over_bound_rejects(bundle_tree, tmp_path, monkeypatch):
    over = aggregates(a_rows(terminal=50, total=100))
    assert M._terminal_fraction(over, criteria)["passes"] is False
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch,
                           census_rows=census(terminal=30))
    assert record["verdict"] == M.PREFLIGHT_FAIL
    assert any("terminal_fraction" in reason for reason in record["failures"])


# --- the verdict vocabulary --------------------------------------------------

def evidence_tree(tmp_path, monkeypatch, *, census_rows=None, a=None,
                  sizing=True, scaled=True):
    """A COMPLETE on-disk evidence set in which every document REPRODUCES.

    Nothing here is hand-written: the census and crossover CSVs are emitted by
    Task 7's serializer from the artifact's own cases, the cohort by Task 4b's
    matcher, and the sizing ladder by Task 8. That is the only way the fixture
    can exercise revision 38 -- a hand-built document would be refused, which is
    the whole point.
    """
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as D
    from scripts.GPU.alphazero import fpu_provenance
    from scripts.GPU.alphazero import v18_cohort_matcher as CM
    from scripts.GPU.alphazero import v18_control_pool as CP
    from scripts.GPU.alphazero import v18_selector_sizing as S

    # A suite run always has untracked files, so the tree is never clean here.
    # At Execution Step 5 it is, by construction. Stub the environment fact, not
    # the check that reads it.
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)
    # The checkpoint and both replay reservoirs live under gitignored logs/.
    # `test_the_search_inputs_are_recomputed_against_their_pins` asserts
    # production calls the real authenticator, so stubbing it here cannot hide a
    # missing call.
    frozen_inputs = {"checkpoint_sha1": "a" * 40, "a_reservoir_sha1": "b" * 40,
                     "census_reservoir_sha1": "c" * 40}
    monkeypatch.setattr(D, "_authenticate_search_inputs",
                        lambda phase: dict(frozen_inputs))

    if scaled:
        # The AUTHORITATIVE object, patched before anything reads it -- never a
        # value taken from the sizing record. monkeypatch restores it.
        monkeypatch.setattr(criteria, "SIZING",
                            {**criteria.SIZING, **SCALED_SIZING})

    rows = census() if census_rows is None else census_rows
    a_list = a_rows() if a is None else a
    cases = [dict(r) for r in rows] + [dict(r) for r in a_list]

    root = Path(tempfile.mkdtemp(dir=str(tmp_path), prefix="evidence"))

    # The criteria artifact is emitted AFTER the patch, so `load_verified_criteria`
    # still performs a genuine re-derivation against the module it will read.
    criteria_path = root / "frozen_preflight_criteria.json"
    criteria.emit_frozen_criteria(str(criteria_path))
    criteria_sha1 = sha1_of(criteria_path)

    # The universe loader re-emits from the authenticated replay reservoir, which
    # a suite test has no access to. It is stubbed -- and
    # `test_the_universe_is_authenticated_not_merely_hashed` asserts production
    # calls it, so removing the call is still caught.
    universe = {"run_kind": "shipped_only_preflight_source_universe",
                "all_game_ids": sorted({r["game_content_sha1"] for r in rows}),
                "git_commit": fpu_provenance.git_commit(), "worktree_clean": True}
    universe_path = root / "universe.json"
    universe_path.write_bytes(canonical(universe))
    universe_sha1 = sha1_of(universe_path)
    monkeypatch.setattr(D, "load_verified_universe",
                        lambda path: (json.loads(Path(path).read_bytes()),
                                      sha1_of(Path(path))))

    # Task 7 publishes the leaves as their own CSV and strips them from the
    # artifact's cases, so the fixture does the same.
    from scripts.GPU.alphazero.diagnose_v18_residual_preflight import _csv_bytes
    residual_path = root / "residual_rows.csv"
    residual_path.write_bytes(_csv_bytes(
        [leaf for c in cases for leaf in c["residual_leaves"]],
        M.RESIDUAL_COLUMNS))
    cases = [{k: v for k, v in c.items() if k != "residual_leaves"}
             for c in cases]

    census_path = root / "census_positions.csv"
    census_path.write_bytes(M._reproduce_census_bytes(cases))
    crossover_path = root / "crossover_tables.csv"
    crossover_path.write_bytes(M._reproduce_crossover_bytes(cases))

    cohort, report = CM.match_cohort(rows, a_list)
    cohort_doc = {"run_kind": "v18_matched_control_cohort",
                  "matched_cohort": [CM._project(r) for r in cohort],
                  "matching_report": report,
                  "census_sha1": sha1_of(census_path),
                  "criteria_sha1": criteria_sha1,
                  "universe_sha1": universe_sha1}
    cohort_path = root / "matched_cohort.json"
    cohort_path.write_bytes(canonical(cohort_doc))

    import scripts.GPU.alphazero.diagnose_v18_residual_preflight as task7
    artifact_doc = {
        "run_kind": task7.RUN_KIND, "cases": cases,
        # The frozen Task 7 ENVELOPE. An artifact that merely parses is not
        # evidence; it must describe the measurement the plan authorized.
        "scientific_interpretation_forbidden": True,
        "search_execution_mode": task7.SEARCH_EXECUTION_MODE,
        "simulations": task7.SIMULATIONS, "add_noise": task7.ADD_NOISE,
        "c_puct": task7.FROZEN_C_PUCT,
        "batching_triple": list(task7.BATCHING_TRIPLE),
        "cap_grid": list(CAPS), "a_source_path": task7.A_SOURCE,
        # REAL authorities, not placeholders. A fixture that stamps "e"*40 into
        # the field under test cannot tell a real artifact from a fabricated one.
        "a_source_sha1": CP.FORBIDDEN_SOURCE_SHA1S["gate_A"],
        "population_order": list(task7.POPULATIONS),
        "seed_audit": task7.assert_seed_sets_disjoint(cases),
        "authenticated_search_inputs": dict(frozen_inputs),
        "source_sha1s": {path: fpu_provenance.file_sha1(path)
                         for path in task7.MEASUREMENT_SOURCE_MODULES},
        "git_commit": fpu_provenance.git_commit(), "worktree_clean": True,
        "runtime_identity_bracketed": True,
        "criteria_sha1": criteria_sha1, "universe_sha1": universe_sha1,
        "census_positions_sha1": sha1_of(census_path),
        "crossover_tables_sha1": sha1_of(crossover_path),
        "residual_rows_sha1": sha1_of(residual_path),
        # RECOMPUTED from the cases, because the boundary recomputes them too.
        "pooled_reach_numerator": sum(c["exposed_positive_mass_numerator"]
                                      for c in cases
                                      if c["population"] == "selected_a"),
        "pooled_reach_denominator": sum(c["exposed_positive_mass_denominator"]
                                        for c in cases
                                        if c["population"] == "selected_a"),
        "terminal_depth2_total": sum(c["terminal_depth2"] for c in cases),
        "total_depth2_total": sum(c["total_depth2"] for c in cases),
    }
    artifact_path = root / "preflight_artifact.json"
    artifact_path.write_bytes(canonical(artifact_doc))

    paths = {"preflight_artifact_path": str(artifact_path),
             "census_path": str(census_path),
             "crossover_tables_path": str(crossover_path),
             "residual_rows_path": str(residual_path),
             "cohort_path": str(cohort_path),
             "criteria_path": str(criteria_path),
             "universe_path": str(universe_path)}

    if sizing:
        cutoff = S.exposure_cutoff(
            S.matched_control_rows(cohort_doc["matched_cohort"], rows))
        ladder = S.sizing_ladder(rows, S.role_predicates(cutoff),
                                 all_game_ids=universe["all_game_ids"])
        record = {
            "run_kind": "v18_preflight_sizing", "ladder": ladder,
            "exposure_cutoff": cutoff,
            "sizing_status": ("SIZING_PASSES"
                              if S.smallest_qualifying_tier(ladder) is not None
                              else "SIZING_FAILS"),
            "smallest_qualifying_tier": S.smallest_qualifying_tier(ladder),
            "recommended_operational_size":
                S.recommended_operational_size(ladder),
            "census_sha1": sha1_of(census_path), "criteria_sha1": criteria_sha1,
            "universe_sha1": universe_sha1,
            "matched_cohort_sha1": sha1_of(cohort_path)}
        sizing_path = root / "sizing.json"
        sizing_path.write_bytes(canonical(record))
        paths["sizing_path"] = str(sizing_path)
    return paths


def sha1_of(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def canonical(payload):
    from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import canonical_json_bytes
    return canonical_json_bytes(payload)


def full_evaluate(tree, tmp_path, monkeypatch, **over):
    """The PRIVATE evaluator over a verified tree -- fast bootstrap, fixture
    anchor. `evaluate` itself takes none of those, which is the point."""
    paths = evidence_tree(tmp_path, monkeypatch, **over)
    verified = M.load_verified_inputs(**paths)
    return M._evaluate_verified(verified, tree["path"],
                                separation_replicates=FAST_REPLICATES,
                                anchor=tree["anchor"])


def test_missing_sizing_record_yields_provisional_not_pass(bundle_tree, tmp_path, monkeypatch):
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch, sizing=False)
    assert record["verdict"] == M.PROVISIONAL_PASS
    assert record["verdict"] != M.PREFLIGHT_PASS
    assert record["sizing"]["state"] == "absent"
    assert "authorizes nothing" in record["sizing"]["note"]


def test_completed_but_failing_sizing_yields_PREFLIGHT_FAIL_not_provisional(bundle_tree, tmp_path, monkeypatch):
    """A completed sizing run that cannot fill the geometry is a real negative
    result. Mapping it to provisional would mean an infeasible selector could
    never formally reject v18."""
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch,
                           census_rows=starved_census())
    assert record["verdict"] == M.PREFLIGHT_FAIL
    assert record["verdict"] != M.PROVISIONAL_PASS
    assert any("sizing" in reason for reason in record["failures"])


def test_full_pass_requires_both_mechanism_and_sizing(bundle_tree, tmp_path, monkeypatch):
    assert full_evaluate(bundle_tree, tmp_path, monkeypatch)["verdict"] == M.PREFLIGHT_PASS
    assert full_evaluate(bundle_tree, tmp_path, monkeypatch, sizing=False)["verdict"] != M.PREFLIGHT_PASS
    assert full_evaluate(bundle_tree, tmp_path, monkeypatch,
                         a=a_rows(leaves=_UNREACHED_LEAVES))["verdict"] == M.PREFLIGHT_FAIL


def test_the_verdict_vocabulary_is_exactly_three_states():
    assert {M.PREFLIGHT_FAIL, M.PROVISIONAL_PASS, M.PREFLIGHT_PASS} == {
        "PREFLIGHT_FAIL", "MECHANISM_PREFLIGHT_PROVISIONAL_PASS",
        "PREFLIGHT_PASS"}
    assert M.PROVISIONAL_PASS != M.PREFLIGHT_PASS


def test_verdict_is_byte_reproducible(bundle_tree, tmp_path, monkeypatch):
    from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import canonical_json_bytes
    first = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    second = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["scientific_interpretation_forbidden"] is True
    assert first["run_kind"] == "v18_preflight_verdict"


def test_verdict_records_every_input_sha1(bundle_tree, tmp_path, monkeypatch):
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    for name in M._REQUIRED_INPUT_SHA1S:
        assert len(record[name]) == 40
    assert set(M._REQUIRED_INPUT_SHA1S) == {
        "criteria_sha1", "universe_sha1", "census_sha1", "crossover_tables_sha1",
        "residual_rows_sha1", "matched_cohort_sha1", "preflight_artifact_sha1"}
    assert record["sizing_sha1"]


def test_verdict_records_both_a6400_capture_sha1s(bundle_tree, tmp_path, monkeypatch):
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    live = hashlib.sha1((bundle_tree["root"] / "run1.json").read_bytes()).hexdigest()
    assert record["a6400_capture_run_1_sha1"] == live
    assert record["a6400_capture_run_2_sha1"] == live
    assert record["a6400_cases_authenticated"] == 30


def test_verdict_records_the_bundle_document_sha1(bundle_tree, tmp_path, monkeypatch):
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert record["a6400_reference_bundle_sha1"] == hashlib.sha1(
        Path(bundle_tree["path"]).read_bytes()).hexdigest()


def test_a_tampered_bundle_stops_the_verdict_entirely(bundle_tree, tmp_path, monkeypatch):
    """No verdict is produced over an unverified bundle."""
    path = rewrite(bundle_tree, capture_run_1_sha1="0" * 40)
    with pytest.raises(ValueError, match="changed after the bundle was written"):
        full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert path == bundle_tree["path"]


def test_a_sizing_record_without_a_digest_is_refused(bundle_tree, tmp_path, monkeypatch):
    """`_evaluate_verified` is private, but it still refuses a verified bundle
    that carries a sizing record no digest was computed for -- the state that
    would exist if someone assembled the inputs by hand."""
    paths = evidence_tree(tmp_path, monkeypatch)
    verified = M.load_verified_inputs(**paths)
    verified["sha1s"].pop("sizing_sha1")
    with pytest.raises(ValueError, match="without a digest"):
        M._evaluate_verified(verified, bundle_tree["path"],
                             separation_replicates=FAST_REPLICATES,
                             anchor=bundle_tree["anchor"])


@pytest.mark.parametrize("name", M._REQUIRED_INPUT_SHA1S)
def test_the_pure_evaluator_refuses_a_verified_set_missing_a_digest(bundle_tree, tmp_path, name, monkeypatch):
    """`load_verified_inputs` always supplies all five, so this guard protects
    the private evaluator against being called with a hand-assembled set."""
    paths = evidence_tree(tmp_path, monkeypatch)
    verified = M.load_verified_inputs(**paths)
    verified["sha1s"].pop(name)
    with pytest.raises(ValueError, match="missing input sha1"):
        M._evaluate_verified(verified, bundle_tree["path"],
                             separation_replicates=FAST_REPLICATES,
                             anchor=bundle_tree["anchor"])


# --- the verified-input boundary (revision 37) -------------------------------

def edit_json(path, **changes):
    payload = json.loads(Path(path).read_bytes())
    payload.update(changes)
    Path(path).write_bytes(canonical(payload))


def test_a_changed_cohort_cannot_pass_under_its_stale_digest(tmp_path,
                                                             monkeypatch):
    """Evaluating a DIFFERENT cohort while every document still records the old
    one's digest."""
    paths = evidence_tree(tmp_path, monkeypatch)
    M.load_verified_inputs(**paths)                        # coherent as written
    payload = json.loads(Path(paths["cohort_path"]).read_bytes())
    payload["matched_cohort"] = payload["matched_cohort"][:29]
    Path(paths["cohort_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="matched_cohort_sha1"):
        M.load_verified_inputs(**paths)


def test_edited_reach_totals_cannot_pass(tmp_path, monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch)
    before = M.load_verified_inputs(**paths)["sha1s"]["preflight_artifact_sha1"]
    edit_json(paths["preflight_artifact_path"], pooled_reach_numerator=0.0)
    assert sha1_of(paths["preflight_artifact_path"]) != before


@pytest.mark.parametrize("target, field", [
    ("preflight_artifact_path", "census_positions_sha1"),
    ("preflight_artifact_path", "crossover_tables_sha1"),
    ("preflight_artifact_path", "criteria_sha1"),
    ("preflight_artifact_path", "universe_sha1"),
    ("cohort_path", "census_sha1"),
    ("cohort_path", "criteria_sha1"),
    ("cohort_path", "universe_sha1"),
    ("sizing_path", "census_sha1"),
    ("sizing_path", "matched_cohort_sha1"),
])
def test_every_embedded_binding_is_cross_checked(tmp_path, monkeypatch, target,
                                                 field):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths[target], **{field: "0" * 40})
    with pytest.raises(ValueError, match=field):
        M.load_verified_inputs(**paths)


def test_rows_are_derived_from_the_verified_artifact_not_supplied(tmp_path,
                                                                  monkeypatch):
    import inspect
    verified = M.load_verified_inputs(**evidence_tree(tmp_path, monkeypatch))
    assert len(verified["a_rows"]) == 30
    assert {r["population"] for r in verified["census_rows"]} == {"census"}
    signature = inspect.signature(M.evaluate).parameters
    assert "a_rows" not in signature and "census_rows" not in signature


def test_the_production_api_has_no_gate_changing_parameters():
    import inspect
    public = set(inspect.signature(M.evaluate).parameters)
    assert public == {"preflight_artifact_path", "census_path",
                      "crossover_tables_path", "residual_rows_path",
                      "cohort_path", "criteria_path", "universe_path",
                      "a_reference_bundle_path", "sizing_path"}
    for forbidden in ("separation_replicates", "criteria_module", "anchor",
                      "input_sha1s", "tiers", "trials"):
        assert forbidden not in public, forbidden
    private = set(inspect.signature(M._evaluate_verified).parameters)
    assert {"separation_replicates", "criteria_module", "anchor"} <= private
    assert not ({"tiers", "trials"} & set(
        inspect.signature(M.load_verified_inputs).parameters))


def test_the_criteria_file_is_re_derived_from_the_committed_module(tmp_path,
                                                                   monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["criteria_path"], separation={"min_auc": 0.1})
    with pytest.raises(ValueError, match="does not re-derive"):
        M.load_verified_inputs(**paths)


@pytest.mark.parametrize("target", ["preflight_artifact_path", "cohort_path",
                                    "sizing_path"])
def test_a_mislabelled_document_is_refused(tmp_path, monkeypatch, target):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths[target], run_kind="something_else")
    with pytest.raises(ValueError, match="run_kind"):
        M.load_verified_inputs(**paths)


# --- reproduction, not agreement (revision 38) -------------------------------

def test_the_universe_is_authenticated_not_merely_hashed(tmp_path, monkeypatch):
    """`load_verified_universe` re-emits the record from the authenticated
    source. Hashing the file would accept a coherently substituted universe."""
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as D
    paths = evidence_tree(tmp_path, monkeypatch)
    seen = []
    real = D.load_verified_universe
    monkeypatch.setattr(D, "load_verified_universe",
                        lambda path: (seen.append(path), real(path))[1])
    M.load_verified_inputs(**paths)
    assert seen == [paths["universe_path"]]


@pytest.mark.parametrize("field", ["exposure_primary_0.50", "sign_dominance"])
def test_edited_case_fields_break_census_reproduction(tmp_path, monkeypatch,
                                                      field):
    """The defect revision 38 closes: the stored census digest stays valid while
    the CASES the verdict evaluates are edited."""
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    payload["cases"][0][field] = 999.0
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="census CSV does not reproduce"):
        M.load_verified_inputs(**paths)


def test_edited_crossover_predictions_break_reproduction(tmp_path, monkeypatch):
    """R_min is derived from these tables, so they are evidence."""
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    table = payload["cases"][0]["crossover"]["1.25"]
    table["predicted_capped_replies"] = 1.0
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="crossover tables do not reproduce"):
        M.load_verified_inputs(**paths)


@pytest.mark.parametrize("field", ["pooled_reach_numerator",
                                   "terminal_depth2_total"])
def test_artifact_aggregates_are_recomputed_from_their_own_cases(tmp_path,
                                                                 monkeypatch,
                                                                 field):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], **{field: 0.0})
    with pytest.raises(ValueError, match=f"recomputing it"):
        M.load_verified_inputs(**paths)


def test_a_hand_picked_cohort_does_not_reproduce(tmp_path, monkeypatch):
    """The coherent attack: choose a friendlier cohort, regenerate its report,
    and update every digest that names it. Only re-running the matcher catches
    it."""
    from scripts.GPU.alphazero import v18_cohort_matcher as CM
    paths = evidence_tree(tmp_path, monkeypatch)
    rows = census()
    document = json.loads(Path(paths["cohort_path"]).read_bytes())
    # Swap one matched control for another admissible row from a different game.
    chosen = {entry["game_content_sha1"] for entry in document["matched_cohort"]}
    replacement = next(r for r in rows
                       if r["game_content_sha1"] not in chosen
                       and r["phase"] == "late")
    document["matched_cohort"][0] = CM._project(replacement)
    Path(paths["cohort_path"]).write_bytes(canonical(document))
    # ... and update every digest that names the cohort, coherently.
    edit_json(paths["sizing_path"],
              matched_cohort_sha1=sha1_of(paths["cohort_path"]))
    with pytest.raises(ValueError, match="cohort does not reproduce"):
        M.load_verified_inputs(**paths)


def test_a_fabricated_sizing_ladder_does_not_reproduce(tmp_path, monkeypatch):
    """An internally consistent all-success ladder passes Task 8's arithmetic
    reconciliation. Re-running the ladder is what refuses it."""
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _binomial_lower_bound
    paths = evidence_tree(tmp_path, monkeypatch,
                          census_rows=starved_census())
    payload = json.loads(Path(paths["sizing_path"]).read_bytes())
    assert payload["sizing_status"] == "SIZING_FAILS"
    trials = criteria.SIZING["trials_per_probabilistic_tier"]
    bound = _binomial_lower_bound(trials, trials, criteria.SIZING["alpha"])
    for tier in payload["ladder"]:
        if not tier["degenerate_full_universe"]:
            tier.update({"n_successes": trials, "success_rate": 1.0,
                         "lower_bound_95": bound, "meets_criterion": True,
                         "failure_reasons": {}, "witness_trial_index": 0})
    smallest = min(t["n_games"] for t in payload["ladder"]
                   if t["meets_criterion"])
    payload.update({"sizing_status": "SIZING_PASSES",
                    "smallest_qualifying_tier": smallest,
                    "recommended_operational_size": max(
                        t["n_games"] for t in payload["ladder"])})
    Path(paths["sizing_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="sizing ladder does not reproduce"):
        M.load_verified_inputs(**paths)


def test_a_flipped_sizing_status_does_not_reproduce(tmp_path, monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch,
                          census_rows=starved_census())
    edit_json(paths["sizing_path"], sizing_status="SIZING_PASSES")
    with pytest.raises(ValueError, match="re-running Task 8 gives"):
        M.load_verified_inputs(**paths)


# --- the scaled criteria are a TEST seam, never a production one -------------

def test_the_production_sizing_parameters_are_frozen():
    """The scaled ladder above exists so the suite is practical. These are the
    values the real Execution-Phase run uses, and nothing in the module may
    override them."""
    assert criteria.SIZING["probabilistic_tiers"] == (200, 300, 400, 500, 600, 700)
    assert criteria.SIZING["degenerate_tier"] == 800
    assert criteria.SIZING["trials_per_probabilistic_tier"] == 299
    assert criteria.SIZING["alpha"] == 0.05
    assert criteria.SIZING["minimum_lower_bound"] == 0.99


def test_the_scaled_criteria_do_not_leak_out_of_a_test(tmp_path, monkeypatch):
    """monkeypatch restores the authoritative object, so a later test cannot
    inherit a scaled ladder."""
    evidence_tree(tmp_path, monkeypatch)
    assert criteria.SIZING["probabilistic_tiers"] == SCALED_SIZING["probabilistic_tiers"]
    monkeypatch.undo()
    assert criteria.SIZING["probabilistic_tiers"] == (200, 300, 400, 500, 600, 700)
    assert criteria.SIZING["trials_per_probabilistic_tier"] == 299


def test_reproduction_reads_its_ladder_parameters_from_the_criteria_module(
        tmp_path, monkeypatch):
    """The SPY: production reproduction must take tiers and trials from the
    committed criteria, never from the sizing document it is checking."""
    from scripts.GPU.alphazero import v18_selector_sizing as S
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["sizing_path"]).read_bytes())
    # A document that ASKS for a different ladder shape.
    payload["probabilistic_tiers"] = [1]
    payload["trials_per_probabilistic_tier"] = 1
    Path(paths["sizing_path"]).write_bytes(canonical(payload))

    seen = {}
    real = S.sizing_ladder

    def spy(rows, predicates, **kwargs):
        seen.update(kwargs)
        return real(rows, predicates, **kwargs)

    monkeypatch.setattr(S, "sizing_ladder", spy)
    M.load_verified_inputs(**paths)          # the extra keys change nothing ...
    # ... because tiers and trials were never taken from the document: the
    # ladder is re-run on `criteria.SIZING`'s defaults alone.
    assert set(seen) == {"all_game_ids"}
    assert "tiers" not in seen and "trials" not in seen


# --- capture metadata --------------------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("mcts_sims", 400),
    ("gate_list", ["A", "B"]),
    ("run_kind", "v17_prechange_abcd"),
    ("mode", "v17_prechange_abcd"),
])
def test_a_capture_taken_under_the_wrong_settings_is_refused(bundle_tree, field,
                                                             value):
    """Digests prove a file did not change; they say nothing about what it is.
    Two identical 400-simulation captures authenticate perfectly."""
    capture = json.loads(json.dumps(bundle_tree["capture"]))
    capture[field] = value
    raw = json.dumps(capture, sort_keys=True).encode()
    for name in ("run1.json", "run2.json"):
        (bundle_tree["root"] / name).write_bytes(raw)
    digest = hashlib.sha1(raw).hexdigest()
    path = rewrite(bundle_tree, capture_run_1_sha1=digest,
                   capture_run_2_sha1=digest)
    with pytest.raises(ValueError, match="frozen A/6,400 mode requires"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


def test_captures_must_be_RAW_byte_identical_not_merely_equivalent(bundle_tree):
    """The builder compares raw bytes. Canonical-JSON equality would accept two
    files the builder itself refuses."""
    raw1 = (bundle_tree["root"] / "run1.json").read_bytes()
    reordered = json.dumps(json.loads(raw1), sort_keys=True, indent=2).encode()
    assert reordered != raw1 and json.loads(reordered) == json.loads(raw1)
    (bundle_tree["root"] / "run2.json").write_bytes(reordered)
    path = rewrite(bundle_tree,
                   capture_run_2_sha1=hashlib.sha1(reordered).hexdigest())
    with pytest.raises(ValueError, match="NOT byte-identical"):
        M.load_verified_a6400_bundle(path, anchor=bundle_tree["anchor"])


# --- the frozen guards Execution Step 6 reports ------------------------------

def test_the_verdict_reports_the_efficiency_floor_and_development_guards(
        bundle_tree, tmp_path, monkeypatch):
    guards = full_evaluate(bundle_tree, tmp_path, monkeypatch)["derived_guards"]
    assert guards["conversion_efficiency_min"] == \
        criteria.CONVERSION_EFFICIENCY_MIN["value"] == 0.5
    assert guards["conversion_efficiency_basis"] == "normative"
    assert guards["min_lost_replies"] == {"development": 20, "held_out": 30}
    assert guards["stable_leader_min_fraction"] == 0.75
    assert "not this preflight" in guards["applies_to"]


def test_step_six_reports_every_derived_quantity(bundle_tree, tmp_path,
                                                 monkeypatch):
    record = full_evaluate(bundle_tree, tmp_path, monkeypatch)
    assert record["thresholds"]["r_min"] is not None
    assert record["thresholds"]["r_max"] == criteria.R_MAX_RULE["value"]
    assert record["thresholds"]["exposure_cutoff"] == 53.0
    assert record["revisit_form"] in ("paired", "candidate_only_floor")
    assert record["derived_guards"]["conversion_efficiency_min"] == 0.5


def test_a_ladder_with_a_skipped_tier_is_refused():
    """A tier larger than the universe. `UNIVERSE["on_insufficient_games"]` is
    STOP, so this cannot happen on a legitimate run -- and letting it reach the
    reconciliation produces a bare KeyError instead of saying what is wrong."""
    ladder = [{"n_games": 200, "meets_criterion": False,
               "skipped": "only 122 games in the universe"}]
    record = {"ladder": ladder, "sizing_status": "SIZING_FAILS"}
    with pytest.raises(ValueError, match="skipped tier"):
        M._sizing_state(record, {"sizing_sha1": "a" * 40})


def test_sizing_state_re_derives_the_status_from_the_ladder():
    """Defense in depth behind `_reproduce_sizing`: even handed a record the
    boundary never saw, `_sizing_state` re-derives rather than reads."""
    ladder = sizing_ladder_over(criteria.SIZING, passing=None)
    record = {"ladder": ladder, "sizing_status": "SIZING_PASSES",
              "smallest_qualifying_tier": None,
              "recommended_operational_size": None}
    with pytest.raises(ValueError, match="re-deriving it from its own ladder"):
        M._sizing_state(record, {"sizing_sha1": "a" * 40})


def test_sizing_state_reruns_task_eight_reconciliation():
    """An incoherent ladder -- successes that its own rate contradicts."""
    ladder = sizing_ladder_over(criteria.SIZING, passing=None)
    ladder[0]["n_successes"] = ladder[0]["n_trials"]      # rate now disagrees
    record = {"ladder": ladder, "sizing_status": "SIZING_FAILS",
              "smallest_qualifying_tier": None,
              "recommended_operational_size": None}
    with pytest.raises(ValueError, match="refusing to emit"):
        M._sizing_state(record, {"sizing_sha1": "a" * 40})


def sizing_ladder_over(sizing_spec, passing=None):
    """A frozen-shape ladder at whatever tier/trial scale `sizing_spec` names."""
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _binomial_lower_bound
    floor = sizing_spec["minimum_lower_bound"]
    out = []
    for n in list(sizing_spec["probabilistic_tiers"]) + [
            sizing_spec["degenerate_tier"]]:
        trials = (1 if n == sizing_spec["degenerate_tier"]
                  else sizing_spec["trials_per_probabilistic_tier"])
        successes = trials if (passing is not None and n >= passing
                               and trials > 1) else 0
        bound = _binomial_lower_bound(successes, trials, sizing_spec["alpha"])
        out.append({"n_games": n, "n_trials": trials, "n_successes": successes,
                    "success_rate": successes / trials, "lower_bound_95": bound,
                    "meets_criterion": bound >= floor,
                    "degenerate_full_universe": n == sizing_spec["degenerate_tier"],
                    "failure_reasons": ({"qualify: x": trials - successes}
                                        if trials - successes else {}),
                    "witness_trial_index": 0 if bound >= floor else None})
    return out


# --- reach is bound to the leaf evidence (revision 39) -----------------------

def test_a_coherent_reach_edit_is_refused(tmp_path, monkeypatch):
    """THE attack this closes: raise every selected-A case numerator AND the
    artifact total to match, leaving residual_rows.csv untouched.

    Census and crossover reproduction both still pass -- neither carries these
    fields -- and `_recompute_aggregates` agrees, because the artifact now
    agrees with itself. Only the leaf evidence disagrees.
    """
    paths = evidence_tree(tmp_path, monkeypatch)
    M.load_verified_inputs(**paths)                        # coherent as written

    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    inflated = 0.0
    for case_row in payload["cases"]:
        if case_row["population"] == "selected_a":
            case_row["exposed_positive_mass_numerator"] = \
                case_row["exposed_positive_mass_denominator"]
            inflated += case_row["exposed_positive_mass_denominator"]
    payload["pooled_reach_numerator"] = inflated          # ... and the total
    before = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    assert payload["pooled_reach_numerator"] != before["pooled_reach_numerator"]
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))

    # The census and crossover CSVs are untouched and still reproduce ...
    cases = payload["cases"]
    assert M._reproduce_census_bytes(cases) == \
        Path(paths["census_path"]).read_bytes()
    assert M._reproduce_crossover_bytes(cases) == \
        Path(paths["crossover_tables_path"]).read_bytes()
    # ... and the artifact agrees with itself. Only the leaves refuse.
    with pytest.raises(ValueError, match="unpublished leaf evidence"):
        M.load_verified_inputs(**paths)


def test_edited_residual_rows_break_the_reach_reproduction(tmp_path,
                                                           monkeypatch):
    """The other direction: change the leaves and leave the cases alone."""
    paths = evidence_tree(tmp_path, monkeypatch)
    text = Path(paths["residual_rows_path"]).read_text()
    lines = text.splitlines()
    header, first = lines[0].split(","), lines[1].split(",")
    first[header.index("leaf_terminating_backups")] = "999"
    lines[1] = ",".join(first)
    Path(paths["residual_rows_path"]).write_text("\n".join(lines) + "\n")
    # The artifact's recorded digest no longer describes the file ...
    with pytest.raises(ValueError, match="residual_rows_sha1"):
        M.load_verified_inputs(**paths)


def test_the_residual_rows_digest_is_recorded(tmp_path, monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch)
    verified = M.load_verified_inputs(**paths)
    assert verified["sha1s"]["residual_rows_sha1"] == \
        sha1_of(paths["residual_rows_path"])
    assert "residual_rows_sha1" in M._REQUIRED_INPUT_SHA1S


def test_reach_numerator_counts_only_leaves_the_strongest_cap_pulls_DOWN():
    """`clip_direction == +1`: clipped AND positive. A negative residual of the
    same magnitude is pulled UP and contributes to the denominator only."""
    rows = []
    for residual, raw_leaf, backups in ((0.8, 1.0, 10), (-0.8, 1.0, 10)):
        row = {"case_id": "c", "raw_leaf": raw_leaf,
               "leaf_terminating_backups": backups, "residual": residual}
        for cap in CAPS:
            row[f"would_clip_{cap}"] = int(abs(residual) > cap)
        rows.append(row)
    masses = M._reach_masses_from_residual_rows(rows)
    assert masses["c"] == (10.0, 20.0)


# --- the Task 7 execution envelope -------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("simulations", 6400),
    ("add_noise", True),
    ("search_execution_mode", "batched"),
    ("c_puct", 2.5),
    ("batching_triple", [1, 1, 1]),
    ("cap_grid", [0.5]),
    ("scientific_interpretation_forbidden", False),
    ("worktree_clean", False),
    ("runtime_identity_bracketed", False),
    ("a_source_path", "logs/eval/somewhere_else.csv"),
    ("population_order", ["census"]),
])
def test_an_artifact_outside_the_frozen_envelope_is_refused(tmp_path,
                                                            monkeypatch, field,
                                                            value):
    """`run_kind` alone admits evidence measured at the wrong budget, with noise
    on, through the batched route, or from a dirty tree."""
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], **{field: value})
    with pytest.raises(ValueError, match="frozen Task 7 envelope requires"):
        M.load_verified_inputs(**paths)


@pytest.mark.parametrize("field", ["a_source_sha1", "git_commit"])
def test_the_envelope_requires_canonical_identities(tmp_path, monkeypatch,
                                                    field):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], **{field: "NOT-A-SHA"})
    with pytest.raises(ValueError, match="not a canonical sha1"):
        M.load_verified_inputs(**paths)


@pytest.mark.parametrize("field", ["source_sha1s", "authenticated_search_inputs",
                                   "seed_audit"])
def test_the_envelope_requires_the_authenticated_state(tmp_path, monkeypatch,
                                                       field):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], **{field: {}})
    with pytest.raises(ValueError, match="authenticated state is unrecorded"):
        M.load_verified_inputs(**paths)


def test_the_envelope_requires_every_measurement_source_module(tmp_path,
                                                               monkeypatch):
    """A change to any of these changes a measured number, so an artifact that
    omits one cannot describe the code that produced it."""
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as task7
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    dropped = sorted(payload["source_sha1s"])[0]
    payload["source_sha1s"].pop(dropped)
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    assert dropped in task7.MEASUREMENT_SOURCE_MODULES
    with pytest.raises(ValueError, match="do not match the live measurement"):
        M.load_verified_inputs(**paths)


def test_a_zero_reach_denominator_is_a_failure_not_a_pass():
    """`REACH["on_zero_denominator"]` is PREFLIGHT_FAIL.

    Coverage regained: rewriting the reach tests around leaf evidence dropped
    the zero-denominator case, and the mutation sweep caught that it had. Leaves
    whose raw value is non-positive carry zero weight, so the denominator is
    zero while the rows still exist.
    """
    weightless = ((0.0, 0.8, 10), (-0.4, 0.2, 10))
    result = M._reach(aggregates(a_rows(leaves=weightless)), criteria)
    assert result["value"] is None
    assert result["passes"] is False
    assert result["note"] == criteria.REACH["on_zero_denominator"] == "PREFLIGHT_FAIL"


# --- provenance: WRONG-BUT-CANONICAL, not absent or malformed ----------------
#
# Absent and malformed values were already refused. Everything below is a
# perfectly well-formed SHA-1 or a plausible dictionary that simply is not the
# right one -- the class the revision-39 fixture's placeholders hid.

def test_a_canonical_but_wrong_a_source_sha1_is_refused(tmp_path, monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], a_source_sha1="a" * 40)
    with pytest.raises(ValueError, match="frozen gate-A source"):
        M.load_verified_inputs(**paths)


def test_the_a_source_pin_is_the_control_pools_frozen_one():
    from scripts.GPU.alphazero import v18_control_pool as CP
    assert _SHA1_OK(CP.FORBIDDEN_SOURCE_SHA1S["gate_A"])


def _SHA1_OK(value):
    return bool(M._SHA1_RE.match(value))


def test_a_fabricated_commit_is_refused(tmp_path, monkeypatch):
    """A commit that does not exist, in canonical form."""
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], git_commit="1" * 40)
    with pytest.raises(ValueError, match="does not describe the code that would run"):
        M.load_verified_inputs(**paths)


def test_records_disagreeing_about_the_commit_are_refused(tmp_path, monkeypatch):
    """The artifact, the criteria record and the universe record must all name
    the same commit."""
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["universe_path"], git_commit="2" * 40)
    with pytest.raises(ValueError, match="does not describe the code that would run"):
        M.load_verified_inputs(**paths)


def test_a_record_from_a_dirty_worktree_is_refused(tmp_path, monkeypatch):
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"], worktree_clean=False)
    with pytest.raises(ValueError, match="frozen Task 7 envelope|dirty worktree"):
        M.load_verified_inputs(**paths)


def test_canonical_but_wrong_module_hashes_are_refused(tmp_path, monkeypatch):
    """Every value a well-formed SHA-1, every one of them wrong."""
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as task7
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"],
              source_sha1s={path: "0" * 40
                            for path in task7.MEASUREMENT_SOURCE_MODULES})
    with pytest.raises(ValueError, match="do not match the live measurement"):
        M.load_verified_inputs(**paths)


def test_one_stale_module_hash_is_refused(tmp_path, monkeypatch):
    """A single module recorded at an older version. The keys are complete and
    every value is canonical."""
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    stale = sorted(payload["source_sha1s"])[0]
    payload["source_sha1s"][stale] = "f" * 40
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="do not match the live measurement"):
        M.load_verified_inputs(**paths)


def test_a_fabricated_checkpoint_identity_is_refused(tmp_path, monkeypatch):
    """Canonical, plausible, and not what re-hashing the pinned checkpoint
    gives."""
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    payload["authenticated_search_inputs"]["checkpoint_sha1"] = "9" * 40
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="authenticated_search_inputs do not reproduce"):
        M.load_verified_inputs(**paths)


@pytest.mark.parametrize("field", ["a_reservoir_sha1", "census_reservoir_sha1"])
def test_a_fabricated_reservoir_identity_is_refused(tmp_path, monkeypatch, field):
    paths = evidence_tree(tmp_path, monkeypatch)
    payload = json.loads(Path(paths["preflight_artifact_path"]).read_bytes())
    payload["authenticated_search_inputs"][field] = "8" * 40
    Path(paths["preflight_artifact_path"]).write_bytes(canonical(payload))
    with pytest.raises(ValueError, match="authenticated_search_inputs do not reproduce"):
        M.load_verified_inputs(**paths)


def test_the_search_inputs_are_recomputed_against_their_pins(tmp_path,
                                                             monkeypatch):
    """The CALL: production re-hashes the checkpoint and both reservoirs rather
    than reading what the artifact claims. The suite stubs the authenticator
    (its files are gitignored), so this is what proves the call happens."""
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as D
    paths = evidence_tree(tmp_path, monkeypatch)
    seen = []
    stub = D._authenticate_search_inputs
    monkeypatch.setattr(D, "_authenticate_search_inputs",
                        lambda phase: (seen.append(phase), stub(phase))[1])
    M.load_verified_inputs(**paths)
    assert seen == ["opening"]


def test_a_self_declared_seed_audit_is_refused(tmp_path, monkeypatch):
    """`seed_audit` is a CONCLUSION about the cases, so it is recomputed."""
    paths = evidence_tree(tmp_path, monkeypatch)
    edit_json(paths["preflight_artifact_path"],
              seed_audit={"intersection": 999})
    with pytest.raises(ValueError, match="seed_audit does not reproduce"):
        M.load_verified_inputs(**paths)


def test_the_recomputed_seed_audit_is_the_frozen_selected_a_shape():
    """30 rows, 27 unique seeds, 3 duplicate groups -- accepted historical
    provenance, and what the fixture's A plies are built to reproduce."""
    from scripts.GPU.alphazero import diagnose_v18_residual_preflight as task7
    audit = task7.assert_seed_sets_disjoint(census() + a_rows())
    assert criteria.SEED_POLICY["selected_a"]["n_rows"] == 30
    assert criteria.SEED_POLICY["selected_a"]["unique_seeds"] == 27
    assert criteria.SEED_POLICY["selected_a"]["duplicate_groups"] == 3
    assert audit
