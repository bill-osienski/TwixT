"""v18 shipped-only residual preflight -- MEASUREMENT ONLY. Plan Task 7.

Runs shipped search over the selected-A rows and the frozen census, derives
every v18 metric post hoc, and emits four artifacts as one transaction. It COMPUTES NO VERDICT:
no PASS/FAIL, no derived threshold, no exposure cutoff. Judgement lives in Task
9, and the separation is enforced by a test rather than described here.

A rows establish REACH AND SEPARATION ONLY. Every numeric threshold derives from
the non-A cohort (spec Sec 2.2.3), which is why each row is tagged with its
`population`: the boundary has to be mechanically checkable downstream.

No v18 cap exists. `mcts.py` is unmodified, every search here is shipped search,
and `assert_shipped_search_config` refuses anything else.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import fpu_provenance
from . import v18_preflight_criteria as criteria
from .eval_runner import EvalConfig, cfg_from
from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .mcts import MCTS
from .position_probe_cases import position_state
from . import capture_v18_a6400 as capture_v18
from . import v18_control_pool as control_pool
from .v18_control_pool import SELECTED_UNIVERSE
from .v18_provisional_backup import IDENTITY_CAP
from .v18_crossover import assert_synchronous_tree, crossover_for_tree
from .v18_tree_walk import (eligible_depth2_pairs, residual, terminating_backups,
                            walk)

RUN_KIND = "shipped_only_preflight"
SIMULATIONS = 400
POPULATIONS = ("selected_a", "census")

# --- the seed rule ----------------------------------------------------------
# `game_idx` is reservoir-local, so ONE base seed would let an A game and a
# census game with the same index collide onto the same seed. Two bases, frozen.
SEED_POLICY = criteria.SEED_POLICY
BASE_SEEDS = {name: SEED_POLICY[name]["base"] for name in POPULATIONS}

# --- the ONE search route ---------------------------------------------------
# This constant is true BECAUSE of the single `search_with_root` call in
# `search_one` below. Any edit to that route must move this constant with it.
# `search_from_root` is never imported, referenced or called anywhere in this
# module: it backs up ALL waiters on a pending leaf with one expansion value,
# which would silently break the crossover substitution.
SEARCH_EXECUTION_MODE = "synchronous"

BATCHING_TRIPLE = (14, 48, 8)
FROZEN_C_PUCT = 1.5

# ONE constant, used at the search call AND at artifact emission. Two
# independent `False` literals would let a change at the call site run noisy
# search while the artifact still published `add_noise: false`.
ADD_NOISE = False

# Every module whose bytes determine a measured number. The producer itself is
# included: a change here changes the measurement, so omitting it would leave
# the artifact unable to describe the code that made it.
MEASUREMENT_SOURCE_MODULES = (
    # the search itself
    "scripts/GPU/alphazero/mcts.py",
    "scripts/GPU/alphazero/opening_diagnostics.py",   # MCTS selection calls it
    "scripts/GPU/alphazero/eval_runner.py",
    "scripts/GPU/alphazero/evaluator.py",
    "scripts/GPU/alphazero/local_evaluator.py",
    "scripts/GPU/alphazero/probe_eval.py",            # loads the checkpoint
    "scripts/GPU/alphazero/network.py",
    "scripts/GPU/alphazero/game/__init__.py",         # exports TwixtState + consts
    "scripts/GPU/alphazero/game/twixt_state.py",
    # state reconstruction and identity
    "scripts/GPU/alphazero/position_probe_cases.py",
    "scripts/GPU/alphazero/goal_line_trigger_probe_cases.py",
    "scripts/GPU/alphazero/fpu_state_hash.py",
    "scripts/GPU/alphazero/fpu_provenance.py",
    # canonical artifact bytes, and Task 4's game-identity implementation
    "scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py",
    "scripts/GPU/alphazero/diagnose_fpu_baseline_policy_mass.py",
    "scripts/GPU/alphazero/fpu_v17_provenance.py",
    # v18 derivation
    "scripts/GPU/alphazero/v18_provisional_backup.py",
    "scripts/GPU/alphazero/v18_tree_walk.py",
    "scripts/GPU/alphazero/v18_crossover.py",
    "scripts/GPU/alphazero/v18_preflight_criteria.py",
    "scripts/GPU/alphazero/v18_control_pool.py",
    "scripts/GPU/alphazero/capture_v18_a6400.py",
    # the frozen v17 capture dependency capture_v18_a6400 imports
    "scripts/GPU/alphazero/capture_v17_abcd_selected_moves.py",
    # the producer itself: a change here changes the measurement
    "scripts/GPU/alphazero/diagnose_v18_residual_preflight.py",
)

CENSUS_SCHEMA = criteria.CENSUS_SCHEMA
CAP_GRID = criteria.CAP_GRID


def shipped_config():
    """Shipped search at the frozen budget and batching triple."""
    return cfg_from(EvalConfig(mcts_sims=SIMULATIONS, mcts_eval_batch_size=14,
                               mcts_stall_flush_sims=48))


def assert_shipped_search_config(cfg) -> None:
    """Refuse anything that is not shipped search.

    No v18 cap field exists yet, so its presence at all is a defect; and a
    nonzero FPU of any family would make this a different experiment.
    """
    for attribute in ("v18_provisional_backup_cap", "provisional_backup_cap",
                      "depth2_cap"):
        value = getattr(cfg, attribute, None)
        if value is not None and value != IDENTITY_CAP:
            raise ValueError(
                f"not shipped search: {attribute}={value!r}. No positive-cap "
                f"configuration may run in the preflight")
    if getattr(cfg, "fpu_value", 0.0) != 0.0:
        raise ValueError(
            f"not shipped search: fpu_value={cfg.fpu_value!r}, expected 0.0")
    for attribute in ("fpu_policy_mass_reduction",
                      "fpu_shipped_policy_mass_reduction"):
        if getattr(cfg, attribute, None) is not None:
            raise ValueError(
                f"not shipped search: {attribute}="
                f"{getattr(cfg, attribute)!r}, expected None")
    if cfg.n_simulations != SIMULATIONS:
        raise ValueError(
            f"not the frozen budget: n_simulations={cfg.n_simulations}, "
            f"expected {SIMULATIONS}")


# --- seeds ------------------------------------------------------------------


def derived_seed(case: Dict) -> int:
    """Per-population seed, by the policy frozen in the criteria.

    ASYMMETRIC on purpose. Selected-A keeps the historical XOR rule because the
    frozen A artifacts were produced under it. The census cannot: `game_idx` and
    `position_ply` are both < 1024, so XOR admits at most 1024 distinct values
    and the 1,974-row census collapses to 841 -- 1,133 forced duplicates. Its
    seed is a digest over the replay CONTENT hash, which has no such ceiling and
    does not depend on reservoir-local `game_idx`.
    """
    population = case["population"]
    policy = SEED_POLICY.get(population)
    if policy is None:
        raise ValueError(f"unknown population {population!r}")
    if policy["rule"] == "historical_xor":
        return (policy["base"] ^ int(case["game_idx"])
                ^ int(case["position_ply"]))
    if policy["rule"] == "sha1_digest":
        material = "|".join((policy["domain_tag"], str(policy["base"]),
                             str(case["game_content_sha1"]),
                             str(case["position_ply"]))).encode()
        return int.from_bytes(hashlib.sha1(material).digest()[:8], "big")
    raise ValueError(f"unknown seed rule {policy['rule']!r}")


def derived_seed_sets(cases: Sequence[Dict]) -> Dict[str, set]:
    out = {name: set() for name in POPULATIONS}
    for case in cases:
        out[case["population"]].add(derived_seed(case))
    return out


def assert_seed_sets_disjoint(cases: Sequence[Dict]) -> Dict:
    """Compare COMPLETE DERIVED SETS, never base seeds.

    XOR with reservoir-local `game_idx` and `position_ply` can collide across
    two different bases, so `20260616 != 20260730` proves nothing. A collision
    is a pre-search STOP: by the execution phase the bases are committed and
    embedded in the emitted criteria, so mutating one would rewrite a frozen
    execution.
    """
    sets = derived_seed_sets(cases)
    counts = {name: sum(1 for c in cases if c["population"] == name)
              for name in POPULATIONS}
    for name in POPULATIONS:
        if SEED_POLICY[name]["require_unique"] and counts[name] != len(sets[name]):
            raise ValueError(
                f"{name}: {counts[name]} rows collapse to {len(sets[name])} "
                f"distinct seeds. Its policy requires uniqueness; two positions "
                f"on one seed are not independent searches")
    intersection = sets["selected_a"] & sets["census"]
    if intersection:
        raise ValueError(
            f"derived seed sets intersect in {len(intersection)} value(s), e.g. "
            f"{sorted(intersection)[:3]}. This is a STOP before any evaluator "
            f"call; changing a base seed requires a criteria amendment, a "
            f"commit and a restart from Execution step 1")
    # Selected-A's three duplicate groups are ACCEPTED HISTORICAL PROVENANCE:
    # the frozen A artifacts were produced under that rule, so it is recorded,
    # not repaired.
    audit = {"selected_a_size": len(sets["selected_a"]),
             "census_size": len(sets["census"]),
             "selected_a_cases": counts["selected_a"],
             "census_cases": counts["census"],
             "duplicate_seeds_within_population": {
                 name: counts[name] - len(sets[name]) for name in POPULATIONS},
             "intersection_size": 0,
             # The COMPLETE asymmetric policy. Reporting one XOR `rule` was a
             # false label: the census does not use it.
             "policy": SEED_POLICY,
             "base_seeds": dict(BASE_SEEDS)}
    # The frozen A figures are a claim about the historical source; check them
    # against what was actually derived rather than restating them.
    a_policy = SEED_POLICY["selected_a"]
    if counts["selected_a"] == a_policy["n_rows"]:
        observed = (counts["selected_a"], len(sets["selected_a"]),
                    counts["selected_a"] - len(sets["selected_a"]))
        frozen = (a_policy["n_rows"], a_policy["unique_seeds"],
                  a_policy["duplicate_groups"])
        if observed != frozen:
            raise ValueError(
                f"selected-A seed shape {observed} contradicts the frozen "
                f"criteria {frozen} (rows, unique seeds, duplicates)")
        audit["selected_a_matches_frozen_criteria"] = True
    return audit


# --- authenticated inputs ---------------------------------------------------

_RUNTIME_KEYS = ("git_commit", "worktree_clean")


def load_verified_criteria(path: str):
    """Re-derive the criteria from the COMMITTED MODULE, not from a value
    stored beside the file. A stored hash authenticates nothing when whoever
    edited the payload could edit the hash too."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode())
    expected = criteria.as_dict()
    stripped = {k: v for k, v in payload.items() if k not in _RUNTIME_KEYS}
    # Compare CANONICALLY: the module holds tuples, the artifact round-trips
    # them as lists, so a raw == would always differ.
    if canonical_json_bytes(stripped) != canonical_json_bytes(expected):
        differing = sorted(
            k for k in set(stripped) | set(expected)
            if canonical_json_bytes(stripped.get(k))
            != canonical_json_bytes(expected.get(k)))
        raise ValueError(
            f"criteria artifact does not re-derive from the committed module; "
            f"differing keys: {differing[:5]}")
    return payload, hashlib.sha1(raw).hexdigest()


def load_verified_universe(path: str):
    """Authenticate the frozen universe record against the committed module."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode())
    if payload.get("run_kind") != "shipped_only_preflight_source_universe":
        raise ValueError(f"universe record has run_kind "
                         f"{payload.get('run_kind')!r}")
    if payload.get("scientific_interpretation_forbidden") is not True:
        raise ValueError("universe record does not forbid interpretation")
    if payload.get("selection_is_independent_of_residual_exposure") is not True:
        raise ValueError("universe record does not assert residual independence")
    _reject_fixture_universe(payload)
    # RE-DERIVE the geometry with the committed Task 4 implementation. Label
    # checks alone would let a record with a substituted census pass while the
    # artifact still recorded a legitimate universe SHA-1.
    # Internal arithmetic only proves SELF-consistency: a coherently
    # substituted census reconciles perfectly. Reproduce the record instead.
    control_pool._assert_record_reconciles(payload)
    reproduce_universe(raw)
    if (canonical_json_bytes(payload.get("selection_inputs"))
            != canonical_json_bytes(control_pool.SELECTION_INPUTS)):
        raise ValueError(
            "universe record's selection_inputs do not match the committed "
            "Task 4 module")
    if (canonical_json_bytes(payload.get("forbidden_sources"))
            != canonical_json_bytes([s["name"]
                                     for s in control_pool.FORBIDDEN_SOURCES])):
        raise ValueError("universe record's forbidden source list has drifted")
    return payload, hashlib.sha1(raw).hexdigest()


def _reject_fixture_universe(payload: Dict) -> None:
    """Production measures only the authenticated selected source.

    A fixture record has no replay reservoir behind it, so nothing it claims
    can be reproduced -- accepting one would let the whole chain pass on a
    record that was never generated from real data.
    """
    selected = payload.get("selected_universe")
    if selected == "fixture":
        raise ValueError(
            "universe record is a FIXTURE record; production measures only the "
            "authenticated selected source")
    if canonical_json_bytes(selected) != canonical_json_bytes(SELECTED_UNIVERSE):
        raise ValueError(
            "universe record's selected_universe does not match the committed "
            "SELECTED_UNIVERSE")


def reproduce_universe(supplied: bytes) -> None:
    """Re-emit the Task 4 record from the AUTHENTICATED source and byte-compare.

    `_assert_record_reconciles` proves only that a record agrees with itself, so
    a coherently substituted census -- one whose counts and geometry were
    recomputed to match -- passes it untouched. Only regenerating from the
    authenticated reservoir establishes that these rows are the census the
    frozen universe actually yields.
    """
    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch) / "universe.json"
        control_pool.freeze_source_universe(
            str(fresh), SELECTED_UNIVERSE["name"], criteria.SIZING["seed"])
        if fresh.read_bytes() != supplied:
            raise ValueError(
                "universe record does not reproduce: re-emitting it from the "
                "authenticated source yields different bytes, so the supplied "
                "census is not the one the frozen universe produces")


def assert_runtime_matches_records(*records, expected_commit: str) -> None:
    """Every authenticated record must describe the CAPTURED commit.

    `expected_commit` comes from the opening runtime identity, never from a
    fresh read: rereading HEAD here would compare the records against whatever
    the tree happens to be at this instant rather than against the state the
    measurement is bracketed by.

    A record emitted at another commit describes different code, and one
    emitted from a dirty tree describes code that was never committed at all.
    """
    head = expected_commit
    for name, payload in records:
        if payload.get("git_commit") != head:
            raise ValueError(
                f"{name} records git_commit {payload.get('git_commit')!r}, "
                f"HEAD is {head!r}: it does not describe the code that would run")
        if payload.get("worktree_clean") is not True:
            raise ValueError(
                f"{name} was emitted from a dirty worktree, so it describes "
                f"code that was never committed")


def census_cases_from_universe(universe_payload: Dict) -> List[Dict]:
    """The census population, derived EXACTLY from the verified Task 4 record.

    Nothing is caller-supplied. `source_universe_ordinal` is the row's rank in
    the record's own content-SHA-ascending `all_game_ids`, so it cannot be
    asserted independently of the universe it claims to index.
    """
    ordinals = {sha: i for i, sha in enumerate(universe_payload["all_game_ids"])}
    cases = []
    for row in universe_payload["census_positions"]:
        game = row["game_content_sha1"]
        if game not in ordinals:
            raise ValueError(
                f"census row names game {game[:12]}... absent from all_game_ids")
        cases.append({
            "population": "census",
            "case_id": f"census_{game[:12]}_{row['position_ply']}",
            "source_universe_ordinal": ordinals[game],
            "game_content_sha1": game,
            "game_idx": row["game_idx"],
            "position_ply": row["position_ply"],
            "side_to_move": row["side_to_move"],
            # Task 4 names it `canonical_sha1`; the census schema names it
            # `canonical_state_sha1`. One quantity, converted here rather than
            # carried under two names.
            "canonical_state_sha1": row["canonical_sha1"],
            "phase": row["phase"],
            "replay_path": row["replay_path"],
        })
    return cases


A_SOURCE = ("logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/"
            "position_probe_cases.csv")
A_CHECKPOINT_ROW_FILTER = "0001"


def selected_a_cases() -> Tuple[List[Dict], str]:
    """The 30 reach rows, from the AUTHENTICATED A source and the authenticated
    seed20115 reservoir. Reach and separation ONLY.

    Returns `(cases, a_source_sha1)`: the artifact records which bytes these
    rows came from, and the bytes hashed are the bytes parsed.
    """
    raw = Path(A_SOURCE).read_bytes()
    actual = hashlib.sha1(raw).hexdigest()
    pinned = control_pool.FORBIDDEN_SOURCE_SHA1S["gate_A"]
    if actual != pinned:
        raise ValueError(
            f"A source {A_SOURCE} hashes {actual}, pinned {pinned}")
    # The canonical identities below are reconstructed from replay bytes, so
    # bind that reservoir BEFORE deriving them.
    capture_v18.authenticate_replay_reservoir("pre_derivation")
    rows = [r for r in csv.DictReader(raw.decode().splitlines())
            if r["checkpoint"] == A_CHECKPOINT_ROW_FILTER]
    if len(rows) != 30:
        raise ValueError(f"A source yielded {len(rows)} rows, expected 30")
    cases = []
    for row in rows:
        capture_v18.canonical_state_sha1_for(row)   # binds the reservoir path
        cases.append({
            "population": "selected_a",
            "case_id": row["case_id"],
            "source_universe_ordinal": "",
            "game_content_sha1": hashlib.sha1(
                Path(row["replay_path"]).read_bytes()).hexdigest(),
            "game_idx": int(row["game_idx"]),
            "position_ply": int(row["position_ply"]),
            "side_to_move": row["side_to_move"],
            "canonical_state_sha1": capture_v18.canonical_state_sha1_for(row),
            "phase": _phase_of(int(row["position_ply"])),
            "replay_path": row["replay_path"],
        })
    return cases, actual


def _phase_of(ply: int) -> str:
    for name in ("opening", "early_mid", "midgame", "late"):
        low, high = criteria.PHASE_WINDOWS[name]
        if ply >= low and (high is None or ply <= high):
            return name
    raise ValueError(f"ply {ply} matched no phase window")


# --- measurement ------------------------------------------------------------


def _state_for(case: Dict):
    replay = json.loads(Path(case["replay_path"]).read_bytes().decode())
    return position_state(replay, int(case["position_ply"]),
                          case["side_to_move"])


def search_one(evaluator, cfg, case: Dict):
    """THE single search call of this module.

    `SEARCH_EXECUTION_MODE` is true because of this line: `search_with_root` is
    the synchronous per-sim path, "NOT search_from_root's batched waiter path"
    (mcts.py:528-535).
    """
    seed = derived_seed(case)
    _counts, _value, root = MCTS(evaluator, cfg, random.Random(seed)) \
        .search_with_root(_state_for(case), add_noise=ADD_NOISE)
    return root


def build_row(root, caps, c_puct: float, meta: Dict) -> Dict:
    """One measured position. Pure post-hoc derivation from a finished tree."""
    if "search_execution_mode" in meta:
        raise ValueError(
            "meta carries search_execution_mode: a caller-supplied mode is an "
            "unaudited claim about a route this module did not take")
    assert_synchronous_tree(root, SIMULATIONS,
                            search_execution_mode=SEARCH_EXECUTION_MODE)

    record = walk(root, caps)
    row = dict(meta)
    row.update({
        "root_value_stm": root.value_sum / root.visit_count,
        "n_legal": len(root.priors or {}),
        "eligible_depth2_leaves": record["eligible_depth2_leaves"],
        "replies": record["replies"],
        "explored_replies": record["explored_replies"],
        "depth_ge3_backups": record["depth_ge3_backups"],
        "depth_ge3_fraction": record["depth_ge3_fraction"],
        "follow_up_visits_per_reply": record["follow_up_visits_per_reply"],
        "positive_mass": record["positive_mass"],
        "negative_mass": record["negative_mass"],
        "sign_dominance": record["sign_dominance"],
        "terminal_depth2": record["terminal_depth2"],
        "total_depth2": record["total_depth2"],
        "seed": derived_seed(meta),
        "walk": record,
    })

    strongest = str(criteria.STRONGEST_CAP)
    primary = record["per_cap"][strongest]
    row[criteria.PRIMARY_EXPOSURE_COLUMN] = \
        primary["contribution_weighted_positive_mass"]
    row["exposure_descriptive_count"] = primary["positive_count"]
    row["exposure_descriptive_clipped_mass"] = primary["clipped_amount_total"]
    row["exposed_positive_mass_numerator"] = primary["exposed_positive_mass_numerator"]
    row["exposed_positive_mass_denominator"] = primary["exposed_positive_mass_denominator"]

    for cap in caps:
        per_cap = record["per_cap"][str(cap)]
        row[f"would_clip_{cap}"] = per_cap["would_clip_count"]
        row[f"clipped_amount_{cap}"] = per_cap["clipped_amount_total"]
        row[f"revisit_to_depth3_rate_{cap}"] = per_cap["revisit_to_depth3_rate"]

    row["crossover"] = {str(cap): crossover_for_tree(root, cap, c_puct)
                        for cap in caps}
    row["residual_leaves"] = _residual_leaves(root, caps, meta)
    return row


def _residual_leaves(root, caps, meta) -> List[Dict]:
    out = []
    for parent, leaf in eligible_depth2_pairs(root):
        entry = {
            "population": meta["population"],
            "case_id": meta["case_id"],
            "game_idx": meta["game_idx"],
            "position_ply": meta["position_ply"],
            "side_to_move": meta["side_to_move"],
            "canonical_state_sha1": meta["canonical_state_sha1"],
            "raw_parent": parent.nn_value,
            "raw_leaf": leaf.nn_value,
            "residual": residual(parent, leaf),
            "leaf_visit_count": leaf.visit_count,
            "leaf_terminating_backups": terminating_backups(leaf),
            "leaf_has_depth3_child": any(
                c.visit_count > 0 for c in leaf.children.values()),
        }
        for cap in caps:
            binds = abs(entry["residual"]) > cap
            entry[f"would_clip_{cap}"] = int(binds)
            entry[f"clipped_amount_{cap}"] = (
                abs(entry["residual"]) - cap if binds else 0.0)
        out.append(entry)
    return out


# --- emission ---------------------------------------------------------------


def _csv_bytes(rows: Sequence[Dict], columns: Sequence[str]) -> bytes:
    """Serialize to BYTES, never straight to disk: the artifact set is published
    as one transaction, and a hash cannot be recorded for a file that was
    already written."""
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns),
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buffer.getvalue().encode()


def publish_atomically(out_dir: str, payloads: Dict[str, bytes]) -> None:
    """Publish the complete artifact set, or nothing.

    Writing the CSVs directly and the JSON afterwards leaves partial scientific
    outputs behind when a later step fails -- and a reader cannot tell a
    complete set from an aborted one. Everything is staged in a sibling
    directory and moved into place with a single rename.
    """
    out = Path(out_dir)
    if out.exists():
        raise ValueError(
            f"{out} already exists; the measurement publishes a complete set "
            f"into a fresh directory and never merges into an existing one")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(out.parent),
                                    prefix=f".{out.name}.staging-"))
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        os.replace(str(staging), str(out))
    except BaseException:
        for leftover in staging.glob("*"):
            leftover.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def _assert_full_config(cfg) -> None:
    """The COMPLETE frozen configuration, not just the cap fields."""
    assert_shipped_search_config(cfg)
    if (cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits) \
            != BATCHING_TRIPLE:
        raise ValueError(
            f"batching triple {(cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits)} "
            f"!= frozen {BATCHING_TRIPLE}")
    if cfg.c_puct != FROZEN_C_PUCT:
        raise ValueError(f"c_puct {cfg.c_puct} != frozen {FROZEN_C_PUCT}")
    if ADD_NOISE is not False:
        raise ValueError(
            f"ADD_NOISE is {ADD_NOISE!r}: shipped preflight search is "
            f"noiseless; refusing before any search")


def measure(cases: Sequence[Dict], *, evaluator, cfg) -> List[Dict]:
    """Search and derive every row. No I/O, no authentication -- callers do both."""
    return [build_row(search_one(evaluator, cfg, case), CAP_GRID, cfg.c_puct,
                      dict(case)) for case in cases]


def _derive_cases(universe_payload: Dict) -> Tuple[List[Dict], str]:
    """Both populations, selected-A first. The ONLY source of measured rows."""
    a_cases, a_source_sha1 = selected_a_cases()
    return a_cases + census_cases_from_universe(universe_payload), a_source_sha1


def _make_evaluator():
    return _default_evaluator()


def _assert_output_is_fresh(out_dir: str) -> None:
    if Path(out_dir).exists():
        raise ValueError(
            f"{out_dir} already exists; the measurement publishes a complete "
            f"set into a fresh directory and never merges into an existing one")


def run_preflight(*, out_dir: str, criteria_path: str,
                  universe_path: str) -> Dict:
    """The frozen measurement. SELF-CONSTRAINED: it derives its own populations.

    There is deliberately no `cases` parameter and no `c_puct` parameter on the
    production path. A caller that supplies rows can measure arbitrary replays
    while the artifact still records a legitimate universe SHA-1; a caller that
    supplies `c_puct` changes the selection arithmetic the crossover mirrors.
    """
    # Freshness FIRST: discovering an occupied destination after the census has
    # run would waste the whole measurement. The publish-time check stays as
    # TOCTOU protection.
    _assert_output_is_fresh(out_dir)

    # OPEN THE BRACKET HERE, before any evidence is verified or derived.
    # Criteria verification, universe re-emission and case derivation all read
    # the source tree; a clean HEAD move DURING them would otherwise become the
    # "opening" identity, the closing check would agree with it, and the
    # artifact would bind the new HEAD to records authenticated under the old
    # one. Everything that authenticates evidence now happens inside it.
    opening_runtime = runtime_identity()
    if opening_runtime["worktree_clean"] is not True:
        raise ValueError(
            "refusing to measure on a dirty worktree: the artifact would record "
            "a commit that does not describe the code that ran")

    criteria_payload, criteria_sha1 = load_verified_criteria(criteria_path)
    universe_payload, universe_sha1 = load_verified_universe(universe_path)
    # Checked against the CAPTURED commit, never a fresh read of HEAD.
    assert_runtime_matches_records(("criteria", criteria_payload),
                                   ("universe", universe_payload),
                                   expected_commit=opening_runtime["git_commit"])

    # Selected-A FIRST, census second: 30 cheap rows expose a failure before the
    # long census, and both land in ONE bound artifact set for Task 9.
    cases, a_source_sha1 = _derive_cases(universe_payload)
    for case in cases:
        if case["population"] not in POPULATIONS:
            raise ValueError(f"unknown population {case['population']!r}")
    seed_audit = assert_seed_sets_disjoint(cases)

    cfg = shipped_config()
    _assert_full_config(cfg)
    if SEARCH_EXECUTION_MODE != "synchronous":
        raise ValueError(
            f"SEARCH_EXECUTION_MODE is {SEARCH_EXECUTION_MODE!r}: this module "
            f"has exactly one route and it is synchronous; refusing before any "
            f"search, and writing no artifact")

    # OPENING authentication of every mutable search input. The runtime bracket
    # is already open, from before any evidence was verified.
    opening = _authenticate_search_inputs("opening")

    evaluator = _make_evaluator()
    rows = measure(cases, evaluator=evaluator, cfg=cfg)

    # CLOSING authentication, before any byte is published.
    closing = _authenticate_search_inputs("closing")
    if closing != opening:
        raise ValueError(
            f"search inputs changed during the measurement: {opening} -> "
            f"{closing}. No artifact written")
    closing_runtime = runtime_identity()
    if closing_runtime != opening_runtime:
        changed = sorted(
            k for k in set(opening_runtime) | set(closing_runtime)
            if opening_runtime.get(k) != closing_runtime.get(k))
        moved = sorted(
            path for path, sha in opening_runtime["source_sha1s"].items()
            if closing_runtime["source_sha1s"].get(path) != sha)
        raise ValueError(
            f"the running code changed during the measurement: {changed}, "
            f"moved sources {moved[:3]}. The trees were produced by the OPENING "
            f"state, so no artifact is written")

    census_payload = _csv_bytes(rows, CENSUS_SCHEMA)
    residual_columns = (
        ["population", "case_id", "game_idx", "position_ply", "side_to_move",
         "canonical_state_sha1", "raw_parent", "raw_leaf", "residual",
         "leaf_visit_count", "leaf_terminating_backups", "leaf_has_depth3_child"]
        + [f"{p}_{c}" for c in CAP_GRID for p in ("would_clip", "clipped_amount")])
    residual_payload = _csv_bytes(
        [leaf for row in rows for leaf in row["residual_leaves"]], residual_columns)

    crossover_columns = ["population", "case_id", "cap",
                         "predicted_shipped_replies", "predicted_capped_replies",
                         "predicted_reply_delta", "predicted_reply_reduction",
                         "excluded_terminal", "excluded_synthetic"]
    crossover_rows = []
    for row in rows:
        for cap, table in row["crossover"].items():
            crossover_rows.append({
                "population": row["population"], "case_id": row["case_id"],
                "cap": cap,
                "predicted_shipped_replies": table["predicted_shipped_replies"],
                "predicted_capped_replies": table["predicted_capped_replies"],
                "predicted_reply_delta": table["predicted_reply_delta"],
                # SIGNED, never clamped.
                "predicted_reply_reduction": table["predicted_reply_reduction"],
                "excluded_terminal": sum(n["excluded_terminal"]
                                         for n in table["per_node"]),
                "excluded_synthetic": sum(n["excluded_synthetic"]
                                          for n in table["per_node"]),
            })
    crossover_payload = _csv_bytes(crossover_rows, crossover_columns)

    reach_numerator = sum(r["exposed_positive_mass_numerator"] for r in rows
                          if r["population"] == "selected_a")
    reach_denominator = sum(r["exposed_positive_mass_denominator"] for r in rows
                            if r["population"] == "selected_a")
    artifact = {
        "run_kind": RUN_KIND,
        "scientific_interpretation_forbidden": True,
        "scope_boundary": criteria.SCOPE_BOUNDARY,
        "search_execution_mode": SEARCH_EXECUTION_MODE,
        "simulations": SIMULATIONS,
        "add_noise": ADD_NOISE,
        "c_puct": cfg.c_puct,
        "batching_triple": list(BATCHING_TRIPLE),
        "cap_grid": list(CAP_GRID),
        "criteria_sha1": criteria_sha1,
        "universe_sha1": universe_sha1,
        "a_source_path": A_SOURCE,
        "a_source_sha1": a_source_sha1,
        "seed_audit": seed_audit,
        "authenticated_search_inputs": opening,
        "populations": {name: sum(1 for r in rows if r["population"] == name)
                        for name in POPULATIONS},
        "population_order": list(POPULATIONS),
        "cases": [{k: v for k, v in row.items() if k != "residual_leaves"}
                  for row in rows],
        "pooled_reach_numerator": reach_numerator,
        "pooled_reach_denominator": reach_denominator,
        # DESCRIPTIVE ONLY -- never a gate, a selector, or an exclusion rule.
        # A row whose shipped denominator is zero has an undefined ROW-LEVEL
        # ratio; it stays in the corpus and in the pooled sum regardless. This
        # records how often that happens so the reader can see it rather than
        # infer it from blanks in the CSV.
        "undefined_reply_reduction_by_population": {
            name: {str(cap): sum(
                1 for r in rows
                if r["population"] == name
                and r["crossover"][str(cap)]["predicted_reply_reduction"] is None)
                for cap in CAP_GRID}
            for name in POPULATIONS},
        "terminal_depth2_total": sum(r["terminal_depth2"] for r in rows),
        "total_depth2_total": sum(r["total_depth2"] for r in rows),
        # The CSVs are bound INTO the artifact, so a reader can prove the four
        # files belong to one run.
        "census_positions_sha1": hashlib.sha1(census_payload).hexdigest(),
        "residual_rows_sha1": hashlib.sha1(residual_payload).hexdigest(),
        "crossover_tables_sha1": hashlib.sha1(crossover_payload).hexdigest(),
        # The OPENING identity: the code that actually produced these trees.
        "source_sha1s": opening_runtime["source_sha1s"],
        "git_commit": opening_runtime["git_commit"],
        "worktree_clean": opening_runtime["worktree_clean"],
        "runtime_identity_bracketed": True,
    }
    artifact_payload = canonical_json_bytes(artifact)
    publish_atomically(out_dir, {
        "census_positions.csv": census_payload,
        "residual_rows.csv": residual_payload,
        "crossover_tables.csv": crossover_payload,
        "preflight_artifact.json": artifact_payload,
    })

    out = Path(out_dir)
    return {"rows": rows,
            "census_positions": str(out / "census_positions.csv"),
            "residual_rows": str(out / "residual_rows.csv"),
            "crossover_tables": str(out / "crossover_tables.csv"),
            "preflight_artifact": str(out / "preflight_artifact.json"),
            "preflight_artifact_sha1": hashlib.sha1(artifact_payload).hexdigest()}


def runtime_identity() -> Dict:
    """HEAD, the worktree state, and every result-determining source SHA-1.

    Captured BEFORE the evaluator and recomputed after the last search. A
    multi-hour run gives HEAD and the source tree ample time to move, and
    reading them only at emission would let the artifact describe bytes other
    than the ones that were imported and executed.
    """
    return {
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
        "source_sha1s": {path: fpu_provenance.file_sha1(path)
                         for path in MEASUREMENT_SOURCE_MODULES},
    }


def _authenticate_search_inputs(phase: str) -> Dict[str, str]:
    """Checkpoint and BOTH replay reservoirs, before the evaluator and again
    after the last search."""
    return {
        "checkpoint_sha1": capture_v18.authenticate_checkpoint(phase),
        "a_reservoir_sha1": capture_v18.authenticate_replay_reservoir(
            "opening" if phase == "opening" else "closing"),
        "census_reservoir_sha1": _authenticate_census_reservoir(),
    }


def _authenticate_census_reservoir() -> str:
    spec = next(u for u in control_pool.CANDIDATE_UNIVERSES
                if u["name"] == SELECTED_UNIVERSE["name"])
    paths = sorted(str(p) for p in Path(spec["replay_dir"]).glob("game_*.json"))
    actual = fpu_provenance.replay_data_sha1(paths)
    if actual != SELECTED_UNIVERSE["replay_data_sha1"]:
        raise ValueError(
            f"census reservoir replay_data_sha1 {actual} != pinned "
            f"{SELECTED_UNIVERSE['replay_data_sha1']}; no artifact written")
    return actual


def _default_evaluator():
    from .eval_runner import _default_evaluator_factory
    return _default_evaluator_factory(SELECTED_UNIVERSE["anchor_checkpoint"])


def build_parser():
    """Exactly three paths in, nothing else.

    No `--c-puct`, no gate subset, no population selector, no mode: every
    scientific parameter is frozen in the criteria or derived from the
    authenticated shipped config. A flag that can change a measured number is a
    flag that makes the artifact unreproducible.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", required=True,
                        help="the frozen criteria artifact")
    parser.add_argument("--universe", required=True,
                        help="the frozen Task 4 universe record")
    parser.add_argument("--out", required=True,
                        help="output directory; must NOT already exist")
    return parser


def main(argv=None) -> int:
    """ONE fixed production command. It derives both populations itself,
    selected-A first and the census second, into one bound artifact set."""
    args = build_parser().parse_args(argv)
    result = run_preflight(out_dir=args.out, criteria_path=args.criteria,
                           universe_path=args.universe)
    print(f"wrote {args.out}  preflight_artifact_sha1="
          f"{result['preflight_artifact_sha1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
