"""v18 A/6,400 capture parameterization and reference-bundle builder.

Spec Sec 2.2.2 requires a PARAMETERIZATION, not a fork, with the v17 default
preserved byte-for-byte. Plan Task 6.

WHY THIS IS A SEPARATE MODULE, NOT AN EDIT TO capture_v17_abcd_selected_moves.

That file is frozen v17 EVIDENCE. Two facts are recorded about it in the tracked
v17 baseline record `logs/eval/fpu_v17_baseline_policy_mass/prechange_baseline.json`:

    selected_move_capture.capture_tool_sha1     2ce39bb56b479ae792e20fa6b493e157b3b89d05
    source_sha1s.v17_pure_dependencies[...]     2ce39bb56b479ae792e20fa6b493e157b3b89d05

The first states which version of the tool produced the v17 selected-move
artifact `162c9a5a...`. Editing the tool in place would make that statement
FALSE -- the file on disk would no longer be the file that produced the
artifact -- and "fixing" the record would assert that a completed, closed
experiment was produced by a tool version that did not exist when it ran. The
v17 policy-mass line is closed; its evidence must stay frozen.

So the v17 module is imported and reused UNCHANGED. This is a parameterization
in the sense Sec 2.2.2 means: the shared per-case machinery -- `load_gate_cases`,
`selected_move_for`, `mcts_config`'s primitives, the gate table, the seeds and
the batching triple -- has exactly one implementation, over there. Nothing is
copied. The legacy mode DELEGATES to `v17.capture()` verbatim, so the default
document is byte-identical by construction rather than by inspection.

The mode fixes every scientific parameter. There is deliberately no
caller-nominated reference: a caller free to nominate its own authentication
source is not authenticated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from . import capture_v17_abcd_selected_moves as v17
from . import fpu_provenance
from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .fpu_state_hash import canonical_state_sha1
from .position_probe_cases import position_state
from .v18_control_pool import FORBIDDEN_REPLAY_RESERVOIRS, SELECTED_UNIVERSE

# Re-exported from the v17 module so this module states no constant of its own.
CHECKPOINT = v17.CHECKPOINT
CHECKPOINT_ROW_FILTER = v17.CHECKPOINT_ROW_FILTER
OUT = v17.OUT
MCTS_SIMS = v17.MCTS_SIMS
EVAL_BATCH_SIZE = v17.EVAL_BATCH_SIZE
STALL_FLUSH_SIMS = v17.STALL_FLUSH_SIMS
PENDING_VIRTUAL_VISITS = v17.PENDING_VIRTUAL_VISITS
GATES = v17.GATES
TOLERANCE = v17.TOLERANCE
sha1 = v17.sha1
load_gate_cases = v17.load_gate_cases
selected_move_for = v17.selected_move_for

A6400_SOURCE = ("logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/"
                "position_probe_cases.csv")
A6400_SOURCE_SHA1 = "a17d4737c747e2799253bebbc3d0261e0e697114"
A6400_EXPECTED_CASES = 30

# Every row of the frozen source points into this reservoir, and the canonical
# position hashes are RECONSTRUCTED from its replay bytes -- so those bytes are
# part of the authentication chain. The pin is IMPORTED from the control pool,
# never restated here: one definition, one aggregate.
A6400_REPLAY_RESERVOIR = next(
    r for r in FORBIDDEN_REPLAY_RESERVOIRS if r["name"] == "seed20115")

# The checkpoint the whole experiment is pinned to, IMPORTED from the already
# authenticated definition rather than restated. `_default_evaluator_factory`
# loads these bytes, so they must be bound before the evaluator exists.
assert CHECKPOINT == SELECTED_UNIVERSE["anchor_checkpoint"], CHECKPOINT
A6400_CHECKPOINT_SHA1 = SELECTED_UNIVERSE["checkpoint_sha1s"][CHECKPOINT]

MODES = {
    "v17_prechange_abcd": {
        "gates": ("A", "B", "C", "D"),
        "mcts_sims": MCTS_SIMS,
        "auth_source": None,        # each gate authenticates against its own CSV
        "auth_sha1": None,
        "base_seed": None,          # per-gate
        "seed_rule": None,
        "batching": (EVAL_BATCH_SIZE, STALL_FLUSH_SIMS, PENDING_VIRTUAL_VISITS),
        "legacy_schema": True,
    },
    "v18_preflight_a6400": {
        "gates": ("A",),
        "mcts_sims": 6400,
        "auth_source": A6400_SOURCE,
        "auth_sha1": A6400_SOURCE_SHA1,
        "base_seed": 20260616,
        "seed_rule": "base ^ game_idx ^ position_ply",
        "batching": (EVAL_BATCH_SIZE, STALL_FLUSH_SIMS, PENDING_VIRTUAL_VISITS),
        "legacy_schema": False,
    },
}

# The v17 document's exact top-level key set. New fields appear ONLY in the v18
# mode: emitting them unconditionally would change the default bytes and break
# the byte-identity regression this task also demands.
LEGACY_DOCUMENT_KEYS = (
    "record_kind", "schema_version", "experiment", "scope",
    "scientific_interpretation", "checkpoint", "checkpoint_sha1", "mcts",
    "authentication", "source_sha1s", "gates",
)
# The v18 document's EXACT key set -- the one `capture()` constructs. Advertising
# "legacy plus additions" while emitting a different shape means the advertised
# schema is never what a consumer receives.
V18_DOCUMENT_KEYS = (
    "record_kind", "schema_version", "run_kind",
    "scientific_interpretation_forbidden", "mode", "mcts_sims", "gate_list",
    "auth_source", "auth_source_sha1", "replay_reservoir",
    "replay_reservoir_sha1", "checkpoint", "checkpoint_sha1", "mcts",
    "source_case_count", "cases", "authentication",
)

# Equal `case_id` SETS are not enough: authentication compares against a
# DIFFERENT artifact, so every identifying field must agree.
CASE_IDENTITY = ("case_id", "game_idx", "position_ply", "side_to_move",
                 "replay_path", "canonical_state_sha1")

A6400_BUNDLE_KEYS = (
    "artifact_kind", "schema_version",
    "capture_run_1_path", "capture_run_1_sha1",
    "capture_run_2_path", "capture_run_2_sha1",
    "byte_identical",
    "historical_source_path", "historical_source_sha1",
    "authentication",
    "run_kind", "scientific_interpretation_forbidden",
)


def _read_bytes(path):
    """Named seam: the ONE read of a frozen artifact."""
    with open(path, "rb") as f:
        return f.read()


def sha1_bytes(raw):
    return hashlib.sha1(raw).hexdigest()


def _parse_source_rows(raw):
    return list(csv.DictReader(raw.decode().splitlines()))


def mcts_config(mcts_sims: int = MCTS_SIMS):
    """The v17 configuration with the sim count parameterized. The batching
    triple is preserved at every sim count."""
    return cfg_from(EvalConfig(mcts_sims=mcts_sims,
                               mcts_eval_batch_size=EVAL_BATCH_SIZE,
                               mcts_stall_flush_sims=STALL_FLUSH_SIMS))


def resolve_gates(s):
    """Parse a gate subset, rejecting any name outside A/B/C/D."""
    gates = tuple(g.strip() for g in s.split(",") if g.strip())
    unknown = [g for g in gates if g not in GATES]
    if unknown or not gates:
        raise ValueError(f"unknown gate(s) {unknown or s!r}; known: {sorted(GATES)}")
    return gates


def document_keys(mode):
    """Top-level keys the emitted document carries in this mode."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}")
    if MODES[mode]["legacy_schema"]:
        return tuple(LEGACY_DOCUMENT_KEYS)
    return tuple(V18_DOCUMENT_KEYS)


def record_envelope(mode):
    """The v18 scope stamps. The v17 default carries none of them, by design."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}")
    if MODES[mode]["legacy_schema"]:
        return {}
    return {"run_kind": mode, "scientific_interpretation_forbidden": True}


def _identity(row):
    """The full six-field identity, STRICT and NORMALIZED.

    Every field must be present: `.get()`-style leniency let a source and a
    capture that BOTH omit `canonical_state_sha1` compare as `None == None`, so
    an incomplete capture authenticated against an incomplete source.

    `game_idx` and `position_ply` are normalized to `int`. The historical CSV
    supplies them as strings (`"347"`, `"73"`) while a capture emits integers,
    so an unnormalized comparison reports all 30 cases missing AND all 30
    unexpected -- a total failure that says nothing about the data.
    """
    missing = [f for f in CASE_IDENTITY
               if row.get(f) is None or row.get(f) == ""]
    if missing:
        raise ValueError(
            f"case set cannot be computed: row {row.get('case_id')!r} is "
            f"missing identity field(s) {missing}")
    return (str(row["case_id"]), int(row["game_idx"]), int(row["position_ply"]),
            str(row["side_to_move"]), str(row["replay_path"]),
            str(row["canonical_state_sha1"]))


def canonical_state_sha1_for(row):
    """The REAL canonical position hash, reconstructed from replay bytes.

    The historical CSV carries no such column, so it is derived rather than
    read. A metadata digest would authenticate nothing about the position.
    """
    replay = json.loads(_read_bytes(row["replay_path"]).decode())
    state = position_state(replay, int(row["position_ply"]), row["side_to_move"])
    return canonical_state_sha1(state)


RESERVOIR_PHASES = ("pre_derivation", "opening", "closing")


def authenticate_replay_reservoir(phase="opening"):
    """Bind the reservoir the canonical hashes are reconstructed from, using the
    aggregate pinned in `v18_control_pool` -- imported, never duplicated.

    `phase` names WHY this call happens, so the real three-check structure is
    observable: `pre_derivation` (before canonical hashes are reconstructed),
    `opening` (before the evaluator exists) and `closing` (after the last
    search). Each guards a different span of the run.
    """
    if phase not in RESERVOIR_PHASES:
        raise ValueError(f"unknown phase {phase!r}; known: {RESERVOIR_PHASES}")
    res = A6400_REPLAY_RESERVOIR
    paths = sorted(str(p) for p in Path(res["dir"]).glob("game_*.json"))
    if len(paths) != res["n_games"]:
        raise ValueError(
            f"{res['name']}: {len(paths)} replays, {res['n_games']} pinned")
    actual = fpu_provenance.replay_data_sha1(paths)
    if actual != res["replay_data_sha1"]:
        raise ValueError(
            f"{res['name']}: replay_data_sha1 {actual} != pinned "
            f"{res['replay_data_sha1']} ({phase})")
    return actual


def authenticate_checkpoint(when="opening"):
    """Bind the weights the evaluator searches with.

    Hashing the checkpoint AFTER the searches would let the artifact name bytes
    other than the ones actually loaded.
    """
    actual = sha1(CHECKPOINT)
    if actual != A6400_CHECKPOINT_SHA1:
        raise ValueError(
            f"checkpoint {CHECKPOINT} hashes {actual}, pinned "
            f"{A6400_CHECKPOINT_SHA1} ({when}); no artifact written")
    return actual


def _assert_well_formed(label, rows, expected_cases):
    if expected_cases is not None and len(rows) != expected_cases:
        raise ValueError(
            f"case set: {label} has {len(rows)} rows, expected exactly "
            f"{expected_cases}")
    case_ids = [str(r["case_id"]) for r in rows]
    if len(set(case_ids)) != len(case_ids):
        dupes = sorted({c for c in case_ids if case_ids.count(c) > 1})[:3]
        raise ValueError(f"case set: {label} has duplicate case_id(s) {dupes}")
    identities = [_identity(r) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"case set: {label} has duplicate full identities")
    return identities


def authentication_report(source_rows, captured):
    """One entry per case, `case_id` ascending. PURE AND TOTAL: it reports, it
    does not raise -- raising is `authenticate_against`'s job."""
    by_id = {r["case_id"]: r for r in captured}
    out = []
    for src in sorted(source_rows, key=lambda r: r["case_id"]):
        cap = by_id.get(src["case_id"])
        if cap is None:
            out.append({"case_id": src["case_id"], "ok": False,
                        "reason": "missing_from_capture"})
            continue
        frozen_value = float(src["probe_black_root_value"])
        frozen_share = float(src["probe_top1_share"])
        got_value = float(cap["recomputed_black_value_repr"])
        got_share = float(cap["top_share_repr"])
        value_delta = abs(got_value - frozen_value)
        share_delta = abs(got_share - frozen_share)
        out.append({
            "case_id": src["case_id"],
            "identity": list(_identity(src)),
            "frozen_black_root_value_repr": repr(frozen_value),
            "captured_black_value_repr": repr(got_value),
            "abs_value_delta_repr": repr(value_delta),
            "frozen_top1_share_repr": repr(frozen_share),
            "captured_top_share_repr": repr(got_share),
            "abs_top_share_delta_repr": repr(share_delta),
            "ok": value_delta < TOLERANCE and share_delta < TOLERANCE,
        })
    return out


def authenticate_against(source_rows, captured, expected_cases=None):
    """Exact identity on the full tuple, then BOTH statistics within TOLERANCE.

    Returns the authentication report on success, so the bundle's 30-entry block
    has a defined provenance rather than being rebuilt from somewhere else.
    """
    src_list = _assert_well_formed("source", source_rows, expected_cases)
    cap_list = _assert_well_formed("capture", captured, expected_cases)
    if len(src_list) != len(cap_list):
        raise ValueError(
            f"case set: source has {len(src_list)} rows, capture has "
            f"{len(cap_list)}")
    src_ids, cap_ids = set(src_list), set(cap_list)
    if src_ids != cap_ids:
        missing = sorted(map(str, src_ids - cap_ids))[:3]
        extra = sorted(map(str, cap_ids - src_ids))[:3]
        raise ValueError(
            f"case set mismatch on the full identity tuple {CASE_IDENTITY}: "
            f"{len(src_ids - cap_ids)} missing (e.g. {missing}), "
            f"{len(cap_ids - src_ids)} unexpected (e.g. {extra})")

    report = authentication_report(source_rows, captured)
    for entry in report:
        if entry["ok"]:
            continue
        if float(entry["abs_top_share_delta_repr"]) >= TOLERANCE:
            raise ValueError(
                f"{entry['case_id']}: top_share differs by "
                f"{entry['abs_top_share_delta_repr']}, tolerance {TOLERANCE}")
        raise ValueError(
            f"{entry['case_id']}: black value differs by "
            f"{entry['abs_value_delta_repr']}, tolerance {TOLERANCE}")
    return report


def _load_frozen_a6400_source():
    """The frozen historical A/6,400 source.

    TAKES NO ARGUMENTS: the path is not a parameter, so substitution is
    impossible by construction. Authenticates THE BYTES IT PARSES -- one read,
    hashed and parsed as the same in-memory object. Hashing a path and then
    parsing a path is two reads, and a file changed between them would
    authenticate one byte sequence and parse another.
    """
    raw = _read_bytes(MODES["v18_preflight_a6400"]["auth_source"])
    actual = sha1_bytes(raw)
    if actual != MODES["v18_preflight_a6400"]["auth_sha1"]:
        raise ValueError(
            f"frozen A/6400 source hashes {actual}, pinned "
            f"{MODES['v18_preflight_a6400']['auth_sha1']}")
    rows = _parse_source_rows(raw)
    if len(rows) != A6400_EXPECTED_CASES:
        raise ValueError(
            f"frozen A/6400 source has {len(rows)} rows, expected "
            f"{A6400_EXPECTED_CASES}")
    return _enrich_source_rows(rows)


def _enrich_source_rows(rows):
    """Normalize the numeric identity fields and supply the canonical position
    hash the historical CSV does not carry.

    Derivation reads replay bytes, so the reservoir is bound FIRST -- and only
    when a derivation is actually needed, because binding bytes nobody reads
    proves nothing. Rows that already carry the hash pass through, which keeps
    this idempotent.
    """
    if any(not r.get("canonical_state_sha1") for r in rows):
        authenticate_replay_reservoir("pre_derivation")
    return [dict(r,
                 game_idx=int(r["game_idx"]),
                 position_ply=int(r["position_ply"]),
                 canonical_state_sha1=(r.get("canonical_state_sha1")
                                       or canonical_state_sha1_for(r)))
            for r in rows]


def preflight_case_identities(source_rows, gate_rows):
    """Compare the FULL identity sets BEFORE any evaluator is constructed.

    A 30 x 6,400-simulation capture is expensive. Every identity defect --
    string-vs-int, a missing canonical hash, a duplicate, a wrong case set --
    is detectable from metadata alone, so it must be detected before a single
    simulation runs rather than after.
    """
    src = _assert_well_formed("source", source_rows, A6400_EXPECTED_CASES)
    gate = _assert_well_formed("gate rows", gate_rows, A6400_EXPECTED_CASES)
    if set(src) != set(gate):
        missing = sorted(map(str, set(src) - set(gate)))[:3]
        extra = sorted(map(str, set(gate) - set(src)))[:3]
        raise ValueError(
            f"case set mismatch BEFORE search: {len(set(src) - set(gate))} "
            f"missing (e.g. {missing}), {len(set(gate) - set(src))} unexpected "
            f"(e.g. {extra}); refusing to spend a 6,400-sim capture")
    return src


def assert_writable_out(out) -> str:
    """Refuse a missing destination, and refuse the frozen v17 evidence path.

    `OUT` is re-exported from the v17 module for identity comparisons, and it is
    ALSO the path of the frozen v17 selected-move artifact `162c9a5a...` that
    `prechange_baseline.json` pins as the output of capture tool `2ce39bb5...`.
    That file is gitignored, so an overwrite destroys evidence git cannot
    restore -- unlike `prechange_baseline.json` beside it, which is tracked.

    So there is no default destination, and the one path that looks like the
    obvious default is precisely the one refused. The check runs BEFORE mode
    dispatch, so neither rejection constructs an evaluator, reads a replay, nor
    delegates to `v17.capture()`.
    """
    if not out:
        raise ValueError(
            "capture() requires an explicit out path: there is deliberately no "
            "default, because the only obvious-looking default is the frozen "
            "v17 evidence artifact")
    if Path(out).resolve() == Path(OUT).resolve():
        raise ValueError(
            f"refusing to write {out}: that is the protected frozen v17 "
            f"evidence artifact {OUT}. Capture to a fresh path and compare "
            f"against it; never through it")
    return str(out)


def capture(mode: str = "v17_prechange_abcd", out=None):
    """Run a capture in the named mode.

    The legacy mode DELEGATES to `v17.capture()` unchanged, so the default
    document is byte-identical by construction.
    """
    assert_writable_out(out)
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
    spec_mode = MODES[mode]
    if spec_mode["legacy_schema"]:
        return v17.capture()

    cfg = mcts_config(spec_mode["mcts_sims"])
    assert (cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits) \
        == spec_mode["batching"], cfg
    actual = sha1(spec_mode["auth_source"])
    if actual != spec_mode["auth_sha1"]:
        raise ValueError(
            f"auth source hashes {actual}, pinned {spec_mode['auth_sha1']}")

    # --- everything below the evaluator line costs GPU time -----------------
    source_rows = _load_frozen_a6400_source()
    gate_rows = []
    for gate in spec_mode["gates"]:
        for row in load_gate_cases(gate):
            gate_rows.append(dict(row,
                                  game_idx=int(row["game_idx"]),
                                  position_ply=int(row["position_ply"]),
                                  canonical_state_sha1=canonical_state_sha1_for(row)))
    preflight_case_identities(source_rows, gate_rows)

    # OPENING authentication of every mutable input the searches will consume.
    # `selected_move_for` reopens replay files throughout a multi-hour run, and
    # the evaluator loads the checkpoint bytes, so both are bound BEFORE the
    # evaluator exists and re-bound after the last search.
    opening = {
        "checkpoint_sha1": authenticate_checkpoint("opening"),
        "replay_reservoir_sha1": authenticate_replay_reservoir("opening"),
    }
    # --- only now is a 6,400-sim capture worth starting ---------------------

    evaluator = _default_evaluator_factory(CHECKPOINT)
    cases = []
    for row in gate_rows:
        result = selected_move_for(evaluator, row, cfg, spec_mode["base_seed"],
                                   spec_mode["seed_rule"])
        cases.append({"case_id": row["case_id"],
                      "game_idx": row["game_idx"],
                      "position_ply": row["position_ply"],
                      "side_to_move": row["side_to_move"],
                      "replay_path": row["replay_path"],
                      "canonical_state_sha1": row["canonical_state_sha1"],
                      **result})

    # CLOSING authentication, before the document exists. A replay or a
    # checkpoint that changed mid-run would otherwise be described by identities
    # taken at a different moment than the bytes actually searched.
    closing_checkpoint = authenticate_checkpoint("closing")
    closing_reservoir = authenticate_replay_reservoir("closing")
    if (closing_checkpoint != opening["checkpoint_sha1"]
            or closing_reservoir != opening["replay_reservoir_sha1"]):
        raise ValueError(
            f"search inputs changed during the capture: checkpoint "
            f"{opening['checkpoint_sha1']} -> {closing_checkpoint}, reservoir "
            f"{opening['replay_reservoir_sha1']} -> {closing_reservoir}. "
            f"No artifact written")

    document = {
        "record_kind": "v18_preflight_a6400_selected_moves",
        "schema_version": 1,
        **record_envelope(mode),
        "mode": mode,
        "mcts_sims": spec_mode["mcts_sims"],
        "gate_list": list(spec_mode["gates"]),
        "auth_source": spec_mode["auth_source"],
        "auth_source_sha1": spec_mode["auth_sha1"],
        "replay_reservoir": A6400_REPLAY_RESERVOIR["dir"],
        "replay_reservoir_sha1": opening["replay_reservoir_sha1"],
        "checkpoint": CHECKPOINT,
        "checkpoint_sha1": opening["checkpoint_sha1"],
        "mcts": {"n_simulations": spec_mode["mcts_sims"],
                 "batching_triple": list(spec_mode["batching"]),
                 "add_noise": False,
                 "base_seed": spec_mode["base_seed"],
                 "seed_rule": spec_mode["seed_rule"]},
        "source_case_count": len(source_rows),
        "cases": cases,
        "authentication": authenticate_against(source_rows, cases,
                                               A6400_EXPECTED_CASES),
    }
    if set(document) != set(document_keys(mode)):
        raise ValueError(
            f"emitted v18 document keys {sorted(set(document))} do not match "
            f"the frozen schema {sorted(set(document_keys(mode)))}")
    return document


def build_a6400_bundle_document(run1_path, run1_sha1, run2_path, run2_sha1,
                                byte_identical, source_path, source_sha1,
                                authentication):
    """PURE: already-loaded, already-authenticated inputs in, dict out.

    The bundle never carries its own digest -- a file cannot contain the hash of
    its own complete bytes. `build_a6400_reference_bundle` RETURNS it instead.
    """
    document = {
        "artifact_kind": "v18_a6400_reference_bundle",
        "schema_version": 1,
        "capture_run_1_path": run1_path,
        "capture_run_1_sha1": run1_sha1,
        "capture_run_2_path": run2_path,
        "capture_run_2_sha1": run2_sha1,
        "byte_identical": byte_identical,
        "historical_source_path": source_path,
        "historical_source_sha1": source_sha1,
        "authentication": authentication,
        "run_kind": "v18_a6400_reference_bundle",
        "scientific_interpretation_forbidden": True,
    }
    if set(document) != set(A6400_BUNDLE_KEYS):
        raise ValueError("bundle key set drifted from A6400_BUNDLE_KEYS")
    return document


def build_a6400_reference_bundle(run1_path, run2_path, out_path):
    """Emit the reference bundle; return the SHA-1 of the bytes written.

    A bundle over two DIFFERING captures is not a valid artifact: its single
    `authentication` block would be undefined as to which run it describes. The
    builder refuses rather than recording `byte_identical: false`.
    """
    raw1, raw2 = _read_bytes(run1_path), _read_bytes(run2_path)
    if raw1 != raw2:
        raise ValueError(
            f"captures are not byte-identical: {run1_path} and {run2_path}. A "
            f"reference bundle over differing runs has no defined "
            f"authentication block; nothing written")

    doc1 = json.loads(raw1.decode())
    doc2 = json.loads(raw2.decode())
    source_rows = _load_frozen_a6400_source()
    report1 = authenticate_against(source_rows, doc1["cases"])
    report2 = authenticate_against(source_rows, doc2["cases"])
    if canonical_json_bytes(report1) != canonical_json_bytes(report2):
        raise ValueError(
            "authentication is not a pure function of the capture bytes: two "
            "byte-identical captures produced different reports")

    document = build_a6400_bundle_document(
        run1_path=run1_path, run1_sha1=sha1_bytes(raw1),
        run2_path=run2_path, run2_sha1=sha1_bytes(raw2),
        byte_identical=True,
        source_path=MODES["v18_preflight_a6400"]["auth_source"],
        source_sha1=MODES["v18_preflight_a6400"]["auth_sha1"],
        authentication=report1)

    return _atomic_write(out_path, canonical_json_bytes(document))


def _atomic_write(out_path, payload):
    """Temp file plus rename: a mid-write failure leaves the destination
    untouched and no temp file behind."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(payload)
        os.replace(tmp, str(out))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return sha1_bytes(payload)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="v17_prechange_abcd", choices=sorted(MODES))
    # REQUIRED, and no default: see `assert_writable_out`.
    ap.add_argument("--out", required=True,
                    help="destination path; must not be the frozen v17 "
                         "evidence artifact")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    record = capture(mode=args.mode, out=args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, sort_keys=True, indent=1)
        f.write("\n")
    print(f"wrote {args.out}  sha1={sha1(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
