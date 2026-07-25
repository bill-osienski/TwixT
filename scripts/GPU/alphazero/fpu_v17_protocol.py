"""v17 protocol emission, config derivation, and tamper detection.

Task 4 of the v17 plan. A v17 stage runs against an immutable *protocol*; its
*config* is a pure function of that protocol, so re-deriving the config and
byte-comparing it against the persisted one detects tampering with either --
before any evaluator is loaded.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 pinned once as `fpu_v17_provenance.FROZEN_DESIGN_SHA1`) §12.

Reuses, rather than reimplements, the v16 primitives that already have the
required semantics: `canonical_json_bytes` (sorted keys, ASCII, no NaN, single
trailing newline -- so artifacts are byte-reproducible) and `write_atomic`
(temp-file + `os.replace`, idempotent on identical bytes, refuses to overwrite
different bytes). Those modules are not modified.

Import-pure: no evaluator, MLX, or search import anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from . import fpu_v17_provenance as prov
from .fpu_dev_reservoir_protocol import (
    EXIT_GATE_FAIL,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_USAGE,
    WriteStatus,
    canonical_json_bytes,
    write_atomic,
)

__all__ = [
    "EXIT_OK", "EXIT_USAGE", "EXIT_MISMATCH", "EXIT_GATE_FAIL", "WriteStatus",
    "PROTOCOL_SCHEMA_VERSION", "CONFIG_SCHEMA_VERSION",
    "PROTOCOL_KEYS", "PROTOCOL_OPTIONAL_KEYS", "CONFIG_KEYS",
    "build_protocol", "derive_config", "emit", "load_json", "protocol_sha1",
    "verify_config_matches", "load_verified",
]

PROTOCOL_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1

# Exact required key sets. A document missing or gaining a top-level key is
# refused rather than partially interpreted.
PROTOCOL_KEYS = frozenset({
    "schema_version", "artifact_kind", "run_kind", "coefficient", "base_seed",
    "games", "board_size", "checkpoints", "provenance"})
PROTOCOL_OPTIONAL_KEYS = frozenset({"extra"})
CONFIG_KEYS = frozenset({
    "schema_version", "artifact_kind", "run_kind", "scientific", "formula_id",
    "config_field", "coefficient", "shipped_branch", "mcts", "seed_range",
    "board_size", "checkpoints", "frozen_design_sha1", "protocol_sha1"})


def protocol_sha1(protocol: Mapping[str, Any]) -> str:
    """SHA-1 of the protocol's CANONICAL bytes.

    The config embeds this, so the config is bound to the COMPLETE protocol --
    including the provenance block, which no other derived field reads. Without
    it, editing e.g. `provenance.formula_id` would leave the config verifying.
    """
    return hashlib.sha1(canonical_json_bytes(protocol)).hexdigest()


def _validate_shape(doc: Mapping[str, Any], *, artifact_kind: str,
                    required: frozenset, optional: frozenset,
                    schema_version: int) -> None:
    if doc.get("artifact_kind") != artifact_kind:
        raise prov.ProtocolViolation(
            f"expected a {artifact_kind} document, got artifact_kind="
            f"{doc.get('artifact_kind')!r}")
    if doc.get("schema_version") != schema_version:
        raise prov.ProtocolViolation(
            f"{artifact_kind} schema_version {doc.get('schema_version')!r} != "
            f"the supported {schema_version}")
    keys = set(doc)
    if missing := sorted(required - keys):
        raise prov.ProtocolViolation(
            f"{artifact_kind} is missing required keys {missing}")
    if unknown := sorted(keys - required - optional):
        raise prov.ProtocolViolation(
            f"{artifact_kind} has unknown keys {unknown}")


def build_protocol(*, run_kind: str,
                   coefficient: Optional[float] = None,
                   base_seed: Optional[int] = None,
                   games: Optional[int] = None,
                   board_size: int = 24,
                   checkpoints: Optional[Mapping[str, str]] = None,
                   source_files: Any = (),
                   extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Validate every frozen rule, then build the immutable protocol document.

    Raises `ProtocolViolation` before touching the filesystem or a checkpoint's
    weights. A scientific `run_kind` additionally requires a clean worktree.
    """
    prov.validate_run_kind(run_kind)
    prov.validate_coefficient(coefficient)
    board_size = prov.validate_board_size(board_size)
    prov.verify_frozen_design()
    prov.require_clean_worktree(run_kind)
    if prov.SEED_RANGES[run_kind] is not None:
        if base_seed is None or games is None:
            raise prov.ProtocolViolation(
                f"run_kind {run_kind!r} requires base_seed and games")
        prov.validate_seed_range(run_kind, base_seed, games)
    elif base_seed is not None or games is not None:
        raise prov.ProtocolViolation(
            f"run_kind {run_kind!r} generates no games; base_seed/games must be "
            f"omitted")
    doc: Dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": "protocol",
        "run_kind": run_kind,
        "coefficient": coefficient,
        "base_seed": base_seed,
        "games": games,
        "board_size": board_size,
        "checkpoints": dict(sorted((checkpoints or {}).items())),
        "provenance": prov.build_provenance(
            run_kind=run_kind, coefficient=coefficient,
            checkpoints=checkpoints, source_files=source_files),
    }
    if extra:
        doc["extra"] = dict(extra)
    return doc


def derive_config(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """The stage config, as a PURE function of the protocol.

    Deterministic and total: same protocol in, byte-identical config out. This
    is what `verify_config_matches` re-runs to detect tampering.
    """
    _validate_shape(protocol, artifact_kind="protocol",
                    required=PROTOCOL_KEYS, optional=PROTOCOL_OPTIONAL_KEYS,
                    schema_version=PROTOCOL_SCHEMA_VERSION)
    run_kind = prov.validate_run_kind(protocol["run_kind"])
    coefficient = prov.validate_coefficient(protocol.get("coefficient"))
    board_size = prov.validate_board_size(protocol.get("board_size"))
    base_seed, games = protocol.get("base_seed"), protocol.get("games")
    if prov.SEED_RANGES[run_kind] is not None:
        prov.validate_seed_range(run_kind, base_seed, games)
    elif base_seed is not None or games is not None:
        raise prov.ProtocolViolation(
            f"run_kind {run_kind!r} generates no games; base_seed/games must be "
            f"null")
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "artifact_kind": "config",
        "run_kind": run_kind,
        "scientific": prov.is_scientific(run_kind),
        "formula_id": prov.FORMULA_ID,
        "config_field": prov.CONFIG_FIELD,
        # None and 0.0 both mean the shipped branch (§2.2); the field still
        # records which was configured so provenance can tell them apart.
        "coefficient": coefficient,
        "shipped_branch": coefficient is None or coefficient == 0.0,
        "mcts": {"n_simulations": prov.MCTS_SIMS,
                 "add_noise": False,
                 **dict(zip(prov.BATCHING_FIELDS, prov.BATCHING))},
        "seed_range": (None if base_seed is None
                       else [base_seed, base_seed + games]),
        "board_size": board_size,
        "checkpoints": dict(sorted((protocol.get("checkpoints") or {}).items())),
        "frozen_design_sha1": prov.FROZEN_DESIGN_SHA1,
        # binds the config to the COMPLETE protocol, provenance included
        "protocol_sha1": protocol_sha1(protocol),
    }


def emit(path: Any, doc: Mapping[str, Any]) -> WriteStatus:
    """Canonical + atomic + immutable write under the v17 output root.

    Re-emitting identical bytes is a no-op (`UNCHANGED`); different bytes at an
    existing path raise rather than clobber a frozen artifact.
    """
    target = prov.validate_output_path(path)
    return write_atomic(target, canonical_json_bytes(doc))


def load_json(path: Any) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def verify_config_matches(protocol: Mapping[str, Any],
                          config: Mapping[str, Any]) -> None:
    """Re-derive the config from the protocol and BYTE-compare.

    Catches tampering with either document, including edits that keep the JSON
    well-formed. Raises `ProtocolViolation`; callers map that to
    `EXIT_MISMATCH`.
    """
    _validate_shape(config, artifact_kind="config", required=CONFIG_KEYS,
                    optional=frozenset(), schema_version=CONFIG_SCHEMA_VERSION)
    expected = canonical_json_bytes(derive_config(protocol))
    actual = canonical_json_bytes(config)
    if expected != actual:
        raise prov.ProtocolViolation(
            "config does not byte-match the config re-derived from its "
            "protocol; one of the two has been modified since emission")


def load_verified(protocol_path: Any, config_path: Any, *,
                  consumer_run_kind: str) -> Dict[str, Any]:
    """Load a protocol/config pair for `consumer_run_kind`, enforcing every
    §12 rule BEFORE an evaluator is loaded. Returns the verified config."""
    prov.validate_run_kind(consumer_run_kind)
    protocol = load_json(protocol_path)
    config = load_json(config_path)
    prov.require_not_tooling_smoke(protocol, consumer_run_kind=consumer_run_kind)
    prov.require_not_tooling_smoke(config, consumer_run_kind=consumer_run_kind)
    if protocol.get("run_kind") != consumer_run_kind:
        raise prov.ProtocolViolation(
            f"protocol run_kind {protocol.get('run_kind')!r} != consumer "
            f"{consumer_run_kind!r}")
    prov.verify_frozen_design()
    prov.require_clean_worktree(consumer_run_kind)
    verify_config_matches(protocol, config)
    prov.validate_batching([config["mcts"][f] for f in prov.BATCHING_FIELDS])
    return config
