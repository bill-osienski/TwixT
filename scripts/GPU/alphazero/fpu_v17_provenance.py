"""Frozen v17 constants and provenance records.

Task 4 of the v17 plan. This module is the single source of truth for every
value the frozen preregistration pins, and for the validators that refuse a run
whose configuration does not match them.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 pinned below as `FROZEN_DESIGN_SHA1`) §2.4, §4, §6.1, §8.1, §10,
§11, §12.

Pure and import-light: hashing/git helpers are reused from `fpu_provenance`
rather than reimplemented, and nothing here imports an evaluator, MLX, or any
search code. No function in this module runs a search or touches a checkpoint's
weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from . import fpu_provenance

SCHEMA_VERSION = 1

# --- the frozen protocol --------------------------------------------------
FROZEN_DESIGN_PATH = ("docs/superpowers/specs/"
                      "2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md")
FROZEN_DESIGN_SHA1 = "944f358c0e3ef66503d2cbb56e31dabd145bafc2"
FROZEN_PLAN_PATH = ("docs/superpowers/plans/"
                    "2026-07-24-v17-baseline-preserving-policy-mass-fpu-plan.md")

FORMULA_ID = "fpu_v17_baseline_policy_mass"
FORMULA = "FPU = fpu_value - r*sqrt(clamp(P_explored, 0, 1)); fpu_value == 0.0"
CONFIG_FIELD = "fpu_shipped_policy_mass_reduction"

# design §4 -- the grid is frozen; §13 forbids extending it after any result.
GRID: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.35, 0.45)

# design §2.4 -- part of the MECHANISM, not a performance knob, because
# P_explored is read from COMPLETED backed-up visits.
BATCHING: Tuple[int, int, int] = (14, 48, 8)
BATCHING_FIELDS = ("eval_batch_size", "stall_flush_sims", "pending_virtual_visits")
MCTS_SIMS = 400
# design §5.2/§6.1/§8.1 -- board 24 throughout. It is also the only size these
# checkpoints play, and the size the n_legal >= 528 - ply geometry assumes.
BOARD_SIZE = 24

# design §12
OUTPUT_ROOT = "logs/eval/fpu_v17_baseline_policy_mass"
RUN_KINDS = ("tooling_smoke", "development", "held_out", "abcd", "strength",
             "external_validation")
SCIENTIFIC_RUN_KINDS = ("development", "held_out", "abcd", "strength",
                        "external_validation")

# Half-open [base, base+games) seed ranges, frozen per stage. `abcd` has none:
# it replays the fixed probe manifests under their own harness base seeds.
SEED_RANGES: Dict[str, Optional[Tuple[int, int]]] = {
    "tooling_smoke": (20309000, 32),        # §5.2; §5.4 match smoke uses 20309100+8
    "development": (20310000, 1600),        # §6.1
    "held_out": (20312000, 2200),           # §8.1
    "abcd": None,                           # §9
    "strength": (20320000, 800),            # §10
    "external_validation": (20330000, 800),  # §11
}
MATCH_SMOKE_SEEDS = (20309100, 8)           # §5.4, inside the tooling_smoke stage

# design §1.2 -- consumed evidence. A v17 run may never reuse these seeds.
CONSUMED_SEED_RANGES = (
    (20300000, 4000),   # v16 production reservoir
    (20280000, 400),    # v16 smoke_v1
    (20290000, 400),    # v16 smoke_v2
)


class ProtocolViolation(ValueError):
    """A frozen-protocol rule was broken. Always raised BEFORE any evaluator
    load, checkpoint read, or search."""


# --- validators -----------------------------------------------------------

def validate_run_kind(run_kind: str) -> str:
    if run_kind not in RUN_KINDS:
        raise ProtocolViolation(
            f"unknown run_kind {run_kind!r}; frozen set is {RUN_KINDS}")
    return run_kind


def is_scientific(run_kind: str) -> bool:
    return validate_run_kind(run_kind) in SCIENTIFIC_RUN_KINDS


def validate_batching(config: Any) -> Tuple[int, int, int]:
    """Design §2.4. Accepts an `MCTSConfig`-like object or a 3-tuple. Rejects
    ANY deviation, including the `MCTSConfig.stall_flush_sims` default of 16 --
    v17 must derive 48 explicitly."""
    if isinstance(config, (tuple, list)):
        triple = tuple(config)
    else:
        try:
            triple = tuple(getattr(config, f) for f in BATCHING_FIELDS)
        except AttributeError as exc:
            raise ProtocolViolation(
                f"cannot read the batching triple {BATCHING_FIELDS} from "
                f"{config!r}") from exc
    if triple != BATCHING:
        raise ProtocolViolation(
            f"batching triple {dict(zip(BATCHING_FIELDS, triple))} != the frozen "
            f"§2.4 triple {dict(zip(BATCHING_FIELDS, BATCHING))}; results at a "
            f"different triple are incomparable, not merely slower")
    return BATCHING


def validate_coefficient(r: Optional[float]) -> Optional[float]:
    """`None` (shipped) and `0.0` (exact zero) are always allowed; any positive
    value must be a frozen grid point. §13 forbids interpolating or extending.

    Type is checked before value: `bool` is a subclass of `int`, so `False`
    would otherwise satisfy `r == 0.0` and silently configure the shipped
    branch, and `True` would satisfy nothing but read as a number.
    """
    if r is None:
        return r
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise ProtocolViolation(
            f"coefficient must be None or a real number, got {r!r} "
            f"({type(r).__name__})")
    if r == 0.0:
        return r
    if r not in GRID:
        raise ProtocolViolation(
            f"coefficient {r!r} is not in the frozen grid {GRID}; the design "
            f"§13 stop rules forbid adding, interpolating, or extending "
            f"coefficients after any scientific result")
    return r


def _require_int(value: Any, *, name: str) -> int:
    """Seeds and counts are exact integers. `bool` is excluded (it subclasses
    `int`), and a float is refused even when integral: `20310000.0 == 20310000`
    would otherwise pass every comparison below while serializing to JSON as a
    different token."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolViolation(
            f"{name} must be an int, got {value!r} ({type(value).__name__})")
    return value


def validate_seed_range(run_kind: str, base_seed: int, games: int) -> Tuple[int, int]:
    """Frozen per-stage seed range, plus disjointness from consumed evidence."""
    validate_run_kind(run_kind)
    _require_int(base_seed, name="base_seed")
    _require_int(games, name="games")
    expected = SEED_RANGES[run_kind]
    if expected is None:
        raise ProtocolViolation(
            f"run_kind {run_kind!r} has no generated seed range; it replays "
            f"fixed manifests")
    # The §5.4 match-smoke block is a SECOND range inside the tooling_smoke
    # stage only. It must never widen any other stage's frozen range.
    allowed = {expected}
    if run_kind == "tooling_smoke":
        allowed.add(MATCH_SMOKE_SEEDS)
    if (base_seed, games) not in allowed:
        raise ProtocolViolation(
            f"seed range [{base_seed}, {base_seed + games}) for {run_kind!r} != "
            f"the frozen [{expected[0]}, {expected[0] + expected[1]})"
            + (f" (or the match-smoke [{MATCH_SMOKE_SEEDS[0]}, "
               f"{MATCH_SMOKE_SEEDS[0] + MATCH_SMOKE_SEEDS[1]}))"
               if run_kind == "tooling_smoke" else ""))
    for c_base, c_games in CONSUMED_SEED_RANGES:
        if base_seed < c_base + c_games and c_base < base_seed + games:
            raise ProtocolViolation(
                f"seed range [{base_seed}, {base_seed + games}) overlaps consumed "
                f"evidence [{c_base}, {c_base + c_games}) (design §1.2)")
    return base_seed, games


def validate_board_size(board_size: Any) -> int:
    if _require_int(board_size, name="board_size") != BOARD_SIZE:
        raise ProtocolViolation(
            f"board_size {board_size!r} != the frozen {BOARD_SIZE}")
    return BOARD_SIZE


def validate_output_path(path: Any) -> Path:
    """Design §12: v17 outputs live only under the v17 root, so no v16 artifact
    can be overwritten or a v17 artifact placed under a v16 root."""
    target = Path(path)
    root = Path(OUTPUT_ROOT)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ProtocolViolation(
            f"output path {target} is not under the v17 root {OUTPUT_ROOT}; v17 "
            f"may not write to a v16 root or overwrite a v16 artifact") from None
    return target


def require_clean_worktree(run_kind: str) -> None:
    """Design §12: a dirty tree is a PRE-RUN REFUSAL for scientific run kinds,
    not merely provenance to record. `tooling_smoke` may run dirty."""
    if is_scientific(run_kind) and not fpu_provenance.worktree_clean():
        raise ProtocolViolation(
            f"run_kind {run_kind!r} is scientific and requires a clean worktree; "
            f"resolve the repository intentionally first (this protocol "
            f"authorizes no automatic commit, stash, deletion, or cleanup)")


def require_not_tooling_smoke(doc: Mapping[str, Any], *, consumer_run_kind: str) -> None:
    """Design §1.2/§5: a tooling-smoke artifact can never feed a scientific
    stage, no matter how well-formed it is."""
    if is_scientific(consumer_run_kind) and doc.get("run_kind") == "tooling_smoke":
        raise ProtocolViolation(
            f"refusing to consume a tooling_smoke artifact in scientific run_kind "
            f"{consumer_run_kind!r}; smoke output has no scientific meaning")


# --- provenance record ----------------------------------------------------

SENTINEL_HASHES = ("none", "missing")

# Stages whose evidence IS a selector corpus: a manifest, its source index and
# the replay set behind it. Only these bind that triple.
#
# `abcd` is NOT exempt from authentication -- it consumes FIXED probe artifacts
# (the canonical A/B/C/D CSVs + the Task 1 freeze) and authenticates those,
# plus checkpoint and source identities. `strength` and `external_validation`
# likewise carry their own evidence-specific bindings. Requiring a selector
# corpus of them would mean inventing placeholders, which is weaker provenance,
# not stronger.
CORPUS_BOUND_RUN_KINDS = ("development", "held_out")


def binds_selector_corpus(run_kind: str) -> bool:
    return validate_run_kind(run_kind) in CORPUS_BOUND_RUN_KINDS

# Keys `extra` may never set. Anything a caller adds must not be able to
# restate the frozen protocol or flip a run's scientific status.
RESERVED_PROVENANCE_KEYS = frozenset({
    "schema_version", "formula_id", "formula", "config_field", "coefficient",
    "grid", "run_kind", "scientific", "scientific_interpretation_forbidden",
    "mcts", "frozen_design", "checkpoints", "source_file_sha1s",
    "manifest_sha1", "source_index_sha1", "replay_data_sha1", "identities",
    "git_commit", "worktree_clean", "runtime",
})


def _require_resolved(label: str, sha1: str) -> str:
    """`fpu_provenance.file_sha1` returns "none"/"missing" instead of raising,
    which is right for a fingerprint but must never be frozen into a v17
    protocol in place of a real hash (the sentinel-leak class the v16
    evidence-chain review caught)."""
    if sha1 in SENTINEL_HASHES:
        raise ProtocolViolation(
            f"refusing to record placeholder hash {sha1!r} for {label}; every "
            f"identity input must be readable at protocol time")
    return sha1


def _source_file_identities(source_files: Iterable[str]) -> Dict[str, str]:
    """`fpu_provenance.source_file_sha1s` keys by BASENAME so a fingerprint is
    checkout-location-independent. That is only sound while basenames are
    unique -- two same-named files in different packages would silently
    collapse to one entry, so assert it rather than assume it."""
    paths = sorted(str(p) for p in source_files)
    names = [Path(p).name for p in paths]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ProtocolViolation(
            f"source file basenames are not unique: {dupes}; basename-keyed "
            f"hashes would silently collapse them into one identity")
    hashes = fpu_provenance.source_file_sha1s(paths)
    for path, name in zip(paths, names):
        _require_resolved(f"source {path}", hashes[name])
    return hashes


def build_provenance(*, run_kind: str,
                     coefficient: Optional[float] = None,
                     checkpoints: Optional[Mapping[str, str]] = None,
                     source_files: Iterable[str] = (),
                     manifest: Optional[str] = None,
                     source_index: Optional[str] = None,
                     replay_paths: Optional[Iterable[str]] = None,
                     extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The §12 provenance block every v17 artifact carries.

    Records the full identity set §12 requires -- checkpoints, source files,
    manifest, source index, and replay DATA -- and refuses any placeholder
    hash. Contains no timestamp, so a canonical artifact built from the same
    inputs is byte-identical across reruns.
    """
    validate_run_kind(run_kind)
    validate_coefficient(coefficient)
    checkpoint_ids = {
        k: _require_resolved(f"checkpoint {k}", fpu_provenance.file_sha1(v))
        for k, v in sorted((checkpoints or {}).items())}
    identities: Dict[str, Optional[str]] = {
        "manifest_sha1": None if manifest is None else
        _require_resolved(f"manifest {manifest}", fpu_provenance.file_sha1(manifest)),
        "source_index_sha1": None if source_index is None else
        _require_resolved(f"source_index {source_index}",
                          fpu_provenance.file_sha1(source_index)),
        # `replay_data_sha1` hashes the CONTENTS in sorted-path order, so it
        # fingerprints the replay data rather than the paths. An unreadable
        # replay would raise there rather than yield a sentinel.
        "replay_data_sha1": None if replay_paths is None else
        fpu_provenance.replay_data_sha1(sorted(str(p) for p in replay_paths)),
    }
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "formula_id": FORMULA_ID,
        "formula": FORMULA,
        "config_field": CONFIG_FIELD,
        "coefficient": coefficient,
        "grid": list(GRID),
        "run_kind": run_kind,
        "scientific": is_scientific(run_kind),
        "mcts": {"n_simulations": MCTS_SIMS,
                 "add_noise": False,
                 **dict(zip(BATCHING_FIELDS, BATCHING))},
        "frozen_design": {"path": FROZEN_DESIGN_PATH, "sha1": FROZEN_DESIGN_SHA1},
        "checkpoints": checkpoint_ids,
        "source_file_sha1s": _source_file_identities(source_files),
        "identities": identities,
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
        "runtime": fpu_provenance.runtime_provenance(),
    }
    if not record["scientific"]:
        record["scientific_interpretation_forbidden"] = True
    if extra:
        collisions = sorted(set(extra) & (RESERVED_PROVENANCE_KEYS | set(record)))
        if collisions:
            raise ProtocolViolation(
                f"extra may not overwrite protected provenance keys {collisions}; "
                f"namespace caller metadata instead")
        record.update(extra)
    return record


def verify_frozen_design(path: str = FROZEN_DESIGN_PATH) -> str:
    """Refuse to proceed if the APPROVED - FROZEN protocol has been edited."""
    actual = fpu_provenance.file_sha1(path)
    if actual != FROZEN_DESIGN_SHA1:
        raise ProtocolViolation(
            f"frozen design {path} has SHA-1 {actual}, expected "
            f"{FROZEN_DESIGN_SHA1}; the preregistration is frozen and results "
            f"may not change its formula, grid, samples, gates, or stop rules")
    return actual
