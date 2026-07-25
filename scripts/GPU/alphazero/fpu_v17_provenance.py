"""Frozen v17 constants and provenance records.

Task 4 of the v17 plan. This module is the single source of truth for every
value the frozen preregistration pins, and for the validators that refuse a run
whose configuration does not match them.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §2.4, §4, §6.1, §8.1, §10,
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
    value must be a frozen grid point. §13 forbids interpolating or extending."""
    if r is None or r == 0.0:
        return r
    if r not in GRID:
        raise ProtocolViolation(
            f"coefficient {r!r} is not in the frozen grid {GRID}; the design "
            f"§13 stop rules forbid adding, interpolating, or extending "
            f"coefficients after any scientific result")
    return r


def validate_seed_range(run_kind: str, base_seed: int, games: int) -> Tuple[int, int]:
    """Frozen per-stage seed range, plus disjointness from consumed evidence."""
    validate_run_kind(run_kind)
    expected = SEED_RANGES[run_kind]
    if expected is None:
        raise ProtocolViolation(
            f"run_kind {run_kind!r} has no generated seed range; it replays "
            f"fixed manifests")
    if (base_seed, games) != expected and (base_seed, games) != MATCH_SMOKE_SEEDS:
        raise ProtocolViolation(
            f"seed range [{base_seed}, {base_seed + games}) for {run_kind!r} != "
            f"the frozen [{expected[0]}, {expected[0] + expected[1]})")
    for c_base, c_games in CONSUMED_SEED_RANGES:
        if base_seed < c_base + c_games and c_base < base_seed + games:
            raise ProtocolViolation(
                f"seed range [{base_seed}, {base_seed + games}) overlaps consumed "
                f"evidence [{c_base}, {c_base + c_games}) (design §1.2)")
    return base_seed, games


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

def build_provenance(*, run_kind: str,
                     coefficient: Optional[float] = None,
                     checkpoints: Optional[Mapping[str, str]] = None,
                     source_files: Iterable[str] = (),
                     extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The §12 provenance block every v17 artifact carries.

    Contains no timestamp, so a canonical artifact built from the same inputs is
    byte-identical across reruns.
    """
    validate_run_kind(run_kind)
    validate_coefficient(coefficient)
    # `fpu_provenance.file_sha1` returns the sentinels "none"/"missing" instead
    # of raising, so an absent input is RECORDED rather than fatal. That is the
    # right behaviour for a fingerprint, but a v17 protocol must never freeze a
    # sentinel in place of a real hash -- refuse instead (the sentinel-leak
    # class the v16 evidence-chain review caught).
    hashed = {**{f"checkpoint:{k}": fpu_provenance.file_sha1(v)
                 for k, v in (checkpoints or {}).items()},
              **{f"source:{p}": fpu_provenance.file_sha1(str(p))
                 for p in source_files}}
    unresolved = sorted(k for k, v in hashed.items() if v in ("none", "missing"))
    if unresolved:
        raise ProtocolViolation(
            f"refusing to record placeholder hashes for {unresolved}; every "
            f"checkpoint and source file must be readable at protocol time")
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
        "checkpoints": {k: fpu_provenance.file_sha1(v)
                        for k, v in sorted((checkpoints or {}).items())},
        "source_file_sha1s": fpu_provenance.source_file_sha1s(sorted(source_files)),
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
        "runtime": fpu_provenance.runtime_provenance(),
    }
    if not record["scientific"]:
        record["scientific_interpretation_forbidden"] = True
    if extra:
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
