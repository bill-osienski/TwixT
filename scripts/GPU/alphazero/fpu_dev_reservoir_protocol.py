"""FPU (policy-mass) v2 reservoir-protocol schema + canonical-JSON/atomic-write
primitives + the `emit-protocol` builder.

Frozen design ref: docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md
  Sec 2.1 (the `reservoir_protocol.json` schema -- the single source of ALL
  declared pre-generation decisions), Sec 3 (CLI stages / exit codes /
  atomicity+immutability contract), Sec 8 (canonical JSON, determinism,
  reviewability).
Pre-op hardening plan ref: docs/superpowers/plans/2026-07-14-fpu-v2-preop-hardening-plan.md
  Task B1 -- the FIRST task of the new Group-2 subsystem (B1-B11), which
  will qualify a generated reservoir zero-GPU (B3-B7) and emit an immutable
  `fpu_dev_corpus_v2_config.json` (B7-B10). B1 lays the foundation ONLY:
  the protocol's field set (`PROTOCOL_SCHEMA_KEYS`), the canonical-JSON
  encoder, the atomic-write primitive, and the schema builder + emitter.
  `emit-gen-command` (B2), `measure_reservoir`/`ReservoirMeasurements` (B3),
  `qualify` (B4-B7), the `V2Config` extension (B8), the `run_screen`
  precheck (B9), the final 11-identity chain (B10) and the CLI `main`
  (B11) are ALL later tasks -- none of that exists here.

=============================================================================
TOOLING ONLY. PURE, stdlib-only module: no evaluator / MCTS / GPU / MLX /
checkpoint import, and no reservoir generation. Every function below is a
plain data transform over already-supplied Python values (dicts, strings,
bytes, paths) -- `write_atomic` is the ONE function that touches the
filesystem, and it does only a read-back-and-compare plus a temp+rename
write, never anything checkpoint/evaluator/MCTS-shaped. Verified at test
time via a subprocess import check (mirrors fpu_dev_corpus_v2.py's own
`test_v2_module_import_pulls_no_gpu_or_mlx`): importing this module leaves
`mlx` and `torch` out of `sys.modules`.

Module-level imports are stdlib ONLY for this task (json, os, tempfile,
enum, pathlib, typing). This module does NOT yet import `fpu_dev_corpus_v2`
-- the design's Sec 6 "import only the shared config schema-key constant"
seam into that module is a LATER task's concern (B3+, once this module has
a config-deriving stage that needs it), not this one.
=============================================================================

What this section does
-----------------------
`PROTOCOL_SCHEMA_KEYS`: the frozen, complete field set of
`reservoir_protocol.json` (design Sec 2.1) -- every declared pre-generation
decision: identity, matchup+anchor, reservoir params (games/base_seed/the
ten result-determining match knobs/save_eval_replays/workers), output-path
relationships, selection settings, and generation provenance. A LATER
schema version may add fields (`config_schema_version` is itself part of
the schema for exactly this reason), but this tuple is what the CURRENT
schema version requires -- no silent omission, mirroring
`fpu_dev_corpus_v2._V2_CONFIG_REQUIRED_KEYS`'s own "no default source, no
default stride" ethos.

`canonical_json_bytes`: the ONE canonical-JSON encoder every emitted
artifact in this subsystem (protocol, and later the config + report) routes
through -- sorted keys (recursively, at every nesting level), `ensure_ascii`,
rejecting non-finite floats (`allow_nan=False` -- "fixed numeric formatting"
means every emitted number is valid, unambiguous JSON, never a
silently-nonstandard `NaN`/`Infinity` token), and a single trailing newline.
Byte-stable across dict-insertion-order permutations, which is what makes a
LATER `protocol_sha1`/`config_sha1` a hash of the DATA, never of incidental
Python construction order.

`write_atomic` / `WriteStatus`: the ONE filesystem primitive both this
task's `emit_protocol` and (a later task's) `run_qualify` use to write an
immutable artifact -- temp file in the SAME directory + `os.replace` (so the
rename is atomic on a single filesystem: no half-written file is ever
visible at the final path), refusing to silently clobber an existing
DIFFERENT artifact (raises `ValueError`) while treating an existing
BYTE-IDENTICAL artifact as a successful no-op (idempotent re-emit).

`build_protocol`: validates a caller-supplied `params` mapping already
carries every `PROTOCOL_SCHEMA_KEYS` field (raising `ValueError` naming
EVERY missing one, not just the first) and returns a canonical protocol
dict containing exactly those fields. It performs NO filesystem I/O and
computes NO hashes -- every value (including each checkpoint's `name:sha1`
`identity` string) is DECLARED by the caller, per the design's derivability
invariant (Sec 2: "the protocol carries every declared decision").
Measuring a REAL generated reservoir against a frozen protocol is
`measure_reservoir`'s job (a later task), not this function's.

`emit_protocol`: freezes a `reservoir_protocol.json` -- `build_protocol` +
`canonical_json_bytes` + `write_atomic`, atomically and immutably.
`--check` (`check=True`) NEVER writes: it recomputes the canonical bytes and
reports whether they match what's on disk (`EXIT_OK`) or not
(`EXIT_MISMATCH`), purely as a read + compare.
"""
from __future__ import annotations

import enum
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

# ---------------------------------------------------------------------------
# Exit-code vocabulary (design Sec 3) -- shared across the WHOLE module's
# eventual CLI. This task only needs EXIT_OK / EXIT_MISMATCH for
# `emit_protocol`'s own `--check` outcome; EXIT_USAGE / EXIT_GATE_FAIL are
# declared here now (not invented ad hoc later) so every later task
# (`qualify`, the CLI `main`) shares this ONE table rather than restating
# it.
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISMATCH = 3
EXIT_GATE_FAIL = 4


# ---------------------------------------------------------------------------
# The frozen `reservoir_protocol.json` field set (design Sec 2.1) -- Task B1.
# ---------------------------------------------------------------------------
PROTOCOL_SCHEMA_KEYS: Tuple[str, ...] = (
    # Identity.
    "protocol_version",
    "no_top_up",
    "config_schema_version",
    # Matchup + anchor (amendment 1 disambiguation): checkpoint_a/b are each
    # a caller-declared {"path": ..., "identity": "name:sha1"} mapping;
    # `anchor` names WHICH of the two ("checkpoint_a" | "checkpoint_b") is
    # the single fpu-off screen anchor.
    "checkpoint_a",
    "checkpoint_b",
    "anchor",
    # Reservoir params: `games` + the ten result-determining match knobs
    # (amendment 4) -- board_size/mcts_sims/mcts_eval_batch_size/
    # mcts_stall_flush_sims/selection_mode/opening_temp_plies/temp_high/
    # temp_low/max_moves/base_seed -- plus `save_eval_replays` (capture is
    # mandatory) and `workers` (operational, recorded, amendment 2).
    "games",
    "base_seed",
    "board_size",
    "mcts_sims",
    "mcts_eval_batch_size",
    "mcts_stall_flush_sims",
    "selection_mode",
    "opening_temp_plies",
    "temp_high",
    "temp_low",
    "max_moves",
    "save_eval_replays",
    "workers",
    # Output relationships (amendment 3).
    "match_summary_path",
    "source_index_path",
    "replay_dir",
    "config_out",
    "report_out",
    # Selection settings (so a later config is derivable from this
    # protocol alone).
    "selection_seed",
    "phase_allocation",
    "late_floors",
    "enumerator_params",
    "new_collapse_stratum",
    "forbidden_manifests",
    "screen_out",
    "select_out",
    # Generation provenance (amendment 8).
    "generation_git_commit",
    "generation_source_sha1s",
)

assert len(PROTOCOL_SCHEMA_KEYS) == len(set(PROTOCOL_SCHEMA_KEYS)), (
    "PROTOCOL_SCHEMA_KEYS has a duplicate key")


# ---------------------------------------------------------------------------
# Canonical JSON (design Sec 8) -- Task B1.
# ---------------------------------------------------------------------------
def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic, byte-reproducible JSON encoding of `obj`.

    - `sort_keys=True`: every dict at every nesting level is emitted with
      its keys in sorted order (`json.dumps` applies this recursively
      through the whole object graph, not just the top level) -- so two
      Python dicts built in different insertion orders but with equal
      contents produce IDENTICAL bytes. This is what makes a later
      `protocol_sha1`/`config_sha1` a hash of the DATA, never of incidental
      construction order (verified across permutations in
      tests/test_fpu_dev_reservoir_protocol.py).
    - `ensure_ascii=True`: every non-ASCII character is `\\uXXXX`-escaped,
      so the output is plain 7-bit text -- diffable/catable/greppable
      anywhere, never dependent on a terminal's encoding.
    - `allow_nan=False`: "fixed numeric formatting" -- a `NaN`/`Infinity`/
      `-Infinity` float would otherwise serialize as a non-standard JSON
      token (valid Python `json`, invalid JSON proper); rejecting it here
      (raises `ValueError`, `json`'s own behavior) means every emitted
      number is unambiguous, standards-conformant JSON.
    - A single trailing newline: `json.dumps` itself never adds one: the
      POSIX text-file convention, and it keeps two canonical files
      diffable/catable without a missing-newline warning.

    Returns `bytes` (ASCII-encoded -- safe given `ensure_ascii=True`; if
    something upstream ever slipped a raw non-ASCII byte through despite
    that flag, `.encode("ascii")` would raise rather than silently emit
    non-canonical bytes).
    """
    text = json.dumps(
        obj, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False)
    return (text + "\n").encode("ascii")


# ---------------------------------------------------------------------------
# Atomic, immutable write primitive (design Sec 3) -- Task B1.
# ---------------------------------------------------------------------------
class WriteStatus(enum.Enum):
    """`write_atomic`'s outcome. Both members are SUCCESS states --
    `UNCHANGED` is the idempotent no-op re-emit of an already-identical
    artifact, not an error. Only a byte-DIFFERENT existing artifact raises
    (see `write_atomic`)."""
    WRITTEN = "written"
    UNCHANGED = "unchanged"


def write_atomic(path: Union[str, Path], data_bytes: bytes) -> WriteStatus:
    """Write `data_bytes` to `path` atomically and immutably.

    - **Absent** -- writes via a temp file in `path`'s OWN directory (so the
      final `os.replace` is an atomic rename on a single filesystem, never
      a cross-filesystem copy that could leave a half-written file visible
      at `path`), then `os.replace`s it into place. Returns `WRITTEN`.
    - **Present, byte-identical** -- a no-op: returns `UNCHANGED` without
      touching the filesystem (idempotent re-emit -- re-running the same
      emitter against the same inputs is always safe).
    - **Present, byte-different** -- refuses: raises `ValueError` and
      leaves the existing artifact completely untouched (no partial/
      clobbering write ever lands). This is the immutability contract: an
      already-frozen artifact is never silently replaced; delete it
      explicitly, or version a new protocol/config, instead.

    Creates `path`'s parent directories if needed (mirrors this project's
    existing atomic-sidecar-write idiom, e.g. trainer.py's iteration-stats
    writer). Any I/O failure mid-write cleans up its own temp file before
    re-raising -- never leaves temp litter behind.
    """
    target = Path(path)
    if target.exists():
        if target.read_bytes() == data_bytes:
            return WriteStatus.UNCHANGED
        raise ValueError(
            f"write_atomic: refusing to overwrite {target} -- an existing "
            f"artifact with DIFFERENT bytes is already there (immutable "
            f"once written; delete it explicitly, or version a new "
            f"protocol/config, rather than silently replacing it)")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return WriteStatus.WRITTEN


# ---------------------------------------------------------------------------
# Protocol schema builder + emitter -- Task B1.
# ---------------------------------------------------------------------------
def build_protocol(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate + assemble a canonical protocol dict from `params`.

    Requires EVERY `PROTOCOL_SCHEMA_KEYS` field to already be present in
    `params` -- raises `ValueError` naming every missing key (not just the
    first), mirroring `fpu_dev_corpus_v2.load_v2_config`'s own "no default
    source, no default stride" required-key check. Extra keys in `params`
    beyond the schema are silently dropped (same permissive-superset
    behavior as `load_v2_config`'s `raw` dict) -- only the schema fields
    make it into the returned protocol.

    Performs NO filesystem I/O and computes NO hashes: every value --
    including each checkpoint's `identity` string -- is taken verbatim from
    `params`, per the design's derivability invariant that the protocol
    carries every DECLARED decision (Sec 2). A later `measure_reservoir`
    stage is what independently MEASURES a real generated reservoir and
    checkpoints against this declaration; this function only assembles what
    the caller already declared.
    """
    missing = sorted(k for k in PROTOCOL_SCHEMA_KEYS if k not in params)
    if missing:
        raise ValueError(
            f"build_protocol: params is missing required key(s): "
            f"{', '.join(missing)}")
    return {k: params[k] for k in PROTOCOL_SCHEMA_KEYS}


def emit_protocol(
        params: Mapping[str, Any], out_path: Union[str, Path], *,
        check: bool = False) -> int:
    """Freeze `params` into a canonical `reservoir_protocol.json` at
    `out_path` (`build_protocol` + `canonical_json_bytes` + `write_atomic`).

    `check=True` NEVER writes: it recomputes `build_protocol(params)`'s
    canonical bytes and reports whether `out_path` already holds exactly
    those bytes (`EXIT_OK`) or not -- including the file not existing at
    all (`EXIT_MISMATCH`) -- as a pure read + compare, with no filesystem
    mutation in either outcome. `check=False` (the default) performs the
    real atomic write via `write_atomic`, which raises `ValueError` if
    `out_path` already holds DIFFERENT bytes (immutability -- see
    `write_atomic`) and is a no-op success on an idempotent re-emit.

    `params` is validated (via `build_protocol`) in BOTH modes -- `--check`
    is a bypass for WRITING, never a bypass for an invalid protocol, so a
    missing param raises `ValueError` here exactly as it does in write
    mode.

    Returns `EXIT_OK` (0) or `EXIT_MISMATCH` (3) (module-level constants,
    design Sec 3's shared exit-code vocabulary) -- never raises for a
    detected mismatch, only for an invalid `params` or a genuine write
    refusal (both propagate from `build_protocol`/`write_atomic`
    unchanged).
    """
    protocol = build_protocol(params)
    data = canonical_json_bytes(protocol)

    if check:
        target = Path(out_path)
        if not target.exists():
            return EXIT_MISMATCH
        return EXIT_OK if target.read_bytes() == data else EXIT_MISMATCH

    write_atomic(out_path, data)
    return EXIT_OK
