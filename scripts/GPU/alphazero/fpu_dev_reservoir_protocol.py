"""FPU (policy-mass) v2 reservoir-protocol schema + canonical-JSON/atomic-write
primitives + the `emit-protocol` builder + the `measure_reservoir` I/O
boundary.

Frozen design ref: docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md
  Sec 2.1 (the `reservoir_protocol.json` schema -- the single source of ALL
  declared pre-generation decisions), Sec 3 (CLI stages / exit codes /
  atomicity+immutability contract), Sec 4 (qualification: the measurement
  boundary), Sec 6 (module boundary / circular-import resolution), Sec 8
  (canonical JSON, determinism, reviewability).
Pre-op hardening plan ref: docs/superpowers/plans/2026-07-14-fpu-v2-preop-hardening-plan.md
  Tasks B1-B3 -- the first three tasks of the new Group-2 subsystem
  (B1-B11), which will qualify a generated reservoir zero-GPU (B3-B7) and
  emit an immutable `fpu_dev_corpus_v2_config.json` (B7-B10). B1 laid the
  foundation: the protocol's field set (`PROTOCOL_SCHEMA_KEYS`), the
  canonical-JSON encoder, the atomic-write primitive, and the schema
  builder + emitter. B2 added `gen_command` -- the exact
  `eval_checkpoint_match` argv derived from an already-frozen protocol, so
  the operator's generation command cannot drift from the frozen
  decisions. B3 adds `ReservoirMeasurements` + `measure_reservoir` -- the
  ONE filesystem-I/O boundary of qualification (Sec 4): it loads and hashes
  a GENERATED reservoir into a frozen, pure-data structure, so every later
  qualification stage (`qualify` protocol-conformance B4, summary-binding
  B5, preflight B6, config-derivation B7) reads only THAT structure and
  performs no I/O of its own. The `V2Config` extension (B8), the
  `run_screen` precheck (B9), the final 11-identity chain (B10) and the CLI
  `main` (B11) are ALL later tasks -- none of that exists here.

=============================================================================
TOOLING ONLY. No evaluator / MCTS / GPU / MLX / checkpoint-WEIGHTS import,
and no reservoir generation. Through B2, every function was a plain data
transform over already-supplied Python values, and `write_atomic` was the
ONE function that touched the filesystem (a read-back-and-compare plus a
temp+rename write). B3 adds a SECOND, deliberately isolated filesystem
toucher: `measure_reservoir` is now the ONE AND ONLY function in this
module -- and, by design Sec 4, in the WHOLE qualification pipeline -- that
reads a GENERATED reservoir off disk: the JSONL index, its replay sidecars,
the match summary, the two reservoir checkpoints + the anchor, the
generation-source modules, the v2 corpus source files, and the forbidden
manifests, hashing/loading every one of them into `ReservoirMeasurements`.
It still never touches a checkpoint's WEIGHTS (only its file BYTES, for a
sha1), never runs the evaluator/MCTS, and never imports GPU/MLX.
`ReservoirMeasurements` itself is frozen, pure data: constructing one
directly (as every later B4-B6 test will) performs NO I/O at all -- it is
an ordinary `@dataclass(frozen=True)` with no `__post_init__`. Verified at
test time via a subprocess import check (mirrors fpu_dev_corpus_v2.py's own
`test_v2_module_import_pulls_no_gpu_or_mlx`): importing this module leaves
`mlx` and `torch` out of `sys.modules`.

Module-level imports are stdlib ONLY, PLUS one intra-package import: `from
.fpu_dev_corpus_v2 import _V2_CORPUS_SOURCES` (design Sec 6's "import only
the shared ... constant" seam -- here narrowed to the v2 corpus's own
result-determining source-file tuple, needed so THIS module's
`source_file_sha1s` measurement can include the v2 corpus sources without
duplicating that list). This is deliberately NOT the Sec 6 circular-import
risk: `fpu_dev_corpus_v2.py` is itself import-pure (verified by its own
`test_v2_module_import_pulls_no_gpu_or_mlx`), and the cycle Sec 6 actually
warns about runs the OTHER direction -- `fpu_dev_corpus_v2.run_screen`
importing (part of) THIS module -- which stays a lazy, in-function import,
a LATER task's concern (B9), not this one. Nothing else is imported from
`fpu_dev_corpus_v2` here: no `V2Config`, no `run_screen`, no evaluator/MCTS
plumbing (verified: tests/test_fpu_dev_reservoir_protocol.py::
test_module_imports_only_v2_corpus_sources_from_fpu_dev_corpus_v2).
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

`gen_command`: the exact `eval_checkpoint_match` argv derived from an
ALREADY-FROZEN `protocol` dict (design Sec 2.1's "every explicit generation
flag needed to reconstruct the command is present"; Sec 3's
`emit-gen-command` -- "so generation cannot drift from the frozen
decisions"). A pure data transform (dict lookups + `str()` formatting) --
it performs NO validation of its own (that is `build_protocol`'s job, and
is expected to have already run before a protocol reaches this function)
and no filesystem I/O. Deterministic: every value is read from `protocol`
by an explicit key (never by iterating `protocol`'s own items), so the
same protocol dict always produces the same argv regardless of Python
dict-construction/iteration order. `--save-eval-replays` is
`eval_checkpoint_match`'s `action="store_true"` flag: emitted bare (no
following value) and ONLY when `protocol["save_eval_replays"]` is true --
never emitted with a `false`/`0` value when it is false, since that is not
how `store_true` works. `protocol["source_index_path"]` is deliberately
NEVER emitted as a flag: `eval_checkpoint_match` has no such flag, and
instead derives that JSONL path itself from `--output`'s stem
(`eval_checkpoint_match._write_outputs`: `f"{stem}_games.jsonl"`) --
`source_index_path` exists in the protocol only so a LATER qualification
stage can verify the generator's derivation rule was actually followed.

`ReservoirMeasurements`: a frozen (`@dataclass(frozen=True)`), pure-data
structure holding everything a generated reservoir's identity requires
(design Sec 4): the loaded JSONL rows (`jsonl_rows`, one dict per line --
EVERY field `eval_checkpoint_match._write_outputs` wrote, not the narrower
subset `build_fpu_dev_corpus.load_game_index` keeps, since a LATER stage
(B5) reconstructs full `EvalGameResult` rows from them); each game's
replay sidecar keyed by `game_idx` (`sidecars_by_idx`); the loaded match
summary (`summary`); the THREE checkpoint identities
(`checkpoint_identities` -- `reservoir_a`/`reservoir_b`/`anchor`, each a
`name:sha1` string, mirroring `fpu_dev_corpus_v2.v2_screen_provenance`'s
own `checkpoint_identity` idiom); the generation-source hashes + git
commit (`generation_source_sha1s`, `generation_git_commit`); three
whole-file hashes (`source_index_sha1`, `replay_data_sha1` -- over the
replay DATA, not paths, via `fpu_provenance.replay_data_sha1` -- and
`match_summary_sha1`); the v2 corpus's own result-determining source files
PLUS this qualification module itself (`source_file_sha1s`); and the
forbidden manifests' hashes (`forbidden_manifest_sha1s`). Every field is a
plain Python value (list, dict, str) -- constructing a
`ReservoirMeasurements` directly never touches a filesystem, which is
exactly what lets B4 (protocol conformance), B5 (summary binding) and B6
(preflight) stay genuinely pure: each takes a `ReservoirMeasurements` and
performs no I/O of its own.

`measure_reservoir`: the ONE function in the whole qualification pipeline
that touches a GENERATED reservoir's filesystem (design Sec 4/Sec 6).
Reads the JSONL index at `protocol["source_index_path"]` into
`jsonl_rows` (file order -- contiguity/ordering is B4's concern, not this
measurement boundary's); reads every row's replay sidecar (keyed by
`game_idx`) into `sidecars_by_idx`; reads the match summary at
`protocol["match_summary_path"]`; hashes `checkpoint_a`/`checkpoint_b`/the
anchor (whichever of the two `protocol["anchor"]` names) into
`checkpoint_identities`; hashes the THIRTEEN generation-source modules
(`GENERATION_SOURCE_MODULES`, design Sec 2.1 amendment 8) and reads the
current `git_commit()`; whole-file-hashes the index and the match summary,
and content-hashes the replay DATA (`fpu_provenance.replay_data_sha1`,
contents not paths, over every row's `replay_path`); hashes
`QUALIFICATION_SOURCE_FILES` -- `fpu_dev_corpus_v2._V2_CORPUS_SOURCES`
PLUS this module itself (design Sec 2.2 amendment 4: "the qualification
module is result-determining for the corpus it produces"); and hashes
`protocol["forbidden_manifests"]`. Every hash/read routes through the SAME
`fpu_provenance` helpers `fpu_dev_corpus_v2.py`'s own `v2_screen_
provenance` uses (`file_sha1`, `source_file_sha1s`, `replay_data_sha1`,
`git_commit`) -- reused, never reimplemented. It FAILS LOUD on any missing
declared path (a `FileNotFoundError` naming it): those `fpu_provenance`
helpers deliberately swallow `OSError` into a stable `"missing"`/`"none"`
sentinel, which -- being stable -- would sail through the config's §5
re-derive-and-byte-compare, so `measure_reservoir` existence-guards every
path it hashes FIRST (`_require_readable_files`), ensuring no sentinel ever
enters the tamper-evident measurements. Beyond that existence guard it
performs NO protocol-conformance validation (that is B4/B5/B6's job, over
the measurements it returns) and loads NO evaluator/MCTS/GPU/checkpoint
weights -- only file BYTES.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

from . import fpu_provenance
from .fpu_dev_corpus_v2 import _V2_CORPUS_SOURCES

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


# ---------------------------------------------------------------------------
# Exact generation command (design Sec 2.1/Sec 3) -- Task B2.
# ---------------------------------------------------------------------------
def gen_command(protocol: Mapping[str, Any]) -> List[str]:
    """The exact `eval_checkpoint_match` argv derived from an
    already-frozen `protocol` (design Sec 2.1: "Every explicit generation
    flag needed to reconstruct the command is present [...]"; Sec 3:
    `emit-gen-command` -- "print the exact `eval_checkpoint_match` command
    derived from the frozen protocol [...] so generation cannot drift from
    the frozen decisions").

    A pure data transform, zero-GPU and zero-I/O: every flag's value is
    read from `protocol` by an explicit key (never by iterating
    `protocol`'s own items), so the SAME protocol dict always produces the
    SAME argv -- independent of Python dict-construction/insertion order,
    and this function never runs anything itself. Flag order mirrors
    `eval_checkpoint_match._build_arg_parser()`'s own `add_argument`
    sequence, so the reconstructed command lines up with that CLI's own
    declaration order.

    - `--checkpoint-a` / `--checkpoint-b` -- each checkpoint's `"path"`
      (never its `"identity"`: the generator takes a filesystem path to
      run, not the `name:sha1` identity string that a LATER qualification
      stage verifies against the generated reservoir).
    - `--games` / `--board-size` / `--mcts-sims` /
      `--mcts-eval-batch-size` / `--mcts-stall-flush-sims` /
      `--selection-mode` / `--opening-temp-plies` / `--temp-high` /
      `--temp-low` / `--max-moves` / `--workers` / `--base-seed` -- each
      maps one-to-one onto its same-named protocol field, stringified
      (`json` round-trips an int/float to the same digits `str` produces,
      so this changes no value's meaning).
    - `--save-eval-replays` is `eval_checkpoint_match`'s
      `action="store_true"` flag: emitted BARE (no following value) and
      ONLY when `protocol["save_eval_replays"]` is true; omitted entirely
      when false (never emitted with a `false`/`0` value -- that is not
      how `store_true` works).
    - `--replay-dir` is always emitted from `protocol["replay_dir"]`,
      regardless of `save_eval_replays` -- the protocol declares it
      unconditionally (design Sec 2.1 "Output relationships"; it would
      simply go unused by the generator when replay capture is off).
    - `--output` is `protocol["match_summary_path"]`.
    - `protocol["source_index_path"]` is deliberately NEVER emitted:
      `eval_checkpoint_match` has no such flag -- it derives that JSONL
      path itself from `--output`'s stem
      (`eval_checkpoint_match._write_outputs`: `f"{stem}_games.jsonl"`).
      `source_index_path` exists in the protocol only so a later
      qualification stage can verify the generator's derivation rule was
      actually followed (spec Sec 4.1 "Output-path relationships").

    Assumes `protocol` is already a valid, fully-populated protocol dict
    (e.g. `build_protocol`'s return value, or a `reservoir_protocol.json`
    already loaded from disk) -- a missing field raises `KeyError` from the
    plain dict lookup rather than a friendlier message, since re-validating
    an already-frozen protocol is `build_protocol`'s job, not this one's.
    """
    checkpoint_a_path = protocol["checkpoint_a"]["path"]
    checkpoint_b_path = protocol["checkpoint_b"]["path"]

    argv: List[str] = [
        ".venv/bin/python", "-m", "scripts.GPU.alphazero.eval_checkpoint_match",
    ]
    argv += ["--checkpoint-a", str(checkpoint_a_path)]
    argv += ["--checkpoint-b", str(checkpoint_b_path)]
    argv += ["--games", str(protocol["games"])]
    argv += ["--board-size", str(protocol["board_size"])]
    argv += ["--mcts-sims", str(protocol["mcts_sims"])]
    argv += ["--mcts-eval-batch-size", str(protocol["mcts_eval_batch_size"])]
    argv += ["--mcts-stall-flush-sims", str(protocol["mcts_stall_flush_sims"])]
    argv += ["--selection-mode", str(protocol["selection_mode"])]
    argv += ["--opening-temp-plies", str(protocol["opening_temp_plies"])]
    argv += ["--temp-high", str(protocol["temp_high"])]
    argv += ["--temp-low", str(protocol["temp_low"])]
    argv += ["--max-moves", str(protocol["max_moves"])]
    argv += ["--workers", str(protocol["workers"])]
    argv += ["--base-seed", str(protocol["base_seed"])]
    if protocol["save_eval_replays"]:
        argv.append("--save-eval-replays")
    argv += ["--replay-dir", str(protocol["replay_dir"])]
    argv += ["--output", str(protocol["match_summary_path"])]
    return argv


# ---------------------------------------------------------------------------
# Result-determining source-file sets (design Sec 2.1 amendment 8, Sec 2.2
# amendment 4) -- Task B3. Pinned as module-level tuples of PATHS, mirroring
# `fpu_dev_corpus_v2._V2_MODULE_DIR` / `_V2_CORPUS_SOURCES`'s own idiom, so a
# reviewer can see the exact frozen file set without running anything.
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent

# The THIRTEEN generation-source modules (spec Sec 2.1 amendment 8) -- the
# modules whose bytes, given the checkpoint bytes, determine the generated
# games and the summary. Order mirrors the spec's own presentation order.
GENERATION_SOURCE_MODULES: Tuple[Path, ...] = (
    _MODULE_DIR / "eval_checkpoint_match.py",
    _MODULE_DIR / "eval_runner.py",
    _MODULE_DIR / "mcts.py",
    _MODULE_DIR / "opening_diagnostics.py",
    _MODULE_DIR / "evaluator.py",
    _MODULE_DIR / "game" / "twixt_state.py",
    _MODULE_DIR / "game" / "__init__.py",
    _MODULE_DIR / "eval_replay.py",
    _MODULE_DIR / "probe_eval.py",
    _MODULE_DIR / "network.py",
    _MODULE_DIR / "local_evaluator.py",
    _MODULE_DIR / "eval_summary.py",
    _MODULE_DIR / "eval_elo.py",
)
assert len(GENERATION_SOURCE_MODULES) == 13, GENERATION_SOURCE_MODULES

# The v2 corpus's own result-determining source set (`_V2_CORPUS_SOURCES`,
# imported from `fpu_dev_corpus_v2`) PLUS this qualification module itself
# (spec Sec 2.2 amendment 4: "the qualification module is result-determining
# for the corpus it produces" -- added to the v2 set ONLY, this module never
# touches v1's own `_CORPUS_SOURCES`).
QUALIFICATION_SOURCE_FILES: Tuple[Path, ...] = _V2_CORPUS_SOURCES + (
    _MODULE_DIR / "fpu_dev_reservoir_protocol.py",
)
assert len(QUALIFICATION_SOURCE_FILES) == len(set(QUALIFICATION_SOURCE_FILES)), (
    "QUALIFICATION_SOURCE_FILES has a duplicate entry")


# ---------------------------------------------------------------------------
# ReservoirMeasurements -- Task B3 (design Sec 4).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ReservoirMeasurements:
    """Everything `measure_reservoir` reads/hashes off a generated
    reservoir, frozen into ONE pure-data structure (design Sec 4/Sec 6) --
    the boundary that lets every later qualification stage (B4 protocol
    conformance, B5 summary binding, B6 preflight) take THIS structure and
    perform no I/O of its own. An ordinary `@dataclass(frozen=True)` with no
    `__post_init__`: constructing one directly is always pure -- see this
    module's own tests for the proof.

    jsonl_rows: every row of `protocol["source_index_path"]`, in FILE order,
      each dict carrying every field `eval_checkpoint_match._write_outputs`
      wrote (task_id, pairing_id, game_idx, red/black_checkpoint, winner,
      winner_checkpoint, reason, n_moves, red/black_score, replay_path) --
      NOT the narrower subset `build_fpu_dev_corpus.load_game_index` keeps,
      since B5 reconstructs full `EvalGameResult` rows from these.
    sidecars_by_idx: each row's replay sidecar (the JSON `eval_replay.
      write_replay` wrote), keyed by `game_idx` (int).
    summary: the loaded `protocol["match_summary_path"]` JSON, verbatim.
    checkpoint_identities: THREE `name:sha1` identities -- `reservoir_a`
      (`protocol["checkpoint_a"]["path"]`'s bytes), `reservoir_b`
      (`protocol["checkpoint_b"]["path"]`'s bytes), and `anchor` (whichever
      of the two `protocol["anchor"]` names -- the SAME file as `reservoir_a`
      or `reservoir_b`, hashed again under its own role name, per the
      design's "three distinct roles" -- Sec 2.1 amendment 1).
    generation_source_sha1s: `{basename: sha1}` over `GENERATION_SOURCE_
      MODULES` (the 13 spec Sec 2.1 modules).
    generation_git_commit: `fpu_provenance.git_commit()` at measurement time
      (the Sec 10 trust boundary applies: this proves the sources AS THEY
      EXIST AT QUALIFY TIME, not that those exact bytes executed).
    source_index_sha1 / match_summary_sha1: whole-file hashes of
      `protocol["source_index_path"]` / `protocol["match_summary_path"]`.
    replay_data_sha1: a single content hash (not path-based) over every
      row's replay sidecar, via `fpu_provenance.replay_data_sha1`.
    source_file_sha1s: `{basename: sha1}` over `QUALIFICATION_SOURCE_FILES`
      (the v2 corpus sources PLUS this qualification module itself).
    forbidden_manifest_sha1s: `{basename: sha1}` over
      `protocol["forbidden_manifests"]`.
    """
    jsonl_rows: List[dict]
    sidecars_by_idx: Dict[int, dict]
    summary: dict
    checkpoint_identities: Dict[str, str]
    generation_source_sha1s: Dict[str, str]
    generation_git_commit: str
    source_index_sha1: str
    replay_data_sha1: str
    match_summary_sha1: str
    source_file_sha1s: Dict[str, str]
    forbidden_manifest_sha1s: Dict[str, str]


def _checkpoint_identity(path: Union[str, Path]) -> str:
    """`name:sha1` -- mirrors `fpu_dev_corpus_v2.v2_screen_provenance`'s own
    `checkpoint_identity` idiom (`f"{Path(checkpoint).name}:{fpu_provenance.
    file_sha1(checkpoint)}"`)."""
    return f"{Path(path).name}:{fpu_provenance.file_sha1(path)}"


def _require_readable_files(paths: Any, *, kind: str) -> None:
    """Raise `FileNotFoundError` naming the FIRST declared path that is
    absent (or not a regular file), BEFORE any hashing.

    This is the fail-loud guard that makes `measure_reservoir`'s promise
    true. `fpu_provenance.file_sha1` / `source_file_sha1s` deliberately
    SWALLOW `OSError` and return a STABLE `"missing"` (or `"none"` for a
    falsy path) sentinel rather than raising -- appropriate for a
    best-effort run-context fingerprint, but WRONG for this measurement
    boundary: a stable sentinel would (a) silently produce a partial
    measurement and (b) keep hard-matching cleanly through the config's
    re-derive-and-byte-compare (spec Sec 5), so a genuinely-absent
    checkpoint / forbidden manifest / source file would pass UNDETECTED into
    the tamper-evident config's `expected_fingerprints`. Existence-checking
    every path we are about to hash closes that gap, so a sentinel can never
    enter `ReservoirMeasurements`. (`is_file()` rather than `exists()`: a
    directory at a checkpoint path would also make `file_sha1` fall back to
    the sentinel, so it must be rejected too.)"""
    for p in paths:
        if not p or not Path(p).is_file():
            raise FileNotFoundError(
                f"measure_reservoir: {kind} path is missing or not a regular "
                f"file: {p!r}")


def _load_jsonl_rows(path: Union[str, Path]) -> List[dict]:
    """Read a `*_games.jsonl` index file into a list of dicts, one per
    non-blank line, in FILE order (`eval_checkpoint_match._write_outputs`'s
    own comment: "already sorted by (pairing_id, game_idx)" -- contiguity
    is re-verified, not re-sorted, by a LATER stage, B4). Every field
    `json.dumps(asdict(r))` wrote is preserved verbatim -- unlike v1's
    narrower `build_fpu_dev_corpus.load_game_index` (game_idx/n_moves/
    winner/replay_path only), B5 needs every field to reconstruct
    `EvalGameResult` rows for the summary<->JSONL binding check (spec Sec
    4.1)."""
    rows: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_sidecars(jsonl_rows: List[dict]) -> Dict[int, dict]:
    """Read every row's replay sidecar (design Sec 2.1: `save_eval_replays:
    true` is mandatory, so every row is expected to carry a `replay_path`),
    keyed by `game_idx` (int) -- the same key B4's replay-linkage check
    (spec Sec 4.1) looks each row up by."""
    sidecars: Dict[int, dict] = {}
    for row in jsonl_rows:
        sidecar_path = row["replay_path"]
        sidecars[int(row["game_idx"])] = json.loads(Path(sidecar_path).read_text())
    return sidecars


def measure_reservoir(protocol: Mapping[str, Any]) -> ReservoirMeasurements:
    """Load + hash a GENERATED reservoir into a `ReservoirMeasurements`
    (design Sec 4) -- the ONE filesystem-I/O function in the whole
    qualification pipeline. See the module docstring's `measure_reservoir`
    paragraph and `ReservoirMeasurements`'s own docstring for the full field
    derivation. Performs NO validation (that is B4/B5/B6's job over the
    returned measurements) and loads NO evaluator/MCTS/GPU/checkpoint
    weights -- only file BYTES, via the same `fpu_provenance` helpers
    `fpu_dev_corpus_v2.v2_screen_provenance` reuses.

    Assumes `protocol` is already a valid, fully-populated protocol dict
    (e.g. `build_protocol`'s return value, or a `reservoir_protocol.json`
    already loaded from disk) -- a missing field raises `KeyError`, exactly
    like `gen_command`, since re-validating an already-frozen protocol is
    `build_protocol`'s job, not this one's.

    FAILS LOUD on any missing declared path: a `FileNotFoundError` (an
    `OSError` subclass) is raised -- naming the path -- for a missing
    checkpoint, forbidden manifest, generation-source module, or v2 source
    file (via the `_require_readable_files` existence guard, so
    `fpu_provenance`'s OSError-swallowing `"missing"` sentinel can NEVER
    enter the measurements -- see that helper's own docstring), and for a
    missing index / summary / replay sidecar (via the raw reads below). This
    is the whole point of the I/O boundary the evidence chain trusts: a
    genuinely-absent input is a hard error here, never a silent partial
    measurement that would still byte-compare cleanly in the config's §5
    re-derive check. Beyond this existence guard it performs NO
    protocol-conformance validation (that is B4/B5/B6's job over the
    returned measurements) and loads NO evaluator/MCTS/GPU/checkpoint
    weights -- only file BYTES.
    """
    anchor_role = protocol["anchor"]
    # Existence guard BEFORE hashing (see `_require_readable_files`): the two
    # reservoir checkpoints + the anchor, the forbidden manifests, and both
    # code-source sets are hashed via `fpu_provenance` helpers that swallow
    # OSError into a stable `"missing"` sentinel -- so absence must be caught
    # HERE or it would leak into the tamper-evident config. (The index /
    # summary / replay sidecars fail loud on their own in the raw reads
    # below, so they need no separate guard.)
    _require_readable_files(
        [protocol["checkpoint_a"]["path"], protocol["checkpoint_b"]["path"],
         protocol[anchor_role]["path"]], kind="checkpoint")
    _require_readable_files(protocol["forbidden_manifests"],
                            kind="forbidden manifest")
    _require_readable_files(GENERATION_SOURCE_MODULES,
                            kind="generation source module")
    _require_readable_files(QUALIFICATION_SOURCE_FILES,
                            kind="qualification source file")

    jsonl_rows = _load_jsonl_rows(protocol["source_index_path"])
    sidecars_by_idx = _load_sidecars(jsonl_rows)
    summary = json.loads(Path(protocol["match_summary_path"]).read_text())

    checkpoint_identities = {
        "reservoir_a": _checkpoint_identity(protocol["checkpoint_a"]["path"]),
        "reservoir_b": _checkpoint_identity(protocol["checkpoint_b"]["path"]),
        "anchor": _checkpoint_identity(protocol[anchor_role]["path"]),
    }

    replay_paths = [row["replay_path"] for row in jsonl_rows]

    return ReservoirMeasurements(
        jsonl_rows=jsonl_rows,
        sidecars_by_idx=sidecars_by_idx,
        summary=summary,
        checkpoint_identities=checkpoint_identities,
        generation_source_sha1s=fpu_provenance.source_file_sha1s(
            GENERATION_SOURCE_MODULES),
        generation_git_commit=fpu_provenance.git_commit(),
        source_index_sha1=fpu_provenance.file_sha1(protocol["source_index_path"]),
        replay_data_sha1=fpu_provenance.replay_data_sha1(replay_paths),
        match_summary_sha1=fpu_provenance.file_sha1(protocol["match_summary_path"]),
        source_file_sha1s=fpu_provenance.source_file_sha1s(
            QUALIFICATION_SOURCE_FILES),
        forbidden_manifest_sha1s=fpu_provenance.source_file_sha1s(
            protocol["forbidden_manifests"]),
    )
