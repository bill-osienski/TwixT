"""FPU (policy-mass) v2 reservoir-protocol schema + canonical-JSON/atomic-write
primitives + the `emit-protocol` builder + the `measure_reservoir` I/O
boundary + config derivation + the qualification report state machine.

Frozen design ref: docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md
  Sec 2.1 (the `reservoir_protocol.json` schema -- the single source of ALL
  declared pre-generation decisions), Sec 2.2 (the `fpu_dev_corpus_v2_
  config.json` schema -- the COMPLETE, derivable field set), Sec 3 (CLI
  stages / exit codes / atomicity+immutability contract / the report state
  machine), Sec 4 (qualification: the measurement boundary), Sec 4.1
  (protocol conformance, IN FULL: every check including
  summary-binding-by-reconstruction, amendments 3 + 5), Sec 4.2 (the
  geometric preflight, now WIRED into the pure qualification decision), Sec
  6 (module boundary / circular-import resolution, including the "preflight
  injection for tests" clarification), Sec 7 (no-top-up & versioning --
  MISMATCH regenerates under the same protocol, GATE-FAIL retires it), Sec 8
  (canonical JSON, determinism, reviewability).
Pre-op hardening plan ref: docs/superpowers/plans/2026-07-14-fpu-v2-preop-hardening-plan.md
  Tasks B1-B11 -- the eleven tasks of the new Group-2 subsystem, which
  qualify a generated reservoir zero-GPU (B3-B7), emit an immutable
  `fpu_dev_corpus_v2_config.json` (B7-B10), and surface the whole pipeline
  through the CLI `main` (B11). B1 laid the
  foundation: the protocol's field set (`PROTOCOL_SCHEMA_KEYS`), the
  canonical-JSON encoder, the atomic-write primitive, and the schema
  builder + emitter. B2 added `gen_command` -- the exact
  `eval_checkpoint_match` argv derived from an already-frozen protocol, so
  the operator's generation command cannot drift from the frozen
  decisions. B3 added `ReservoirMeasurements` + `measure_reservoir` -- the
  ONE filesystem-I/O boundary of qualification (Sec 4): it loads and hashes
  a GENERATED reservoir into a frozen, pure-data structure, so every later
  qualification stage reads only THAT structure and performs no I/O of its
  own. B4 added `check_protocol_conformance` -- the FIRST such stage: every
  Sec 4.1 protocol-vs-reservoir check EXCEPT summary-binding (B5) and the
  geometric preflight (B6), returning the FIRST failing check's reason
  (design's "any failure is MISMATCH exit 3" -- one outcome, not an
  accumulated list). B5 added `check_summary_binding` -- the SECOND stage,
  completing Sec 4.1: reconstructs `eval_runner.EvalGameResult` rows from
  `measurements.jsonl_rows` and calls the REAL, pure `eval_summary.
  summarize_match` to prove the supplied `measurements.summary` really IS
  that reconstruction's output (catching a summary swapped in from a
  DIFFERENT run), plus the separate `git_commit`-vs-protocol check; also
  adds `reason_histogram` for the qualification report. B6 adds
  `qualify_core` -- the pure qualification DECISION (Sec 4.2/Sec 6):
  composes B4 -> B5 -> an INJECTED geometric `preflight`, short-circuiting
  to `QualifyStatus.MISMATCH` on either of the first two (the preflight is
  never reached), else running `preflight(measurements)` and mapping
  feasible -> `OK` / infeasible -> `GATE_FAIL`. Its default `preflight`
  (`default_preflight`) is a thin, PURE wrapper -- deliberately NOT
  `fpu_dev_corpus_v2.v2_preflight_source` (that function is the I/O
  wrapper, per its own docstring) -- that builds v2 proposals from the
  ALREADY-LOADED `measurements.sidecars_by_idx` (no second disk read of
  data `measure_reservoir`, B3, already loaded) and hands them to the pure
  `fpu_dev_corpus_v2.v2_geometry_feasibility` core. B7 adds config
  derivation + the report state machine + the operator entry point (Sec
  2.2/Sec 3/Sec 7): `derive_config(protocol, measurements, *,
  protocol_path)` is the PURE `config = derive(protocol, reservoir)`
  function itself -- every Sec 2.2 field, carried from the protocol
  (verbatim, or RENAMED where the config's own field name differs, e.g.
  `eval_batch_size` <- `protocol["mcts_eval_batch_size"]`) or MEASURED
  (`expected_fingerprints`, including the three checkpoint identities that
  REPLACE the legacy single `checkpoint_identity`); `protocol_sha1` is
  computed from the IN-MEMORY protocol via `canonical_json_bytes` (never a
  file read), by construction equal to `fpu_provenance.file_sha1` of a
  canonically-emitted protocol file. `write_report`/`is_passed`/
  `is_retired` implement the Sec 3 report state machine: a MISMATCH report
  is REPLACEABLE (the next attempt may overwrite it), while an OK (PASS) or
  GATE_FAIL report is TERMINAL and IMMUTABLE (`write_report` raises rather
  than overwrite either). `run_qualify` is the operator entry point that
  OWNS the I/O this whole pipeline was built to keep out of every earlier
  stage: it loads the protocol file, calls `measure_reservoir` (B3, the ONE
  filesystem-I/O function that reads a GENERATED reservoir), runs the pure
  `qualify_core` (B6), and -- unless `check=True` -- emits the config +
  report per the state machine (OK: config + PASS report; GATE_FAIL:
  retirement report, config refused; MISMATCH: replaceable report, no
  config). It refuses outright (no measurement at all) against an
  already-RETIRED protocol, and refuses to re-qualify (no re-emit; use
  `--check`) an already-PASSED one. B8 (in fpu_dev_corpus_v2.py) extended
  `V2Config`/`load_v2_config` with the Sec 2.2 fields and added this module to
  the v2 corpus source set. B9 adds `precheck_before_screen(config, *,
  measure=measure_reservoir, preflight=default_preflight)` -- the
  `run_screen` pre-evaluator gate (Sec 5/Sec 6): re-derives + byte-compares
  the WHOLE config against a fresh `(protocol, measurements)` (the real
  config-tamper check -- a hash-only recheck cannot see an edited
  `selection_seed`/`select_out`, since neither carries a hash of its own),
  plus a per-identity hash recheck, an explicit protocol-binding check, and a
  defensive preflight repeat. `fpu_dev_corpus_v2.run_screen` calls it via a
  LAZY, function-body-local import (never at that module's top level -- Sec
  6's circular-import resolution: this module already top-level-imports FROM
  `fpu_dev_corpus_v2`, so a top-level import back would cycle). The final
  11-identity chain (B10) lives entirely in `fpu_dev_corpus_v2.py` (this
  module is only cross-referenced from its remediation prose, e.g.
  `_fingerprint_mismatch`'s own docstring below) -- no B10 code touches this
  file. B11 adds `main(argv=None) -> int` (Sec 3): three argparse
  subcommands -- `emit-protocol` (freeze a `reservoir_protocol.json` from a
  `--params-json` params mapping, via `emit_protocol`), `emit-gen-command`
  (print `gen_command`'s exact, shell-joined argv for an already-frozen
  `--protocol`), and `qualify` (`--protocol`, `--check`) -- dispatching to
  the functions above and returning `run_qualify`'s own status as the
  process exit code. Pure CLI glue, not a fourth qualification stage: see
  the `_parse_args`/`main` paragraph near the end of this docstring for the
  full per-subcommand contract. No `--mode select`/`screen` subcommand
  exists here (design Sec 3 draws this module's CLI boundary at
  protocol/qualify only) -- those two stages stay in
  `fpu_dev_corpus_v2.main`.

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
`mlx` and `torch` out of `sys.modules`. B4 introduces NO new filesystem
toucher: `check_protocol_conformance` reads only its two already-in-memory
arguments (`protocol`, a `ReservoirMeasurements`) -- `measure_reservoir`
remains the ONE AND ONLY I/O function in the whole qualification pipeline,
and every B4 test fabricates a `ReservoirMeasurements` directly (never via
`measure_reservoir`), exactly as this paragraph's B3 sentence anticipated
("every later B4-B6 test will"). B5 likewise introduces NO new filesystem
toucher, but it DOES introduce this module's first import of production
game/search code: `check_summary_binding` needs the REAL `eval_runner.
EvalGameResult` and `eval_summary.summarize_match` (not reimplementations of
either), so it imports both -- LAZILY, INSIDE the function itself, never at
module scope. This is deliberate, not a style preference: `eval_summary`
imports `eval_runner`, and `eval_runner` imports `.mcts` (which in turn
imports `.evaluator` and `.opening_diagnostics`) -- so a MODULE-level import
would violate this docstring's own opening line ("No evaluator / MCTS ...
import"), even though that whole chain is independently confirmed
mlx/torch-free (grepped `^import mlx`/`^import torch`/`^from mlx`/`^from
torch` across `eval_runner.py`, `eval_summary.py`, `mcts.py`,
`evaluator.py`, `opening_diagnostics.py`, `eval_replay.py`,
`game/twixt_state.py`, `game/__init__.py`: none). Verified at test time
(test_module_import_does_not_pull_eval_runner_or_eval_summary, the same
subprocess idiom as the mlx/torch check): merely IMPORTING this module --
never calling `check_summary_binding` -- leaves `eval_runner`/
`eval_summary`/`mcts`/`evaluator` out of `sys.modules`. B6 likewise
introduces NO new filesystem toucher and NO new lazy import: `qualify_core`
and its default `preflight` (`default_preflight`) read only `protocol` and
the already-loaded `measurements` -- specifically `measurements.
sidecars_by_idx`, the replay sidecars B3's `measure_reservoir` already
loaded, never a second read of a `replay_path` off disk (unlike
`fpu_dev_corpus_v2.v2_preflight_source`, which is deliberately NOT used
here for exactly that reason -- see `default_preflight`'s own docstring).
It only WIDENS an existing MODULE-level import (see the next paragraph):
two more names from the already-imported `fpu_dev_corpus_v2` -- both Task
2/Task 4 functions from that module's own PURE SECTION (NO MCTS / evaluator
/ GPU / MLX, per that module's own docstring), so this adds no
import-purity risk despite widening the import surface from one name to
three.

Module-level imports are stdlib ONLY, PLUS one intra-package import: `from
.fpu_dev_corpus_v2 import (_V2_CORPUS_SOURCES, enumerate_v2_proposals,
v2_geometry_feasibility)` (design Sec 6's "import only the shared ...
constant" seam -- here narrowed to exactly three names: the v2 corpus's own
result-determining source-file tuple (needed so THIS module's
`source_file_sha1s` measurement can include the v2 corpus sources without
duplicating that list), plus the two PURE functions B6's `default_preflight`
composes). This is deliberately NOT the Sec 6 circular-import risk:
`fpu_dev_corpus_v2.py` is itself import-pure (verified by its own
`test_v2_module_import_pulls_no_gpu_or_mlx`), and the cycle Sec 6 actually
warns about runs the OTHER direction -- `fpu_dev_corpus_v2.run_screen`
importing (part of) THIS module -- which stays a lazy, in-function import,
a LATER task's concern (B9), not this one. Nothing else is imported from
`fpu_dev_corpus_v2` here: no `V2Config`, no `run_screen`, no evaluator/MCTS
plumbing (verified: tests/test_fpu_dev_reservoir_protocol.py::
test_module_imports_only_pure_names_from_fpu_dev_corpus_v2). B4 adds no new
import beyond two more names from the already-imported `typing` module
(`Callable`, `Optional`) -- the import surface is otherwise identical to
B3's. B5 adds NO new MODULE-level import at all: its two new production
dependencies (`eval_runner.EvalGameResult`, `eval_summary.summarize_match`)
are function-local (see above) -- the module-level import surface is
byte-identical to B4's. B6 widens the `fpu_dev_corpus_v2` import from one
name to three (above) -- its only module-level import change; `enum` (for
`QualifyStatus`) is already imported (B1, for `WriteStatus`). B7 adds ONE
new stdlib import, `hashlib` (for `protocol_sha1 = hashlib.sha1(
canonical_json_bytes(protocol)).hexdigest()`, computed from the already
in-memory protocol -- never a file read) -- no new import from
`fpu_dev_corpus_v2` (still exactly the same three names B6 established) and
no new lazy import: `derive_config`/`write_report`/`is_passed`/`is_retired`/
`run_qualify` read/write only via B1's `canonical_json_bytes`/
`write_atomic`, stdlib `json`/`pathlib`, and the already-imported B3/B6
production functions (`measure_reservoir`, `qualify_core`,
`default_preflight`). B11 adds TWO new stdlib imports, `argparse` (the
three `main` subcommands) and `shlex` (`emit-gen-command`'s shell-joined
argv) -- no new import from `fpu_dev_corpus_v2` (still exactly the same
three names B6 established) and no new lazy import: `main` only calls
`emit_protocol` (B1), `gen_command` (B2), and `run_qualify` (B7) -- three
functions already defined/imported in this same module -- and reads at
most one caller-supplied JSON document per invocation (`--params-json` or
`--protocol`) via the already-imported `json`/`pathlib`.
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

`ConformanceResult` / `check_protocol_conformance`: the first qualification
stage (design Sec 4.1) -- PURE over an already-built `ReservoirMeasurements`
(B3) and the frozen `protocol` dict: no filesystem I/O, no evaluator/MCTS/
GPU. Runs, in the spec's own presentation order, every Sec 4.1
protocol-vs-reservoir check EXCEPT summary-binding-by-reconstruction (a
SEPARATE stage below, B5's `check_summary_binding`, since it calls the real
`eval_summary.summarize_match`) and the geometric preflight (B6): game count
(JSONL rows AND replay sidecars
each == `protocol["games"]`), `game_idx` contiguity (`0..games-1`, no
gaps/dupes), each game's sidecar `seed == base_seed + game_idx` (the ONLY
place a per-game seed is recorded -- `EvalGameResult` itself carries none),
the matchup (`checkpoint_identities["reservoir_a"/"reservoir_b"]` == the
protocol's declared `name:sha1` identities, and every row's checkpoints
resolve to the pair), between-games model-color parity (even `game_idx` ->
checkpoint-A red), replay linkage (every row's sidecar exists and its own
`game_idx`/colors/`board_size` agree with the row), sidecar `"moves"`-list
well-formedness (`_check_sidecar_moves_wellformed` -- a REVIEW-FIX addition,
not part of the original Sec 4.1 list; see its own docstring), the ten
result-determining match knobs (`TEN_MATCH_KNOBS`) PLUS `workers` recorded
in `summary["config"]`, output-path derivation (`source_index_path` ==
`<match_summary_path stem>_games.jsonl`; every replay's parent directory ==
the declared `replay_dir`), within-game move-player parity (red on even
ply, distinct from the between-games color check), and generation
provenance (`generation_source_sha1s`/`generation_git_commit` == the
protocol's, subject to the Sec 10 trust boundary). `ConformanceResult`
carries `ok` plus the FIRST failing check's reason ONLY -- checks
short-circuit at the first failure (design Sec 4.1: "any failure is
MISMATCH exit 3", a single regenerate-under-the-same-protocol outcome)
rather than accumulating every gate like
`diagnose_fpu_policy_mass.SafetyVerdict`'s multi-reason tuple; a LATER
task's report (B7) is where the FULL per-check diagnostic detail belongs.
`_validate_protocol_shape` closes the one gap B1's `build_protocol`
deliberately left open (key PRESENCE only, Sec 2.1): before any check runs,
it validates the SHAPE of `checkpoint_a`/`checkpoint_b` (`path`+`identity`
sub-keys) and that `anchor` is one of the two literal role names, raising
`ValueError` (a USAGE error against a malformed protocol DOCUMENT) rather
than letting a bare KeyError/TypeError surface from deep inside a check --
deliberately minimal, covering only the nested shapes this module's OWN
checks dereference, not a full re-validation of `build_protocol`'s job.

`check_summary_binding`: the SECOND qualification stage (design Sec 4.1
amendments 3 + 5) -- PURE over `measurements` (+ `protocol`, for the
separate `generation_git_commit` check): no filesystem I/O, no
evaluator/MCTS/GPU (`eval_runner`/`eval_summary` are imported LAZILY,
inside this function -- see the TOOLING ONLY section above for why).
Reconstructs `eval_runner.EvalGameResult` rows from EVERY `measurements.
jsonl_rows` entry (`EvalGameResult(**row)` -- the row already carries every
one of that dataclass's fields; `EvalGameResult` itself has no `seed`
field, so the row's absent `seed` key is simply never read), ORDERED BY
`game_idx`, then calls the REAL `eval_summary.summarize_match(results,
a_ckpt, b_ckpt, pairing_id, config)` with `a_ckpt`/`b_ckpt`/`pairing_id`/
`config` read from `measurements.summary` ITSELF (`checkpoint_a`,
`checkpoint_b`, `pairing_id`, `config` -- exactly the values
`eval_checkpoint_match.run_match` originally passed, since `summarize_match`
writes each straight through into its own output verbatim) -- so a
faithful summary's own recorded pass-through fields reconstruct EXACTLY,
and only the fields that DEPEND on `results` (games/state_caps/board_full/
color_bias/avg_plies/a_wins/b_wins/a_score/rates/CI/elo/verdict/color
stats) are genuinely RECOMPUTED from the JSONL. Requires the reconstructed
summary's COMPLETE output to equal `measurements.summary`, via
`_strip_cli_stamped_keys` on BOTH sides -- EXCLUDING ONLY `generated_at`/
`git_commit` (`summarize_match` is "no time, no git"; `eval_checkpoint_
match.run_match` stamps both AFTER calling it) -- the whole dict, minus
those two keys, never a hand-picked partial aggregate list (design Sec 4.1:
"with no second partial aggregate list to drift"). SEPARATELY requires
`measurements.summary["git_commit"] == protocol["generation_git_commit"]`
-- independent of the body compare, which excludes `git_commit` entirely,
so a body-faithful summary stamped with the WRONG commit is still caught.
Returns `ConformanceResult(ok=True)` when both checks pass, else
`ConformanceResult(ok=False, reason=...)` naming which of the two failed
(body compare runs first). Assumes `measurements.jsonl_rows` entries are
already `EvalGameResult`-shaped -- a malformed row raises `TypeError` from
the dataclass constructor itself, exactly like `gen_command`/
`measure_reservoir`'s own "assumes already-valid input" contract; this is
not a NEW validation layer.

`reason_histogram`: the full termination-reason histogram over
`jsonl_rows` (design Sec 4.1 / Sec 3's report state machine: "the computed
reason histogram") -- counts of every `reason` value (`"win"`,
`"state_cap"`, `"board_full"`, ...) `EvalGameResult.reason` may carry.
Pure (a plain dict-counting loop, no I/O) and, unlike every `_check_*`/
`check_summary_binding` function above, produces data for the
qualification REPORT (a LATER task's, B7's, `report_out`) rather than a
pass/fail conformance compare -- `measurements.summary` has no such field
to check it against (design Sec 4.1: "the summary has no such field to
compare against").

`QualifyStatus` / `QualifyResult` / `default_preflight` / `qualify_core`:
Task B6 (design Sec 4.2, Sec 6) -- the pure qualification DECISION.
`qualify_core(protocol, measurements, *, preflight=default_preflight) ->
QualifyResult` composes `check_protocol_conformance` (B4) -> `check_summary_
binding` (B5) -> the INJECTED `preflight`, in that order, stopping at the
FIRST failure: a conformance OR summary-binding defect returns
`QualifyStatus.MISMATCH` (the preflight is NEVER called -- design Sec 4.2:
"any failure is MISMATCH exit 3"; "a corrupt/incomplete output that breaks
preflight's INPUTS is a MISMATCH, not a gate failure") -- this ordering
matters (a B5 review flagged getting it right): conformance runs FIRST so a
structurally-short reservoir (e.g. an empty/truncated JSONL index) is
caught by its game-count check before summary binding ever reconstructs
anything, since `check_summary_binding` calls the real `eval_summary.
summarize_match`, which raises a bare `ValueError` on an empty `results`
list rather than returning a clean `ConformanceResult`. Only once BOTH pass
does `preflight(measurements)` run: `feasible=True` -> `QualifyStatus.OK`;
`feasible=False` -> `QualifyStatus.GATE_FAIL`, with the preflight result's
`binding_constraint` as the reason (design Sec 4.2: "the ONLY GATE-FAIL
condition is a protocol-faithful reservoir with infeasible geometry").

`default_preflight` is the real-world default: a thin, PURE wrapper that
builds v2 proposals from `measurements.sidecars_by_idx` (the replay
sidecars B3's `measure_reservoir` already loaded -- never a second disk
read) via the REAL `enumerate_v2_proposals`, and hands them to the pure
`fpu_dev_corpus_v2.v2_geometry_feasibility` core, returning its
`V2PreflightReport` verbatim. Deliberately NOT `fpu_dev_corpus_v2.
v2_preflight_source`: that function is the I/O wrapper (it re-reads each
`rec["replay_path"]` off disk) -- calling it here would make `qualify_core`,
documented pure over `protocol` + `measurements` alone, perform a hidden
SECOND filesystem read of data already sitting in `measurements`, breaking
this module's "`measure_reservoir` is the ONE filesystem-I/O function"
invariant. It mirrors `v2_preflight_source`'s own "the SOURCE INDEX
record's game_idx is authoritative" rule -- `sidecars_by_idx` is already
keyed by each JSONL row's OWN `game_idx` (B3's `_load_sidecars`), so this
overrides the replay dict's own `game_idx` key with that authoritative int
before calling `enumerate_v2_proposals`, the exact same override
`v2_preflight_source` performs, just sourced from an already-loaded dict
instead of a second disk read.

`preflight` is an INJECTED dependency (design Sec 6's "preflight injection
for tests" clarification): a test may supply a fake `measurements ->
<object with .feasible / .binding_constraint>` callable, so a small
fabricated `ReservoirMeasurements` (far too small to ever clear the real
240-row/4-phase geometric quotas) can still exercise the OK/GATE_FAIL
branches directly; a SEPARATE test exercises the real `default_preflight`
on a genuinely large synthetic reservoir sized to clear (or just miss) the
real quotas (see tests/test_fpu_dev_reservoir_protocol.py). Every
`QualifyResult.report` records `conformance`'s outcome, and (when reached)
`summary_binding`'s and `preflight`'s, as plain dicts (`{"ok", "reason"}`
for the first two; `{"feasible", "binding_constraint"}` for the third) plus
the unconditionally-computed `reason_histogram(measurements.jsonl_rows)`
(B5) -- `summary_binding`/`preflight` are `None` in the report when that
stage was never reached. Deliberately minimal per stage, mirroring
`ConformanceResult`'s own "first failing reason only, no accumulated list"
precedent -- the full persisted report artifact is a LATER task's concern
(B7's `write_report`), which can always re-derive richer detail by calling
each stage again. `qualify_core` performs NO filesystem I/O of its own
(`check_summary_binding`'s lazy production import is a CODE import, not
reservoir-data I/O) -- `measure_reservoir` (B3) remains the ONE
filesystem-I/O function in the whole qualification pipeline.

REVIEW FIX (this task, over the committed B6 work): a reviewer reproduced a
raw, uncaught `KeyError: 'moves'` escaping `qualify_core` for a corrupt
sidecar (a `"moves"` key deleted) that still passed BOTH `check_protocol_
conformance` and `check_summary_binding` -- `_check_move_player_parity`
softens an absent `"moves"` key to `sidecar.get("moves") or []` (vacuously
passing), so `default_preflight` was the FIRST stage to dereference
`sidecar["moves"]` (via `enumerate_v2_proposals` -> `build_fpu_dev_corpus.
per_ply_n_legal`), raising raw instead of the spec-mandated MISMATCH (design
Sec 4.2: "a corrupt/incomplete output that breaks preflight's inputs is a
MISMATCH, not a gate failure"). Two independent layers now close this: (1) a
new B4 check, `_check_sidecar_moves_wellformed`, requiring the minimum shape
the LATER per-move derefs consume (`"moves"` present, a `list`, every element
a mapping carrying every `_REQUIRED_MOVE_FIELDS` key -- int-convertible
`"n_legal"` for `per_ply_n_legal`, PLUS `"ply"`/`"player"` for the later
`_check_move_player_parity` -- all three unconditionally written by
`eval_replay.ply_record`, so a genuine reservoir never takes
`per_ply_n_legal`'s sparse-reconstruction fallback, which this check
deliberately does not validate the shape of); and (2) `qualify_core` itself
now wraps its `preflight(measurements)` call in a narrow `try/except
(KeyError, TypeError, ValueError, IndexError)`, mapping any such escaping
data-shape exception to `QualifyStatus.MISMATCH` -- belt-and-suspenders for
any corrupt shape the conformance check does not enumerate. The except clause
is deliberately narrow, not a bare `except`: a genuine LOGIC bug inside
`v2_geometry_feasibility`/`enumerate_v2_proposals` (e.g. tripping one of their
own internal `assert`s) still raises `AssertionError` uncaught -- only
garden-variety data-shape complaints become MISMATCH. NOTE the layers are
NOT redundant here: the `_check_move_player_parity` raw crash is INSIDE
`check_protocol_conformance`, which `qualify_core`'s preflight-scoped
try/except does NOT cover -- so for that specific field the CONFORMANCE check
(layer 1) is the ONLY thing standing between a tampered sidecar and a raw
exit-1 crash. A follow-up review extended layer 1 from `"n_legal"`-only to
the full `_REQUIRED_MOVE_FIELDS` set to close exactly that gap.

`_parse_args` / `main` -- Task B11 (design Sec 3), the CLI. Three argparse
subcommands over the pure functions + `run_qualify` above -- pure glue,
performing no qualification/derivation logic of its own:

`emit-protocol` freezes a `reservoir_protocol.json` from a caller-supplied
params mapping, loaded from a required `--params-json <path>` (a JSON file
holding every `PROTOCOL_SCHEMA_KEYS` field) rather than one flag per schema
field: several fields are themselves nested dicts/lists
(`checkpoint_a`/`checkpoint_b`, `phase_allocation`, `late_floors`,
`enumerator_params`, `forbidden_manifests`, ...) that argparse has no
natural per-flag encoding for, so a single reviewable JSON document an
operator edits directly is "the simplest thing that lets an operator freeze
a protocol deterministically" (task B11 brief) -- mirroring how
`fpu_dev_corpus_v2.main`'s own `--config` already works. Calls
`emit_protocol(params, out_path=--out, check=--check)` and returns its
status (`EXIT_OK`/`EXIT_MISMATCH`) verbatim; `--check` never writes
(`emit_protocol`'s own contract). A malformed `--params-json` (unreadable,
unparseable, or missing a required schema key) propagates raw from
`json.loads`/`build_protocol` -- this module's established "assumes
already-valid input" convention (`gen_command`, `measure_reservoir`,
`run_qualify` all do the same), not a new friendlier-message layer.

`emit-gen-command` loads an already-frozen protocol from a required
`--protocol <path>` and prints `gen_command(protocol)` -- and ONLY that --
shell-joined via stdlib `shlex.join`, so the printed line is directly
copy-pasteable (or pipeable) as a real shell command; zero-GPU, zero-write.
Always returns `EXIT_OK` (a malformed/missing `--protocol` propagates raw,
same convention as above).

`qualify` loads a frozen protocol from a required `--protocol <path>` and
calls `run_qualify(protocol_path, check=--check)` UNCHANGED -- no
`preflight=` override -- so the CLI always wires the REAL
`default_preflight` (design Sec 11's test-matrix note: "The CLI always
wires the real preflight"; only this module's own tests inject a fake one,
directly against `run_qualify`/`qualify_core`, never through `main`).
Returns `run_qualify`'s own exit code verbatim
(`EXIT_OK`/`EXIT_MISMATCH`/`EXIT_GATE_FAIL`, or `EXIT_USAGE` for an
already-PASSED protocol re-qualified without `--check`) -- `main` performs
NO status-to-exit-code mapping of its own; `run_qualify` (B7) already IS
that mapping. `qualify` NEVER launches generation (design's TOOLING ONLY
constraint, restated at the CLI boundary): its whole call chain --
`run_qualify` -> `measure_reservoir` (the ONE filesystem-I/O function in
the qualification pipeline) -> `qualify_core` -> `default_preflight` --
only ever reads/hashes an ALREADY-GENERATED reservoir, never runs the
evaluator/MCTS or loads checkpoint weights.

None of the three subcommands takes `--mode`/`select`/`screen` (design Sec
3 draws this module's CLI boundary at protocol/qualify only): those two
stages stay in `fpu_dev_corpus_v2.main`, a deliberately separate CLI over a
deliberately separate module.

Argparse's OWN usage errors -- a missing required flag, an unrecognized
subcommand, or no subcommand at all (`add_subparsers(..., required=True)`)
-- raise `SystemExit(2)` from inside `_parse_args`, before `main`'s own
body ever runs; `2` is already `EXIT_USAGE`'s value (design Sec 3), so no
separate mapping is needed. `if __name__ == "__main__": raise
SystemExit(main())` at the module foot mirrors `fpu_dev_corpus_v2.py`'s own
CLI convention exactly.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from . import fpu_provenance
from .fpu_dev_corpus_v2 import (
    _V2_CONFIG_REQUIRED_KEYS,
    _V2_CORPUS_SOURCES,
    enumerate_v2_proposals,
    v2_geometry_feasibility,
)

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
# imported from `fpu_dev_corpus_v2`) -- which, as of Task B8, ALREADY
# includes this qualification module itself (spec Sec 2.2 amendment 4: "the
# qualification module is result-determining for the corpus it produces" --
# added to the v2 set ONLY; `fpu_dev_corpus_v2._V2_CORPUS_SOURCES` never
# touches v1's own `build_fpu_dev_corpus._CORPUS_SOURCES`). No further
# concatenation happens here: appending this module's own path a SECOND time
# would silently double-count it in `source_file_sha1s` (a basename-keyed
# dict would just overwrite, but the no-duplicates assert below exists
# precisely to catch this before that point) -- Task B8 is the single source
# of truth for "is this module in the v2 source set," not this line.
QUALIFICATION_SOURCE_FILES: Tuple[Path, ...] = _V2_CORPUS_SOURCES
assert len(QUALIFICATION_SOURCE_FILES) == len(set(QUALIFICATION_SOURCE_FILES)), (
    "QUALIFICATION_SOURCE_FILES has a duplicate entry")
assert (_MODULE_DIR / "fpu_dev_reservoir_protocol.py") in QUALIFICATION_SOURCE_FILES, (
    "QUALIFICATION_SOURCE_FILES must include this qualification module "
    "itself -- fpu_dev_corpus_v2._V2_CORPUS_SOURCES should carry it "
    "(Task B8, spec Sec 2.2 amendment 4)")


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


# ---------------------------------------------------------------------------
# Protocol conformance -- Task B4 (design Sec 4.1). PURE over `protocol` +
# `measurements` (B3's `ReservoirMeasurements`): every check below is a
# plain dict/string/Path comparison over already-in-memory values -- no
# filesystem I/O, no evaluator/MCTS/GPU. `measure_reservoir` remains the
# ONE filesystem-I/O function in the whole qualification pipeline.
# ---------------------------------------------------------------------------

# The two checkpoint roles a protocol's `anchor` field may name -- also the
# two top-level protocol keys `_validate_protocol_shape` shape-checks
# (design Sec 2.1: "checkpoint_a and checkpoint_b ... plus anchor:
# 'checkpoint_a' | 'checkpoint_b'").
_CHECKPOINT_ROLES: Tuple[str, str] = ("checkpoint_a", "checkpoint_b")

# The ten result-determining match knobs (design Sec 2.1 amendment 4) --
# recorded verbatim in `measurements.summary["config"]`
# (`eval_checkpoint_match.run_match`: `config_dict = {**asdict(config),
# "base_seed": base_seed, "workers": workers}`) and checked against the
# SAME-named `protocol` fields by `_check_match_config_knobs`. `workers` is
# checked alongside these (spec Sec 4.1: "the summary's recorded
# config.workers equals protocol.workers") but is deliberately NOT one of
# the ten -- it is operational, not result-determining (design Sec 2.1
# amendment 2) -- so it stays a separate name, not an eleventh entry here.
TEN_MATCH_KNOBS: Tuple[str, ...] = (
    "board_size", "mcts_sims", "mcts_eval_batch_size", "mcts_stall_flush_sims",
    "selection_mode", "opening_temp_plies", "temp_high", "temp_low",
    "max_moves", "base_seed",
)


@dataclasses.dataclass(frozen=True)
class ConformanceResult:
    """`check_protocol_conformance`'s result (design Sec 4.1).

    `ok=True` (`reason=None`): every check passed. `ok=False`: the FIRST
    failing check's reason, a short human-readable string naming which
    check tripped (`"game_count: ..."`, `"seed: ..."`, `"matchup: ..."`,
    etc.) -- checks stop at the first failure rather than accumulating every
    gate like `diagnose_fpu_policy_mass.SafetyVerdict`'s multi-reason
    `reasons` tuple, since design Sec 4.1 defines a SINGLE outcome for any
    conformance defect ("any failure is MISMATCH exit 3", one
    regenerate-under-the-same-protocol action regardless of how many checks
    would have failed). The full per-check diagnostic detail -- e.g. a
    termination-reason histogram -- belongs to the report a LATER task (B7)
    writes, not to this result.
    """
    ok: bool
    reason: Optional[str] = None


def _validate_protocol_shape(protocol: Mapping[str, Any]) -> None:
    """Validate the nested protocol shapes `check_protocol_conformance`
    actually dereferences, before any check runs.

    B1's `build_protocol` validates top-level `PROTOCOL_SCHEMA_KEYS`
    PRESENCE only (design Sec 2.1) -- it never inspects what is INSIDE
    `checkpoint_a`/`checkpoint_b`, nor constrains `anchor` to its two-member
    enum (a gap flagged in the B3 report: "`build_protocol` not
    enum-constraining `anchor`"). This module's checks are the first to
    DEREFERENCE those nested shapes (`protocol["checkpoint_a"]["identity"]`,
    `protocol["checkpoint_b"]["path"]`, the `anchor` string), so this raises
    a clear `ValueError` naming the problem -- a USAGE error against a
    malformed protocol DOCUMENT -- rather than letting a bare
    KeyError/TypeError surface from deep inside a later check. Deliberately
    minimal: validates only `checkpoint_a`/`checkpoint_b` each being a
    mapping with `path`/`identity` keys, and `anchor` being one of the two
    literal role names -- NOT a full re-validation of `build_protocol`'s own
    job (top-level key presence, or the other schema fields this module
    never dereferences into).
    """
    for role in _CHECKPOINT_ROLES:
        ckpt = protocol.get(role)
        if (not isinstance(ckpt, Mapping) or "path" not in ckpt
                or "identity" not in ckpt):
            raise ValueError(
                f"check_protocol_conformance: protocol[{role!r}] must be a "
                f"mapping with 'path' and 'identity' keys, got {ckpt!r}")
    if protocol.get("anchor") not in _CHECKPOINT_ROLES:
        raise ValueError(
            f"check_protocol_conformance: protocol['anchor'] must be one "
            f"of {_CHECKPOINT_ROLES!r}, got {protocol.get('anchor')!r}")


def _check_game_count(protocol: Mapping[str, Any],
                      measurements: ReservoirMeasurements) -> Optional[str]:
    """Game count (design Sec 4.1): exactly `protocol['games']` JSONL rows
    AND exactly that many replay sidecars -- checked as two independent
    sub-conditions so a reviewer can tell (from the reason text) which of
    the two collections is short, but reported as a SINGLE `"game_count"`
    check (whichever of the two disagrees first)."""
    games = protocol["games"]
    n_rows = len(measurements.jsonl_rows)
    if n_rows != games:
        return (f"game_count: protocol declares games={games} but "
                f"jsonl_rows has {n_rows} row(s)")
    n_sidecars = len(measurements.sidecars_by_idx)
    if n_sidecars != games:
        return (f"game_count: protocol declares games={games} but "
                f"sidecars_by_idx has {n_sidecars} entry/entries")
    return None


def _check_contiguous_game_idx(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Contiguity (design Sec 4.1): `jsonl_rows`' `game_idx` values run
    `0..games-1` with no gaps and no duplicates."""
    games = protocol["games"]
    idxs = sorted(int(row["game_idx"]) for row in measurements.jsonl_rows)
    expected = list(range(games))
    if idxs != expected:
        return (f"contiguity: jsonl_rows game_idx values {idxs} are not "
                f"exactly contiguous 0..{games - 1} (a gap or a duplicate)")
    return None


def _check_seed(protocol: Mapping[str, Any],
                measurements: ReservoirMeasurements) -> Optional[str]:
    """Seed range (design Sec 4.1): each game's RECORDED seed --
    `EvalGameResult` itself carries no `seed` field (`eval_runner.
    EvalGameResult`'s field set), so the only place a per-game seed is
    recorded is its replay sidecar (`eval_replay.build_replay_dict`) --
    equals the half-open range's `base_seed + game_idx`
    (`eval_runner.build_pairing_tasks`: `seed=base_seed + offset + g`,
    `offset=0` for the reservoir's single pairing). A row whose sidecar is
    altogether MISSING is skipped here -- that is `_check_replay_linkage`'s
    failure to report, not a spurious empty-seed mismatch."""
    base_seed = protocol["base_seed"]
    for row in measurements.jsonl_rows:
        game_idx = int(row["game_idx"])
        sidecar = measurements.sidecars_by_idx.get(game_idx)
        if sidecar is None:
            continue
        expected_seed = base_seed + game_idx
        actual_seed = sidecar.get("seed")
        if actual_seed != expected_seed:
            return (f"seed: game_idx={game_idx} sidecar seed={actual_seed!r}, "
                    f"expected base_seed({base_seed})+game_idx({game_idx})="
                    f"{expected_seed}")
    return None


def _check_matchup(protocol: Mapping[str, Any],
                   measurements: ReservoirMeasurements) -> Optional[str]:
    """Matchup (design Sec 4.1): the measured `checkpoint_identities`
    (`reservoir_a`/`reservoir_b`, B3) equal the protocol's DECLARED
    `name:sha1` identities, and every JSONL row's
    `red_checkpoint`/`black_checkpoint` PATHS resolve to exactly the
    `{checkpoint_a, checkpoint_b}` path pair (as a set -- WHICH one is red
    on a given game is `_check_color_parity`'s separate concern)."""
    ckpt_a_identity = protocol["checkpoint_a"]["identity"]
    ckpt_b_identity = protocol["checkpoint_b"]["identity"]
    actual_a = measurements.checkpoint_identities.get("reservoir_a")
    if actual_a != ckpt_a_identity:
        return (f"matchup: checkpoint_identities['reservoir_a']={actual_a!r} "
                f"!= protocol checkpoint_a identity={ckpt_a_identity!r}")
    actual_b = measurements.checkpoint_identities.get("reservoir_b")
    if actual_b != ckpt_b_identity:
        return (f"matchup: checkpoint_identities['reservoir_b']={actual_b!r} "
                f"!= protocol checkpoint_b identity={ckpt_b_identity!r}")

    ckpt_a_path = protocol["checkpoint_a"]["path"]
    ckpt_b_path = protocol["checkpoint_b"]["path"]
    expected_paths = {ckpt_a_path, ckpt_b_path}
    for row in measurements.jsonl_rows:
        row_paths = {row["red_checkpoint"], row["black_checkpoint"]}
        if row_paths != expected_paths:
            return (f"matchup: game_idx={row['game_idx']} red/black "
                    f"checkpoints {row_paths} do not resolve to "
                    f"{{checkpoint_a, checkpoint_b}}={expected_paths}")
    return None


def _check_color_parity(protocol: Mapping[str, Any],
                        measurements: ReservoirMeasurements) -> Optional[str]:
    """Model-color parity BETWEEN games (design Sec 4.1 amendment 4):
    even `game_idx` -> checkpoint-A plays red; odd -> checkpoint-B plays red
    (`eval_runner.build_pairing_tasks`'s own rule: `red, black = (a_ckpt,
    b_ckpt) if g % 2 == 0 else (b_ckpt, a_ckpt)`)."""
    ckpt_a_path = protocol["checkpoint_a"]["path"]
    ckpt_b_path = protocol["checkpoint_b"]["path"]
    for row in measurements.jsonl_rows:
        game_idx = int(row["game_idx"])
        expected_red = ckpt_a_path if game_idx % 2 == 0 else ckpt_b_path
        if row["red_checkpoint"] != expected_red:
            parity = "even" if game_idx % 2 == 0 else "odd"
            return (f"color_parity: game_idx={game_idx} ({parity}) expected "
                    f"red_checkpoint={expected_red!r}, got "
                    f"{row['red_checkpoint']!r}")
    return None


def _check_replay_linkage(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Replay linkage (design Sec 4.1): every row has a linked sidecar in
    `sidecars_by_idx`, and that sidecar's OWN `game_idx`, colors, and
    `board_size` agree with the row (`seed` agreement is `_check_seed`'s
    separate concern, so it is not re-checked here)."""
    board_size = protocol["board_size"]
    for row in measurements.jsonl_rows:
        game_idx = int(row["game_idx"])
        sidecar = measurements.sidecars_by_idx.get(game_idx)
        if sidecar is None:
            return (f"replay_linkage: game_idx={game_idx} has no linked "
                    f"sidecar in sidecars_by_idx")
        sidecar_game_idx = sidecar.get("game_idx")
        if sidecar_game_idx is None or int(sidecar_game_idx) != game_idx:
            return (f"replay_linkage: game_idx={game_idx} sidecar "
                    f"game_idx={sidecar_game_idx!r} does not match "
                    f"the row")
        if (sidecar.get("red_checkpoint") != row["red_checkpoint"]
                or sidecar.get("black_checkpoint") != row["black_checkpoint"]):
            return (f"replay_linkage: game_idx={game_idx} sidecar colors "
                    f"(red={sidecar.get('red_checkpoint')!r}, "
                    f"black={sidecar.get('black_checkpoint')!r}) do not "
                    f"match the row (red={row['red_checkpoint']!r}, "
                    f"black={row['black_checkpoint']!r})")
        if sidecar.get("board_size") != board_size:
            return (f"replay_linkage: game_idx={game_idx} sidecar "
                    f"board_size={sidecar.get('board_size')!r} != protocol "
                    f"board_size={board_size!r}")
    return None


# The fields EVERY move record in a sidecar's `"moves"` list must carry --
# the UNION of what the two conformance/preflight consumers dereference per
# move, so that once `_check_sidecar_moves_wellformed` passes, no LATER
# per-move access in `check_protocol_conformance` (nor in `default_preflight`)
# can raw-crash: `build_fpu_dev_corpus.per_ply_n_legal` reads `"n_legal"`
# (`int(m["n_legal"])`) and `_check_move_player_parity` reads `"ply"` +
# `"player"` (`record["ply"] % 2`, `record["player"] != ...`). Kept as a
# module-level tuple so the check and this provenance comment stay a single
# source of truth (a NEW per-move access elsewhere must add its field here).
# Presence alone is guarded for all three here; the two whose VALUE an
# unwrapped consumer OPERATES on -- `n_legal` (`int(...)`) and `ply`
# (`... % 2`) -- are additionally int-validated inside the check itself.
_REQUIRED_MOVE_FIELDS: Tuple[str, ...] = ("n_legal", "ply", "player")


def _check_sidecar_moves_wellformed(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Sidecar `"moves"`-list well-formedness -- a REVIEW-FIX addition to
    B4, not part of the original Sec 4.1 list (see the module docstring's
    "REVIEW FIX" paragraph for the full story).

    Requires the minimum shape the LATER per-move derefs in conformance +
    preflight consume: `"moves"` PRESENT, a `list`, and every element a
    mapping carrying every `_REQUIRED_MOVE_FIELDS` key -- with the two whose
    VALUE a downstream UNWRAPPED consumer operates on ALSO int-validated:
    `"n_legal"` (int-convertible, since `build_fpu_dev_corpus.
    per_ply_n_legal`'s FAST path does `int(m["n_legal"])`) and `"ply"`
    (int-convertible, since `_check_move_player_parity` does `record["ply"]
    % 2` -- arithmetic that a non-int `ply` would raise `TypeError` on).
    `"player"` is presence-only: `_check_move_player_parity` only compares
    it with `!=`, which never raises on any value. This is the ONLY path a
    protocol-conformant reservoir ever exercises -- `eval_replay.ply_record`
    unconditionally writes all three (`{"ply", "player", ..., "n_legal"}`)
    onto every move it records, so a genuine reservoir never falls into
    `per_ply_n_legal`'s sparse-reconstruction FALLBACK (which needs a much
    deeper shape -- `"row"`/`"col"` + `goal_line_trigger_probe_cases.
    position_state`'s whole-game replay). This check deliberately does NOT
    validate that deeper fallback shape -- do not over-validate a path a
    conformant reservoir never takes.

    Without this check, a corrupt/incomplete sidecar (a `"moves"` key
    entirely absent, or a move missing `"n_legal"`/`"ply"`/`"player"`) sails
    past the other B4 checks -- `_check_move_player_parity` (below) softens
    an absent `"moves"` key to `sidecar.get("moves") or []`, vacuously
    passing, but then RAW-CRASHES on `record["ply"]`/`record["player"]` for
    a present-but-fieldless move record -- and past B5 too, so
    `default_preflight` (B6) becomes the first stage to dereference
    `sidecar["moves"]`' `"n_legal"` (via `enumerate_v2_proposals` ->
    `per_ply_n_legal`). EITHER raw crash violates the spec-mandated MISMATCH
    (design Sec 4.2: "a corrupt/incomplete output that breaks preflight's
    inputs is a MISMATCH, not a gate failure") -- and the `_check_move_
    player_parity` crash is INSIDE `check_protocol_conformance` itself, which
    `qualify_core` does NOT wrap in its preflight-only try/except, so only
    catching it HERE keeps the exit-3 contract. Placed in
    `_CONFORMANCE_CHECKS` right after `_check_replay_linkage` -- once a row's
    sidecar is known to exist and match the row, THIS check validates its
    `"moves"` list's shape, ahead of every LATER check that also reads move
    records (`_check_move_player_parity`).

    Returns the FIRST malformed game's reason (naming its `game_idx` and
    the exact problem), mirroring every other `_check_*` helper's
    first-failure contract; `None` when every linked sidecar's `"moves"` is
    well-formed. A row with no linked sidecar at all is skipped (`_check_
    replay_linkage`'s own failure to report), the same guard `_check_seed`
    already uses.
    """
    for row in measurements.jsonl_rows:
        game_idx = int(row["game_idx"])
        sidecar = measurements.sidecars_by_idx.get(game_idx)
        if sidecar is None:
            continue
        if "moves" not in sidecar:
            return (f"sidecar_moves: game_idx={game_idx} sidecar has no "
                    f"'moves' key")
        moves = sidecar["moves"]
        if not isinstance(moves, list):
            return (f"sidecar_moves: game_idx={game_idx} sidecar 'moves' "
                    f"is not a list (got {type(moves).__name__})")
        for i, m in enumerate(moves):
            if not isinstance(m, Mapping):
                return (f"sidecar_moves: game_idx={game_idx} moves[{i}] is "
                        f"not a mapping (got {type(m).__name__})")
            for field in _REQUIRED_MOVE_FIELDS:
                if field not in m:
                    return (f"sidecar_moves: game_idx={game_idx} moves[{i}] is "
                            f"missing required field {field!r}")
            try:
                int(m["n_legal"])
            except (TypeError, ValueError):
                return (f"sidecar_moves: game_idx={game_idx} moves[{i}] "
                        f"'n_legal'={m['n_legal']!r} is not int-convertible")
            # `_check_move_player_parity` (a LATER, unwrapped conformance
            # check) does `record["ply"] % 2` -- arithmetic, not a deref --
            # so a present-but-non-int `ply` (`"abc"`/`None`/`[1,2]`/`{}`)
            # would raise a raw `TypeError` INSIDE check_protocol_conformance
            # (which qualify_core does NOT wrap). Value-validate it here, the
            # SAME int-convertibility guard `n_legal` already gets. `player`
            # needs no such guard: `_check_move_player_parity` only compares
            # it with `!=`, which never raises.
            try:
                int(m["ply"])
            except (TypeError, ValueError):
                return (f"sidecar_moves: game_idx={game_idx} moves[{i}] "
                        f"'ply'={m['ply']!r} is not int-convertible")
    return None


def _check_match_config_knobs(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """The ten result-determining match knobs (`TEN_MATCH_KNOBS`) PLUS
    `workers`, recorded in `measurements.summary["config"]`
    (`eval_checkpoint_match.run_match`'s `config_dict`), each equal the
    SAME-named `protocol` field."""
    config = measurements.summary.get("config") or {}
    for knob in TEN_MATCH_KNOBS:
        expected = protocol[knob]
        actual = config.get(knob)
        if actual != expected:
            return (f"match_config: summary['config'][{knob!r}]={actual!r} "
                    f"!= protocol[{knob!r}]={expected!r}")
    expected_workers = protocol["workers"]
    actual_workers = config.get("workers")
    if actual_workers != expected_workers:
        return (f"match_config: summary['config']['workers']="
                f"{actual_workers!r} != protocol['workers']="
                f"{expected_workers!r}")
    return None


def _check_output_path_derivation(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Output-path derivation (design Sec 4.1 amendment 3):
    `protocol['source_index_path']` equals
    `<protocol['match_summary_path'] stem>_games.jsonl` (the
    `eval_checkpoint_match._write_outputs` rule) -- a pure protocol
    self-consistency check needing no `measurements` field; and every row's
    `replay_path` lives directly under the protocol's declared
    `replay_dir` (`eval_replay.write_replay`: `path = os.path.join(
    replay_dir, replay_filename(game_idx))`), proving the generator
    actually wrote into the DECLARED directory. (Existence of the files
    themselves was already enforced by `measure_reservoir`'s own
    `_require_readable_files`/raw-read guards at measurement time -- not
    re-checked here, since this stage is pure over already-loaded data.)"""
    stem, _ext = os.path.splitext(protocol["match_summary_path"])
    expected_index = f"{stem}_games.jsonl"
    if protocol["source_index_path"] != expected_index:
        return (f"output_path: protocol source_index_path="
                f"{protocol['source_index_path']!r} != derived "
                f"{expected_index!r} (<match_summary_path stem>_games.jsonl)")

    replay_dir = Path(protocol["replay_dir"])
    for row in measurements.jsonl_rows:
        actual_dir = Path(row["replay_path"]).parent
        if actual_dir != replay_dir:
            return (f"output_path: game_idx={row['game_idx']} replay_path="
                    f"{row['replay_path']!r} is not directly under protocol "
                    f"replay_dir={protocol['replay_dir']!r}")
    return None


def _check_move_player_parity(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Within-game move-player parity (design Sec 4.1, checked separately
    from the between-games `_check_color_parity` above): within each
    replay's `moves` (`eval_replay.ply_record`), the mover alternates by
    ply -- red on even ply, black on odd (`eval_runner.play_eval_game`:
    `TwixtState(..., to_move="red", ...)`, red moves first at ply 0)."""
    for game_idx, sidecar in measurements.sidecars_by_idx.items():
        for record in sidecar.get("moves") or []:
            ply = record["ply"]
            expected_player = "red" if ply % 2 == 0 else "black"
            if record["player"] != expected_player:
                return (f"move_player_parity: game_idx={game_idx} ply={ply} "
                        f"player={record['player']!r}, expected "
                        f"{expected_player!r}")
    return None


def _check_generation_provenance(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> Optional[str]:
    """Generation provenance (design Sec 4.1/Sec 2.1 amendment 8): the
    measured `generation_source_sha1s` (over the Sec 2.1 thirteen-module
    list, B3) and `generation_git_commit` equal the protocol's declared
    values -- subject to the Sec 10 trust boundary (this proves the sources
    AS THEY EXIST AT QUALIFY TIME, not that those exact bytes executed)."""
    expected_sources = protocol["generation_source_sha1s"]
    if measurements.generation_source_sha1s != expected_sources:
        return (f"generation_provenance: generation_source_sha1s="
                f"{measurements.generation_source_sha1s!r} != protocol "
                f"generation_source_sha1s={expected_sources!r}")
    expected_commit = protocol["generation_git_commit"]
    if measurements.generation_git_commit != expected_commit:
        return (f"generation_provenance: generation_git_commit="
                f"{measurements.generation_git_commit!r} != protocol "
                f"generation_git_commit={expected_commit!r}")
    return None


# Spec Sec 4.1's own presentation order -- also the order checks run in:
# `check_protocol_conformance` returns the FIRST failure, so this ordering
# only affects WHICH single reason is reported when more than one check
# would have failed (each defect test in the test suite isolates exactly
# one broken check, so the order never changes a defect test's outcome).
# `_check_sidecar_moves_wellformed` is the ONE entry NOT from that spec list
# (a review-fix addition -- see its own docstring): inserted right after
# `_check_replay_linkage`, so a sidecar's `"moves"` shape is validated
# immediately once that sidecar is known to exist and match its row, ahead
# of every LATER check that also reads move records
# (`_check_move_player_parity`).
_CONFORMANCE_CHECKS: Tuple[
    Callable[[Mapping[str, Any], ReservoirMeasurements], Optional[str]], ...
] = (
    _check_game_count,
    _check_contiguous_game_idx,
    _check_seed,
    _check_matchup,
    _check_color_parity,
    _check_replay_linkage,
    _check_sidecar_moves_wellformed,
    _check_match_config_knobs,
    _check_output_path_derivation,
    _check_move_player_parity,
    _check_generation_provenance,
)


def check_protocol_conformance(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> ConformanceResult:
    """The first qualification stage (design Sec 4.1) -- PURE over an
    already-built `ReservoirMeasurements` (B3) and the frozen `protocol`
    dict: no filesystem I/O, no evaluator/MCTS/GPU (`measure_reservoir`
    remains the ONE I/O function in the whole qualification pipeline).

    Runs every Sec 4.1 protocol-vs-reservoir check EXCEPT summary-binding-
    by-reconstruction (a SEPARATE stage, `check_summary_binding` below (B5)
    -- it calls the real `eval_summary.summarize_match`) and the geometric
    preflight (B6): game count,
    `game_idx` contiguity, per-game seed, the matchup (identities + every
    row's checkpoints), between-games model-color parity, replay linkage,
    sidecar `"moves"`-list well-formedness (`_check_sidecar_moves_
    wellformed` -- a REVIEW-FIX addition, not from the original Sec 4.1
    list), the ten match knobs + `workers`, output-path derivation,
    within-game move-player parity, and generation provenance -- see each
    `_check_*` helper's own docstring for its exact rule.

    Returns `ConformanceResult(ok=True)` when every check passes, else
    `ConformanceResult(ok=False, reason=<first failing check's reason>)` --
    checks stop at the FIRST failure (design Sec 4.1: "any failure is
    MISMATCH exit 3", a single outcome, not an accumulated list).

    `_validate_protocol_shape` runs first and RAISES `ValueError` (a usage
    error, not a `ConformanceResult`) for a malformed protocol DOCUMENT --
    e.g. a `checkpoint_a` missing its `identity` key, or an `anchor` outside
    its two-member enum -- closing the nested-shape validation gap B1's
    `build_protocol` deliberately left open (key presence only).
    """
    _validate_protocol_shape(protocol)
    for check in _CONFORMANCE_CHECKS:
        reason = check(protocol, measurements)
        if reason is not None:
            return ConformanceResult(ok=False, reason=reason)
    return ConformanceResult(ok=True, reason=None)


# ---------------------------------------------------------------------------
# Summary <-> JSONL binding by reconstruction -- Task B5 (design Sec 4.1
# amendments 3, 5). PURE over `protocol` + `measurements`: no filesystem
# I/O, no evaluator/MCTS/GPU -- `eval_summary`/`eval_runner` are imported
# LAZILY, inside `check_summary_binding` itself (see the module docstring's
# TOOLING ONLY section for why: importing them at MODULE level would
# transitively pull `mcts`/`evaluator`/`opening_diagnostics` -- this
# module's own declared "No evaluator / MCTS ... import" contract -- even
# though that chain is independently confirmed mlx/torch-free).
# ---------------------------------------------------------------------------

# The two keys `eval_checkpoint_match.run_match` stamps onto `summarize_
# match`'s otherwise-pure output AFTER calling it (`summary["git_commit"] =
# _git_commit()`; `summary["generated_at"] = datetime.now(...).isoformat()`,
# lines 69-70) -- `summarize_match` itself never produces either (this
# module's own "no time, no git"). Excluded from the summary<->JSONL body
# compare (spec Sec 4.1 amendment 5); `git_commit` is separately compared
# against `protocol["generation_git_commit"]` by `check_summary_binding`.
_SUMMARY_CLI_STAMPED_KEYS: Tuple[str, str] = ("generated_at", "git_commit")


def _strip_cli_stamped_keys(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """`summary` minus `_SUMMARY_CLI_STAMPED_KEYS` -- the two CLI-stamped
    keys `eval_summary.summarize_match` never produces. Applied to BOTH the
    reconstructed and the supplied summary before comparing them, so the
    comparison is symmetric (never assumes which side may or may not carry
    the two keys)."""
    return {k: v for k, v in summary.items()
            if k not in _SUMMARY_CLI_STAMPED_KEYS}


def check_summary_binding(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements) -> ConformanceResult:
    """Summary <-> JSONL binding by reconstruction (design Sec 4.1
    amendments 3, 5) -- Task B5, the SECOND qualification stage. PURE over
    `measurements` (+ `protocol`, for the separate `generation_git_commit`
    check): no filesystem I/O, no evaluator/MCTS/GPU.

    Reconstructs `eval_runner.EvalGameResult` rows from EVERY `measurements.
    jsonl_rows` entry (`EvalGameResult(**row)` -- the row already carries
    every one of that dataclass's fields, task_id through replay_path, per
    `ReservoirMeasurements.jsonl_rows`'s own docstring; `EvalGameResult` has
    no `seed` field, so the row's absent `seed` key is simply never read),
    ORDERED BY `game_idx`, then calls the REAL, pure `eval_summary.
    summarize_match(results, a_ckpt, b_ckpt, pairing_id, config)` --
    `a_ckpt`/`b_ckpt`/`pairing_id`/`config` read from `measurements.summary`
    ITSELF (`checkpoint_a`, `checkpoint_b`, `pairing_id`, `config` -- exactly
    the values `eval_checkpoint_match.run_match` originally passed, since
    `summarize_match` writes each straight through into its own output
    verbatim; see that function's `base` dict) -- so a faithful summary's
    own recorded pass-through fields reconstruct EXACTLY, and only the
    fields that DEPEND on `results` (games/state_caps/board_full/
    color_bias/avg_plies/a_wins/b_wins/a_score/rates/CI/elo/verdict/color
    stats) are genuinely RECOMPUTED from the JSONL -- exactly what makes
    this a check that "a summary from a DIFFERENT run with the same
    settings" paired with THIS reservoir cannot pass (its recomputed numbers
    would not match the pass-through fields it was itself filed with).

    Requires the reconstructed summary's COMPLETE output to equal
    `measurements.summary`, comparing both through `_strip_cli_stamped_keys`
    -- EXCLUDING ONLY `generated_at`/`git_commit` -- never a hand-picked
    partial aggregate list (design Sec 4.1: "with no second partial
    aggregate list to drift"). SEPARATELY requires `measurements.
    summary["git_commit"] == protocol["generation_git_commit"]` --
    independent of the body compare, which excludes `git_commit` entirely,
    so a body-faithful summary stamped with the WRONG commit is still
    caught. The two checks run in that order (body, then git_commit); each
    returns immediately on its own failure.

    Returns `ConformanceResult(ok=True)` when both checks pass, else
    `ConformanceResult(ok=False, reason=...)` -- mirrors `check_protocol_
    conformance`'s single-reason contract.

    Assumes `measurements.jsonl_rows` entries are already `EvalGameResult`-
    shaped (every field present, no unexpected extras) -- a malformed row
    raises `TypeError` from the dataclass constructor itself, exactly like
    `gen_command`/`measure_reservoir`'s own "assumes already-valid input"
    contract; this is not a new validation layer over `measurements` itself
    (that remains `measure_reservoir`'s and B4's job).
    """
    from .eval_runner import EvalGameResult
    from .eval_summary import summarize_match

    summary = measurements.summary
    results = [
        EvalGameResult(**row)
        for row in sorted(measurements.jsonl_rows, key=lambda r: int(r["game_idx"]))
    ]
    reconstructed = summarize_match(
        results,
        summary.get("checkpoint_a"),
        summary.get("checkpoint_b"),
        summary.get("pairing_id"),
        summary.get("config"),
    )

    actual_body = _strip_cli_stamped_keys(summary)
    reconstructed_body = _strip_cli_stamped_keys(reconstructed)
    if reconstructed_body != actual_body:
        return ConformanceResult(
            ok=False,
            reason=(f"summary_binding: reconstructed summary body "
                    f"{reconstructed_body!r} != supplied summary body "
                    f"{actual_body!r} (excluding generated_at/git_commit)"))

    expected_commit = protocol["generation_git_commit"]
    actual_commit = summary.get("git_commit")
    if actual_commit != expected_commit:
        return ConformanceResult(
            ok=False,
            reason=(f"summary_binding: measurements.summary['git_commit']="
                    f"{actual_commit!r} != protocol['generation_git_commit']="
                    f"{expected_commit!r}"))

    return ConformanceResult(ok=True, reason=None)


def reason_histogram(jsonl_rows: List[dict]) -> Dict[str, int]:
    """The full termination-reason histogram over `jsonl_rows` (design Sec
    4.1 / Sec 3's report state machine: "the computed reason histogram") --
    counts of every `reason` value (`"win"`, `"state_cap"`, `"board_full"`,
    `"unknown_error"`, ... -- whatever `EvalGameResult.reason` actually
    carries) across every row. Pure: a plain dict-counting loop, no I/O, no
    dependency on `check_summary_binding` or any `_check_*` helper.

    For the qualification REPORT (a LATER task's, B7's, `report_out`), NOT
    a conformance compare -- `measurements.summary` has no such field to
    check it against (design Sec 4.1: "the summary has no such field to
    compare against"), so unlike every function above this returns a plain
    `dict`, never a `ConformanceResult`.
    """
    histogram: Dict[str, int] = {}
    for row in jsonl_rows:
        reason = row["reason"]
        histogram[reason] = histogram.get(reason, 0) + 1
    return histogram


# ---------------------------------------------------------------------------
# qualify_core -- Task B6 (design Sec 4.2, Sec 6). PURE: composes B4's
# `check_protocol_conformance`, B5's `check_summary_binding`, and an
# INJECTED geometric-feasibility `preflight` -- no filesystem I/O, no
# evaluator/MCTS/GPU of its own (though `check_summary_binding`, which it
# calls, still LAZILY imports `eval_runner`/`eval_summary` -- see that
# function's own docstring; `qualify_core` introduces no NEW import).
# ---------------------------------------------------------------------------

class QualifyStatus(enum.Enum):
    """`qualify_core`'s three possible outcomes (design Sec 4.2).

    `OK` -- every stage passed; the caller may proceed to emit the config (a
      LATER task, B7).
    `MISMATCH` -- a protocol-conformance (B4) or summary-binding (B5)
      defect: "regenerate under the same protocol" (spec Sec 4.1/Sec 7). The
      preflight is never reached in this case. A REVIEW-FIX addition adds a
      THIRD MISMATCH path: the preflight itself WAS reached but raised a
      data-shape exception (`KeyError`/`TypeError`/`ValueError`/
      `IndexError`) on corrupt/incomplete input that slipped past
      conformance and binding -- caught by `qualify_core`'s own guard (see
      its docstring) and likewise mapped to MISMATCH, never a raw crash.
    `GATE_FAIL` -- a protocol-FAITHFUL reservoir whose geometry cannot
      support the v2 corpus: "retire this protocol version" (spec Sec
      4.2/Sec 7).

    Distinct from the module's `EXIT_*` int constants (design Sec 3's CLI
    exit-code vocabulary) -- mapping a `QualifyStatus` to its process exit
    code is the CLI's job (a LATER task, B11), not this pure core's.
    """
    OK = "OK"
    MISMATCH = "MISMATCH"
    GATE_FAIL = "GATE_FAIL"


@dataclasses.dataclass(frozen=True)
class QualifyResult:
    """`qualify_core`'s result (design Sec 4.2).

    `status`: a `QualifyStatus` (`OK` / `MISMATCH` / `GATE_FAIL`).
    `reason`: `None` on `OK`; on `MISMATCH`, the tripped stage's own
      `ConformanceResult.reason` (conformance's, or -- only when conformance
      itself passed -- summary binding's, or -- a REVIEW-FIX addition, only
      when BOTH passed -- a synthesized reason naming the exception an
      injected/default preflight raised on corrupt/incomplete input); on
      `GATE_FAIL`, the preflight result's `binding_constraint`.
    `report`: a plain dict recording every stage's own outcome PLUS the full
      termination-reason histogram (`reason_histogram(measurements.
      jsonl_rows)`, B5, computed UNCONDITIONALLY -- even on an early
      MISMATCH, since it is a cheap, pure fact about the JSONL alone) --
      `{"conformance": {"ok", "reason"}, "summary_binding": {"ok", "reason"}
      | None, "preflight": {"feasible", "binding_constraint"} | None,
      "reason_histogram": {...}}`. `summary_binding`/`preflight` are `None`
      when that stage was never REACHED (conformance short-circuits both; a
      conformance-level MISMATCH never even attempts summary binding, per
      this stage's own sequencing). Deliberately minimal per stage --
      mirrors `ConformanceResult`'s own "first failing reason only, no
      accumulated list" precedent; the FULL per-check diagnostic detail (if
      ever needed beyond `ok`/`reason`/`feasible`/`binding_constraint`)
      belongs to a LATER task's persisted report artifact (B7's
      `write_report`), which can always re-derive it by calling each stage
      again.
    """
    status: QualifyStatus
    reason: Optional[str]
    report: Dict[str, Any]


def default_preflight(measurements: ReservoirMeasurements) -> Any:
    """`qualify_core`'s DEFAULT `preflight` (design Sec 4.2/Sec 6) -- a thin,
    PURE wrapper that builds v2 proposals from the ALREADY-LOADED
    `measurements.sidecars_by_idx` and hands them to the pure
    `fpu_dev_corpus_v2.v2_geometry_feasibility` core, returning its
    `V2PreflightReport` verbatim.

    Deliberately NOT `fpu_dev_corpus_v2.v2_preflight_source`: that function
    is the I/O wrapper (that module's own docstring: "the ONE impure
    function [in fpu_dev_corpus_v2.py] ... stdlib json/pathlib only") -- it
    takes `records` and RE-READS each `rec["replay_path"]` off disk. Calling
    it here would make `qualify_core` -- documented pure over `protocol` +
    `measurements` alone -- perform a SECOND, hidden filesystem read of
    exactly the replay data `measure_reservoir` (B3) already loaded into
    `measurements.sidecars_by_idx`, breaking this module's own
    "`measure_reservoir` is the ONE filesystem-I/O function in the whole
    qualification pipeline" invariant. Design Sec 6 pins this: "the pure
    qualification core accepts the preflight as an injected dependency" --
    THIS is that dependency's real-world default; the CLI (a later task,
    B11) always wires this same function, never `v2_preflight_source`.

    Mirrors `v2_preflight_source`'s own "the SOURCE INDEX record's game_idx
    is authoritative" rule -- `measurements.sidecars_by_idx` is already
    keyed by each JSONL row's OWN `game_idx` (B3's `_load_sidecars`), so
    this overrides the replay dict's own `game_idx` key with that
    authoritative int (`{**sidecar, "game_idx": game_idx}`) before calling
    the REAL `enumerate_v2_proposals` -- the exact same override
    `v2_preflight_source` performs, just sourced from an already-loaded dict
    instead of a second disk read, so the preflight can never drift from
    the enumeration the (a later, operator) screen stage will actually use.
    """
    proposals_by_game: Dict[int, List[dict]] = {}
    for game_idx, sidecar in measurements.sidecars_by_idx.items():
        replay = {**sidecar, "game_idx": game_idx}
        proposals_by_game[game_idx] = enumerate_v2_proposals(replay)
    return v2_geometry_feasibility(proposals_by_game)


def qualify_core(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements,
        *,
        preflight: Callable[[ReservoirMeasurements], Any] = default_preflight,
) -> QualifyResult:
    """The pure qualification decision (design Sec 4.2/Sec 6) -- composes
    B4's `check_protocol_conformance`, B5's `check_summary_binding`, and the
    (injectable) geometric `preflight`, in that ORDER, short-circuiting at
    the first failure:

      1. `check_protocol_conformance(protocol, measurements)` -- FIRST, so a
         structurally-broken reservoir (e.g. an empty or short JSONL index)
         is caught here, before summary binding ever runs (`check_summary_
         binding` reconstructs `EvalGameResult` rows and calls the real
         `eval_summary.summarize_match`, which would raise a raw exception
         on a malformed/empty `jsonl_rows` rather than return a clean
         `ConformanceResult` -- conformance's `_check_game_count` catches
         that mismatch FIRST, so binding never sees it).
      2. `check_summary_binding(protocol, measurements)` -- SECOND, only
         once conformance has already passed.
      3. `preflight(measurements)` -- LAST, only once both prior stages have
         passed; the geometric feasibility (design Sec 4.2).

    A conformance OR summary-binding failure returns `QualifyStatus.
    MISMATCH` with that stage's own reason -- the preflight is NEVER called
    (design Sec 4.2: "any failure is MISMATCH exit 3"; "a corrupt/incomplete
    output that breaks preflight's INPUTS is a MISMATCH, not a gate
    failure"). Only once both pass does `preflight` run: `feasible=True` ->
    `QualifyStatus.OK`; `feasible=False` -> `QualifyStatus.GATE_FAIL` with
    the preflight result's `binding_constraint` as the reason (design Sec
    4.2: "the ONLY GATE-FAIL condition is a protocol-faithful reservoir with
    infeasible geometry").

    REVIEW FIX (this task): `preflight(measurements)` itself is now called
    inside a narrow `try/except (KeyError, TypeError, ValueError,
    IndexError)`. A reviewer reproduced a corrupt sidecar (a `"moves"` key
    deleted) that passed BOTH conformance and binding yet made
    `default_preflight` raise a raw `KeyError` -- exactly the "corrupt/
    incomplete output that breaks preflight's inputs" case design Sec 4.2
    says must be MISMATCH, not a crash. `_check_sidecar_moves_wellformed`
    (B4, see its own docstring) now catches THAT specific shape earlier, at
    the conformance stage -- but this except clause is the general
    belt-and-suspenders: ANY data-shape exception a conformance check does
    not (or cannot, for an entirely custom-injected `preflight`) enumerate
    still becomes `QualifyStatus.MISMATCH`, with a synthesized reason naming
    the exception, rather than escaping `qualify_core` raw. Deliberately
    NARROW, not a bare `except`: a genuine LOGIC bug inside
    `v2_geometry_feasibility`/`enumerate_v2_proposals` (e.g. tripping one of
    THEIR OWN internal `assert`s) still raises `AssertionError`, uncaught --
    only garden-variety data-shape complaints are reclassified.

    `preflight` defaults to `default_preflight` (this module's own PURE
    wrapper over `v2_geometry_feasibility`, above) but is an INJECTED
    dependency (design Sec 6's "preflight injection for tests"): a test may
    supply a fake `measurements -> <object with .feasible/.binding_
    constraint>` callable, so a small fabricated `ReservoirMeasurements` (far
    too small to ever clear the real 240-row/4-phase geometric quotas) can
    still exercise the OK/GATE_FAIL branches directly -- only a test of the
    REAL default needs a genuinely large synthetic reservoir. `.feasible` is
    read as a direct attribute (the ONE required signal every preflight
    result must carry -- a fake omitting it should raise `AttributeError`
    loudly, not silently default); `.binding_constraint` is read via
    `getattr(..., None)` since it is only ever CONSULTED when infeasible (a
    feasible=True fake need not bother setting it). Both are part of the
    SAME documented duck-typed contract (design Sec 6) -- the asymmetry is
    each attribute's own optionality, not an inconsistency.

    PURE: reads only `protocol`, `measurements`, and whatever `preflight`
    itself reads (the default reads only `measurements`) -- no filesystem
    I/O of its own (`check_summary_binding`'s lazy `eval_runner`/
    `eval_summary` import is a CODE import, not a filesystem read of
    reservoir data; `measure_reservoir`, B3, remains the ONE filesystem-I/O
    function in the whole qualification pipeline).

    Every branch's `report` records that stage's own outcome
    (`ConformanceResult` as `{"ok", "reason"}`) plus
    `reason_histogram(measurements.jsonl_rows)` (B5) -- computed
    unconditionally, even on an early MISMATCH, since it is a cheap, pure
    fact about the JSONL alone (design Sec 4.1: "qualify computes the full
    termination-reason histogram... into the report"). `summary_binding`/
    `preflight` are `None` in the report when that stage was never reached
    OR when `preflight` was reached but raised (its own result was never
    obtained). The `report` dict is built ONCE, incrementally updated as
    each stage is reached, rather than re-literalled at every return branch.
    """
    conformance = check_protocol_conformance(protocol, measurements)
    histogram = reason_histogram(measurements.jsonl_rows)
    report: Dict[str, Any] = {
        "conformance": dataclasses.asdict(conformance),
        "summary_binding": None,
        "preflight": None,
        "reason_histogram": histogram,
    }

    if not conformance.ok:
        return QualifyResult(
            status=QualifyStatus.MISMATCH, reason=conformance.reason, report=report)

    binding = check_summary_binding(protocol, measurements)
    report["summary_binding"] = dataclasses.asdict(binding)
    if not binding.ok:
        return QualifyResult(
            status=QualifyStatus.MISMATCH, reason=binding.reason, report=report)

    try:
        preflight_result = preflight(measurements)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        reason = (f"preflight: raised {type(exc).__name__}: {exc} -- treated "
                  f"as a corrupt/incomplete input (MISMATCH, not a gate "
                  f"failure)")
        return QualifyResult(status=QualifyStatus.MISMATCH, reason=reason, report=report)

    preflight_report = {
        "feasible": bool(preflight_result.feasible),
        "binding_constraint": getattr(preflight_result, "binding_constraint", None),
    }
    report["preflight"] = preflight_report
    if preflight_result.feasible:
        return QualifyResult(status=QualifyStatus.OK, reason=None, report=report)
    return QualifyResult(
        status=QualifyStatus.GATE_FAIL,
        reason=preflight_report["binding_constraint"],
        report=report)


# ---------------------------------------------------------------------------
# Config derivation -- Task B7 (design Sec 2.2, the derivability invariant
# "config = derive(protocol, reservoir)", Sec 2). PURE: reads only the
# already in-memory `protocol` dict, the already-loaded `measurements` (B3),
# and the caller-supplied `protocol_path` STRING -- no filesystem I/O, no
# hash not already computed by `measure_reservoir` or `canonical_json_bytes`
# itself. `measure_reservoir` (B3) remains the ONE filesystem-I/O function
# in the whole qualification pipeline; this function never touches it.
# ---------------------------------------------------------------------------

def derive_config(
        protocol: Mapping[str, Any],
        measurements: ReservoirMeasurements,
        *,
        protocol_path: Union[str, Path]) -> Dict[str, Any]:
    """The canonical `fpu_dev_corpus_v2_config.json` (design Sec 2.2) -- a
    PURE deterministic function of `(protocol, measurements, protocol_path)`
    (design Sec 2: "config = derive(protocol, reservoir)"). Every value is
    either carried from `protocol` (verbatim, or renamed per amendment 1),
    the caller-supplied `protocol_path` itself, or a MEASURED identity read
    from `measurements` (B3) -- never a fresh filesystem read or hash of its
    own. `measure_reservoir` (B3) remains the ONE filesystem-I/O function in
    the whole qualification pipeline; calling `derive_config` twice on the
    SAME `(protocol, measurements, protocol_path)` always returns an EQUAL
    dict (and, once passed through `canonical_json_bytes`, IDENTICAL bytes)
    -- this is the re-derivability the Sec 5 pre-screen tamper check depends
    on ("re-derive the canonical config from the pinned protocol + reservoir
    and byte-compare it against the supplied config").

    `protocol_path` is a REQUIRED keyword-only parameter, not something this
    function derives or reads: the config's own `protocol_path` field must
    record WHERE the frozen `reservoir_protocol.json` this config was
    derived from actually lives, and there is no way to recover a file path
    from the in-memory `protocol` dict alone (the dict has no such key --
    `PROTOCOL_SCHEMA_KEYS` records `config_out`/`report_out`/etc, paths the
    protocol itself POINTS AT, never the protocol's OWN path). Threading it
    in as a parameter -- rather than, say, reading it back off disk inside
    this function -- is what keeps `derive_config` pure: `run_qualify` (the
    I/O-owning caller, below) is the ONLY place that knows the path it just
    read `protocol` from, and passes it straight through.

    `expected_fingerprints["protocol_sha1"]` is computed from the IN-MEMORY
    `protocol` dict via `hashlib.sha1(canonical_json_bytes(protocol))` --
    NOT by hashing the file at `protocol_path` (that would be a SECOND,
    redundant filesystem read this function has no business performing). By
    construction this equals `fpu_provenance.file_sha1(protocol_path)` for
    any protocol file that was itself written via `emit_protocol`/
    `write_atomic` (B1): that emitter writes EXACTLY `canonical_json_bytes
    (protocol)`'s bytes to disk, so a whole-file SHA1 of those bytes is the
    same computation over the same bytes, just reached by reading a file
    instead of re-serializing the dict -- pinned as its own test
    (`test_derive_config_protocol_sha1_equals_file_sha1_of_a_canonically_
    emitted_protocol_file`), not merely asserted in prose.

    Returns a dict with EXACTLY the complete spec Sec 2.2 field set (no more,
    no less): the thirteen decisions carried from the protocol (`source_
    index_path`, `seed_range`, `selection_seed`, `phase_allocation`,
    `late_floors`, `enumerator_params`, `new_collapse_stratum`, `checkpoint`
    = the anchor path, `forbidden_manifests`, `screen_out`, `select_out`,
    and the two RENAMED throughput knobs `eval_batch_size`/`stall_flush_sims`
    <- `protocol["mcts_eval_batch_size"]`/`["mcts_stall_flush_sims"]`), the
    five new top-level paths (`config_schema_version`, `protocol_path`,
    `match_summary_path`, `replay_dir`, `report_out`), and the nested
    `expected_fingerprints` block of nine measured identities -- nineteen
    top-level keys in all. The field set lives HERE, as this literal dict
    (heterogeneous by nature: some values carried from the protocol, some
    computed, some read from `measurements` -- so no key-only tuple could
    build them); it is GUARDED by the test file's own independent key-set
    literals (`_DERIVED_CONFIG_TOP_LEVEL_KEYS` / `_EXPECTED_FINGERPRINTS_
    KEYS` in tests/test_fpu_dev_reservoir_protocol.py, spelled out from the
    spec rather than imported from here -- the correct anti-tautology guard),
    plus a full golden-dict equality test. Task B8 defines its OWN required-
    key set on the `fpu_dev_corpus_v2` side (`_V2_CONFIG_REQUIRED_KEYS` /
    `V2Config`, which cannot top-level-import from this module without
    recreating the Sec 6 import cycle), hard-matched against this same
    nineteen-key set for the current `config_schema_version`. Every value is
    already JSON-shaped (lists, not tuples; plain dicts) so the returned dict
    is already exactly what `canonical_json_bytes` will serialize -- no
    container-type surprises at the caller's `canonical_json_bytes(derive_
    config(...))` call site.
    """
    anchor_role = protocol["anchor"]
    base_seed = protocol["base_seed"]
    games = protocol["games"]
    protocol_sha1 = hashlib.sha1(canonical_json_bytes(protocol)).hexdigest()
    checkpoint_identities = measurements.checkpoint_identities

    return {
        # Carried from the protocol, verbatim (Sec 2.2 "Carried from the
        # protocol").
        "source_index_path": protocol["source_index_path"],
        "seed_range": [base_seed, base_seed + games],   # half-open [a, b)
        "selection_seed": protocol["selection_seed"],
        "phase_allocation": protocol["phase_allocation"],
        "late_floors": protocol["late_floors"],
        "enumerator_params": protocol["enumerator_params"],
        "new_collapse_stratum": protocol["new_collapse_stratum"],
        "checkpoint": protocol[anchor_role]["path"],   # the ANCHOR, singular
        "forbidden_manifests": list(protocol["forbidden_manifests"]),
        "screen_out": protocol["screen_out"],
        "select_out": protocol["select_out"],
        # Carried, RENAMED (amendment 1) -- the screen-anchor MCTS
        # throughput knobs.
        "eval_batch_size": protocol["mcts_eval_batch_size"],
        "stall_flush_sims": protocol["mcts_stall_flush_sims"],
        # New top-level paths (amendments 1, 2).
        "config_schema_version": protocol["config_schema_version"],
        "protocol_path": str(protocol_path),
        "match_summary_path": protocol["match_summary_path"],
        "replay_dir": protocol["replay_dir"],
        "report_out": protocol["report_out"],
        # Measured identities (Sec 2.2 "expected_fingerprints (extended)").
        "expected_fingerprints": {
            "protocol_sha1": protocol_sha1,
            "source_index_sha1": measurements.source_index_sha1,
            "replay_data_sha1": measurements.replay_data_sha1,
            "match_summary_sha1": measurements.match_summary_sha1,
            "source_file_sha1s": dict(measurements.source_file_sha1s),
            "forbidden_manifest_sha1s": dict(measurements.forbidden_manifest_sha1s),
            "reservoir_checkpoint_a_identity": checkpoint_identities["reservoir_a"],
            "reservoir_checkpoint_b_identity": checkpoint_identities["reservoir_b"],
            "anchor_checkpoint_identity": checkpoint_identities["anchor"],
        },
    }


# ---------------------------------------------------------------------------
# Report state machine -- Task B7 (design Sec 3). `write_report` persists a
# `QualifyResult` as canonical JSON, tagged with an explicit `"status"`
# marker; `is_passed`/`is_retired` classify an EXISTING report file by that
# marker alone (no re-qualification, no filesystem access beyond the report
# file itself).
#
# The three `QualifyStatus` outcomes map onto exactly two write policies
# (design Sec 3):
#   - MISMATCH is REPLACEABLE: the next `qualify` attempt (after complete
#     regeneration under the SAME protocol) may overwrite it -- a mismatch
#     never burns a protocol version.
#   - OK (PASS) and GATE_FAIL are each TERMINAL and IMMUTABLE:
#     `write_report` raises rather than overwrite either -- a passed
#     protocol is thereafter reviewed with `--check` only (never
#     re-qualified, `run_qualify` below); a gate-failed protocol's version
#     is permanently RETIRED (`run_qualify` refuses to run again at all).
# ---------------------------------------------------------------------------

# The two TERMINAL status values (design Sec 3) -- `write_report` refuses to
# overwrite a report already carrying either one. `QualifyStatus.MISMATCH`
# is deliberately absent from this tuple: it is the one REPLACEABLE status.
_TERMINAL_REPORT_STATUSES: Tuple[str, str] = (
    QualifyStatus.OK.value, QualifyStatus.GATE_FAIL.value)


def _read_report_status(path: Union[str, Path]) -> Optional[str]:
    """The persisted `"status"` marker of the report at `path`, or `None`
    when no report exists there yet (the ordinary, expected state for a
    protocol that has never been qualified). Reads and parses the file when
    present -- a PRESENT-but-corrupt report (unparseable JSON, or JSON
    missing the `"status"` key) raises rather than silently returning
    `None`: `is_retired` guards `run_qualify` against re-running a
    permanently-blocked protocol, so silently treating a corrupt retirement
    record as "not retired" would defeat the one property that guard exists
    to enforce. The single implementation `is_passed`/`is_retired`/
    `write_report` all share, so the report's on-disk shape (the `"status"`
    key `_qualify_result_document` writes) has exactly one reader.
    """
    target = Path(path)
    if not target.is_file():
        return None
    document = json.loads(target.read_text())
    return document["status"]


def is_passed(report_path: Union[str, Path]) -> bool:
    """True iff the report at `report_path` records a PASS (`QualifyStatus.
    OK`) -- design Sec 3: "A passed protocol is thereafter reviewed with
    `--check`, never re-qualified." `False` both when no report exists yet
    and when it records MISMATCH or GATE_FAIL -- only an actual PASS
    counts."""
    return _read_report_status(report_path) == QualifyStatus.OK.value


def is_retired(report_path: Union[str, Path]) -> bool:
    """True iff the report at `report_path` records a GATE_FAIL (design
    Sec 3: "records the protocol version as RETIRED and permanently
    prevents config emission for that protocol"). `False` both when no
    report exists yet and when it records MISMATCH or PASS."""
    return _read_report_status(report_path) == QualifyStatus.GATE_FAIL.value


def _qualify_result_document(qualify_result: QualifyResult) -> Dict[str, Any]:
    """`qualify_result` as a plain, JSON-shaped dict -- the exact document
    `write_report` persists. `status` is the Enum's own `.value` string
    (`"OK"`/`"MISMATCH"`/`"GATE_FAIL"`) -- the explicit marker `is_passed`/
    `is_retired` classify a report by; `reason`/`report` are carried through
    verbatim (already plain JSON-shaped values -- see `QualifyResult`/
    `qualify_core`'s own docstrings, B6)."""
    return {
        "status": qualify_result.status.value,
        "reason": qualify_result.reason,
        "report": qualify_result.report,
    }


def write_report(path: Union[str, Path], qualify_result: QualifyResult) -> None:
    """Persist `qualify_result` as canonical JSON at `path` (design Sec 3),
    implementing the report state machine's write policy:

    - **Absent**, or an existing report whose `"status"` is `MISMATCH` --
      writes/REPLACES freely. A MISMATCH report is explicitly the
      REPLACEABLE state (design Sec 3): the next `qualify` attempt's content
      may legitimately differ from the old one (e.g. a different failing
      check after the operator regenerates the reservoir), so this does NOT
      route through `write_atomic`'s own refuse-overwrite-DIFFERENT guard
      (that guard exists for IMMUTABLE artifacts; a MISMATCH report is the
      opposite). The stale file is removed first, so the ONE reused
      `write_atomic` primitive performs the actual write via its own
      "absent" branch, rather than a second, parallel implementation of its
      temp+rename mechanics existing just for this case.
    - **An existing PASS or GATE_FAIL report** -- raises `ValueError` and
      leaves the existing report file completely untouched (checked, and
      refused, BEFORE any write is attempted -- the stale-MISMATCH removal
      above never runs in this branch). Both are TERMINAL (design Sec 3): a
      PASS is reviewed with `--check`, never re-qualified; a GATE_FAIL
      permanently retires the protocol version. Mirrors `write_atomic`'s own
      "refuses to overwrite ... raises `ValueError`" idiom exactly, one
      level up: the object being protected here is the REPORT's recorded
      OUTCOME, not (only) its bytes.
    """
    target = Path(path)
    existing_status = _read_report_status(target)
    # DELIBERATELY stricter than the base Sec 3 prose's "idempotent-on-byte-
    # identical": a terminal report is refused UNCONDITIONALLY, even when the
    # incoming bytes would be identical. Correction 3 (the PASS-terminal
    # amendment) supersedes that base prose -- a passed protocol is
    # "reviewed with --check, NEVER re-qualified", so re-reaching this write
    # for a terminal report is itself the error, regardless of byte-equality.
    # In practice `run_qualify`'s `is_retired`/`is_passed` guards short-
    # circuit before ever re-entering here for a terminal report, so this
    # raise is a defensive backstop, not a normal path. Do NOT "fix" this
    # toward literal byte-identical idempotency -- that would silently undo
    # correction 3.
    if existing_status in _TERMINAL_REPORT_STATUSES:
        raise ValueError(
            f"write_report: refusing to overwrite {target} -- an existing "
            f"report already records {existing_status!r}, a TERMINAL and "
            f"immutable outcome (a PASS is reviewed with --check, never "
            f"re-qualified; a GATE_FAIL permanently retires the protocol "
            f"version)")
    if existing_status is not None:
        # The only remaining classified value is QualifyStatus.MISMATCH --
        # replaceable. Remove the stale file so `write_atomic` below takes
        # its own "absent" branch (see this function's own docstring for why
        # this is not routed through write_atomic's refuse-overwrite-
        # different guard).
        target.unlink()
    data = canonical_json_bytes(_qualify_result_document(qualify_result))
    write_atomic(target, data)


# ---------------------------------------------------------------------------
# run_qualify -- Task B7 (design Sec 3/Sec 7). The OPERATOR entry point that
# OWNS the I/O this whole pipeline was built to keep out of every earlier,
# pure stage: it loads the frozen protocol, calls `measure_reservoir` (B3 --
# the ONE filesystem-I/O function that reads a GENERATED reservoir), runs
# the pure `qualify_core` (B6), and -- unless `check=True` -- emits the
# config + report per the Sec 3 state machine. Never runs the evaluator/
# MCTS/generation itself (design's TOOLING ONLY constraint) -- everything it
# calls (`measure_reservoir`, `qualify_core`, `derive_config`,
# `write_atomic`, `write_report`) is either pure or restricted to reading/
# hashing already-generated files.
# ---------------------------------------------------------------------------

# `QualifyStatus` -> the design Sec 3 exit-code vocabulary, for the three
# outcomes `qualify_core` can return. The fourth vocabulary member,
# `EXIT_USAGE`, belongs to `run_qualify`'s OWN pre-measurement refusals
# (already-PASSED without `--check`) -- never a `qualify_core` outcome, so
# it is deliberately not a value in this mapping.
_EXIT_CODE_FOR_STATUS: Dict[QualifyStatus, int] = {
    QualifyStatus.OK: EXIT_OK,
    QualifyStatus.MISMATCH: EXIT_MISMATCH,
    QualifyStatus.GATE_FAIL: EXIT_GATE_FAIL,
}


def run_qualify(
        protocol_path: Union[str, Path],
        *,
        check: bool = False,
        preflight: Callable[[ReservoirMeasurements], Any] = default_preflight,
) -> int:
    """Qualify the reservoir declared by the frozen protocol at
    `protocol_path` (design Sec 3/Sec 7) and, on success, emit the immutable
    `fpu_dev_corpus_v2_config.json`. Returns the design's Sec 3 exit-code
    vocabulary (`EXIT_OK`/`EXIT_MISMATCH`/`EXIT_GATE_FAIL`, plus
    `EXIT_USAGE` for the PASS-terminal refusal below) -- never raises for an
    ordinary qualification outcome, only for a malformed `protocol_path` /
    protocol document (propagated from the plain `json.loads`/dict-lookup
    calls below, exactly like `measure_reservoir`/`gen_command`'s own
    "assumes already-valid input" contract) or a write refused by
    `write_atomic`/`write_report` (which should never trigger in the normal
    flow below -- each is only ever reached once per report/config path per
    the state-machine checks that precede it).

    `preflight` is an INJECTED dependency, defaulting to the real
    `default_preflight` -- forwarded straight through to `qualify_core`
    (B6). This mirrors the same "preflight injection for tests" principle
    design Sec 6 establishes for `qualify_core` itself: the real geometric
    feasibility only turns OK on a genuinely large reservoir (>= ~120
    protocol-conformant games, the same boundary `qualify_core`'s own
    real-preflight tests exercise) -- far too heavy to fabricate ON DISK
    (unlike `qualify_core`'s own tests, this function's tests cannot
    fabricate a `ReservoirMeasurements` directly; `run_qualify` calls the
    real `measure_reservoir`, so they must write real files under
    `tmp_path`) for every test of THIS function's own, DIFFERENT concern:
    the report state machine and config emission, not geometric feasibility
    (already thoroughly covered where it belongs, B6). `measure_reservoir`
    itself is deliberately NOT injectable here -- it is, by design, the ONE
    filesystem-I/O function in the whole qualification pipeline, and this
    function's whole job is to OWN that one real call, against a real (if
    small, in tests) on-disk reservoir.

    The Sec 3 state machine, in the order this function checks it:

    1. **Already RETIRED** (`is_retired(protocol["report_out"])`) --
       refuses UNCONDITIONALLY, even under `--check`: returns
       `EXIT_GATE_FAIL` immediately, calling `measure_reservoir` NOT AT ALL
       (design Sec 3: "GATE-FAIL ... qualify refuses to run again against a
       protocol whose report_out carries a retirement record" -- no
       carve-out for review). No config is ever written in this branch.
    2. **Already PASSED** (`is_passed(...)`) **and `check=False`** --
       refuses re-qualification: returns `EXIT_USAGE`, again calling
       `measure_reservoir` not at all. This is the PASS-terminal rule
       (design Sec 3: "A passed protocol is thereafter reviewed with
       `--check`, never re-qualified") -- `EXIT_USAGE` (not
       `EXIT_OK`/`EXIT_MISMATCH`/`EXIT_GATE_FAIL`) because this refusal is
       not a finding about the CURRENT reservoir at all (it is never even
       measured); it is a usage error against the ALREADY-COMPLETE prior
       qualification, matching the design's own "usage/IO" bucket for exit
       code 2. **Already PASSED with `check=True`** falls through to step 3
       instead -- exactly the "reviewed with `--check`" path.
    3. Otherwise (never qualified, a replaceable MISMATCH, or a PASS being
       reviewed under `--check`): calls `measure_reservoir(protocol)` (the
       ONE real I/O) then `qualify_core(protocol, measurements,
       preflight=preflight)`.
       - **`check=True`** -- returns the status's mapped exit code and
         writes NOTHING (design Sec 8: "`--check` recomputes and diffs but
         never writes"), regardless of which status was computed.
       - **`check=False`, status OK** -- `derive_config(protocol,
         measurements, protocol_path=str(protocol_path))`, emits the config
         atomically to `protocol["config_out"]`, then `write_report`s a PASS
         record to `protocol["report_out"]`; returns `EXIT_OK`.
       - **`check=False`, status GATE_FAIL** -- `write_report`s the
         retirement record (no config is ever derived or written); returns
         `EXIT_GATE_FAIL`.
       - **`check=False`, status MISMATCH** -- `write_report`s the
         (replaceable) mismatch record; returns `EXIT_MISMATCH`.
    """
    protocol = json.loads(Path(protocol_path).read_text())
    report_out = protocol["report_out"]

    if is_retired(report_out):
        return EXIT_GATE_FAIL
    if is_passed(report_out) and not check:
        return EXIT_USAGE

    measurements = measure_reservoir(protocol)
    result = qualify_core(protocol, measurements, preflight=preflight)

    if check:
        return _EXIT_CODE_FOR_STATUS[result.status]

    if result.status == QualifyStatus.OK:
        config = derive_config(
            protocol, measurements, protocol_path=str(protocol_path))
        write_atomic(protocol["config_out"], canonical_json_bytes(config))
        write_report(report_out, result)
        return EXIT_OK

    write_report(report_out, result)
    return _EXIT_CODE_FOR_STATUS[result.status]


# ---------------------------------------------------------------------------
# precheck_before_screen -- Task B9 (design Sec 5/Sec 6). The `run_screen`
# pre-evaluator gate: called BEFORE any checkpoint/evaluator work, so an
# hours-long screen never starts against a stale or tampered
# config/protocol/reservoir. `run_screen` (fpu_dev_corpus_v2.py) LAZILY
# imports this function inside its own body -- never at that module's top
# level -- which is what keeps the Sec 6 circular-import risk
# one-directional: this module already top-level-imports FROM
# fpu_dev_corpus_v2 (`_V2_CONFIG_REQUIRED_KEYS`, `_V2_CORPUS_SOURCES`,
# `enumerate_v2_proposals`, `v2_geometry_feasibility`), so a top-level import
# back from that module into this one would cycle.
# ---------------------------------------------------------------------------

def _fingerprint_mismatch(key: str, supplied: Any, fresh: Any, *,
                          protocol_path: Union[str, Path]) -> ValueError:
    """One `expected_fingerprints[key]` hard-match failure (Task B9 step 2),
    formatted like the rest of this module's raises: names the identity,
    what each side says, and what to do. When both sides are `{basename:
    sha1}` blocks (`source_file_sha1s`/`forbidden_manifest_sha1s`), narrows
    the report to just the differing basename(s) -- dumping the whole block
    twice over would bury the one fact an operator needs (WHICH file
    changed), mirroring `fpu_dev_corpus_v2._identity_mismatch`'s own
    narrowing for the same reason.

    Deliberately a DISTINCT helper, not a shared import, even though (Task
    B10) `SCREEN_IDENTITY_KEYS` (eleven identities) is now a strict superset
    of this module's `expected_fingerprints` (nine): the two call sites need
    DIFFERENT remediation advice, not just different formatting. This
    function fires BEFORE a screen exists at all (`precheck_before_screen`,
    a 2-source config-vs-fresh-recompute check), where the one correct fix is
    always "re-qualify"; `fpu_dev_corpus_v2._identity_mismatch` fires AFTER
    an (hours-long, expensive) screen exists (a 2-or-3-source A/B/C
    hard-match), where "re-qualify" is not even meaningful (qualify never
    reads a screen) and the honest per-identity advice varies ("restore the
    file", "re-screen", "point --config at the right file", ...). Reusing
    `fpu_dev_corpus_v2._IDENTITY_REMEDIATION`'s text here would therefore
    give WRONG advice at this call site, not merely duplicate it -- so B10
    leaves this generic message as is (considered per the B10 brief's
    unify-or-explain-why note) rather than importing that table."""
    if isinstance(supplied, Mapping) and isinstance(fresh, Mapping):
        names = sorted(set(supplied) | set(fresh))
        differing = [n for n in names if supplied.get(n) != fresh.get(n)]
        detail = "differing basename(s): " + ", ".join(
            f"{n}: {supplied.get(n)!r} -> {fresh.get(n)!r}" for n in differing)
    else:
        detail = f"config declares {supplied!r}, fresh recompute is {fresh!r}"
    return ValueError(
        f"precheck_before_screen: expected_fingerprints[{key!r}] MISMATCH -- "
        f"{detail}. The reservoir/protocol/summary/source-file/checkpoint "
        f"this identity covers changed since the config at {protocol_path!r} "
        f"was derived -- re-qualify (fpu_dev_reservoir_protocol qualify) "
        f"before screening.")


# ---------------------------------------------------------------------------
# The shared design-Sec-5 "re-derive + byte-compare" (Task B10 correction).
# Spec Sec 5 requires the config to be re-derived-and-byte-compared at BOTH
# check points -- the pre-GPU `precheck_before_screen` (below) AND, at select,
# `fpu_dev_corpus_v2.select_final_manifest` ("reservoir/config identities are
# checked TWICE, pre-GPU and at select"). These helpers are the ONE
# implementation both call sites reuse (select via a lazy import -- Sec 6:
# fpu_dev_corpus_v2 must not top-level-import this module), so the tamper check
# can never drift between the two stages.
#
# Split into a FRONT half (`measure_and_rederive_config`) and a BACK half
# (`assert_config_byte_equals_rederivation`) because `precheck_before_screen`
# interleaves its per-identity `expected_fingerprints` hash recheck BETWEEN
# them (so a byte-changed reservoir/checkpoint/summary fails with a SHARP
# per-identity message -- "replay_data_sha1", not the whole-document diff);
# `select` needs no such interleave, so it uses the `rederive_and_assert_
# config_unchanged` convenience that runs both halves back to back.
# ---------------------------------------------------------------------------

def measure_and_rederive_config(
        config: Any,
        *,
        measure: Callable[[Mapping[str, Any]], ReservoirMeasurements] = measure_reservoir,
) -> Tuple[ReservoirMeasurements, Dict[str, Any]]:
    """Load `config.protocol_path`, `measure` the reservoir, and re-derive the
    canonical config from `(protocol, measurements)` via `derive_config`.
    Returns `(measurements, recomputed_config)` -- the FRONT half of the Sec 5
    re-derive-and-byte-compare, factored out so a caller that ALSO wants the
    per-identity hash recheck / geometric preflight (`precheck_before_screen`)
    can reuse the SAME `measurements`/`recomputed_config` without a second real
    `measure`. `measure` is injectable (default the real `measure_reservoir`,
    the ONE filesystem-I/O function) exactly as `precheck_before_screen`'s is,
    so a test can drive the check over a fabricated reservoir. Pure apart from
    `measure`'s one real read; no evaluator/MCTS/GPU."""
    protocol = json.loads(Path(config.protocol_path).read_text())
    measurements = measure(protocol)
    recomputed_config = derive_config(
        protocol, measurements, protocol_path=config.protocol_path)
    return measurements, recomputed_config


def assert_config_byte_equals_rederivation(
        config: Any, recomputed_config: Mapping[str, Any]) -> None:
    """THE Sec 5 config-tamper check (the BACK half): byte-compare the supplied
    `config` against an already-computed fresh `recomputed_config`
    (`canonical_json_bytes` both), raising `ValueError` -- naming the differing
    top-level key(s) -- on any diff. This is what catches an edited config field
    that carries NO hash of its own (`selection_seed`, `select_out`, a floor,
    ...), which a per-identity `expected_fingerprints` hash recheck structurally
    cannot see. Pure: no I/O (the caller supplies `recomputed_config`, e.g. from
    `measure_and_rederive_config`). `config` is duck-typed (reads only the
    `_V2_CONFIG_REQUIRED_KEYS` attrs + `eval_batch_size`/`stall_flush_sims`)."""
    supplied_config = {key: getattr(config, key) for key in _V2_CONFIG_REQUIRED_KEYS}
    supplied_config["eval_batch_size"] = config.eval_batch_size
    supplied_config["stall_flush_sims"] = config.stall_flush_sims
    if canonical_json_bytes(supplied_config) != canonical_json_bytes(recomputed_config):
        # Diff PER KEY on canonical JSON -- the SAME type-insensitive normalization
        # the pass/fail decision above uses -- NOT raw `!=`. `V2Config` stores
        # `seed_range`/`forbidden_manifests` as TUPLES while `derive_config` emits
        # them as JSON LISTS, so `tuple != list` is always True even when the
        # CONTENTS match; a raw-`!=` diff would name those two keys in EVERY raise
        # regardless of the real tamper (a false positive on the operator-facing
        # forgery-investigation path). Canonicalizing each side first makes
        # `differing` name only the genuinely-changed key(s).
        differing = sorted(
            k for k in recomputed_config
            if canonical_json_bytes(supplied_config.get(k))
            != canonical_json_bytes(recomputed_config[k]))
        raise ValueError(
            "the supplied config does not byte-equal a fresh re-derivation from "
            f"(protocol, reservoir) -- differing top-level key(s): {differing}. "
            "Some config field was edited after qualification (design Sec 5's "
            "re-derive + byte-compare tamper check -- it catches ANY edited "
            "field, hashed or not, e.g. selection_seed, select_out, that a "
            "per-identity expected_fingerprints hash recheck cannot see).")


def rederive_and_assert_config_unchanged(
        config: Any,
        *,
        measure: Callable[[Mapping[str, Any]], ReservoirMeasurements] = measure_reservoir,
) -> None:
    """The WHOLE Sec 5 re-derive-and-byte-compare as ONE call: `measure_and_
    rederive_config` then `assert_config_byte_equals_rederivation`. This is the
    entry point `fpu_dev_corpus_v2.select_final_manifest` lazily imports and
    runs at select time -- it needs exactly this and nothing more (its identity
    hard-match already covers what `precheck_before_screen`'s step-2 recheck
    does, and Sec 5 does not require the geometric preflight at select).
    `precheck_before_screen` does NOT use this convenience -- it must interleave
    its per-identity recheck between the two halves (see this section's banner),
    so it calls the two underlying helpers directly. Raises `ValueError` on any
    config tamper; returns `None` on success."""
    _measurements, recomputed_config = measure_and_rederive_config(
        config, measure=measure)
    assert_config_byte_equals_rederivation(config, recomputed_config)


def precheck_before_screen(
        config: Any,
        *,
        measure: Callable[[Mapping[str, Any]], ReservoirMeasurements] = measure_reservoir,
        preflight: Callable[[ReservoirMeasurements], Any] = default_preflight,
) -> None:
    """`run_screen`'s pre-evaluator gate (design Sec 5/Sec 6, Task B9).
    Raises `ValueError` on ANY failure; returns `None` on success. `config`
    is duck-typed -- a real `V2Config`, or anything exposing the same
    attributes (`protocol_path`, `expected_fingerprints`, and every
    `_V2_CONFIG_REQUIRED_KEYS` field) -- this module never imports
    `V2Config` itself (see this section's banner).

    Five checks, run in this order:

    1. Loads the frozen protocol from `config.protocol_path` and calls
       `measure(protocol)` (default `measure_reservoir`, B3) -- the ONE
       real-world filesystem read this function performs, through its
       injected dependency.
    2. Recomputes `derive_config`'s (B7) `expected_fingerprints` block from
       the FRESH `(protocol, measurements)` and hard-matches it, identity by
       identity, against `config.expected_fingerprints` -- catches a
       reservoir, protocol, match-summary, result-determining source file,
       or checkpoint that changed on disk since this config was derived
       (`_fingerprint_mismatch` names which one).
    3. Re-derives the WHOLE canonical config (`derive_config(protocol,
       measurements, protocol_path=config.protocol_path)`) and
       byte-compares it (`canonical_json_bytes`) against the supplied
       `config` -- THE real config-tamper check (design Sec 5). Unlike step
       2, which can only ever see the nine HASHED identities, this compares
       the entire derived document, so it catches an edit to ANY field --
       hashed or not: `selection_seed`, `select_out`, `phase_allocation`,
       .... Step 2 could never see a `selection_seed` edit (it carries no
       hash of its own); this step is what makes the tamper check complete.
    4. Explicitly re-confirms the config binds THIS protocol
       (`expected_fingerprints["protocol_sha1"]` against the freshly-loaded
       protocol's own hash). Already implied by steps 2 and 3
       (`protocol_sha1` is one of the nine fingerprints step 2 checks, and
       is part of the whole-config comparison step 3 makes) -- kept as its
       own explicit check so a config that binds a completely different
       protocol fails with a message naming exactly that, rather than a
       generic fingerprint diff.
    5. Repeats the geometric preflight DEFENSIVELY (`preflight(
       measurements)`, default `default_preflight`, B6) -- the same
       feasibility gate `qualify_core` ran when this config was first
       emitted, re-run here in case the reservoir's *content* regressed
       into infeasibility in a way steps 2-4 cannot see. Infeasible ->
       raise, stopping before the evaluator loads.

    PURE apart from `measure`'s one real read: no evaluator/MCTS/GPU of its
    own. Deliberately does NOT call `check_protocol_conformance` or
    `check_summary_binding` (B4/B5) a second time -- those already ran once,
    at `qualify` time, verifying the RESERVOIR against the PROTOCOL; this
    function verifies the CONFIG against the (protocol, reservoir) pair --
    the one thing `qualify_core` never checked, since it takes no `config`
    parameter at all.
    """
    # (1) Load the protocol, measure the reservoir, re-derive the canonical
    # config (the SHARED front half -- `select_final_manifest` runs the exact
    # same re-derivation at select time via `rederive_and_assert_config_
    # unchanged`, design Sec 5's "checked twice").
    measurements, recomputed_config = measure_and_rederive_config(
        config, measure=measure)
    recomputed_fingerprints = recomputed_config["expected_fingerprints"]
    expected_fingerprints = config.expected_fingerprints or {}

    # (2) Hash recheck: every pinned identity, hard-matched individually so
    # a failure names exactly which one drifted. Deliberately BEFORE the
    # whole-config byte-compare (3) so a byte-changed reservoir/checkpoint/
    # summary fails with the SHARP per-identity message ("replay_data_sha1"),
    # not the coarser whole-document "expected_fingerprints differs" diff (3)
    # would report -- this ordering is why precheck calls the two shared
    # halves directly rather than the `rederive_and_assert_config_unchanged`
    # convenience.
    for key in sorted(recomputed_fingerprints):
        supplied_value = expected_fingerprints.get(key)
        fresh_value = recomputed_fingerprints[key]
        if supplied_value != fresh_value:
            raise _fingerprint_mismatch(
                key, supplied_value, fresh_value,
                protocol_path=config.protocol_path)

    # (3) The real config-tamper check (the SHARED back half): byte-compare the
    # WHOLE canonical config -- catches an edited field that carries no hash of
    # its own (selection_seed, select_out, ...).
    assert_config_byte_equals_rederivation(config, recomputed_config)

    # (4) The config must bind THIS protocol -- already implied by (2)/(3),
    # re-checked explicitly for a message that names the failure directly.
    supplied_protocol_sha1 = expected_fingerprints.get("protocol_sha1")
    measured_protocol_sha1 = recomputed_fingerprints["protocol_sha1"]
    if supplied_protocol_sha1 != measured_protocol_sha1:
        raise ValueError(
            f"precheck_before_screen: config does not bind the protocol at "
            f"{config.protocol_path!r} -- expected_fingerprints["
            f"'protocol_sha1']={supplied_protocol_sha1!r} but that protocol "
            f"hashes to {measured_protocol_sha1!r}. The config was derived "
            f"from a DIFFERENT protocol (or the protocol changed since).")

    # (5) Repeat the geometric preflight, defensively.
    preflight_result = preflight(measurements)
    if not preflight_result.feasible:
        raise ValueError(
            f"precheck_before_screen: geometric preflight is INFEASIBLE on "
            f"defensive re-check -- binding constraint: "
            f"{getattr(preflight_result, 'binding_constraint', None)}. "
            f"Stopping BEFORE the evaluator loads (an hours-long screen "
            f"must never start on an infeasible reservoir).")


# ---------------------------------------------------------------------------
# CLI -- Task B11 (design Sec 3). Three argparse subcommands over the pure
# functions + `run_qualify` above: `emit-protocol`, `emit-gen-command`,
# `qualify`. Pure glue -- see this module's own docstring (the
# `_parse_args`/`main` paragraph, near the end) for the full per-subcommand
# contract. No `--mode select`/`screen` here: those stay in
# `fpu_dev_corpus_v2.main`.
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """Build the CLI's argparse parser and parse `argv` (design Sec 3).
    Mirrors `fpu_dev_corpus_v2._parse_v2_args`'s own "build + parse in one
    function" shape. A missing required flag, an unrecognized subcommand, or
    no subcommand at all each raise `SystemExit(2)` -- argparse's OWN usage
    errors, already `EXIT_USAGE`'s value (design Sec 3), so `main` needs no
    separate mapping for them.
    """
    ap = argparse.ArgumentParser(
        description="FPU v2 reservoir-protocol CLI (design Sec 3): freeze a "
                    "reservoir_protocol.json from declared params, print its "
                    "exact eval_checkpoint_match generation command, and "
                    "qualify a GENERATED reservoir against it -- zero-GPU; "
                    "qualify never launches generation. No --mode "
                    "select/screen here -- those stay in "
                    "fpu_dev_corpus_v2.main.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_emit_protocol = sub.add_parser(
        "emit-protocol",
        help="freeze a reservoir_protocol.json from a --params-json params "
             "mapping; --check never writes.")
    p_emit_protocol.add_argument(
        "--params-json", required=True,
        help="path to a JSON file holding the params mapping -- every "
             "PROTOCOL_SCHEMA_KEYS field (build_protocol's required set).")
    p_emit_protocol.add_argument(
        "--out", required=True,
        help="path to write (or, with --check, verify) the "
             "reservoir_protocol.json.")
    p_emit_protocol.add_argument(
        "--check", action="store_true",
        help="never writes: recompute the canonical protocol bytes and "
             "report whether --out already holds them (exit 0) or not "
             "(exit 3).")

    p_emit_gen_command = sub.add_parser(
        "emit-gen-command",
        help="print the exact, shell-joined eval_checkpoint_match argv "
             "derived from an already-frozen --protocol (zero-GPU).")
    p_emit_gen_command.add_argument(
        "--protocol", required=True,
        help="path to the frozen reservoir_protocol.json.")

    p_qualify = sub.add_parser(
        "qualify",
        help="qualify a GENERATED reservoir against its frozen --protocol "
             "(zero-GPU: reads + validates only, never launches "
             "generation); --check never writes.")
    p_qualify.add_argument(
        "--protocol", required=True,
        help="path to the frozen reservoir_protocol.json.")
    p_qualify.add_argument(
        "--check", action="store_true",
        help="never writes: report the qualification outcome only "
             "(reviews an already-PASSED protocol without re-qualifying).")

    return ap.parse_args(argv)


# `qualify`'s one status print's exit-code -> label lookup -- purely
# cosmetic (no test asserts on this wording, only on the returned exit code
# and on-disk state): `.get(status, status)` at the call site keeps this
# total even for a status somehow outside this table.
_EXIT_STATUS_LABELS: Dict[int, str] = {
    EXIT_OK: "OK",
    EXIT_MISMATCH: "MISMATCH",
    EXIT_GATE_FAIL: "GATE_FAIL",
    EXIT_USAGE: "USAGE (already PASSED -- re-run with --check to review)",
}


def main(argv: Optional[List[str]] = None) -> int:
    """The module's CLI entry point (design Sec 3, Task B11) -- dispatches
    to `emit_protocol`/`gen_command`/`run_qualify`, the pure/I/O-owning
    functions this whole module built, and returns the Sec 3 exit-code
    vocabulary (`EXIT_OK`/`EXIT_USAGE`/`EXIT_MISMATCH`/`EXIT_GATE_FAIL`) as
    the process exit code. See this module's own docstring (the
    `_parse_args`/`main` paragraph) for the full per-subcommand contract
    (the `--params-json` rationale, "the CLI always wires the real
    preflight", the exit-code mapping already living in `run_qualify`).

    Pure CLI glue: reads at most ONE caller-supplied JSON document per
    subcommand (`--params-json`, or `--protocol`) and makes exactly ONE call
    into `emit_protocol`/`gen_command`/`run_qualify` -- no qualification or
    derivation logic of its own. A malformed document propagates raw
    (`json.loads`/`build_protocol`'s own exceptions), exactly like every
    other "assumes already-valid input" function in this module.

    `qualify` NEVER launches generation: it forwards `--protocol`/`--check`
    straight into `run_qualify` UNCHANGED (no `preflight=` override), so the
    real `default_preflight` -- and only `measure_reservoir`'s read/hash of
    an ALREADY-GENERATED reservoir -- ever runs; no evaluator/MCTS/GPU/
    checkpoint-weights load anywhere in this call chain.
    """
    args = _parse_args(argv)

    if args.command == "emit-protocol":
        params = json.loads(Path(args.params_json).read_text())
        status = emit_protocol(params, args.out, check=args.check)
        if args.check:
            verdict = "MATCH" if status == EXIT_OK else "MISMATCH"
            print(f"[fpu-dev-reservoir-protocol] emit-protocol --check "
                  f"{args.out}: {verdict}")
        elif status == EXIT_OK:
            print(f"[fpu-dev-reservoir-protocol] emit-protocol: wrote "
                  f"{args.out}")
        return status

    if args.command == "emit-gen-command":
        protocol = json.loads(Path(args.protocol).read_text())
        print(shlex.join(gen_command(protocol)))
        return EXIT_OK

    if args.command == "qualify":
        status = run_qualify(args.protocol, check=args.check)
        print(f"[fpu-dev-reservoir-protocol] qualify "
              f"{'--check ' if args.check else ''}{args.protocol}: "
              f"{_EXIT_STATUS_LABELS.get(status, status)} (exit {status})")
        return status

    raise AssertionError(   # unreachable: argparse's own subparsers guard this
        f"main: unreachable command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
