"""Tests for `fpu_dev_reservoir_protocol.py` Task B1: the frozen
`reservoir_protocol.json` field set, the canonical-JSON encoder, the
atomic/immutable write primitive, and the `build_protocol` / `emit_protocol`
schema builder + emitter.

Frozen design ref: docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md
  Sec 2.1 (protocol schema), Sec 3 (atomicity/immutability + `--check`
  contract), Sec 8 (canonical JSON, determinism).
Pre-op hardening plan ref: docs/superpowers/plans/2026-07-14-fpu-v2-preop-hardening-plan.md
  Task B1.

Pure stdlib only -- no evaluator/MCTS/GPU/MLX/checkpoint anywhere in this
file. `test_module_import_pulls_no_gpu_or_mlx` proves the module itself is
import-clean via a subprocess (the SAME idiom
tests/test_fpu_dev_corpus_v2.py::test_v2_module_import_pulls_no_gpu_or_mlx
uses -- a plain in-process `import` check would be unreliable here because
another test module in the same pytest session may have already pulled mlx
into `sys.modules` first).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import (
    EXIT_GATE_FAIL,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_USAGE,
    PROTOCOL_SCHEMA_KEYS,
    WriteStatus,
    build_protocol,
    canonical_json_bytes,
    emit_protocol,
    write_atomic,
)

# ---------------------------------------------------------------------------
# Shared fixture: a complete, valid `params` mapping covering every
# PROTOCOL_SCHEMA_KEYS field with plausible (not necessarily production)
# values. Mirrors the `_v2_config_fixture(**overrides)` idiom the plan's
# Task B8 names for `V2Config` -- one shared helper so the schema lives in
# one place across this file's tests.
# ---------------------------------------------------------------------------
def _protocol_params(**overrides) -> dict:
    params = {
        # Identity.
        "protocol_version": 1,
        "no_top_up": True,
        "config_schema_version": 1,
        # Matchup + anchor.
        "checkpoint_a": {"path": "checkpoints/calib020_0001.safetensors",
                          "identity": "calib020_0001:aaaaaaaa"},
        "checkpoint_b": {"path": "checkpoints/0379.safetensors",
                          "identity": "0379:bbbbbbbb"},
        "anchor": "checkpoint_a",
        # Reservoir params: games + the ten knobs + save_eval_replays +
        # workers.
        "games": 6,
        "base_seed": 900000,
        "board_size": 24,
        "mcts_sims": 400,
        "mcts_eval_batch_size": 14,
        "mcts_stall_flush_sims": 48,
        "selection_mode": "visit_count",
        "opening_temp_plies": 6,
        "temp_high": 1.0,
        "temp_low": 0.1,
        "max_moves": 300,
        "save_eval_replays": True,
        "workers": 4,
        # Output relationships.
        "match_summary_path": "runs/reservoir_v2/match_summary.json",
        "source_index_path": "runs/reservoir_v2/match_summary_games.jsonl",
        "replay_dir": "runs/reservoir_v2/match_summary_replays",
        "config_out": "runs/reservoir_v2/fpu_dev_corpus_v2_config.json",
        "report_out": "runs/reservoir_v2/qualify_report.json",
        # Selection settings.
        "selection_seed": 20260714,
        "phase_allocation": {
            "target|opening": {"tuning": 30, "frozen_check": 15}},
        "late_floors": {"b300_399": 12, "b200_299": 12},
        "enumerator_params": {"min_ply_gap": 12, "max_per_game": 2},
        "new_collapse_stratum": "ply_bucket",
        "forbidden_manifests": ["manifests/v1_controls.csv"],
        "screen_out": "runs/reservoir_v2/fpu_dev_source_screen.csv",
        "select_out": "runs/reservoir_v2/fpu_dev_manifest_v2.csv",
        # Generation provenance.
        "generation_git_commit": "abc123def456",
        "generation_source_sha1s": {"eval_checkpoint_match.py": "sha1a"},
    }
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# PROTOCOL_SCHEMA_KEYS
# ---------------------------------------------------------------------------

def test_protocol_schema_keys_is_the_exact_spec_2_1_field_set():
    """Pins both the exact membership AND the grouped source order (design
    Sec 2.1's own presentation order) so an accidental duplication, typo, or
    silent drop can never pass unnoticed."""
    assert PROTOCOL_SCHEMA_KEYS == (
        "protocol_version", "no_top_up", "config_schema_version",
        "checkpoint_a", "checkpoint_b", "anchor",
        "games", "base_seed", "board_size", "mcts_sims",
        "mcts_eval_batch_size", "mcts_stall_flush_sims", "selection_mode",
        "opening_temp_plies", "temp_high", "temp_low", "max_moves",
        "save_eval_replays", "workers",
        "match_summary_path", "source_index_path", "replay_dir",
        "config_out", "report_out",
        "selection_seed", "phase_allocation", "late_floors",
        "enumerator_params", "new_collapse_stratum", "forbidden_manifests",
        "screen_out", "select_out",
        "generation_git_commit", "generation_source_sha1s",
    )


def test_protocol_schema_keys_has_no_duplicates():
    assert len(PROTOCOL_SCHEMA_KEYS) == len(set(PROTOCOL_SCHEMA_KEYS)) == 34


def test_protocol_schema_keys_fixture_covers_every_key():
    """The shared fixture itself must supply every schema field -- otherwise
    every other test in this file would be silently under-testing."""
    assert set(_protocol_params()) == set(PROTOCOL_SCHEMA_KEYS)


# ---------------------------------------------------------------------------
# canonical_json_bytes
# ---------------------------------------------------------------------------

def test_canonical_json_bytes_stable_across_top_level_key_order():
    a = {"z": 1, "a": 2, "m": 3}
    b = {"a": 2, "m": 3, "z": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_canonical_json_bytes_stable_across_nested_key_order():
    a = {"outer": {"z": 1, "a": {"y": 2, "b": 3}}}
    b = {"outer": {"a": {"b": 3, "y": 2}, "z": 1}}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_canonical_json_bytes_permutation_stability_on_a_full_protocol_dict():
    """The load-bearing case: a complete protocol-shaped dict, assembled in
    forward vs reversed schema-key order, is byte-identical -- what will
    make a later `protocol_sha1` a hash of the DATA, never of incidental
    Python dict-construction order."""
    params = _protocol_params()
    forward = {k: params[k] for k in PROTOCOL_SCHEMA_KEYS}
    backward = {k: params[k] for k in reversed(PROTOCOL_SCHEMA_KEYS)}
    assert list(forward) != list(backward)  # actually different construction order
    assert canonical_json_bytes(forward) == canonical_json_bytes(backward)


def test_canonical_json_bytes_emits_sorted_keys():
    text = canonical_json_bytes({"z": 1, "a": 2}).decode("ascii")
    assert text.index('"a"') < text.index('"z"')


def test_canonical_json_bytes_returns_bytes_with_single_trailing_newline():
    data = canonical_json_bytes({"k": 1})
    assert isinstance(data, bytes)
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")


def test_canonical_json_bytes_ensure_ascii_escapes_non_ascii():
    data = canonical_json_bytes({"name": "café"})
    assert b"\\u00e9" in data
    assert "café".encode("utf-8") not in data
    # round-trips back to the original string
    assert json.loads(data.decode("ascii"))["name"] == "café"


def test_canonical_json_bytes_rejects_non_finite_floats():
    """"Fixed numeric formatting" -- every emitted number must be valid,
    unambiguous JSON; a NaN/Infinity float would otherwise serialize as a
    non-standard JSON token (`NaN`, `Infinity`)."""
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("-inf")})


def test_canonical_json_bytes_round_trips_a_full_protocol():
    protocol = build_protocol(_protocol_params())
    data = canonical_json_bytes(protocol)
    assert json.loads(data.decode("ascii")) == protocol


# ---------------------------------------------------------------------------
# write_atomic / WriteStatus
# ---------------------------------------------------------------------------

def test_write_atomic_writes_when_absent(tmp_path):
    target = tmp_path / "out.json"
    status = write_atomic(target, b"hello\n")
    assert status is WriteStatus.WRITTEN
    assert target.read_bytes() == b"hello\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]  # no tmp litter


def test_write_atomic_accepts_str_path(tmp_path):
    target = tmp_path / "out.json"
    status = write_atomic(str(target), b"hi\n")
    assert status is WriteStatus.WRITTEN
    assert target.read_bytes() == b"hi\n"


def test_write_atomic_is_idempotent_on_byte_identical(tmp_path):
    target = tmp_path / "out.json"
    write_atomic(target, b"hello\n")
    status = write_atomic(target, b"hello\n")
    assert status is WriteStatus.UNCHANGED
    assert target.read_bytes() == b"hello\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]  # no tmp litter


def test_write_atomic_raises_on_overwrite_different(tmp_path):
    target = tmp_path / "out.json"
    write_atomic(target, b"hello\n")
    with pytest.raises(ValueError):
        write_atomic(target, b"goodbye\n")
    # the existing artifact is preserved byte-for-byte -- never partially
    # clobbered by the refused write.
    assert target.read_bytes() == b"hello\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]  # no tmp litter


def test_write_atomic_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "dir" / "out.json"
    status = write_atomic(target, b"hi\n")
    assert status is WriteStatus.WRITTEN
    assert target.read_bytes() == b"hi\n"


def test_write_atomic_never_leaves_a_temp_file_on_the_refused_path(tmp_path):
    target = tmp_path / "out.json"
    write_atomic(target, b"hello\n")
    for _ in range(3):
        with pytest.raises(ValueError):
            write_atomic(target, b"different\n")
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


# ---------------------------------------------------------------------------
# build_protocol
# ---------------------------------------------------------------------------

def test_build_protocol_includes_every_schema_key():
    protocol = build_protocol(_protocol_params())
    assert set(protocol) == set(PROTOCOL_SCHEMA_KEYS)


def test_build_protocol_preserves_values():
    params = _protocol_params()
    protocol = build_protocol(params)
    assert protocol["games"] == 6
    assert protocol["base_seed"] == 900000
    assert protocol["checkpoint_a"] == params["checkpoint_a"]
    assert protocol["checkpoint_b"] == params["checkpoint_b"]
    assert protocol["anchor"] == "checkpoint_a"
    assert protocol["board_size"] == 24
    assert protocol["mcts_sims"] == 400
    assert protocol["mcts_eval_batch_size"] == 14
    assert protocol["mcts_stall_flush_sims"] == 48
    assert protocol["selection_mode"] == "visit_count"
    assert protocol["opening_temp_plies"] == 6
    assert protocol["temp_high"] == 1.0
    assert protocol["temp_low"] == 0.1
    assert protocol["max_moves"] == 300
    assert protocol["save_eval_replays"] is True
    assert protocol["workers"] == 4
    assert protocol["match_summary_path"] == params["match_summary_path"]
    assert protocol["source_index_path"] == params["source_index_path"]
    assert protocol["replay_dir"] == params["replay_dir"]
    assert protocol["config_out"] == params["config_out"]
    assert protocol["report_out"] == params["report_out"]
    assert protocol["selection_seed"] == 20260714
    assert protocol["phase_allocation"] == params["phase_allocation"]
    assert protocol["late_floors"] == params["late_floors"]
    assert protocol["enumerator_params"] == params["enumerator_params"]
    assert protocol["new_collapse_stratum"] == "ply_bucket"
    assert protocol["forbidden_manifests"] == params["forbidden_manifests"]
    assert protocol["screen_out"] == params["screen_out"]
    assert protocol["select_out"] == params["select_out"]
    assert protocol["generation_git_commit"] == "abc123def456"
    assert protocol["generation_source_sha1s"] == params["generation_source_sha1s"]


def test_build_protocol_does_not_invent_extra_keys():
    params = _protocol_params(some_unrelated_bookkeeping_field="ignored")
    protocol = build_protocol(params)
    assert set(protocol) == set(PROTOCOL_SCHEMA_KEYS)
    assert "some_unrelated_bookkeeping_field" not in protocol


def test_build_protocol_rejects_a_single_missing_param():
    params = _protocol_params()
    del params["games"]
    with pytest.raises(ValueError, match="games"):
        build_protocol(params)


def test_build_protocol_names_every_missing_param():
    params = _protocol_params()
    del params["games"]
    del params["anchor"]
    del params["workers"]
    with pytest.raises(ValueError) as exc_info:
        build_protocol(params)
    msg = str(exc_info.value)
    assert "anchor" in msg
    assert "games" in msg
    assert "workers" in msg


def test_build_protocol_rejects_empty_params():
    with pytest.raises(ValueError):
        build_protocol({})


@pytest.mark.parametrize("key", PROTOCOL_SCHEMA_KEYS)
def test_build_protocol_rejects_each_individually_missing_key(key):
    """Every one of the 34 fields is independently required -- not just the
    handful exercised by the other tests above."""
    params = _protocol_params()
    del params[key]
    with pytest.raises(ValueError, match=key):
        build_protocol(params)


# ---------------------------------------------------------------------------
# emit_protocol
# ---------------------------------------------------------------------------

def test_emit_protocol_writes_canonical_bytes(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    params = _protocol_params()
    rc = emit_protocol(params, out)
    assert rc == EXIT_OK
    assert out.read_bytes() == canonical_json_bytes(build_protocol(params))


def test_emit_protocol_is_idempotent_on_reemit(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    params = _protocol_params()
    emit_protocol(params, out)
    before = out.read_bytes()
    rc = emit_protocol(params, out)
    assert rc == EXIT_OK
    assert out.read_bytes() == before


def test_emit_protocol_raises_on_conflicting_reemit(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    emit_protocol(_protocol_params(), out)
    before = out.read_bytes()
    with pytest.raises(ValueError):
        emit_protocol(_protocol_params(games=9600), out)
    assert out.read_bytes() == before  # immutable: refused write never lands


def test_emit_protocol_check_true_never_writes_when_absent(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    rc = emit_protocol(_protocol_params(), out, check=True)
    assert rc == EXIT_MISMATCH
    assert not out.exists()


def test_emit_protocol_check_true_reports_match_without_writing(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    params = _protocol_params()
    emit_protocol(params, out)
    before = out.read_bytes()
    before_mtime = out.stat().st_mtime_ns

    rc = emit_protocol(params, out, check=True)

    assert rc == EXIT_OK
    assert out.read_bytes() == before
    assert out.stat().st_mtime_ns == before_mtime  # never touched
    assert [p.name for p in tmp_path.iterdir()] == ["reservoir_protocol.json"]


def test_emit_protocol_check_true_detects_mismatch_without_writing(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    emit_protocol(_protocol_params(), out)
    before = out.read_bytes()

    rc = emit_protocol(_protocol_params(games=9600), out, check=True)

    assert rc == EXIT_MISMATCH
    assert out.read_bytes() == before  # --check NEVER writes
    assert [p.name for p in tmp_path.iterdir()] == ["reservoir_protocol.json"]


def test_emit_protocol_propagates_missing_param_in_write_mode(tmp_path):
    out = tmp_path / "reservoir_protocol.json"
    params = _protocol_params()
    del params["games"]
    with pytest.raises(ValueError):
        emit_protocol(params, out)
    assert not out.exists()


def test_emit_protocol_propagates_missing_param_in_check_mode(tmp_path):
    """--check still validates params -- it is not a bypass for an invalid
    protocol, only a bypass for WRITING."""
    out = tmp_path / "reservoir_protocol.json"
    params = _protocol_params()
    del params["games"]
    with pytest.raises(ValueError):
        emit_protocol(params, out, check=True)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Exit-code vocabulary (design Sec 3) -- shared module-wide constants.
# ---------------------------------------------------------------------------

def test_exit_code_constants_match_spec_sec_3():
    assert (EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_GATE_FAIL) == (0, 2, 3, 4)


# ---------------------------------------------------------------------------
# Import purity
# ---------------------------------------------------------------------------

def test_module_import_pulls_no_gpu_or_mlx():
    """Importing the module must never load mlx/torch -- checked via a fresh
    subprocess (an in-process check would be unreliable: another test module
    already imported in this same pytest session may have pulled mlx in
    first)."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; "
         "import scripts.GPU.alphazero.fpu_dev_reservoir_protocol as m; "
         "print(sorted(k for k in sys.modules if 'mlx' in k or 'torch' in k))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_module_does_not_yet_import_fpu_dev_corpus_v2():
    """B1 scope guard (plan Task B1 / spec Sec 6): the shared
    schema-key-constant import seam into `fpu_dev_corpus_v2` is a LATER
    task's concern (B3+, once a config-deriving stage needs it) -- not
    this one. Parsed via `ast` (not a raw substring check) so a legitimate
    prose mention of the module's name -- e.g. this file's own docstring
    naming `fpu_dev_corpus_v2_config.json`, the artifact -- can never
    false-positive; only a real `import`/`from ... import` node counts.
    The module's own `__file__` (via `inspect`, on the object already
    imported at the top of this file) locates the source -- an ABSOLUTE
    path, never a cwd-relative guess."""
    import ast
    import inspect

    module_path = Path(inspect.getfile(build_protocol))
    tree = ast.parse(module_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            imported_modules.update(
                f"{node.module}.{alias.name}" for alias in node.names)
    assert not any("fpu_dev_corpus_v2" in m for m in imported_modules), (
        imported_modules)
