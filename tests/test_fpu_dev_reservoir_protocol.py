"""Tests for `fpu_dev_reservoir_protocol.py` Tasks B1-B3: the frozen
`reservoir_protocol.json` field set, the canonical-JSON encoder, the
atomic/immutable write primitive, the `build_protocol` / `emit_protocol`
schema builder + emitter, `gen_command`, and the `ReservoirMeasurements` /
`measure_reservoir` I/O boundary.

Frozen design ref: docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md
  Sec 2.1 (protocol schema), Sec 3 (atomicity/immutability + `--check`
  contract), Sec 4 (qualification measurement boundary), Sec 6 (module
  boundary / circular-import resolution), Sec 8 (canonical JSON,
  determinism).
Pre-op hardening plan ref: docs/superpowers/plans/2026-07-14-fpu-v2-preop-hardening-plan.md
  Tasks B1, B2, B3.

Pure stdlib only, PLUS the same `fpu_provenance` helpers the module under
test itself reuses (used here as the independent "hand-computed" oracle for
every B3 hash assertion -- the same convention
tests/test_fpu_dev_corpus_v2.py uses for `v2_screen_provenance`) -- no
evaluator/MCTS/GPU/MLX/checkpoint WEIGHTS anywhere in this file.
`test_module_import_pulls_no_gpu_or_mlx` proves the module itself is
import-clean via a subprocess (the SAME idiom
tests/test_fpu_dev_corpus_v2.py::test_v2_module_import_pulls_no_gpu_or_mlx
uses -- a plain in-process `import` check would be unreliable here because
another test module in the same pytest session may have already pulled mlx
into `sys.modules` first).
"""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.GPU.alphazero import fpu_provenance
from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _V2_CORPUS_SOURCES
from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import (
    ConformanceResult,
    EXIT_GATE_FAIL,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_USAGE,
    GENERATION_SOURCE_MODULES,
    PROTOCOL_SCHEMA_KEYS,
    QUALIFICATION_SOURCE_FILES,
    ReservoirMeasurements,
    TEN_MATCH_KNOBS,
    WriteStatus,
    build_protocol,
    canonical_json_bytes,
    check_protocol_conformance,
    emit_protocol,
    gen_command,
    measure_reservoir,
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
        "selection_mode": "opening_temperature",
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
    assert protocol["selection_mode"] == "opening_temperature"
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
# gen_command -- Task B2.
# ---------------------------------------------------------------------------

def test_gen_command_produces_the_exact_argv_for_a_fixed_protocol():
    """The load-bearing case: every flag maps to the right protocol field
    with the right value, in one fixed, fully-specified argv -- so an
    accidental typo'd flag name, swapped value, or dropped flag can never
    pass unnoticed."""
    protocol = build_protocol(_protocol_params())
    assert gen_command(protocol) == [
        ".venv/bin/python", "-m", "scripts.GPU.alphazero.eval_checkpoint_match",
        "--checkpoint-a", "checkpoints/calib020_0001.safetensors",
        "--checkpoint-b", "checkpoints/0379.safetensors",
        "--games", "6",
        "--board-size", "24",
        "--mcts-sims", "400",
        "--mcts-eval-batch-size", "14",
        "--mcts-stall-flush-sims", "48",
        "--selection-mode", "opening_temperature",
        "--opening-temp-plies", "6",
        "--temp-high", "1.0",
        "--temp-low", "0.1",
        "--max-moves", "300",
        "--workers", "4",
        "--base-seed", "900000",
        "--save-eval-replays",
        "--replay-dir", "runs/reservoir_v2/match_summary_replays",
        "--output", "runs/reservoir_v2/match_summary.json",
    ]


def test_gen_command_invocation_prefix():
    """Matches how `eval_checkpoint_match` is invoked elsewhere in this repo
    (e.g. docs/post-game-analysis.md) -- module invocation via `-m`, not a
    script path."""
    argv = gen_command(build_protocol(_protocol_params()))
    assert argv[:3] == [
        ".venv/bin/python", "-m", "scripts.GPU.alphazero.eval_checkpoint_match",
    ]


def test_gen_command_uses_checkpoint_path_not_identity():
    """`checkpoint_a`/`checkpoint_b` are each `{"path", "identity"}` --
    the generator takes a filesystem path (the `name:sha1` identity is
    qualification's job to verify AFTER generation, not a generator arg)."""
    protocol = build_protocol(_protocol_params())
    argv = gen_command(protocol)
    assert argv[argv.index("--checkpoint-a") + 1] == protocol["checkpoint_a"]["path"]
    assert argv[argv.index("--checkpoint-b") + 1] == protocol["checkpoint_b"]["path"]
    assert protocol["checkpoint_a"]["identity"] not in argv
    assert protocol["checkpoint_b"]["identity"] not in argv


def test_gen_command_omits_save_eval_replays_flag_when_false():
    """`--save-eval-replays` is `eval_checkpoint_match`'s `store_true` flag
    -- when `save_eval_replays` is false it must be OMITTED entirely, never
    emitted with a `false`/`0` value (that is not how `store_true` works).
    `--replay-dir` is still emitted (protocol Sec 2.1: declared
    unconditionally) even though it would go unused by the generator in
    that case."""
    protocol = build_protocol(_protocol_params(save_eval_replays=False))
    argv = gen_command(protocol)
    assert "--save-eval-replays" not in argv
    assert argv[argv.index("--replay-dir") + 1] == protocol["replay_dir"]


def test_gen_command_emits_bare_save_eval_replays_flag_when_true():
    """A bare flag -- the token immediately after it is the NEXT flag, not
    a value for `--save-eval-replays` itself."""
    protocol = build_protocol(_protocol_params(save_eval_replays=True))
    argv = gen_command(protocol)
    idx = argv.index("--save-eval-replays")
    assert argv[idx + 1].startswith("--")


def test_gen_command_does_not_emit_source_index_path_as_a_flag():
    """`source_index_path` is a `PROTOCOL_SCHEMA_KEYS` field (so a later
    stage can verify the generator's derivation), but `eval_checkpoint_match`
    has no such flag -- it derives the JSONL path itself from `--output`'s
    stem. Neither a `--source-index-path` flag nor the bare value may
    appear anywhere in the argv."""
    protocol = build_protocol(_protocol_params())
    argv = gen_command(protocol)
    assert not any("source-index" in tok for tok in argv)
    assert protocol["source_index_path"] not in argv


def test_gen_command_output_stem_implies_source_index_path():
    """Pins the derivability claim itself: the frozen protocol's declared
    `source_index_path` agrees with what `eval_checkpoint_match._write_outputs`
    would actually derive from the emitted `--output` value
    (`f"{stem}_games.jsonl"`) -- so the missing flag above is provably not
    a silent gap, just a value the generator computes on its own."""
    protocol = build_protocol(_protocol_params())
    argv = gen_command(protocol)
    output = argv[argv.index("--output") + 1]
    stem, _ext = os.path.splitext(output)
    assert f"{stem}_games.jsonl" == protocol["source_index_path"]


def test_gen_command_all_elements_are_strings():
    protocol = build_protocol(_protocol_params())
    argv = gen_command(protocol)
    assert all(isinstance(tok, str) for tok in argv)


def test_gen_command_is_deterministic():
    """The same protocol -- even two SEPARATELY constructed but
    equal-valued protocol dicts -- always produces the same argv."""
    protocol_1 = build_protocol(_protocol_params())
    protocol_2 = build_protocol(_protocol_params())
    assert protocol_1 is not protocol_2
    assert gen_command(protocol_1) == gen_command(protocol_2)
    assert gen_command(protocol_1) == gen_command(protocol_1)  # repeat call


def test_gen_command_reflects_overridden_numeric_knobs():
    """Every one of the ten match knobs, plus `games`/`workers`, is READ
    from the protocol rather than hardcoded -- overriding each
    independently changes the corresponding flag's value."""
    protocol = build_protocol(_protocol_params(
        games=42, board_size=18, mcts_sims=111, mcts_eval_batch_size=7,
        mcts_stall_flush_sims=9, selection_mode="argmax",
        opening_temp_plies=3, temp_high=2.5, temp_low=0.25, max_moves=99,
        base_seed=555, workers=2,
    ))
    argv = gen_command(protocol)

    def value_of(flag):
        return argv[argv.index(flag) + 1]

    assert value_of("--games") == "42"
    assert value_of("--board-size") == "18"
    assert value_of("--mcts-sims") == "111"
    assert value_of("--mcts-eval-batch-size") == "7"
    assert value_of("--mcts-stall-flush-sims") == "9"
    assert value_of("--selection-mode") == "argmax"
    assert value_of("--opening-temp-plies") == "3"
    assert value_of("--temp-high") == "2.5"
    assert value_of("--temp-low") == "0.25"
    assert value_of("--max-moves") == "99"
    assert value_of("--base-seed") == "555"
    assert value_of("--workers") == "2"


def test_gen_command_reflects_overridden_matchup_and_output_paths():
    protocol = build_protocol(_protocol_params(
        checkpoint_a={"path": "checkpoints/other_a.safetensors",
                      "identity": "other_a:cccccccc"},
        checkpoint_b={"path": "checkpoints/other_b.safetensors",
                      "identity": "other_b:dddddddd"},
        match_summary_path="runs/other/match_summary.json",
        source_index_path="runs/other/match_summary_games.jsonl",
        replay_dir="runs/other/match_summary_replays",
    ))
    argv = gen_command(protocol)

    def value_of(flag):
        return argv[argv.index(flag) + 1]

    assert value_of("--checkpoint-a") == "checkpoints/other_a.safetensors"
    assert value_of("--checkpoint-b") == "checkpoints/other_b.safetensors"
    assert value_of("--output") == "runs/other/match_summary.json"
    assert value_of("--replay-dir") == "runs/other/match_summary_replays"


# ---------------------------------------------------------------------------
# ReservoirMeasurements / measure_reservoir -- Task B3.
#
# `measure_reservoir` is the ONLY filesystem I/O in the whole qualification
# pipeline (design Sec 4/Sec 6). Every assertion below either (a) proves a
# `ReservoirMeasurements` field equals an INDEPENDENTLY hand-computed value
# -- via the SAME `fpu_provenance` helpers `measure_reservoir` itself calls,
# the established oracle convention this codebase already uses for
# `v2_screen_provenance` (tests/test_fpu_dev_corpus_v2.py) -- or (b) proves
# constructing a `ReservoirMeasurements` directly touches no disk at all.
# ---------------------------------------------------------------------------

def _mini_sidecar(game_idx: int, seed: int) -> dict:
    """A minimal, schema-shaped replay sidecar (mirrors `eval_replay.
    build_replay_dict`'s field set) for one fabricated game. Model color
    alternates by `game_idx` parity (design Sec 4.1's convention), though
    B3 itself validates none of this -- only a LATER stage (B4) does;
    `measure_reservoir` loads it verbatim."""
    red_is_a = (game_idx % 2 == 0)
    return {
        "schema_version": 1,
        "pairing_id": "calib020_0001_vs_0379",
        "game_idx": game_idx,
        "task_id": game_idx,
        "seed": seed,
        "board_size": 24,
        "red_checkpoint": ("checkpoints/calib020_0001.safetensors" if red_is_a
                            else "checkpoints/0379.safetensors"),
        "black_checkpoint": ("checkpoints/0379.safetensors" if red_is_a
                              else "checkpoints/calib020_0001.safetensors"),
        "winner": "red" if game_idx == 0 else "black",
        "winner_checkpoint": "checkpoints/calib020_0001.safetensors",
        "reason": "win" if game_idx == 0 else "state_cap",
        "n_moves": 42 if game_idx == 0 else 300,
        "moves": [],
    }


def _mini_jsonl_row(game_idx: int, replay_path: Path) -> dict:
    """A full `EvalGameResult`-shaped row -- every field
    `eval_checkpoint_match._write_outputs` writes via
    `json.dumps(asdict(r))` (task_id, pairing_id, game_idx,
    red/black_checkpoint, winner, winner_checkpoint, reason, n_moves,
    red/black_score, replay_path) -- NOT the narrower subset
    `build_fpu_dev_corpus.load_game_index` keeps (game_idx/n_moves/winner/
    replay_path only): a LATER stage (B5) reconstructs full
    `EvalGameResult` rows from `measurements.jsonl_rows`, so every field
    must survive `measure_reservoir` verbatim."""
    red_is_a = (game_idx % 2 == 0)
    return {
        "task_id": game_idx,
        "pairing_id": "calib020_0001_vs_0379",
        "game_idx": game_idx,
        "red_checkpoint": ("checkpoints/calib020_0001.safetensors" if red_is_a
                            else "checkpoints/0379.safetensors"),
        "black_checkpoint": ("checkpoints/0379.safetensors" if red_is_a
                              else "checkpoints/calib020_0001.safetensors"),
        "winner": "red" if game_idx == 0 else "black",
        "winner_checkpoint": "checkpoints/calib020_0001.safetensors",
        "reason": "win" if game_idx == 0 else "state_cap",
        "n_moves": 42 if game_idx == 0 else 300,
        "red_score": 1.0 if game_idx == 0 else 0.0,
        "black_score": 0.0 if game_idx == 0 else 1.0,
        "replay_path": str(replay_path),
    }


def _write_mini_reservoir(tmp_path: Path, *, base_seed: int = 900000,
                          n_games: int = 2, protocol_overrides=None):
    """Fabricate a tiny, fully ON-DISK reservoir under `tmp_path`: real
    (fake-content) checkpoint files, real replay sidecars, a real JSONL
    index linking to them, a real match-summary JSON, and a real
    forbidden-manifest file -- so `measure_reservoir` has something genuine
    to read and hash.

    Returns `(protocol, info)`: `protocol` is a complete, valid
    `build_protocol`-shaped dict pointing at every fabricated path; `info`
    is a plain dict of the raw fabricated data (`rows`, `sidecars`,
    `summary`, and every path) so each test can independently recompute its
    own "hand-computed" expectation without re-deriving the fixture's own
    internals a second time."""
    ckpt_a_path = tmp_path / "checkpoints" / "calib020_0001.safetensors"
    ckpt_b_path = tmp_path / "checkpoints" / "0379.safetensors"
    ckpt_a_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_a_path.write_bytes(b"fake-checkpoint-a-bytes")
    ckpt_b_path.write_bytes(b"fake-checkpoint-b-bytes-different")

    replay_dir = tmp_path / "replays"
    replay_paths = []
    sidecars = {}
    rows = []
    for game_idx in range(n_games):
        sidecar = _mini_sidecar(game_idx, base_seed + game_idx)
        replay_path = replay_dir / f"game_{game_idx:06d}.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(sidecar))
        replay_paths.append(replay_path)
        sidecars[game_idx] = sidecar
        rows.append(_mini_jsonl_row(game_idx, replay_path))

    index_path = tmp_path / "match_summary_games.jsonl"
    with open(index_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "pairing_id": "calib020_0001_vs_0379",
        "checkpoint_a": "checkpoints/calib020_0001.safetensors",
        "checkpoint_b": "checkpoints/0379.safetensors",
        "games": n_games,
        "git_commit": "abc123def456",
        "generated_at": "2026-07-14T00:00:00+00:00",
    }
    summary_path = tmp_path / "match_summary.json"
    summary_path.write_text(json.dumps(summary))

    manifest_path = tmp_path / "manifests" / "v1_controls.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("state_sha1\nabc\n")

    params = _protocol_params(
        games=n_games,
        checkpoint_a={"path": str(ckpt_a_path), "identity": "placeholder-a"},
        checkpoint_b={"path": str(ckpt_b_path), "identity": "placeholder-b"},
        anchor="checkpoint_a",
        base_seed=base_seed,
        source_index_path=str(index_path),
        match_summary_path=str(summary_path),
        replay_dir=str(replay_dir),
        forbidden_manifests=[str(manifest_path)],
    )
    if protocol_overrides:
        params.update(protocol_overrides)
    protocol = build_protocol(params)

    info = {
        "rows": rows,
        "sidecars": sidecars,
        "summary": summary,
        "checkpoint_a_path": ckpt_a_path,
        "checkpoint_b_path": ckpt_b_path,
        "index_path": index_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "replay_paths": replay_paths,
    }
    return protocol, info


# ---------------------------------------------------------------------------
# ReservoirMeasurements -- shape, immutability, purity of construction.
# ---------------------------------------------------------------------------

_RESERVOIR_MEASUREMENTS_FIELDS = (
    "jsonl_rows", "sidecars_by_idx", "summary", "checkpoint_identities",
    "generation_source_sha1s", "generation_git_commit", "source_index_sha1",
    "replay_data_sha1", "match_summary_sha1", "source_file_sha1s",
    "forbidden_manifest_sha1s",
)


def test_reservoir_measurements_has_exactly_the_spec_sec_4_fields():
    field_names = {f.name for f in dataclasses.fields(ReservoirMeasurements)}
    assert field_names == set(_RESERVOIR_MEASUREMENTS_FIELDS)
    assert len(_RESERVOIR_MEASUREMENTS_FIELDS) == 11


def test_reservoir_measurements_is_frozen():
    measurements = ReservoirMeasurements(
        jsonl_rows=[], sidecars_by_idx={}, summary={},
        checkpoint_identities={}, generation_source_sha1s={},
        generation_git_commit="unknown", source_index_sha1="none",
        replay_data_sha1="none", match_summary_sha1="none",
        source_file_sha1s={}, forbidden_manifest_sha1s={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        measurements.summary = {"tampered": True}


def test_constructing_reservoir_measurements_directly_does_no_io():
    """Every field is loaded from a path that DOES NOT EXIST on this
    machine -- if construction performed any I/O it would raise; it does
    not, because `ReservoirMeasurements` is an ordinary `@dataclass(frozen=
    True)` with no `__post_init__`, and every field here is a plain,
    already-in-memory Python value (this is the whole point of the
    measurement boundary: B4-B6 construct/consume these directly, with no
    disk in the loop)."""
    fake_path = "/definitely/does/not/exist/on/this/machine.json"
    measurements = ReservoirMeasurements(
        jsonl_rows=[{"game_idx": 0, "replay_path": fake_path}],
        sidecars_by_idx={0: {"game_idx": 0}},
        summary={"pairing_id": "fake"},
        checkpoint_identities={"reservoir_a": "a:deadbeef",
                               "reservoir_b": "b:deadbeef",
                               "anchor": "a:deadbeef"},
        generation_source_sha1s={"eval_runner.py": "deadbeef"},
        generation_git_commit="deadbeef",
        source_index_sha1="deadbeef",
        replay_data_sha1="deadbeef",
        match_summary_sha1="deadbeef",
        source_file_sha1s={"fpu_dev_reservoir_protocol.py": "deadbeef"},
        forbidden_manifest_sha1s={"v1_controls.csv": "deadbeef"},
    )
    assert measurements.summary == {"pairing_id": "fake"}
    assert measurements.jsonl_rows == [{"game_idx": 0, "replay_path": fake_path}]
    assert measurements.checkpoint_identities["reservoir_a"] == "a:deadbeef"


# ---------------------------------------------------------------------------
# measure_reservoir -- loaded data (jsonl_rows / sidecars_by_idx / summary).
# ---------------------------------------------------------------------------

def test_measure_reservoir_returns_a_reservoir_measurements_instance(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert isinstance(measurements, ReservoirMeasurements)


def test_measure_reservoir_loads_jsonl_rows_verbatim(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.jsonl_rows == info["rows"]


def test_measure_reservoir_loads_sidecars_keyed_by_game_idx(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.sidecars_by_idx == info["sidecars"]
    assert set(measurements.sidecars_by_idx) == {0, 1}


def test_measure_reservoir_loads_the_match_summary(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.summary == info["summary"]


# ---------------------------------------------------------------------------
# measure_reservoir -- checkpoint_identities (reservoir_a/reservoir_b/anchor).
# ---------------------------------------------------------------------------

def test_measure_reservoir_checkpoint_identities_reservoir_a_and_b(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    expected_a = (f"{info['checkpoint_a_path'].name}:"
                  f"{fpu_provenance.file_sha1(str(info['checkpoint_a_path']))}")
    expected_b = (f"{info['checkpoint_b_path'].name}:"
                  f"{fpu_provenance.file_sha1(str(info['checkpoint_b_path']))}")
    assert measurements.checkpoint_identities["reservoir_a"] == expected_a
    assert measurements.checkpoint_identities["reservoir_b"] == expected_b
    # The two fake checkpoints have DIFFERENT bytes -- proves this isn't
    # accidentally hashing the same file twice.
    assert expected_a != expected_b


def test_measure_reservoir_checkpoint_identities_exact_key_set(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert set(measurements.checkpoint_identities) == {
        "reservoir_a", "reservoir_b", "anchor"}


def test_measure_reservoir_anchor_identity_resolves_checkpoint_a_when_declared(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)  # anchor == "checkpoint_a"
    measurements = measure_reservoir(protocol)
    assert (measurements.checkpoint_identities["anchor"]
            == measurements.checkpoint_identities["reservoir_a"])
    assert (measurements.checkpoint_identities["anchor"]
            != measurements.checkpoint_identities["reservoir_b"])


def test_measure_reservoir_anchor_identity_resolves_checkpoint_b_when_declared(tmp_path):
    protocol, _info = _write_mini_reservoir(
        tmp_path, protocol_overrides={"anchor": "checkpoint_b"})
    measurements = measure_reservoir(protocol)
    assert (measurements.checkpoint_identities["anchor"]
            == measurements.checkpoint_identities["reservoir_b"])
    assert (measurements.checkpoint_identities["anchor"]
            != measurements.checkpoint_identities["reservoir_a"])


# ---------------------------------------------------------------------------
# measure_reservoir -- generation_source_sha1s / generation_git_commit.
# ---------------------------------------------------------------------------

def test_measure_reservoir_generation_source_sha1s_covers_the_thirteen_modules(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert set(measurements.generation_source_sha1s) == {
        "eval_checkpoint_match.py", "eval_runner.py", "mcts.py",
        "opening_diagnostics.py", "evaluator.py", "twixt_state.py",
        "__init__.py", "eval_replay.py", "probe_eval.py", "network.py",
        "local_evaluator.py", "eval_summary.py", "eval_elo.py",
    }
    assert len(measurements.generation_source_sha1s) == 13


def test_measure_reservoir_generation_source_sha1s_matches_hand_computed_hashes(tmp_path):
    """Cross-checks against the SAME oracle (`fpu_provenance.source_file_
    sha1s`) called independently over the pinned 13-module list -- these
    are REAL project source files (not fixture-fabricated), so this also
    proves `measure_reservoir` reads THIS repo's generation sources, not a
    fixture stand-in."""
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    expected = fpu_provenance.source_file_sha1s(GENERATION_SOURCE_MODULES)
    assert measurements.generation_source_sha1s == expected


def test_measure_reservoir_generation_git_commit_matches_fpu_provenance(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.generation_git_commit == fpu_provenance.git_commit()
    assert measurements.generation_git_commit != "unknown"  # this IS a git repo


# ---------------------------------------------------------------------------
# measure_reservoir -- source_index_sha1 / replay_data_sha1 / match_summary_sha1.
# ---------------------------------------------------------------------------

def test_measure_reservoir_source_index_sha1_matches_hand_computed_hash(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.source_index_sha1 == fpu_provenance.file_sha1(
        str(info["index_path"]))


def test_measure_reservoir_replay_data_sha1_matches_hand_computed_hash(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    expected = fpu_provenance.replay_data_sha1(
        [str(p) for p in info["replay_paths"]])
    assert measurements.replay_data_sha1 == expected


def test_measure_reservoir_match_summary_sha1_matches_hand_computed_hash(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert measurements.match_summary_sha1 == fpu_provenance.file_sha1(
        str(info["summary_path"]))


# ---------------------------------------------------------------------------
# measure_reservoir -- source_file_sha1s / forbidden_manifest_sha1s.
# ---------------------------------------------------------------------------

def test_measure_reservoir_source_file_sha1s_matches_qualification_source_files(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    expected = fpu_provenance.source_file_sha1s(QUALIFICATION_SOURCE_FILES)
    assert measurements.source_file_sha1s == expected


def test_measure_reservoir_source_file_sha1s_includes_the_qualification_module_itself(tmp_path):
    """Spec Sec 2.2 amendment 4: 'the qualification module is
    result-determining for the corpus it produces' -- `fpu_dev_reservoir_
    protocol.py` (THIS module) must be one of the hashed sources, not just
    the pre-existing v1/v2 corpus set."""
    import inspect
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    assert "fpu_dev_reservoir_protocol.py" in measurements.source_file_sha1s
    this_module_path = Path(inspect.getfile(measure_reservoir))
    assert (measurements.source_file_sha1s["fpu_dev_reservoir_protocol.py"]
            == fpu_provenance.file_sha1(str(this_module_path)))


def test_measure_reservoir_source_file_sha1s_includes_the_v2_corpus_sources(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    for path in _V2_CORPUS_SOURCES:
        assert path.name in measurements.source_file_sha1s


def test_measure_reservoir_forbidden_manifest_sha1s_matches_hand_computed_hash(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    expected = fpu_provenance.source_file_sha1s([str(info["manifest_path"])])
    assert measurements.forbidden_manifest_sha1s == expected
    assert "v1_controls.csv" in measurements.forbidden_manifest_sha1s


def test_measure_reservoir_forbidden_manifest_sha1s_empty_when_declared_empty(tmp_path):
    protocol, _info = _write_mini_reservoir(
        tmp_path, protocol_overrides={"forbidden_manifests": []})
    measurements = measure_reservoir(protocol)
    assert measurements.forbidden_manifest_sha1s == {}


# ---------------------------------------------------------------------------
# measure_reservoir -- the load-bearing full cross-check + dynamism + errors.
# ---------------------------------------------------------------------------

def test_measure_reservoir_full_field_set_matches_independent_hand_computation(tmp_path):
    """The load-bearing case (mirrors `test_gen_command_produces_the_exact_
    argv_for_a_fixed_protocol`'s role for B2): every one of the 11 fields,
    recomputed independently (never by calling `measure_reservoir` a second
    time), matches -- so an accidental swapped path, wrong dict key, or
    silently-dropped field can never pass unnoticed."""
    protocol, info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)

    checkpoint_a_identity = (
        f"{info['checkpoint_a_path'].name}:"
        f"{fpu_provenance.file_sha1(str(info['checkpoint_a_path']))}")
    checkpoint_b_identity = (
        f"{info['checkpoint_b_path'].name}:"
        f"{fpu_provenance.file_sha1(str(info['checkpoint_b_path']))}")

    expected = ReservoirMeasurements(
        jsonl_rows=info["rows"],
        sidecars_by_idx=info["sidecars"],
        summary=info["summary"],
        checkpoint_identities={
            "reservoir_a": checkpoint_a_identity,
            "reservoir_b": checkpoint_b_identity,
            "anchor": checkpoint_a_identity,   # fixture declares anchor="checkpoint_a"
        },
        generation_source_sha1s=fpu_provenance.source_file_sha1s(
            GENERATION_SOURCE_MODULES),
        generation_git_commit=fpu_provenance.git_commit(),
        source_index_sha1=fpu_provenance.file_sha1(str(info["index_path"])),
        replay_data_sha1=fpu_provenance.replay_data_sha1(
            [str(p) for p in info["replay_paths"]]),
        match_summary_sha1=fpu_provenance.file_sha1(str(info["summary_path"])),
        source_file_sha1s=fpu_provenance.source_file_sha1s(
            QUALIFICATION_SOURCE_FILES),
        forbidden_manifest_sha1s=fpu_provenance.source_file_sha1s(
            [str(info["manifest_path"])]),
    )
    assert measurements == expected


def test_measure_reservoir_reads_the_declared_paths_not_a_fixed_location(tmp_path):
    """Two INDEPENDENT mini-reservoirs, in two different subdirectories,
    produce DIFFERENT reservoir-scoped measurements -- proving
    `measure_reservoir` really reads `protocol[...]`'s paths dynamically
    rather than some memoized or hardcoded location -- while agreeing on
    the shared, repo-wide identities (the generation sources and the v2
    corpus sources do not depend on which reservoir was measured)."""
    protocol_1, _info_1 = _write_mini_reservoir(tmp_path / "one")
    protocol_2, _info_2 = _write_mini_reservoir(tmp_path / "two", base_seed=111111)

    measurements_1 = measure_reservoir(protocol_1)
    measurements_2 = measure_reservoir(protocol_2)

    assert measurements_1.jsonl_rows != measurements_2.jsonl_rows
    assert measurements_1.source_index_sha1 != measurements_2.source_index_sha1
    assert measurements_1.sidecars_by_idx != measurements_2.sidecars_by_idx
    assert measurements_1.generation_source_sha1s == measurements_2.generation_source_sha1s
    assert measurements_1.source_file_sha1s == measurements_2.source_file_sha1s


def test_measure_reservoir_raises_when_source_index_path_is_missing(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    protocol["source_index_path"] = str(tmp_path / "does_not_exist.jsonl")
    with pytest.raises(FileNotFoundError):
        measure_reservoir(protocol)


def test_measure_reservoir_raises_when_match_summary_path_is_missing(tmp_path):
    protocol, _info = _write_mini_reservoir(tmp_path)
    protocol["match_summary_path"] = str(tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        measure_reservoir(protocol)


def test_measure_reservoir_raises_when_a_replay_sidecar_is_missing(tmp_path):
    protocol, info = _write_mini_reservoir(tmp_path)
    info["replay_paths"][0].unlink()
    with pytest.raises(FileNotFoundError):
        measure_reservoir(protocol)


def test_measure_reservoir_raises_naming_a_missing_reservoir_checkpoint(tmp_path):
    """A missing checkpoint must FAIL LOUD, not silently bake
    `fpu_provenance`'s `"missing"` sentinel into `checkpoint_identities`
    (which -- being STABLE -- would keep hard-matching cleanly through the
    config's re-derive-and-byte-compare, silently passing a genuinely-absent
    network). The raise names the offending path so the operator can act."""
    protocol, info = _write_mini_reservoir(tmp_path)
    info["checkpoint_a_path"].unlink()
    with pytest.raises(FileNotFoundError, match="calib020_0001"):
        measure_reservoir(protocol)


def test_measure_reservoir_raises_naming_a_missing_checkpoint_b(tmp_path):
    """The same fail-loud guarantee for checkpoint B (not just A/anchor) --
    proves the guard covers every one of the three checkpoint roles, not
    only the anchor role that happens to alias A in the fixture."""
    protocol, info = _write_mini_reservoir(tmp_path)
    info["checkpoint_b_path"].unlink()
    with pytest.raises(FileNotFoundError, match="0379"):
        measure_reservoir(protocol)


def test_measure_reservoir_raises_naming_a_missing_forbidden_manifest(tmp_path):
    """A missing forbidden-manifest path must FAIL LOUD, not silently bake a
    `"missing"` sentinel into `forbidden_manifest_sha1s` -- the exact
    silent-partial-measurement the tamper-evident config cannot tolerate
    (a genuinely-absent manifest would otherwise re-derive/byte-compare
    cleanly). The raise names the offending path."""
    protocol, info = _write_mini_reservoir(tmp_path)
    info["manifest_path"].unlink()
    with pytest.raises(FileNotFoundError, match="v1_controls"):
        measure_reservoir(protocol)


def test_measure_reservoir_no_missing_sentinel_leaks_into_any_field(tmp_path):
    """Belt-and-braces over the whole clean-reservoir measurement: NONE of
    the string/dict identity fields may EVER carry `fpu_provenance`'s
    `"missing"`/`"none"` sentinel on a well-formed reservoir -- the guard's
    net effect is that a sentinel can only mean 'this specific input was
    absent', and on a complete reservoir there are none."""
    protocol, _info = _write_mini_reservoir(tmp_path)
    measurements = measure_reservoir(protocol)
    flat = [measurements.source_index_sha1, measurements.replay_data_sha1,
            measurements.match_summary_sha1, measurements.generation_git_commit]
    flat += list(measurements.checkpoint_identities.values())
    flat += list(measurements.generation_source_sha1s.values())
    flat += list(measurements.source_file_sha1s.values())
    flat += list(measurements.forbidden_manifest_sha1s.values())
    for value in flat:
        assert "missing" not in value, value
        assert value != "none", value


# ---------------------------------------------------------------------------
# GENERATION_SOURCE_MODULES / QUALIFICATION_SOURCE_FILES -- Task B3.
# ---------------------------------------------------------------------------

def test_generation_source_modules_is_the_exact_spec_2_1_thirteen_module_list():
    """Pins both membership AND order (design Sec 2.1's own presentation
    order) -- mirrors `test_protocol_schema_keys_is_the_exact_spec_2_1_
    field_set`'s pinning style."""
    names = [p.name for p in GENERATION_SOURCE_MODULES]
    assert names == [
        "eval_checkpoint_match.py", "eval_runner.py", "mcts.py",
        "opening_diagnostics.py", "evaluator.py", "twixt_state.py",
        "__init__.py", "eval_replay.py", "probe_eval.py", "network.py",
        "local_evaluator.py", "eval_summary.py", "eval_elo.py",
    ]
    assert len(GENERATION_SOURCE_MODULES) == 13


def test_generation_source_modules_paths_all_exist_on_disk():
    """Sanity: a typo'd filename would otherwise silently hash to
    `fpu_provenance`'s `"missing"` sentinel instead of failing loud."""
    missing = [str(p) for p in GENERATION_SOURCE_MODULES if not p.exists()]
    assert missing == []


def test_generation_source_modules_paths_are_distinguishable_by_basename():
    """`game/twixt_state.py` and `game/__init__.py` share no basename with
    any other of the 13 -- so keying by basename (as `fpu_provenance.
    source_file_sha1s` does) never silently collides two different files."""
    paths_by_name = {}
    for p in GENERATION_SOURCE_MODULES:
        assert p.name not in paths_by_name, (p.name, paths_by_name[p.name], p)
        paths_by_name[p.name] = p


def test_qualification_source_files_extends_v2_corpus_sources_with_self():
    import inspect
    this_module_path = Path(inspect.getfile(measure_reservoir)).resolve()
    assert QUALIFICATION_SOURCE_FILES == tuple(_V2_CORPUS_SOURCES) + (this_module_path,)


def test_qualification_source_files_has_no_duplicates():
    assert len(QUALIFICATION_SOURCE_FILES) == len(set(QUALIFICATION_SOURCE_FILES))


def test_qualification_source_files_are_distinguishable_by_basename():
    """`fpu_provenance.source_file_sha1s` keys by BASENAME -- so two entries
    sharing a basename (even from different subdirs) would silently collide,
    one overwriting the other's hash. Mirrors the same guard for
    `GENERATION_SOURCE_MODULES`."""
    paths_by_name = {}
    for p in QUALIFICATION_SOURCE_FILES:
        assert p.name not in paths_by_name, (p.name, paths_by_name[p.name], p)
        paths_by_name[p.name] = p


# ---------------------------------------------------------------------------
# check_protocol_conformance / ConformanceResult -- Task B4 (design Sec 4.1).
#
# PURE over an already-built `ReservoirMeasurements` + `protocol`: every
# fixture below is fabricated DIRECTLY (no disk, no `measure_reservoir`
# call) -- mirrors the B3 "constructing a ReservoirMeasurements directly ...
# performs NO I/O" convention this file already established.
# ---------------------------------------------------------------------------

def _ply_record(ply: int) -> dict:
    """One minimal, schema-shaped `eval_replay.ply_record` dict -- alternates
    red (even ply) / black (odd ply), matching `eval_runner.play_eval_game`
    (`TwixtState(..., to_move="red", ...)`: red moves first, ply 0)."""
    return {
        "ply": ply, "player": "red" if ply % 2 == 0 else "black",
        "row": 1, "col": 1, "root_value": 0.0, "root_top1_share": 1.0,
        "selected_visit_rank": 1, "selected_visit_count": 1,
        "root_total_visits": 1, "n_legal": 1,
    }


def _conformant_reservoir(games: int = 6, n_moves: int = 4, **protocol_overrides):
    """Build a FULLY conformant `(protocol, measurements)` pair -- the
    shared clean baseline every `check_protocol_conformance` defect test
    below mutates exactly ONE field of. A faithful, internally consistent
    reservoir mirroring the real shapes `eval_checkpoint_match`/
    `eval_runner`/`eval_replay` produce (design Sec 4.1): balanced colors
    (even `game_idx` -> checkpoint-A red, `eval_runner.build_pairing_tasks`'s
    own rule), seeds `base_seed + game_idx`, replay sidecars filed directly
    under `replay_dir`, a `summary["config"]` carrying the ten match knobs +
    `workers` verbatim from the protocol, and per-ply mover alternation (red
    on even ply). Fabricated directly -- no disk, no `measure_reservoir`
    call.

    `protocol_overrides` flow into the shared `_protocol_params()` fixture
    (e.g. `replay_dir=...`, `checkpoint_a=...`) -- the fabricated rows /
    sidecars / summary are all DERIVED from the resulting protocol, so an
    override stays self-consistent (e.g. overriding `replay_dir` moves
    where the fabricated rows' `replay_path`s point too).
    """
    params = _protocol_params(games=games, **protocol_overrides)
    protocol = build_protocol(params)

    ckpt_a_path = protocol["checkpoint_a"]["path"]
    ckpt_b_path = protocol["checkpoint_b"]["path"]
    ckpt_a_id = protocol["checkpoint_a"]["identity"]
    ckpt_b_id = protocol["checkpoint_b"]["identity"]
    base_seed = protocol["base_seed"]
    replay_dir = protocol["replay_dir"]
    board_size = protocol["board_size"]

    moves = [_ply_record(ply) for ply in range(n_moves)]

    jsonl_rows = []
    sidecars_by_idx = {}
    for game_idx in range(games):
        red_is_a = (game_idx % 2 == 0)
        red_ckpt = ckpt_a_path if red_is_a else ckpt_b_path
        black_ckpt = ckpt_b_path if red_is_a else ckpt_a_path
        seed = base_seed + game_idx
        replay_path = f"{replay_dir}/game_{game_idx:06d}.json"

        jsonl_rows.append({
            "task_id": game_idx, "pairing_id": "calib020_0001_vs_0379",
            "game_idx": game_idx, "red_checkpoint": red_ckpt,
            "black_checkpoint": black_ckpt, "winner": "red",
            "winner_checkpoint": red_ckpt, "reason": "win",
            "n_moves": n_moves, "red_score": 1.0, "black_score": 0.0,
            "replay_path": replay_path,
        })
        sidecars_by_idx[game_idx] = {
            "schema_version": 1, "pairing_id": "calib020_0001_vs_0379",
            "game_idx": game_idx, "task_id": game_idx, "seed": seed,
            "board_size": board_size, "red_checkpoint": red_ckpt,
            "black_checkpoint": black_ckpt, "winner": "red",
            "winner_checkpoint": red_ckpt, "reason": "win",
            "n_moves": n_moves, "moves": [dict(m) for m in moves],
        }

    summary_config = {
        "board_size": protocol["board_size"],
        "mcts_sims": protocol["mcts_sims"],
        "mcts_eval_batch_size": protocol["mcts_eval_batch_size"],
        "mcts_stall_flush_sims": protocol["mcts_stall_flush_sims"],
        "selection_mode": protocol["selection_mode"],
        "opening_temp_plies": protocol["opening_temp_plies"],
        "temp_high": protocol["temp_high"], "temp_low": protocol["temp_low"],
        "max_moves": protocol["max_moves"], "base_seed": base_seed,
        "workers": protocol["workers"],
    }
    summary = {
        "pairing_id": "calib020_0001_vs_0379",
        "checkpoint_a": ckpt_a_path, "checkpoint_b": ckpt_b_path,
        "games": games,
        "config": summary_config,
        "git_commit": protocol["generation_git_commit"],
        "generated_at": "2026-07-14T00:00:00+00:00",
    }

    measurements = ReservoirMeasurements(
        jsonl_rows=jsonl_rows,
        sidecars_by_idx=sidecars_by_idx,
        summary=summary,
        checkpoint_identities={
            "reservoir_a": ckpt_a_id, "reservoir_b": ckpt_b_id,
            "anchor": ckpt_a_id if protocol["anchor"] == "checkpoint_a" else ckpt_b_id,
        },
        generation_source_sha1s=dict(protocol["generation_source_sha1s"]),
        generation_git_commit=protocol["generation_git_commit"],
        source_index_sha1="index-sha1-placeholder",
        replay_data_sha1="replay-data-sha1-placeholder",
        match_summary_sha1="summary-sha1-placeholder",
        source_file_sha1s={"fpu_dev_reservoir_protocol.py": "src-sha1-placeholder"},
        forbidden_manifest_sha1s={"v1_controls.csv": "manifest-sha1-placeholder"},
    )
    return protocol, measurements


# ---------------------------------------------------------------------------
# TEN_MATCH_KNOBS
# ---------------------------------------------------------------------------

def test_ten_match_knobs_is_the_exact_spec_2_1_amendment_4_set():
    assert TEN_MATCH_KNOBS == (
        "board_size", "mcts_sims", "mcts_eval_batch_size", "mcts_stall_flush_sims",
        "selection_mode", "opening_temp_plies", "temp_high", "temp_low",
        "max_moves", "base_seed",
    )
    assert "workers" not in TEN_MATCH_KNOBS  # operational, checked separately


# ---------------------------------------------------------------------------
# ConformanceResult -- shape / immutability.
# ---------------------------------------------------------------------------

def test_conformance_result_ok_shape():
    result = ConformanceResult(ok=True, reason=None)
    assert result.ok is True
    assert result.reason is None


def test_conformance_result_mismatch_shape():
    result = ConformanceResult(ok=False, reason="game_count: mismatch")
    assert result.ok is False
    assert "game_count" in result.reason


def test_conformance_result_is_frozen():
    result = ConformanceResult(ok=True, reason=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ok = False


# ---------------------------------------------------------------------------
# _conformant_reservoir fixture self-check + the clean/ok case.
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_ok_on_clean_reservoir():
    protocol, measurements = _conformant_reservoir()
    result = check_protocol_conformance(protocol, measurements)
    assert result == ConformanceResult(ok=True, reason=None)


def test_check_protocol_conformance_performs_no_io():
    """PURE -- reads only `measurements`/`protocol`, never touches disk.
    Every path-shaped field here points somewhere that does not exist on
    this machine; if `check_protocol_conformance` performed any I/O it
    would raise, not return a clean result (mirrors B3's own
    `test_constructing_reservoir_measurements_directly_does_no_io`)."""
    protocol, measurements = _conformant_reservoir(
        match_summary_path="/definitely/does/not/exist/match_summary.json",
        source_index_path="/definitely/does/not/exist/match_summary_games.jsonl",
        replay_dir="/definitely/does/not/exist/replays",
        checkpoint_a={"path": "/definitely/does/not/exist/a.safetensors",
                      "identity": "a:deadbeef"},
        checkpoint_b={"path": "/definitely/does/not/exist/b.safetensors",
                      "identity": "b:deadbeef"},
    )
    result = check_protocol_conformance(protocol, measurements)
    assert result == ConformanceResult(ok=True, reason=None)


# ---------------------------------------------------------------------------
# _validate_protocol_shape -- the B1-deferred nested-shape/enum validation.
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_rejects_non_mapping_checkpoint_a():
    protocol, measurements = _conformant_reservoir()
    protocol = {**protocol, "checkpoint_a": "not-a-mapping"}
    with pytest.raises(ValueError, match="checkpoint_a"):
        check_protocol_conformance(protocol, measurements)


def test_check_protocol_conformance_rejects_checkpoint_b_missing_identity():
    protocol, measurements = _conformant_reservoir()
    protocol = {**protocol, "checkpoint_b": {"path": "checkpoints/0379.safetensors"}}
    with pytest.raises(ValueError, match="checkpoint_b"):
        check_protocol_conformance(protocol, measurements)


def test_check_protocol_conformance_rejects_invalid_anchor_enum():
    protocol, measurements = _conformant_reservoir()
    protocol = {**protocol, "anchor": "checkpoint_c"}
    with pytest.raises(ValueError, match="anchor"):
        check_protocol_conformance(protocol, measurements)


# ---------------------------------------------------------------------------
# game_count
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_game_count_jsonl_rows():
    protocol, measurements = _conformant_reservoir()
    rows = measurements.jsonl_rows[:-1]  # drop the last row: 5 rows, games=6
    bad = dataclasses.replace(measurements, jsonl_rows=rows)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "game_count" in result.reason


def test_check_protocol_conformance_wrong_game_count_sidecars():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    del sidecars[5]
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "game_count" in result.reason


# ---------------------------------------------------------------------------
# contiguity
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_non_contiguous_game_idx():
    protocol, measurements = _conformant_reservoir()
    rows = [dict(r) for r in measurements.jsonl_rows]
    rows[5]["game_idx"] = 2  # duplicate 2, leaves a gap at 5
    bad = dataclasses.replace(measurements, jsonl_rows=rows)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "contiguity" in result.reason


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_seed():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    sidecars[3] = {**sidecars[3], "seed": sidecars[3]["seed"] + 1}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "seed" in result.reason


# ---------------------------------------------------------------------------
# matchup
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_matchup_identity():
    protocol, measurements = _conformant_reservoir()
    bad_identities = {**measurements.checkpoint_identities,
                      "reservoir_a": "tampered:deadbeef"}
    bad = dataclasses.replace(measurements, checkpoint_identities=bad_identities)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "matchup" in result.reason


def test_check_protocol_conformance_wrong_matchup_row_checkpoint():
    protocol, measurements = _conformant_reservoir()
    rows = [dict(r) for r in measurements.jsonl_rows]
    rows[0]["red_checkpoint"] = "checkpoints/some_other_model.safetensors"
    bad = dataclasses.replace(measurements, jsonl_rows=rows)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "matchup" in result.reason


# ---------------------------------------------------------------------------
# color_parity
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_color_parity():
    protocol, measurements = _conformant_reservoir()
    rows = [dict(r) for r in measurements.jsonl_rows]
    # game_idx=0 is even -> should be red=checkpoint_a; swap red/black so the
    # SET check (matchup) still passes but parity is now wrong.
    rows[0]["red_checkpoint"], rows[0]["black_checkpoint"] = (
        rows[0]["black_checkpoint"], rows[0]["red_checkpoint"])
    bad = dataclasses.replace(measurements, jsonl_rows=rows)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "color_parity" in result.reason


# ---------------------------------------------------------------------------
# replay_linkage
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_replay_linkage_missing_sidecar():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    orphan = dict(sidecars[3])
    del sidecars[3]
    orphan["game_idx"] = 99
    sidecars[99] = orphan  # keeps len(sidecars_by_idx) == games == 6
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "replay_linkage" in result.reason


def test_check_protocol_conformance_replay_linkage_game_idx_mismatch():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    sidecars[2] = {**sidecars[2], "game_idx": 99}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "replay_linkage" in result.reason


def test_check_protocol_conformance_replay_linkage_color_mismatch():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    sidecars[1] = {**sidecars[1],
                   "red_checkpoint": "checkpoints/unrelated.safetensors"}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "replay_linkage" in result.reason


def test_check_protocol_conformance_replay_linkage_sidecar_game_idx_none():
    """Hardening edge case: a sidecar with an explicitly-`None` `game_idx`
    (as opposed to a simply-missing key) must report a clean MISMATCH, not
    crash with a `TypeError` from `int(None)`."""
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    sidecars[2] = {**sidecars[2], "game_idx": None}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "replay_linkage" in result.reason


def test_check_protocol_conformance_replay_linkage_board_size_mismatch():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    sidecars[0] = {**sidecars[0], "board_size": 18}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "replay_linkage" in result.reason


# ---------------------------------------------------------------------------
# match_config -- the ten knobs + workers.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("knob,bad_value", [
    ("board_size", 18),
    ("mcts_sims", 111),
    ("mcts_eval_batch_size", 7),
    ("mcts_stall_flush_sims", 9),
    ("selection_mode", "argmax"),
    ("opening_temp_plies", 3),
    ("temp_high", 2.5),
    ("temp_low", 0.25),
    ("max_moves", 99),
    ("base_seed", 555),
])
def test_check_protocol_conformance_wrong_match_config_knob(knob, bad_value):
    """Every one of the ten result-determining match knobs is independently
    checked -- mirrors `test_build_protocol_rejects_each_individually_
    missing_key`'s per-field coverage style for B1."""
    protocol, measurements = _conformant_reservoir()
    config = dict(measurements.summary["config"])
    assert config[knob] != bad_value  # sanity: fixture's own value differs
    config[knob] = bad_value
    bad_summary = {**measurements.summary, "config": config}
    bad = dataclasses.replace(measurements, summary=bad_summary)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "match_config" in result.reason
    assert knob in result.reason


def test_check_protocol_conformance_wrong_workers():
    protocol, measurements = _conformant_reservoir()
    config = dict(measurements.summary["config"])
    config["workers"] = config["workers"] + 1
    bad_summary = {**measurements.summary, "config": config}
    bad = dataclasses.replace(measurements, summary=bad_summary)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "match_config" in result.reason
    assert "workers" in result.reason


# ---------------------------------------------------------------------------
# output_path -- source_index_path derivation + replay_dir.
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_source_index_path():
    protocol, measurements = _conformant_reservoir(
        source_index_path="runs/reservoir_v2/WRONG_games.jsonl")
    result = check_protocol_conformance(protocol, measurements)
    assert result.ok is False
    assert "output_path" in result.reason


def test_check_protocol_conformance_wrong_replay_dir():
    protocol, measurements = _conformant_reservoir()
    rows = [dict(r) for r in measurements.jsonl_rows]
    rows[4]["replay_path"] = "some/other/dir/game_000004.json"
    bad = dataclasses.replace(measurements, jsonl_rows=rows)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "output_path" in result.reason


# ---------------------------------------------------------------------------
# move_player_parity
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_move_player_parity():
    protocol, measurements = _conformant_reservoir()
    sidecars = dict(measurements.sidecars_by_idx)
    moves = [dict(m) for m in sidecars[2]["moves"]]
    moves[1] = {**moves[1], "player": "red"}  # ply=1 is odd -> should be black
    sidecars[2] = {**sidecars[2], "moves": moves}
    bad = dataclasses.replace(measurements, sidecars_by_idx=sidecars)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "move_player_parity" in result.reason


# ---------------------------------------------------------------------------
# generation_provenance
# ---------------------------------------------------------------------------

def test_check_protocol_conformance_wrong_generation_source_sha1s():
    protocol, measurements = _conformant_reservoir()
    bad_sources = {**measurements.generation_source_sha1s,
                   "eval_checkpoint_match.py": "tampered0000"}
    bad = dataclasses.replace(measurements, generation_source_sha1s=bad_sources)
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "generation_provenance" in result.reason


def test_check_protocol_conformance_wrong_generation_git_commit():
    protocol, measurements = _conformant_reservoir()
    bad = dataclasses.replace(measurements,
                              generation_git_commit="deadbeefdeadbeef")
    result = check_protocol_conformance(protocol, bad)
    assert result.ok is False
    assert "generation_provenance" in result.reason


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


def test_module_imports_only_v2_corpus_sources_from_fpu_dev_corpus_v2():
    """B3 narrows (not lifts) the B1 scope guard (plan Task B3 / spec Sec 6
    "import only the shared ... constant" seam): `fpu_dev_corpus_v2` IS now
    imported (for `_V2_CORPUS_SOURCES`, needed by `QUALIFICATION_SOURCE_
    FILES`), but ONLY that one name -- never `V2Config`, `run_screen`,
    `load_v2_config`, or a bare `import fpu_dev_corpus_v2` that would pull
    in its whole surface. This is what keeps the Sec 6 circular-import risk
    one-directional: `fpu_dev_corpus_v2.run_screen` -> (lazily, a LATER
    task, B9) this module -> `fpu_dev_corpus_v2` (top-level, THIS import,
    already-proven-import-pure) is not a cycle, since it never imports
    `run_screen` back.

    Parsed via `ast` (not a raw substring check) so a legitimate prose
    mention of the module's name -- e.g. this file's own docstring naming
    `fpu_dev_corpus_v2_config.json`, the artifact -- can never
    false-positive; only real `import`/`from ... import` nodes count. The
    module's own `__file__` (via `inspect`, on an object already imported
    at the top of this file) locates the source -- an ABSOLUTE path, never
    a cwd-relative guess."""
    import ast
    import inspect

    module_path = Path(inspect.getfile(build_protocol))
    tree = ast.parse(module_path.read_text())
    whole_module_imports = set()
    from_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            whole_module_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "fpu_dev_corpus_v2" in node.module:
                from_imports.update(alias.name for alias in node.names)
    assert not any("fpu_dev_corpus_v2" in m for m in whole_module_imports), (
        whole_module_imports)
    assert from_imports == {"_V2_CORPUS_SOURCES"}, from_imports
