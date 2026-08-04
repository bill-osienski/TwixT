"""Atlas Stage 2 -- CLI: staged chronology, validation, operator stops."""
import json
import subprocess
import sys
from pathlib import Path

BASE = 20400000


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.build_atlas_corpus", *args],
        capture_output=True, text=True, check=False,
    )


def _ckpt(tmp_path, content=b"weights"):
    p = Path(tmp_path) / "ck.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _block(tmp_path, start_index, n_games, n_moves=200, ck_sha=None):
    """A synthetic but VALID block: production manifest plus seed-consistent
    sidecars. The production fields are mandatory -- load_block rejects a block
    generated under smoke settings."""
    d = Path(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "block_manifest.json").write_text(json.dumps({
        "base_seed": BASE, "start_index": start_index, "n_games": n_games,
        "active_size": 24, "n_simulations": 400, "max_moves": 280,
        "batching": [14, 48, 8], "add_noise": True,
        "checkpoint_path": "x", "checkpoint_sha1": ck_sha or ("0" * 40),
        "git_head": "deadbeef", "worktree_clean": True,
    }))
    for i in range(start_index, start_index + n_games):
        (d / f"game_{i:06d}.json").write_text(json.dumps({
            "game_idx": i, "seed": BASE + i, "n_moves": n_moves,
            "start_player": "red" if i % 2 == 0 else "black",
        }))
    return str(d)


def test_emit_protocol_freezes_the_checkpoint_and_needs_no_N(tmp_path):
    ck = _ckpt(tmp_path)
    r = _run("emit-protocol", "--base-seed", str(BASE), "--sampling-seed",
             "20260804", "--checkpoint", str(ck))
    assert r.returncode == 0, r.stderr
    p = json.loads(r.stdout)
    assert p["checkpoint_path"] == str(ck) and len(p["checkpoint_sha1"]) == 40
    assert p["max_seed_range_games"] == 480
    assert p["seed_range"] == [BASE, BASE + 480]
    assert p["pilot_games"] == 24
    assert "n_target" not in p          # N does not exist before the pilot ladder


def test_emit_pilot_command_is_pre_pilot_and_needs_no_N(tmp_path):
    ck = _ckpt(tmp_path)
    r = _run("emit-pilot-command", "--base-seed", str(BASE),
             "--sampling-seed", "1", "--checkpoint", str(ck), "--out-dir", "root")
    assert r.returncode == 0, r.stderr
    assert "OPERATOR STOP" in r.stdout and "NOT AUTHORIZED" in r.stdout
    assert "--start-index 0 --n-games 24" in r.stdout
    assert "root/pilot" in r.stdout              # its OWN directory
    assert "--n-target" not in r.stdout


def test_continuation_command_consumes_the_REAL_size_output(tmp_path):
    """Drive the real `size` subcommand into the real consumer. Hand-writing the
    artifact here would repeat the surrogate mistake this project has paid for
    twice."""
    ck = _ckpt(tmp_path)
    import hashlib
    sha = hashlib.sha1(ck.read_bytes()).hexdigest()
    pilot = _block(tmp_path / "p", 0, 24, ck_sha=sha)
    sz = _run("size", "--sidecar-dir", pilot, "--base-seed", str(BASE),
              "--n-target", "240")
    assert sz.returncode == 0, sz.stderr
    art = tmp_path / "size.json"
    art.write_text(sz.stdout)
    assert json.loads(sz.stdout)["G_total"] == 280      # the formula's answer

    r = _run("emit-continuation-command", "--base-seed", str(BASE),
             "--pilot-dir", pilot, "--n-target", "240",
             "--size-artifact", str(art), "--checkpoint", str(ck),
             "--out-dir", "root")
    assert r.returncode == 0, r.stderr
    assert "--start-index 24 --n-games 256" in r.stdout     # 280 - 24
    assert "root/continuation" in r.stdout and "root/pilot" not in r.stdout


def test_continuation_command_rejects_a_tampered_size_artifact(tmp_path):
    """A hand-edited G_total must not authorize the expensive block."""
    ck = _ckpt(tmp_path)
    import hashlib
    sha = hashlib.sha1(ck.read_bytes()).hexdigest()
    pilot = _block(tmp_path / "p2", 0, 24, ck_sha=sha)
    art = tmp_path / "tampered.json"
    art.write_text(json.dumps({"verdict": "OK", "G_total": 480, "g_cont": 456}))
    r = _run("emit-continuation-command", "--base-seed", str(BASE),
             "--pilot-dir", pilot, "--n-target", "240",
             "--size-artifact", str(art), "--checkpoint", str(ck),
             "--out-dir", "root")
    assert r.returncode == 2
    assert "does not match a recomputation" in r.stderr


def test_continuation_command_rejects_a_mismatched_checkpoint(tmp_path):
    ck = _ckpt(tmp_path)
    other = Path(tmp_path) / "other.safetensors"
    other.write_bytes(b"different weights")
    pilot = _block(tmp_path / "p6", 0, 24, ck_sha="0" * 40)   # pilot used another
    sz = _run("size", "--sidecar-dir", pilot, "--base-seed", str(BASE),
              "--n-target", "240")
    art = tmp_path / "sz6.json"
    art.write_text(sz.stdout)
    r = _run("emit-continuation-command", "--base-seed", str(BASE),
             "--pilot-dir", pilot, "--n-target", "240",
             "--size-artifact", str(art), "--checkpoint", str(ck),
             "--out-dir", "root")
    assert r.returncode == 2 and "does not match the pilot" in r.stderr


def test_pilot_gate_passes_and_no_gos_with_exit_3(tmp_path):
    ok = _block(tmp_path / "pg", 0, 24)
    r = _run("pilot-gate", "--sidecar-dir", ok, "--base-seed", str(BASE),
             "--sampling-seed", "7")
    assert r.returncode == 0 and json.loads(r.stdout)["verdict"] == "PASS"

    short = _block(tmp_path / "pg2", 0, 24, n_moves=60)
    r2 = _run("pilot-gate", "--sidecar-dir", short, "--base-seed", str(BASE),
              "--sampling-seed", "7")
    assert r2.returncode == 3
    assert json.loads(r2.stdout)["verdict"] == "PHASE_GEOMETRY_NO_GO"


def test_cli_rejects_a_block_with_a_tampered_seed(tmp_path):
    d = Path(_block(tmp_path / "t", 0, 24))
    bad = json.loads((d / "game_000005.json").read_text())
    bad["seed"] = BASE + 9999
    (d / "game_000005.json").write_text(json.dumps(bad))
    r = _run("pilot-gate", "--sidecar-dir", str(d), "--base-seed", str(BASE),
             "--sampling-seed", "7")
    assert r.returncode != 0
    assert "seed" in (r.stderr + r.stdout).lower()


def test_size_reports_g_total_and_binding_subset(tmp_path):
    d = _block(tmp_path / "sz", 0, 24)
    r = _run("size", "--sidecar-dir", d, "--base-seed", str(BASE), "--n-target", "240")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["verdict"] == "OK" and out["G_total"] <= 480 and out["binding_subset"]


def test_assign_succeeds_at_the_AUTHORIZED_continuation_size(tmp_path):
    """At N=200 with a full-coverage pilot the frozen rule gives G_total=240, so
    the continuation is exactly 216 games. The demand is 176; the 40-game slack
    IS the 20% margin, not spare capacity to draw on."""
    pilot = _block(tmp_path / "p", 0, 24)
    cont = _block(tmp_path / "c", 24, 216)
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", cont,
             "--base-seed", str(BASE), "--n-target", "200", "--sampling-seed", "7")
    assert r.returncode == 0, r.stderr
    assert len(json.loads(r.stdout)["rows"]) == 176


def test_assign_rejects_an_oversized_continuation_as_a_top_up(tmp_path):
    """400 games where the rule authorizes 216 is an unauthorized top-up: the
    surplus becomes matching capacity the sizing rule never granted."""
    pilot = _block(tmp_path / "p3", 0, 24)
    cont = _block(tmp_path / "c3", 24, 400)
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", cont,
             "--base-seed", str(BASE), "--n-target", "200", "--sampling-seed", "7")
    assert r.returncode == 2
    assert "requires exactly [24, 240)" in r.stderr


def test_assign_shortfall_exits_4_with_no_partial_corpus(tmp_path):
    pilot = _block(tmp_path / "p4", 0, 24)
    short = _block(tmp_path / "c4", 24, 216, n_moves=50)
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", short,
             "--base-seed", str(BASE), "--n-target", "200", "--sampling-seed", "7")
    out = json.loads(r.stdout)
    assert r.returncode == 4 and out["verdict"] == "ASSIGNMENT_SHORTFALL"
    assert "rows" not in out and out["min_cut_cells"]


def test_assign_rejects_blocks_from_different_checkpoints(tmp_path):
    pilot = _block(tmp_path / "p5", 0, 24)
    cont = Path(_block(tmp_path / "c5", 24, 216, ck_sha="1" * 40))
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", str(cont),
             "--base-seed", str(BASE), "--n-target", "200", "--sampling-seed", "7")
    assert r.returncode == 2 and "checkpoint_sha1" in r.stderr


def test_cli_refuses_a_disallowed_n(tmp_path):
    d = _block(tmp_path / "n", 0, 24)
    r = _run("size", "--sidecar-dir", d, "--base-seed", str(BASE), "--n-target", "250")
    assert r.returncode != 0
