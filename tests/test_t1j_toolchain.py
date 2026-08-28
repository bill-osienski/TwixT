"""Toolchain path resolution and hash verification. NO EXECUTION.

Nothing here starts a JVM, loads a model, draws a seed or queries T1j. It
resolves a filesystem root and hashes files.

The rule this module exists to enforce: the local root comes from an EXPLICIT
setting, is never under /tmp, and is never silently substituted. The previous
arrangement pinned test fixtures at a path inside a session scratchpad, which
was wiped, taking the whole qualification suite red with it.
"""
import hashlib
import json

import pytest

from scripts.GPU.alphazero import t1j_toolchain as TC
from scripts.GPU.alphazero.e4_screen_integration import PINNED_JDK


def _fake_root(tmp_path, jar_bytes=b"JAR", rel_bytes=b"REL"):
    (tmp_path / "jdk").mkdir()
    (tmp_path / "a.jar").write_bytes(jar_bytes)
    (tmp_path / "jdk" / "release").write_bytes(rel_bytes)
    lock = {"layout": {"jar": "a.jar", "jdk_home": "jdk"},
            "jar": {"sha256": hashlib.sha256(b"JAR").hexdigest(), "bytes": 3},
            "jdk": {"components": {"release": hashlib.sha256(b"REL").hexdigest()}}}
    return str(tmp_path), lock


# ------------------------------------------------------------ root resolution

def test_an_explicit_root_wins_and_its_source_is_reported(tmp_path):
    root, _ = _fake_root(tmp_path)
    resolved, source = TC.resolve_root(root)
    assert resolved == root and source == "explicit"


def test_the_environment_variable_is_used_when_no_explicit_root(tmp_path, monkeypatch):
    root, _ = _fake_root(tmp_path)
    monkeypatch.setenv(TC.ENV_VAR, root)
    assert TC.resolve_root() == (root, "environment")


def test_the_recorded_default_is_named_not_silent(monkeypatch):
    """A default is acceptable only if it announces itself; the ban is on a
    SILENT substitution, not on having a documented location."""
    monkeypatch.delenv(TC.ENV_VAR, raising=False)
    resolved, source = TC.resolve_root()
    assert source == "recorded-default"
    assert resolved == TC.DEFAULT_ROOT


@pytest.mark.parametrize("bad", ["/tmp/x", "/private/tmp/x", "/tmp"])
def test_a_tmp_root_is_refused_however_it_arrives(bad, monkeypatch):
    with pytest.raises(TC.ToolchainError, match="ephemeral"):
        TC.resolve_root(bad)
    monkeypatch.setenv(TC.ENV_VAR, bad)
    with pytest.raises(TC.ToolchainError, match="ephemeral"):
        TC.resolve_root()


def test_a_missing_root_is_refused_rather_than_substituted(tmp_path):
    with pytest.raises(TC.ToolchainError, match="does not exist"):
        TC.resolve_root(str(tmp_path / "nope"))


# ------------------------------------------------- hash verification before use

def test_verified_paths_returns_paths_only_after_checking_hashes(tmp_path):
    root, lock = _fake_root(tmp_path)
    got = TC.verified_paths(root=root, lock=lock)
    assert got["jar"].endswith("a.jar") and got["jdk_home"].endswith("jdk")
    assert got["source"] == "explicit" and got["verified"] == 2


def test_a_tampered_jar_is_refused(tmp_path):
    root, lock = _fake_root(tmp_path, jar_bytes=b"XXX")
    with pytest.raises(TC.ToolchainError, match="sha256"):
        TC.verified_paths(root=root, lock=lock)


def test_a_tampered_jdk_component_is_refused(tmp_path):
    root, lock = _fake_root(tmp_path, rel_bytes=b"XXX")
    with pytest.raises(TC.ToolchainError, match="sha256"):
        TC.verified_paths(root=root, lock=lock)


def test_a_missing_component_is_refused(tmp_path):
    root, lock = _fake_root(tmp_path)
    (tmp_path / "jdk" / "release").unlink()
    with pytest.raises(TC.ToolchainError, match="missing"):
        TC.verified_paths(root=root, lock=lock)


def test_a_wrong_size_jar_is_refused_even_if_the_hash_field_is_absent(tmp_path):
    root, lock = _fake_root(tmp_path)
    lock["jar"]["bytes"] = 999
    with pytest.raises(TC.ToolchainError, match="bytes"):
        TC.verified_paths(root=root, lock=lock)


# ------------------------------------------------ the lock agrees with the code

def test_the_lock_does_not_duplicate_the_pinned_jdk_hashes_by_hand():
    assert TC.load_lock()["jdk"]["components"] == PINNED_JDK


def test_the_lock_jar_hash_matches_what_e1_recorded():
    import pathlib, re
    e1 = pathlib.Path("docs/superpowers/2026-08-24-t1j-e1-artifact-integrity.md").read_text()
    assert TC.load_lock()["jar"]["sha256"] == re.search(r'\| sha256 \| `([0-9a-f]{64})` \|', e1).group(1)


def test_the_lock_records_sources_and_layout():
    lock = TC.load_lock()
    assert lock["jar"]["url"].startswith("https://github.com/johannesSchwagereit/T1j")
    assert "adoptium" in lock["jdk"]["url"] and lock["jdk"]["version"] == "17.0.20.1+1"
    assert set(lock["layout"]) == {"jar", "jdk_home"}


# --------------------------------------------- the REAL, recovered toolchain

def test_the_recovered_toolchain_verifies_against_the_lock():
    got = TC.verified_paths()
    assert got["verified"] == 1 + len(PINNED_JDK) == 5
    assert "/private/tmp" not in got["root"] and "/tmp/" not in got["root"]
