"""_git_provenance against purpose-built repositories.

These never consult the ambient worktree, so they cannot silently skip and
cannot pass for the wrong reason.
"""
import pathlib
import subprocess

import pytest

from scripts.GPU.alphazero.eval_readout_match import _git_provenance, _sha1


def _init_repo(path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=path, check=True)
    return str(path)


@pytest.fixture
def clean_repo(tmp_path):
    return _init_repo(tmp_path / "clean_repo")


@pytest.fixture
def dirty_repo(tmp_path):
    path = tmp_path / "dirty_repo"
    _init_repo(path)
    (path / "uncommitted.txt").write_text("dirty\n")
    return str(path)


def test_clean_repo_yields_a_commit_and_no_null_fields(clean_repo):
    p = _git_provenance(clean_repo)
    assert len(p["git_commit"]) == 40
    assert p["worktree_clean"] is True


def test_dirty_repo_is_refused(dirty_repo):
    with pytest.raises(RuntimeError, match="dirty"):
        _git_provenance(dirty_repo)


def test_untracked_file_counts_as_dirty(clean_repo):
    # CONSTRUCTED: an untracked file is a real difference in what ran.
    (pathlib.Path(clean_repo) / "new.txt").write_text("x\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _git_provenance(clean_repo)


def test_a_modified_tracked_file_counts_as_dirty(clean_repo):
    (pathlib.Path(clean_repo) / "seed.txt").write_text("changed\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _git_provenance(clean_repo)


def test_a_non_repository_raises_rather_than_recording_null(tmp_path):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="provenance"):
        _git_provenance(str(plain))


def test_sha1_hashes_a_real_file(tmp_path):
    f = tmp_path / "ckpt.bin"
    f.write_bytes(b"abc")
    # sha1("abc") is a fixed, externally checkable value.
    assert _sha1(str(f)) == "a9993e364706816aba3e25717850c26c9cd0d89d"


def test_sha1_raises_on_an_unreadable_file(tmp_path):
    with pytest.raises(RuntimeError, match="hash checkpoint"):
        _sha1(str(tmp_path / "missing.bin"))
