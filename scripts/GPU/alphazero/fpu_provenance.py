"""Pure, stdlib-only provenance helpers for the FPU staged diagnostic and the
dev-corpus builder (design §12.5).

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md

=============================================================================
NO MCTS / evaluator / GPU / MLX / numpy import here. `importlib.metadata.version
("mlx")` reads installed-package METADATA and never imports mlx, so this module
-- and every module that imports it -- stays GPU/MLX-free at import time
(verified: importing it leaves `mlx` out of `sys.modules`).
=============================================================================

Every result-determining input that a git commit + a replay PATH would MISS is
fingerprinted here: source-file BYTES (uncommitted edits), replay-DATA bytes
(contents, not paths), a clean-worktree flag, and runtime identity. These feed
the SHARED selection-context fingerprint (hard-matched across stages) and the
RECORDED run-context fingerprint (git/worktree/runtime provenance).
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata as _metadata
from pathlib import Path
from typing import Dict, Iterable, Optional


def file_sha1(path: Optional[str]) -> str:
    """Streaming SHA1 of a file's bytes. Returns the sentinel ``"none"`` for an
    absent path and ``"missing"`` for an unreadable one, so a fingerprint records
    WHICH input was absent rather than crashing (identical semantics to the
    diagnostic's former private `_file_sha1`)."""
    if not path:
        return "none"
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "missing"


def source_file_sha1s(paths: Iterable[str]) -> Dict[str, str]:
    """``{basename: sha1}`` over each effective result-determining source file.
    Keyed by basename (checkout-location-independent; the callers' inputs have
    distinct basenames), so the shared selection-context matches across stages
    regardless of the absolute repo path."""
    return {Path(p).name: file_sha1(str(p)) for p in paths}


def replay_data_sha1(replay_paths: Iterable[str]) -> str:
    """A single deterministic SHA1 over the CONTENTS of the replay files, taken
    in sorted-by-path order (so it is order-independent-by-path) and sensitive to
    every byte -- it fingerprints the replay DATA, not the paths. Empty input
    hashes to the SHA1 of nothing (a stable, well-defined value)."""
    h = hashlib.sha1()
    for p in sorted(str(x) for x in replay_paths):
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """``git rev-parse HEAD``, or ``"unknown"`` on any failure (the logic the
    diagnostic's former private `_git_commit` used)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def worktree_clean() -> bool:
    """True iff ``git status --porcelain`` is empty (no staged / unstaged /
    untracked changes). A git commit alone cannot detect uncommitted edits; this
    flag does. Any git failure -> False (conservative: not known-clean)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        return out.strip() == ""
    except Exception:
        return False


def runtime_provenance() -> Dict[str, Optional[str]]:
    """Interpreter / package / platform identity (RECORDED, not a hard equality
    gate -- belongs to the run-context fingerprint). Keys: ``python_version``,
    ``mlx_version`` (None if mlx is not installed -- read from package METADATA,
    NOT via an mlx import), ``platform``, ``machine``."""
    try:
        mlx_version: Optional[str] = _metadata.version("mlx")
    except Exception:
        mlx_version = None
    return {
        "python_version": sys.version,
        "mlx_version": mlx_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
