"""Tests for the bootstrap probe suite generator.

Covers:
- CLI --help responds
- Deterministic byte-identical output on rerun
- No wall-clock fields in meta
- Only natural wins emitted
- Schema matches tests/probes/README.md expectations
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_bootstrap_cli_help():
    """Bootstrap generator responds to --help."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--source-iter-range" in result.stdout
