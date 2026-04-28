"""Parity guard: regenerating --tier forced must produce a byte-identical
output to the committed tests/probes/twixt_probes.json. This is the safety
gate for the build_probe_suite.py refactor — if it fails after a refactor,
the refactor is wrong.

Reads selection_rules from the committed file's meta block, so the test
follows whatever args the committed suite used (not pinned to a literal
iter range).

Assumed stable inputs: scripts/GPU/logs/games/iter_NNNN_game_MMM.json for
the iter range in meta.selection_rules.source_iter_range. If those are
moved/edited, this test will fail and the committed suite must be
regenerated against the new replay set with a deliberate commit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_SUITE = PROJECT_ROOT / "tests" / "probes" / "twixt_probes.json"


def test_tier_forced_byte_identical_to_committed_suite(tmp_path):
    committed_bytes = COMMITTED_SUITE.read_bytes()
    meta = json.loads(committed_bytes)["meta"]
    rules = meta["selection_rules"]
    src_min, src_max = rules["source_iter_range"]

    out_path = tmp_path / "regenerated_twixt_probes.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_probe_suite.py"),
        "--tier", "forced",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", str(src_min), str(src_max),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"build_probe_suite.py exited {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    regenerated_bytes = out_path.read_bytes()
    if regenerated_bytes != committed_bytes:
        # Surface the first diff for debuggability.
        from difflib import unified_diff
        diff = "\n".join(unified_diff(
            committed_bytes.decode().splitlines(),
            regenerated_bytes.decode().splitlines(),
            fromfile="committed/twixt_probes.json",
            tofile="regenerated/twixt_probes.json",
            lineterm="",
            n=3,
        ))
        raise AssertionError(
            "Regenerated forced suite differs from committed suite.\n"
            "First 50 lines of diff:\n"
            + "\n".join(diff.splitlines()[:50])
        )
