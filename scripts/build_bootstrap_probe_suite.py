"""Backward-compatibility shim.

The real implementation now lives in scripts/build_probe_suite.py. This
shim preserves the existing CLI/cron invocation
(`build_bootstrap_probe_suite.py --source-iter-range MIN MAX`) by injecting
`--tier forced` and forwarding to the new entrypoint.

DO NOT add new flags here. Add them to build_probe_suite.py instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    real = Path(__file__).resolve().parent / "build_probe_suite.py"
    args = [sys.executable, str(real), "--tier", "forced", *sys.argv[1:]]
    import os
    os.execv(sys.executable, args)
