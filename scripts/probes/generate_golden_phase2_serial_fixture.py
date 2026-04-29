"""One-off generator for tests/probes/golden/phase2_serial_tiny_*.json.

Run once after each intentional change to the serial Phase 2 output schema.
Commit the generated files. NOT in CI.

Usage:
    .venv/bin/python scripts/probes/generate_golden_phase2_serial_fixture.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from scripts.GPU.alphazero import probe_eval
    from scripts.build_probe_suite import _build_arg_parser, _run_strong_advantage
    from tests.test_probe_phase2_parallel import _SAMPLE_CENTRAL, _SAMPLE_EDGE

    out_dir = repo_root / "tests" / "probes" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    class _FakeNet:
        def eval(self):
            return self

    def stub(state, sims, seed):
        v = 0.6 + 0.001 * (seed % 7)
        t1 = 0.4 + 0.001 * (seed % 5)
        return (v, t1)

    def fake_extract(games, **kw):
        return [_SAMPLE_CENTRAL, _SAMPLE_EDGE], []

    target = out_dir / "phase2_serial_tiny.json"
    fake_ckpt = out_dir / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    try:
        with patch.object(probe_eval, "extract_strong_advantage_candidates", fake_extract), \
             patch.object(probe_eval, "_default_mcts_labeler", stub), \
             patch.object(probe_eval, "load_network_for_scoring", lambda p: (_FakeNet(), 30, 128, 6)), \
             patch.object(probe_eval, "_set_default_labeler_network", lambda *a, **kw: None), \
             patch.object(probe_eval, "_set_default_labeler_mcts_config", lambda *a, **kw: None):

            ap = _build_arg_parser()
            args = ap.parse_args([
                "--tier", "strong_advantage",
                "--input", "scripts/GPU/logs/games",  # ignored, Phase 1 mocked
                "--source-iter-range", "70", "70",
                "--label-checkpoint", str(fake_ckpt),
                "--label-mcts-sims", "10",
                "--label-mcts-repeats", "2",
                "--magnitude-threshold", "0.45",
                "--out", str(target),
                "--force",
            ])
            args.label_workers_requested = args.label_workers
            rc = _run_strong_advantage(args)
            assert rc == 0
    finally:
        if fake_ckpt.exists():
            fake_ckpt.unlink()

    draft = target.with_suffix(".draft.json")
    audit = target.parent / "candidates_strong_advantage.json"
    final_draft = out_dir / "phase2_serial_tiny_expected.draft.json"
    final_audit = out_dir / "phase2_serial_tiny_expected.audit.json"
    shutil.move(str(draft), str(final_draft))
    shutil.move(str(audit), str(final_audit))
    print(f"Golden fixtures written:\n  {final_draft}\n  {final_audit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
