#!/usr/bin/env python3
"""V3, research side: the validated protocol's evaluator on the student, over the Phase 2 corpus.

    MLX_ENABLE_TF32=0 .venv/bin/python tests/parity/htb_mirror/python_side.py <out.json>

The student is loaded by the H-TB runs' own loader (eval_runner._default_evaluator_factory: LocalGPUEvaluator,
compile=True, eval mode), wrapped in mirror_averaging.MirrorAveragingEvaluator. Each position goes through infer()
exactly as MCTS._expand_batch feeds it (B=1, int32 move arrays, float32 mask). The state digest and eval mode are
checked at load and at the end. No gates here: htb_mirror/compare.mjs judges.
"""
import hashlib, json, pathlib, platform, sys
from importlib.metadata import version

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import eval_runner as ER, htb_pilot as H, search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero.game.twixt_state import TwixtState  # noqa: E402
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator  # noqa: E402

STUDENT = pathlib.Path.home() / "twixt-cache/htb-pilot/student.safetensors"
STUDENT_SHA256 = "0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5"
STUDENT_DIGEST = "586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b"


def main():
    out = pathlib.Path(sys.argv[1])
    sha = hashlib.sha256(STUDENT.read_bytes()).hexdigest()
    assert sha == STUDENT_SHA256, sha
    base = ER._default_evaluator_factory(str(STUDENT))
    assert H._digest(base.network) == STUDENT_DIGEST
    M._require_eval(base.network, "load")
    ev = MirrorAveragingEvaluator(base)
    corpus = json.loads((ROOT / "tests/parity/corpus.json").read_text())
    positions = []
    for p in corpus["primary"] + corpus["edge"]:
        state = TwixtState.from_moves([tuple(m) for m in p["moves"]])
        legal = state.legal_moves()
        n = len(legal)
        board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
        rows = np.array([[r for r, _ in legal]], np.int32)
        cols = np.array([[c for _, c in legal]], np.int32)
        mask = np.ones((1, n), np.float32)
        pri, val = ev.infer(board, rows, cols, mask, 24)
        positions.append({"id": p["id"], "stratum": p["stratum"], "to_move": state.to_move, "n_legal": n,
                          "legal_moves": [[int(r), int(c)] for r, c in legal],
                          "priors": [float(x) for x in pri[0, :n]], "value": float(val[0])})
    assert H._digest(base.network) == STUDENT_DIGEST
    M._require_eval(base.network, "end")
    report = {"side": "research MirrorAveragingEvaluator(student)", "student_sha256": sha,
              "state_digest": STUDENT_DIGEST, "environment": {"python": platform.python_version(),
              "mlx": version("mlx"), "numpy": np.__version__}, "positions": positions}
    out.write_text(json.dumps(report))
    print(f"wrote {out} ({len(positions)} positions)")


if __name__ == "__main__":
    main()
