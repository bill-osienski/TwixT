#!/usr/bin/env python3
"""Python half of the Phase 2 parity measurement.

    .venv/bin/python tests/parity/run_python_side.py <model_dir> <out.json>

Produces, for every corpus position: the Python state encoding (as a hash of
its exact float32 bytes), the legal-move list, and the policy/value outputs of
BOTH native MLX and Python ONNX Runtime.

It applies no gates and reaches no verdict. Comparison and the pass/fail rule
live in compare.mjs, so that neither half of the measurement can decide its own
result.

Specification: docs/superpowers/2026-08-13-phase2-parity-specification.md
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402

from scripts.GPU.alphazero.game.twixt_state import TwixtState, NUM_CHANNELS  # noqa: E402
from scripts.GPU.alphazero.network import create_network  # noqa: E402

MAX_MOVES = 576
BOARD_SIZE = 24
# Shared synthetic side-to-move values for the value-perspective check. Fixed,
# model-independent, and identical on both sides, so the comparison isolates the
# conversion logic instead of re-measuring the S3 model difference.
PERSPECTIVE_PROBES = [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0]
# Positions given the move-order equivariance check, and the fixed permutation
# seed used to shuffle their move lists.
EQUIVARIANCE_POSITIONS = 10
EQUIVARIANCE_SEED = 20260813


def tensor_hash(arr: np.ndarray) -> str:
    """SHA-256 over canonical little-endian float32 bytes in C order."""
    canonical = np.ascontiguousarray(arr, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def pad_moves(moves):
    rows = np.zeros(MAX_MOVES, dtype=np.int64)
    cols = np.zeros(MAX_MOVES, dtype=np.int64)
    mask = np.zeros(MAX_MOVES, dtype=np.float32)
    for i, (r, c) in enumerate(moves[:MAX_MOVES]):
        rows[i], cols[i], mask[i] = r, c, 1.0
    return rows, cols, mask


def red_perspective(value: float, to_move: str) -> float:
    """Independent implementation of the server's red-perspective conversion.

    Written from the rule, not copied from the JavaScript: the value is from the
    side to move, so it is negated when black is to move, then clamped.
    """
    red = value if to_move == "red" else -value
    return max(-1.0, min(1.0, red))


def deterministic_permutation(n: int, seed: int) -> list[int]:
    """Fisher-Yates under a fixed PRNG, so both halves permute identically."""
    idx = list(range(n))
    state = seed & 0xFFFFFFFF
    for i in range(n - 1, 0, -1):
        # mulberry32, matching tests/parity/generate_corpus.mjs
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (state ^ (state >> 15)) * (1 | state) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        r = ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
        j = int(r * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_python_side.py <model_dir> <out.json>", file=sys.stderr)
        return 2
    model_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2])

    corpus = json.loads((PROJECT_ROOT / "tests/parity/corpus.json").read_text())
    positions = corpus["primary"] + corpus["edge"]

    manifest = json.loads((model_dir / "manifest.json").read_text())
    graph_path = model_dir / manifest["graph"]["filename"]

    # Native MLX, from the source checkpoint named by the manifest.
    ckpt = PROJECT_ROOT / manifest["provenance"]["source_checkpoint_path"]
    net = create_network(hidden=128, n_blocks=6)
    net.load_weights(str(ckpt))
    net.eval()

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])

    results = []
    for pos in positions:
        moves = [tuple(m) for m in pos["moves"]]
        state = TwixtState.from_moves(moves)
        legal = [tuple(m) for m in state.legal_moves()]
        n_legal = len(legal)

        chw = np.ascontiguousarray(state.to_tensor(), dtype=np.float32)
        nchw = chw[None, ...]

        # --- native MLX (channels-last) ---
        hwc = np.transpose(chw, (1, 2, 0))
        mlx_policy, mlx_value = net(mx.array(hwc[None, ...]), legal)
        mx.eval(mlx_policy, mlx_value)
        mlx_logits = np.array(mlx_policy, dtype=np.float64).reshape(-1)[:n_legal]
        mlx_val = float(np.array(mlx_value).reshape(-1)[0])

        # --- Python ONNX Runtime ---
        rows, cols, mask = pad_moves(legal)
        ort_out = session.run(
            ["policy_logits", "value"],
            {"board": nchw, "move_rows": rows, "move_cols": cols, "move_mask": mask},
        )
        ort_full = np.array(ort_out[0], dtype=np.float64).reshape(-1)
        ort_logits = ort_full[:n_legal]
        ort_val = float(np.array(ort_out[1]).reshape(-1)[0])

        entry = {
            "id": pos["id"],
            "stratum": pos["stratum"],
            "ply": pos["ply"],
            "to_move": state.to_move,
            "n_legal": n_legal,
            "legal_moves": [list(m) for m in legal],
            "encoding_sha256": tensor_hash(nchw),
            "encoding_shape": list(nchw.shape),
            "mlx": {"logits": mlx_logits.tolist(), "value": mlx_val},
            "ort_py": {
                "logits": ort_logits.tolist(),
                "value": ort_val,
                # Masked tail, checked exactly rather than approximately.
                "mask_tail_all_neg1e9": bool(np.all(ort_full[n_legal:] == -1e9))
                if n_legal < MAX_MOVES
                else True,
                "mask_tail_count": int(MAX_MOVES - n_legal),
            },
            "red_perspective_probes": [red_perspective(v, state.to_move) for v in PERSPECTIVE_PROBES],
        }
        results.append(entry)

    # --- move-order equivariance on the first N primary positions ---
    #
    # BOTH Python surfaces are permuted, native MLX included. The specification
    # assigns this exact gate to S2, S3 and S4, and MLX is the reference
    # endpoint for S2 and S4 -- measuring only ONNX Runtime would leave the
    # reference side of two surfaces unmeasured while still reporting them.
    equivariance = []
    for pos in corpus["primary"][:EQUIVARIANCE_POSITIONS]:
        state = TwixtState.from_moves([tuple(m) for m in pos["moves"]])
        legal = [tuple(m) for m in state.legal_moves()]
        n_legal = len(legal)
        perm = deterministic_permutation(n_legal, EQUIVARIANCE_SEED)
        permuted = [legal[i] for i in perm]

        chw = np.ascontiguousarray(state.to_tensor(), dtype=np.float32)
        nchw = chw[None, ...]

        rows, cols, mask = pad_moves(permuted)
        out = session.run(
            ["policy_logits", "value"],
            {"board": nchw, "move_rows": rows, "move_cols": cols, "move_mask": mask},
        )
        permuted_logits = np.array(out[0], dtype=np.float64).reshape(-1)[:n_legal]

        hwc = np.transpose(chw, (1, 2, 0))
        mlx_perm_policy, _ = net(mx.array(hwc[None, ...]), permuted)
        mx.eval(mlx_perm_policy)
        mlx_permuted_logits = np.array(mlx_perm_policy, dtype=np.float64).reshape(-1)[:n_legal]

        equivariance.append(
            {
                "id": pos["id"],
                "permutation": perm,
                "permuted_logits": permuted_logits.tolist(),
                "mlx_permuted_logits": mlx_permuted_logits.tolist(),
            }
        )

    model = onnx.load(str(graph_path), load_external_data=False)
    payload = {
        "schema": "twixt-parity-side/1",
        "side": "python",
        "specification": "docs/superpowers/2026-08-13-phase2-parity-specification.md",
        "corpus_sha256": hashlib.sha256(
            (PROJECT_ROOT / "tests/parity/corpus.json").read_bytes()
        ).hexdigest(),
        "model_dir": str(model_dir),
        "model_id": manifest["model_id"],
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "scratch_export_path": os.environ.get("PHASE2_SCRATCH_EXPORT", "(not recorded)"),
        "source_checkpoint_sha1": manifest["provenance"]["source_checkpoint_sha1"],
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system().lower()} {platform.machine()}",
            "mlx": version("mlx"),
            "mlx_metal": version("mlx-metal"),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "onnx_producer": f"{model.producer_name} {model.producer_version}",
            "onnx_opset": model.opset_import[0].version,
            "ort_providers": session.get_providers(),
        },
        "constants": {
            "num_channels": NUM_CHANNELS,
            "board_size": BOARD_SIZE,
            "max_moves": MAX_MOVES,
            "perspective_probes": PERSPECTIVE_PROBES,
            "equivariance_positions": EQUIVARIANCE_POSITIONS,
            "equivariance_seed": EQUIVARIANCE_SEED,
        },
        "positions": results,
        "equivariance": equivariance,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path} ({len(results)} positions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
