"""End-to-end JS/Python tensor parity for 30-channel input (Phase 2)."""
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.GPU.alphazero.game.twixt_state import TwixtState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tests" / "js_parity_runner.mjs"


def _run_js(active_size, moves):
    """Invoke the Node runner and return the flattened tensor as a numpy array."""
    result = subprocess.run(
        [
            "node",
            str(RUNNER),
            "--active-size",
            str(active_size),
            "--moves",
            json.dumps(moves),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"JS runner failed (rc={result.returncode}): {result.stderr}"
    )
    flat = json.loads(result.stdout)
    return np.array(flat, dtype=np.float32)


def test_js_py_tensor_parity_empty_state():
    """Empty state tensors agree exactly across JS and Python."""
    state = TwixtState(active_size=8)
    py_tensor = state.to_tensor()  # (C, H, W)

    flat = _run_js(8, [])
    C, H, W = py_tensor.shape
    js_tensor = flat.reshape(C, H * W).reshape(C, H, W)

    assert np.allclose(py_tensor, js_tensor, atol=1e-6), (
        f"mismatch on empty state: max diff = "
        f"{np.max(np.abs(py_tensor - js_tensor))}"
    )


def test_js_py_tensor_parity_with_moves():
    """A small scripted game produces identical tensors in JS and Python."""
    # red top edge, black left edge, red bottom edge, black right edge
    moves = [[0, 3], [4, 0], [7, 5], [4, 7]]

    state = TwixtState(active_size=8)
    for (r, c) in moves:
        state = state.apply_move((r, c))
    py_tensor = state.to_tensor()

    flat = _run_js(8, moves)
    C, H, W = py_tensor.shape
    js_tensor = flat.reshape(C, H * W).reshape(C, H, W)

    assert np.allclose(py_tensor, js_tensor, atol=1e-6), (
        f"mismatch with moves: max diff = "
        f"{np.max(np.abs(py_tensor - js_tensor))}"
    )
