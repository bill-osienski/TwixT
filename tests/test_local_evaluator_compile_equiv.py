"""compile=True must produce identical inference to compile=False across
different input shapes. Guards the eval MLX-compile fix (graph keyed on
active_size while B and M vary) against MLX version drift.

Integration-marked: does real MLX inference (no checkpoint needed; uses a
fresh randomly-initialised network)."""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _batch(ev, states):
    boards = np.stack(
        [np.transpose(ev.build_input_tensor(s), (1, 2, 0)) for s in states]
    ).astype(np.float32)
    max_m = max(len(s.legal_moves()) for s in states)
    b = len(states)
    rows = np.zeros((b, max_m), np.int32)
    cols = np.zeros((b, max_m), np.int32)
    mask = np.zeros((b, max_m), np.float32)
    for bi, s in enumerate(states):
        for j, (r, c) in enumerate(s.legal_moves()):
            rows[bi, j] = r
            cols[bi, j] = c
            mask[bi, j] = 1.0
    return boards, rows, cols, mask, states[0].active_size


def test_compiled_matches_uncompiled_across_shapes():
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    net = create_network()
    plain = LocalGPUEvaluator(net, compile=False)
    comp = LocalGPUEvaluator(net, compile=True)

    # Two states at different plies => different #legal-moves (different M).
    s1 = TwixtState(active_size=24, to_move="red")
    s2 = s1.apply_move(next(iter(s1.legal_moves())))

    # Shape A: batch of 1.  Shape B: batch of 2 (different B and M).
    for states in ([s1], [s1, s2]):
        b, r, c, m, asz = _batch(plain, states)
        p1, v1 = plain.infer(b, r, c, m, asz)
        p2, v2 = comp.infer(b, r, c, m, asz)
        assert np.allclose(p1, p2, atol=1e-5), "priors diverged compiled vs not"
        assert np.allclose(v1, v2, atol=1e-5), "values diverged compiled vs not"
