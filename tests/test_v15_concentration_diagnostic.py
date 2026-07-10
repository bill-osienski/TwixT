import random
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import (
    per_child_metrics, classify_concentration)

import json

from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import (
    raw_black_value, search_for_row)
from tests.goal_line_probe_fixtures import legal_replay


def _root_with_children(specs):
    # specs: list of (move_rc, visit_count, q_value). Builds a root whose
    # invariants (visit_count sum, value_sum = q*visits) match the real backup.
    root = MCTSNode(state=None)
    for (rc, vc, q) in specs:
        ch = MCTSNode(state=None, parent=root, move=encode_move(*rc),
                      visit_count=vc, value_sum=q * vc)
        root.children[ch.move] = ch
    root.visit_count = sum(vc for _, vc, _ in specs)
    root.value_sum = sum(-q * vc for _, vc, q in specs)   # single sign flip child->root
    return root


def test_contributions_sum_to_root_q():
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 60, 0.2), ((3, 3), 40, 0.1)])
    m = per_child_metrics(root)
    assert abs(sum(c["child_contribution_share"] for c in m) - root.q_value) < 1e-9
    assert abs(sum(c["visit_share"] for c in m) - 1.0) < 1e-9


def test_visit_share_and_contribution_values():
    # child (1,1): 300/400 visits, q=-0.9 -> contribution = 0.75 * 0.9 = +0.675 (root perspective)
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 100, 0.0)])
    m = {tuple(c["move"]): c for c in per_child_metrics(root)}
    assert abs(m[(1, 1)]["visit_share"] - 0.75) < 1e-9
    assert abs(m[(1, 1)]["child_contribution_share"] - 0.675) < 1e-9


def test_positive_contribution_share_normalizes_positive_mass():
    # one optimistic child (+contribution), one defensive (-contribution)
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 100, +0.5)])
    m = {tuple(c["move"]): c for c in per_child_metrics(root)}
    assert m[(2, 2)]["child_contribution_share"] < 0                      # defensive
    assert m[(2, 2)]["positive_contribution_share"] == 0.0               # -> 0 positive mass
    assert abs(m[(1, 1)]["positive_contribution_share"] - 1.0) < 1e-9    # carries all positive mass
    assert abs(sum(c["positive_contribution_share"] for c in m.values()) - 1.0) < 1e-9


def test_classify_concentrated():
    # top child carries ~96% of the positive backup mass
    root = _root_with_children([((1, 1), 380, -0.9), ((2, 2), 20, -0.1)])
    label, share = classify_concentration(per_child_metrics(root))
    assert label == "concentrated" and share >= 0.70


def test_classify_broad():
    # positive backup spread across many similar children
    specs = [((r, r), 40, -0.2) for r in range(1, 11)]     # 10 children, equal
    label, share = classify_concentration(per_child_metrics(_root_with_children(specs)))
    assert label == "broad" and share < 0.40


def test_raw_black_value_converts_to_black_perspective():
    # _teacher_infer returns the value in the state's OWN to-move perspective.
    # raw_black_value must flip it when the state is red-to-move, and not
    # otherwise. A fake evaluator returns a fixed value for any board.
    class _FakeEvaluator:
        def build_input_tensor(self, state):
            import numpy as np
            return np.zeros((3, state.active_size, state.active_size), dtype=np.float32)

        def infer(self, board, rows, cols, mask, active_size):
            import numpy as np
            return np.zeros((1, rows.shape[1]), dtype=np.float32), np.array([0.7], dtype=np.float32)

    replay = legal_replay(4)
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    black_state = position_state(replay, 1, "black")   # ply 1 -> black to move
    red_state = position_state(replay, 2, "red")       # ply 2 -> red to move
    # Tolerance 1e-6 (not 1e-9): the fake evaluator's np.float32(0.7) upcasts
    # to 0.699999988079071 (diff ~1.19e-8), so a 1e-9 tolerance fails on
    # float32 round-off alone, unrelated to the sign conversion under test.
    # 1e-6 matches this module's own assertion tolerances (main()'s sign
    # sanity checks) and still trivially catches a sign-flip bug (diff ~1.4).
    assert abs(raw_black_value(black_state, _FakeEvaluator()) - 0.7) < 1e-6
    assert abs(raw_black_value(red_state, _FakeEvaluator()) + 0.7) < 1e-6


def test_search_for_row_reconstructs_state_and_seeds_deterministically(tmp_path):
    # search_for_row must reconstruct the root from replay_path/position_ply,
    # pass the row_seed-derived seed to search_fn, and return the search's
    # (state, side, root_value_stm, root) without touching them.
    replay = legal_replay(6)
    replay_path = tmp_path / "game_000007.json"
    replay_path.write_text(json.dumps(replay))
    # legal_replay starts with red to move, so an ODD position_ply is
    # black-to-move; position_state raises if this disagrees with side_to_move.
    row = {"case_id": "c", "game_idx": 7, "position_ply": 1,
           "side_to_move": "black", "replay_path": str(replay_path)}

    seen = {}
    sentinel_root = object()

    def fake_search_fn(state, seed):
        seen["state"], seen["seed"] = state, seed
        return {"counts": 1}, 0.25, sentinel_root

    state, side, root_value_stm, root = search_for_row(
        row, fake_search_fn, pos_base_seed=20260616, goal_base_seed=20260614)

    assert seen["seed"] == 20260616 ^ 7 ^ 1      # row_seed's position-probe branch
    assert side == "black" and state.to_move == "black"
    assert root_value_stm == 0.25 and root is sentinel_root
    assert state is seen["state"]
