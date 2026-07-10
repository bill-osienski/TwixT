import random

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move


def _stub_value_fn():
    def f(state):
        return {}, 0.0
    return f


def _synthetic_root(fpu):
    """Root with two candidate moves, each prior 0.01:
      A = a decent VISITED reply for the mover -- child q in the child's own
          perspective = -0.1, so the mover (parent) sees -(-0.1) = +0.1 --
          visited 100 times;
      B = UNVISITED (no child node).
    Arithmetic (c_puct=1.5, sqrt_parent = sqrt(101) = 10.0499):
      score_A = 0.1 + 1.5*0.01*10.0499/(1+100) = 0.1 + 0.00149 = 0.10149
      score_B = fpu + 1.5*0.01*10.0499/(1+0)   = fpu + 0.15075
    So at fpu=0.0, B (0.15075) outranks A (0.10149) -- the legacy pathology,
    an unexplored move beating a decent visited reply. At fpu=-0.5, B is
    -0.34925 and A wins. The two scores are far apart => no rng tie-break."""
    cfg = MCTSConfig(n_simulations=1, c_puct=1.5, fpu_value=fpu)
    m = MCTS(_stub_value_fn(), cfg, random.Random(0))
    A, B = encode_move(0, 0), encode_move(1, 1)
    root = MCTSNode(state=None, visit_count=100)
    root.priors = {A: 0.01, B: 0.01}
    root.children[A] = MCTSNode(state=None, parent=root, move=A,
                                visit_count=100, value_sum=-10.0)  # q_value=-0.1
    return m, root, A, B


def test_fpu_value_default_is_zero():
    assert MCTSConfig().fpu_value == 0.0


def test_fpu_zero_reproduces_legacy_unvisited_wins():
    # fpu=0.0 IS the old hardcoded q=0.0: the unvisited move B wins.
    m, root, A, B = _synthetic_root(fpu=0.0)
    assert m._select_child(root)[0] == B


def test_negative_fpu_makes_the_mover_keep_the_good_visited_child():
    # same root, fpu=-0.5 lowers B's assumed value below A's real value.
    # (If _select_child ignored fpu_value, B would still win -- this discriminates.)
    m, root, A, B = _synthetic_root(fpu=-0.5)
    assert m._select_child(root)[0] == A
