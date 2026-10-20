"""ROOT-ONLY SEQUENTIAL-HALVING VARIANT -- an isolated search prototype. Not production, not wired into any
caller, and not full Gumbel AlphaZero (Danihelka et al., ICLR 2022): only the root's allocation and final
decision change; descent below the root is the existing PUCT, through MCTS._run_single_simulation's
root_move_override seam.

Frozen variant (no parameters):
* plies 0-19: ordinary MCTS.search_with_root, unchanged (last_choice = None; the caller's readout applies);
* from ply 20, 24x24 only, no root noise: CANDIDATES = the 16 highest-prior legal moves (fewer if fewer legal),
  ties by (row, col); zero Gumbel noise.
* score(a) = log(max(prior_a, PRIOR_FLOOR)) + (C_VISIT + max_b N(b)) * C_SCALE * q_normalized(a), with
  q_normalized = (q + 1) / 2 and q the candidate's mean value from the root player's side. C_VISIT 50 and
  C_SCALE 1.0 are the paper's Go/chess settings; the fixed (q + 1) / 2 mapping replaces the paper's
  completed-Q rescaling (an adaptation with no published guarantee). Rank ties: higher prior, then (row, col).
* budget N = n_simulations: with m candidates there are R = ceil(log2 m) visiting rounds. Each round but the
  last gives every survivor N // (R * survivors) more visits, then keeps the top ceil(survivors / 2) by score.
  The last round splits the rest of N evenly over its survivors, any remainder one visit each in rank order
  (prior order when nothing has been visited, i.e. m <= 2). Visits run round-robin in rank order. For m = 16:
  6 / 12 / 25 / 54 more visits per round (cumulative 6 / 18 / 43 / 97); m = 1 gives all N to the one move.
* the move to play (last_choice) is the top-ranked final survivor; it replaces the caller's readout.
Known limit, not patched: a move outside the 16 candidates (e.g. a low-prior winning move) is never searched.
"""
import math

from .mcts import MCTS, MCTSNode, decode_move, encode_move

FROM_PLY = 20
CANDIDATES = 16
C_VISIT = 50.0
C_SCALE = 1.0
PRIOR_FLOOR = 1e-12
BOARD = 24


def score(prior, q_root, max_visits):
    return math.log(max(prior, PRIOR_FLOOR)) + (C_VISIT + max_visits) * C_SCALE * (q_root + 1.0) / 2.0


class RootSequentialHalvingMCTS(MCTS):
    last_choice = None          # (row, col) to play, or None when the ordinary search ran
    last_rounds = None          # survivors (row, col) entering each visiting round

    def search_with_root(self, root_state, add_noise=True):
        if root_state.ply < FROM_PLY:
            self.last_choice = self.last_rounds = None
            return super().search_with_root(root_state, add_noise=add_noise)
        if add_noise:
            raise ValueError("the halving variant runs without root noise")
        if root_state.active_size != BOARD:
            raise ValueError(f"24x24 only, got active_size={root_state.active_size}")
        if root_state.is_terminal():
            raise ValueError("no search from a terminal root")
        n = self.config.n_simulations
        root = MCTSNode(state=root_state)
        self._expand(root)
        prior = root.priors

        def rank(mids):
            top = max(c.visit_count for c in root.children.values())
            return sorted(mids, key=lambda m: (-score(prior[m], -root.children[m].q_value, top), -prior[m],
                                               decode_move(m)))

        survivors = sorted(prior, key=lambda m: (-prior[m], decode_move(m)))[:CANDIDATES]
        rounds = (len(survivors) - 1).bit_length()
        self.last_rounds, used = [], 0
        for _ in range(rounds - 1):
            each = n // (rounds * len(survivors))
            if each < 1:
                raise ValueError(f"{n} simulations cannot visit {len(survivors)} survivors")
            self.last_rounds.append([decode_move(m) for m in survivors])
            self._visit(root, {m: each for m in survivors})
            used += each * len(survivors)
            survivors = rank(survivors)[:-(-len(survivors) // 2)]
        each, extra = divmod(n - used, len(survivors))
        if each < 1:
            raise ValueError(f"{n} simulations cannot visit {len(survivors)} survivors")
        self.last_rounds.append([decode_move(m) for m in survivors])
        self._visit(root, {m: each + (i < extra) for i, m in enumerate(survivors)})
        self.last_choice = decode_move(rank(survivors)[0])

        visit_counts = {}
        for (r, c) in root.state.legal_moves():
            child = root.children.get(encode_move(r, c))
            visit_counts[(r, c)] = child.visit_count if child else 0
        self._capture_final_root_stats(root)
        return visit_counts, root.q_value, root

    def _visit(self, root, allotment):
        """Round-robin root-override simulations, in the allotment's (rank) order."""
        for t in range(max(allotment.values())):
            for m, k in allotment.items():
                if t < k:
                    self._run_single_simulation(root, root_move_override=m)
