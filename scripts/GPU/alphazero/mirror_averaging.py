"""PROTOTYPE, not production -- nothing in the trainer, eval runner or browser imports this.

MirrorAveragingEvaluator wraps an evaluator and answers each infer() with the average of the network's
predictions for the batch and its left-right reflection: the reflected boards come from self_play's
_mirror_position_lr (with the 27<->28 connectivity swap, 0c417882), the reflected legal moves are (r, S-1-c) in the
same order, so reflected move j IS the mirror of original move j and the mapped-back average is index-aligned.
Priors: mean of the two masked probability distributions (no renormalization -- a mean of two distributions is
one). Values: mean of the two (both side-to-move). Two underlying infer() calls per call; weights, search and
readout untouched.
"""
import numpy as np

from .self_play import _mirror_position_lr


class MirrorAveragingEvaluator:
    def __init__(self, base):
        self.base = base
        self.network = base.network

    def build_input_tensor(self, state):
        return self.base.build_input_tensor(state)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        S = int(active_size)
        m_boards = np.stack([_mirror_position_lr(b, [], [], S)[0] for b in boards]).astype(np.float32)
        m_cols = np.where(move_mask > 0, S - 1 - move_cols, move_cols).astype(np.int32)
        p, v = self.base.infer(boards, move_rows, move_cols, move_mask, active_size)
        q, w = self.base.infer(m_boards, move_rows, m_cols, move_mask, active_size)
        return ((p + q) / 2).astype(np.float32), ((v + w) / 2).astype(np.float32)
