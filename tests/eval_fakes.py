"""GPU-free deterministic fakes for eval tests.

FakeEvaluator implements the Evaluator protocol MCTS expects
(build_input_tensor + infer) with uniform priors and a fixed value, so
games are deterministic per seed and need no checkpoint or GPU.

Factories are module-level functions so they pickle under the spawn
multiprocessing context (lambdas do not).
"""
from __future__ import annotations

import numpy as np


class FakeEvaluator:
    def __init__(self, value: float = 0.0):
        self._value = float(value)

    def build_input_tensor(self, state) -> np.ndarray:
        # (C, H, W); contents ignored by infer. Minimal C=1.
        return np.zeros((1, state.active_size, state.active_size), dtype=np.float32)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        mask = move_mask.astype(np.float32)
        row_sums = mask.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        priors = (mask / row_sums).astype(np.float32)
        values = np.full((mask.shape[0],), self._value, dtype=np.float32)
        return priors, values


def fake_evaluator_factory(path: str) -> FakeEvaluator:
    """Picklable factory: ignores path, returns a fresh FakeEvaluator."""
    return FakeEvaluator(value=0.0)


def counting_factory(path: str) -> FakeEvaluator:
    """Sequential-only counting factory (process-local dict)."""
    counting_factory.calls[path] = counting_factory.calls.get(path, 0) + 1
    return FakeEvaluator(value=0.0)


counting_factory.calls = {}
