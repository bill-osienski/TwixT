"""RemoteEvaluator: Evaluator for worker processes (CPU-only).

Implements the Evaluator protocol by sending inference requests to the
main process via a shared queue, then waiting for responses on a
dedicated per-worker response queue.

Key features:
- Mailbox for out-of-order response handling
- 60s timeout for crash detection
- Backpressure via bounded request queue
"""
from __future__ import annotations
import itertools
import queue
from typing import Tuple, Dict, Any
import numpy as np

from .ipc_messages import InferenceRequest, InferenceResponse


class RemoteEvaluator:
    """Evaluator that sends inference requests to main process."""

    def __init__(self, worker_id: int, request_queue: Any, response_queue: Any):
        """Initialize remote evaluator.

        Args:
            worker_id: Unique ID for this worker
            request_queue: Shared queue for sending requests to server
            response_queue: Per-worker queue for receiving responses
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_counter = itertools.count(1)
        self._mailbox: Dict[int, InferenceResponse] = {}  # For out-of-order responses

    def build_input_tensor(self, state) -> np.ndarray:
        """Build the (C, H, W) input tensor for the training network.

        Workers always use the current (30-channel) tensor format — training
        never runs against legacy 24-channel checkpoints. Mirrors
        LocalGPUEvaluator.build_input_tensor so MCTS's tensor construction is
        uniform across evaluator backends.
        """
        return state.to_tensor()

    def infer(
        self,
        boards: np.ndarray,
        move_rows: np.ndarray,
        move_cols: np.ndarray,
        move_mask: np.ndarray,
        active_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Send inference request and wait for response.

        Args:
            boards: Board states (B, H, W, C) float32
            move_rows: Move row indices (B, M) int32
            move_cols: Move column indices (B, M) int32
            move_mask: Valid move mask (B, M) float32
            active_size: Curriculum board size

        Returns:
            priors: Policy probabilities (B, M) float32
            values: Value estimates (B,) float32

        Raises:
            RuntimeError: If server doesn't respond within 60 seconds
        """
        request_id = next(self._req_counter)

        req = InferenceRequest(
            worker_id=self.worker_id,
            request_id=request_id,
            boards=boards,
            move_rows=move_rows,
            move_cols=move_cols,
            move_mask=move_mask,
            active_size=int(active_size),
        )

        # Send request (backpressure here if request_queue full)
        self.request_queue.put(req)

        # Check mailbox first (out-of-order responses)
        if request_id in self._mailbox:
            resp = self._mailbox.pop(request_id)
            return resp.priors, resp.values

        # Block until our response arrives (with timeout for crash detection)
        while True:
            try:
                resp = self.response_queue.get(timeout=60.0)
            except queue.Empty:
                raise RuntimeError(
                    f"Worker {self.worker_id}: inference server unresponsive (60s timeout)"
                )

            if isinstance(resp, InferenceResponse):
                if resp.request_id == request_id:
                    return resp.priors, resp.values
                # Not ours - stash for later
                self._mailbox[resp.request_id] = resp
