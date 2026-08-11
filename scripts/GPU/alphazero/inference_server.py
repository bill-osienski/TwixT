"""InferenceServer: GPU inference batching for multi-process self-play.

Runs in the MAIN process (same process as GPU/MLX). Pulls inference requests
from workers, batches them efficiently, runs GPU inference, and routes
responses back to the appropriate workers.

Key design:
- max_batch_rows caps TOTAL rows per GPU call (not request count)
- Groups by active_size for curriculum safety
- Crash-safe with error reporting to stats_queue
"""
from __future__ import annotations
import time
import queue
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .ipc_messages import InferenceRequest, InferenceResponse, StopSignal


class InferenceServer:
    """Batches inference requests from workers, runs GPU inference."""

    def __init__(
        self,
        evaluators: Dict[str, Any],  # model_id -> LocalGPUEvaluator
        request_queue,
        response_queues: Dict[Tuple[int, str], Any],  # (worker_id, model_id) -> Queue
        max_batch_rows: int = 14,  # Cap on TOTAL rows per GPU call
        flush_ms: int = 2,  # Batch collection timeout (ms)
        stats_queue: Optional[Any] = None,  # For error reporting
    ):
        """Initialize inference server.

        Args:
            evaluators: model_id -> LocalGPUEvaluator. ONE server thread owns
                the device and serves every network in this map; two servers on
                one Metal device abort the process (do-not-repeat #50).
            request_queue: Single shared queue for incoming requests
            response_queues: Queues keyed by (worker_id, model_id), so each
                remote evaluator owns its own return path
            max_batch_rows: Max total rows per GPU batch (default 14)
            flush_ms: Max time to wait for batch to fill (default 2ms)
            stats_queue: Optional queue for error reporting
        """
        if not evaluators:
            raise ValueError("InferenceServer requires at least one evaluator")
        self.evaluators = dict(evaluators)
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_rows = max_batch_rows
        self.flush_ms = flush_ms
        self.stats_queue = stats_queue
        self._running = True
        self._error: Optional[Exception] = None

        # Telemetry, global and per model. Per-model counters exist so a smoke
        # can prove neither network starved rather than assuming it.
        self._batches_flushed = 0
        self._total_requests = 0
        self._model_requests: Dict[str, int] = {m: 0 for m in self.evaluators}
        self._model_rows: Dict[str, int] = {m: 0 for m in self.evaluators}
        self._model_batches: Dict[str, int] = {m: 0 for m in self.evaluators}

    def model_telemetry(self) -> Dict[str, Dict[str, int]]:
        """Per-model requests / rows / device batches."""
        return {
            m: {"requests": self._model_requests[m],
                "rows": self._model_rows[m],
                "batches": self._model_batches[m]}
            for m in self.evaluators
        }

    def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False

    def run_forever(self) -> None:
        """Main server loop. Wrapped in try/except for crash safety."""
        try:
            self._run_loop()
        except Exception as e:
            self._error = e
            self._running = False
            # Report error so trainer can fail fast
            if self.stats_queue:
                self.stats_queue.put({"type": "server_error", "error": str(e)})
            raise

    def _run_loop(self) -> None:
        """Core batching loop."""
        pending: List[InferenceRequest] = []
        pending_rows = 0
        t0 = time.time()

        while self._running:
            # 1) Pull at least one item (blocking briefly)
            try:
                item = self.request_queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            if isinstance(item, StopSignal):
                break
            elif isinstance(item, InferenceRequest):
                pending.append(item)
                pending_rows += item.boards.shape[0]

            # 2) Keep draining until row budget hit (non-blocking)
            while pending_rows < self.max_batch_rows:
                try:
                    item2 = self.request_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item2, StopSignal):
                    self._running = False
                    break
                if isinstance(item2, InferenceRequest):
                    pending.append(item2)
                    pending_rows += item2.boards.shape[0]

            if not pending:
                continue

            # 3) Flush: row budget hit or timeout
            elapsed_ms = (time.time() - t0) * 1000.0
            if pending_rows < self.max_batch_rows and elapsed_ms < self.flush_ms and self._running:
                continue

            # 4) Run batched inference
            self._flush(pending)
            pending.clear()
            pending_rows = 0
            t0 = time.time()

        # Final tail flush
        if pending:
            self._flush(pending)

    def _flush(self, batch: List[InferenceRequest]) -> None:
        """Process pending requests.

        Grouping is by **(model_id, active_size)**: distinct networks cannot
        share a forward pass, and active_size grouping is retained for mixed
        curriculum. Each resulting sub-batch is inferred by its own model and
        its responses are routed by **(worker_id, model_id)**, so a reply can
        only ever reach the evaluator that asked for it.
        """
        # Group by (model_id, active_size). A batch must never mix models --
        # distinct networks cannot share a forward pass -- and active_size
        # grouping is kept for mixed curriculum.
        groups: Dict[Any, List[InferenceRequest]] = defaultdict(list)
        for req in batch:
            groups[(req.model_id, req.active_size)].append(req)

        for (model_id, active_size), reqs in groups.items():
            # Sub-batch if group exceeds row budget
            subbatch: List[InferenceRequest] = []
            subbatch_rows = 0

            for req in reqs:
                req_rows = req.boards.shape[0]
                if subbatch_rows + req_rows > self.max_batch_rows and subbatch:
                    # Flush current subbatch
                    self._infer_and_respond(subbatch, active_size, model_id)
                    subbatch = []
                    subbatch_rows = 0
                subbatch.append(req)
                subbatch_rows += req_rows

            # Flush remaining
            if subbatch:
                self._infer_and_respond(subbatch, active_size, model_id)

    def _infer_and_respond(self, reqs: List[InferenceRequest], active_size: int,
                           model_id: str) -> None:
        """Run GPU inference and send responses to workers.

        Pads all requests to a common max_M (move dimension) before batching,
        since different requests may have different numbers of legal moves.
        Trims priors back to original M when sending responses.
        """
        # First pass: collect metadata and find common max_M
        req_meta = []  # List of (batch_size, orig_M) tuples
        max_M = 0
        total_B = 0
        for req in reqs:
            b = int(req.boards.shape[0])
            m = int(req.move_rows.shape[1])
            # Sanity check: all move arrays must have same M
            assert req.move_cols.shape[1] == m and req.move_mask.shape[1] == m, (
                f"Shape mismatch: move_rows M={m}, move_cols M={req.move_cols.shape[1]}, "
                f"move_mask M={req.move_mask.shape[1]}"
            )
            # Sanity check: board batch size matches move batch sizes
            assert req.move_rows.shape[0] == b and req.move_cols.shape[0] == b and req.move_mask.shape[0] == b, (
                f"Batch mismatch: boards B={b}, move_rows B={req.move_rows.shape[0]}, "
                f"move_cols B={req.move_cols.shape[0]}, move_mask B={req.move_mask.shape[0]}"
            )
            req_meta.append((b, m))
            max_M = max(max_M, m)
            total_B += b

        # Guard: terminal states should never reach inference (filtered in MCTS expand)
        assert max_M > 0, "Inference batch has max_M=0 (no legal moves) - terminal states shouldn't be here"

        # Stack boards (all have same H, W, C - simple concatenation)
        boards = np.concatenate([r.boards for r in reqs], axis=0)

        # Fast path: if all requests have same M, skip padding
        if all(m == max_M for _, m in req_meta):
            move_rows = np.concatenate([r.move_rows for r in reqs], axis=0)
            move_cols = np.concatenate([r.move_cols for r in reqs], axis=0)
            move_mask = np.concatenate([r.move_mask for r in reqs], axis=0)
        else:
            # Pad move arrays to common max_M
            move_rows = np.zeros((total_B, max_M), dtype=np.int32)
            move_cols = np.zeros((total_B, max_M), dtype=np.int32)
            move_mask = np.zeros((total_B, max_M), dtype=np.float32)

            row_offset = 0
            for req, (b, m) in zip(reqs, req_meta):
                move_rows[row_offset:row_offset + b, :m] = req.move_rows
                move_cols[row_offset:row_offset + b, :m] = req.move_cols
                move_mask[row_offset:row_offset + b, :m] = req.move_mask
                row_offset += b

            assert row_offset == total_B, f"row_offset={row_offset} != total_B={total_B}"

        # Run inference on THIS model. Fail closed on an unknown id: falling
        # back to a default evaluator would answer with the wrong network.
        evaluator = self.evaluators.get(model_id)
        if evaluator is None:
            raise KeyError(
                f"unknown model_id {model_id!r}; registered: "
                f"{sorted(self.evaluators)}"
            )
        priors, values = evaluator.infer(
            boards, move_rows, move_cols, move_mask, active_size
        )

        # Split results and send to each worker (trim priors to original M)
        offset = 0
        for req, (b, m) in zip(reqs, req_meta):
            # Trim priors back to original M (remove padding columns)
            priors_trimmed = priors[offset:offset + b, :m]
            resp = InferenceResponse(
                request_id=req.request_id,
                priors=priors_trimmed,
                values=values[offset:offset + b],
            )
            # Routed by (worker_id, model_id): each evaluator owns its own
            # return path, so request_id counters cannot collide across models.
            self.response_queues[req.route].put(resp)
            offset += b

        # Telemetry
        self._batches_flushed += 1
        self._model_batches[model_id] = self._model_batches.get(model_id, 0) + 1
        self._model_requests[model_id] = self._model_requests.get(model_id, 0) + len(reqs)
        self._model_rows[model_id] = self._model_rows.get(model_id, 0) + int(total_B)
        self._total_requests += len(reqs)
