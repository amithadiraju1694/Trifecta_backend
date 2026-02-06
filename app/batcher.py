from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .ort_pool import OrtSessionPool


@dataclass
class BatchItem:
    """Single queued inference request and completion future."""
    tensor: np.ndarray
    future: asyncio.Future

class AsyncBatcher:
    """Collect small requests into larger batches for more efficient inference."""

    def __init__(
        self,
        pool: OrtSessionPool,
        input_name: str,
        output_name: str,
        max_batch_size: int,
        batch_timeout_ms: float,
    ) -> None:
        """Store batching settings and shared resources for fast reuse across requests.

        Performance: avoids re-creating expensive session/pool objects per request and
        precomputes limits so batching decisions are cheap during inference.
        """
        self._pool = pool
        self._input_name = input_name
        self._output_name = output_name
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_timeout_ms = max(0.0, float(batch_timeout_ms))
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the background batching loop once.

        Performance: keeps a hot task ready to merge requests immediately, avoiding per-call
        startup overhead and reducing latency spikes.
        """
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background batching loop safely.

        Performance: prevents wasted work and frees the event loop when batching is not needed,
        keeping throughput stable for other tasks.
        """
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def enqueue(self, tensor: np.ndarray) -> np.ndarray:
        """Queue a request and wait for its slice of the batched output.

        Performance: lets many callers share one larger inference call, which is usually faster
        on GPU/CPU than running many small calls.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(BatchItem(tensor=tensor, future=future))
        return await future

    async def _run_loop(self) -> None:
        """Continuously build batches, run inference, and split results back to callers.

        Performance: packs multiple requests into one model call (better hardware utilization)
        while using a short timeout to balance throughput and latency.
        """
        # `pending` holds one item that didn't fit in the previous batch; it is processed first next loop.
        pending: BatchItem | None = None
        while True:
            if pending is not None:
                item = pending
                pending = None
            else:
                item = await self._queue.get()
            items = [item]
            # batch_size tracks total rows across requests (single request = its row count).
            batch_size = item.tensor.shape[0]
            start = time.perf_counter()

            while batch_size < self._max_batch_size:
                remaining_ms = self._batch_timeout_ms - (time.perf_counter() - start) * 1000.0
                if remaining_ms <= 0:
                    break
                # still time remaining, so wait for more requests
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), remaining_ms / 1000.0)
                except asyncio.TimeoutError:
                    break
                # If we got more requests, check if we exceeded max batch size.
                if batch_size + next_item.tensor.shape[0] > self._max_batch_size:
                    pending = next_item
                    break
                items.append(next_item)
                batch_size += next_item.tensor.shape[0]

            # Flow comes here if there's time out or we got more items than max batch size
            try:
                # Single request stays as-is; multiple requests are concatenated for one shared inference.
                batch_tensor = np.concatenate([it.tensor for it in items], axis=0)
                # Model inference (single or batched) is executed via the shared ONNX pool for speed.
                outputs = await self._pool.run(
                    {self._input_name: batch_tensor}, [self._output_name]
                )
                logits = outputs[0]

                offset = 0
                for it in items:
                    b = it.tensor.shape[0]
                    # Assign each request its slice from the batched output.
                    it.future.set_result(logits[offset : offset + b])
                    offset += b
            except Exception as exc:  # pragma: no cover - best effort for service robustness
                for it in items:
                    if not it.future.done():
                        it.future.set_exception(exc)
