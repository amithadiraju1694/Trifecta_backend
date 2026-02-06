from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import onnxruntime as ort


def build_session(
    onnx_path: str,
    device_id: Optional[int],
    use_cuda: bool,
) -> ort.InferenceSession:
    """Create one ONNX Runtime inference session with optimized providers."""
    sess_options = ort.SessionOptions()
    # Enable highest graph optimization level for faster execution.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Disable memory pattern cache because dynamic batch shapes can vary.
    sess_options.enable_mem_pattern = False
    # Keep per-session CPU threading predictable in a multi-session setup.
    sess_options.intra_op_num_threads = 1

    providers: List = []
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        # Provider options tune CUDA execution behavior for throughput stability.
        provider_options = {
            "device_id": device_id if device_id is not None else 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": "1",
        }
        providers.append(("CUDAExecutionProvider", provider_options))
    providers.append("CPUExecutionProvider")

    try:
        return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    except Exception:
        if providers and isinstance(providers[0], tuple) and providers[0][0] == "CUDAExecutionProvider":
            return ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        raise


@dataclass
class OrtSessionRunner:
    """Pair one session with its concurrency limiter."""
    session: ort.InferenceSession
    semaphore: asyncio.Semaphore


class OrtSessionPool:
    """Round-robin pool that runs inference across multiple runtime sessions."""

    def __init__(self, sessions: Iterable[ort.InferenceSession], max_concurrency_per_gpu: int) -> None:
        """Create session runners and internal scheduling primitives."""
        self._sessions: List[ort.InferenceSession] = list(sessions)
        self._runners: List[OrtSessionRunner] = [
            # Semaphore limits concurrent calls per session to protect memory/latency.
            OrtSessionRunner(session=s, semaphore=asyncio.Semaphore(max_concurrency_per_gpu))
            for s in self._sessions
        ]
        if not self._runners:
            raise ValueError("At least one ONNX Runtime session is required")
        self._rr = itertools.cycle(range(len(self._runners)))
        self._lock = asyncio.Lock()

    async def _next_runner(self) -> OrtSessionRunner:
        """Select the next runner using a lock-protected round-robin iterator."""
        async with self._lock:
            idx = next(self._rr)
            return self._runners[idx]

    async def run(self, input_feed: Dict[str, object], output_names: List[str]):
        """Run inference asynchronously on one pooled session."""
        runner = await self._next_runner()
        async with runner.semaphore:
            # run() is blocking; offload to a thread to keep the event loop responsive
            return await asyncio.to_thread(runner.session.run, output_names, input_feed)

    def warmup_sync(self, input_feed: Dict[str, object], output_names: List[str]) -> None:
        """Run one synchronous warmup inference on the first session."""
        # best-effort warmup on the first session
        try:
            self._sessions[0].run(output_names, input_feed)
        except Exception:
            pass
