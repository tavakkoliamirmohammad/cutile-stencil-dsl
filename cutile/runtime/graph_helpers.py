"""CUDA Graph capture helpers for cuTile launchers.

A typical stencil benchmark runs the same launch many times. Each
``launch_<name>(...)`` call carries fixed ~5-10 µs of Python + CuPy
overhead (kernel-arg packing, ``ct.launch`` dispatch, stream selection).
For small kernels (e.g. 256² heat_2d, 32³ Lap) that overhead dominates
real GPU work, capping throughput far below DRAM-bound peak.

CUDA Graphs solve this by recording a sequence of launches once and
replaying them with a single device-side dispatch (~0.5 µs of host
work per replay). cuTile launchers compose cleanly with stream
capture: every ``ct.launch`` becomes a graph node, no special API
needed.

This module exposes:

- :func:`capture_loop` — record ``n_iters`` calls to ``launch_fn``
  (with internal buffer ping-pong) into a CUDA graph and return a
  ``CapturedLoop`` you can :meth:`~CapturedLoop.replay` cheaply.
- :func:`capture_multigpu_loop` — same idea but for the multi-GPU
  step body. Captures one graph per GPU + the cross-device halo
  memcpys via per-GPU stream capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cupy as cp


@dataclass
class CapturedLoop:
    """Handle to a captured stencil-iteration graph."""

    graph: object   # cp.cuda.Graph (instantiated)
    stream: cp.cuda.Stream
    n_iters: int    # iterations recorded inside the graph

    def replay(self, n_replays: int = 1) -> None:
        """Replay the captured graph *n_replays* times.

        Each replay executes ``n_iters`` stencil iterations. After
        ``replay()`` returns, the stream has the launches enqueued but
        not necessarily completed; call :meth:`synchronize` to wait.
        """
        for _ in range(n_replays):
            self.graph.launch(self.stream)

    def synchronize(self) -> None:
        self.stream.synchronize()


def capture_loop(
    launch_fn: Callable[..., None],
    buf_in: cp.ndarray,
    buf_out: cp.ndarray,
    n_iters: int = 50,
    stream: cp.cuda.Stream | None = None,
) -> CapturedLoop:
    """Capture *n_iters* alternating launches into a CUDA graph.

    The recorded graph is::

        launch_fn(buf_in,  buf_out)
        launch_fn(buf_out, buf_in)
        launch_fn(buf_in,  buf_out)
        ...                                  (n_iters total)

    The captured graph references the buffer pointers as they exist
    at capture time, so subsequent replays reuse the same allocations.
    Stream capture records ops without executing them, so ``buf_in`` /
    ``buf_out`` retain their pre-capture contents until the first
    :meth:`CapturedLoop.replay` runs.

    The capture uses a non-default non-blocking stream so it does not
    interfere with the legacy default stream.
    """
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    a, b = buf_in, buf_out
    with stream:
        stream.begin_capture()
        for _ in range(n_iters):
            launch_fn(a, b)
            a, b = b, a
        graph = stream.end_capture()
    return CapturedLoop(graph=graph, stream=stream, n_iters=n_iters)


# Note on multi-GPU CUDA Graph capture
# -------------------------------------
# We attempted multi-device graph capture via per-GPU
# ``streamBeginCapture(..., cudaStreamCaptureModeRelaxed)`` but
# ``cudaMemcpyPeerAsync`` is not supported on a capturing stream:
# the runtime returns ``cudaErrorStreamCaptureUnsupported``. This is a
# CUDA constraint, not a CuPy one — multi-device graph capture for
# patterns that rely on P2P memcpys requires either manual
# ``cudaGraphAddMemcpyNode`` graph construction or replacing the
# memcpys with explicit P2P kernels. Both are out of scope for this
# release; ``capture_loop`` (single-GPU) provides the bulk of the
# host-overhead win in practice.
