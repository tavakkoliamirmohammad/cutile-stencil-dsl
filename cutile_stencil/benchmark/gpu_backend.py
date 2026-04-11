"""GPU timing backend using cupy.cuda.Event for wall-clock kernel measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GPUTimingResult:
    """Result of GPU kernel timing."""
    kernel_name: str
    elapsed_ms: float
    n_iterations: int
    avg_ms: float
    domain_size: int  # Total points


def time_gpu_kernel(
    launch_fn: Callable,
    args: tuple,
    warmup: int = 5,
    iterations: int = 20,
    kernel_name: str = "kernel",
    domain_size: int = 0,
) -> GPUTimingResult:
    """Time a GPU kernel launch using cupy.cuda.Event.

    Parameters
    ----------
    launch_fn : callable
        The kernel launch function to time.
    args : tuple
        Arguments to pass to launch_fn.
    warmup : int
        Number of warmup iterations (not timed).
    iterations : int
        Number of timed iterations.
    kernel_name : str
        Name for the result.
    domain_size : int
        Total domain points (for throughput calculation).

    Returns
    -------
    GPUTimingResult
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("cupy is required for GPU timing")

    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        launch_fn(*args)
    stream.synchronize()

    # Timed iterations
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(iterations):
        launch_fn(*args)
    end.record(stream)
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    avg_ms = elapsed_ms / iterations

    return GPUTimingResult(
        kernel_name=kernel_name,
        elapsed_ms=elapsed_ms,
        n_iterations=iterations,
        avg_ms=avg_ms,
        domain_size=domain_size,
    )
