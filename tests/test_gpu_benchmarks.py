"""Test that GPU benchmark script is importable and structures are correct."""

import pytest

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


@gpu_required
def test_gpu_benchmark_runs():
    """Smoke test: run benchmarks on smallest configs only."""
    from examples.gpu_benchmarks import benchmark_gpu, _load_kernel, compute_throughput
    from cutile_stencil import stencil, compile as stencil_compile
    import cupy as cp
    import numpy as np

    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    result = stencil_compile(heat, domain=(256,))
    launch = _load_kernel(result)

    u = cp.random.randn(258).astype(cp.float64)
    out = cp.zeros_like(u)

    ms = benchmark_gpu(launch, u, out, warmup=2, iterations=5)
    assert ms > 0
    assert compute_throughput((256,), ms) > 0
