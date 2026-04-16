"""CuPy baseline: GPU array-slicing stencils (no tiling, no shared memory)."""

from __future__ import annotations

import cupy as cp

from benchmarks.stencils import STENCIL_META, full_shape, interior_size


def _cupy_heat_1d(u):
    return 0.25 * u[:-2] + 0.5 * u[1:-1] + 0.25 * u[2:]


def _cupy_heat_2d(u):
    return 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])


def _cupy_laplacian_2d_5pt(u):
    return (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        - 4 * u[1:-1, 1:-1]
    )


def _cupy_laplacian_2d_9pt(u):
    return (
        -u[:-4, 2:-2] + 16 * u[1:-3, 2:-2] - 30 * u[2:-2, 2:-2]
        + 16 * u[3:-1, 2:-2] - u[4:, 2:-2]
        - u[2:-2, :-4] + 16 * u[2:-2, 1:-3] - 30 * u[2:-2, 2:-2]
        + 16 * u[2:-2, 3:-1] - u[2:-2, 4:]
    ) / 12.0


def _cupy_laplacian_3d_7pt(u):
    return (
        u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]
        - 6 * u[1:-1, 1:-1, 1:-1]
    )


_CUPY_FNS = {
    "heat_1d": _cupy_heat_1d,
    "heat_2d": _cupy_heat_2d,
    "laplacian_2d_5pt": _cupy_laplacian_2d_5pt,
    "laplacian_2d_9pt": _cupy_laplacian_2d_9pt,
    "laplacian_3d_7pt": _cupy_laplacian_3d_7pt,
}


def bench_cupy(
    name: str,
    shape: tuple[int, ...],
    warmup: int = 30,
    iters: int = 100,
) -> dict:
    """Benchmark a CuPy stencil. Returns dict with time_ms, gpoints_per_s, gbytes_per_s."""
    fn = _CUPY_FNS[name]
    meta = STENCIL_META[name]
    u = cp.random.randn(*shape).astype(cp.float64)

    for _ in range(warmup):
        _ = fn(u)
    cp.cuda.Device(0).synchronize()

    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters):
        _ = fn(u)
    e2.record()
    e2.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(e1, e2) / iters
    domain = tuple(s - 2 * h for s, h in zip(shape, meta["halo"]))
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "name": name,
        "framework": "CuPy",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
    }
