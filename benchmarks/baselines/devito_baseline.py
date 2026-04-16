"""Devito baseline: stencil DSL comparison.

Devito uses symbolic math (SymPy) to define stencil equations and generates
optimized C/OpenMP/OpenACC/CUDA code.

GPU execution requires environment variables:
    export DEVITO_PLATFORM=nvidiaX
    export DEVITO_ARCH=cuda
"""

from __future__ import annotations

import time

import numpy as np

from benchmarks.stencils import STENCIL_META, interior_size


def _make_devito_heat_1d(shape):
    from devito import Grid, TimeFunction, Eq, Operator

    nx = shape[0] - 2
    grid = Grid(shape=(nx,), extent=(1.0,))
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=0)
    t, x = u.dimensions
    eq = Eq(u.forward, 0.25 * u[t, x - 1] + 0.5 * u[t, x] + 0.25 * u[t, x + 1])
    return Operator([eq], name="heat_1d"), u


def _make_devito_heat_2d(shape):
    from devito import Grid, TimeFunction, Eq, Operator

    nx, ny = shape[0] - 2, shape[1] - 2
    grid = Grid(shape=(nx, ny), extent=(1.0, 1.0))
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=0)
    t, x, y = u.dimensions
    eq = Eq(u.forward, 0.25 * (u[t, x - 1, y] + u[t, x + 1, y]
                                + u[t, x, y - 1] + u[t, x, y + 1]))
    return Operator([eq], name="heat_2d"), u


def _make_devito_laplacian_2d_5pt(shape):
    from devito import Grid, TimeFunction, Eq, Operator

    nx, ny = shape[0] - 2, shape[1] - 2
    grid = Grid(shape=(nx, ny), extent=(1.0, 1.0))
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=0)
    eq = Eq(u.forward, u.dx2 + u.dy2)
    return Operator([eq], name="laplacian_2d_5pt"), u


def _make_devito_laplacian_3d_7pt(shape):
    from devito import Grid, TimeFunction, Eq, Operator

    nx, ny, nz = shape[0] - 2, shape[1] - 2, shape[2] - 2
    grid = Grid(shape=(nx, ny, nz), extent=(1.0, 1.0, 1.0))
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=0)
    eq = Eq(u.forward, u.dx2 + u.dy2 + u.dz2)
    return Operator([eq], name="laplacian_3d_7pt"), u


_DEVITO_BUILDERS = {
    "heat_1d": _make_devito_heat_1d,
    "heat_2d": _make_devito_heat_2d,
    "laplacian_2d_5pt": _make_devito_laplacian_2d_5pt,
    "laplacian_3d_7pt": _make_devito_laplacian_3d_7pt,
}


def bench_devito(
    name: str,
    domain: tuple[int, ...],
    warmup: int = 5,
    iters: int = 20,
) -> dict:
    """Benchmark a Devito stencil. Returns dict with time_ms, gpoints_per_s, gbytes_per_s."""
    if name not in _DEVITO_BUILDERS:
        raise ValueError(f"No Devito builder for {name!r}. Available: {list(_DEVITO_BUILDERS)}")

    meta = STENCIL_META[name]
    halo = meta["halo"]
    shape = tuple(d + 2 * h for d, h in zip(domain, halo))

    op, u = _DEVITO_BUILDERS[name](shape)
    u.data[:] = np.random.randn(*u.data.shape).astype(np.float64)

    for _ in range(warmup):
        op.apply(time_M=1)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        op.apply(time_M=1)
        times.append(time.perf_counter() - t0)

    elapsed_ms = float(np.median(times)) * 1000
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "name": name,
        "framework": "Devito",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
    }
