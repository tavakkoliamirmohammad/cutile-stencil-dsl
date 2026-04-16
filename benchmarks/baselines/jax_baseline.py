"""JAX/XLA baseline: JIT-compiled stencils via array slicing."""

from __future__ import annotations

import os
import time

import numpy as np

from benchmarks.stencils import STENCIL_META, interior_size

# Set CUDA path for XLA before importing JAX
for _cuda_path in [
    os.environ.get("CUDA_HOME", ""),
    "/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-x86_64/gcc-8.5.0/cuda-13.2.0-hc3gf4kzlz2qzpq6r5kekfbdxgjccyou",
    "/usr/local/cuda",
]:
    if _cuda_path and os.path.exists(_cuda_path):
        os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={_cuda_path}")
        break

import jax
import jax.numpy as jnp


@jax.jit
def _jax_heat_1d(u):
    return 0.25 * u[:-2] + 0.5 * u[1:-1] + 0.25 * u[2:]


@jax.jit
def _jax_heat_2d(u):
    return 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])


@jax.jit
def _jax_laplacian_2d_5pt(u):
    return (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        - 4 * u[1:-1, 1:-1]
    )


@jax.jit
def _jax_laplacian_2d_9pt(u):
    return (
        -u[:-4, 2:-2] + 16 * u[1:-3, 2:-2] - 30 * u[2:-2, 2:-2]
        + 16 * u[3:-1, 2:-2] - u[4:, 2:-2]
        - u[2:-2, :-4] + 16 * u[2:-2, 1:-3] - 30 * u[2:-2, 2:-2]
        + 16 * u[2:-2, 3:-1] - u[2:-2, 4:]
    ) / 12.0


@jax.jit
def _jax_laplacian_3d_7pt(u):
    return (
        u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]
        - 6 * u[1:-1, 1:-1, 1:-1]
    )


_JAX_FNS = {
    "heat_1d": _jax_heat_1d,
    "heat_2d": _jax_heat_2d,
    "laplacian_2d_5pt": _jax_laplacian_2d_5pt,
    "laplacian_2d_9pt": _jax_laplacian_2d_9pt,
    "laplacian_3d_7pt": _jax_laplacian_3d_7pt,
}


def bench_jax(
    name: str,
    shape: tuple[int, ...],
    warmup: int = 30,
    iters: int = 100,
) -> dict:
    """Benchmark a JAX stencil. Returns dict with time_ms, gpoints_per_s, gbytes_per_s."""
    fn = _JAX_FNS[name]
    meta = STENCIL_META[name]
    u = jnp.array(np.random.randn(*shape).astype(np.float64))

    for _ in range(warmup):
        _ = fn(u).block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(u).block_until_ready()
        times.append(time.perf_counter() - t0)

    elapsed_ms = float(np.median(times)) * 1000
    domain = tuple(s - 2 * h for s, h in zip(shape, meta["halo"]))
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "name": name,
        "framework": "JAX",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
    }
