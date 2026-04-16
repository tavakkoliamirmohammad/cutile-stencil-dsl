"""Canonical stencil definitions for all baselines.

Each stencil is defined as:
1. A cuTile @stencil decorated function (for compiler benchmarks)
2. A plain numpy-style function (for CuPy/NumPy baselines)
3. Metadata (flops, loads, arithmetic intensity)
"""

from __future__ import annotations

import math

from cutile import stencil


# -- cuTile stencil definitions --

@stencil(ndim=1, order=2)
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


@stencil(ndim=2, order=2)
def heat_2d(u, i, j):
    return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


@stencil(ndim=2, order=2)
def laplacian_2d_5pt(u, i, j):
    return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]


@stencil(ndim=2, order=4)
def laplacian_2d_9pt(u, i, j):
    return (
        -u[i - 2, j] + 16 * u[i - 1, j] - 30 * u[i, j] + 16 * u[i + 1, j] - u[i + 2, j]
        - u[i, j - 2] + 16 * u[i, j - 1] - 30 * u[i, j] + 16 * u[i, j + 1] - u[i, j + 2]
    ) / 12.0


@stencil(ndim=3, order=2)
def laplacian_3d_7pt(u, i, j, k):
    return (
        u[i - 1, j, k] + u[i + 1, j, k]
        + u[i, j - 1, k] + u[i, j + 1, k]
        + u[i, j, k - 1] + u[i, j, k + 1]
        - 6 * u[i, j, k]
    )


# -- NumPy/CuPy-style slice functions --
# Operate on the full array (with halo) and return the interior.

def heat_1d_sliced(u):
    return 0.25 * u[:-2] + 0.5 * u[1:-1] + 0.25 * u[2:]


def heat_2d_sliced(u):
    return 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])


def laplacian_2d_5pt_sliced(u):
    return (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        - 4 * u[1:-1, 1:-1]
    )


def laplacian_2d_9pt_sliced(u):
    return (
        -u[:-4, 2:-2] + 16 * u[1:-3, 2:-2] - 30 * u[2:-2, 2:-2]
        + 16 * u[3:-1, 2:-2] - u[4:, 2:-2]
        - u[2:-2, :-4] + 16 * u[2:-2, 1:-3] - 30 * u[2:-2, 2:-2]
        + 16 * u[2:-2, 3:-1] - u[2:-2, 4:]
    ) / 12.0


def laplacian_3d_7pt_sliced(u):
    return (
        u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]
        - 6 * u[1:-1, 1:-1, 1:-1]
    )


# -- Stencil metadata --

STENCIL_META = {
    "heat_1d": {
        "cutile_fn": heat_1d,
        "slice_fn": heat_1d_sliced,
        "ndim": 1,
        "order": 2,
        "halo": (1,),
        "flops_per_point": 4,
        "loads_per_point": 3,
        "stores_per_point": 1,
        "dtype_bytes": 8,
    },
    "heat_2d": {
        "cutile_fn": heat_2d,
        "slice_fn": heat_2d_sliced,
        "ndim": 2,
        "order": 2,
        "halo": (1, 1),
        "flops_per_point": 4,
        "loads_per_point": 4,
        "stores_per_point": 1,
        "dtype_bytes": 8,
    },
    "laplacian_2d_5pt": {
        "cutile_fn": laplacian_2d_5pt,
        "slice_fn": laplacian_2d_5pt_sliced,
        "ndim": 2,
        "order": 2,
        "halo": (1, 1),
        "flops_per_point": 6,
        "loads_per_point": 5,
        "stores_per_point": 1,
        "dtype_bytes": 8,
    },
    "laplacian_2d_9pt": {
        "cutile_fn": laplacian_2d_9pt,
        "slice_fn": laplacian_2d_9pt_sliced,
        "ndim": 2,
        "order": 4,
        "halo": (2, 2),
        "flops_per_point": 18,
        "loads_per_point": 9,
        "stores_per_point": 1,
        "dtype_bytes": 8,
    },
    "laplacian_3d_7pt": {
        "cutile_fn": laplacian_3d_7pt,
        "slice_fn": laplacian_3d_7pt_sliced,
        "ndim": 3,
        "order": 2,
        "halo": (1, 1, 1),
        "flops_per_point": 8,
        "loads_per_point": 7,
        "stores_per_point": 1,
        "dtype_bytes": 8,
    },
}

# Domain sizes per dimensionality
DOMAINS = {
    1: [(2**16,), (2**18,), (2**20,), (2**22,), (2**24,)],
    2: [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
    3: [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)],
}


def full_shape(domain: tuple[int, ...], halo: tuple[int, ...]) -> tuple[int, ...]:
    """Domain + halo padding on each side."""
    return tuple(d + 2 * h for d, h in zip(domain, halo))


def interior_size(domain: tuple[int, ...]) -> int:
    """Number of interior points."""
    return math.prod(domain)


def arithmetic_intensity(name: str) -> float:
    """Operational intensity in flops/byte."""
    m = STENCIL_META[name]
    return m["flops_per_point"] / ((m["loads_per_point"] + m["stores_per_point"]) * m["dtype_bytes"])
