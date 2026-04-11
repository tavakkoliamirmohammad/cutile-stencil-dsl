"""Tile decomposition and halo overhead optimisation."""

from __future__ import annotations

import itertools
import math
from typing import List, Optional, Tuple

from cutile_stencil.dsl.types import HardwareSpec, StencilSpec, TileConfig


def _tile_memory(tile_sizes: Tuple[int, ...], halo: Tuple[int, ...],
                 dtype_bytes: int, num_arrays: int = 2) -> int:
    """Shared-memory footprint for a tile with halos (input + output)."""
    expanded = 1
    for t, h in zip(tile_sizes, halo):
        expanded *= (t + 2 * h)
    base = 1
    for t in tile_sizes:
        base *= t
    return (expanded + base) * dtype_bytes * num_arrays


def _overhead(tile_sizes: Tuple[int, ...], halo: Tuple[int, ...]) -> float:
    """Fraction of loaded data that is redundant halo."""
    expanded = 1
    base = 1
    for t, h in zip(tile_sizes, halo):
        expanded *= (t + 2 * h)
        base *= t
    if expanded == 0:
        return 1.0
    return 1.0 - base / expanded


def compute_tile_config(
    spec: StencilSpec,
    domain_size: Tuple[int, ...],
    hw: HardwareSpec | None = None,
    candidate_sizes: List[int] | None = None,
    candidate_mode: str = "power_of_2",
) -> TileConfig:
    """Find optimal tile sizes that fit in shared memory.

    Enumerates candidate tile sizes via N-dimensional product,
    picks the configuration that minimises halo overhead while fitting
    in the shared-memory budget.

    Parameters
    ----------
    spec : StencilSpec
        Stencil specification with ndim and halo info.
    domain_size : tuple of int
        Domain extents per dimension.
    hw : HardwareSpec, optional
        GPU hardware parameters. Defaults to HardwareSpec().
    candidate_sizes : list of int, optional
        Tile size candidates to enumerate. When provided, ``candidate_mode``
        is ignored.
    candidate_mode : str, optional
        Predefined set of tile size candidates to use when ``candidate_sizes``
        is None. One of ``"power_of_2"``, ``"multiples_of_32"``, or
        ``"multiples_of_16"``. Defaults to ``"power_of_2"``.
    """
    if hw is None:
        hw = HardwareSpec()

    if candidate_sizes is None:
        _CANDIDATE_MODES = {
            "power_of_2": [32, 64, 128, 256, 512, 1024],
            "multiples_of_32": [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512],
            "multiples_of_16": [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512],
        }
        if candidate_mode not in _CANDIDATE_MODES:
            raise ValueError(
                f"Unknown candidate_mode '{candidate_mode}'. "
                f"Choose from: {list(_CANDIDATE_MODES.keys())}"
            )
        candidate_sizes = _CANDIDATE_MODES[candidate_mode]

    ndim = spec.ndim
    halo = spec.halo_widths or (spec.order // 2,) * ndim
    budget = hw.shared_mem_bytes
    num_arrays = len(spec.inputs) + 1  # inputs + output

    best: TileConfig | None = None
    best_overhead = float("inf")

    # Unified N-dim enumeration via itertools.product
    for combo in itertools.product(candidate_sizes, repeat=ndim):
        tile = combo
        mem = _tile_memory(tile, halo, hw.dtype_bytes, num_arrays)
        if mem <= budget:
            oh = _overhead(tile, halo)
            if oh < best_overhead:
                best_overhead = oh
                nt = tuple(math.ceil(d / t) for d, t in zip(domain_size, tile))
                best = TileConfig(tile, halo, nt, oh)

    if best is None:
        # Fallback: smallest possible tile
        tile = (32,) * ndim
        nt = tuple(math.ceil(d / 32) for d in domain_size)
        best = TileConfig(tile, halo, nt, _overhead(tile, halo))

    return best
