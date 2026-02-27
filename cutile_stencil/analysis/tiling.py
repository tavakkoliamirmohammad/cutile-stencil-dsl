"""Tile decomposition and halo overhead optimisation."""

from __future__ import annotations

import math
from typing import Tuple

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
) -> TileConfig:
    """Find optimal power-of-2 tile sizes that fit in shared memory.

    Enumerates candidate tile sizes from 32 up to the domain extent,
    picks the configuration that minimises halo overhead while fitting
    in the shared-memory budget.
    """
    if hw is None:
        hw = HardwareSpec()

    ndim = spec.ndim
    halo = spec.halo_widths or (spec.order // 2,) * ndim
    budget = hw.shared_mem_bytes
    num_arrays = len(spec.inputs) + 1  # inputs + output

    candidates = [32, 64, 128, 256, 512, 1024]

    best: TileConfig | None = None
    best_overhead = float("inf")

    if ndim == 1:
        for ts in candidates:
            tile = (ts,)
            mem = _tile_memory(tile, halo, hw.dtype_bytes, num_arrays)
            if mem <= budget:
                oh = _overhead(tile, halo)
                if oh < best_overhead:
                    best_overhead = oh
                    nt = (math.ceil(domain_size[0] / ts),)
                    best = TileConfig(tile, halo, nt, oh)
    elif ndim == 2:
        for tx in candidates:
            for ty in candidates:
                tile = (tx, ty)
                mem = _tile_memory(tile, halo, hw.dtype_bytes, num_arrays)
                if mem <= budget:
                    oh = _overhead(tile, halo)
                    if oh < best_overhead:
                        best_overhead = oh
                        nt = (math.ceil(domain_size[0] / tx),
                              math.ceil(domain_size[1] / ty))
                        best = TileConfig(tile, halo, nt, oh)
    else:  # 3D
        for tx in candidates:
            for ty in candidates:
                for tz in candidates:
                    tile = (tx, ty, tz)
                    mem = _tile_memory(tile, halo, hw.dtype_bytes, num_arrays)
                    if mem <= budget:
                        oh = _overhead(tile, halo)
                        if oh < best_overhead:
                            best_overhead = oh
                            nt = (math.ceil(domain_size[0] / tx),
                                  math.ceil(domain_size[1] / ty),
                                  math.ceil(domain_size[2] / tz))
                            best = TileConfig(tile, halo, nt, oh)

    if best is None:
        # Fallback: smallest possible tile
        tile = (32,) * ndim
        nt = tuple(math.ceil(d / 32) for d in domain_size)
        best = TileConfig(tile, halo, nt, _overhead(tile, halo))

    spec.tile_sizes = best.tile_sizes
    spec.halo_widths = best.halo_widths
    return best
