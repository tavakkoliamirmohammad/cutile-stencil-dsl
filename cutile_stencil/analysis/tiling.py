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
) -> TileConfig:
    """Find optimal power-of-2 tile sizes that fit in shared memory.

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
        Tile size candidates to enumerate. Defaults to [32, 64, 128, 256, 512, 1024].
    """
    if hw is None:
        hw = HardwareSpec()

    if candidate_sizes is None:
        candidate_sizes = [32, 64, 128, 256, 512, 1024]

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

    spec.tile_sizes = best.tile_sizes
    spec.halo_widths = best.halo_widths
    return best
