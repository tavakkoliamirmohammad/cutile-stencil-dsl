"""Temporal blocking analysis.

Given a tile configuration, compute the maximum number of time steps
that can be fused into a single kernel launch while keeping the expanded
tile (with growing halo) inside shared memory.
"""

from __future__ import annotations

from cutile_stencil.dsl.types import HardwareSpec, StencilSpec, TileConfig, TemporalConfig


def compute_temporal_config(
    spec: StencilSpec,
    tile_cfg: TileConfig,
    hw: HardwareSpec | None = None,
    max_steps: int = 16,
) -> TemporalConfig:
    """Find the largest temporal blocking factor T such that the expanded
    tile ``(tile + 2*T*halo)^ndim * dtype_bytes`` still fits in shared memory.
    """
    if hw is None:
        hw = HardwareSpec()

    halo = tile_cfg.halo_widths
    tile = tile_cfg.tile_sizes
    num_arrays = len(spec.inputs) + 1

    best_t = 1
    for t in range(1, max_steps + 1):
        expanded_volume = 1
        for ts, h in zip(tile, halo):
            expanded_volume *= (ts + 2 * t * h)
        base_volume = 1
        for ts in tile:
            base_volume *= ts
        # Need space for expanded input + base output
        mem = (expanded_volume + base_volume) * hw.dtype_bytes * num_arrays
        if mem <= hw.shared_mem_bytes:
            best_t = t
        else:
            break

    expanded_halo = tuple(h * best_t for h in halo)
    # Bandwidth reduction: we load once for T steps instead of T loads
    reduction = best_t if best_t > 0 else 1

    return TemporalConfig(
        steps=best_t,
        expanded_halo=expanded_halo,
        bandwidth_reduction_factor=float(reduction),
    )
