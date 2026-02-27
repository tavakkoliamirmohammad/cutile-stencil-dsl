"""Core dataclasses for the stencil DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class OffsetAccess:
    """A single array access with integer offsets, e.g. u[i+1] → ('u', (1,))."""
    array_name: str
    offsets: Tuple[int, ...]


@dataclass
class StencilSpec:
    """Complete specification of a stencil extracted from a decorated function."""
    name: str
    ndim: int
    order: int
    inputs: Tuple[str, ...]
    output: str
    update_fn: Callable
    accesses: List[OffsetAccess] = field(default_factory=list)
    dtype: str = "float64"
    tile_sizes: Optional[Tuple[int, ...]] = None
    halo_widths: Optional[Tuple[int, ...]] = None
    temporal_steps: int = 1


@dataclass(frozen=True)
class HardwareSpec:
    """Target GPU hardware parameters for tiling analysis."""
    peak_bandwidth_gbs: float = 1000.0   # GB/s
    shared_mem_bytes: int = 49152        # 48 KiB default
    sm_count: int = 108
    dtype_bytes: int = 8                 # float64 = 8


@dataclass
class TileConfig:
    """Result of tile decomposition analysis."""
    tile_sizes: Tuple[int, ...]
    halo_widths: Tuple[int, ...]
    num_tiles: Tuple[int, ...]
    overhead_fraction: float


@dataclass
class TemporalConfig:
    """Result of temporal blocking analysis."""
    steps: int
    expanded_halo: Tuple[int, ...]
    bandwidth_reduction_factor: float


@dataclass
class RooflineResult:
    """Analytical roofline model result."""
    flops_per_point: int
    bytes_per_point: int
    arithmetic_intensity: float
    bound: str          # "memory" or "compute"
    peak_gpoints_s: float
