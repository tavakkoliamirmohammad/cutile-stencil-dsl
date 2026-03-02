"""Core dataclasses for the stencil DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class OffsetAccess:
    """A single array access with integer offsets, e.g. u[i+1] -> ('u', (1,))."""
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
    boundary: object = None  # BoundarySpec, set to object to avoid circular import
    closure_constants: dict = field(default_factory=dict)  # Captured scope variables

    def validate(self) -> List[str]:
        """Return a list of warnings about potential issues."""
        warnings = []
        if not self.accesses:
            warnings.append("No accesses extracted; run extract_footprint() first")
        if self.halo_widths is None:
            warnings.append("Halo widths not set; run compute_halo() first")
        elif all(h == 0 for h in self.halo_widths):
            warnings.append("All halo widths are zero; stencil may be trivial")
        if len(self.inputs) == 0:
            warnings.append("No input arrays specified")
        return warnings


@dataclass(frozen=True)
class HardwareSpec:
    """Target GPU hardware parameters for tiling analysis."""
    peak_bandwidth_gbs: float = 1000.0   # GB/s
    shared_mem_bytes: int = 49152        # 48 KiB default
    sm_count: int = 108
    dtype_bytes: int = 8                 # float64 = 8
    peak_gflops: float = 10000.0         # Peak GFLOPS

    @classmethod
    def from_preset(cls, preset, dtype: str = "float64") -> HardwareSpec:
        """Create a HardwareSpec from a GPUPreset.

        Parameters
        ----------
        preset : GPUPreset or str
            A GPUPreset instance or a preset name (e.g., "H100_SXM").
        dtype : str
            Data type to select peak GFLOPS (float64 or float32).
        """
        if isinstance(preset, str):
            from cutile_stencil.config import GPU_PRESETS
            preset = GPU_PRESETS[preset]
        dtype_bytes = 4 if dtype in ("float32", "fp32") else 8
        return cls(
            peak_bandwidth_gbs=preset.peak_bandwidth_gbs,
            shared_mem_bytes=preset.shared_mem_bytes,
            sm_count=preset.sm_count,
            dtype_bytes=dtype_bytes,
            peak_gflops=preset.peak_gflops(dtype),
        )


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


@dataclass(frozen=True)
class BrickLayout:
    """Brick decomposition layout parameters.

    The domain is divided into bricks of size ``brick_sizes`` per dimension.
    Each brick is stored with halo padding from its neighbours so that stencil
    application is self-contained within a brick.

    Bricked shape: ``(nb0, nb1, ..., B0+2*h0, B1+2*h1, ...)``
    where outer dims are brick indices and inner dims are padded brick data.
    """
    brick_sizes: Tuple[int, ...]

    def validate(self, ndim: int, halo_widths: Tuple[int, ...]) -> None:
        """Raise ValueError if brick_sizes don't match ndim or are too small."""
        if len(self.brick_sizes) != ndim:
            raise ValueError(
                f"brick_sizes has {len(self.brick_sizes)} dims, expected {ndim}"
            )
        for d, (bs, hw) in enumerate(zip(self.brick_sizes, halo_widths)):
            if bs < 2 * hw:
                raise ValueError(
                    f"brick_sizes[{d}]={bs} < 2*halo={2 * hw}"
                )

    def num_bricks(self, domain: Tuple[int, ...]) -> Tuple[int, ...]:
        """Number of bricks per dimension (domain must be divisible)."""
        result = []
        for d, (n, bs) in enumerate(zip(domain, self.brick_sizes)):
            if n % bs != 0:
                raise ValueError(
                    f"domain[{d}]={n} not divisible by brick_sizes[{d}]={bs}"
                )
            result.append(n // bs)
        return tuple(result)

    def bricked_shape(
        self, domain: Tuple[int, ...], halo_widths: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Shape of the bricked array: (nb0, nb1, ..., B0+2*h0, B1+2*h1, ...)."""
        nbs = self.num_bricks(domain)
        padded = tuple(bs + 2 * h for bs, h in zip(self.brick_sizes, halo_widths))
        return nbs + padded
