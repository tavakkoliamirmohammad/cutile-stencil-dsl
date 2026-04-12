"""Configuration system -- GPU presets, hardware specs, tiling, and benchmark configs.

Provides configurable dataclasses with sensible defaults for GPU hardware
specifications, tile size enumeration, and benchmarking parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPU preset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUPreset:
    """Hardware specification for a specific GPU model."""

    name: str
    peak_bandwidth_gbs: float
    shared_mem_bytes: int
    sm_count: int
    peak_gflops_fp64: float
    peak_gflops_fp32: float

    def peak_gflops(self, dtype: str = "float64") -> float:
        """Return peak GFLOPS for the given dtype string."""
        if dtype in ("float32", "fp32"):
            return self.peak_gflops_fp32
        return self.peak_gflops_fp64


# ---------------------------------------------------------------------------
# GPU presets from official datasheets
# ---------------------------------------------------------------------------

H100_SXM = GPUPreset(
    name="H100_SXM",
    peak_bandwidth_gbs=3350.0,
    shared_mem_bytes=232 * 1024,
    sm_count=132,
    peak_gflops_fp64=33500.0,
    peak_gflops_fp32=67000.0,
)

H100_PCIe = GPUPreset(
    name="H100_PCIe",
    peak_bandwidth_gbs=2039.0,
    shared_mem_bytes=232 * 1024,
    sm_count=114,
    peak_gflops_fp64=24000.0,
    peak_gflops_fp32=48000.0,
)

A100_80GB = GPUPreset(
    name="A100_80GB",
    peak_bandwidth_gbs=2039.0,
    shared_mem_bytes=164 * 1024,
    sm_count=108,
    peak_gflops_fp64=9746.0,
    peak_gflops_fp32=19492.0,
)

A100_40GB = GPUPreset(
    name="A100_40GB",
    peak_bandwidth_gbs=1555.0,
    shared_mem_bytes=164 * 1024,
    sm_count=108,
    peak_gflops_fp64=9746.0,
    peak_gflops_fp32=19492.0,
)

V100 = GPUPreset(
    name="V100",
    peak_bandwidth_gbs=900.0,
    shared_mem_bytes=96 * 1024,
    sm_count=80,
    peak_gflops_fp64=7066.0,
    peak_gflops_fp32=14131.0,
)

RTX4090 = GPUPreset(
    name="RTX4090",
    peak_bandwidth_gbs=1008.0,
    shared_mem_bytes=100 * 1024,
    sm_count=128,
    peak_gflops_fp64=1290.0,
    peak_gflops_fp32=82580.0,
)

RTX_PRO_6000 = GPUPreset(
    name="RTX_PRO_6000",
    peak_bandwidth_gbs=1792.0,
    shared_mem_bytes=228 * 1024,
    sm_count=188,
    peak_gflops_fp64=1953.0,
    peak_gflops_fp32=125000.0,
)

GPU_PRESETS: Dict[str, GPUPreset] = {
    "H100_SXM": H100_SXM,
    "H100_PCIe": H100_PCIe,
    "A100_80GB": A100_80GB,
    "A100_40GB": A100_40GB,
    "V100": V100,
    "RTX4090": RTX4090,
    "RTX_PRO_6000": RTX_PRO_6000,
}


# ---------------------------------------------------------------------------
# Hardware spec (runtime-resolved from preset or auto-detect)
# ---------------------------------------------------------------------------

@dataclass
class HardwareSpec:
    """Resolved hardware parameters used by the analysis and codegen layers."""

    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    peak_bandwidth_gbs: float = 900.0
    peak_gflops: float = 7066.0
    sm_count: int = 80

    @classmethod
    def from_preset(cls, name: str, dtype: str = "float64") -> HardwareSpec:
        """Create a HardwareSpec from a named GPU preset."""
        preset = GPU_PRESETS[name]
        dtype_bytes = 4 if dtype in ("float32", "fp32") else 8
        return cls(
            shared_mem_bytes=preset.shared_mem_bytes,
            dtype_bytes=dtype_bytes,
            peak_bandwidth_gbs=preset.peak_bandwidth_gbs,
            peak_gflops=preset.peak_gflops(dtype),
            sm_count=preset.sm_count,
        )

    @classmethod
    def auto_detect(cls, dtype: str = "float64") -> HardwareSpec:
        """Try to detect the current GPU and build a HardwareSpec.

        Tries pynvml first, then torch.cuda. Falls back to V100 defaults
        if no GPU can be detected.
        """
        preset = _auto_detect_gpu()
        if preset is None:
            # Fall back to conservative V100 defaults
            preset = V100
        dtype_bytes = 4 if dtype in ("float32", "fp32") else 8
        return cls(
            shared_mem_bytes=preset.shared_mem_bytes,
            dtype_bytes=dtype_bytes,
            peak_bandwidth_gbs=preset.peak_bandwidth_gbs,
            peak_gflops=preset.peak_gflops(dtype),
            sm_count=preset.sm_count,
        )


def _auto_detect_gpu() -> Optional[GPUPreset]:
    """Try to detect the current GPU and return a matching preset.

    Tries pynvml first, then torch.cuda.  Returns ``None`` if no GPU is
    detected or the GPU name does not match any known preset.
    """
    gpu_name: Optional[str] = None

    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Try torch.cuda
    if gpu_name is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    if gpu_name is None:
        return None

    gpu_upper = gpu_name.upper()
    for preset_name, preset in GPU_PRESETS.items():
        if (preset_name.replace("_", " ") in gpu_upper
                or preset_name.replace("_", "") in gpu_upper):
            return preset

    # Heuristic matching for common GPU families
    if "RTX PRO 6000" in gpu_upper or "RTXPRO6000" in gpu_upper:
        return RTX_PRO_6000
    if "H100" in gpu_upper:
        return H100_SXM if "SXM" in gpu_upper else H100_PCIe
    if "A100" in gpu_upper:
        return A100_80GB if "80" in gpu_upper else A100_40GB
    if "V100" in gpu_upper:
        return V100
    if "4090" in gpu_upper:
        return RTX4090

    return None


# ---------------------------------------------------------------------------
# Tiling and benchmark configs
# ---------------------------------------------------------------------------

@dataclass
class TilingConfig:
    """Configuration for tile size enumeration."""

    candidate_sizes: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024],
    )
    max_temporal_steps: int = 16


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters."""

    grid_sizes: Dict[int, Tuple[int, ...]] = field(default_factory=lambda: {
        1: (1024 * 1024,),
        2: (1024, 1024),
        3: (64, 64, 64),
    })
    warmup_iters: int = 5
    benchmark_iters: int = 20


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_HW = HardwareSpec()
DEFAULT_TILING = TilingConfig()
DEFAULT_BENCHMARK = BenchmarkConfig()
