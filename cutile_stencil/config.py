"""Configuration system — GPU presets, tiling, benchmark, and solver configs.

Eliminates all hardcoded GPU specs, tile sizes, benchmark grids, and solver
tolerances by providing configurable dataclasses with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GPUPreset:
    """Hardware specification for a specific GPU model."""
    name: str
    peak_bandwidth_gbs: float
    shared_mem_bytes: int
    sm_count: int
    peak_gflops_fp64: float
    peak_gflops_fp32: float
    dtype_bytes: int = 8

    def peak_gflops(self, dtype: str = "float64") -> float:
        if dtype in ("float32", "fp32"):
            return self.peak_gflops_fp32
        return self.peak_gflops_fp64


# GPU presets from official datasheets
H100_SXM = GPUPreset(
    name="H100_SXM",
    peak_bandwidth_gbs=3350.0,
    shared_mem_bytes=232 * 1024,  # 232 KiB configurable
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
    shared_mem_bytes=164 * 1024,  # 164 KiB configurable
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
    shared_mem_bytes=96 * 1024,  # 96 KiB configurable
    sm_count=80,
    peak_gflops_fp64=7066.0,
    peak_gflops_fp32=14131.0,
)

RTX4090 = GPUPreset(
    name="RTX4090",
    peak_bandwidth_gbs=1008.0,
    shared_mem_bytes=100 * 1024,  # 100 KiB
    sm_count=128,
    peak_gflops_fp64=1290.0,
    peak_gflops_fp32=82580.0,
)

# NVIDIA RTX PRO 6000 Blackwell (GB202 full die)
# Specs from NVIDIA datasheet: 24064 CUDA cores, 188 SMs, 96 GB GDDR7, 512-bit bus
# FP32: 125 TFLOPS (workstation edition), FP64: 1/64 rate (RTX-class)
# Max-Q variant: 300W TDP, may sustain ~80% of peak clocks under load
RTX_PRO_6000 = GPUPreset(
    name="RTX_PRO_6000",
    peak_bandwidth_gbs=1792.0,    # 512-bit GDDR7 @ 14 Gbps per pin
    shared_mem_bytes=228 * 1024,  # 228 KiB per SM (Blackwell architecture)
    sm_count=188,
    peak_gflops_fp64=1953.0,      # 125000/64 — RTX-class 1/64 FP64 rate
    peak_gflops_fp32=125000.0,    # 125 TFLOPS from datasheet
)

# Max-Q variant: lower sustained clocks due to 300W power limit (vs 600W full)
RTX_PRO_6000_MAXQ = GPUPreset(
    name="RTX_PRO_6000_MAXQ",
    peak_bandwidth_gbs=1792.0,    # Same memory subsystem
    shared_mem_bytes=228 * 1024,  # Same Blackwell SM
    sm_count=188,
    peak_gflops_fp64=1562.0,      # ~80% of full edition under power limit
    peak_gflops_fp32=100000.0,    # ~80% of 125 TFLOPS
)

GPU_PRESETS: Dict[str, GPUPreset] = {
    "H100_SXM": H100_SXM,
    "H100_PCIe": H100_PCIe,
    "A100_80GB": A100_80GB,
    "A100_40GB": A100_40GB,
    "V100": V100,
    "RTX4090": RTX4090,
    "RTX_PRO_6000": RTX_PRO_6000,
    "RTX_PRO_6000_MAXQ": RTX_PRO_6000_MAXQ,
}


def auto_detect_gpu() -> Optional[GPUPreset]:
    """Try to detect the current GPU and return a matching preset.

    Tries pynvml first, then torch.cuda. Returns None if no GPU detected.
    """
    gpu_name = None

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

    gpu_name_upper = gpu_name.upper()
    for preset_name, preset in GPU_PRESETS.items():
        if preset_name.replace("_", " ") in gpu_name_upper or preset_name.replace("_", "") in gpu_name_upper:
            return preset

    # Heuristic matching
    if "RTX PRO 6000" in gpu_name_upper or "RTXPRO6000" in gpu_name_upper:
        return RTX_PRO_6000_MAXQ if "MAX-Q" in gpu_name_upper or "MAXQ" in gpu_name_upper else RTX_PRO_6000
    if "H100" in gpu_name_upper:
        return H100_SXM if "SXM" in gpu_name_upper else H100_PCIe
    if "A100" in gpu_name_upper:
        return A100_80GB if "80" in gpu_name_upper else A100_40GB
    if "V100" in gpu_name_upper:
        return V100
    if "4090" in gpu_name_upper:
        return RTX4090

    return None


@dataclass
class TilingConfig:
    """Configuration for tile size enumeration."""
    candidate_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024])
    candidate_mode: str = "power_of_2"
    max_temporal_steps: int = 16


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark grid sizes."""
    grid_1d: int = 1024 * 1024
    grid_2d: Tuple[int, int] = (1024, 1024)
    grid_3d: Tuple[int, int, int] = (64, 64, 64)
    warmup_iterations: int = 5
    benchmark_iterations: int = 20


@dataclass
class SolverConfig:
    """Configuration for iterative solvers."""
    cg_tol: float = 1e-10
    cg_max_iter: int = 1000
    mixed_inner_tol: float = 1e-4
    mixed_outer_tol: float = 1e-10
    mixed_max_outer: int = 20
    mixed_max_inner: int = 200
    persistent_spinlock_size: int = 4
    tile_size: int = 256
    dtype: str = "float64"


# Default configs
DEFAULT_TILING = TilingConfig()
DEFAULT_BENCHMARK = BenchmarkConfig()
DEFAULT_SOLVER = SolverConfig()
