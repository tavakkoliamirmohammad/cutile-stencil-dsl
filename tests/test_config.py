"""Tests for the configuration system."""

import pytest
from cutile_stencil.config import (
    GPUPreset, GPU_PRESETS, auto_detect_gpu,
    H100_SXM, A100_80GB, V100, RTX4090,
    TilingConfig, BenchmarkConfig, SolverConfig,
    DEFAULT_TILING, DEFAULT_BENCHMARK, DEFAULT_SOLVER,
)
from cutile_stencil.dsl.types import HardwareSpec


class TestGPUPresets:
    def test_all_presets_exist(self):
        expected = {"H100_SXM", "H100_PCIe", "A100_80GB", "A100_40GB", "V100", "RTX4090",
                    "RTX_PRO_6000", "RTX_PRO_6000_MAXQ"}
        assert set(GPU_PRESETS.keys()) == expected

    def test_h100_sxm_specs(self):
        h = H100_SXM
        assert h.sm_count == 132
        assert h.peak_bandwidth_gbs == 3350.0
        assert h.shared_mem_bytes == 232 * 1024

    def test_preset_peak_gflops_dtype(self):
        h = H100_SXM
        fp64 = h.peak_gflops("float64")
        fp32 = h.peak_gflops("float32")
        assert fp32 > fp64  # FP32 is always faster

    def test_v100_specs(self):
        assert V100.sm_count == 80
        assert V100.shared_mem_bytes == 96 * 1024

    def test_rtx4090_low_fp64(self):
        # RTX4090 has poor FP64 but great FP32
        assert RTX4090.peak_gflops_fp64 < RTX4090.peak_gflops_fp32


class TestHardwareSpecFromPreset:
    def test_from_preset_object(self):
        hw = HardwareSpec.from_preset(H100_SXM)
        assert hw.sm_count == 132
        assert hw.shared_mem_bytes == 232 * 1024
        assert hw.peak_gflops == H100_SXM.peak_gflops_fp64

    def test_from_preset_string(self):
        hw = HardwareSpec.from_preset("A100_80GB")
        assert hw.sm_count == 108
        assert hw.peak_bandwidth_gbs == 2039.0

    def test_from_preset_fp32(self):
        hw = HardwareSpec.from_preset(H100_SXM, dtype="float32")
        assert hw.dtype_bytes == 4
        assert hw.peak_gflops == H100_SXM.peak_gflops_fp32

    def test_from_preset_invalid(self):
        with pytest.raises(KeyError):
            HardwareSpec.from_preset("NOT_A_GPU")


class TestTilingConfig:
    def test_defaults(self):
        tc = TilingConfig()
        assert tc.candidate_sizes == [32, 64, 128, 256, 512, 1024]
        assert tc.max_temporal_steps == 16

    def test_custom(self):
        tc = TilingConfig(candidate_sizes=[16, 32], max_temporal_steps=8)
        assert len(tc.candidate_sizes) == 2
        assert tc.max_temporal_steps == 8


class TestBenchmarkConfig:
    def test_defaults(self):
        bc = BenchmarkConfig()
        assert bc.grid_1d == 1024 * 1024
        assert bc.grid_2d == (1024, 1024)
        assert bc.grid_3d == (64, 64, 64)

    def test_custom(self):
        bc = BenchmarkConfig(grid_1d=256, grid_2d=(128, 128))
        assert bc.grid_1d == 256


class TestSolverConfig:
    def test_defaults(self):
        sc = SolverConfig()
        assert sc.cg_tol == 1e-10
        assert sc.mixed_inner_tol == 1e-4
        assert sc.mixed_outer_tol == 1e-10
        assert sc.persistent_spinlock_size == 4

    def test_custom(self):
        sc = SolverConfig(cg_tol=1e-8, cg_max_iter=500)
        assert sc.cg_tol == 1e-8
        assert sc.cg_max_iter == 500


class TestAutoDetectGPU:
    def test_returns_none_or_preset(self):
        # On machines without GPU, should return None
        result = auto_detect_gpu()
        assert result is None or isinstance(result, GPUPreset)
