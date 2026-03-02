"""Tests for tile configuration and temporal blocking analysis."""

import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec, TileConfig
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def _make_heat_spec():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    spec = heat._stencil_spec
    assert spec.ndim == 1
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)
    return spec


def _make_lap2d_spec():
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
    spec = lap2d._stencil_spec
    assert spec.ndim == 2
    assert spec.order == 2
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 2)
    return spec


def test_tile_config_1d():
    spec = _make_heat_spec()
    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    cfg = compute_tile_config(spec, (1024,), hw)
    assert isinstance(cfg, TileConfig)
    assert len(cfg.tile_sizes) == 1
    assert cfg.tile_sizes[0] >= 32
    assert cfg.overhead_fraction >= 0
    assert cfg.overhead_fraction < 1


def test_tile_fits_in_shared_memory():
    spec = _make_heat_spec()
    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    cfg = compute_tile_config(spec, (1024,), hw)
    # Check memory: (tile + 2*halo) * 2_arrays * dtype_bytes <= budget
    expanded = cfg.tile_sizes[0] + 2 * cfg.halo_widths[0]
    base = cfg.tile_sizes[0]
    mem = (expanded + base) * hw.dtype_bytes * 2  # 2 arrays (input + output)
    assert mem <= hw.shared_mem_bytes


def test_tile_config_2d():
    spec = _make_lap2d_spec()
    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    cfg = compute_tile_config(spec, (256, 256), hw)
    assert len(cfg.tile_sizes) == 2
    assert all(t >= 32 for t in cfg.tile_sizes)


def test_temporal_blocking():
    spec = _make_heat_spec()
    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    cfg = compute_tile_config(spec, (1024,), hw)
    temp = compute_temporal_config(spec, cfg, hw)
    assert temp.steps >= 1
    assert temp.bandwidth_reduction_factor >= 1.0
    assert len(temp.expanded_halo) == 1


def test_temporal_blocking_small_memory():
    """With very small shared memory, temporal blocking should be T=1."""
    spec = _make_heat_spec()
    hw = HardwareSpec(shared_mem_bytes=512, dtype_bytes=8)
    cfg = compute_tile_config(spec, (1024,), hw)
    temp = compute_temporal_config(spec, cfg, hw)
    assert temp.steps >= 1


def test_roofline_1d():
    spec = _make_heat_spec()
    roof = roofline_analysis(spec)
    assert roof.flops_per_point > 0
    assert roof.bytes_per_point > 0
    assert roof.arithmetic_intensity > 0
    assert roof.bound in ("memory", "compute")
    assert roof.peak_gpoints_s > 0


def test_roofline_2d():
    spec = _make_lap2d_spec()
    roof = roofline_analysis(spec)
    assert roof.flops_per_point > 0
    # 2D Laplacian has 5 loads + 1 store
    assert roof.bytes_per_point > 0
