"""Tests for tiling visualization."""

import pytest
from cutile_stencil.dsl.types import TileConfig, TemporalConfig


pytest.importorskip("matplotlib")


from cutile_stencil.analysis.visualization import visualize_tiling


class TestVisualizeTiling:
    def test_1d_returns_figure(self):
        cfg = TileConfig(tile_sizes=(128,), halo_widths=(1,), num_tiles=(8,), overhead_fraction=0.015)
        fig = visualize_tiling(cfg, domain=(1024,))
        assert fig is not None
        assert len(fig.axes) == 1

    def test_2d_returns_figure(self):
        cfg = TileConfig(tile_sizes=(64, 64), halo_widths=(1, 1), num_tiles=(4, 4), overhead_fraction=0.03)
        fig = visualize_tiling(cfg, domain=(256, 256))
        assert fig is not None

    def test_3d_returns_2d_slice(self):
        cfg = TileConfig(
            tile_sizes=(32, 32, 32), halo_widths=(1, 1, 1),
            num_tiles=(2, 2, 2), overhead_fraction=0.06,
        )
        fig = visualize_tiling(cfg, domain=(64, 64, 64))
        assert fig is not None
        assert "3D" in fig.axes[0].get_title() or "slice" in fig.axes[0].get_title()

    def test_2d_with_temporal(self):
        cfg = TileConfig(tile_sizes=(64, 64), halo_widths=(1, 1), num_tiles=(4, 4), overhead_fraction=0.03)
        temp = TemporalConfig(steps=4, expanded_halo=(4, 4), bandwidth_reduction_factor=4.0)
        fig = visualize_tiling(cfg, domain=(256, 256), temporal_config=temp)
        assert fig is not None

    def test_custom_halo_widths(self):
        cfg = TileConfig(tile_sizes=(128,), halo_widths=(1,), num_tiles=(8,), overhead_fraction=0.015)
        fig = visualize_tiling(cfg, domain=(1024,), halo_widths=(2,))
        assert fig is not None
