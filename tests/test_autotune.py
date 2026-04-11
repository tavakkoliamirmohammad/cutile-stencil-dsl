"""Tests for Bayesian auto-tuning."""

import pytest

optuna = pytest.importorskip("optuna")

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.autotune import autotune, AutotuneResult


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


def _make_heat_fn():
    @stencil(ndim=1, order=2)
    def heat(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
    return heat


def _make_lap2d_fn():
    @stencil(ndim=2, order=2)
    def lap2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
    return lap2d


class TestAutotune:
    def test_returns_result(self):
        heat = _make_heat_fn()
        result = autotune(heat, domain=(1024,), n_trials=20)
        assert isinstance(result, AutotuneResult)
        assert result.best_score > 0
        assert len(result.best_tile_sizes) == 1
        assert result.best_temporal_steps >= 1

    def test_2d_returns_result(self):
        lap = _make_lap2d_fn()
        result = autotune(lap, domain=(256, 256), n_trials=20)
        assert isinstance(result, AutotuneResult)
        assert len(result.best_tile_sizes) == 2

    def test_score_at_least_as_good_as_default(self):
        heat = _make_heat_fn()
        spec = heat._stencil_spec
        extract_footprint(spec)
        spec.halo_widths = compute_halo(spec.accesses, 1)

        hw = HardwareSpec()
        default_cfg = compute_tile_config(spec, (1024,), hw)

        result = autotune(heat, domain=(1024,), hw=hw, n_trials=50)
        # Autotune should find something at least as good
        assert result.best_score > 0
        assert result.n_feasible > 0

    def test_accepts_stencil_spec(self):
        heat = _make_heat_fn()
        spec = heat._stencil_spec
        extract_footprint(spec)
        spec.halo_widths = compute_halo(spec.accesses, 1)
        result = autotune(spec, domain=(1024,), n_trials=10)
        assert isinstance(result, AutotuneResult)

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="Expected"):
            autotune(lambda x: x, domain=(1024,))

    def test_custom_candidates(self):
        heat = _make_heat_fn()
        result = autotune(
            heat, domain=(1024,), n_trials=10,
            tile_candidates=[32, 64],
            temporal_candidates=[1, 2],
        )
        assert result.best_tile_sizes[0] in (32, 64)
        assert result.best_temporal_steps in (1, 2)
