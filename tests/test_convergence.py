"""Tests for convergence history tracking."""

import math
import pytest
from cutile_stencil.solvers.convergence import ConvergenceHistory


class TestConvergenceHistory:
    def test_basic_creation(self):
        history = ConvergenceHistory(
            iterations=10,
            residual_norms=[1.0, 0.5, 0.25, 0.125],
            converged=True,
            final_residual=0.125,
            tolerance=1e-3,
            max_iterations=100,
        )
        assert history.iterations == 10
        assert history.converged is True
        assert len(history.residual_norms) == 4

    def test_convergence_rate(self):
        # Each residual is half the previous → rate = 0.5
        history = ConvergenceHistory(
            iterations=4,
            residual_norms=[1.0, 0.5, 0.25, 0.125],
            converged=True,
            final_residual=0.125,
            tolerance=1e-3,
            max_iterations=100,
        )
        rate = history.convergence_rate
        assert rate is not None
        assert abs(rate - 0.5) < 1e-10

    def test_convergence_rate_single_iteration(self):
        history = ConvergenceHistory(
            iterations=1,
            residual_norms=[1.0],
            converged=False,
            final_residual=1.0,
            tolerance=1e-10,
            max_iterations=100,
        )
        assert history.convergence_rate is None

    def test_summary_converged(self):
        history = ConvergenceHistory(
            iterations=50,
            residual_norms=[1.0] * 50,
            converged=True,
            final_residual=1e-11,
            tolerance=1e-10,
            max_iterations=1000,
        )
        s = history.summary()
        assert "converged" in s
        assert "50" in s

    def test_summary_not_converged(self):
        history = ConvergenceHistory(
            iterations=1000,
            residual_norms=[1.0] * 1000,
            converged=False,
            final_residual=0.1,
            tolerance=1e-10,
            max_iterations=1000,
        )
        s = history.summary()
        assert "NOT converged" in s

    def test_monotonic_residuals(self):
        """Well-conditioned CG should have monotonically decreasing residuals."""
        norms = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        history = ConvergenceHistory(
            iterations=6,
            residual_norms=norms,
            converged=True,
            final_residual=0.3125,
            tolerance=1.0,
            max_iterations=100,
        )
        for i in range(1, len(norms)):
            assert norms[i] <= norms[i - 1]
        assert history.convergence_rate is not None
        assert history.convergence_rate < 1.0


class TestConvergencePlot:
    def test_plot_returns_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        history = ConvergenceHistory(
            iterations=5,
            residual_norms=[1.0, 0.5, 0.25, 0.125, 0.0625],
            converged=True,
            final_residual=0.0625,
            tolerance=0.01,
            max_iterations=100,
        )
        fig = history.plot()
        assert fig is not None
