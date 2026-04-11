"""Convergence history tracking for iterative solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConvergenceHistory:
    """Structured convergence data from an iterative solve."""
    iterations: int
    residual_norms: List[float]
    converged: bool
    final_residual: float
    tolerance: float
    max_iterations: int

    @property
    def convergence_rate(self) -> Optional[float]:
        """Average convergence rate (ratio of successive residuals).

        Returns None if fewer than 2 iterations.
        """
        norms = self.residual_norms
        if len(norms) < 2:
            return None
        rates = []
        for i in range(1, len(norms)):
            if norms[i - 1] > 0:
                rates.append(norms[i] / norms[i - 1])
        return sum(rates) / len(rates) if rates else None

    def summary(self) -> str:
        """One-line summary string."""
        status = "converged" if self.converged else "NOT converged"
        return (
            f"{status} in {self.iterations} iterations, "
            f"final residual={self.final_residual:.2e}, "
            f"tol={self.tolerance:.2e}"
        )

    def plot(self, ax=None):
        """Plot convergence curve (residual norm vs iteration).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        else:
            fig = ax.get_figure()

        iters = list(range(len(self.residual_norms)))
        ax.semilogy(iters, self.residual_norms, "b-o", markersize=3, label="||r||")
        ax.axhline(y=self.tolerance, color="r", linestyle="--", alpha=0.7, label=f"tol={self.tolerance:.0e}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title(f"CG Convergence ({self.summary()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
