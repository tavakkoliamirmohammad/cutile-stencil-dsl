"""Benchmark suite: runs all stencils at multiple sizes, prints comparison table.

Usage:
    python examples/benchmark_suite.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.benchmark.runner import BenchmarkRunner
from cutile_stencil.benchmark.report import BenchmarkReport


def _define_stencils():
    """Define a set of benchmark stencils."""
    clear()

    @stencil()
    def heat_1d(u, i):
        return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

    @stencil()
    def laplacian_2d(u, i, j):
        return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

    @stencil()
    def advection_1d(u, i):
        return u[i] - 0.5 * (u[i] - u[i - 1])

    stencils = [heat_1d, laplacian_2d, advection_1d]

    # Extract footprints and halos
    for fn in stencils:
        spec = fn._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, spec.ndim)

    return stencils


def main():
    print("=" * 60)
    print("cuTile Stencil DSL — Benchmark Suite")
    print("=" * 60)

    stencils = _define_stencils()
    report = BenchmarkReport()

    for fn in stencils:
        spec = fn._stencil_spec
        runner = BenchmarkRunner(spec)

        # Choose grid sizes based on dimensionality
        if spec.ndim == 1:
            grid_sizes = [(256,), (1024,), (4096,)]
        elif spec.ndim == 2:
            grid_sizes = [(32, 32), (64, 64), (128, 128)]
        else:
            grid_sizes = [(16, 16, 16), (32, 32, 32)]

        print(f"\nBenchmarking: {spec.name} ({spec.ndim}D)")
        pred = runner.roofline_prediction()
        print(f"  Roofline: {pred['peak_gpoints_s']:.2f} Gpts/s ({pred['bound']})")

        for gs in grid_sizes:
            result = runner.run_numpy(gs, iterations=10, warmup=3)
            report.add(result)
            grid_str = "x".join(str(g) for g in gs)
            print(f"  Grid {grid_str}: {result.gpoints_per_sec:.4f} Gpts/s, "
                  f"{result.ms_per_iteration:.2f} ms/iter")

    print()
    print(report.to_table())

    # Also show roofline comparison for the first stencil
    spec0 = stencils[0]._stencil_spec
    runner0 = BenchmarkRunner(spec0)
    pred0 = runner0.roofline_prediction()
    print()
    print(report.roofline_comparison(pred0))


if __name__ == "__main__":
    main()
