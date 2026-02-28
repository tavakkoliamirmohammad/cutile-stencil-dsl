"""Benchmark runner for stencil performance evaluation.

Provides NumPy and optional Numba backends for measuring stencil throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from cutile_stencil.dsl.types import StencilSpec, HardwareSpec
from cutile_stencil.reference.stencil_ref import apply_stencil
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.roofline import roofline_analysis


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    backend: str
    grid_size: Tuple[int, ...]
    dtype: str
    elapsed_seconds: float
    gpoints_per_sec: float
    gbytes_per_sec: float
    flops_per_sec: float
    iterations: int

    @property
    def ms_per_iteration(self) -> float:
        return (self.elapsed_seconds / self.iterations) * 1000.0


class BenchmarkRunner:
    """Run stencil benchmarks on various backends.

    Parameters
    ----------
    spec : StencilSpec
        The stencil to benchmark.
    hw : HardwareSpec, optional
        Hardware spec for roofline comparison.
    """

    def __init__(self, spec: StencilSpec, hw: HardwareSpec | None = None):
        self.spec = spec
        self.hw = hw or HardwareSpec()
        # Ensure footprint is extracted
        if not spec.accesses:
            extract_footprint(spec)
        if spec.halo_widths is None:
            spec.halo_widths = compute_halo(spec.accesses, spec.ndim)

    def run_numpy(
        self,
        grid_size: Tuple[int, ...],
        iterations: int = 20,
        warmup: int = 5,
        dtype: str = "float64",
    ) -> BenchmarkResult:
        """Benchmark using NumPy reference implementation.

        Parameters
        ----------
        grid_size : tuple of int
            Interior grid size per dimension.
        iterations : int
            Number of timed iterations.
        warmup : int
            Warmup iterations (not timed).
        dtype : str
            Data type for arrays.
        """
        spec = self.spec
        halo = spec.halo_widths
        np_dtype = getattr(np, dtype)

        # Create full array with halo
        full_size = tuple(g + 2 * h for g, h in zip(grid_size, halo))
        u = np.random.randn(*full_size).astype(np_dtype)

        # Create multiple input arrays for multi-input stencils
        # (apply_stencil uses the first input for all)

        # Warmup
        for _ in range(warmup):
            apply_stencil(u, spec)

        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            apply_stencil(u, spec)
        elapsed = time.perf_counter() - start

        # Compute metrics
        total_points = 1
        for g in grid_size:
            total_points *= g
        points_per_sec = (total_points * iterations) / elapsed
        gpoints_per_sec = points_per_sec / 1e9

        # Bytes per point
        roof = roofline_analysis(spec, self.hw)
        gbytes_per_sec = (points_per_sec * roof.bytes_per_point) / 1e9
        flops_per_sec = (points_per_sec * roof.flops_per_point)

        return BenchmarkResult(
            name=spec.name,
            backend="numpy",
            grid_size=grid_size,
            dtype=dtype,
            elapsed_seconds=elapsed,
            gpoints_per_sec=gpoints_per_sec,
            gbytes_per_sec=gbytes_per_sec,
            flops_per_sec=flops_per_sec,
            iterations=iterations,
        )

    def run_numba(
        self,
        grid_size: Tuple[int, ...],
        iterations: int = 20,
        warmup: int = 5,
        dtype: str = "float64",
    ) -> Optional[BenchmarkResult]:
        """Benchmark using Numba JIT.

        Returns None if numba is not available.
        """
        from cutile_stencil.benchmark.numba_backend import numba_apply_stencil
        if numba_apply_stencil is None:
            return None

        spec = self.spec
        halo = spec.halo_widths
        np_dtype = getattr(np, dtype)
        full_size = tuple(g + 2 * h for g, h in zip(grid_size, halo))
        u = np.random.randn(*full_size).astype(np_dtype)

        fn = numba_apply_stencil(spec)
        if fn is None:
            return None

        # Warmup (includes JIT compilation)
        for _ in range(warmup):
            fn(u)

        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            fn(u)
        elapsed = time.perf_counter() - start

        total_points = 1
        for g in grid_size:
            total_points *= g
        points_per_sec = (total_points * iterations) / elapsed
        gpoints_per_sec = points_per_sec / 1e9

        roof = roofline_analysis(spec, self.hw)
        gbytes_per_sec = (points_per_sec * roof.bytes_per_point) / 1e9
        flops_per_sec = (points_per_sec * roof.flops_per_point)

        return BenchmarkResult(
            name=spec.name,
            backend="numba",
            grid_size=grid_size,
            dtype=dtype,
            elapsed_seconds=elapsed,
            gpoints_per_sec=gpoints_per_sec,
            gbytes_per_sec=gbytes_per_sec,
            flops_per_sec=flops_per_sec,
            iterations=iterations,
        )

    def scaling_study(
        self,
        grid_sizes: List[Tuple[int, ...]],
        backend: str = "numpy",
        iterations: int = 20,
        warmup: int = 5,
        dtype: str = "float64",
    ) -> List[BenchmarkResult]:
        """Run benchmarks across multiple grid sizes."""
        results = []
        for gs in grid_sizes:
            if backend == "numba":
                r = self.run_numba(gs, iterations, warmup, dtype)
            else:
                r = self.run_numpy(gs, iterations, warmup, dtype)
            if r is not None:
                results.append(r)
        return results

    def roofline_prediction(self) -> dict:
        """Get theoretical roofline prediction."""
        roof = roofline_analysis(self.spec, self.hw)
        return {
            "flops_per_point": roof.flops_per_point,
            "bytes_per_point": roof.bytes_per_point,
            "arithmetic_intensity": roof.arithmetic_intensity,
            "bound": roof.bound,
            "peak_gpoints_s": roof.peak_gpoints_s,
        }
