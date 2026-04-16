"""Tests for benchmark baselines and plotting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.stencils import STENCIL_META, full_shape


# -- Helpers --

def _has_gpu() -> bool:
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _has_jax_gpu() -> bool:
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def _has_devito() -> bool:
    try:
        import devito
        return True
    except ImportError:
        return False


gpu_required = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
jax_gpu_required = pytest.mark.skipif(not _has_jax_gpu(), reason="No JAX GPU available")
devito_required = pytest.mark.skipif(not _has_devito(), reason="Devito not installed")


# -- Stencil slice correctness tests (CPU only, no GPU needed) --

class TestSliceFunctions:
    """Verify slice functions produce correct shapes."""

    @pytest.mark.parametrize("name", list(STENCIL_META.keys()))
    def test_output_shape(self, name):
        meta = STENCIL_META[name]
        # Use small domains
        small_domains = {1: (32,), 2: (16, 16), 3: (8, 8, 8)}
        domain = small_domains[meta["ndim"]]
        shape = full_shape(domain, meta["halo"])
        u = np.random.randn(*shape).astype(np.float64)
        result = meta["slice_fn"](u)
        assert result.shape == domain, f"Expected {domain}, got {result.shape}"


# -- CuPy baseline tests --

@gpu_required
class TestCuPyBaseline:
    """CuPy baseline produces timing results."""

    def test_heat_2d(self):
        from benchmarks.baselines.cupy_baseline import bench_cupy
        result = bench_cupy("heat_2d", (66, 66), warmup=2, iters=5)
        assert result["time_ms"] > 0
        assert result["framework"] == "CuPy"
        assert result["gpoints_per_s"] > 0

    def test_laplacian_3d_7pt(self):
        from benchmarks.baselines.cupy_baseline import bench_cupy
        result = bench_cupy("laplacian_3d_7pt", (18, 18, 18), warmup=2, iters=5)
        assert result["time_ms"] > 0


# -- JAX baseline tests --

@jax_gpu_required
class TestJAXBaseline:
    """JAX/XLA baseline produces timing results."""

    def test_heat_2d(self):
        from benchmarks.baselines.jax_baseline import bench_jax
        result = bench_jax("heat_2d", (66, 66), warmup=2, iters=5)
        assert result["time_ms"] > 0
        assert result["framework"] == "JAX"


# -- Devito baseline tests --

@devito_required
class TestDevitoBaseline:
    """Devito baseline produces timing results."""

    def test_laplacian_2d(self):
        from benchmarks.baselines.devito_baseline import bench_devito
        result = bench_devito("laplacian_2d_5pt", (32, 32), warmup=2, iters=5)
        assert result["time_ms"] > 0
        assert result["framework"] == "Devito"


# -- Hand-written cuTile tests --

@gpu_required
class TestHandwrittenBaseline:
    """Hand-written cuTile kernels produce timing results."""

    def test_heat_2d(self):
        from benchmarks.baselines.handwritten_cutile import bench_handwritten
        result = bench_handwritten("heat_2d", (66, 66), warmup=2, iters=5)
        assert result["time_ms"] > 0
        assert result["framework"] == "Hand-cuTile"


# -- Plotting smoke tests --

class TestPlotting:
    """Plotting module runs without errors on mock data."""

    def test_generate_figures_with_mock_data(self):
        from benchmarks.plotting import generate_all_figures

        mock_data = {
            "gpu": {"name": "Test GPU", "n_gpus": 1, "peak_bw_gbs": 1000.0},
            "timestamp": "2026-04-16",
            "warmup": 10,
            "iters": 50,
            "baselines": ["cupy"],
            "results": [
                {
                    "stencil": "heat_2d",
                    "ndim": 2,
                    "domain": [512, 512],
                    "npoints": 262144,
                    "cutile": {
                        "framework": "cuTile-DSL",
                        "time_ms": 0.5,
                        "gpoints_per_s": 2.0,
                        "gbytes_per_s": 200.0,
                        "tile_sizes": [32, 32],
                        "temporal_steps": 1,
                    },
                    "cupy": {
                        "framework": "CuPy",
                        "time_ms": 1.5,
                        "gpoints_per_s": 0.7,
                        "gbytes_per_s": 70.0,
                    },
                },
                {
                    "stencil": "heat_2d",
                    "ndim": 2,
                    "domain": [1024, 1024],
                    "npoints": 1048576,
                    "cutile": {
                        "framework": "cuTile-DSL",
                        "time_ms": 0.3,
                        "gpoints_per_s": 3.5,
                        "gbytes_per_s": 350.0,
                        "tile_sizes": [32, 32],
                        "temporal_steps": 1,
                    },
                    "cupy": {
                        "framework": "CuPy",
                        "time_ms": 1.0,
                        "gpoints_per_s": 1.0,
                        "gbytes_per_s": 100.0,
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "figures"
            generate_all_figures(mock_data, outdir)
            pdfs = list(outdir.glob("*.pdf"))
            pngs = list(outdir.glob("*.png"))
            assert len(pdfs) >= 3, f"Expected >=3 PDFs, got {len(pdfs)}: {pdfs}"
            assert len(pngs) >= 3, f"Expected >=3 PNGs, got {len(pngs)}: {pngs}"
