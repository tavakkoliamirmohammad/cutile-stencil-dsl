"""Tests for the benchmarking framework."""

import json
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.benchmark.runner import BenchmarkRunner, BenchmarkResult
from cutile_stencil.benchmark.report import BenchmarkReport


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
    accesses = extract_footprint(spec)
    spec.halo_widths = compute_halo(accesses, 1)
    return spec


class TestBenchmarkRunner:
    def test_numpy_benchmark(self):
        spec = _make_heat_spec()
        runner = BenchmarkRunner(spec)
        result = runner.run_numpy((256,), iterations=3, warmup=1)
        assert isinstance(result, BenchmarkResult)
        assert result.backend == "numpy"
        assert result.gpoints_per_sec > 0
        assert result.elapsed_seconds > 0
        assert result.iterations == 3

    def test_numpy_benchmark_2d(self):
        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        spec = lap._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)
        runner = BenchmarkRunner(spec)
        result = runner.run_numpy((32, 32), iterations=3, warmup=1)
        assert result.grid_size == (32, 32)
        assert result.gpoints_per_sec > 0

    def test_scaling_study(self):
        spec = _make_heat_spec()
        runner = BenchmarkRunner(spec)
        results = runner.scaling_study(
            grid_sizes=[(64,), (128,), (256,)],
            backend="numpy",
            iterations=3,
            warmup=1,
        )
        assert len(results) == 3
        # Larger grids should take more time (or same)
        for r in results:
            assert r.gpoints_per_sec > 0

    def test_roofline_prediction(self):
        spec = _make_heat_spec()
        runner = BenchmarkRunner(spec)
        pred = runner.roofline_prediction()
        assert "flops_per_point" in pred
        assert "peak_gpoints_s" in pred
        assert pred["peak_gpoints_s"] > 0

    def test_numba_returns_none_if_unavailable(self):
        """Numba benchmark gracefully handles missing numba."""
        spec = _make_heat_spec()
        runner = BenchmarkRunner(spec)
        # May return None if numba not installed, or a result if it is
        result = runner.run_numba((256,), iterations=3, warmup=1)
        assert result is None or isinstance(result, BenchmarkResult)

    def test_ms_per_iteration(self):
        spec = _make_heat_spec()
        runner = BenchmarkRunner(spec)
        result = runner.run_numpy((256,), iterations=5, warmup=1)
        expected = (result.elapsed_seconds / 5) * 1000
        assert abs(result.ms_per_iteration - expected) < 1e-10


class TestBenchmarkReport:
    def test_table_output(self):
        result = BenchmarkResult(
            name="heat", backend="numpy", grid_size=(256,),
            dtype="float64", elapsed_seconds=0.01,
            gpoints_per_sec=0.5, gbytes_per_sec=8.0,
            flops_per_sec=1e9, iterations=10,
        )
        report = BenchmarkReport([result])
        table = report.to_table()
        assert "heat" in table
        assert "numpy" in table

    def test_csv_output(self):
        result = BenchmarkResult(
            name="heat", backend="numpy", grid_size=(256,),
            dtype="float64", elapsed_seconds=0.01,
            gpoints_per_sec=0.5, gbytes_per_sec=8.0,
            flops_per_sec=1e9, iterations=10,
        )
        report = BenchmarkReport([result])
        csv_str = report.to_csv()
        assert "heat" in csv_str
        assert "numpy" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

    def test_json_output(self):
        result = BenchmarkResult(
            name="heat", backend="numpy", grid_size=(256,),
            dtype="float64", elapsed_seconds=0.01,
            gpoints_per_sec=0.5, gbytes_per_sec=8.0,
            flops_per_sec=1e9, iterations=10,
        )
        report = BenchmarkReport([result])
        json_str = report.to_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["name"] == "heat"
        assert data[0]["backend"] == "numpy"

    def test_roofline_comparison(self):
        result = BenchmarkResult(
            name="heat", backend="numpy", grid_size=(256,),
            dtype="float64", elapsed_seconds=0.01,
            gpoints_per_sec=0.5, gbytes_per_sec=8.0,
            flops_per_sec=1e9, iterations=10,
        )
        report = BenchmarkReport([result])
        pred = {"peak_gpoints_s": 100.0, "bound": "memory",
                "arithmetic_intensity": 0.1}
        comparison = report.roofline_comparison(pred)
        assert "Roofline" in comparison
        assert "%" in comparison

    def test_add_result(self):
        report = BenchmarkReport()
        assert len(report.results) == 0
        result = BenchmarkResult(
            name="test", backend="numpy", grid_size=(100,),
            dtype="float64", elapsed_seconds=0.01,
            gpoints_per_sec=1.0, gbytes_per_sec=1.0,
            flops_per_sec=1e9, iterations=5,
        )
        report.add(result)
        assert len(report.results) == 1


class TestBenchmarkSuite:
    def test_suite_runs(self):
        """Benchmark suite example runs end-to-end."""
        from examples.benchmark_suite import main
        main()
