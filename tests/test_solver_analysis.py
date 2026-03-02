"""Tests for the solver analysis pipeline."""

import pytest

from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.solvers.stencil_bridge import laplacian_stencil_spec
from cutile_stencil.solvers.solver_analysis import (
    analyze_dot,
    analyze_axpy,
    analyze_dia_spmv,
    analyze_stencil_operator,
    analyze_cg_iteration,
    SolverAnalysisResult,
    SolverKernelProfile,
    print_solver_analysis,
)


class TestAnalyzeDot:
    def test_flops(self):
        prof = analyze_dot(1000)
        assert prof.flops_per_element == 2  # mul + add

    def test_loads(self):
        prof = analyze_dot(1000)
        assert prof.loads_per_element == 2  # a, b

    def test_memory_bound(self):
        hw = HardwareSpec(peak_bandwidth_gbs=2000.0, peak_gflops=10000.0)
        prof = analyze_dot(1000, hw)
        assert prof.bound == "memory"

    def test_name(self):
        prof = analyze_dot(100)
        assert prof.name == "dot"


class TestAnalyzeAxpy:
    def test_flops(self):
        prof = analyze_axpy(1000)
        assert prof.flops_per_element == 2  # mul + add

    def test_stores(self):
        prof = analyze_axpy(1000)
        assert prof.stores_per_element == 1  # y

    def test_memory_bound(self):
        hw = HardwareSpec(peak_bandwidth_gbs=2000.0, peak_gflops=10000.0)
        prof = analyze_axpy(1000, hw)
        assert prof.bound == "memory"


class TestAnalyzeDiaSpmv:
    def test_3_diags_flops(self):
        prof = analyze_dia_spmv(1000, 3)
        assert prof.flops_per_element == 5  # 3 muls + 2 adds

    def test_5_diags_flops(self):
        prof = analyze_dia_spmv(1000, 5)
        assert prof.flops_per_element == 9  # 5 muls + 4 adds

    def test_loads(self):
        prof = analyze_dia_spmv(1000, 3)
        assert prof.loads_per_element == 6  # 3 diag_vals + 3 x gathers


class TestAnalyzeStencilOperator:
    def test_1d_laplacian(self):
        spec = laplacian_stencil_spec(1)
        hw = HardwareSpec()
        prof = analyze_stencil_operator(spec, (1000,), hw)
        assert prof.name == "stencil_laplacian_1d"
        assert prof.flops_per_element > 0
        assert prof.bytes_per_element > 0

    def test_matches_pipeline_roofline(self):
        """Analysis result should wrap the pipeline's roofline result."""
        from cutile_stencil.dsl.pipeline import analyze as pipeline_analyze

        spec = laplacian_stencil_spec(1)
        hw = HardwareSpec()
        ar = pipeline_analyze(spec, (1000,), hw)
        prof = analyze_stencil_operator(spec, (1000,), hw)

        assert prof.flops_per_element == ar.roofline.flops_per_point
        assert prof.arithmetic_intensity == ar.roofline.arithmetic_intensity
        assert prof.bound == ar.roofline.bound


class TestAnalyzeCGIteration:
    def test_returns_result(self):
        hw = HardwareSpec()
        op = analyze_dia_spmv(1000, 3, hw)
        result = analyze_cg_iteration(op, 1000, hw)
        assert isinstance(result, SolverAnalysisResult)

    def test_kernel_count(self):
        hw = HardwareSpec()
        op = analyze_dia_spmv(1000, 3, hw)
        result = analyze_cg_iteration(op, 1000, hw)
        # 1 operator + 3 dots + 2 axpys = 6
        assert len(result.kernel_profiles) == 6

    def test_total_flops(self):
        hw = HardwareSpec()
        op = analyze_dia_spmv(1000, 3, hw)
        result = analyze_cg_iteration(op, 1000, hw)
        # operator(5) + 3*dot(2) + 2*axpy(2) = 5 + 6 + 4 = 15
        assert result.total_flops_per_element == 15

    def test_bottleneck_identified(self):
        hw = HardwareSpec()
        op = analyze_dia_spmv(1000, 3, hw)
        result = analyze_cg_iteration(op, 1000, hw)
        assert result.bottleneck in [p.name for p in result.kernel_profiles]
        assert result.bottleneck_bound in ("memory", "compute")

    def test_stencil_operator_iteration(self):
        spec = laplacian_stencil_spec(1)
        hw = HardwareSpec()
        op = analyze_stencil_operator(spec, (1000,), hw)
        result = analyze_cg_iteration(op, 1000, hw)
        assert result.total_flops_per_element > 0


class TestPrintSolverAnalysis:
    """print_solver_analysis runs without error (smoke test)."""

    def test_no_crash(self, capsys):
        hw = HardwareSpec()
        op = analyze_dia_spmv(1000, 3, hw)
        result = analyze_cg_iteration(op, 1000, hw)
        print_solver_analysis(result)
        captured = capsys.readouterr()
        assert "Bottleneck" in captured.out
        assert "CG Iteration Analysis" in captured.out
