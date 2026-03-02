"""Tests for the analysis pipeline, compile, decorator validation, and CFL analysis."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import StencilSpec, HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.pipeline import analyze, AnalysisResult, compile, CompileResult
from cutile_stencil.dsl.boundary import BoundarySpec
from cutile_stencil.analysis.footprint import compute_max_cfl, extract_footprint
from cutile_stencil.config import GPU_PRESETS


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestAnalyzePipeline:
    def test_one_call_analysis_1d(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        spec = heat._stencil_spec
        assert spec.ndim == 1
        assert spec.order == 2
        result = analyze(spec, domain=(1024,))
        assert isinstance(result, AnalysisResult)
        assert result.tile_config.tile_sizes[0] >= 32
        assert result.temporal_config.steps >= 1
        assert result.roofline.flops_per_point > 0
        assert result.spec.halo_widths == (1,)

    def test_one_call_analysis_2d(self):
        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        spec = lap._stencil_spec
        assert spec.ndim == 2
        assert spec.order == 2
        result = analyze(spec, domain=(256, 256))
        assert len(result.tile_config.tile_sizes) == 2
        assert result.roofline.bound in ("memory", "compute")

    def test_with_hardware_preset(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        assert heat._stencil_spec.ndim == 1
        assert heat._stencil_spec.order == 2
        hw = HardwareSpec.from_preset("H100_SXM")
        result = analyze(heat._stencil_spec, domain=(1024,), hw=hw)
        assert result.roofline.peak_gpoints_s > 0

    def test_warnings_empty_for_good_spec(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = analyze(heat._stencil_spec, domain=(1024,))
        assert len(result.warnings) == 0

    def test_temporal_steps_set(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = analyze(heat._stencil_spec, domain=(1024,))
        assert result.spec.temporal_steps >= 1


class TestDecoratorValidation:
    def test_too_few_params(self):
        with pytest.raises(ValueError, match="at least"):
            @stencil(ndim=2, order=2)
            def bad(i, j):
                return i + j

    def test_no_return(self):
        with pytest.raises(ValueError, match="return"):
            @stencil(ndim=1, order=2)
            def bad(u, i):
                u[i] + u[i + 1]  # No return!

    def test_valid_stencil_passes(self):
        @stencil(ndim=1, order=2)
        def ok(u, i):
            return u[i]

        assert ok._stencil_spec.ndim == 1


class TestStencilSpecValidate:
    def test_no_accesses_warning(self):
        @stencil(ndim=1, order=2)
        def s(u, i):
            return u[i]

        spec = s._stencil_spec
        # Don't extract footprint
        warnings = spec.validate()
        assert any("accesses" in w.lower() for w in warnings)

    def test_no_halo_warning(self):
        @stencil(ndim=1, order=2)
        def s(u, i):
            return u[i]

        spec = s._stencil_spec
        extract_footprint(spec)
        # Don't compute halo
        spec.halo_widths = None
        warnings = spec.validate()
        assert any("halo" in w.lower() for w in warnings)

    def test_clean_spec_no_warnings(self):
        @stencil(ndim=1, order=2)
        def s(u, i):
            return 0.5 * u[i - 1] + 0.5 * u[i + 1]

        spec = s._stencil_spec
        from cutile_stencil.analysis.footprint import compute_halo
        extract_footprint(spec)
        spec.halo_widths = compute_halo(spec.accesses, 1)
        warnings = spec.validate()
        assert len(warnings) == 0


class TestCFLAnalysis:
    def test_cfl_1d_order2(self):
        @stencil(ndim=1, order=2)
        def s(u, i):
            return u[i - 1] + u[i + 1]

        assert s._stencil_spec.ndim == 1
        assert s._stencil_spec.order == 2
        cfl = compute_max_cfl(s._stencil_spec)
        assert cfl > 0
        assert cfl == 1.0  # 1 / (1 * (2/2))

    def test_cfl_2d_order2(self):
        @stencil(ndim=2, order=2)
        def s(u, i, j):
            return u[i - 1, j] + u[i + 1, j]

        assert s._stencil_spec.ndim == 2
        assert s._stencil_spec.order == 2
        cfl = compute_max_cfl(s._stencil_spec)
        assert cfl == 0.5  # 1 / (2 * 1)

    def test_cfl_higher_order_smaller(self):
        @stencil(ndim=1, order=4)
        def s4(u, i):
            return u[i - 2] + u[i + 2]

        @stencil(ndim=1, order=2)
        def s2(u, i):
            return u[i - 1] + u[i + 1]

        assert s4._stencil_spec.ndim == 1
        assert s4._stencil_spec.order == 4
        assert s2._stencil_spec.ndim == 1
        assert s2._stencil_spec.order == 2
        cfl4 = compute_max_cfl(s4._stencil_spec)
        cfl2 = compute_max_cfl(s2._stencil_spec)
        assert cfl4 < cfl2  # Higher order = tighter CFL


class TestCompile:
    def test_compile_1d_returns_compile_result(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        assert heat._stencil_spec.ndim == 1
        assert heat._stencil_spec.order == 2
        result = compile(heat, domain=(1024,))
        assert isinstance(result, CompileResult)
        assert isinstance(result.analysis, AnalysisResult)
        assert result.spec.name == "heat"
        assert len(result.code) > 0
        ast.parse(result.code)

    def test_compile_2d(self):
        @stencil(ndim=2, order=2)
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        assert lap._stencil_spec.ndim == 2
        assert lap._stencil_spec.order == 2
        result = compile(lap, domain=(256, 256))
        assert isinstance(result, CompileResult)
        ast.parse(result.code)
        assert ".slice(axis=" in result.code

    def test_compile_accepts_spec_directly(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = compile(heat._stencil_spec, domain=(1024,))
        assert isinstance(result, CompileResult)

    def test_compile_with_hw(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        hw = HardwareSpec.from_preset("H100_SXM")
        result = compile(heat, domain=(1024,), hw=hw)
        assert result.analysis.roofline.peak_gpoints_s > 0

    def test_compile_emit_to_file(self, tmp_path):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = compile(heat, domain=(1024,))
        out = result.emit_to_file(tmp_path / "heat_kernel.py")
        assert out.exists()
        ast.parse(out.read_text())

    def test_compile_invalid_input_raises(self):
        with pytest.raises(TypeError, match="Expected"):
            compile("not a stencil", domain=(1024,))

    def test_compile_code_uses_slice(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = compile(heat, domain=(1024,))
        assert ".slice(axis=" in result.code
        assert "HALO" in result.code

    def test_compile_importable_from_top_level(self):
        from cutile_stencil import compile, CompileResult
        assert callable(compile)
        assert CompileResult is not None


class TestGPUPresetsExport:
    def test_presets_importable(self):
        from cutile_stencil import GPU_PRESETS
        assert "H100_SXM" in GPU_PRESETS

    def test_analyze_importable(self):
        from cutile_stencil import analyze
        assert callable(analyze)

    def test_boundary_types_importable(self):
        from cutile_stencil import BoundaryType, BoundarySpec
        bs = BoundarySpec.periodic(2)
        assert bs.conditions[0][0].bc_type == BoundaryType.PERIODIC
