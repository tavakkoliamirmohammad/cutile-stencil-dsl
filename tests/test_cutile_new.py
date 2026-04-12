"""Comprehensive tests for the new cuTile stencil compiler architecture.

Covers all 4 levels of the testing pyramid:
  Level 1:    IR unit tests (parser, normalization, analysis, optimization passes)
  Level 1.5:  Lowering tests (IR -> Python source generation)
  Level 1.75: Compile API tests (top-level compile() function)
  Level 2:    Reference tests (CPU NumPy executor)
  Level 3:    GPU convergence tests (generated kernel vs CPU reference)
"""

import ast

import numpy as np
import pytest

from cutile.frontend.decorator import stencil
from cutile.frontend.parser import parse_stencil
from cutile.runtime.launcher import compile as stencil_compile
from cutile.runtime.pipeline import Pipeline
from cutile.lowering.normalize import NormalizePass
from cutile.passes.analysis.footprint import AnalysisPass
from cutile.passes.analysis.roofline import RooflinePass
from cutile.passes.tiling import TilingPass
from cutile.passes.temporal import TemporalBlockingPass
from cutile.passes.boundary import BoundaryPass
from cutile.passes.fusion import FusionPass
from cutile.passes.bricked import BrickedLayoutPass
from cutile.passes.decompose import DecompositionPass
from cutile.passes.halo import HaloExchangePass
from cutile.passes.verify import VerifyPass
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from cutile.reference.stencil_ref import apply_stencil
from cutile.config import HardwareSpec, GPU_PRESETS
from xdsl.context import Context


# ===================================================================
# LEVEL 1: IR UNIT TESTS (no GPU required)
# ===================================================================


class TestFrontendParser:
    """Test Python AST -> Dialect 1 IR conversion."""

    def test_1d_heat_produces_ir(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        assert heat._ir is not None
        assert heat._ndim == 1
        assert heat._order == 2

    def test_2d_heat_produces_ir(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        assert heat2d._ir is not None
        assert heat2d._ndim == 2

    def test_3d_laplacian_produces_ir(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k]
                + u[i + 1, j, k]
                + u[i, j - 1, k]
                + u[i, j + 1, k]
                + u[i, j, k - 1]
                + u[i, j, k + 1]
            ) / 6.0

        assert lap3d._ir is not None
        assert lap3d._ndim == 3

    def test_4th_order_stencil(self):
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / 12.0

        assert lap4._order == 4

    def test_preserves_original_function(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        assert callable(heat._fn)

    def test_ir_has_access_ops(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ir_str = str(heat2d._ir)
        assert "cutile_stencil.access" in ir_str

    def test_ir_has_arith_ops(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        ir_str = str(heat._ir)
        assert "arith.mulf" in ir_str or "arith.addf" in ir_str

    def test_dtype_float32(self):
        @stencil(ndim=1, order=2, dtype="float32")
        def heat_f32(u, i):
            return 0.5 * u[i - 1] + 0.5 * u[i + 1]

        assert heat_f32._dtype == "float32"

    def test_bare_decorator_syntax(self):
        """@stencil without arguments should also work (ndim/order auto-inferred)."""
        @stencil
        def heat_bare(u, i):
            return 0.5 * u[i - 1] + 0.5 * u[i + 1]

        assert heat_bare._ir is not None

    def test_multi_input_stencil(self):
        @stencil(ndim=2, order=2)
        def wave(u, v, i, j):
            return u[i, j] + 0.1 * (
                v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]
            )

        assert wave._ir is not None
        assert wave._ndim == 2


class TestNormalization:
    """Test Dialect 1 -> Dialect 2 (standard stencil dialect) conversion."""

    def test_normalize_produces_stencil_apply(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ctx = Context()
        module = heat2d._ir.clone()
        NormalizePass().apply(ctx, module)
        ir_str = str(module)
        assert "stencil.apply" in ir_str
        assert "stencil.access" in ir_str
        assert "stencil.return" in ir_str

    def test_normalize_preserves_offsets(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return u[i - 1] + u[i + 1]

        ctx = Context()
        module = heat._ir.clone()
        NormalizePass().apply(ctx, module)
        ir_str = str(module)
        # Should contain the offset -1 and +1 in some form
        assert "-1" in ir_str
        assert "1" in ir_str

    def test_normalize_preserves_metadata(self):
        @stencil(ndim=2, order=2, dtype="float64")
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ctx = Context()
        module = heat2d._ir.clone()
        NormalizePass().apply(ctx, module)
        ir_str = str(module)
        # Metadata should be attached as attributes
        assert "ndim" in ir_str
        assert "order" in ir_str

    def test_normalize_1d(self):
        @stencil(ndim=1, order=2)
        def heat1d(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        ctx = Context()
        module = heat1d._ir.clone()
        NormalizePass().apply(ctx, module)
        ir_str = str(module)
        assert "stencil.apply" in ir_str

    def test_normalize_3d(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        ctx = Context()
        module = lap3d._ir.clone()
        NormalizePass().apply(ctx, module)
        ir_str = str(module)
        assert "stencil.apply" in ir_str


class TestAnalysisPasses:
    """Test footprint and roofline analysis pass correctness."""

    def _normalized_module(self, stencil_fn):
        """Produce a normalized module ready for analysis."""
        ctx = Context()
        module = stencil_fn._ir.clone()
        NormalizePass().apply(ctx, module)
        return module, ctx

    def _run_analysis(self, stencil_fn):
        """Run normalize + footprint + roofline on a stencil."""
        module, ctx = self._normalized_module(stencil_fn)
        AnalysisPass().apply(ctx, module)
        RooflinePass().apply(ctx, module)
        return module

    def _get_apply_attrs(self, module):
        """Get attributes dict from the first ApplyOp in the module."""
        from xdsl.dialects.stencil import ApplyOp

        for op in module.walk():
            if isinstance(op, ApplyOp):
                return dict(op.attributes)
        pytest.fail("No stencil.ApplyOp found in the module")

    def test_halo_widths_1d(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        module = self._run_analysis(heat)
        attrs = self._get_apply_attrs(module)
        assert "halo_widths" in attrs
        halo = [a.data for a in attrs["halo_widths"]]
        assert halo == [1]

    def test_halo_widths_2d(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module = self._run_analysis(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "halo_widths" in attrs
        halo = [a.data for a in attrs["halo_widths"]]
        assert halo == [1, 1]

    def test_halo_widths_4th_order(self):
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return -u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]

        module = self._run_analysis(lap4)
        attrs = self._get_apply_attrs(module)
        assert "halo_widths" in attrs
        halo = [a.data for a in attrs["halo_widths"]]
        assert halo == [2]

    def test_halo_widths_3d(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        module = self._run_analysis(lap3d)
        attrs = self._get_apply_attrs(module)
        halo = [a.data for a in attrs["halo_widths"]]
        assert halo == [1, 1, 1]

    def test_roofline_memory_bound(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module = self._run_analysis(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "bound" in attrs
        assert attrs["bound"].data == "memory"

    def test_flop_count(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module = self._run_analysis(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "flops" in attrs
        flops = attrs["flops"].data
        # 0.25 * (a + b + c + d) -> 3 adds + 1 mul = 4 flops
        assert flops == 4

    def test_unique_loads(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module = self._run_analysis(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "unique_loads" in attrs
        assert attrs["unique_loads"].data == 4

    def test_arithmetic_intensity(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module = self._run_analysis(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "arithmetic_intensity" in attrs
        ai = attrs["arithmetic_intensity"].value.data
        # AI = 4 / ((4 + 1) * 8) = 0.1
        assert abs(ai - 0.1) < 1e-9


class TestOptimizationPasses:
    """Test tiling, temporal blocking, decomposition, and verification passes."""

    def _run_through_tiling(self, stencil_fn, **kwargs):
        """Run normalize -> analysis -> tiling on a stencil."""
        ctx = Context()
        module = stencil_fn._ir.clone()
        NormalizePass().apply(ctx, module)
        AnalysisPass().apply(ctx, module)
        TilingPass(**kwargs).apply(ctx, module)
        return module, ctx

    def _get_apply_attrs(self, module):
        from xdsl.dialects.stencil import ApplyOp
        for op in module.walk():
            if isinstance(op, ApplyOp):
                return dict(op.attributes)
        pytest.fail("No stencil.ApplyOp found")

    def test_tiling_produces_power_of_2(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module, _ = self._run_through_tiling(heat2d)
        attrs = self._get_apply_attrs(module)
        assert "tile_sizes" in attrs
        tiles = [a.data for a in attrs["tile_sizes"]]
        for t in tiles:
            assert t > 0 and (t & (t - 1)) == 0, f"tile {t} not power of 2"

    def test_tiling_respects_shared_mem_budget(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        # Small budget forces smaller tiles
        module_small, _ = self._run_through_tiling(heat2d, shared_mem_bytes=4096)
        attrs_small = self._get_apply_attrs(module_small)
        tiles_small = [a.data for a in attrs_small["tile_sizes"]]

        # Large budget allows bigger tiles
        module_large, _ = self._run_through_tiling(heat2d, shared_mem_bytes=256 * 1024)
        attrs_large = self._get_apply_attrs(module_large)
        tiles_large = [a.data for a in attrs_large["tile_sizes"]]

        assert tiles_large[0] >= tiles_small[0]

    def test_tiling_1d(self):
        @stencil(ndim=1, order=2)
        def heat1d(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        module, _ = self._run_through_tiling(heat1d)
        attrs = self._get_apply_attrs(module)
        tiles = [a.data for a in attrs["tile_sizes"]]
        assert len(tiles) == 1
        assert tiles[0] > 0

    def test_temporal_blocking_sets_attributes(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        module, ctx = self._run_through_tiling(heat)
        TemporalBlockingPass(shared_mem_bytes=49152).apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "temporal_steps" in attrs
        assert attrs["temporal_steps"].data >= 1
        assert "expanded_halo" in attrs
        assert "temporal_buffer_count" in attrs

    def test_temporal_blocking_increases_with_shared_mem(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        # Small shared mem -> few temporal steps
        module_small, ctx1 = self._run_through_tiling(heat, shared_mem_bytes=4096)
        TemporalBlockingPass(shared_mem_bytes=4096).apply(ctx1, module_small)
        attrs_small = self._get_apply_attrs(module_small)
        t_small = attrs_small["temporal_steps"].data

        # Large shared mem -> more temporal steps
        module_large, ctx2 = self._run_through_tiling(heat, shared_mem_bytes=256 * 1024)
        TemporalBlockingPass(shared_mem_bytes=256 * 1024).apply(ctx2, module_large)
        attrs_large = self._get_apply_attrs(module_large)
        t_large = attrs_large["temporal_steps"].data

        assert t_large >= t_small

    def test_boundary_pass_attaches_padding_mode(self):
        @stencil(ndim=2, order=2, boundary="dirichlet")
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ctx = Context()
        module = heat2d._ir.clone()
        NormalizePass().apply(ctx, module)
        BoundaryPass().apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "padding_mode" in attrs
        assert attrs["padding_mode"].data == "zero"

    def test_boundary_pass_periodic(self):
        @stencil(ndim=1, order=2, boundary="periodic")
        def heat(u, i):
            return 0.5 * u[i - 1] + 0.5 * u[i + 1]

        ctx = Context()
        module = heat._ir.clone()
        NormalizePass().apply(ctx, module)
        BoundaryPass().apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "boundary_fn" in attrs
        assert attrs["boundary_fn"].data == "host_side"

    def test_fusion_pass_marks_group(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ctx = Context()
        module = heat2d._ir.clone()
        NormalizePass().apply(ctx, module)
        FusionPass().apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "fusion_group" in attrs
        assert attrs["fusion_group"].data == 0

    def test_bricked_layout_pass(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        ctx = Context()
        module = heat2d._ir.clone()
        NormalizePass().apply(ctx, module)
        BrickedLayoutPass(brick_size=64).apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "layout" in attrs
        assert attrs["layout"].data == "bricked"
        assert "brick_size" in attrs
        assert attrs["brick_size"].data == 64

    def test_decomposition_sets_multi_gpu_attrs(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module, ctx = self._run_through_tiling(heat2d)
        DecompositionPass(num_gpus=4, topology=(2, 2)).apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "num_gpus" in attrs
        assert attrs["num_gpus"].data == 4
        assert "topology" in attrs
        assert "split_axis" in attrs

    def test_halo_exchange_pass(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        module, ctx = self._run_through_tiling(heat2d)
        DecompositionPass(num_gpus=4, topology=(2, 2)).apply(ctx, module)
        HaloExchangePass().apply(ctx, module)
        attrs = self._get_apply_attrs(module)
        assert "halo_exchange_enabled" in attrs
        assert attrs["halo_exchange_enabled"].data == 1
        assert "halo_exchange_width" in attrs

    def test_verify_pass_accepts_valid_ir(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        module, ctx = self._run_through_tiling(heat)
        # VerifyPass should pass on valid IR without raising
        VerifyPass().apply(ctx, module)

    def test_verify_pass_checks_power_of_2(self):
        """VerifyPass should detect non-power-of-2 tile sizes."""
        from xdsl.dialects.stencil import ApplyOp
        from xdsl.dialects.builtin import ArrayAttr, IntAttr

        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        ctx = Context()
        module = heat._ir.clone()
        NormalizePass().apply(ctx, module)
        AnalysisPass().apply(ctx, module)
        # Manually set a bad tile size
        for op in module.walk():
            if isinstance(op, ApplyOp):
                op.attributes["tile_sizes"] = ArrayAttr([IntAttr(100)])
                break
        with pytest.raises(ValueError, match="not a power of 2"):
            VerifyPass().apply(ctx, module)


# ===================================================================
# LEVEL 1.5: LOWERING TESTS (no GPU required)
# ===================================================================


class TestLowering:
    """Test IR -> Python source generation via lower_stencil_to_python."""

    def test_1d_produces_valid_python(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        code = lower_stencil_to_python(heat._ir, tile_sizes=(128,), halo_widths=(1,))
        ast.parse(code)

    def test_2d_produces_valid_python(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1))
        ast.parse(code)

    def test_3d_produces_valid_python(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        code = lower_stencil_to_python(lap3d._ir, tile_sizes=(32, 32, 32), halo_widths=(1, 1, 1))
        ast.parse(code)

    def test_has_kernel_decorator(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1))
        assert "@ct.kernel" in code

    def test_has_launcher(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1))
        assert "def launch_heat2d" in code

    def test_has_ct_imports(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.5 * u[i - 1] + 0.5 * u[i + 1]

        code = lower_stencil_to_python(heat._ir, tile_sizes=(128,), halo_widths=(1,))
        assert "import cuda.tile as ct" in code
        assert "import cupy as cp" in code

    def test_has_slice_ops(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1))
        assert ".slice(" in code
        assert "ct.load(" in code
        assert "ct.store(" in code

    def test_temporal_blocking_generates_loop(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(
            heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1), temporal_steps=3
        )
        ast.parse(code)
        assert "range(3)" in code
        assert "bufs" in code

    def test_temporal_blocking_buffer_swap(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        code = lower_stencil_to_python(
            heat._ir, tile_sizes=(128,), halo_widths=(1,), temporal_steps=5
        )
        ast.parse(code)
        assert "range(5)" in code
        assert "bufs.append" in code
        assert "bufs[_step]" in code

    def test_boundary_generates_periodic_function(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        code = lower_stencil_to_python(
            heat._ir, tile_sizes=(128,), halo_widths=(1,),
            boundary_spec={"type": "periodic"},
        )
        ast.parse(code)
        assert "apply_boundary" in code
        assert "periodic" in code.lower()

    def test_boundary_generates_dirichlet_function(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(
            heat2d._ir, tile_sizes=(64, 64), halo_widths=(1, 1),
            boundary_spec={"type": "dirichlet", "value": 0.0},
        )
        ast.parse(code)
        assert "apply_boundary" in code

    def test_defaults_when_no_tile_halo(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        code = lower_stencil_to_python(heat2d._ir)
        ast.parse(code)
        # Defaults: TX=64, TY=64 for 2D; HX=1, HY=1 for order=2
        assert "TX, TY = 64, 64" in code
        assert "HX, HY = 1, 1" in code


# ===================================================================
# LEVEL 1.75: COMPILE API TESTS (no GPU required)
# ===================================================================


class TestCompileAPI:
    """Test the top-level compile() function."""

    def test_compile_1d(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat)
        assert result.name == "heat"
        assert result.ndim == 1
        assert len(result.halo_widths) == 1
        assert len(result.tile_sizes) == 1
        ast.parse(result.code)

    def test_compile_2d(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        result = stencil_compile(heat2d)
        assert result.ndim == 2
        assert len(result.tile_sizes) == 2
        ast.parse(result.code)

    def test_compile_3d(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        result = stencil_compile(lap3d)
        assert result.ndim == 3
        ast.parse(result.code)

    def test_compile_has_analysis(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        result = stencil_compile(heat2d)
        assert "flops" in result.analysis
        assert "bound" in result.analysis

    def test_compile_halo_widths_correct(self):
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return -u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]

        result = stencil_compile(lap4)
        assert result.halo_widths == (2,)

    def test_compile_emit_to_file(self, tmp_path):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat)
        path = tmp_path / "heat_kernel.py"
        result.emit_to_file(str(path))
        assert path.exists()
        ast.parse(path.read_text())

    def test_compile_no_temporal_blocking(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat, temporal_blocking=False)
        assert result.temporal_steps == 1

    def test_compile_with_temporal_blocking(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat, temporal_blocking=True)
        assert result.temporal_steps >= 1

    def test_custom_pipeline(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        pipeline = Pipeline.single_gpu(shared_mem_bytes=49152)
        result = stencil_compile(heat, pipeline=pipeline)
        ast.parse(result.code)

    def test_compile_preserves_ref_fn(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat)
        assert result._ref_fn is not None
        assert callable(result._ref_fn)

    def test_compile_code_has_ct_kernel(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        result = stencil_compile(heat2d)
        assert "@ct.kernel" in result.code
        assert "ct.launch(" in result.code


# ===================================================================
# LEVEL 2: REFERENCE TESTS (no GPU, NumPy only)
# ===================================================================


class TestReference:
    """Test CPU reference implementation."""

    def test_1d_heat_smoothing(self):
        """1D heat equation should smooth a delta function."""
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        u = np.zeros(10)
        u[5] = 1.0
        result = apply_stencil(u, heat, ndim=1, halo_widths=(1,))
        # Interior should be smoothed
        assert result[5] < 1.0  # peak reduced
        assert result[4] > 0.0  # neighbors increased
        assert result[6] > 0.0

    def test_2d_laplacian_constant_is_zero(self):
        """2D Laplacian of a constant field should be zero."""
        def lap(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]

        u = np.ones((10, 10))
        result = apply_stencil(u, lap, ndim=2, halo_widths=(1, 1))
        assert np.allclose(result[1:-1, 1:-1], 0.0, atol=1e-14)

    def test_3d_laplacian_constant_is_zero(self):
        """3D Laplacian of a constant field should be zero."""
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
                - 6 * u[i, j, k]
            )

        u = np.ones((8, 8, 8))
        result = apply_stencil(u, lap3d, ndim=3, halo_widths=(1, 1, 1))
        assert np.allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-14)

    def test_1d_identity(self):
        """Identity stencil u[i] should return the same array."""
        def identity(u, i):
            return u[i]

        u = np.arange(10, dtype=float)
        result = apply_stencil(u, identity, ndim=1, halo_widths=(1,))
        np.testing.assert_array_equal(result[1:-1], u[1:-1])

    def test_2d_averaging(self):
        """2D averaging stencil on uniform field."""
        def avg(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        u = 5.0 * np.ones((8, 8))
        result = apply_stencil(u, avg, ndim=2, halo_widths=(1, 1))
        assert np.allclose(result[1:-1, 1:-1], 5.0, atol=1e-14)

    def test_time_march(self):
        """time_march should produce a trajectory of snapshots."""
        from cutile.reference.stencil_ref import time_march

        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        u0 = np.zeros(20)
        u0[10] = 1.0
        history = time_march(u0, heat, ndim=1, halo_widths=(1,), steps=5)
        assert len(history) == 6  # initial + 5 steps
        # Energy should dissipate (max decreases each step)
        for k in range(1, len(history)):
            assert np.max(history[k]) <= np.max(history[k - 1]) + 1e-14


# ===================================================================
# CONFIG TESTS
# ===================================================================


class TestConfig:
    """Test configuration module."""

    def test_gpu_presets_exist(self):
        assert "H100_SXM" in GPU_PRESETS
        assert "A100_80GB" in GPU_PRESETS
        assert "RTX_PRO_6000" in GPU_PRESETS

    def test_all_presets_have_positive_bandwidth(self):
        for name, preset in GPU_PRESETS.items():
            assert preset.peak_bandwidth_gbs > 0, f"{name} has zero bandwidth"

    def test_hardware_spec_creation(self):
        hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
        assert hw.shared_mem_bytes == 49152

    def test_hardware_spec_from_preset(self):
        hw = HardwareSpec.from_preset("H100_SXM", "float64")
        assert hw.shared_mem_bytes > 0
        assert hw.peak_bandwidth_gbs > 0
        assert hw.dtype_bytes == 8

    def test_hardware_spec_from_preset_float32(self):
        hw = HardwareSpec.from_preset("H100_SXM", "float32")
        assert hw.dtype_bytes == 4

    def test_preset_peak_gflops(self):
        preset = GPU_PRESETS["H100_SXM"]
        assert preset.peak_gflops("float64") > 0
        assert preset.peak_gflops("float32") > preset.peak_gflops("float64")


# ===================================================================
# PIPELINE TESTS
# ===================================================================


class TestPipeline:
    """Test the Pipeline class."""

    def test_pipeline_single_gpu_factory(self):
        pipeline = Pipeline.single_gpu()
        assert len(pipeline.passes) > 0

    def test_pipeline_runs_on_module(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        pipeline = Pipeline.single_gpu()
        module = heat2d._ir.clone()
        result = pipeline.run(module)
        assert result is module  # modifies in-place and returns it

    def test_pipeline_add_chaining(self):
        pipeline = Pipeline()
        result = pipeline.add(NormalizePass()).add(AnalysisPass())
        assert result is pipeline
        assert len(pipeline.passes) == 2

    def test_pipeline_custom_passes(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        pipeline = Pipeline()
        pipeline.add(NormalizePass())
        pipeline.add(AnalysisPass())
        pipeline.add(RooflinePass())
        module = heat2d._ir.clone()
        pipeline.run(module)

        from xdsl.dialects.stencil import ApplyOp
        for op in module.walk():
            if isinstance(op, ApplyOp):
                assert "halo_widths" in op.attributes
                assert "flops" in op.attributes
                return
        pytest.fail("No ApplyOp found after running custom pipeline")


# ===================================================================
# LEVEL 3: GPU CONVERGENCE TESTS (requires GPU)
# ===================================================================

try:
    import cupy as cp

    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


@gpu_required
class TestGPUConvergence:
    """GPU convergence tests: generated kernel vs CPU reference."""

    def _validate_stencil(self, stencil_fn, domain, atol=1e-10):
        """Compile stencil, run on GPU, compare to CPU reference."""
        result = stencil_compile(stencil_fn, temporal_blocking=False)
        mod = result.load_module()
        launch_fn = getattr(mod, f"launch_{result.name}")

        ndim = result.ndim
        halo = result.halo_widths

        # Create input with halo padding
        shape = tuple(d + 2 * h for d, h in zip(domain, halo))
        np.random.seed(42)
        u_np = np.random.randn(*shape).astype(np.float64)

        # GPU
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)
        launch_fn(u_gpu, out_gpu)
        cp.cuda.Device(0).synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        cpu_result = apply_stencil(u_np, stencil_fn._fn, ndim=ndim, halo_widths=halo)

        # Compare interior
        slices = tuple(slice(h, -h) for h in halo)
        gpu_interior = gpu_result[slices]
        cpu_interior = cpu_result[slices]

        maxdiff = np.max(np.abs(gpu_interior - cpu_interior))
        assert np.allclose(gpu_interior, cpu_interior, atol=atol), (
            f"GPU vs CPU max diff = {maxdiff:.2e}"
        )

    def test_1d_heat_convergence(self):
        @stencil(ndim=1, order=2)
        def heat1d(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        self._validate_stencil(heat1d, domain=(512,))

    def test_2d_heat_convergence(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        self._validate_stencil(heat2d, domain=(128, 128))

    def test_3d_laplacian_convergence(self):
        @stencil(ndim=3, order=2)
        def lap3d(u, i, j, k):
            return (
                u[i - 1, j, k] + u[i + 1, j, k]
                + u[i, j - 1, k] + u[i, j + 1, k]
                + u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

        self._validate_stencil(lap3d, domain=(32, 32, 32))

    def test_4th_order_convergence(self):
        @stencil(ndim=1, order=4)
        def lap4(u, i):
            return (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / 12.0

        self._validate_stencil(lap4, domain=(256,))

    def test_temporal_blocking_code_is_valid(self):
        """Temporal blocking should produce syntactically valid code."""
        @stencil(ndim=1, order=2)
        def heat1d(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        T = 3
        result_nt = stencil_compile(heat1d, temporal_blocking=False)
        code_t = lower_stencil_to_python(
            heat1d._ir,
            tile_sizes=result_nt.tile_sizes,
            halo_widths=(1,),
            temporal_steps=T,
        )
        ast.parse(code_t)
        assert f"range({T})" in code_t


@gpu_required
class TestGPUBenchmark:
    """Basic benchmark smoke tests."""

    def test_benchmark_runs(self):
        @stencil(ndim=2, order=2)
        def heat2d(u, i, j):
            return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

        result = stencil_compile(heat2d)
        metrics = result.benchmark()
        assert "gpoints_per_s" in metrics
        assert "gbytes_per_s" in metrics
        assert metrics["gpoints_per_s"] > 0

    def test_validate_runs(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

        result = stencil_compile(heat)
        ok = result.validate()
        assert ok is True
