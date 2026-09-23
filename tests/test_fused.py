"""Fused multi-output kernels through the standard emitter.

Several stencils over one domain (CNS hypterm: five outputs from eight
fields; Gray-Scott u/v) compile into a single kernel that loads each distinct
(field, offset) once and stores every output.  Inputs are merged by
*parameter name*, which the frontend records as ``arg_names``, so stencils
that read different subsets of the fields fuse correctly.  Tile and hint
selection use the merged access count, exactly as for a single stencil.
"""

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutile import stencil
from cutile.dialects.cutile_stencil.dialect import FuncOp
from cutile.lowering.fusion_emitter import lower_fused_stencils_to_python
from cutile.runtime import regcheck
from cutile.runtime.launcher import compile_fused


def _pair():
    @stencil(ndim=2, order=2)
    def a(u, v, i, j):
        return u[i + 1, j] + 0.3 * v[i, j]

    @stencil(ndim=2, order=2)
    def b(v, w, i, j):
        return 0.3 * v[i, j] - w[i, j - 1]

    return a, b


def _func_op(fn):
    return next(op for op in fn._ir.body.ops if isinstance(op, FuncOp))


class TestArgNames:
    def test_frontend_records_array_parameter_names(self):
        a, b = _pair()
        assert [s.data for s in _func_op(a).arg_names.data] == ["u", "v"]
        assert [s.data for s in _func_op(b).arg_names.data] == ["v", "w"]

    def test_only_subscripted_parameters_are_recorded(self):
        @stencil(ndim=1, order=2)
        def f(u, unused, i):
            return u[i - 1] + u[i + 1]

        assert [s.data for s in _func_op(f).arg_names.data] == ["u"]


class TestFusedLowering:
    def _code(self):
        a, b = _pair()
        return lower_fused_stencils_to_python([a._ir, b._ir], tile_sizes=(1, 256), halo_widths=(1, 16))

    def test_inputs_are_merged_by_name(self):
        code = self._code()
        ast.parse(code)
        assert "def a_b_kernel(u, v, w, output_0, output_1, consts, TX: ConstInt, TY: ConstInt, HX: ConstInt, HY: ConstInt):" in code
        assert "def launch_a_b(u, v, w, a_out, b_out, stream=None):" in code

    def test_shared_access_is_loaded_once_and_hoisted(self):
        code = self._code()
        assert code.count("ct.load(v_0_0,") == 1
        assert "t_v_0_0 = ct.load(v_0_0," in code

    def test_single_use_access_is_still_loaded_inline(self):
        code = self._code()
        assert "t_u_p1_0 =" not in code
        assert "result_0 = ct.load(u_p1_0," in code

    def test_each_output_is_computed_and_stored(self):
        code = self._code()
        assert "result_0 = " in code and "result_1 = " in code
        assert "ct.store(out_0, index=(bx, by), tile=result_0)" in code
        assert "ct.store(out_1, index=(bx, by), tile=result_1)" in code

    def test_exact_constants_are_shared_across_outputs(self):
        code = self._code()
        assert "_c0" in code and "_c1" not in code
        assert "_KERNEL_CONSTANTS = [0.3]" in code

    def test_temporal_blocking_is_rejected(self):
        a, b = _pair()
        with pytest.raises(NotImplementedError):
            lower_fused_stencils_to_python([a._ir, b._ir], tile_sizes=(1, 256), halo_widths=(1, 16), temporal_steps=2)


class TestCompileFused:
    def test_compile_fused_is_part_of_the_public_api(self):
        import cutile
        assert cutile.compile_fused is compile_fused
        assert "compile_fused" in cutile.__all__

    def test_result_describes_the_fused_kernel(self):
        a, b = _pair()
        r = compile_fused([a, b])
        assert r.name == "a_b"
        assert r.ndim == 2
        assert r.inputs == ("u", "v", "w")
        assert r.outputs == ("a", "b")
        assert r.halo_widths == (1, 16)
        assert r.tile_sizes == (1, 256)      # 4 merged accesses: narrow tier
        assert r.kernel_hints == {}
        assert "def launch_a_b(u, v, w, a_out, b_out, stream=None):" in r.code

    def test_tile_tier_and_probe_use_the_merged_kernel(self, monkeypatch):
        from wide_stencils import box27

        @stencil(ndim=3, order=2)
        def lap(v, i, j, k):
            return (v[i - 1, j, k] + v[i + 1, j, k] + v[i, j - 1, k] + v[i, j + 1, k]
                    + v[i, j, k - 1] + v[i, j, k + 1] - 6.0 * v[i, j, k])

        calls = []
        monkeypatch.setattr(regcheck, "probe_resources", lambda **kw: calls.append(kw) or regcheck.Resources(96, 0))
        r = compile_fused([box27, lap])          # 27 + 7 = 34 merged accesses: wide tier
        assert r.tile_sizes == (1, 2, 64)
        assert r.kernel_hints == {}
        assert r.resources == regcheck.Resources(96, 0)
        assert calls[0]["num_inputs"] == 2 and calls[0]["num_outputs"] == 2

    def test_explicit_occupancy_wins(self):
        a, b = _pair()
        r = compile_fused([a, b], occupancy=8)
        assert r.kernel_hints == {"occupancy": 8}
        assert "@ct.kernel(occupancy=8)" in r.code

    def test_temporal_blocking_is_rejected(self):
        a, b = _pair()
        with pytest.raises(NotImplementedError):
            compile_fused([a, b], temporal_steps=2)


try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
class TestFusedGPU:
    def test_fused_outputs_match_the_reference(self):
        a, b = _pair()
        assert compile_fused([a, b]).validate(atol=1e-12)

    def test_member_with_an_unused_parameter_validates(self):
        # the launcher takes only the fields a stencil reads, but its Python
        # reference still has the full parameter list
        @stencil(ndim=1, order=2)
        def p(u, unused, i):
            return u[i - 1] - 2.0 * u[i] + u[i + 1]

        @stencil(ndim=1, order=2)
        def q(unused, v, i):
            return 0.5 * v[i + 1]

        r = compile_fused([p, q])
        assert r.inputs == ("u", "v")
        assert r.validate(atol=1e-12)

    def test_three_field_products_match_the_reference(self):
        @stencil(ndim=3, order=2)
        def flux(m, u, p, i, j, k):
            return 0.8 * (m[i + 1, j, k] * u[i + 1, j, k] - m[i - 1, j, k] * u[i - 1, j, k]) + p[i, j, k]

        @stencil(ndim=3, order=2)
        def div(m, w, i, j, k):
            return m[i, j, k + 1] * w[i, j, k + 1] - m[i, j, k - 1] * w[i, j, k - 1]

        r = compile_fused([flux, div])
        assert r.inputs == ("m", "u", "p", "w")
        assert r.validate(atol=1e-12)
