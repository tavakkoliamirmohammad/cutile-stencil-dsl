"""Element-wise ``max``/``min``/``abs`` and ``where`` in stencil bodies.

bricklib's ``cond`` benchmark is a 7-point stencil of ``max(u, 0)`` terms
followed by an absolute value.  The frontend lowers the Python builtins and
``where(cond, a, b)`` to ``arith.maximumf``/``minimumf``, ``math.absf`` and
``arith.cmpf`` + ``arith.select``; the emitter maps them to ``ct.maximum``,
``ct.minimum``, ``ct.abs`` and ``ct.where``; the NumPy reference executor
runs the same Python function with array-aware builtins so validation still
works.
"""

import ast

import numpy as np
import pytest

from cutile import stencil, where
from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
from cutile.reference.stencil_ref import apply_stencil


def _cond():
    @stencil(ndim=3, order=2)
    def cond(u, i, j, k):
        return abs(0.5 * max(u[i, j, k], 0.0) + 0.25 * max(u[i + 1, j, k], 0.0)
                   + 0.25 * max(u[i - 1, j, k], 0.0) - 0.125 * min(u[i, j + 1, k], 0.0))
    return cond


def _clip():
    @stencil(ndim=1, order=2)
    def clip(u, i):
        return where(u[i] > 0.5, u[i - 1], u[i + 1]) + where(u[i] <= 0.0, 1.0, 0.0)
    return clip


def _result_line(code):
    return next(l for l in code.splitlines() if l.strip().startswith("result ="))


class TestParse:
    def test_max_min_abs_lower_to_arith_and_math_ops(self):
        ir = str(_cond()._ir)
        assert "arith.maximumf" in ir
        assert "arith.minimumf" in ir
        assert "math.absf" in ir

    def test_where_lowers_to_cmpf_and_select(self):
        ir = str(_clip()._ir)
        assert "arith.cmpf" in ir
        assert "arith.select" in ir

    def test_unsupported_call_is_rejected(self):
        with pytest.raises(Exception, match="Unsupported"):
            @stencil(ndim=1, order=2)
            def bad(u, i):
                return round(u[i])


class TestEmit:
    def test_cond_emits_cutile_maximum_and_abs(self):
        code = lower_stencil_to_python(_cond()._ir, tile_sizes=(1, 1, 256), halo_widths=(1, 1, 16))
        ast.parse(code)
        line = _result_line(code)
        assert line.count("ct.maximum(") == 3
        assert "ct.minimum(" in line
        assert line.strip().startswith("result = ct.abs(")

    def test_where_emits_cutile_where_with_comparison(self):
        code = lower_stencil_to_python(_clip()._ir, tile_sizes=(1024,), halo_widths=(16,))
        ast.parse(code)
        line = _result_line(code)
        assert "ct.where(" in line
        assert "> 0.5" in line and "<= 0.0" in line

    def test_constant_max_is_folded(self):
        @stencil(ndim=1, order=2)
        def f(u, i):
            return max(2.0, 3.0) * u[i] + abs(-0.5) * u[i - 1]

        code = lower_stencil_to_python(f._ir, tile_sizes=(1024,), halo_widths=(16,))
        line = _result_line(code)
        assert "ct.maximum" not in line and "ct.abs" not in line
        assert "3.0 *" in line and "0.5 *" in line


class TestReference:
    def test_reference_runs_python_builtins_element_wise(self):
        u = np.random.rand(6, 6, 6) - 0.5
        out = apply_stencil(u, _cond()._fn, 3, (1, 1, 1))
        c = u[1:-1, 1:-1, 1:-1]
        exp = np.abs(0.5 * np.maximum(c, 0.0) + 0.25 * np.maximum(u[2:, 1:-1, 1:-1], 0.0)
                     + 0.25 * np.maximum(u[:-2, 1:-1, 1:-1], 0.0) - 0.125 * np.minimum(u[1:-1, 2:, 1:-1], 0.0))
        assert np.allclose(out[1:-1, 1:-1, 1:-1], exp)

    def test_reference_where(self):
        u = np.random.rand(20)
        out = apply_stencil(u, _clip()._fn, 1, (1,))
        c = u[1:-1]
        exp = np.where(c > 0.5, u[:-2], u[2:]) + np.where(c <= 0.0, 1.0, 0.0)
        assert np.allclose(out[1:-1], exp)


try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")
class TestGPU:
    def test_cond_matches_reference_on_gpu(self):
        from cutile import compile as stencil_compile
        result = stencil_compile(_cond(), temporal_blocking=False)
        np.random.seed(3)
        u = np.random.randn(64, 64, 64 + 2 * result.halo_widths[2] - 2)
        assert result.validate(u_input=u, atol=1e-12)

    def test_where_matches_reference_on_gpu(self):
        from cutile import compile as stencil_compile
        result = stencil_compile(_clip(), temporal_blocking=False)
        assert result.validate(atol=1e-12)
