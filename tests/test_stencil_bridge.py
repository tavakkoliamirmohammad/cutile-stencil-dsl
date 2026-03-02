"""Tests for the SpMV-as-stencil bridge."""

import ast

import numpy as np
import pytest

from cutile_stencil.solvers.stencil_bridge import (
    dia_to_stencil_spec,
    laplacian_stencil_spec,
    stencil_spmv_kernel,
)
from cutile_stencil.solvers.formats import laplacian_1d_dia, laplacian_2d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv
from cutile_stencil.reference.stencil_ref import apply_stencil, _ArrayProxy


class TestDiaToStencilSpec:
    """dia_to_stencil_spec creates correct StencilSpec objects."""

    def test_1d_identity(self):
        """Single offset (0,) with coeff 1 = identity stencil."""
        spec = dia_to_stencil_spec("id", 1, [(0,)], [1.0])
        assert spec.name == "id"
        assert spec.ndim == 1
        assert spec.order == 0
        assert spec.halo_widths == (0,)
        assert len(spec.accesses) == 1
        assert spec.accesses[0].offsets == (0,)

    def test_1d_laplacian_accesses(self):
        spec = dia_to_stencil_spec("lap1d", 1, [(-1,), (0,), (1,)], [-1.0, 2.0, -1.0])
        assert spec.ndim == 1
        assert spec.order == 2
        assert spec.halo_widths == (1,)
        assert len(spec.accesses) == 3
        offsets = [a.offsets for a in spec.accesses]
        assert (-1,) in offsets
        assert (0,) in offsets
        assert (1,) in offsets

    def test_2d_five_point_halo(self):
        offsets = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
        coeffs = [-1.0, -1.0, 4.0, -1.0, -1.0]
        spec = dia_to_stencil_spec("lap2d", 2, offsets, coeffs)
        assert spec.ndim == 2
        assert spec.halo_widths == (1, 1)
        assert len(spec.accesses) == 5

    def test_3d_seven_point(self):
        offsets = [
            (0, 0, -1), (0, -1, 0), (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
        ]
        coeffs = [-1.0] * 3 + [6.0] + [-1.0] * 3
        spec = dia_to_stencil_spec("lap3d", 3, offsets, coeffs)
        assert spec.ndim == 3
        assert spec.halo_widths == (1, 1, 1)
        assert len(spec.accesses) == 7

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            dia_to_stencil_spec("bad", 1, [(0,)], [1.0, 2.0])

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2 elements"):
            dia_to_stencil_spec("bad", 2, [(0,)], [1.0])

    def test_update_fn_callable(self):
        spec = dia_to_stencil_spec("test", 1, [(-1,), (0,), (1,)], [1.0, -2.0, 1.0])
        assert callable(spec.update_fn)


class TestLaplacianStencilSpec:
    """laplacian_stencil_spec convenience constructor."""

    def test_1d(self):
        spec = laplacian_stencil_spec(1)
        assert spec.name == "laplacian_1d"
        assert spec.ndim == 1
        assert spec.halo_widths == (1,)

    def test_2d(self):
        spec = laplacian_stencil_spec(2)
        assert spec.name == "laplacian_2d"
        assert spec.ndim == 2
        assert spec.halo_widths == (1, 1)

    def test_3d(self):
        spec = laplacian_stencil_spec(3)
        assert spec.name == "laplacian_3d"
        assert spec.ndim == 3
        assert spec.halo_widths == (1, 1, 1)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="Unsupported ndim"):
            laplacian_stencil_spec(4)

    def test_dtype_propagated(self):
        spec = laplacian_stencil_spec(1, dtype="float32")
        assert spec.dtype == "float32"


class TestStencilMatchesDIA:
    """Stencil application matches DIA SpMV on interior points."""

    def test_1d_laplacian_matches_dia_interior(self):
        N = 20
        halo = 1
        np.random.seed(42)
        u_full = np.random.randn(N + 2 * halo)

        # DIA SpMV
        A = laplacian_1d_dia(N)
        y_dia = dia_spmv(A, u_full[halo:-halo])

        # Stencil apply
        spec = laplacian_stencil_spec(1)
        out = apply_stencil(u_full, spec)
        y_stencil = out[halo:-halo]

        # Interior rows (1:-1) match because DIA boundary rows lack halo data
        np.testing.assert_allclose(y_dia[1:-1], y_stencil[1:-1], atol=1e-12)

    def test_1d_laplacian_update_fn(self):
        """The update_fn produces correct values via _ArrayProxy."""
        spec = laplacian_stencil_spec(1)
        u = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        proxy = _ArrayProxy(u, (1,))
        result = spec.update_fn(proxy, 0)
        # -u[i-1] + 2*u[i] - u[i+1] for interior [1,4,9]:
        # -0 + 2 - 4 = -2, -1 + 8 - 9 = -2, -4 + 18 - 16 = -2
        expected = np.array([-2.0, -2.0, -2.0])
        np.testing.assert_allclose(result, expected)


class TestStencilSpmvKernel:
    """stencil_spmv_kernel produces valid Python code."""

    def test_1d_valid_python(self):
        spec = laplacian_stencil_spec(1)
        code = stencil_spmv_kernel(spec, (256,))
        ast.parse(code)

    def test_2d_valid_python(self):
        spec = laplacian_stencil_spec(2)
        code = stencil_spmv_kernel(spec, (32, 32))
        ast.parse(code)

    def test_contains_launch_function(self):
        spec = laplacian_stencil_spec(1)
        code = stencil_spmv_kernel(spec, (128,))
        assert f"def launch_{spec.name}" in code
