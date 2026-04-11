"""Tests for Jacobi preconditioner generation."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import StencilSpec, OffsetAccess
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint
from cutile_stencil.solvers.preconditioner import (
    generate_jacobi_preconditioner,
    get_center_coefficient,
)


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestCenterCoefficient:
    def test_1d_laplacian(self):
        @stencil(ndim=1, order=2)
        def lap1d(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap1d._stencil_spec
        extract_footprint(spec)
        coeff = get_center_coefficient(spec)
        # 2 non-center accesses → center = -2
        assert coeff == -2.0

    def test_2d_laplacian(self):
        @stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        spec = lap2d._stencil_spec
        extract_footprint(spec)
        coeff = get_center_coefficient(spec)
        # 4 non-center accesses → center = -4
        assert coeff == -4.0

    def test_empty_accesses(self):
        spec = StencilSpec(
            name="empty", ndim=1, order=2,
            inputs=("u",), output="result",
            update_fn=lambda u, i: u,
        )
        coeff = get_center_coefficient(spec)
        assert coeff == 1.0


class TestJacobiKernelGeneration:
    def test_generates_valid_python(self):
        @stencil(ndim=1, order=2)
        def lap1d(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap1d._stencil_spec
        extract_footprint(spec)
        code = generate_jacobi_preconditioner(spec)
        ast.parse(code)

    def test_contains_kernel_and_launcher(self):
        @stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        spec = lap2d._stencil_spec
        extract_footprint(spec)
        code = generate_jacobi_preconditioner(spec)
        assert "@ct.kernel" in code
        assert "jacobi_precondition_kernel" in code
        assert "launch_jacobi_precondition" in code

    def test_inv_diag_correct(self):
        @stencil(ndim=1, order=2)
        def lap1d(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap1d._stencil_spec
        extract_footprint(spec)
        code = generate_jacobi_preconditioner(spec)
        # Center coeff = -2, so INV_DIAG = -0.5
        assert "INV_DIAG = -0.5" in code

    def test_custom_tile_size(self):
        spec = StencilSpec(
            name="test", ndim=1, order=2,
            inputs=("u",), output="result",
            update_fn=lambda u, i: u,
            accesses=[OffsetAccess("u", (-1,)), OffsetAccess("u", (0,)), OffsetAccess("u", (1,))],
        )
        code = generate_jacobi_preconditioner(spec, tile_size=512)
        assert "TILE = 512" in code
