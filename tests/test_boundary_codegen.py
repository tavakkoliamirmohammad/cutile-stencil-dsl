"""Tests for boundary condition code generation."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.dsl.registry import clear
from cutile_stencil.dsl.boundary import BoundarySpec, BoundaryType
from cutile_stencil.dsl.pipeline import compile


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestDirichletCodegen:
    def test_1d_dirichlet_has_padding_mode(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "PaddingMode.ZERO" in result.code

    def test_2d_dirichlet_has_padding_mode(self):
        @stencil(ndim=2, order=2, boundary=BoundarySpec.dirichlet(2))
        def lap(u, i, j):
            return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
        result = compile(lap, domain=(256, 256))
        assert "PaddingMode.ZERO" in result.code

    def test_dirichlet_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.dirichlet(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestPeriodicCodegen:
    def test_1d_periodic_has_boundary_fill(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.periodic(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "apply_boundary" in result.code
        assert "periodic" in result.code.lower()

    def test_periodic_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.periodic(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestNeumannCodegen:
    def test_1d_neumann_has_boundary_fill(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.neumann(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "apply_boundary" in result.code
        assert "Neumann" in result.code or "neumann" in result.code.lower()

    def test_neumann_valid_python(self):
        @stencil(ndim=1, order=2, boundary=BoundarySpec.neumann(1))
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        ast.parse(result.code)


class TestNoBoundary:
    def test_no_boundary_no_padding_mode(self):
        @stencil(ndim=1, order=2)
        def heat(u, i):
            return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]
        result = compile(heat, domain=(1024,))
        assert "PaddingMode" not in result.code
        assert "apply_boundary" not in result.code
