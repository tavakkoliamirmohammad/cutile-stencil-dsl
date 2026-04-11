"""Tests for preconditioned CG solver."""

import ast
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.solvers.stencil_cg import generate_stencil_cg


@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestPreconditionedCG:
    def test_generates_valid_python(self):
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,), preconditioned=True)
        ast.parse(code)

    def test_contains_preconditioner(self):
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,), preconditioned=True)
        assert "jacobi" in code.lower() or "precondition" in code.lower()
        assert "INV_DIAG" in code

    def test_unpreconditioned_has_no_preconditioner(self):
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,))
        assert "INV_DIAG" not in code

    def test_preconditioned_uses_r_dot_z(self):
        """Preconditioned CG should use dot(r, z) not dot(r, r) for beta."""
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,), preconditioned=True)
        assert "r_dot_z" in code
        # The standard r_dot_r variable used in unpreconditioned CG should not appear
        assert "r_dot_r" not in code

    def test_unpreconditioned_uses_r_dot_r(self):
        """Standard CG should use dot(r, r) for beta."""
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,))
        assert "r_dot_r" in code
        assert "r_dot_z" not in code

    def test_preconditioned_contains_launch_jacobi(self):
        """Preconditioned code should call launch_jacobi_precondition."""
        @stencil(ndim=1, order=2)
        def lap(u, i):
            return u[i - 1] - 2 * u[i] + u[i + 1]
        spec = lap._stencil_spec
        code = generate_stencil_cg(spec, domain=(256,), preconditioned=True)
        assert "launch_jacobi_precondition" in code

    def test_preconditioned_2d_generates_valid_python(self):
        """Preconditioned CG works for a 2D stencil."""
        @stencil(ndim=2, order=2)
        def lap2d(u, i, j):
            return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        spec = lap2d._stencil_spec
        code = generate_stencil_cg(spec, domain=(64, 64), preconditioned=True)
        ast.parse(code)
        assert "INV_DIAG" in code
