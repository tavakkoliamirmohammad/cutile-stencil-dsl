"""Tests for the matrix-free stencil CG solver."""

import ast

import numpy as np
import pytest

from cutile_stencil.solvers.stencil_bridge import laplacian_stencil_spec
from cutile_stencil.solvers.stencil_cg import generate_stencil_cg
from cutile_stencil.reference.stencil_cg_ref import stencil_cg_solve
from cutile_stencil.reference.stencil_ref import apply_stencil
from cutile_stencil.reference.cg_ref import cg_solve
from cutile_stencil.solvers.formats import laplacian_1d_dia, laplacian_2d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv


class TestGenerateStencilCG:
    """generate_stencil_cg produces valid Python code."""

    def test_1d_valid_python(self):
        spec = laplacian_stencil_spec(1)
        code = generate_stencil_cg(spec, (256,))
        ast.parse(code)

    def test_2d_valid_python(self):
        spec = laplacian_stencil_spec(2)
        code = generate_stencil_cg(spec, (32, 32))
        ast.parse(code)

    def test_contains_stencil_cg_solve(self):
        spec = laplacian_stencil_spec(1)
        code = generate_stencil_cg(spec, (128,))
        assert "def stencil_cg_solve(" in code

    def test_contains_stencil_launcher(self):
        spec = laplacian_stencil_spec(1)
        code = generate_stencil_cg(spec, (128,))
        assert "launch_laplacian_1d" in code

    def test_contains_dot_and_axpy(self):
        spec = laplacian_stencil_spec(1)
        code = generate_stencil_cg(spec, (128,))
        assert "def dot_kernel(" in code
        assert "def axpy_kernel(" in code


class TestStencilCGRef1D:
    """NumPy reference stencil CG converges for 1D Laplacian."""

    def test_1d_poisson_converges(self):
        N = 50
        spec = laplacian_stencil_spec(1)
        halo = 1

        # Known solution: u = sin(pi*x)
        h = 1.0 / (N + 1)
        x_grid = np.linspace(h, 1 - h, N)
        u_exact = np.zeros(N + 2 * halo)
        u_exact[halo:-halo] = np.sin(np.pi * x_grid)

        # RHS
        Au = apply_stencil(u_exact, spec)
        b = np.zeros(N + 2 * halo)
        b[halo:-halo] = Au[halo:-halo]

        x_sol, iters, residuals = stencil_cg_solve(spec, b, tol=1e-10, max_iter=500)
        assert residuals[-1] < 1e-10
        error = np.max(np.abs(x_sol[halo:-halo] - u_exact[halo:-halo]))
        assert error < 1e-8

    def test_1d_random_rhs(self):
        N = 30
        spec = laplacian_stencil_spec(1)
        halo = 1
        np.random.seed(99)

        b = np.zeros(N + 2 * halo)
        b[halo:-halo] = np.random.randn(N)

        x_sol, iters, residuals = stencil_cg_solve(spec, b, tol=1e-10, max_iter=500)
        assert residuals[-1] < 1e-8

        # Verify: A @ x ≈ b on interior
        Ax = apply_stencil(x_sol, spec)
        np.testing.assert_allclose(Ax[halo:-halo], b[halo:-halo], atol=1e-7)


class TestStencilCGRef2D:
    """NumPy reference stencil CG converges for 2D Laplacian."""

    def test_2d_random_rhs(self):
        Nx, Ny = 8, 8
        spec = laplacian_stencil_spec(2)
        halo = (1, 1)
        np.random.seed(42)

        b = np.zeros((Nx + 2, Ny + 2))
        b[1:-1, 1:-1] = np.random.randn(Nx, Ny)

        x_sol, iters, residuals = stencil_cg_solve(spec, b, tol=1e-8, max_iter=500)
        assert residuals[-1] < 1e-6

        Ax = apply_stencil(x_sol, spec)
        np.testing.assert_allclose(Ax[1:-1, 1:-1], b[1:-1, 1:-1], atol=1e-5)


class TestStencilCGMatchesDIACG:
    """Matrix-free stencil CG gives same solution as DIA-based CG."""

    def test_1d_matches_dia_cg(self):
        N = 30
        halo = 1
        spec = laplacian_stencil_spec(1)
        np.random.seed(7)

        # Build RHS
        b_interior = np.random.randn(N)
        b_full = np.zeros(N + 2 * halo)
        b_full[halo:-halo] = b_interior

        # DIA-based CG
        A = laplacian_1d_dia(N)
        A_fn = lambda v: dia_spmv(A, v)
        x_dia, _, _ = cg_solve(A_fn, b_interior, tol=1e-10, max_iter=500)

        # Stencil-based CG
        x_stencil, _, _ = stencil_cg_solve(spec, b_full, tol=1e-10, max_iter=500)

        # Compare interior solutions
        np.testing.assert_allclose(
            x_stencil[halo:-halo], x_dia, atol=1e-7,
            err_msg="Stencil CG and DIA CG solutions differ",
        )
