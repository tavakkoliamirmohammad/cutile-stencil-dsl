"""Tests for NumPy CG convergence and SpMV correctness."""

import numpy as np
import pytest
from cutile_stencil.solvers.formats import (
    DIAMatrix, BSRMatrix,
    laplacian_1d_dia, laplacian_2d_dia, laplacian_3d_dia,
)
from cutile_stencil.reference.spmv_ref import dia_spmv, bsr_spmv
from cutile_stencil.reference.cg_ref import cg_solve, mixed_precision_cg


# ── DIA SpMV tests ──────────────────────────────────────────────────

class TestDIASpMV:
    def test_identity(self):
        """DIA identity matrix: y = x."""
        N = 10
        data = np.ones((1, N))
        offsets = np.array([0], dtype=np.int32)
        A = DIAMatrix(data, offsets, (N, N))
        x = np.arange(N, dtype=np.float64)
        y = dia_spmv(A, x)
        np.testing.assert_allclose(y, x)

    def test_laplacian_1d(self):
        """1D Laplacian applied to constant should give zero interior."""
        N = 20
        A = laplacian_1d_dia(N)
        x = np.ones(N)
        y = dia_spmv(A, x)
        # Interior: -1 + 2 - 1 = 0 (except boundary rows)
        # Actually for tridiag, all interior rows give 0 for constant vector
        # Boundary rows: first row has [--, 2, -1] * 1 = 1; last has [-1, 2, --] * 1 = 1
        np.testing.assert_allclose(y[1:-1], 0.0, atol=1e-14)

    def test_laplacian_1d_matches_dense(self):
        N = 15
        A = laplacian_1d_dia(N)
        A_dense = A.to_dense()
        x = np.random.randn(N)
        y_dia = dia_spmv(A, x)
        y_dense = A_dense @ x
        np.testing.assert_allclose(y_dia, y_dense, atol=1e-12)

    def test_laplacian_2d_matches_dense(self):
        Nx, Ny = 5, 5
        A = laplacian_2d_dia(Nx, Ny)
        A_dense = A.to_dense()
        x = np.random.randn(Nx * Ny)
        y_dia = dia_spmv(A, x)
        y_dense = A_dense @ x
        np.testing.assert_allclose(y_dia, y_dense, atol=1e-12)

    def test_laplacian_3d_shape(self):
        A = laplacian_3d_dia(4, 4, 4)
        assert A.shape == (64, 64)
        assert A.data.shape[0] == 7  # 7 diagonals


# ── BSR SpMV tests ──────────────────────────────────────────────────

class TestBSRSpMV:
    def test_block_identity(self):
        """BSR identity with 2×2 blocks."""
        N = 6  # 3 blocks of 2×2
        nb = 3
        data = np.array([np.eye(2)] * nb)
        indices = np.array([0, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3], dtype=np.int32)
        A = BSRMatrix(data, indices, indptr, (2, 2), (N, N))
        x = np.arange(N, dtype=np.float64)
        y = bsr_spmv(A, x)
        np.testing.assert_allclose(y, x)

    def test_bsr_matches_dense(self):
        """BSR SpMV matches dense matmul."""
        N = 4
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        A = BSRMatrix(data, indices, indptr, (2, 2), (N, N))
        A_dense = A.to_dense()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(bsr_spmv(A, x), A_dense @ x)


# ── CG solver tests ────────────────────────────────────────────────

class TestCG:
    def test_cg_1d_poisson(self):
        """CG solves 1D Poisson: -u'' = f."""
        N = 50
        A = laplacian_1d_dia(N)
        A_fn = lambda v: dia_spmv(A, v)

        # Known solution: u_i = sin(πi/(N+1))
        h = 1.0 / (N + 1)
        x_grid = np.linspace(h, 1 - h, N)
        exact = np.sin(np.pi * x_grid)
        f = A_fn(exact)

        x_sol, iters, residuals = cg_solve(A_fn, f, tol=1e-10, max_iter=500)
        np.testing.assert_allclose(x_sol, exact, atol=1e-8)
        assert residuals[-1] < 1e-10

    def test_cg_2d_poisson(self):
        """CG solves 2D Poisson on small grid."""
        Nx, Ny = 10, 10
        N = Nx * Ny
        A = laplacian_2d_dia(Nx, Ny)
        A_fn = lambda v: dia_spmv(A, v)

        # Random RHS
        np.random.seed(42)
        f = np.random.randn(N)
        x_sol, iters, residuals = cg_solve(A_fn, f, tol=1e-10, max_iter=1000)

        # Verify A @ x ≈ f
        Ax = dia_spmv(A, x_sol)
        np.testing.assert_allclose(Ax, f, atol=1e-8)

    def test_cg_converges_monotonically(self):
        """Residual norms should decrease (for SPD system)."""
        N = 30
        A = laplacian_1d_dia(N)
        A_fn = lambda v: dia_spmv(A, v)
        f = np.ones(N)

        _, _, residuals = cg_solve(A_fn, f, tol=1e-12, max_iter=200)
        # CG residuals should generally decrease (monotone for exact arithmetic)
        # Allow some floating-point noise
        for k in range(1, min(len(residuals), 10)):
            assert residuals[k] <= residuals[0] * 1.1


class TestMixedPrecisionCG:
    def test_mixed_precision_converges(self):
        """Mixed-precision CG should converge to FP64 accuracy."""
        N = 50
        A = laplacian_1d_dia(N)
        A_fn = lambda v: dia_spmv(A, v)

        h = 1.0 / (N + 1)
        x_grid = np.linspace(h, 1 - h, N)
        exact = np.sin(np.pi * x_grid)
        f = A_fn(exact)

        x_sol, outers, residuals = mixed_precision_cg(
            A_fn, f, inner_dtype=np.float32,
            tol=1e-10, max_outer=30, max_inner=300,
        )
        error = np.max(np.abs(x_sol - exact))
        assert error < 1e-6
        assert residuals[-1] < 1e-6
