"""Poisson equation solved with CG using NumPy reference solver.

Solves: -∇²u = f on [0,1] with u=0 on boundary.
Discretised as A·u = f where A is the 1D tridiagonal Laplacian.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil.solvers.formats import laplacian_1d_dia, laplacian_2d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv
from cutile_stencil.reference.cg_ref import cg_solve
from cutile_stencil.solvers.kernels import generate_dia_spmv, generate_dot, generate_axpy


def main():
    # ── 1D Poisson ──────────────────────────────────────────────────
    print("=" * 60)
    print("1D Poisson equation: -u'' = f, u(0)=u(1)=0")
    print("=" * 60)

    N = 100
    h = 1.0 / (N + 1)
    A = laplacian_1d_dia(N)

    # RHS: f(x) = sin(πx) → exact solution u(x) = sin(πx) / π²
    x = np.linspace(h, 1 - h, N)
    f = np.sin(np.pi * x) * h**2  # scaled by h²

    A_fn = lambda v: dia_spmv(A, v)
    x_sol, iters, residuals = cg_solve(A_fn, f, tol=1e-10, max_iter=500)

    exact = np.sin(np.pi * x) / (np.pi**2)
    error = np.max(np.abs(x_sol - exact))

    print(f"  Grid points: {N}")
    print(f"  CG converged in {iters} iterations")
    print(f"  Final residual: {residuals[-1]:.2e}")
    print(f"  Max error vs exact: {error:.2e}")
    print(f"  ✓ Solution verified" if error < 1e-3 else "  ✗ Error too large")

    # ── 2D Poisson ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("2D Poisson equation on 20×20 grid")
    print("=" * 60)

    Nx, Ny = 20, 20
    h2 = 1.0 / (Nx + 1)
    A2 = laplacian_2d_dia(Nx, Ny)
    N2 = Nx * Ny

    # RHS: constant source
    f2 = np.ones(N2) * h2**2

    A2_fn = lambda v: dia_spmv(A2, v)
    x2_sol, iters2, res2 = cg_solve(A2_fn, f2, tol=1e-10, max_iter=2000)

    print(f"  Grid: {Nx}×{Ny} = {N2} unknowns")
    print(f"  CG converged in {iters2} iterations")
    print(f"  Final residual: {res2[-1]:.2e}")

    # Verify A @ x ≈ f
    Ax = dia_spmv(A2, x2_sol)
    rel_error = np.linalg.norm(Ax - f2) / np.linalg.norm(f2)
    print(f"  Relative residual: {rel_error:.2e}")
    print(f"  ✓ Solution verified" if rel_error < 1e-8 else "  ✗ Error too large")

    # ── Generate cuTile kernels ─────────────────────────────────────
    print()
    print("=" * 60)
    print("Generated cuTile solver kernels")
    print("=" * 60)

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    import ast
    for name, gen_fn in [("dia_spmv", generate_dia_spmv),
                          ("dot", generate_dot),
                          ("axpy", generate_axpy)]:
        code = gen_fn()
        path = os.path.join(gen_dir, f"{name}_kernel.py")
        with open(path, "w") as fh:
            fh.write(code)
        ast.parse(code)
        print(f"  {path} ✓")


if __name__ == "__main__":
    main()
