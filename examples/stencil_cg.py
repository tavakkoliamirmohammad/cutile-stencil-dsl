#!/usr/bin/env python3
"""Matrix-free CG solver using the stencil pipeline — end-to-end demo.

Demonstrates:
1. Defining the Laplacian as a stencil (via the bridge)
2. Generating a matrix-free CG solver
3. Solving 1D Poisson with the NumPy reference CG
4. Running the solver analysis pipeline
"""

import numpy as np

from cutile_stencil.solvers.stencil_bridge import laplacian_stencil_spec
from cutile_stencil.solvers.stencil_cg import generate_stencil_cg
from cutile_stencil.reference.stencil_cg_ref import stencil_cg_solve
from cutile_stencil.reference.stencil_ref import apply_stencil
from cutile_stencil.solvers.solver_analysis import (
    analyze_stencil_operator,
    analyze_cg_iteration,
    print_solver_analysis,
)
from cutile_stencil.dsl.types import HardwareSpec


def main():
    # ── 1. Define Laplacian stencil ──────────────────────────────
    print("=" * 60)
    print("Matrix-Free CG with Stencil Operator")
    print("=" * 60)

    spec = laplacian_stencil_spec(1)
    print(f"\nStencil: {spec.name}")
    print(f"  ndim={spec.ndim}, order={spec.order}, halo={spec.halo_widths}")
    print(f"  accesses: {[(a.array_name, a.offsets) for a in spec.accesses]}")

    # ── 2. Generate matrix-free CG code ──────────────────────────
    N = 256
    domain = (N,)
    code = generate_stencil_cg(spec, domain)
    print(f"\nGenerated matrix-free CG solver ({len(code)} chars)")
    print(f"  First 3 lines: {code.splitlines()[:3]}")

    # ── 3. Solve 1D Poisson with NumPy reference ─────────────────
    print("\n" + "-" * 60)
    print("Solving 1D Poisson: A @ u = f")
    print("-" * 60)

    halo = 1
    h = 1.0 / (N + 1)
    x_grid = np.linspace(h, 1 - h, N)

    # Known solution
    u_exact = np.zeros(N + 2 * halo)
    u_exact[halo:-halo] = np.sin(np.pi * x_grid)

    # Compute RHS: b = A @ u_exact
    Au = apply_stencil(u_exact, spec)
    b = np.zeros(N + 2 * halo)
    b[halo:-halo] = Au[halo:-halo]

    # Solve
    x_sol, iters, residuals = stencil_cg_solve(
        spec, b, tol=1e-10, max_iter=500,
    )

    error = np.max(np.abs(x_sol[halo:-halo] - u_exact[halo:-halo]))
    print(f"  Converged in {iters} iterations")
    print(f"  Final residual: {residuals[-1]:.2e}")
    print(f"  Max error vs exact: {error:.2e}")

    # ── 4. Solver analysis ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("Solver Analysis (A100 80GB)")
    print("-" * 60)

    hw = HardwareSpec.from_preset("A100_80GB")
    op_profile = analyze_stencil_operator(spec, domain, hw)
    result = analyze_cg_iteration(op_profile, N, hw, spec.dtype)
    print()
    print_solver_analysis(result)


if __name__ == "__main__":
    main()
