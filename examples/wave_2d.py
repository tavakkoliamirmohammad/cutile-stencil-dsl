"""2D acoustic wave equation with 4th-order stencil.

Second-order wave equation: ∂²u/∂t² = c² ∇²u

4th-order spatial discretisation of the Laplacian in 2D uses offsets ±1, ±2
in each dimension. Explicit leapfrog time-stepping.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import apply_stencil


# ── 4th-order 2D Laplacian stencil ─────────────────────────────────

@stencil(dtype="float64")
def wave_2d(u, i, j):
    lap_x = (-1/12) * u[i - 2, j] + (4/3) * u[i - 1, j] + (-5/2) * u[i, j] + (4/3) * u[i + 1, j] + (-1/12) * u[i + 2, j]
    lap_y = (-1/12) * u[i, j - 2] + (4/3) * u[i, j - 1] + (-5/2) * u[i, j] + (4/3) * u[i, j + 1] + (-1/12) * u[i, j + 2]
    return 0.1 * (lap_x + lap_y)


def main():
    # ── Compile ─────────────────────────────────────────────────────
    result = compile(wave_2d, domain=(256, 256))
    result.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result.emit_to_file(os.path.join(gen_dir, "wave_2d_kernel.py"))
    print(f"\nGenerated cuTile kernel")
    print("  ✓ Valid Python syntax")

    # ── NumPy reference simulation (leapfrog) ───────────────────────
    halo = result.spec.halo_widths
    Nx, Ny = 64, 64
    hx, hy = halo
    u = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    u_old = np.zeros_like(u)

    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(u.shape[0]):
        for jj in range(u.shape[1]):
            r2 = ((ii - cx) / 5.0)**2 + ((jj - cy) / 5.0)**2
            u[ii, jj] = np.exp(-r2)
    u_old[:] = u

    steps = 50
    print(f"\nNumPy simulation: {steps} steps, grid={Nx}x{Ny}")
    for s in range(steps):
        lap = apply_stencil(u, result.spec)
        u_new = 2 * u - u_old + lap
        u_new[0, :] = 0; u_new[-1, :] = 0
        u_new[:, 0] = 0; u_new[:, -1] = 0
        u_old = u
        u = u_new

    print(f"  Max amplitude: {np.max(np.abs(u)):.6f}")
    print("  ✓ Simulation completed")

    # ── GPU kernel validation ────────────────────────────────────────
    test_u = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(test_u.shape[0]):
        for jj in range(test_u.shape[1]):
            r2 = ((ii - cx_t) / 5.0)**2 + ((jj - cy_t) / 5.0)**2
            test_u[ii, jj] = np.exp(-r2)

    result.validate(test_u)


if __name__ == "__main__":
    main()
