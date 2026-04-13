"""2D acoustic wave equation with 4th-order stencil (new cuTile API).

Second-order wave equation: d^2u/dt^2 = c^2 nabla^2 u

4th-order spatial discretisation of the Laplacian in 2D uses offsets +/-1, +/-2
in each dimension. Explicit leapfrog time-stepping.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import apply_stencil


# -- 4th-order 2D Laplacian stencil ----------------------------------------

@stencil
def wave_2d(u, i, j):
    return 0.1 * ((-1/12) * u[i - 2, j] + (4/3) * u[i - 1, j] + (-5/2) * u[i, j] + (4/3) * u[i + 1, j] + (-1/12) * u[i + 2, j] + (-1/12) * u[i, j - 2] + (4/3) * u[i, j - 1] + (-5/2) * u[i, j] + (4/3) * u[i, j + 1] + (-1/12) * u[i, j + 2])


def main():
    print("=" * 60)
    print("2D Acoustic Wave (4th-order, new cuTile API)")
    print("=" * 60)

    # -- Compile ------------------------------------------------------------
    result = compile(wave_2d)
    code = result.code

    print(f"\nCompilation summary:")
    print(f"  ndim:           {result.ndim}")
    print(f"  halo_widths:    {result.halo_widths}")
    print(f"  tile_sizes:     {result.tile_sizes}")
    print(f"  temporal_steps: {result.temporal_steps}")
    if result.analysis:
        for k, v in result.analysis.items():
            print(f"  {k}: {v}")

    # -- Validate generated code is valid Python ----------------------------
    ast.parse(code)
    print("\n  [OK] Generated code is valid Python syntax")

    # -- NumPy reference simulation (leapfrog) ------------------------------
    halo = result.halo_widths
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
        lap = apply_stencil(u, wave_2d._fn, ndim=2, halo_widths=halo)
        u_new = 2 * u - u_old + lap
        u_new[0, :] = 0; u_new[-1, :] = 0
        u_new[:, 0] = 0; u_new[:, -1] = 0
        u_old = u
        u = u_new

    print(f"  Max amplitude: {np.max(np.abs(u)):.6f}")
    print("  [OK] Simulation completed")


if __name__ == "__main__":
    main()
