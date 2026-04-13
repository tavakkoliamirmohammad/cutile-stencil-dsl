"""1D upwind advection scheme (new cuTile API).

Solves du/dt + c * du/dx = 0 with first-order upwind differencing.
For positive velocity c > 0, the stencil is left-biased (asymmetric):
    u_new[i] = u[i] - CFL * (u[i] - u[i-1])

This exercises asymmetric stencils in the new compiler.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import time_march


CFL = 0.5  # CFL number (must be <= 1 for stability)


@stencil
def advection_upwind(u, i):
    return u[i] - CFL * (u[i] - u[i - 1])


def main():
    print("=" * 60)
    print("1D Upwind Advection: du/dt + c * du/dx = 0 (new cuTile API)")
    print("=" * 60)

    # -- Compile ------------------------------------------------------------
    result = compile(advection_upwind)
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

    # -- NumPy simulation ---------------------------------------------------
    halo = result.halo_widths
    N = 200
    h = halo[0]
    u0 = np.zeros(N + 2 * h)
    u0[h + N // 4:h + N // 2] = 1.0

    steps = 80
    print(f"\nSimulation: {steps} steps, N={N}, CFL={CFL}")

    history = time_march(
        u0, advection_upwind._fn, ndim=1, halo_widths=halo, steps=steps,
    )
    u_final = history[-1]

    initial_mass = np.sum(u0[h:-h])
    final_mass = np.sum(u_final[h:-h])
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass:   {final_mass:.6f}")
    print(f"  Max value:    {u_final.max():.6f}")
    print("  [OK] Advection simulation completed")

    initial_center_of_mass = np.sum(
        np.arange(N) * u0[h:-h]
    ) / max(initial_mass, 1e-10)
    final_center_of_mass = np.sum(
        np.arange(N) * u_final[h:-h]
    ) / max(final_mass, 1e-10)
    if final_mass > 0.01:
        assert final_center_of_mass >= initial_center_of_mass - 1, \
            "Wave should move right"
        print("  [OK] Wave propagation direction verified")


if __name__ == "__main__":
    main()
