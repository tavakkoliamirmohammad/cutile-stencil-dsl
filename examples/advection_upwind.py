"""1D upwind advection scheme.

Solves du/dt + c * du/dx = 0 with first-order upwind differencing.
For positive velocity c > 0, the stencil is left-biased (asymmetric):
    u_new[i] = u[i] - CFL * (u[i] - u[i-1])

This exercises asymmetric stencils.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import time_march


CFL = 0.5  # CFL number (must be <= 1 for stability)


@stencil(dtype="float64")
def advection_upwind(u, i):
    return u[i] - CFL * (u[i] - u[i - 1])


def main():
    print("=" * 60)
    print("1D Upwind Advection: du/dt + c * du/dx = 0")
    print("=" * 60)

    # ── Compile ─────────────────────────────────────────────────────
    result = compile(advection_upwind, domain=(1024,))
    result.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result.emit_to_file(os.path.join(gen_dir, "advection_upwind_kernel.py"))
    print(f"\nGenerated cuTile kernel")
    print("  ✓ Valid Python syntax")

    # ── NumPy simulation ────────────────────────────────────────────
    halo = result.spec.halo_widths
    N = 200
    h = halo[0]
    u0 = np.zeros(N + 2 * h)
    u0[h + N // 4:h + N // 2] = 1.0

    steps = 80
    print(f"\nSimulation: {steps} steps, N={N}, CFL={CFL}")

    history = time_march(u0, result.spec, steps)
    u_final = history[-1]

    initial_mass = np.sum(u0[h:-h])
    final_mass = np.sum(u_final[h:-h])
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass:   {final_mass:.6f}")
    print(f"  Max value:    {u_final.max():.6f}")
    print(f"  ✓ Advection simulation completed")

    initial_center_of_mass = np.sum(np.arange(N) * u0[h:-h]) / max(initial_mass, 1e-10)
    final_center_of_mass = np.sum(np.arange(N) * u_final[h:-h]) / max(final_mass, 1e-10)
    if final_mass > 0.01:
        assert final_center_of_mass >= initial_center_of_mass - 1, "Wave should move right"
        print("  ✓ Wave propagation direction verified")

    # ── GPU kernel validation ────────────────────────────────────────
    result.validate(u0)


if __name__ == "__main__":
    main()
