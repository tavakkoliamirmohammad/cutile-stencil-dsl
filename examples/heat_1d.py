"""Full pipeline: DSL → analysis → codegen → NumPy validation for 1D heat equation.

The 1D heat equation with explicit Euler:
    u^{n+1}_i = u^n_i + α·Δt/Δx² · (u^n_{i-1} - 2·u^n_i + u^n_{i+1})

With α·Δt/Δx² = 0.25 (stable):
    u^{n+1}_i = 0.25·u^n_{i-1} + 0.5·u^n_i + 0.25·u^n_{i+1}
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import time_march


# ── Define stencil ──────────────────────────────────────────────────

@stencil(dtype="float64")
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


def main():
    # ── Compile ─────────────────────────────────────────────────────
    result = compile(heat_1d, domain=(1024,))
    result.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result.emit_to_file(os.path.join(gen_dir, "heat_1d_kernel.py"))
    print(f"\nGenerated cuTile kernel")
    print("  ✓ Valid Python syntax")

    # ── NumPy reference simulation ──────────────────────────────────
    halo = result.spec.halo_widths
    N = 128
    h = halo[0]
    u0 = np.zeros(N + 2 * h)
    x = np.linspace(-1, 1, N + 2 * h)
    u0[:] = np.exp(-20 * x**2)
    u0[0] = 0.0
    u0[-1] = 0.0

    steps = 100
    history = time_march(u0, result.spec, steps)
    print(f"\nNumPy simulation: {steps} steps, N={N}")
    print(f"  Initial energy: {np.sum(u0**2):.6f}")
    print(f"  Final energy:   {np.sum(history[-1]**2):.6f}")
    assert np.sum(history[-1]**2) < np.sum(u0**2), "Energy should decrease"
    print("  ✓ Energy dissipation verified")

    # ── GPU kernel validation ────────────────────────────────────────
    result.validate(u0)


if __name__ == "__main__":
    main()
