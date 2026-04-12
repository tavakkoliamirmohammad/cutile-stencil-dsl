"""1D heat equation ported to the new cuTile compiler API.

The 1D heat equation with explicit Euler:
    u^{n+1}_i = u^n_i + a*dt/dx^2 * (u^n_{i-1} - 2*u^n_i + u^n_{i+1})

With a*dt/dx^2 = 0.25 (stable):
    u^{n+1}_i = 0.25*u^n_{i-1} + 0.5*u^n_i + 0.25*u^n_{i+1}

Gaussian initial condition, energy dissipation check.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import apply_stencil, time_march


# -- Define stencil --------------------------------------------------------

@stencil(ndim=1, order=1, dtype="float64")
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


def main():
    print("=" * 60)
    print("1D Heat Equation (new cuTile API)")
    print("=" * 60)

    # -- Compile ------------------------------------------------------------
    result = compile(heat_1d)
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

    # -- NumPy reference simulation -----------------------------------------
    halo = result.halo_widths
    N = 128
    h = halo[0]
    u0 = np.zeros(N + 2 * h)
    x = np.linspace(-1, 1, N + 2 * h)
    u0[:] = np.exp(-20 * x**2)
    u0[0] = 0.0
    u0[-1] = 0.0

    steps = 100
    history = time_march(u0, heat_1d._fn, ndim=1, halo_widths=halo, steps=steps)
    print(f"\nNumPy simulation: {steps} steps, N={N}")
    print(f"  Initial energy: {np.sum(u0**2):.6f}")
    print(f"  Final energy:   {np.sum(history[-1]**2):.6f}")
    assert np.sum(history[-1]**2) < np.sum(u0**2), "Energy should decrease"
    print("  [OK] Energy dissipation verified")


if __name__ == "__main__":
    main()
