"""1D FDTD Maxwell solver -- E-H fields (new cuTile API).

Solves Maxwell's equations in 1D:
    dE/dt = (1/eps) * dH/dx
    dH/dt = (1/mu)  * dE/dx

Uses the Yee staggered grid: E and H are offset by half a cell.
Separate stencils for each field update (two-field coupling).
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import _ArrayProxy


# Physical parameters (normalized)
c0 = 1.0       # speed of light
dx = 0.01       # spatial step
dt = 0.005      # time step (CFL: dt < dx/c0)
eps = 1.0       # permittivity
mu = 1.0        # permeability


@stencil
def update_E(E, H, i):
    """Update E field: E^{n+1}[i] = E^n[i] + (dt/(eps*dx)) * (H[i] - H[i-1])"""
    return E[i] + (dt / (eps * dx)) * (H[i] - H[i - 1])


@stencil
def update_H(E, H, i):
    """Update H field: H^{n+1}[i] = H^n[i] + (dt/(mu*dx)) * (E[i+1] - E[i])"""
    return H[i] + (dt / (mu * dx)) * (E[i + 1] - E[i])


def main():
    print("=" * 60)
    print("1D FDTD Maxwell Solver -- E-H fields (new cuTile API)")
    print("=" * 60)

    # -- Compile both kernels -----------------------------------------------
    result_E = compile(update_E)
    result_H = compile(update_H)

    print(f"\nCompilation summary (E-kernel):")
    print(f"  ndim:           {result_E.ndim}")
    print(f"  halo_widths:    {result_E.halo_widths}")
    print(f"  tile_sizes:     {result_E.tile_sizes}")
    print(f"  temporal_steps: {result_E.temporal_steps}")
    if result_E.analysis:
        for key, val in result_E.analysis.items():
            print(f"  {key}: {val}")

    # -- Validate generated code is valid Python ----------------------------
    ast.parse(result_E.code)
    ast.parse(result_H.code)
    print("\n  [OK] Generated E-update and H-update kernel code is valid Python syntax")

    halo_E = result_E.halo_widths
    halo_H = result_H.halo_widths

    # -- NumPy FDTD simulation ----------------------------------------------
    N = 500
    h = max(halo_E[0], halo_H[0])
    E = np.zeros(N + 2 * h)
    H = np.zeros(N + 2 * h)

    source_pos = N // 4 + h
    t0 = 40
    spread = 12

    steps = 200
    print(f"\nSimulation: {steps} steps, N={N}")

    for s in range(steps):
        # Update H first (leapfrog)
        proxy_E_h = _ArrayProxy(E, halo_H)
        proxy_H_h = _ArrayProxy(H, halo_H)
        result_H_step = update_H._fn(proxy_E_h, proxy_H_h, 0)
        H_new = H.copy()
        h_interior = tuple(
            slice(hw_, n - hw_) for hw_, n in zip(halo_H, H.shape)
        )
        H_new[h_interior] = result_H_step
        H = H_new

        # Update E
        proxy_E_e = _ArrayProxy(E, halo_E)
        proxy_H_e = _ArrayProxy(H, halo_E)
        result_E_step = update_E._fn(proxy_E_e, proxy_H_e, 0)
        E_new = E.copy()
        e_interior = tuple(
            slice(hw_, n - hw_) for hw_, n in zip(halo_E, E.shape)
        )
        E_new[e_interior] = result_E_step
        E = E_new

        E[source_pos] += np.exp(-0.5 * ((s - t0) / spread) ** 2)

    print(f"  Max |E|: {np.max(np.abs(E)):.6f}")
    print(f"  Max |H|: {np.max(np.abs(H)):.6f}")

    total_energy = np.sum(E ** 2) + np.sum(H ** 2)
    print(f"  Total energy (E^2 + H^2): {total_energy:.6f}")
    assert np.max(np.abs(E)) < 100.0, "E field blew up"
    assert np.max(np.abs(H)) < 100.0, "H field blew up"
    print("  [OK] FDTD simulation stable")


if __name__ == "__main__":
    main()
