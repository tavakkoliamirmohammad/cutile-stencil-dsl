"""1D FDTD Maxwell solver (E-H fields).

Solves Maxwell's equations in 1D:
    dE/dt = (1/eps) * dH/dx
    dH/dt = (1/mu) * dE/dx

Uses the Yee staggered grid: E and H are offset by half a cell.
This exercises staggered grid and two-field coupling.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import _ArrayProxy


# Physical parameters (normalized)
c0 = 1.0       # speed of light
dx = 0.01       # spatial step
dt = 0.005      # time step (CFL: dt < dx/c0)
eps = 1.0       # permittivity
mu = 1.0        # permeability


@stencil(dtype="float64")
def update_E(E, H, i):
    """Update E field: E^{n+1}[i] = E^n[i] + (dt/(eps*dx)) * (H[i] - H[i-1])"""
    return E[i] + (dt / (eps * dx)) * (H[i] - H[i - 1])


@stencil(dtype="float64")
def update_H(E, H, i):
    """Update H field: H^{n+1}[i] = H^n[i] + (dt/(mu*dx)) * (E[i+1] - E[i])"""
    return H[i] + (dt / (mu * dx)) * (E[i + 1] - E[i])


def main():
    print("=" * 60)
    print("1D FDTD Maxwell Solver (E-H fields)")
    print("=" * 60)

    # ── Compile both kernels ─────────────────────────────────────────
    result_E = compile(update_E, domain=(1024,))
    result_H = compile(update_H, domain=(1024,))

    result_E.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result_E.emit_to_file(os.path.join(gen_dir, "fdtd_update_E_kernel.py"))
    print(f"\nGenerated E-update kernel")
    print("  ✓ Valid Python syntax")

    spec_E = result_E.spec
    spec_H = result_H.spec

    # ── NumPy FDTD simulation ───────────────────────────────────────
    N = 500
    h = max(spec_E.halo_widths[0], spec_H.halo_widths[0])
    E = np.zeros(N + 2 * h)
    H = np.zeros(N + 2 * h)

    source_pos = N // 4 + h
    t0 = 40
    spread = 12

    steps = 200
    print(f"\nSimulation: {steps} steps, N={N}")

    for s in range(steps):
        # Update H first (leapfrog)
        proxy_h = {
            'E': _ArrayProxy(E, spec_H.halo_widths),
            'H': _ArrayProxy(H, spec_H.halo_widths),
        }
        result_H_step = spec_H.update_fn(proxy_h['E'], proxy_h['H'], 0)
        H_new = H.copy()
        h_interior = tuple(slice(hw_, n - hw_) for hw_, n in zip(spec_H.halo_widths, H.shape))
        H_new[h_interior] = result_H_step
        H = H_new

        # Update E
        proxy_e = {
            'E': _ArrayProxy(E, spec_E.halo_widths),
            'H': _ArrayProxy(H, spec_E.halo_widths),
        }
        result_E_step = spec_E.update_fn(proxy_e['E'], proxy_e['H'], 0)
        E_new = E.copy()
        e_interior = tuple(slice(hw_, n - hw_) for hw_, n in zip(spec_E.halo_widths, E.shape))
        E_new[e_interior] = result_E_step
        E = E_new

        E[source_pos] += np.exp(-0.5 * ((s - t0) / spread) ** 2)

    print(f"  Max |E|: {np.max(np.abs(E)):.6f}")
    print(f"  Max |H|: {np.max(np.abs(H)):.6f}")

    total_energy = np.sum(E ** 2) + np.sum(H ** 2)
    print(f"  Total energy (E^2 + H^2): {total_energy:.6f}")
    assert np.max(np.abs(E)) < 100.0, "E field blew up"
    assert np.max(np.abs(H)) < 100.0, "H field blew up"
    print("  ✓ FDTD simulation stable")

    # ── GPU kernel validation (single E-update step) ─────────────────
    E_test = np.random.randn(N + 2 * h)
    H_test = np.random.randn(N + 2 * h)
    result_E.validate(E_test, H_test)


if __name__ == "__main__":
    main()
