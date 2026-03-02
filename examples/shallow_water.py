"""2D Shallow Water Equations (linearized).

Solves the linearized shallow water system:
    dh/dt  = -H0 * (d(hu)/dx + d(hv)/dy)
    d(hu)/dt = -g * dh/dx
    d(hv)/dt = -g * dh/dy

Uses central differences. Three coupled fields (h, hu, hv).
This exercises 3 coupled fields and flux stencils.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import _ArrayProxy


# Physical parameters
g = 9.81      # gravity
H0 = 1.0      # mean depth
dt = 0.01
dx = 0.1


@stencil(dtype="float64")
def shallow_water_h(h, hu, hv, i, j):
    """Update height field h."""
    dhu_dx = (hu[i + 1, j] - hu[i - 1, j]) / (2 * dx)
    dhv_dy = (hv[i, j + 1] - hv[i, j - 1]) / (2 * dx)
    return h[i, j] - dt * H0 * (dhu_dx + dhv_dy)


@stencil(dtype="float64")
def shallow_water_hu(h, hu, i, j):
    """Update x-momentum field hu."""
    dh_dx = (h[i + 1, j] - h[i - 1, j]) / (2 * dx)
    return hu[i, j] - dt * g * dh_dx


@stencil(dtype="float64")
def shallow_water_hv(h, hv, i, j):
    """Update y-momentum field hv."""
    dh_dy = (h[i, j + 1] - h[i, j - 1]) / (2 * dx)
    return hv[i, j] - dt * g * dh_dy


def main():
    print("=" * 60)
    print("2D Shallow Water Equations (Linearized)")
    print("=" * 60)

    # ── Compile all three kernels ────────────────────────────────────
    result_h = compile(shallow_water_h, domain=(64, 64))
    result_hu = compile(shallow_water_hu, domain=(64, 64))
    result_hv = compile(shallow_water_hv, domain=(64, 64))

    result_h.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result_h.emit_to_file(os.path.join(gen_dir, "shallow_water_h_kernel.py"))
    result_hu.emit_to_file(os.path.join(gen_dir, "shallow_water_hu_kernel.py"))
    result_hv.emit_to_file(os.path.join(gen_dir, "shallow_water_hv_kernel.py"))
    print(f"\nGenerated h, hu, hv kernels")
    print("  ✓ Valid Python syntax")

    spec_h = result_h.spec
    spec_hu = result_hu.spec
    spec_hv = result_hv.spec

    # Use consistent halo (max across all fields)
    max_halo = tuple(
        max(spec_h.halo_widths[d], spec_hu.halo_widths[d], spec_hv.halo_widths[d])
        for d in range(2)
    )
    for spec in [spec_h, spec_hu, spec_hv]:
        spec.halo_widths = max_halo
    print(f"Unified halo: {max_halo}")

    # ── NumPy simulation ────────────────────────────────────────────
    Nx, Ny = 32, 32
    hx, hy = max_halo

    h = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(h.shape[0]):
        for jj in range(h.shape[1]):
            r2 = ((ii - cx) / 3.0) ** 2 + ((jj - cy) / 3.0) ** 2
            h[ii, jj] = 0.1 * np.exp(-r2)

    initial_energy = np.sum(h[hx:-hx, hy:-hy] ** 2)
    steps = 50
    print(f"\nSimulation: {steps} steps, grid={Nx}x{Ny}")

    for s in range(steps):
        proxy = {
            'h': _ArrayProxy(h, spec_h.halo_widths),
            'hu': _ArrayProxy(hu, spec_h.halo_widths),
            'hv': _ArrayProxy(hv, spec_h.halo_widths),
        }
        result_h_step = spec_h.update_fn(proxy['h'], proxy['hu'], proxy['hv'], 0, 0)
        h_new = h.copy()
        interior = tuple(slice(hw_, s_ - hw_) for hw_, s_ in zip(spec_h.halo_widths, h.shape))
        h_new[interior] = result_h_step

        proxy2 = {
            'h': _ArrayProxy(h, spec_hu.halo_widths),
            'hu': _ArrayProxy(hu, spec_hu.halo_widths),
        }
        result_hu_step = spec_hu.update_fn(proxy2['h'], proxy2['hu'], 0, 0)
        hu_new = hu.copy()
        hu_new[interior] = result_hu_step

        proxy3 = {
            'h': _ArrayProxy(h, spec_hv.halo_widths),
            'hv': _ArrayProxy(hv, spec_hv.halo_widths),
        }
        result_hv_step = spec_hv.update_fn(proxy3['h'], proxy3['hv'], 0, 0)
        hv_new = hv.copy()
        hv_new[interior] = result_hv_step

        h, hu, hv = h_new, hu_new, hv_new

    final_energy = np.sum(h[hx:-hx, hy:-hy] ** 2)
    print(f"  Initial energy (h^2): {initial_energy:.6f}")
    print(f"  Final energy (h^2):   {final_energy:.6f}")
    print(f"  Max |h|: {np.max(np.abs(h)):.6f}")
    print(f"  Max |hu|: {np.max(np.abs(hu)):.6f}")
    print(f"  Max |hv|: {np.max(np.abs(hv)):.6f}")

    assert np.max(np.abs(h)) < 10.0, "h blew up"
    print("  ✓ Simulation stable")

    # ── GPU kernel validation ────────────────────────────────────────
    h_test = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    hu_test = np.zeros_like(h_test)
    hv_test = np.zeros_like(h_test)
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(h_test.shape[0]):
        for jj in range(h_test.shape[1]):
            r2 = ((ii - cx_t) / 3.0)**2 + ((jj - cy_t) / 3.0)**2
            h_test[ii, jj] = 0.1 * np.exp(-r2)

    result_h.validate(h_test, hu_test, hv_test)
    result_hu.validate(h_test, hu_test)
    result_hv.validate(h_test, hv_test)


if __name__ == "__main__":
    main()
