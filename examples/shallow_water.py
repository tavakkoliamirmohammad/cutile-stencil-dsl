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

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.reference.stencil_ref import _ArrayProxy


# Physical parameters
g = 9.81      # gravity
H0 = 1.0      # mean depth
dt = 0.01
dx = 0.1


@stencil(ndim=2, order=2, dtype="float64")
def shallow_water_h(h, hu, hv, i, j):
    """Update height field h."""
    dhu_dx = (hu[i + 1, j] - hu[i - 1, j]) / (2 * dx)
    dhv_dy = (hv[i, j + 1] - hv[i, j - 1]) / (2 * dx)
    return h[i, j] - dt * H0 * (dhu_dx + dhv_dy)


@stencil(ndim=2, order=2, dtype="float64")
def shallow_water_hu(h, hu, i, j):
    """Update x-momentum field hu."""
    dh_dx = (h[i + 1, j] - h[i - 1, j]) / (2 * dx)
    return hu[i, j] - dt * g * dh_dx


@stencil(ndim=2, order=2, dtype="float64")
def shallow_water_hv(h, hv, i, j):
    """Update y-momentum field hv."""
    dh_dy = (h[i, j + 1] - h[i, j - 1]) / (2 * dx)
    return hv[i, j] - dt * g * dh_dy


def main():
    print("=" * 60)
    print("2D Shallow Water Equations (Linearized)")
    print("=" * 60)

    spec_h = shallow_water_h._stencil_spec
    spec_hu = shallow_water_hu._stencil_spec
    spec_hv = shallow_water_hv._stencil_spec

    # Compute halos and use the maximum across all fields for consistent arrays
    max_halo = [0, 0]
    for name, spec in [("h", spec_h), ("hu", spec_hu), ("hv", spec_hv)]:
        accesses = extract_footprint(spec)
        halo = compute_halo(accesses, 2)
        print(f"{name}-stencil: {len(accesses)} accesses, halo={halo}, inputs={spec.inputs}")
        for d in range(2):
            max_halo[d] = max(max_halo[d], halo[d])

    # Use consistent halo for all fields
    unified_halo = tuple(max_halo)
    for spec in [spec_h, spec_hu, spec_hv]:
        spec.halo_widths = unified_halo
    print(f"Unified halo: {unified_halo}")

    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    tile_h = compute_tile_config(spec_h, (64, 64), hw)
    print(f"\nTile config: {tile_h.tile_sizes}")

    roof = roofline_analysis(spec_h, hw)
    print(f"Roofline (h): {roof.flops_per_point} FLOPs/pt, AI={roof.arithmetic_intensity:.3f}")

    # NumPy simulation
    Nx, Ny = 32, 32
    hx, hy = spec_h.halo_widths

    h = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    # Gaussian height perturbation in center
    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(h.shape[0]):
        for jj in range(h.shape[1]):
            r2 = ((ii - cx) / 3.0) ** 2 + ((jj - cy) / 3.0) ** 2
            h[ii, jj] = 0.1 * np.exp(-r2)

    initial_energy = np.sum(h[hx:-hx, hy:-hy] ** 2)
    steps = 50
    print(f"\nSimulation: {steps} steps, grid={Nx}x{Ny}")

    for s in range(steps):
        # Update h
        proxy = {
            'h': _ArrayProxy(h, spec_h.halo_widths),
            'hu': _ArrayProxy(hu, spec_h.halo_widths),
            'hv': _ArrayProxy(hv, spec_h.halo_widths),
        }
        result_h = spec_h.update_fn(proxy['h'], proxy['hu'], proxy['hv'], 0, 0)
        h_new = h.copy()
        interior = tuple(slice(hw_, s_ - hw_) for hw_, s_ in zip(spec_h.halo_widths, h.shape))
        h_new[interior] = result_h

        # Update hu
        proxy2 = {
            'h': _ArrayProxy(h, spec_hu.halo_widths),
            'hu': _ArrayProxy(hu, spec_hu.halo_widths),
        }
        result_hu = spec_hu.update_fn(proxy2['h'], proxy2['hu'], 0, 0)
        hu_new = hu.copy()
        hu_new[interior] = result_hu

        # Update hv
        proxy3 = {
            'h': _ArrayProxy(h, spec_hv.halo_widths),
            'hv': _ArrayProxy(hv, spec_hv.halo_widths),
        }
        result_hv = spec_hv.update_fn(proxy3['h'], proxy3['hv'], 0, 0)
        hv_new = hv.copy()
        hv_new[interior] = result_hv

        h, hu, hv = h_new, hu_new, hv_new

    final_energy = np.sum(h[hx:-hx, hy:-hy] ** 2)
    print(f"  Initial energy (h^2): {initial_energy:.6f}")
    print(f"  Final energy (h^2):   {final_energy:.6f}")
    print(f"  Max |h|: {np.max(np.abs(h)):.6f}")
    print(f"  Max |hu|: {np.max(np.abs(hu)):.6f}")
    print(f"  Max |hv|: {np.max(np.abs(hv)):.6f}")

    # Should remain bounded (stable)
    assert np.max(np.abs(h)) < 10.0, "h blew up"
    print("  ✓ Simulation stable")


if __name__ == "__main__":
    main()
