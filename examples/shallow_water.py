"""2D Shallow Water Equations -- linearized (new cuTile API).

Solves the linearized shallow water system:
    dh/dt   = -H0 * (d(hu)/dx + d(hv)/dy)
    d(hu)/dt = -g * dh/dx
    d(hv)/dt = -g * dh/dy

Uses central differences. Three coupled fields (h, hu, hv), each
compiled as a separate stencil.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import _ArrayProxy


# Physical parameters
g = 9.81      # gravity
H0 = 1.0      # mean depth
dt = 0.01
dx = 0.1


@stencil
def shallow_water_h(h, hu, hv, i, j):
    return h[i, j] - dt * H0 * ((hu[i + 1, j] - hu[i - 1, j]) / (2 * dx) + (hv[i, j + 1] - hv[i, j - 1]) / (2 * dx))


@stencil
def shallow_water_hu(h, hu, i, j):
    return hu[i, j] - dt * g * (h[i + 1, j] - h[i - 1, j]) / (2 * dx)


@stencil
def shallow_water_hv(h, hv, i, j):
    return hv[i, j] - dt * g * (h[i, j + 1] - h[i, j - 1]) / (2 * dx)


def main():
    print("=" * 60)
    print("2D Shallow Water Equations -- Linearized (new cuTile API)")
    print("=" * 60)

    # -- Compile all three kernels ------------------------------------------
    result_h = compile(shallow_water_h)
    result_hu = compile(shallow_water_hu)
    result_hv = compile(shallow_water_hv)

    print(f"\nCompilation summary (h-kernel):")
    print(f"  ndim:           {result_h.ndim}")
    print(f"  halo_widths:    {result_h.halo_widths}")
    print(f"  tile_sizes:     {result_h.tile_sizes}")
    print(f"  temporal_steps: {result_h.temporal_steps}")
    if result_h.analysis:
        for key, val in result_h.analysis.items():
            print(f"  {key}: {val}")

    # -- Validate generated code is valid Python ----------------------------
    ast.parse(result_h.code)
    ast.parse(result_hu.code)
    ast.parse(result_hv.code)
    print("\n  [OK] Generated h, hu, hv kernel code is valid Python syntax")

    halo_h = result_h.halo_widths
    halo_hu = result_hu.halo_widths
    halo_hv = result_hv.halo_widths

    # Use consistent halo (max across all fields)
    max_halo = tuple(
        max(halo_h[d], halo_hu[d], halo_hv[d])
        for d in range(2)
    )
    print(f"  Unified halo: {max_halo}")

    # -- NumPy simulation ---------------------------------------------------
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

    interior = tuple(slice(hw_, s_ - hw_) for hw_, s_ in zip(max_halo, h.shape))

    for s in range(steps):
        # h update
        proxy_h = _ArrayProxy(h, max_halo)
        proxy_hu = _ArrayProxy(hu, max_halo)
        proxy_hv = _ArrayProxy(hv, max_halo)
        h_step = shallow_water_h._fn(proxy_h, proxy_hu, proxy_hv, 0, 0)
        h_new = h.copy()
        h_new[interior] = h_step

        # hu update
        proxy_h2 = _ArrayProxy(h, max_halo)
        proxy_hu2 = _ArrayProxy(hu, max_halo)
        hu_step = shallow_water_hu._fn(proxy_h2, proxy_hu2, 0, 0)
        hu_new = hu.copy()
        hu_new[interior] = hu_step

        # hv update
        proxy_h3 = _ArrayProxy(h, max_halo)
        proxy_hv3 = _ArrayProxy(hv, max_halo)
        hv_step = shallow_water_hv._fn(proxy_h3, proxy_hv3, 0, 0)
        hv_new = hv.copy()
        hv_new[interior] = hv_step

        h, hu, hv = h_new, hu_new, hv_new

    final_energy = np.sum(h[hx:-hx, hy:-hy] ** 2)
    print(f"  Initial energy (h^2): {initial_energy:.6f}")
    print(f"  Final energy (h^2):   {final_energy:.6f}")
    print(f"  Max |h|: {np.max(np.abs(h)):.6f}")
    print(f"  Max |hu|: {np.max(np.abs(hu)):.6f}")
    print(f"  Max |hv|: {np.max(np.abs(hv)):.6f}")

    assert np.max(np.abs(h)) < 10.0, "h blew up"
    print("  [OK] Simulation stable")


if __name__ == "__main__":
    main()
