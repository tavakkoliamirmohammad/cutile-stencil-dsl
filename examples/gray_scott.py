"""Gray-Scott reaction-diffusion system (new cuTile API).

Two coupled fields (u, v) with diffusion and reaction:
    du/dt = Du * laplacian(u) - u*v^2 + F*(1 - u)
    dv/dt = Dv * laplacian(v) + u*v^2 - (F + k)*v

Separate stencils are compiled for each field update, since the
@stencil decorator currently handles single-output stencils.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import _ArrayProxy


# Gray-Scott parameters
Du = 0.16
Dv = 0.08
F = 0.035
k = 0.065
dt = 1.0


@stencil
def gray_scott_u(u, v, i, j):
    return u[i, j] + dt * (Du * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]) - u[i, j] * v[i, j] * v[i, j] + F * (1.0 - u[i, j]))


@stencil
def gray_scott_v(u, v, i, j):
    return v[i, j] + dt * (Dv * (v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]) + u[i, j] * v[i, j] * v[i, j] - (F + k) * v[i, j])


def main():
    print("=" * 60)
    print("Gray-Scott Reaction-Diffusion (new cuTile API)")
    print("=" * 60)

    # -- Compile both kernels -----------------------------------------------
    result_u = compile(gray_scott_u)
    result_v = compile(gray_scott_v)

    print(f"\nCompilation summary (u-kernel):")
    print(f"  ndim:           {result_u.ndim}")
    print(f"  halo_widths:    {result_u.halo_widths}")
    print(f"  tile_sizes:     {result_u.tile_sizes}")
    print(f"  temporal_steps: {result_u.temporal_steps}")
    if result_u.analysis:
        for key, val in result_u.analysis.items():
            print(f"  {key}: {val}")

    # -- Validate generated code is valid Python ----------------------------
    ast.parse(result_u.code)
    ast.parse(result_v.code)
    print("\n  [OK] Generated u and v kernel code is valid Python syntax")

    halo_u = result_u.halo_widths
    halo_v = result_v.halo_widths

    # -- NumPy simulation ---------------------------------------------------
    Nx, Ny = 64, 64
    hx, hy = halo_u

    u = np.ones((Nx + 2 * hx, Ny + 2 * hy))
    v = np.zeros_like(u)

    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    r = 5
    u[cx - r:cx + r, cy - r:cy + r] = 0.50
    v[cx - r:cx + r, cy - r:cy + r] = 0.25

    steps = 100
    print(f"\nSimulation: {steps} steps, grid={Nx}x{Ny}")

    for s in range(steps):
        proxy_u_arr = _ArrayProxy(u, halo_u)
        proxy_v_arr = _ArrayProxy(v, halo_u)
        idx_args = [0, 0]
        result_u_step = gray_scott_u._fn(proxy_u_arr, proxy_v_arr, *idx_args)
        interior = tuple(slice(h, n - h) for h, n in zip(halo_u, u.shape))
        u_new = u.copy()
        u_new[interior] = result_u_step

        proxy_u_arr2 = _ArrayProxy(u, halo_v)
        proxy_v_arr2 = _ArrayProxy(v, halo_v)
        result_v_step = gray_scott_v._fn(proxy_u_arr2, proxy_v_arr2, *idx_args)
        v_new = v.copy()
        v_new[interior] = result_v_step

        u = u_new
        v = v_new

    print(f"  u: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    print(f"  v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    assert u.min() >= -0.1 and u.max() <= 1.1, \
        f"u out of range: [{u.min()}, {u.max()}]"
    assert v.min() >= -0.1 and v.max() <= 1.1, \
        f"v out of range: [{v.min()}, {v.max()}]"
    print("  [OK] Solution in valid range")


if __name__ == "__main__":
    main()
