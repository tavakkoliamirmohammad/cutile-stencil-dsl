"""Gray-Scott reaction-diffusion system.

Two coupled fields (u, v) with diffusion and reaction:
    du/dt = Du * laplacian(u) - u*v^2 + F*(1 - u)
    dv/dt = Dv * laplacian(v) + u*v^2 - (F + k)*v

This exercises multi-input stencils (two arrays u, v).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import _ArrayProxy


# Gray-Scott parameters
Du = 0.16
Dv = 0.08
F = 0.035
k = 0.065
dt = 1.0


@stencil(ndim=2, order=2, dtype="float64")
def gray_scott_u(u, v, i, j):
    lap = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
    return u[i, j] + dt * (Du * lap - u[i, j] * v[i, j] * v[i, j] + F * (1.0 - u[i, j]))


@stencil(ndim=2, order=2, dtype="float64")
def gray_scott_v(u, v, i, j):
    lap = v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]
    return v[i, j] + dt * (Dv * lap + u[i, j] * v[i, j] * v[i, j] - (F + k) * v[i, j])


def main():
    print("=" * 60)
    print("Gray-Scott Reaction-Diffusion System")
    print("=" * 60)

    # ── Compile both kernels ────────────────────────────────────────
    result_u = compile(gray_scott_u, domain=(64, 64))
    result_v = compile(gray_scott_v, domain=(64, 64))

    result_u.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result_u.emit_to_file(os.path.join(gen_dir, "gray_scott_u_kernel.py"))
    result_v.emit_to_file(os.path.join(gen_dir, "gray_scott_v_kernel.py"))
    print(f"\nGenerated u and v kernels")
    print("  ✓ Valid Python syntax")

    spec_u = result_u.spec
    spec_v = result_v.spec
    halo_u = spec_u.halo_widths
    halo_v = spec_v.halo_widths

    # ── NumPy simulation ────────────────────────────────────────────
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
        proxies_u = {
            'u': _ArrayProxy(u, halo_u),
            'v': _ArrayProxy(v, halo_u),
        }
        idx_args = [0, 0]
        result_u_step = spec_u.update_fn(proxies_u['u'], proxies_u['v'], *idx_args)
        interior = tuple(slice(h, n - h) for h, n in zip(halo_u, u.shape))
        u_new = u.copy()
        u_new[interior] = result_u_step

        proxies_v = {
            'u': _ArrayProxy(u, halo_v),
            'v': _ArrayProxy(v, halo_v),
        }
        result_v_step = spec_v.update_fn(proxies_v['u'], proxies_v['v'], *idx_args)
        v_new = v.copy()
        v_new[interior] = result_v_step

        u = u_new
        v = v_new

    print(f"  u: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    print(f"  v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    assert u.min() >= -0.1 and u.max() <= 1.1, f"u out of range: [{u.min()}, {u.max()}]"
    assert v.min() >= -0.1 and v.max() <= 1.1, f"v out of range: [{v.min()}, {v.max()}]"
    print("  ✓ Solution in valid range")

    # ── GPU kernel validation ────────────────────────────────────────
    u_test = np.ones((Nx + 2 * hx, Ny + 2 * hy))
    v_test = np.zeros_like(u_test)
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    r_t = 5
    u_test[cx_t - r_t:cx_t + r_t, cy_t - r_t:cy_t + r_t] = 0.50
    v_test[cx_t - r_t:cx_t + r_t, cy_t - r_t:cy_t + r_t] = 0.25

    result_u.validate(u_test, v_test)
    result_v.validate(u_test, v_test)


if __name__ == "__main__":
    main()
