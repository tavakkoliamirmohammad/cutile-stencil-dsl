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

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.reference.stencil_ref import apply_stencil


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

    spec_u = gray_scott_u._stencil_spec
    spec_v = gray_scott_v._stencil_spec

    # Analysis for u equation
    accesses_u = extract_footprint(spec_u)
    halo_u = compute_halo(accesses_u, 2)
    spec_u.halo_widths = halo_u
    print(f"u-stencil: {len(accesses_u)} accesses, halo={halo_u}")
    print(f"  inputs: {spec_u.inputs}")

    # Analysis for v equation
    accesses_v = extract_footprint(spec_v)
    halo_v = compute_halo(accesses_v, 2)
    spec_v.halo_widths = halo_v
    print(f"v-stencil: {len(accesses_v)} accesses, halo={halo_v}")

    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    domain = (64, 64)
    tile_u = compute_tile_config(spec_u, domain, hw)
    print(f"Tile config: {tile_u.tile_sizes}, overhead={tile_u.overhead_fraction:.4f}")

    roof = roofline_analysis(spec_u, hw)
    print(f"Roofline: {roof.flops_per_point} FLOPs/pt, AI={roof.arithmetic_intensity:.3f}")

    # NumPy simulation
    Nx, Ny = 64, 64
    hx, hy = halo_u

    u = np.ones((Nx + 2 * hx, Ny + 2 * hy))
    v = np.zeros_like(u)

    # Seed a small square perturbation in the center
    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    r = 5
    u[cx - r:cx + r, cy - r:cy + r] = 0.50
    v[cx - r:cx + r, cy - r:cy + r] = 0.25

    steps = 100
    print(f"\nSimulation: {steps} steps, grid={Nx}x{Ny}")

    from cutile_stencil.reference.stencil_ref import _ArrayProxy

    for s in range(steps):
        # Apply u equation
        proxies_u = {
            'u': _ArrayProxy(u, halo_u),
            'v': _ArrayProxy(v, halo_u),
        }
        idx_args = [0, 0]
        result_u = spec_u.update_fn(proxies_u['u'], proxies_u['v'], *idx_args)
        interior = tuple(slice(h, n - h) for h, n in zip(halo_u, u.shape))
        u_new = u.copy()
        u_new[interior] = result_u

        # Apply v equation
        proxies_v = {
            'u': _ArrayProxy(u, halo_v),
            'v': _ArrayProxy(v, halo_v),
        }
        result_v = spec_v.update_fn(proxies_v['u'], proxies_v['v'], *idx_args)
        v_new = v.copy()
        v_new[interior] = result_v

        u = u_new
        v = v_new

    print(f"  u: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    print(f"  v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    # Verify u and v are still in physical range [0, 1]
    assert u.min() >= -0.1 and u.max() <= 1.1, f"u out of range: [{u.min()}, {u.max()}]"
    assert v.min() >= -0.1 and v.max() <= 1.1, f"v out of range: [{v.min()}, {v.max()}]"
    print("  ✓ Solution in valid range")

    # ── Code generation ──────────────────────────────────────────────
    from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    gen_u = StencilCodeGenerator(spec_u, tile_u)
    out_path_u = os.path.join(gen_dir, "gray_scott_u_kernel.py")
    gen_u.emit_to_file(out_path_u)
    print(f"\nGenerated: {out_path_u}")

    tile_v = compute_tile_config(spec_v, domain, hw)
    gen_v = StencilCodeGenerator(spec_v, tile_v)
    out_path_v = os.path.join(gen_dir, "gray_scott_v_kernel.py")
    gen_v.emit_to_file(out_path_v)
    print(f"Generated: {out_path_v}")

    import ast
    ast.parse(open(out_path_u).read())
    ast.parse(open(out_path_v).read())
    print("  ✓ Valid Python syntax")

    # ── GPU kernel validation ────────────────────────────────────────
    import importlib.util
    import cupy as cp

    # Fresh test data (simulation modified u, v above)
    u_test = np.ones((Nx + 2 * hx, Ny + 2 * hy))
    v_test = np.zeros_like(u_test)
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    r_t = 5
    u_test[cx_t - r_t:cx_t + r_t, cy_t - r_t:cy_t + r_t] = 0.50
    v_test[cx_t - r_t:cx_t + r_t, cy_t - r_t:cy_t + r_t] = 0.25

    # Validate u-kernel
    mod_spec_u = importlib.util.spec_from_file_location("kernel_u", out_path_u)
    mod_u = importlib.util.module_from_spec(mod_spec_u)
    mod_spec_u.loader.exec_module(mod_u)

    u_gpu = cp.asarray(u_test, dtype=cp.float64)
    v_gpu = cp.asarray(v_test, dtype=cp.float64)
    out_gpu = cp.zeros_like(u_gpu)
    mod_u.launch_gray_scott_u(u_gpu, v_gpu, out_gpu)
    cp.cuda.Device().synchronize()

    gpu_result = cp.asnumpy(out_gpu)
    proxies_ref = {
        'u': _ArrayProxy(u_test, halo_u),
        'v': _ArrayProxy(v_test, halo_u),
    }
    cpu_interior = spec_u.update_fn(proxies_ref['u'], proxies_ref['v'], 0, 0)
    cpu_ref = u_test.copy()
    interior = tuple(slice(h_, n - h_) for h_, n in zip(halo_u, u_test.shape))
    cpu_ref[interior] = cpu_interior
    assert np.allclose(gpu_result[interior], cpu_ref[interior]), "GPU kernel mismatch (u)!"
    print("  ✓ GPU kernel (gray_scott_u) matches NumPy reference")

    # Validate v-kernel
    mod_spec_v = importlib.util.spec_from_file_location("kernel_v", out_path_v)
    mod_v = importlib.util.module_from_spec(mod_spec_v)
    mod_spec_v.loader.exec_module(mod_v)

    out_gpu_v = cp.zeros_like(v_gpu)
    mod_v.launch_gray_scott_v(u_gpu, v_gpu, out_gpu_v)
    cp.cuda.Device().synchronize()

    gpu_result_v = cp.asnumpy(out_gpu_v)
    proxies_ref_v = {
        'u': _ArrayProxy(u_test, halo_v),
        'v': _ArrayProxy(v_test, halo_v),
    }
    cpu_interior_v = spec_v.update_fn(proxies_ref_v['u'], proxies_ref_v['v'], 0, 0)
    cpu_ref_v = v_test.copy()
    interior_v = tuple(slice(h_, n - h_) for h_, n in zip(halo_v, v_test.shape))
    cpu_ref_v[interior_v] = cpu_interior_v
    assert np.allclose(gpu_result_v[interior_v], cpu_ref_v[interior_v]), "GPU kernel mismatch (v)!"
    print("  ✓ GPU kernel (gray_scott_v) matches NumPy reference")


if __name__ == "__main__":
    main()
