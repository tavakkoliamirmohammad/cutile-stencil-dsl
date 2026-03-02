"""2D acoustic wave equation with 4th-order stencil.

Second-order wave equation: ∂²u/∂t² = c² ∇²u

4th-order spatial discretisation of the Laplacian in 2D uses offsets ±1, ±2
in each dimension. Explicit leapfrog time-stepping.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.reference.stencil_ref import apply_stencil


# ── 4th-order 2D Laplacian stencil ─────────────────────────────────
# Coefficients for 4th-order central difference:
#   -1/12 · f(x±2) + 4/3 · f(x±1) - 5/2 · f(x) in each dimension
# Combined with c²Δt²/Δx² = 0.1 and leapfrog: u_new = 2u - u_old + c²Δt²∇²u

@stencil(ndim=2, order=4, dtype="float64")
def wave_2d(u, i, j):
    # 4th-order Laplacian * (c²Δt²/Δx²) with factor 0.1
    lap_x = (-1/12) * u[i - 2, j] + (4/3) * u[i - 1, j] + (-5/2) * u[i, j] + (4/3) * u[i + 1, j] + (-1/12) * u[i + 2, j]
    lap_y = (-1/12) * u[i, j - 2] + (4/3) * u[i, j - 1] + (-5/2) * u[i, j] + (4/3) * u[i, j + 1] + (-1/12) * u[i, j + 2]
    return 0.1 * (lap_x + lap_y)


def main():
    spec = wave_2d._stencil_spec
    print(f"Stencil: {spec.name}, ndim={spec.ndim}, order={spec.order}")

    # ── Analysis ────────────────────────────────────────────────────
    accesses = extract_footprint(spec)
    halo = compute_halo(accesses, spec.ndim)
    spec.halo_widths = halo
    print(f"Footprint: {len(accesses)} accesses, halo = {halo}")

    hw = HardwareSpec(peak_bandwidth_gbs=1000.0, shared_mem_bytes=49152,
                      sm_count=108, dtype_bytes=8)
    domain = (256, 256)
    tile_cfg = compute_tile_config(spec, domain, hw)
    print(f"Tile config: sizes={tile_cfg.tile_sizes}, overhead={tile_cfg.overhead_fraction:.4f}")

    temp_cfg = compute_temporal_config(spec, tile_cfg, hw)
    print(f"Temporal blocking: {temp_cfg.steps} steps, BW reduction={temp_cfg.bandwidth_reduction_factor:.1f}x")

    roof = roofline_analysis(spec, hw)
    print(f"Roofline: {roof.flops_per_point} FLOPs/pt, AI={roof.arithmetic_intensity:.3f}, bound={roof.bound}")

    # ── Code generation ─────────────────────────────────────────────
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    codegen = StencilCodeGenerator(spec, tile_cfg, temp_cfg)
    out_path = os.path.join(gen_dir, "wave_2d_kernel.py")
    codegen.emit_to_file(out_path)
    print(f"\nGenerated cuTile kernel: {out_path}")

    import ast
    ast.parse(open(out_path).read())
    print("  ✓ Valid Python syntax")

    # ── NumPy reference simulation (leapfrog) ───────────────────────
    Nx, Ny = 64, 64
    hx, hy = halo
    u = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    u_old = np.zeros_like(u)

    # Gaussian pulse initial condition
    cx, cy = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(u.shape[0]):
        for jj in range(u.shape[1]):
            r2 = ((ii - cx) / 5.0)**2 + ((jj - cy) / 5.0)**2
            u[ii, jj] = np.exp(-r2)
    u_old[:] = u

    steps = 50
    print(f"\nNumPy simulation: {steps} steps, grid={Nx}x{Ny}")
    for s in range(steps):
        # wave_2d returns the Laplacian contribution; leapfrog update:
        lap = apply_stencil(u, spec)
        u_new = 2 * u - u_old + lap
        # Zero boundaries
        u_new[0, :] = 0; u_new[-1, :] = 0
        u_new[:, 0] = 0; u_new[:, -1] = 0
        u_old = u
        u = u_new

    print(f"  Max amplitude: {np.max(np.abs(u)):.6f}")
    print("  ✓ Simulation completed")

    # ── GPU kernel validation ────────────────────────────────────────
    import importlib.util
    import cupy as cp

    # Fresh test array (simulation modified u above)
    test_u = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(test_u.shape[0]):
        for jj in range(test_u.shape[1]):
            r2 = ((ii - cx_t) / 5.0)**2 + ((jj - cy_t) / 5.0)**2
            test_u[ii, jj] = np.exp(-r2)

    mod_spec = importlib.util.spec_from_file_location("kernel", out_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    u_gpu = cp.asarray(test_u, dtype=cp.float64)
    out_gpu = cp.zeros_like(u_gpu)
    mod.launch_wave_2d(u_gpu, out_gpu)
    cp.cuda.Device().synchronize()

    gpu_result = cp.asnumpy(out_gpu)
    cpu_ref = apply_stencil(test_u, spec)
    # Compare interior only (halo cells differ)
    interior = (slice(hx, -hx), slice(hy, -hy))
    assert np.allclose(gpu_result[interior], cpu_ref[interior], atol=1e-10), "GPU kernel mismatch!"
    print("  ✓ GPU kernel matches NumPy reference")


if __name__ == "__main__":
    main()
