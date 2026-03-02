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

    # ── Code generation ──────────────────────────────────────────────
    from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    tile_hu = compute_tile_config(spec_hu, (64, 64), hw)
    tile_hv = compute_tile_config(spec_hv, (64, 64), hw)

    for name, spec, tile in [("shallow_water_h", spec_h, tile_h),
                              ("shallow_water_hu", spec_hu, tile_hu),
                              ("shallow_water_hv", spec_hv, tile_hv)]:
        gen = StencilCodeGenerator(spec, tile)
        path = os.path.join(gen_dir, f"{name}_kernel.py")
        gen.emit_to_file(path)
        print(f"\nGenerated: {path}")

    import ast
    for name in ["shallow_water_h", "shallow_water_hu", "shallow_water_hv"]:
        ast.parse(open(os.path.join(gen_dir, f"{name}_kernel.py")).read())
    print("  ✓ Valid Python syntax")

    # ── GPU kernel validation ────────────────────────────────────────
    import importlib.util
    import cupy as cp

    # Fresh test data (simulation modified h, hu, hv above)
    h_test = np.zeros((Nx + 2 * hx, Ny + 2 * hy))
    hu_test = np.zeros_like(h_test)
    hv_test = np.zeros_like(h_test)
    cx_t, cy_t = (Nx + 2 * hx) // 2, (Ny + 2 * hy) // 2
    for ii in range(h_test.shape[0]):
        for jj in range(h_test.shape[1]):
            r2 = ((ii - cx_t) / 3.0)**2 + ((jj - cy_t) / 3.0)**2
            h_test[ii, jj] = 0.1 * np.exp(-r2)

    h_gpu = cp.asarray(h_test, dtype=cp.float64)
    hu_gpu = cp.asarray(hu_test, dtype=cp.float64)
    hv_gpu = cp.asarray(hv_test, dtype=cp.float64)
    interior = tuple(slice(hw_, s_ - hw_) for hw_, s_ in zip(unified_halo, h_test.shape))

    # Validate h kernel
    mod_h_spec = importlib.util.spec_from_file_location(
        "k_h", os.path.join(gen_dir, "shallow_water_h_kernel.py"))
    mod_h = importlib.util.module_from_spec(mod_h_spec)
    mod_h_spec.loader.exec_module(mod_h)

    out_h_gpu = cp.zeros_like(h_gpu)
    mod_h.launch_shallow_water_h(h_gpu, hu_gpu, hv_gpu, out_h_gpu)
    cp.cuda.Device().synchronize()

    gpu_result_h = cp.asnumpy(out_h_gpu)
    proxy_ref = {
        'h': _ArrayProxy(h_test, unified_halo),
        'hu': _ArrayProxy(hu_test, unified_halo),
        'hv': _ArrayProxy(hv_test, unified_halo),
    }
    cpu_h = spec_h.update_fn(proxy_ref['h'], proxy_ref['hu'], proxy_ref['hv'], 0, 0)
    cpu_ref_h = h_test.copy()
    cpu_ref_h[interior] = cpu_h
    assert np.allclose(gpu_result_h[interior], cpu_ref_h[interior]), "GPU kernel mismatch (h)!"
    print("  ✓ GPU kernel (shallow_water_h) matches NumPy reference")

    # Validate hu kernel
    mod_hu_spec = importlib.util.spec_from_file_location(
        "k_hu", os.path.join(gen_dir, "shallow_water_hu_kernel.py"))
    mod_hu = importlib.util.module_from_spec(mod_hu_spec)
    mod_hu_spec.loader.exec_module(mod_hu)

    out_hu_gpu = cp.zeros_like(hu_gpu)
    mod_hu.launch_shallow_water_hu(h_gpu, hu_gpu, out_hu_gpu)
    cp.cuda.Device().synchronize()

    gpu_result_hu = cp.asnumpy(out_hu_gpu)
    proxy_ref2 = {
        'h': _ArrayProxy(h_test, unified_halo),
        'hu': _ArrayProxy(hu_test, unified_halo),
    }
    cpu_hu = spec_hu.update_fn(proxy_ref2['h'], proxy_ref2['hu'], 0, 0)
    cpu_ref_hu = hu_test.copy()
    cpu_ref_hu[interior] = cpu_hu
    assert np.allclose(gpu_result_hu[interior], cpu_ref_hu[interior]), "GPU kernel mismatch (hu)!"
    print("  ✓ GPU kernel (shallow_water_hu) matches NumPy reference")

    # Validate hv kernel
    mod_hv_spec = importlib.util.spec_from_file_location(
        "k_hv", os.path.join(gen_dir, "shallow_water_hv_kernel.py"))
    mod_hv = importlib.util.module_from_spec(mod_hv_spec)
    mod_hv_spec.loader.exec_module(mod_hv)

    out_hv_gpu = cp.zeros_like(hv_gpu)
    mod_hv.launch_shallow_water_hv(h_gpu, hv_gpu, out_hv_gpu)
    cp.cuda.Device().synchronize()

    gpu_result_hv = cp.asnumpy(out_hv_gpu)
    proxy_ref3 = {
        'h': _ArrayProxy(h_test, unified_halo),
        'hv': _ArrayProxy(hv_test, unified_halo),
    }
    cpu_hv = spec_hv.update_fn(proxy_ref3['h'], proxy_ref3['hv'], 0, 0)
    cpu_ref_hv = hv_test.copy()
    cpu_ref_hv[interior] = cpu_hv
    assert np.allclose(gpu_result_hv[interior], cpu_ref_hv[interior]), "GPU kernel mismatch (hv)!"
    print("  ✓ GPU kernel (shallow_water_hv) matches NumPy reference")


if __name__ == "__main__":
    main()
