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

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import HardwareSpec
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.roofline import roofline_analysis
from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
from cutile_stencil.reference.stencil_ref import _ArrayProxy


# Physical parameters (normalized)
c0 = 1.0       # speed of light
dx = 0.01       # spatial step
dt = 0.005      # time step (CFL: dt < dx/c0)
eps = 1.0       # permittivity
mu = 1.0        # permeability


@stencil(ndim=1, order=2, dtype="float64")
def update_E(E, H, i):
    """Update E field: E^{n+1}[i] = E^n[i] + (dt/(eps*dx)) * (H[i] - H[i-1])"""
    return E[i] + (dt / (eps * dx)) * (H[i] - H[i - 1])


@stencil(ndim=1, order=2, dtype="float64")
def update_H(E, H, i):
    """Update H field: H^{n+1}[i] = H^n[i] + (dt/(mu*dx)) * (E[i+1] - E[i])"""
    return H[i] + (dt / (mu * dx)) * (E[i + 1] - E[i])


def main():
    print("=" * 60)
    print("1D FDTD Maxwell Solver (E-H fields)")
    print("=" * 60)

    spec_E = update_E._stencil_spec
    spec_H = update_H._stencil_spec

    for name, spec in [("E", spec_E), ("H", spec_H)]:
        accesses = extract_footprint(spec)
        halo = compute_halo(accesses, 1)
        spec.halo_widths = halo
        print(f"{name}-update: {len(accesses)} accesses, halo={halo}, inputs={spec.inputs}")

    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    domain = (1024,)
    tile_E = compute_tile_config(spec_E, domain, hw)
    print(f"\nTile config: {tile_E.tile_sizes}")

    roof = roofline_analysis(spec_E, hw)
    print(f"Roofline (E-update): {roof.flops_per_point} FLOPs/pt, AI={roof.arithmetic_intensity:.3f}")

    # Code generation
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)
    gen = StencilCodeGenerator(spec_E, tile_E)
    out_path = os.path.join(gen_dir, "fdtd_update_E_kernel.py")
    gen.emit_to_file(out_path)
    print(f"\nGenerated: {out_path}")

    import ast
    ast.parse(open(out_path).read())
    print("  ✓ Valid Python syntax")

    # NumPy FDTD simulation
    N = 500
    h = max(spec_E.halo_widths[0], spec_H.halo_widths[0])
    E = np.zeros(N + 2 * h)
    H = np.zeros(N + 2 * h)

    # Gaussian pulse source in the E-field
    source_pos = N // 4 + h
    t0 = 40  # center of pulse
    spread = 12  # width

    steps = 200
    print(f"\nSimulation: {steps} steps, N={N}")

    for s in range(steps):
        # Update H first (leapfrog)
        proxy_h = {
            'E': _ArrayProxy(E, spec_H.halo_widths),
            'H': _ArrayProxy(H, spec_H.halo_widths),
        }
        result_H = spec_H.update_fn(proxy_h['E'], proxy_h['H'], 0)
        H_new = H.copy()
        h_interior = tuple(slice(hw_, n - hw_) for hw_, n in zip(spec_H.halo_widths, H.shape))
        H_new[h_interior] = result_H
        H = H_new

        # Update E
        proxy_e = {
            'E': _ArrayProxy(E, spec_E.halo_widths),
            'H': _ArrayProxy(H, spec_E.halo_widths),
        }
        result_E = spec_E.update_fn(proxy_e['E'], proxy_e['H'], 0)
        E_new = E.copy()
        e_interior = tuple(slice(hw_, n - hw_) for hw_, n in zip(spec_E.halo_widths, E.shape))
        E_new[e_interior] = result_E
        E = E_new

        # Inject source
        E[source_pos] += np.exp(-0.5 * ((s - t0) / spread) ** 2)

    print(f"  Max |E|: {np.max(np.abs(E)):.6f}")
    print(f"  Max |H|: {np.max(np.abs(H)):.6f}")

    # Energy should be bounded (stable FDTD)
    total_energy = np.sum(E ** 2) + np.sum(H ** 2)
    print(f"  Total energy (E^2 + H^2): {total_energy:.6f}")
    assert np.max(np.abs(E)) < 100.0, "E field blew up"
    assert np.max(np.abs(H)) < 100.0, "H field blew up"
    print("  ✓ FDTD simulation stable")

    # ── GPU kernel validation (single E-update step) ─────────────────
    import importlib.util
    import cupy as cp

    mod_spec = importlib.util.spec_from_file_location("kernel", out_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    # Fresh test data for single-step validation
    E_test = np.random.randn(N + 2 * h)
    H_test = np.random.randn(N + 2 * h)

    E_gpu = cp.asarray(E_test, dtype=cp.float64)
    H_gpu = cp.asarray(H_test, dtype=cp.float64)
    out_gpu = cp.zeros_like(E_gpu)
    mod.launch_update_E(E_gpu, H_gpu, out_gpu)
    cp.cuda.Device().synchronize()

    gpu_result = cp.asnumpy(out_gpu)
    # CPU reference
    proxy = {
        'E': _ArrayProxy(E_test, spec_E.halo_widths),
        'H': _ArrayProxy(H_test, spec_E.halo_widths),
    }
    cpu_interior = spec_E.update_fn(proxy['E'], proxy['H'], 0)
    cpu_ref = E_test.copy()
    eh = spec_E.halo_widths[0]
    cpu_ref[eh:-eh] = cpu_interior
    assert np.allclose(gpu_result[eh:-eh], cpu_ref[eh:-eh]), "GPU kernel mismatch!"
    print("  ✓ GPU kernel matches NumPy reference")


if __name__ == "__main__":
    main()
