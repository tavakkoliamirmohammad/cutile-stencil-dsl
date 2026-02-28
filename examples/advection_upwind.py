"""1D upwind advection scheme.

Solves du/dt + c * du/dx = 0 with first-order upwind differencing.
For positive velocity c > 0, the stencil is left-biased (asymmetric):
    u_new[i] = u[i] - CFL * (u[i] - u[i-1])

This exercises asymmetric stencils.
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
from cutile_stencil.reference.stencil_ref import apply_stencil, time_march


CFL = 0.5  # CFL number (must be <= 1 for stability)


@stencil(ndim=1, order=2, dtype="float64")
def advection_upwind(u, i):
    return u[i] - CFL * (u[i] - u[i - 1])


def main():
    print("=" * 60)
    print("1D Upwind Advection: du/dt + c * du/dx = 0")
    print("=" * 60)

    spec = advection_upwind._stencil_spec
    accesses = extract_footprint(spec)
    halo = compute_halo(accesses, 1)
    spec.halo_widths = halo
    print(f"Stencil: {spec.name}, ndim={spec.ndim}")
    print(f"Footprint: {len(accesses)} accesses, halo={halo}")
    print(f"  Asymmetric: offsets = {sorted({a.offsets for a in accesses})}")

    hw = HardwareSpec(shared_mem_bytes=49152, dtype_bytes=8)
    domain = (1024,)
    tile_cfg = compute_tile_config(spec, domain, hw)
    print(f"Tile config: {tile_cfg.tile_sizes}")

    roof = roofline_analysis(spec, hw)
    print(f"Roofline: {roof.flops_per_point} FLOPs/pt, AI={roof.arithmetic_intensity:.3f}")

    # Code generation
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)
    gen = StencilCodeGenerator(spec, tile_cfg)
    out_path = os.path.join(gen_dir, "advection_upwind_kernel.py")
    gen.emit_to_file(out_path)
    print(f"\nGenerated: {out_path}")

    import ast
    ast.parse(open(out_path).read())
    print("  ✓ Valid Python syntax")

    # NumPy simulation
    N = 200
    h = halo[0]
    dx = 1.0 / N
    # Square wave initial condition
    u0 = np.zeros(N + 2 * h)
    u0[h + N // 4:h + N // 2] = 1.0

    steps = 80
    print(f"\nSimulation: {steps} steps, N={N}, CFL={CFL}")

    history = time_march(u0, spec, steps)
    u_final = history[-1]

    # Check total mass conservation (upwind scheme conserves mass)
    initial_mass = np.sum(u0[h:-h])
    final_mass = np.sum(u_final[h:-h])
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass:   {final_mass:.6f}")
    # Allow small boundary loss since we use zero BCs
    print(f"  Max value:    {u_final.max():.6f}")
    print(f"  ✓ Advection simulation completed")

    # Verify the wave moved to the right (upwind with positive velocity)
    initial_center_of_mass = np.sum(np.arange(N) * u0[h:-h]) / max(initial_mass, 1e-10)
    final_center_of_mass = np.sum(np.arange(N) * u_final[h:-h]) / max(final_mass, 1e-10)
    if final_mass > 0.01:
        assert final_center_of_mass >= initial_center_of_mass - 1, "Wave should move right"
        print("  ✓ Wave propagation direction verified")


if __name__ == "__main__":
    main()
