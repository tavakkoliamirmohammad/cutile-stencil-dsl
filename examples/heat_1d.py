"""Full pipeline: DSL → analysis → codegen → NumPy validation for 1D heat equation.

The 1D heat equation with explicit Euler:
    u^{n+1}_i = u^n_i + α·Δt/Δx² · (u^n_{i-1} - 2·u^n_i + u^n_{i+1})

With α·Δt/Δx² = 0.25 (stable):
    u^{n+1}_i = 0.25·u^n_{i-1} + 0.5·u^n_i + 0.25·u^n_{i+1}
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
from cutile_stencil.reference.stencil_ref import apply_stencil, time_march


# ── Define stencil ──────────────────────────────────────────────────

@stencil(ndim=1, order=2, dtype="float64")
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


def main():
    spec = heat_1d._stencil_spec
    print(f"Stencil: {spec.name}, ndim={spec.ndim}, order={spec.order}")

    # ── Analysis ────────────────────────────────────────────────────
    accesses = extract_footprint(spec)
    halo = compute_halo(accesses, spec.ndim)
    spec.halo_widths = halo
    print(f"Footprint: {len(accesses)} accesses, halo = {halo}")
    for a in accesses:
        print(f"  {a.array_name}[{a.offsets}]")

    hw = HardwareSpec(peak_bandwidth_gbs=1000.0, shared_mem_bytes=49152,
                      sm_count=108, dtype_bytes=8)
    domain = (1024,)
    tile_cfg = compute_tile_config(spec, domain, hw)
    print(f"\nTile config: sizes={tile_cfg.tile_sizes}, "
          f"halo={tile_cfg.halo_widths}, overhead={tile_cfg.overhead_fraction:.4f}")

    temp_cfg = compute_temporal_config(spec, tile_cfg, hw)
    print(f"Temporal blocking: {temp_cfg.steps} steps, "
          f"expanded_halo={temp_cfg.expanded_halo}, "
          f"BW reduction={temp_cfg.bandwidth_reduction_factor:.1f}x")

    roof = roofline_analysis(spec, hw)
    print(f"\nRoofline: {roof.flops_per_point} FLOPs/pt, "
          f"{roof.bytes_per_point} B/pt, AI={roof.arithmetic_intensity:.3f}")
    print(f"  Bound: {roof.bound}, peak ≈ {roof.peak_gpoints_s:.2f} Gpts/s")

    # ── Code generation ─────────────────────────────────────────────
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    codegen = StencilCodeGenerator(spec, tile_cfg)
    out_path = os.path.join(gen_dir, "heat_1d_kernel.py")
    codegen.emit_to_file(out_path)
    print(f"\nGenerated cuTile kernel: {out_path}")

    # Verify generated code is valid Python
    import ast
    code = open(out_path).read()
    ast.parse(code)
    print("  ✓ Valid Python syntax")

    # ── NumPy reference simulation ──────────────────────────────────
    N = 128
    h = halo[0]
    u0 = np.zeros(N + 2 * h)
    # Gaussian initial condition
    x = np.linspace(-1, 1, N + 2 * h)
    u0[:] = np.exp(-20 * x**2)
    u0[0] = 0.0
    u0[-1] = 0.0

    steps = 100
    history = time_march(u0, spec, steps)
    print(f"\nNumPy simulation: {steps} steps, N={N}")
    print(f"  Initial energy: {np.sum(u0**2):.6f}")
    print(f"  Final energy:   {np.sum(history[-1]**2):.6f}")
    # Heat equation should dissipate energy
    assert np.sum(history[-1]**2) < np.sum(u0**2), "Energy should decrease"
    print("  ✓ Energy dissipation verified")


if __name__ == "__main__":
    main()
