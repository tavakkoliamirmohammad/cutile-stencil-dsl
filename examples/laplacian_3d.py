"""3D 7-point Laplacian stencil: DSL → analysis → codegen → NumPy validation."""

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
from cutile_stencil.reference.stencil_ref import apply_stencil


@stencil(ndim=3, order=2, dtype="float64")
def laplacian_3d(u, i, j, k):
    return (u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
            - 6 * u[i, j, k])


def main():
    spec = laplacian_3d._stencil_spec
    print(f"Stencil: {spec.name}, ndim={spec.ndim}, order={spec.order}")

    # ── Analysis ────────────────────────────────────────────────────
    accesses = extract_footprint(spec)
    halo = compute_halo(accesses, spec.ndim)
    spec.halo_widths = halo
    print(f"Footprint: {len(accesses)} accesses, halo = {halo}")

    hw = HardwareSpec(peak_bandwidth_gbs=2000.0, shared_mem_bytes=49152,
                      sm_count=132, dtype_bytes=8)
    domain = (64, 64, 64)
    tile_cfg = compute_tile_config(spec, domain, hw)
    print(f"Tile config: sizes={tile_cfg.tile_sizes}, "
          f"num_tiles={tile_cfg.num_tiles}, overhead={tile_cfg.overhead_fraction:.4f}")

    roof = roofline_analysis(spec, hw)
    print(f"Roofline: {roof.flops_per_point} FLOPs/pt, "
          f"AI={roof.arithmetic_intensity:.3f}, bound={roof.bound}")

    # ── Code generation ─────────────────────────────────────────────
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    codegen = StencilCodeGenerator(spec, tile_cfg)
    out_path = os.path.join(gen_dir, "laplacian_3d_kernel.py")
    codegen.emit_to_file(out_path)
    print(f"\nGenerated cuTile kernel: {out_path}")

    import ast
    ast.parse(open(out_path).read())
    print("  ✓ Valid Python syntax")

    # ── NumPy reference ─────────────────────────────────────────────
    N = 16
    h = halo[0]
    u = np.zeros((N + 2 * h, N + 2 * h, N + 2 * h))
    # Point source in center
    c = N // 2 + h
    u[c, c, c] = 1.0

    result = apply_stencil(u, spec)
    lap_center = result[c, c, c]
    print(f"\nNumPy reference: Laplacian at center = {lap_center:.6f}")
    # For a point source, Laplacian at center = -6 * 1.0 = -6.0
    assert abs(lap_center - (-6.0)) < 1e-10, f"Expected -6.0, got {lap_center}"
    print("  ✓ Correctness verified")


if __name__ == "__main__":
    main()
