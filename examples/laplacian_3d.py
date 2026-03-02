"""3D 7-point Laplacian stencil: DSL → analysis → codegen → NumPy validation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile
from cutile_stencil.reference.stencil_ref import apply_stencil


@stencil(dtype="float64")
def laplacian_3d(u, i, j, k):
    return (u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
            - 6 * u[i, j, k])


def main():
    # ── Compile ─────────────────────────────────────────────────────
    result = compile(laplacian_3d, domain=(64, 64, 64))
    result.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result.emit_to_file(os.path.join(gen_dir, "laplacian_3d_kernel.py"))
    print(f"\nGenerated cuTile kernel")
    print("  ✓ Valid Python syntax")

    # ── NumPy reference ─────────────────────────────────────────────
    halo = result.spec.halo_widths
    N = 16
    h = halo[0]
    u = np.zeros((N + 2 * h, N + 2 * h, N + 2 * h))
    c = N // 2 + h
    u[c, c, c] = 1.0

    ref = apply_stencil(u, result.spec)
    lap_center = ref[c, c, c]
    print(f"\nNumPy reference: Laplacian at center = {lap_center:.6f}")
    assert abs(lap_center - (-6.0)) < 1e-10, f"Expected -6.0, got {lap_center}"
    print("  ✓ Correctness verified")

    # ── GPU kernel validation ────────────────────────────────────────
    result.validate(u)


if __name__ == "__main__":
    main()
