"""3D 7-point Laplacian stencil (new cuTile API).

Demonstrates DSL compilation and NumPy validation of the standard
7-point discrete Laplacian operator in three dimensions.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import apply_stencil


@stencil
def laplacian_3d(u, i, j, k):
    return (u[i - 1, j, k] + u[i + 1, j, k]
            + u[i, j - 1, k] + u[i, j + 1, k]
            + u[i, j, k - 1] + u[i, j, k + 1]
            - 6 * u[i, j, k])


def main():
    print("=" * 60)
    print("3D 7-point Laplacian (new cuTile API)")
    print("=" * 60)

    # -- Compile ------------------------------------------------------------
    result = compile(laplacian_3d)
    code = result.code

    print(f"\nCompilation summary:")
    print(f"  ndim:           {result.ndim}")
    print(f"  halo_widths:    {result.halo_widths}")
    print(f"  tile_sizes:     {result.tile_sizes}")
    print(f"  temporal_steps: {result.temporal_steps}")
    if result.analysis:
        for k, v in result.analysis.items():
            print(f"  {k}: {v}")

    # -- Validate generated code is valid Python ----------------------------
    ast.parse(code)
    print("\n  [OK] Generated code is valid Python syntax")

    # -- NumPy reference ----------------------------------------------------
    halo = result.halo_widths
    N = 16
    h = halo[0]
    u = np.zeros((N + 2 * h, N + 2 * h, N + 2 * h))
    c = N // 2 + h
    u[c, c, c] = 1.0

    ref = apply_stencil(u, laplacian_3d._fn, ndim=3, halo_widths=halo)
    lap_center = ref[c, c, c]
    print(f"\nNumPy reference: Laplacian at center = {lap_center:.6f}")
    assert abs(lap_center - (-6.0)) < 1e-10, f"Expected -6.0, got {lap_center}"
    print("  [OK] Correctness verified")


if __name__ == "__main__":
    main()
