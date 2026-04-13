"""2D heat equation with bricked memory layout (new cuTile API).

Demonstrates the bricked workflow concept:
1. Define a @stencil for the 2D heat equation
2. Compile to get generated code
3. Run CPU reference on flat data
4. Verify bricked layout round-trip

TODO: Bricked layout is a pass that has been implemented in the new
architecture (cutile.passes.bricked) but the lowering does not fully
support it yet.  For now, we compile normally and only demonstrate
the CPU reference + bricked layout conversion if available.
"""

import ast

import numpy as np

from cutile import stencil, compile
from cutile.reference.stencil_ref import apply_stencil


# -- Define stencil ---------------------------------------------------------

@stencil
def heat_2d(u, i, j):
    return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


def main():
    print("=" * 60)
    print("2D Heat Equation -- Bricked Layout (new cuTile API)")
    print("=" * 60)

    domain = (128, 128)

    # -- Compile (normal, bricked lowering not yet complete) ----------------
    # TODO: Once bricked lowering is fully supported, pass a layout argument:
    #   from cutile.passes.bricked import BrickLayout
    #   layout = BrickLayout(brick_sizes=(32, 32))
    #   result = compile(heat_2d, layout=layout)
    result = compile(heat_2d)
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

    # -- Flat reference data ------------------------------------------------
    halo = result.halo_widths
    Nx, Ny = domain
    hx, hy = halo
    flat = np.zeros((Nx + 2 * hx, Ny + 2 * hy))

    # Gaussian initial condition
    cx = (Nx + 2 * hx) / 2.0
    cy = (Ny + 2 * hy) / 2.0
    for ii in range(flat.shape[0]):
        for jj in range(flat.shape[1]):
            r2 = ((ii - cx) / 10.0)**2 + ((jj - cy) / 10.0)**2
            flat[ii, jj] = np.exp(-r2)

    # CPU reference (one stencil application on flat data)
    cpu_ref = apply_stencil(flat, heat_2d._fn, ndim=2, halo_widths=halo)
    print(f"\nNumPy reference: domain={Nx}x{Ny}, halo={hx},{hy}")
    print(f"  Max value: {np.max(np.abs(cpu_ref)):.6f}")

    # -- Bricked layout conversion (best-effort) ----------------------------
    brick_sizes = (32, 32)
    try:
        from cutile_stencil.layout.bricks import to_bricks, from_bricks

        bricked = to_bricks(flat, brick_sizes, halo)
        print(f"\nBricked layout: flat {flat.shape} -> bricked {bricked.shape}")

        recovered = from_bricks(bricked, brick_sizes, halo)
        interior = (slice(hx, -hx), slice(hy, -hy))
        assert np.allclose(recovered[interior], flat[interior]), "Round-trip failed!"
        print("  [OK] to_bricks -> from_bricks round-trip verified")
    except ImportError:
        print("\n  [NOTE] cutile_stencil.layout.bricks not available; "
              "skipping bricked layout demo")

    print("  [OK] CPU reference completed")


if __name__ == "__main__":
    main()
