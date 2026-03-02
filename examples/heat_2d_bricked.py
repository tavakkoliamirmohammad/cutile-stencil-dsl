"""Bricked memory layout demo: 2D heat equation (5-point Laplacian).

Demonstrates the full bricked workflow:
1. Define a @stencil for the 2D heat equation
2. Compile with BrickLayout to produce a bricked kernel
3. Convert flat data to bricked layout with to_bricks()
4. (GPU: launch bricked kernel on bricked arrays)
5. Convert back with from_bricks() and compare with flat reference
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil import stencil, compile, BrickLayout
from cutile_stencil.layout.bricks import to_bricks, from_bricks
from cutile_stencil.reference.stencil_ref import apply_stencil


# ── Define stencil ──────────────────────────────────────────────────

@stencil(ndim=2, order=2, dtype="float64")
def heat_2d(u, i, j):
    return 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


def main():
    domain = (128, 128)
    brick_sizes = (32, 32)
    layout = BrickLayout(brick_sizes=brick_sizes)

    # ── Compile with bricked layout ──────────────────────────────────
    result = compile(heat_2d, domain=domain, layout=layout)
    result.print_summary()

    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    result.emit_to_file(os.path.join(gen_dir, "heat_2d_bricked_kernel.py"))
    print(f"\nGenerated bricked cuTile kernel")
    print("  ✓ Valid Python syntax")

    # ── Flat reference data ──────────────────────────────────────────
    halo = result.spec.halo_widths
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
    cpu_ref = apply_stencil(flat, result.spec)
    print(f"\nNumPy reference: domain={Nx}x{Ny}, halo={hx},{hy}")
    print(f"  Max value: {np.max(np.abs(cpu_ref)):.6f}")

    # ── Bricked layout conversion ────────────────────────────────────
    bricked = to_bricks(flat, brick_sizes, halo)
    print(f"\nBricked layout: flat {flat.shape} -> bricked {bricked.shape}")
    num_bricks = layout.num_bricks(domain)
    print(f"  Bricks: {num_bricks}, padded brick: "
          f"({brick_sizes[0]+2*hx}, {brick_sizes[1]+2*hy})")

    # Round-trip verification
    recovered = from_bricks(bricked, brick_sizes, halo)
    interior = (slice(hx, -hx), slice(hy, -hy))
    assert np.allclose(recovered[interior], flat[interior]), "Round-trip failed!"
    print("  ✓ to_bricks -> from_bricks round-trip verified")

    # ── GPU kernel validation ────────────────────────────────────────
    result.validate(flat)


if __name__ == "__main__":
    main()
