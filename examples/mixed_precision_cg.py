"""Mixed-precision CG demo: FP64 outer loop + FP32 inner CG.

Shows how iterative refinement recovers FP64 accuracy while
doing most of the work in FP32 (2× less memory bandwidth).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cutile_stencil.solvers.formats import laplacian_1d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv
from cutile_stencil.reference.cg_ref import cg_solve, mixed_precision_cg
from cutile_stencil.solvers.mixed_precision import generate_mixed_precision_cg


def main():
    print("=" * 60)
    print("Mixed-Precision CG: FP64 outer + FP32 inner")
    print("=" * 60)

    N = 200
    h = 1.0 / (N + 1)
    A = laplacian_1d_dia(N)

    # Known RHS
    x_grid = np.linspace(h, 1 - h, N)
    exact = np.sin(2 * np.pi * x_grid)
    f = dia_spmv(A, exact)  # f = A @ exact_solution

    A_fn = lambda v: dia_spmv(A, v)

    # ── Standard FP64 CG ───────────────────────────────────────────
    x_fp64, iters_64, res_64 = cg_solve(A_fn, f, tol=1e-12, max_iter=1000)
    err_64 = np.max(np.abs(x_fp64 - exact))
    print(f"\nStandard CG (FP64):")
    print(f"  Iterations: {iters_64}")
    print(f"  Final residual: {res_64[-1]:.2e}")
    print(f"  Max error: {err_64:.2e}")

    # ── Mixed-precision CG ──────────────────────────────────────────
    x_mixed, outers, res_mix = mixed_precision_cg(
        A_fn, f, inner_dtype=np.float32,
        tol=1e-12, max_outer=30, max_inner=500, inner_tol=1e-3,
    )
    err_mix = np.max(np.abs(x_mixed - exact))
    print(f"\nMixed-Precision CG (FP64 outer + FP32 inner):")
    print(f"  Outer iterations: {outers}")
    print(f"  Final residual: {res_mix[-1]:.2e}")
    print(f"  Max error: {err_mix:.2e}")

    # Both should achieve similar accuracy
    print(f"\n  FP64 error / Mixed error = {err_64 / err_mix:.2f}")
    assert err_mix < 1e-6, f"Mixed-precision error too large: {err_mix}"
    print("  ✓ Mixed-precision achieves high accuracy")

    # ── Generate cuTile kernel ──────────────────────────────────────
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    code = generate_mixed_precision_cg()
    out_path = os.path.join(gen_dir, "mixed_precision_cg_kernel.py")
    with open(out_path, "w") as fh:
        fh.write(code)

    import ast
    ast.parse(code)
    print(f"\nGenerated: {out_path} ✓")


if __name__ == "__main__":
    main()
